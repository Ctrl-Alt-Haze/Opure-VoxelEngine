// ============================================================================
//  Opure_Server.cpp
//  Opure-VoxelEngine -- HTTP inference server for the UE5 plugin.
//
//  Usage: opure_server [--port N] [--model <path>]
//
//  Endpoints
//  ---------
//    GET  /health     liveness + loaded model geometry and VRAM budget
//    POST /generate   {"prompt":..., "max_tokens":..., "temperature":...}
//                     -> {"text":..., "tokens":[...], "ms_per_token":...}
//    GET  /metrics    the MetricsCollector snapshot from the last request
//
//  Deliberately hand-rolled HTTP/1.1 over Winsock. The gRPC path was tried
//  first and abandoned: CMake FetchContent could not clone grpc under
//  Windows' path-length limit. Keeping this dependency-free means the whole
//  project still builds from Win32 + D3D12 + DXC alone, which is the point of
//  the exercise. It is a small, fixed set of routes, not a general server --
//  no chunked request bodies, no keep-alive, no TLS.
//
//  SERIALISATION. The engine runs on the calling thread and the socket accept
//  loop is single-threaded, so a request is served to completion before the
//  next is accepted. That is correct for a single inference device and is
//  what keeps the GPU command lists serialised; a UE5 client that wants
//  concurrency needs multiple engines, not threads.
// ---------------------------------------------------------------------------

#include "engine/Opure_VoxelEngine.h"
#include "engine/Opure_Graph.h"
#include "engine/Opure_Metrics.h"
#include "model/Opure_Tokenizer.h"
#include "core/Opure_Dx12.h"
#include "core/Opure_Json.h"
#include "core/Opure_Platform.h"

#include <winsock2.h>
#include <ws2tcpip.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace opure {
namespace {

// ---- tiny JSON writer -----------------------------------------------------
//
// The reader in Opure_Json.h parses; nothing in it emits, and pulling in a
// writer for four fixed response shapes would be the only third-party-shaped
// dependency in the project. This covers exactly what the routes need.

void AppendEscaped(std::string& out, std::string_view s)
{
    out += '"';
    for (const char c : s)
    {
        switch (c)
        {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                // Control characters must be escaped to stay valid JSON.
                if (static_cast<unsigned char>(c) < 0x20)
                {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x",
                                  static_cast<unsigned>(static_cast<unsigned char>(c)));
                    out += buf;
                }
                else
                {
                    out += c;
                }
        }
    }
    out += '"';
}

std::string JsonString(std::string_view s)
{
    std::string out;
    AppendEscaped(out, s);
    return out;
}

// ---- HTTP plumbing --------------------------------------------------------

constexpr size_t kMaxBodyBytes = 1u << 20;   // 1 MiB: generous for a prompt

bool SendAll(SOCKET s, std::string_view data)
{
    size_t sent = 0;
    while (sent < data.size())
    {
        const int n = ::send(s, data.data() + sent,
                             static_cast<int>(data.size() - sent), 0);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

void SendResponse(SOCKET s, int status, std::string_view body)
{
    const char* reason = (status == 200) ? "OK"
                       : (status == 404) ? "Not Found"
                       : (status == 405) ? "Method Not Allowed"
                       : (status == 413) ? "Payload Too Large"
                                         : "Bad Request";

    char head[256];
    const int n = std::snprintf(head, sizeof(head),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "\r\n", status, reason, body.size());

    if (SendAll(s, std::string_view(head, static_cast<size_t>(n))) &&
        SendAll(s, body))
    {
        ::shutdown(s, SD_BOTH);
    }
}

void SendJson(SOCKET s, int status, const std::string& json)
{
    SendResponse(s, status, json);
}

std::string_view Trim(std::string_view s)
{
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
    return s;
}

// Case-insensitive search for `name:` in an already-terminated header block.
std::string_view FindHeader(std::string_view head, std::string_view name)
{
    const size_t n = name.size();
    for (size_t i = 0; i + n < head.size(); ++i)
    {
        if (head.compare(i, n, name) != 0) continue;
        // Require the colon to follow immediately, and that this is at the start
        // of a line, so "Content-Type" cannot match inside another header name.
        if (i != 0 && head[i - 1] != '\n') continue;
        size_t j = i + n;
        if (j < head.size() && head[j] != ':') continue;
        ++j;
        while (j < head.size() && (head[j] == ' ' || head[j] == '\t')) ++j;
        size_t end = head.find("\r\n", j);
        if (end == std::string_view::npos) return {};
        return Trim(head.substr(j, end - j));
    }
    return {};
}

// Reads one whole request: header block, then exactly Content-Length body bytes.
//
// The body has to be fetched explicitly. POST bodies routinely arrive in a
// separate TCP segment from the headers, so stopping at the blank line leaves
// the payload unread and the JSON parser sees an empty document. No chunked
// transfer-encoding is supported; a chunked request is rejected rather than
// misparsed.
bool ReadRequest(SOCKET s, std::string& raw, size_t& contentLength, bool& chunked)
{
    raw.clear();
    contentLength = 0;
    chunked = false;

    char buf[4096];
    size_t hdrEnd = std::string::npos;

    for (;;)
    {
        if (hdrEnd == std::string::npos)
        {
            const int n = ::recv(s, buf, sizeof(buf), 0);
            if (n <= 0) return false;
            raw.append(buf, static_cast<size_t>(n));
            hdrEnd = raw.find("\r\n\r\n");
            if (hdrEnd == std::string::npos)
            {
                if (raw.size() > kMaxBodyBytes) return false;
                continue;
            }

            const std::string_view head(raw.data(), hdrEnd);
            if (!FindHeader(head, "Transfer-Encoding").empty()) chunked = true;

            const std::string_view cl = FindHeader(head, "Content-Length");
            if (!cl.empty())
            {
                contentLength = static_cast<size_t>(std::strtoull(
                    std::string(cl).c_str(), nullptr, 10));
            }
            if (contentLength > kMaxBodyBytes) return false;
        }

        const size_t have = raw.size() - (hdrEnd + 4);
        if (have >= contentLength) return true;

        const int n = ::recv(s, buf, sizeof(buf), 0);
        if (n <= 0) return false;       // peer hung up mid-body
        raw.append(buf, static_cast<size_t>(n));
    }
}

// ---- the server -----------------------------------------------------------

class Server {
public:
    bool Init(const std::wstring& modelDir) {
        dev_ = std::make_unique<dx12::Device>();
        // Device's constructor is defaulted; Initialize() is what enumerates the
        // adapter and records the VRAM budget. Skipping it leaves dedicatedVram_
        // at 0 and the VRAM admission gate rejects every model.
        dev_->Initialize();
        gpuName_ = dx12::DeviceName();
        vramTotal_ = dev_->DedicatedVideoMemory();

        lib_ = std::make_unique<shaders::Library>();
        lib_->Initialize(*dev_);

        // VoxelEngine owns the graph, every GPU buffer, the weight upload and the
        // tokenizer. Standing up a second copy of all of that here allocated a
        // second full weight buffer that nothing ever read, on top of the
        // engine's own. Model metadata is read back off the engine's graph.
        engine::EngineOptions opts;
        opts.enablePlasticity = false;
        engine_.Initialize(modelDir, opts, dev_.get(), lib_.get());
        metrics_ = std::make_unique<engine::MetricsCollector>(engine_);

        return true;
    }

    std::string HealthJson() const {
        const auto& cfg = engine_.GraphModel().Config();
        std::string j = "{";
        j += "\"status\":\"ok\",";
        j += "\"gpu\":" + JsonString(gpuName_) + ",";
        j += "\"architecture\":" + JsonString(cfg.architecture) + ",";
        j += "\"layers\":" + std::to_string(cfg.numLayers) + ",";
        j += "\"hidden_size\":" + std::to_string(cfg.hiddenSize) + ",";
        j += "\"num_heads\":" + std::to_string(cfg.numHeads) + ",";
        j += "\"num_kv_heads\":" + std::to_string(cfg.numKvHeads) + ",";
        j += "\"head_dim\":" + std::to_string(cfg.HeadDim()) + ",";
        j += "\"vocab_size\":" + std::to_string(cfg.vocabSize) + ",";
        j += "\"max_positions\":" + std::to_string(cfg.maxPositions) + ",";
        j += "\"vram_used_mib\":" + std::to_string(engine_.GraphModel().Plan().totalBytes >> 20) + ",";
        j += "\"vram_total_mib\":" + std::to_string(vramTotal_ >> 20);
        j += "}";
        return j;
    }

    std::string MetricsJson() {
        const engine::MetricsSnapshot s = metrics_->Snapshot();
        char buf[1024];
        std::snprintf(buf, sizeof(buf),
            "{\"kv_occupancy\":%.4f,\"logit_entropy\":%.4f,\"logit_norm\":%.4f,"
            "\"tokens_per_sec\":%.3f,\"memory_pressure\":%.4f,"
            "\"speculative_acceptance\":%.3f,\"max_attn_score\":%.4f}",
            static_cast<double>(s.kvOccupancy),   static_cast<double>(s.logitEntropy),
            static_cast<double>(s.logitNorm),     static_cast<double>(s.tokensPerSec),
            static_cast<double>(s.memoryPressure),static_cast<double>(s.speculativeAcceptance),
            static_cast<double>(s.maxAttnScore));
        return std::string(buf);
    }

    std::string GenerateJson(const std::string& prompt,
                             int maxTokens, float temperature)
    {
        std::vector<uint32_t> tokens;
        try
        {
            tokens = engine_.Tokenizer().Encode(prompt);
        }
        catch (const std::exception& e)
        {
            return std::string("{\"error\":") + JsonString(e.what()) + "}";
        }

        if (tokens.empty())
        {
            return R"({"error":"prompt tokenized to empty"})";
        }

        engine::EngineOptions opts;
        opts.enablePlasticity = false;
        opts.temperature = temperature;
        opts.maxNewTokens = static_cast<uint32_t>(maxTokens);

        try
        {
            const engine::GenerateResult r = engine_.Generate(tokens, opts);

            // Detokenize into text. A stop token terminates the string rather
            // than being decoded, so it never reaches the client as garbage.
            std::string text;
            for (const uint32_t tok : r.generated)
            {
                if (tok == engine_.Tokenizer().SpecialId("")) break;
                text += engine_.Tokenizer().Decode(tok);
            }

            std::string j = "{\"text\":";
            j += JsonString(text);
            j += ",\"tokens\":[";
            for (size_t i = 0; i < r.generated.size(); ++i)
            {
                if (i) j += ',';
                j += std::to_string(r.generated[i]);
            }
            j += "]";
            j += ",\"prompt_tokens\":" + std::to_string(r.promptTokens.size());
            j += ",\"generated_tokens\":" + std::to_string(r.generated.size());
            char perf[128];
            std::snprintf(perf, sizeof(perf),
                          ",\"ms_per_token\":%.4f,\"total_ms\":%.4f,\"hit_stop\":%s",
                          r.msPerToken, r.lastMs, r.hitStop ? "true" : "false");
            j += perf;
            j += "}";
            return j;
        }
        catch (const std::exception& e)
        {
            return std::string("{\"error\":") + JsonString(e.what()) + "}";
        }
    }

    void Handle(SOCKET s) {
        std::string raw;
        size_t contentLength = 0;
        bool chunked = false;
        if (!ReadRequest(s, raw, contentLength, chunked)) { ::closesocket(s); return; }

        const size_t hdrEnd = raw.find("\r\n\r\n");
        const std::string_view head(raw.data(), hdrEnd);
        // Slice exactly Content-Length bytes: a client that pipelines or pads
        // must not be able to smuggle extra bytes into the parser.
        const std::string body(raw.data() + hdrEnd + 4, contentLength);

        if (chunked)
        {
            SendJson(s, 400, R"({"error":"chunked transfer-encoding not supported; send Content-Length"})");
            ::closesocket(s);
            return;
        }

        // Request line: METHOD SP TARGET SP VERSION
        const size_t sp1 = head.find(' ');
        if (sp1 == std::string_view::npos) {
            SendJson(s, 400, R"({"error":"malformed request line"})");
            ::closesocket(s);
            return;
        }
        const std::string_view method = head.substr(0, sp1);

        const size_t sp2 = head.find(' ', sp1 + 1);
        const std::string_view target =
            (sp2 == std::string_view::npos) ? head.substr(sp1 + 1)
                                            : head.substr(sp1 + 1, sp2 - sp1 - 1);
        // Strip any query string; none of the routes use one.
        const std::string_view route = target.substr(0, target.find('?'));

        if (route == "/health" && method == "GET") {
            SendJson(s, 200, HealthJson());
        } else if (route == "/metrics" && method == "GET") {
            SendJson(s, 200, MetricsJson());
        } else if (route == "/generate" && method == "POST") {
            if (body.size() > kMaxBodyBytes) {
                SendJson(s, 413, R"({"error":"request body too large"})");
            } else {
                std::string prompt;
                int maxTokens = 64;
                float temperature = 0.0f;
                try {
                    const json::Value v = json::ParseObject(body, "/generate");
                    // Value exposes its scalar payload directly; the Get* helpers are keyed
                    // lookups into an *object*, so they cannot be used on the
                    // member Value that Find() already returned.
                    if (const json::Value* p = v.Find("prompt")) {
                        if (!p->IsString()) {
                            SendJson(s, 400, R"({"error":"\"prompt\" must be a string"})");
                            ::closesocket(s);
                            return;
                        }
                        prompt = p->str;
                    }
                    if (const json::Value* m = v.Find("max_tokens")) {
                        if (!m->IsNumber()) {
                            SendJson(s, 400, R"({"error":"\"max_tokens\" must be a number"})");
                            ::closesocket(s);
                            return;
                        }
                        maxTokens = static_cast<int>(m->inum);
                    }
                    if (const json::Value* t = v.Find("temperature")) {
                        if (!t->IsNumber()) {
                            SendJson(s, 400, R"({"error":"\"temperature\" must be a number"})");
                            ::closesocket(s);
                            return;
                        }
                        temperature = static_cast<float>(t->num);
                    }
                }
                catch (const std::exception& e) {
                    SendJson(s, 400, std::string("{\"error\":") + JsonString(e.what()) + "}");
                    ::closesocket(s);
                    return;
                }

                if (prompt.empty()) {
                    SendJson(s, 400, R"({"error":"missing \"prompt\""})");
                } else {
                    // An unvalidated max_tokens would let one request ask for
                    // millions of tokens and wedge the device for hours.
                    maxTokens = (maxTokens < 1) ? 1
                             : ((maxTokens > 2048) ? 2048 : maxTokens);
                    if (temperature < 0.0f) temperature = 0.0f;
                    SendJson(s, 200, GenerateJson(prompt, maxTokens, temperature));
                }
            }
        } else if (route == "/health" || route == "/metrics" || route == "/generate") {
            SendJson(s, 405, R"({"error":"method not allowed"})");
        } else {
            SendJson(s, 404, R"({"error":"no such route"})");
        }

        ::closesocket(s);
    }

private:
    std::unique_ptr<dx12::Device> dev_;
    std::unique_ptr<shaders::Library> lib_;
    // The engine owns the graph, the GPU buffers and the tokenizer.
    engine::VoxelEngine engine_;
    std::unique_ptr<engine::MetricsCollector> metrics_;

    std::string gpuName_;
    uint64_t vramTotal_ = 0;
};

} // namespace

int RunServer(int argc, char** argv)
{
    uint16_t port = 8080;
    std::string modelPath = ".\\models";

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc)
            port = static_cast<uint16_t>(std::stoi(argv[++i]));
        else if (arg == "--model" && i + 1 < argc)
            modelPath = argv[++i];
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "Usage: opure_server [--port N] [--model <path>]\n"
                      << "Routes: GET /health, POST /generate, GET /metrics\n";
            return 0;
        }
    }

    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
    {
        std::cerr << "WSAStartup failed\n";
        return 1;
    }

    Server server;
    if (!server.Init(std::filesystem::path(modelPath).wstring()))
    {
        WSACleanup();
        return 1;
    }

    const SOCKET listenSock = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listenSock == INVALID_SOCKET)
    {
        std::cerr << "socket() failed\n";
        WSACleanup();
        return 1;
    }

    // SO_REUSEADDR so a restart does not have to wait out TIME_WAIT.
    BOOL reuse = TRUE;
    ::setsockopt(listenSock, SOL_SOCKET, SO_REUSEADDR,
                 reinterpret_cast<const char*>(&reuse), sizeof(reuse));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);   // loopback only, by default
    addr.sin_port = htons(port);

    if (::bind(listenSock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR)
    {
        std::cerr << "bind() failed on port " << port << "\n";
        ::closesocket(listenSock);
        WSACleanup();
        return 1;
    }
    if (::listen(listenSock, SOMAXCONN) == SOCKET_ERROR)
    {
        std::cerr << "listen() failed\n";
        ::closesocket(listenSock);
        WSACleanup();
        return 1;
    }

    std::cout << "Opure server listening on http://127.0.0.1:" << port << "\n"
              << "  GET  /health\n"
              << "  POST /generate  {\"prompt\":\"...\",\"max_tokens\":64}\n"
              << "  GET  /metrics\n"
              << "Ctrl+C to stop.\n\n";

    for (;;)
    {
        const SOCKET client = ::accept(listenSock, nullptr, nullptr);
        if (client == INVALID_SOCKET) continue;   // a failed accept must not kill the loop
        server.Handle(client);
    }

    ::closesocket(listenSock);
    WSACleanup();
    return 0;
}

} // namespace opure

int main(int argc, char** argv)
{
    return opure::RunServer(argc, argv);
}