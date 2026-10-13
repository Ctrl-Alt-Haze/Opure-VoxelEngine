// ============================================================================
//  Opure_Graph.cpp
//  Opure-VoxelEngine -- model geometry and VRAM layout.
// ============================================================================
#include "engine/Opure_Graph.h"

#include "core/Opure_Dx12.h"
#include "core/Opure_Json.h"

#include <windows.h>

#include <cstring>
#include <string>

namespace opure::graph
{

namespace
{

// Reads a whole small file into memory. Used for config.json only; the model
// weights go through WeightStreamer's mapping instead.
std::string ReadSmallFile(const std::wstring& path)
{
    HANDLE f = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) { ThrowError("cannot open config.json in the model directory"); }

    LARGE_INTEGER size{};
    if (!::GetFileSizeEx(f, &size) || size.QuadPart <= 0)
    {
        ::CloseHandle(f);
        ThrowError("config.json is empty");
    }
    std::string text(static_cast<size_t>(size.QuadPart), '\0');
    DWORD got = 0;
    const BOOL ok = ::ReadFile(f, text.data(), static_cast<DWORD>(text.size()), &got, nullptr);
    ::CloseHandle(f);
    if (!ok) { ThrowError("ReadFile failed on config.json"); }
    text.resize(got);
    return text;
}

std::wstring JoinPath(const std::wstring& dir, const wchar_t* leaf)
{
    std::wstring p = dir;
    if (!p.empty() && p.back() != L'\\' && p.back() != L'/') { p.push_back(L'\\'); }
    p += leaf;
    return p;
}

uint32_t GetU32(const json::Value& o, const char* key, std::string_view ctx)
{
    return static_cast<uint32_t>(std::max<int64_t>(0, o.GetInt(key, ctx)));
}

uint64_t Align64(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// Two words appended to the KV allocation for the ArgMax result: [best value,
// best token id]. Kept out of the cache proper so a decode pass cannot corrupt
// a real KV entry by writing its result.
constexpr uint64_t kArgmaxTailWords = 2;

} // namespace

// ---------------------------------------------------------------------------
uint32_t ModelConfig::FloatBits(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// ---------------------------------------------------------------------------
ModelConfig ModelConfig::Parse(std::string_view jsonText)
{
    const json::Value root = json::ParseObject(jsonText, "config.json");
    constexpr std::string_view ctx = "config.json";

    ModelConfig c;
    c.modelType    = root.GetStr("model_type", ctx);
    if (const json::Value* a = root.Find("architectures");
        a != nullptr && a->IsArray() && !a->arr.empty() && a->arr[0].IsString())
    {
        c.architecture = a->arr[0].str;
    }
    c.hiddenAct     = root.GetStr("hidden_act", ctx);
    c.vocabSize     = GetU32(root, "vocab_size", ctx);
    c.hiddenSize    = GetU32(root, "hidden_size", ctx);
    c.intermediateSize = GetU32(root, "intermediate_size", ctx);
    c.numLayers     = GetU32(root, "num_hidden_layers", ctx);
    c.numHeads      = GetU32(root, "num_attention_heads", ctx);
    c.maxPositions  = GetU32(root, "max_position_embeddings", ctx);

    // num_key_value_heads is absent on MHA models; defaulting to num_attention_heads
    // is what HF does, so this is not an invented fallback.
    if (const json::Value* v = root.Find("num_key_value_heads"))
    {
        c.numKvHeads = static_cast<uint32_t>(std::max<int64_t>(0, v->inum));
    }
    else
    {
        c.numKvHeads = c.numHeads;
    }

    if (const json::Value* v = root.Find("rope_theta"))
    {
        c.ropeTheta = v->num;
    }
    else
    {
        c.ropeTheta = 10000.0;
    }

    c.rmsNormEpsBits = FloatBits(static_cast<float>(root.GetNum("rms_norm_eps", ctx)));

    c.tieWordEmbeddings = root.GetBool("tie_word_embeddings", true, ctx);
    c.attentionBias     = root.GetBool("attention_bias", false, ctx);
    c.mlpBias           = root.GetBool("mlp_bias", false, ctx);

    if (const json::Value* t = root.Find("torch_dtype"); t != nullptr && t->IsString())
    {
        c.torchDtype = t->str;
    }

    return c;
}

// ---------------------------------------------------------------------------
bool ModelConfig::Validate(std::string& err) const
{
    if (hiddenSize == 0 || numHeads == 0 || numLayers == 0 || vocabSize == 0)
    {
        err = "config.json is missing a required dimension";
        return false;
    }
    if (hiddenSize % numHeads != 0)
    {
        err = "hidden_size " + std::to_string(hiddenSize) +
              " is not divisible by num_attention_heads " + std::to_string(numHeads);
        return false;
    }
    if (numHeads % numKvHeads != 0)
    {
        err = "num_attention_heads " + std::to_string(numHeads) +
              " is not divisible by num_key_value_heads " + std::to_string(numKvHeads);
        return false;
    }
    const uint32_t hd = HeadDim();
    if (hd != 64 && hd != 128)
    {
        // Only two attention entry points are compiled, so anything else has
        // no pipeline. Failing here beats dispatching to the wrong one.
        err = "head_dim " + std::to_string(hd) +
              " has no compiled attention entry point (only 64 and 128 exist)";
        return false;
    }
    if (intermediateSize == 0)
    {
        err = "intermediate_size must be non-zero";
        return false;
    }
    if (maxPositions == 0)
    {
        err = "max_position_embeddings must be non-zero";
        return false;
    }
    if (hiddenAct != "silu")
    {
        err = "hidden_act '" + hiddenAct + "' is not implemented (only silu/SwiGLU is)";
        return false;
    }
    if (attentionBias || mlpBias)
    {
        err = "bias terms are not implemented by this engine";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
uint64_t ModelConfig::WeightBytes() const
{
    const uint64_t h  = hiddenSize;
    const uint64_t ff = intermediateSize;
    const uint64_t kvh = numKvHeads * HeadDim();
    const uint64_t layers = numLayers;

    // Per layer: 2 norms [h] + q [h,h] + k [kvh,h] + v [kvh,h] + o [h,h]
    //                     + gate [ff,h] + up [ff,h] + down [h,ff]
    const uint64_t perLayer =
        2 * h + (h * h) + 2 * (kvh * h) + (h * h) + (ff * h) * 3;

    // Embedding + final norm. lm_head is absent when embeddings are tied, which
    // is the common case and saves vocabSize*hiddenSize bytes.
    const uint64_t embed = static_cast<uint64_t>(vocabSize) * h;
    const uint64_t fnorm = h;

    return (perLayer * layers + embed + fnorm) * 4ull;
}

uint64_t ModelConfig::TotalVramBytes() const
{
    const uint64_t weights = WeightBytes();

    // Activation banks hold every scratch vector a layer needs at once. The
    // widest is the logits vector, which is vocabSize words; the rest are
    // hidden and intermediate sized. Getting this wrong makes Initialize()
    // throw, which is why the engine asserts against the same figure.
    const uint64_t kvHidden = static_cast<uint64_t>(numKvHeads) * HeadDim();
    const uint64_t actWords =
        static_cast<uint64_t>(hiddenSize) * 4 +      // q, attn out, proj, spare
        kvHidden * 2 +                               // k, v
        static_cast<uint64_t>(intermediateSize) * 3 +// gate, up, swiglu out
        vocabSize;                                   // logits
    // Plasticity staging, appended to every activation bank. See
    // VramPlan::plasticityWords.
    const uint64_t plasticityWords =
        std::max<uint64_t>({ static_cast<uint64_t>(hiddenSize) * hiddenSize,
                             kvHidden * hiddenSize,
                             static_cast<uint64_t>(intermediateSize) * hiddenSize,
                             static_cast<uint64_t>(hiddenSize) * intermediateSize });
    const uint64_t actWithStaging =
        Align64((actWords + plasticityWords) * 4, 4096);

    const uint64_t kvBytes = Align64(KvCacheWords() * 4, 4096);

    // Three activation buffers: the ping-pong pair plus the residual bank. See
    // Opure_VoxelEngine.cpp for why two is not enough.
    return weights + actWithStaging * 3 + kvBytes;
}

// ---------------------------------------------------------------------------
Graph::~Graph() { Release(); }

void Graph::Release()
{
    stream_.Close();
    loaded_ = false;
}

void Graph::Load(const std::wstring& modelDir, uint64_t availableVramBytes)
{
    LoadConfigOnly(modelDir, availableVramBytes);
}

// Plans and validates only; does not touch the GPU. The engine calls this once
// to size its buffers, then UploadWeights() once they exist.
void Graph::LoadConfigOnly(const std::wstring& modelDir, uint64_t availableVramBytes)
{
    Release();

    const std::string cfgText = ReadSmallFile(JoinPath(modelDir, L"config.json"));
    cfg_ = ModelConfig::Parse(cfgText);

    std::string err;
    if (!cfg_.Validate(err))
    {
        ThrowError("model configuration rejected: " + err);
    }

    stream_.Open(JoinPath(modelDir, L"model.safetensors"));
    PlanLayout();

    plan_.availableBytes = availableVramBytes;
    plan_.totalBytes     = plan_.weightBytes + plan_.actBytes * 3 + plan_.kvBytes;

    // Admission check. Refusing here is much better than letting
    // CreateCommittedResource fail deep inside a token's first dispatch.
    plan_.fits = plan_.totalBytes <= availableVramBytes;
    if (!plan_.fits)
    {
        plan_.headroom = plan_.totalBytes - availableVramBytes;
    }

    loaded_ = true;
}

void Graph::PlanLayout()
{
    plan_ = VramPlan{};
    const uint32_t h   = cfg_.hiddenSize;
    const uint32_t ff  = cfg_.intermediateSize;
    const uint32_t hd  = cfg_.HeadDim();
    const uint32_t kvh = cfg_.numKvHeads * hd;

    uint64_t cursor = 0;
    auto place = [&](const safetensors::Tensor& t) -> uint64_t
    {
        const uint64_t at = cursor;
        // Tensors are 4-byte aligned by construction; 256 keeps every row of a
        // GEMV 64-byte-segment friendly.
        cursor += Align64(t.ElementCount() * 4, 256);
        return at / 4;   // word offset
    };

    plan_.embedTokens = place(stream_.Find("model.embed_tokens.weight"));
    plan_.finalNorm   = place(stream_.Find("model.norm.weight"));

    plan_.layers.resize(cfg_.numLayers);
    for (uint32_t l = 0; l < cfg_.numLayers; ++l)
    {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        VramPlan::LayerTensors& lt = plan_.layers[l];
        lt.inputNorm   = place(stream_.Find(p + "input_layernorm.weight"));
        lt.postAttNorm = place(stream_.Find(p + "post_attention_layernorm.weight"));
        lt.wq          = place(stream_.Find(p + "self_attn.q_proj.weight"));
        lt.wk          = place(stream_.Find(p + "self_attn.k_proj.weight"));
        lt.wv          = place(stream_.Find(p + "self_attn.v_proj.weight"));
        lt.wo          = place(stream_.Find(p + "self_attn.o_proj.weight"));
        lt.gate        = place(stream_.Find(p + "mlp.gate_proj.weight"));
        lt.up          = place(stream_.Find(p + "mlp.up_proj.weight"));
        lt.down        = place(stream_.Find(p + "mlp.down_proj.weight"));
    }

    plan_.weightWords = cursor;
    plan_.weightBytes = cursor * 4;

    // Plasticity staging. The SignSGD pass may not read O_W and write O_WU in
    // one dispatch (D3D12 leaves that aliasing undefined), so the updated matrix
    // is computed into a scratch range first and copied back by a second
    // dispatch. The range lives in the tail of activation bank A, which the
    // inference graph never touches, so it costs nothing at run time and needs
    // no allocation of its own. Sized for the widest projection the model has.
    const uint64_t kvHidden = static_cast<uint64_t>(cfg_.numKvHeads) * hd;
    plan_.plasticityWords = std::max<uint64_t>({ h * h,
                                                 kvHidden * h,
                                                 ff * h,
                                                 static_cast<uint64_t>(h) * ff });

    // Activation scratch. Sized from the same expression as
    // ModelConfig::TotalVramBytes, because the engine asserts against it.
    plan_.actWords =
        static_cast<uint64_t>(h) * 4 + kvHidden * 2 +
        static_cast<uint64_t>(ff) * 3 + cfg_.vocabSize;
    plan_.plasticityOffset = plan_.actWords;
    plan_.actWords += plan_.plasticityWords;
    plan_.actBytes  = Align64(plan_.actWords * 4, 4096);
    plan_.act2Words = plan_.actWords;

    // The KV cache is sized for the model, and a small tail is appended for the
    // ArgMax result. WITHOUT that tail the argmax would land on the last real
    // cache entry and overwrite it, corrupting attention for the next token.
    plan_.kvWords = cfg_.KvCacheWords() + kArgmaxTailWords;
    plan_.kvBytes = Align64(plan_.kvWords * 4, 4096);

    OPURE_CHECK(h > 0 && ff > 0 && hd > 0, "graph planning produced a zero dimension");
    OPURE_CHECK(kvh == cfg_.numKvHeads * hd, "kv hidden size overflow");
}

Graph::AttEntry Graph::AttentionEntryPoint() const
{
    const uint32_t hd = cfg_.HeadDim();
    if (hd == 64)  { return AttEntry::Hd64; }
    if (hd == 128) { return AttEntry::Hd128; }
    // Validate() already rejects this, so reaching here means Load() was
    // bypassed.
    ThrowError("head_dim " + std::to_string(hd) + " has no attention entry point");
}

// Copies every weight into the GPU weight buffer in planned order.
//
// The source is the memory-mapped file itself, so nothing is staged on the host
// beyond the upload buffer inside Device::UploadRanges. One command list per
// 32 MB batch, not one per tensor.
void Graph::UploadWeights(dx12::Device& dev, dx12::GpuBuffer& weights)
{
    struct Piece { uint64_t wordOffset; std::string name; };
    std::vector<Piece> pieces;
    pieces.reserve(2 + plan_.layers.size() * 9);

    pieces.push_back({ plan_.embedTokens, "model.embed_tokens.weight" });
    pieces.push_back({ plan_.finalNorm,   "model.norm.weight" });

    for (size_t l = 0; l < plan_.layers.size(); ++l)
    {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        const graph::VramPlan::LayerTensors& t = plan_.layers[l];
        pieces.push_back({ t.inputNorm,   p + "input_layernorm.weight" });
        pieces.push_back({ t.postAttNorm, p + "post_attention_layernorm.weight" });
        pieces.push_back({ t.wq, p + "self_attn.q_proj.weight" });
        pieces.push_back({ t.wk, p + "self_attn.k_proj.weight" });
        pieces.push_back({ t.wv, p + "self_attn.v_proj.weight" });
        pieces.push_back({ t.wo, p + "self_attn.o_proj.weight" });
        pieces.push_back({ t.gate, p + "mlp.gate_proj.weight" });
        pieces.push_back({ t.up,   p + "mlp.up_proj.weight" });
        pieces.push_back({ t.down, p + "mlp.down_proj.weight" });
    }

    std::vector<dx12::Device::UploadRange> ranges;
    ranges.reserve(pieces.size());
    for (const Piece& pc : pieces)
    {
        const safetensors::Tensor& t = stream_.Find(pc.name);
        ranges.push_back({ stream_.Data(t), t.ByteCount(), pc.wordOffset * 4 });
    }

    dev.UploadRanges(weights, ranges);
}

} // namespace opure::graph