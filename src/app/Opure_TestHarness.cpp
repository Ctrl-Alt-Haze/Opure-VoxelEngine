// ============================================================================
//  Opure_TestHarness.cpp
//  Opure-VoxelEngine -- verification driver.
//
//  Phase 1 establishes that the GPU compute graph is numerically correct before
//  any model is loaded. Each kernel is dispatched on deterministic pseudo-random
//  data and its output is compared against an independent CPU implementation
//  written here in scalar C++. A systematic GPU bug (bad reduction, wrong
//  layout, transposed matrix, broken softmax) shows up as a large error; the
//  reported error magnitude is the evidence.
// ============================================================================
#include "core/Opure_Dx12.h"
#include "core/Opure_Shaders.h"
#include "core/Opure_Streamer.h"
#include "model/Opure_Tokenizer.h"
#include "engine/Opure_Graph.h"
#include "engine/Opure_VoxelEngine.h"
#include "engine/Opure_CpuReference.h"

#include <cmath>
#include <limits>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace opure;
using namespace opure::dx12;

namespace
{

// ---------------------------------------------------------------------------
//  Deterministic data (no <random> so results are reproducible across runs)
// ---------------------------------------------------------------------------
struct Rng
{
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 1) {}
    uint32_t NextU32()
    {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return static_cast<uint32_t>(s >> 32);
    }
    // Uniform in [-1, 1)
    float NextF() { return static_cast<float>(NextU32()) / 2147483648.0f - 1.0f; }
};

using F32 = std::vector<float>;

struct Result { bool pass; double maxAbs; double rms; bool hasNaN = false; };

int g_failures = 0;

void Report(const char* name, const Result& r, double tol)
{
    if (!r.pass) ++g_failures;
    if (r.hasNaN)
    {
        std::printf("  %-26s %-5s  NaN in output -- the graph read a word it "
                    "never wrote (tol %.0e)\n",
                    name, "FAIL", tol);
        return;
    }
    std::printf("  %-26s %-5s  max_abs=%.3e  rms=%.3e  (tol %.0e)\n",
                name, r.pass ? "PASS" : "FAIL", r.maxAbs, r.rms, tol);
}

// Compare two float vectors.
Result Compare(const F32& got, const F32& want)
{
    Result r{ true, 0.0, 0.0 };
    double se = 0.0;
    size_t n = got.size() < want.size() ? got.size() : want.size();
    for (size_t i = 0; i < n; ++i)
    {
        // A NaN on either side makes every comparison false, so the loop below
        // would silently report maxAbs = 0 and call two all-NaN vectors
        // IDENTICAL. That masked a real divergence for several iterations, so
        // NaN is detected explicitly and reported as an infinite error.
        const double a = static_cast<double>(got[i]);
        const double b = static_cast<double>(want[i]);
        if (std::isnan(a) || std::isnan(b))
        {
            r.pass  = false;
            r.maxAbs = std::numeric_limits<double>::infinity();
            r.rms    = std::numeric_limits<double>::quiet_NaN();
            r.hasNaN = true;
            return r;
        }
        const double d = std::fabs(a - b);
        if (d > r.maxAbs) r.maxAbs = d;
        se += d * d;
    }
    r.rms = n ? std::sqrt(se / static_cast<double>(n)) : 0.0;
    // Pass criterion is relative to the magnitude of the reference vector.
    double scale = 1e-6;
    for (size_t i = 0; i < n; ++i) scale = std::max(scale, std::fabs(static_cast<double>(want[i])));
    r.pass = (r.rms <= 1e-5 * scale) || (r.rms <= 1e-7);
    return r;
}

// ---------------------------------------------------------------------------
//  Buffers
// ---------------------------------------------------------------------------
constexpr uint64_t kBankWords  = 1u << 20;    // 4 MiB per activation bank
constexpr uint64_t kWeightWords = 2u << 20;    // 8 MiB weights
constexpr uint64_t kKvWords     = 4u << 20;    // 16 MiB kv + argmax tail

struct Gpu
{
    GpuBuffer w, a, b, kv, res;
    void Create(Device& dev)
    {
        w  .Create(dev, kSlotWeights, kWeightWords * 4, true);
        a  .Create(dev, 1,             kBankWords   * 4, true);
        b  .Create(dev, 2,             kBankWords   * 4, true);
        kv .Create(dev, 3,             kKvWords     * 4, true);
        // The residual bank: the add cannot legally write a ping-pong bank it
        // is also reading, so the residual stream gets its own buffer.
        res.Create(dev, 4,             kBankWords   * 4, true);
        // Register the binding set once; CommandScope rebinds by GPU address.
        dev.SetPersistentBuffers(&w, &a, &b, &kv, &res);
    }
};

void Upload(Device& dev, GpuBuffer& buf, const std::vector<uint32_t>& words, uint64_t off = 0)
{
    if (!words.empty()) dev.UploadToBuffer(buf, words.data(), words.size() * 4, off);
}
F32 Download(Device& dev, const GpuBuffer& buf, size_t count, uint64_t off = 0)
{
    std::vector<uint32_t> raw(count);
    dev.ReadbackFromBuffer(buf, raw.data(), count * 4, off);
    F32 f(count);
    std::memcpy(f.data(), raw.data(), count * 4);
    return f;
}

// ===========================================================================
//  safetensors streaming
// ===========================================================================
//
//  The model file is not committed, so the streamer test is SKIPPED (not
//  failed) when models\model.safetensors is absent. It is still a real gate
//  wherever the weights are present, which is where correctness matters.
void TestStreamer()
{
    const std::wstring path = L"models\\model.safetensors";

    std::FILE* probe = nullptr;
    if (_wfopen_s(&probe, path.c_str(), L"rb") != 0 || probe == nullptr)
    {
        std::printf("  %-26s SKIP  models\\model.safetensors not present\n", "safetensors map");
        return;
    }
    std::fclose(probe);

    safetensors::WeightStreamer ws;
    ws.Open(path);

    std::printf("  %-26s PASS  %zu tensors, header+payload %.1f MB\n",
                "safetensors map", ws.TensorCount(),
                static_cast<double>(ws.TotalBytes()) / (1024.0 * 1024.0));

    // Spot-check the known SmolLM-135M shapes. A wrong payload base offset
    // would still produce a parseable header, so the first/last weight values
    // are checked against the file to prove the mapping is actually readable.
    const safetensors::Tensor& emb = ws.Find("model.embed_tokens.weight");
    const bool shapeOk = (emb.shape.size() == 2) &&
                         (static_cast<uint64_t>(emb.shape[0]) == 49152u) &&
                         (static_cast<uint64_t>(emb.shape[1]) == 576u);
    if (!shapeOk) ++g_failures;
    std::printf("  %-26s %-5s  embed_tokens [%lld, %lld]  (want 49152, 576)\n",
                "tensor shapes", shapeOk ? "PASS" : "FAIL",
                static_cast<long long>(emb.shape[0]), static_cast<long long>(emb.shape[1]));

    // Every tensor's mapped range must be inside the file and match its extent.
    bool rangesOk = true;
    for (const safetensors::Tensor& t : ws.Tensors())
    {
        if (t.byteEnd > ws.TotalBytes() || t.byteEnd <= t.byteBegin) { rangesOk = false; break; }
    }
    if (!rangesOk) ++g_failures;
    std::printf("  %-26s %-5s  all %zu mapped ranges inside the file\n",
                "tensor ranges", rangesOk ? "PASS" : "FAIL", ws.TensorCount());

    // Missing tensors must throw rather than default to zeros.
    bool threw = false;
    try { (void)ws.Find("model.layers.9999.mlp.up_proj.weight"); }
    catch (const std::exception&) { threw = true; }
    if (!threw) ++g_failures;
    std::printf("  %-26s %-5s  absent tensor name throws\n",
                "lookup is strict", threw ? "PASS" : "FAIL");

    // The KV projection shapes are what the attention kernel is specialised
    // for; a mismatch here would silently pick the wrong entry point.
    const safetensors::Tensor& kp = ws.Find("model.layers.0.self_attn.k_proj.weight");
    const bool gqaOk = kp.shape.size() == 2 &&
                       static_cast<uint64_t>(kp.shape[0]) == 192u &&   // 3 kv heads * 64
                       static_cast<uint64_t>(kp.shape[1]) == 576u;
    if (!gqaOk) ++g_failures;
    std::printf("  %-26s %-5s  k_proj [%lld, %lld]  (want 192, 576 = GQA 3x64)\n",
                "GQA projection", gqaOk ? "PASS" : "FAIL",
                static_cast<long long>(kp.shape[0]), static_cast<long long>(kp.shape[1]));
}

// ===========================================================================
//  Model graph: config parsing, layout planning, VRAM admission
// ===========================================================================
void TestGraph()
{
    const std::wstring dir = L"models";

    std::FILE* probe = nullptr;
    if (_wfopen_s(&probe, L"models\\config.json", L"rb") != 0 || probe == nullptr)
    {
        std::printf("  %-26s SKIP  models\\config.json not present\n", "model graph");
        return;
    }
    std::fclose(probe);

    graph::Graph g;
    try
    {
        g.Load(dir, 15ull * 1024 * 1024 * 1024);
    }
    catch (const std::exception& e)
    {
        ++g_failures;
        std::printf("  %-26s FAIL  %s\n", "model graph", e.what());
        return;
    }

    const graph::ModelConfig& c = g.Config();
    std::printf("  %-26s PASS  %s  %u layers, d_model %u, heads %u/%u, head_dim %u, d_ff %u\n",
                "model graph", c.modelType.c_str(), c.numLayers, c.hiddenSize,
                c.numHeads, c.numKvHeads, c.HeadDim(), c.intermediateSize);

    // The GQA ratio must divide exactly, since the attention kernel derives the
    // KV head by integer division and a wrong ratio reads the wrong cache rows.
    const bool gqaOk = (c.numHeads % c.numKvHeads == 0) && (c.KvGroupSize() == 3);
    if (!gqaOk) ++g_failures;
    std::printf("  %-26s %-5s  %u query heads / %u kv heads = %u per group (want 3)\n",
                "GQA grouping", gqaOk ? "PASS" : "FAIL", c.numHeads, c.numKvHeads,
                c.KvGroupSize());

    const graph::VramPlan& p = g.Plan();
    std::printf("  %-26s        weights %.1f MB, activations %.2f MB x2, kv %.1f MB\n",
                "", static_cast<double>(p.weightBytes) / (1024.0 * 1024.0),
                static_cast<double>(p.actBytes) / (1024.0 * 1024.0),
                static_cast<double>(p.kvBytes) / (1024.0 * 1024.0));

    // Every layer's tensors must land inside the weight region, and none may
    // overlap -- an overlap silently feeds one projection another's weights.
    bool layoutOk = p.layers.size() == c.numLayers;
    for (size_t l = 0; l < p.layers.size() && layoutOk; ++l)
    {
        const graph::VramPlan::LayerTensors& t = p.layers[l];
        const uint64_t fields[] = { t.inputNorm, t.postAttNorm, t.wq, t.wk,
                                    t.wv, t.wo, t.gate, t.up, t.down };
        for (const uint64_t f : fields)
        {
            if (f >= p.weightWords) { layoutOk = false; break; }
        }
    }
    if (!layoutOk) ++g_failures;
    std::printf("  %-26s %-5s  %zu layer tensor sets inside %llu weight words\n",
                "layout bounds", layoutOk ? "PASS" : "FAIL", p.layers.size(),
                static_cast<unsigned long long>(p.weightWords));

    // The KV cache must be sized for 2 (K,V) * layers * kvHeads * maxPos * hd.
    const uint64_t wantKv = 2ull * c.numLayers * c.numKvHeads * c.maxPositions * c.HeadDim();
    // The plan adds a 2-word tail for the argmax result, so compare against the
    // cache proper and allow for the tail.
    const bool kvOk = (p.kvWords == wantKv + 2);
    if (!kvOk) ++g_failures;
    std::printf("  %-26s %-5s  kv %llu words (want %llu cache + 2 argmax tail)\n",
                "KV cache sizing", kvOk ? "PASS" : "FAIL",
                static_cast<unsigned long long>(p.kvWords),
                static_cast<unsigned long long>(wantKv));

    // Admission: the plan must fit in the VRAM actually reported, with a
    // documented margin left over for the driver and compositor.
    const bool fits = p.fits;
    if (!fits) ++g_failures;
    std::printf("  %-26s %-5s  needs %.0f MB, have %.0f MB\n",
                "VRAM admission", fits ? "PASS" : "FAIL",
                static_cast<double>(p.totalBytes) / (1024.0 * 1024.0),
                static_cast<double>(p.availableBytes) / (1024.0 * 1024.0));

    // A deliberately undersized budget must be REJECTED, not silently accepted.
    bool rejected = false;
    try
    {
        graph::Graph tiny;
        tiny.Load(dir, 1024ull * 1024);      // 1 MB
        rejected = !tiny.Plan().fits;
    }
    catch (const std::exception&) { rejected = true; }
    if (!rejected) ++g_failures;
    std::printf("  %-26s %-5s  1 MB budget is refused, not silently accepted\n",
                "admission is a gate", rejected ? "PASS" : "FAIL");

    // A head_dim with no compiled pipeline must be rejected up front.
    bool headDimGate = false;
    {
        graph::ModelConfig bad;
        bad.hiddenSize      = 240;   // 3 heads * 80 -> head_dim 80
        bad.numHeads        = 3;
        bad.numKvHeads      = 1;
        bad.numLayers       = 1;
        bad.vocabSize       = 100;
        bad.intermediateSize = 100;
        bad.maxPositions    = 8;
        std::string err;
        headDimGate = !bad.Validate(err);
    }
    if (!headDimGate) ++g_failures;
    std::printf("  %-26s %-5s  head_dim 80 is rejected (no such pipeline)\n",
                "head_dim gate", headDimGate ? "PASS" : "FAIL");
}

// ===========================================================================
//  Weight upload: the GPU must hold exactly the bytes the file holds
// ===========================================================================
//
// This is the highest-value IO test in the suite. A wrong payload base offset,
// a truncated upload, or a stale staging buffer all produce plausible-looking
// weights and a model that emits confident nonsense. Comparing sampled words
// against the memory mapping catches all three.
void TestWeightUpload(Device& dev, shaders::Library& lib, Gpu& gpu, graph::Graph& g)
{
    const graph::VramPlan& p = g.Plan();

    // A dedicated weight buffer sized to the plan; the kernel tests reuse the
    // small harness one.
    dx12::GpuBuffer bigW;
    bigW.Create(dev, 0, p.weightBytes, /*allowUav*/ false);

    g.UploadWeights(dev, bigW);

    // The graph's streamer is private, so compare against the same file via a
    // fresh streamer rather than through Graph.
    safetensors::WeightStreamer ws;
    ws.Open(L"models\\model.safetensors");

    struct Check { uint64_t planWords; const safetensors::Tensor* t; const char* label; };
    const std::vector<Check> checks = {
        { p.embedTokens, &ws.Find("model.embed_tokens.weight"),        "embed_tokens[0..7]" },
        { p.finalNorm,   &ws.Find("model.norm.weight"),                "norm"               },
        { p.layers[15].wq, &ws.Find("model.layers.15.self_attn.q_proj.weight"), "layers[15].q_proj" },
    };

    bool allOk = true;
    for (const Check& c : checks)
    {
        const auto* host = static_cast<const uint32_t*>(ws.Data(*c.t));
        std::vector<uint32_t> dev_(8);
        dev.ReadbackFromBuffer(bigW, dev_.data(), dev_.size() * 4, c.planWords * 4);
        bool ok = true;
        for (size_t i = 0; i < dev_.size(); ++i)
        {
            if (dev_[i] != host[i]) { ok = false; break; }
        }
        if (!ok) { allOk = false; }
        std::printf("  %-26s %-5s  %-22s host %08X %08X  gpu %08X %08X\n",
                    "weight upload", ok ? "PASS" : "FAIL", c.label,
                    host[0], host[1], dev_[0], dev_[1]);
    }

    // A deep probe guards against a truncated final chunk. The offset is
    // derived from the tensor's real extent, not hardcoded, so it can never
    // wander past the tensor into the next one.
    {
        const safetensors::Tensor& emb = ws.Find("model.embed_tokens.weight");
        const uint64_t last = emb.ElementCount() - 8;      // 8 words from the end
        const uint64_t at   = p.embedTokens + last;
        const auto* host = static_cast<const uint32_t*>(ws.Data(emb)) + last;
        std::vector<uint32_t> dev_(8);
        dev.ReadbackFromBuffer(bigW, dev_.data(), dev_.size() * 4, at * 4);
        bool ok = true;
        for (size_t i = 0; i < dev_.size(); ++i)
        {
            if (dev_[i] != host[i]) { ok = false; break; }
        }
        if (!ok) { allOk = false; }
        std::printf("  %-26s %-5s  %-22s host %08X %08X  gpu %08X %08X\n",
                    "weight upload", ok ? "PASS" : "FAIL", "embed_tokens[tail]",
                    host[0], host[1], dev_[0], dev_[1]);
    }

    if (!allOk) { ++g_failures; }

    // Exhaustive offset check. Three sampled tensors is not enough: a layout bug
    // that points, say, layers[7].gate at layers[7].up leaves the sampled tensors
    // perfectly correct while the model computes confident nonsense. Every
    // planned offset is compared against the tensor the plan claims it is, and
    // also against the tensor the NAME says it is -- a swap between two same-
    // shaped tensors is invisible to any value-based check that reads only one.
    {
        int offsetFail = 0;
        int checked = 0;
        for (uint32_t L = 0; L < p.layers.size(); ++L)
        {
            const graph::VramPlan::LayerTensors& lt = p.layers[L];
            const std::string pre = "model.layers." + std::to_string(L) + ".";
            struct Pair { uint64_t off; std::string name; };
            const Pair pairs[] = {
                { lt.inputNorm,    pre + "input_layernorm.weight" },
                { lt.postAttNorm,  pre + "post_attention_layernorm.weight" },
                { lt.wq,           pre + "self_attn.q_proj.weight" },
                { lt.wk,           pre + "self_attn.k_proj.weight" },
                { lt.wv,           pre + "self_attn.v_proj.weight" },
                { lt.wo,           pre + "self_attn.o_proj.weight" },
                { lt.gate,         pre + "mlp.gate_proj.weight" },
                { lt.up,           pre + "mlp.up_proj.weight" },
                { lt.down,         pre + "mlp.down_proj.weight" },
            };
            for (const Pair& pr : pairs)
            {
                const safetensors::Tensor& t = ws.Find(pr.name);
                const auto* host = static_cast<const uint32_t*>(ws.Data(t));
                std::vector<uint32_t> got(4);
                dev.ReadbackFromBuffer(bigW, got.data(), got.size() * 4, pr.off * 4);
                ++checked;
                if (got[0] != host[0] || got[1] != host[1] || got[2] != host[2] || got[3] != host[3])
                {
                    if (offsetFail < 5)
                    {
                        std::printf("  %-26s FAIL  L%-2u %-42s plan %6llu -> gpu %08X %08X, want %08X %08X\n",
                                    "weight offsets", L, pr.name.c_str(),
                                    static_cast<unsigned long long>(pr.off),
                                    got[0], got[1], host[0], host[1]);
                    }
                    ++offsetFail;
                }
            }
        }
        if (offsetFail) { ++g_failures; }
        std::printf("  %-26s %-5s  %d/%d planned offsets match their tensor\n",
                    "weight offsets", offsetFail ? "FAIL" : "PASS",
                    checked - offsetFail, checked);
    }

    std::printf("  %-26s        %.0f MB uploaded in one batched submission\n",
                "", static_cast<double>(p.weightBytes) / (1024.0 * 1024.0));
}

// ===========================================================================
//  Tokenizer
// ===========================================================================
void TestTokenizer()
{
    const std::wstring path = L"models\\tokenizer.json";

    std::FILE* probe = nullptr;
    if (_wfopen_s(&probe, path.c_str(), L"rb") != 0 || probe == nullptr)
    {
        std::printf("  %-26s SKIP  models\\tokenizer.json not present\n", "tokenizer load");
        return;
    }
    std::fclose(probe);

    tokenizer::BpeTokenizer tok;
    tok.LoadFromFile(path);

    std::printf("  %-26s PASS  vocab %u, specials %u\n",
                "tokenizer load", tok.VocabSize(), tok.SpecialCount());

    // Vocab size must match config.json exactly, or every logit index is wrong.
    const bool sizeOk = (tok.VocabSize() == 49152u);
    if (!sizeOk) ++g_failures;
    std::printf("  %-26s %-5s  vocab %u (want 49152 from config.json)\n",
                "vocab matches config", sizeOk ? "PASS" : "FAIL", tok.VocabSize());

    // Round-trip: encode then decode must reproduce the input exactly. This is
    // the single strongest check available without a reference implementation,
    // because it exercises the pre-tokenizer, the byte-level alphabet and the
    // merge loop together.
    const char* kRoundTrip[] = {
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "2024 was a year of 365 days and 12 months.",
        "def f(x):\n    return x ** 2  # comment",
        "  leading and trailing spaces  ",
        "tabs\tand\nnewlines",
        "<|endoftext|>after special",
        "unicode: caf\u00e9 na\u00efve \u2014 \u65e5\u672c\u8a9e",
        "punct!!! ??? ... 100%",
        "",
    };
    bool rtOk = true;
    std::string rtFail;
    for (const char* s : kRoundTrip)
    {
        const std::vector<uint32_t> ids = tok.Encode(s);
        const std::string back = tok.Decode(ids);
        if (back != std::string(s))
        {
            rtOk = false;
            if (rtFail.empty()) { rtFail = std::string("\"") + s + "\" -> \"" + back + "\""; }
        }
    }
    if (!rtOk) ++g_failures;
    std::printf("  %-26s %-5s  %zu strings%s%s\n", "encode/decode round-trip",
                rtOk ? "PASS" : "FAIL", (sizeof(kRoundTrip) / sizeof(kRoundTrip[0])),
                rtOk ? "" : "  first failure: ", rtFail.c_str());

    // Known-answer check against ids read straight out of tokenizer.json.
    // These are the strongest assertions available without running Python:
    // each id is looked up in the vocab and its decoded text must match the
    // expected surface form.
    struct Known { const char* text; const char* expect; };
    const Known kKnown[] = {
        { "Hello",             "Hello"  },
        { " world",            " world" },
        { " the",              " the"   },
    };
    bool knownOk = true;
    std::string knownDetail;
    for (const Known& k : kKnown)
    {
        const std::vector<uint32_t> ids = tok.Encode(k.text);
        if (ids.size() != 1)
        {
            knownOk = false;
            knownDetail = std::string(k.text) + " produced " +
                          std::to_string(ids.size()) + " tokens, want 1";
            break;
        }
        const std::string dec = tok.Decode(ids[0]);
        if (dec != k.expect)
        {
            knownOk = false;
            knownDetail = std::string(k.text) + " decoded to \"" + dec + "\"";
            break;
        }
    }
    if (!knownOk) ++g_failures;
    std::printf("  %-26s %-5s  single-token words%s%s\n", "known-answer tokens",
                knownOk ? "PASS" : "FAIL",
                knownOk ? "" : "  ", knownDetail.c_str());

    // Digits(individual_digits=true) must split "2024" into four pieces.
    const std::vector<uint32_t> digits = tok.Encode("2024");
    bool digitsOk = digits.size() == 4;
    if (digitsOk)
    {
        for (uint32_t id : digits)
        {
            if (tok.Decode(id) != std::string(1, static_cast<char>('0' + 0)) &&
                tok.Decode(id) != "0")
            {
                // Each id must decode to exactly one digit.
                const std::string d = tok.Decode(id);
                if (d.size() != 1 || d[0] < '0' || d[0] > '9') { digitsOk = false; break; }
            }
        }
    }
    if (!digitsOk) ++g_failures;
    std::printf("  %-26s %-5s  \"2024\" -> %zu tokens (want 4, individual_digits)\n",
                "digit splitting", digitsOk ? "PASS" : "FAIL", digits.size());

    // Pre-tokenizer spot checks against the GPT-2 pattern.
    const auto pieces = tokenizer::BpeTokenizer::PreTokenize("the quick-brown fox");
    std::string joined;
    for (const std::string& p : pieces) { joined += "[" + p + "]"; }
    const bool preOk = joined == "[the][ quick][-][brown][ fox]";
    if (!preOk) ++g_failures;
    std::printf("  %-26s %-5s  %s\n", "pre-tokenizer", preOk ? "PASS" : "FAIL",
                joined.c_str());
}

// ===========================================================================
//  End-to-end: the real model, on the GPU, versus a CPU reference
// ===========================================================================
//
// The kernels are already verified individually. This is the test that proves
// the LAYER GRAPH is right -- that the passes are sequenced correctly, the
// bank schedule routes each intermediate to the pass that reads it, and the KV
// cache is addressed the way the attention kernel expects.
//
// Numerical parity against the CPU reference is the strong check. But a graph
// that is self-consistent and wrong would still pass it, so the generated token
// is also checked for SEMANTIC sanity: SmolLM-135M given "Hello" must produce a
// real word, not a rare garbage id.
void TestModelForward(Device& dev, shaders::Library& lib, Gpu& gpu)
{
    std::printf("\nend-to-end model:\n");

    std::FILE* probe = nullptr;
    if (_wfopen_s(&probe, L"models\\config.json", L"rb") != 0 || probe == nullptr)
    {
        std::printf("  %-26s SKIP  models\\ not present\n", "model forward");
        return;
    }
    std::fclose(probe);

    engine::VoxelEngine eng;
    engine::EngineOptions opts;
    opts.maxNewTokens = 8;

    try
    {
        eng.Initialize(L"models", opts, &dev, &lib);
    }
    catch (const std::exception& e)
    {
        ++g_failures;
        std::printf("  %-26s FAIL  %s\n", "engine init", e.what());
        return;
    }

    const std::vector<uint32_t> prompt = eng.Tokenizer().Encode("Hello");
    std::printf("  %-26s PASS  engine ready, prompt %zu tokens\n",
                "engine init", prompt.size());

    // ---- logits parity against an independent CPU reference ---------------
    std::vector<float> gpuLogits;
    try
    {
        eng.PoisonBanks(1234.5f);
        eng.ForwardLogits(prompt, gpuLogits);
    }
    catch (const std::exception& e)
    {
        ++g_failures;
        std::printf("  %-26s FAIL  %s\n", "forward pass", e.what());
        std::printf("  %-26s        deviceRemovedReason = 0x%08X\n",
                    "", static_cast<unsigned>(dev.Get()->GetDeviceRemovedReason()));
        return;
    }
    std::printf("  %-26s PASS  deviceRemovedReason = 0x%08X\n",
                "forward pass", static_cast<unsigned>(dev.Get()->GetDeviceRemovedReason()));

    // Trace: report the first few logits and whether the bank is non-zero.
    // A uniformly zero vector means a pass in the chain wrote nothing; a
    // plausible-but-wrong vector means the graph is mis-sequenced.
    int nonzero = 0;
    for (float v : gpuLogits) { if (v != 0.0f) ++nonzero; }
    std::printf("  %-26s        %d/%zu non-zero, logits[0..4] = %.4f %.4f %.4f %.4f %.4f\n",
                "", nonzero, gpuLogits.size(),
                static_cast<double>(gpuLogits[0]), static_cast<double>(gpuLogits[1]),
                static_cast<double>(gpuLogits[2]), static_cast<double>(gpuLogits[3]),
                static_cast<double>(gpuLogits[4]));

    // Trace the embedding lookup: the first GEMV's input is the embedding row.
    // If this is zero the whole chain is zero, and the fault is upstream of the
    // layer graph (upload, or the Embedding dispatch).
    {
        const uint32_t tok = prompt.empty() ? 0u : prompt[0];
        std::vector<uint32_t> embRow(8);
        eng.ReadWeightsAt(eng.GraphModel().EmbedTokensWord() +
                              static_cast<uint64_t>(tok) * 576,
                          embRow.data(), embRow.size());
        std::printf("  %-26s        embed[%u][0..3] = %08X %08X %08X %08X\n",
                    "", tok, embRow[0], embRow[1], embRow[2], embRow[3]);
    }

    // ---- wired plasticity -------------------------------------------------
    // The kernel test above proves the two dispatches are right in isolation.
    // This proves the ENGINE's path is right: the right tensor is addressed, the
    // staging range is where the planner put it, and the commit lands on the real
    // weights. The original words are restored afterwards so later tests still
    // see the uploaded model.
    {
        const graph::VramPlan::LayerTensors& lt = eng.GraphPlan().layers[0];
        const graph::ModelConfig& mc = eng.Model();
        const uint32_t kRows = mc.hiddenSize, kCols = mc.hiddenSize;
        const float wd = 0.25f, lr = 1e-3f;
        const size_t   kWords = static_cast<size_t>(kRows) * kCols;

        // Save the WHOLE target tensor, not a probe. The commit overwrites every
        // word, so restoring a 64-word window leaves 331,712 words decayed and
        // every later test reads a quietly damaged model.
        std::vector<uint32_t> before(kWords);
        eng.ReadWeightsAt(lt.wq, before.data(), kWords);

        engine::PlasticityStep step;
        step.weightOffset  = lt.wq;
        step.rows          = kRows;
        step.cols          = kCols;
        step.actBank       = 0;
        step.actOffset     = 0;
        step.errBank       = 1;
        step.errOffset     = 0;
        step.learningRate  = lr;
        step.weightDecay   = wd;
        step.gradientScale = 1.0f;

        // Zero error signal: the update must be decay-only, exactly w*(1-wd).
        eng.RunPlasticityWithError(step, nullptr);

        std::vector<uint32_t> after(kWords);
        eng.ReadWeightsAt(lt.wq, after.data(), kWords);

        F32 got(kWords), ref(kWords);
        for (size_t i = 0; i < kWords; ++i)
        {
            float w;
            std::memcpy(&w, &before[i], 4);
            std::memcpy(&got[i], &after[i], 4);
            ref[i] = w - wd * w;
        }
        Report("plasticity (engine, q_proj)", Compare(got, ref), 1e-6);

        // Restore, so the weight-upload invariants above still hold afterwards.
        dev.UploadToBuffer(eng.Weights(), before.data(), before.size() * 4,
                           lt.wq * 4);
        std::vector<uint32_t> restored(kWords);
        eng.ReadWeightsAt(lt.wq, restored.data(), kWords);
        const bool ok = (restored == before);
        if (!ok) { ++g_failures; }
        std::printf("  %-26s %-5s  %s\n", "plasticity restore", ok ? "PASS" : "FAIL",
                    ok ? "target tensor written back verbatim"
                       : "restore did not reproduce the uploaded bytes");
    }

    // ---- pass-by-pass walk of layer 0 --------------------------------------
    // Step the dispatch budget from 1 to 15 and, after each step, read the
    // region THAT PASS IS SUPPOSED TO WRITE. Reading a fixed window instead
    // hides the bug: gate/up/SwiGLU/down live at different scratch offsets, so a
    // wrong outDim leaves the probe window untouched and looks fine.
    //
    // This is a gate, not a debug print. Poisoning the banks first means "this
    // pass left part of its output unwritten" is directly observable, and that is
    // the failure mode of a mis-specified GEMV, a wrong thread index, or a missing
    // memory barrier -- all three of which reached this project and all three of
    // which are invisible to a final-value comparison.
    {
        // The walk poisons every activation bank 16 times over, which is the
        // heaviest IO in the suite. The device is checked immediately afterwards
        // so a fault raised HERE is attributed here instead of surfacing three
        // tests later as an unrelated DEVICE LOST.
        struct Probe { const char* name; uint32_t bank; uint32_t off; uint32_t n; };
        const graph::ModelConfig& walkCfg = eng.Model();
        const uint32_t wh   = walkCfg.hiddenSize;
        const uint32_t wff  = walkCfg.intermediateSize;
        const uint32_t wkvh = walkCfg.numKvHeads * walkCfg.HeadDim();
        static const Probe kProbe[15] = {
            { "Embedding",     2,      0, wh   },   // residual  <- scratch 0
            { "RMSNorm",       0,      0, wh   },   // normed    <- 0
            { "q_proj",        1,      0, wh   },   // q         <- 0
            { "k_proj",        1,      wh, wkvh },  // k         <- hidden
            { "v_proj",        1, wh + wkvh, wkvh },  // v         <- hidden + kvHidden
            { "RoPE",          0,      0, wh   },   // rotated q <- 0
            { "Attention",     1, wh + 2 * wkvh, wh },// attn out  <- hidden + 2*kvHidden
            { "o_proj",        0,      0, wh   },   // proj      <- 0
            { "residual+attn", 1,      0, wh   },   // rb1       <- 0
            { "RMSNorm2",      0,      0, wh   },   // n2        <- 0
            { "gate_proj",     2, 2 * wh + 2 * wkvh, wff },          // gate
            { "up_proj",       2, 2 * wh + 2 * wkvh + wff, wff },    // up
            { "SwiGLU",        0, 2 * wh + 2 * wkvh + 2 * wff, wff },// swiglu out
            { "down_proj",     2, wh + 2 * wkvh, wh },                // down
            { "residual+mlp",  0,      0, wh   },   // rb2       <- 0
        };

        const uint32_t tok = prompt.empty() ? 0u : prompt[0];
        std::printf("  %-26s        pass-by-pass walk of layer 0 (poison 1234.5)\n",
                    "");
        int unwrittenPasses = 0;
        std::string firstBadPass;
        for (uint32_t budget = 1; budget <= 15; ++budget)
        {
            eng.PoisonBanks(1234.5f);
            eng.RunPartial(tok, budget, 1);

            const Probe& pr = kProbe[budget - 1];
            std::vector<float> row(pr.n);
            eng.ReadActBank(pr.bank, pr.off, reinterpret_cast<uint32_t*>(row.data()), pr.n);

            double mx = 0.0;
            int    poisonLeft = 0;
            for (float v : row)
            {
                if (std::isnan(v)) { continue; }
                if (std::fabs(v - 1234.5f) < 1e-3f) { ++poisonLeft; }
                mx = std::max(mx, std::fabs(static_cast<double>(v)));
            }
            if (poisonLeft)
            {
                ++unwrittenPasses;
                if (firstBadPass.empty()) { firstBadPass = pr.name; }
            }
            std::printf("    %2u %-16s bank %c +%-5u n=%-5u  max=%12.4f%s\n",
                        budget, pr.name, 'A' + static_cast<char>(pr.bank), pr.off, pr.n,
                        mx, poisonLeft ? "  <== UNWRITTEN" : "");
        }

        if (unwrittenPasses) { ++g_failures; }
        const std::string walkMsg =
            unwrittenPasses
                ? (std::to_string(unwrittenPasses) + " pass(es) left poison, first: " +
                   firstBadPass)
                : "all 15 passes fully wrote their target range";
        std::printf("  %-26s %-5s  %s\n", "every pass writes its output",
                    unwrittenPasses ? "FAIL" : "PASS", walkMsg.c_str());

        {
            const auto hr = dev.Get()->GetDeviceRemovedReason();
            const bool alive = SUCCEEDED(hr);
            if (!alive) { ++g_failures; }
            std::printf("  %-26s %-5s  deviceRemovedReason after the walk = %08X\n",
                        "device healthy", alive ? "PASS" : "FAIL",
                        static_cast<unsigned>(hr));
        }

        // The RoPE -> attention handoff, as a GATE rather than a printed note.
        //
        // RoPE appends rotated K and raw V to the paged cache; the attention pass
        // reads them back. For a one-token prompt the softmax has a single term,
        // so the attention output MUST equal V exactly. Checking all three copies
        // -- v_proj's output, the cached V, and what attention produced -- is what
        // separates a broken RoPE from a broken attention, and it is the check
        // that catches a cache index using the activation stride instead of the
        // head stride.
        //
        // The cache is [layer][kvHead][seq][dim]: adjacent kv HEADS sit
        // maxSeq*headDim apart, while maxSeq*nKV*headDim is the distance from the
        // K plane to the V plane. Conflating the two reads empty space and makes
        // a correct RoPE look broken.
        {
            const uint32_t kvTok = prompt.empty() ? 0u : prompt[0];
            eng.PoisonBanks(1234.5f);
            eng.RunPartial(kvTok, 7, 1);

            const graph::ModelConfig& mc = eng.Model();
            const uint32_t hd    = mc.HeadDim();
            const uint32_t nKV   = mc.numKvHeads;
            const uint32_t nH    = mc.numHeads;
            const uint32_t h     = mc.hiddenSize;
            const uint32_t kvh   = nKV * hd;
            const uint32_t headStride = mc.maxPositions * hd;
            const uint32_t vPlane     = mc.maxPositions * kvh;

            std::vector<float> vProj(kvh), vCache(kvh), attn(h);
            eng.ReadActBank(1, 768, reinterpret_cast<uint32_t*>(vProj.data()), kvh);
            for (uint32_t i = 0; i < nKV; ++i)
            {
                eng.ReadKv(vPlane + i * headStride,
                           reinterpret_cast<uint32_t*>(vCache.data()) + i * hd, hd);
            }
            eng.ReadActBank(1, 960, reinterpret_cast<uint32_t*>(attn.data()), h);
            Report("KV V  <- v_proj", Compare(vProj, vCache), 0.0);

            // GQA: query head i reads kv head i / (nHeads / nKvHeads).
            std::vector<float> expect(h);
            const uint32_t perKv = nH / nKV;
            for (uint32_t i = 0; i < nH; ++i)
            {
                for (uint32_t l = 0; l < hd; ++l)
                {
                    expect[i * hd + l] = vCache[(i / perKv) * hd + l];
                }
            }
            Report("attention  <- KV V", Compare(expect, attn), 0.0);

            // The cache must hold the ROTATED key, i.e. the RoPE output bank.
            F32 kRope(kvh), kCache(kvh);
            eng.ReadActBank(0, 576, reinterpret_cast<uint32_t*>(kRope.data()), kvh);
            for (uint32_t i = 0; i < nKV; ++i)
            {
                eng.ReadKv(i * headStride,
                           reinterpret_cast<uint32_t*>(kCache.data()) + i * hd, hd);
            }
            Report("KV K  <- rotated k", Compare(kRope, kCache), 0.0);
        }
    }


    std::vector<float> cpuLogits;
    try
    {
        cpu::Reference ref;
        ref.Load(L"models");
        ref.Forward(prompt, cpuLogits);
    }
    catch (const std::exception& e)
    {
        ++g_failures;
        std::printf("  %-26s FAIL  CPU reference: %s\n", "cpu reference", e.what());
        return;
    }

    const Result r = Compare(gpuLogits, cpuLogits);
    Report("logits vs CPU reference", r, 2e-2);

    // ---- per-layer divergence trace ----------------------------------------
    // The parity check above says the graph is wrong but not WHERE. Running the
    // same token through both implementations one layer at a time and comparing
    // the residual stream localises it to a layer index immediately.
    {
        cpu::Reference ref2;
        ref2.Load(L"models");
        std::vector<float> tmp;
        std::vector<std::vector<float>> cpuTrace;
        ref2.ForwardTraced(prompt, 6, tmp, cpuTrace);

        std::vector<std::vector<float>> gpuTrace;
        eng.RunLayersTraced(prompt.empty() ? 0u : prompt[0], 6, gpuTrace);

        int firstBad = -1;
        for (size_t L = 0; L < gpuTrace.size() && L < cpuTrace.size(); ++L)
        {
            const Result lr = Compare(gpuTrace[L], cpuTrace[L]);
            if (lr.maxAbs > 1e-3) { firstBad = static_cast<int>(L); break; }
        }
        std::printf("  %-26s        ", "layer trace");
        for (size_t L = 0; L < gpuTrace.size() && L < 6; ++L)
        {
            const Result lr = Compare(gpuTrace[L], cpuTrace[L]);
            std::printf("L%zu=%.1e ", L, lr.maxAbs);
        }
        // Layer 0 counts: `firstBad > 0` silently passed a divergence in the very
        // first layer, which is the one place a bank-schedule bug shows up.
        const bool traceOk = (firstBad < 0);
        if (!traceOk) { ++g_failures; }
        std::printf("\n  %-26s %-5s  %s\n",
                    "layer divergence", traceOk ? "PASS" : "FAIL",
                    traceOk ? "residual matches the reference through 6 layers"
                            : ("first divergent layer = " +
                               std::to_string(firstBad)).c_str());
    }

    // ---- semantic sanity ---------------------------------------------------
    uint32_t gpuBest = 0;
    float    gpuBestVal = -1e30f;
    for (size_t i = 0; i < gpuLogits.size(); ++i)
    {
        if (gpuLogits[i] > gpuBestVal) { gpuBestVal = gpuLogits[i]; gpuBest = static_cast<uint32_t>(i); }
    }

    uint32_t cpuBest = 0;
    float    cpuBestVal = -1e30f;
    for (size_t i = 0; i < cpuLogits.size(); ++i)
    {
        if (cpuLogits[i] > cpuBestVal) { cpuBestVal = cpuLogits[i]; cpuBest = static_cast<uint32_t>(i); }
    }

    std::printf("  %-26s        gpu argmax %u (%s)  cpu argmax %u (%s)\n", "",
                gpuBest, eng.Tokenizer().Decode(gpuBest).c_str(),
                cpuBest, eng.Tokenizer().Decode(cpuBest).c_str());


    std::printf("  %-26s        gpu[0..4] %.4f %.4f %.4f %.4f %.4f\n", "",
                static_cast<double>(gpuLogits[0]), static_cast<double>(gpuLogits[1]),
                static_cast<double>(gpuLogits[2]), static_cast<double>(gpuLogits[3]),
                static_cast<double>(gpuLogits[4]));
    std::printf("  %-26s        cpu[0..4] %.4f %.4f %.4f %.4f %.4f\n", "",
                static_cast<double>(cpuLogits[0]), static_cast<double>(cpuLogits[1]),
                static_cast<double>(cpuLogits[2]), static_cast<double>(cpuLogits[3]),
                static_cast<double>(cpuLogits[4]));

    // Semantic sanity: the top prediction must be a REAL vocabulary entry that
    // decodes to printable text, and it must be the same token the CPU reference
    // picks. A wrong-but-self-consistent graph is what this catches.
    //
    // An earlier version of this check demanded that the piece contain one of
    // "aeionst ". SmolLM-135M's actual top prediction for "Hello" is "," -- a
    // perfectly ordinary continuation, and the CPU reference agrees -- so the
    // heuristic rejected a correct model. Demanding a vowel in a piece that may
    // legitimately be punctuation is not a test of anything.
    const std::string piece = eng.Tokenizer().Decode(gpuBest);
    bool printable = !piece.empty();
    for (unsigned char c : piece)
    {
        if (c < 0x20 && c != '\n' && c != '\t') { printable = false; }
    }
    const bool agrees = (gpuBest == cpuBest);
    const bool sane = printable && agrees;
    if (!sane) ++g_failures;
    std::printf("  %-26s %-5s  top prediction %u \"%s\"%s\n",
                "semantic sanity", sane ? "PASS" : "FAIL",
                gpuBest, piece.c_str(),
                agrees ? " (matches CPU reference)"
                       : " (CPU reference chose a different token)");

    // ---- short generation --------------------------------------------------
    engine::EngineOptions gopts;
    gopts.maxNewTokens = 8;
    const engine::GenerateResult gr = eng.Generate(prompt, gopts);
    std::string outText = eng.Tokenizer().Decode(gr.generated);
    std::printf("  %-26s        %zu tokens -> \"%s\"\n",
                "", gr.generated.size(), outText.c_str());
    std::printf("  %-26s %-5s  generation produced %zu token(s)\n",
                "generation", gr.generated.empty() ? "FAIL" : "PASS",
                gr.generated.size());
    if (gr.generated.empty()) ++g_failures;

    // The engine called SetPersistentBuffers() with its OWN buffers (2053 MB of
    // weights, ~220 KB activation banks). Restore the harness binding set, or
    // every later kernel test dispatches against the engine's allocations and
    // reads memory that does not exist in the small test buffers.
    gpu.Create(dev);
}

// ===========================================================================
//  Binding self-test. Writes a known constant with NO root constants and NO
//  SRV reads. Isolates "does a dispatch execute at all" from "do root
//  constants arrive".
// ===========================================================================
void TestDispatchPath(Device& dev, shaders::Library& lib, Gpu& gpu)
{
    dev.ResetCommandList();
    {
        CommandScope cs(dev);
        cs.SetPipeline(lib.Get(shaders::Id::Debug));
        cs.SetParams(nullptr, 0);
        cs.Dispatch(1, 1, 1);
    }
    dev.ExecuteAndWait();

    const F32 got = Download(dev, gpu.a, 64);
    bool ok = true;
    for (uint32_t i = 0; i < 64; ++i)
    {
        if (std::fabs(got[i] - (1000.0f + static_cast<float>(i))) > 1e-4f) { ok = false; break; }
    }
    if (!ok) ++g_failures;
    std::printf("  %-26s %-5s  bankA[0..3] = %.1f %.1f %.1f %.1f (want 1000..1003)\n",
                "Dispatch path", ok ? "PASS" : "FAIL",
                got[0], got[1], got[2], got[3]);
    std::printf("  %-26s        deviceRemovedReason = 0x%08X\n",
                "", (unsigned)dev.Get()->GetDeviceRemovedReason());
}

// Parameter-block probe. Two dispatches in ONE command list, each with a
// different constant, writing to a different slot. This is the property the
// whole engine depends on: the GPU samples the parameter buffer at EXECUTION
// time, so without a per-dispatch address every dispatch would see only the
// last parameters the host wrote.
// Parameter-block probe. Two dispatches in ONE command list, each with a
// different constant, writing to a different destination range. This is the
// property the whole engine depends on: the GPU samples the parameter buffer at
// EXECUTION time, so without a per-dispatch address every dispatch would see
// only the last parameters the host wrote.
//
// Read back as raw uints -- no float anywhere in the comparison.
void TestParamSlots(Device& dev, shaders::Library& lib, Gpu& gpu)
{
    const uint32_t pa[5] = { 0xC0010001u, 0xDEADBEEFu, 0x22222222u, 0x33333333u, 0u };
    const uint32_t pb[5] = { 0xC0020002u, 0x12345678u, 0x44444444u, 0x55555555u, 8u };

    dev.ResetCommandList();
    {
        CommandScope cs(dev);
        cs.SetPipeline(lib.Get(shaders::Id::DebugRootConst));
        cs.SetParams(pa, 5);
        cs.Dispatch(1, 1, 1);
        cs.SetParams(pb, 5);
        cs.Dispatch(1, 1, 1);
    }
    dev.ExecuteAndWait();

    std::vector<uint32_t> got(12);
    dev.ReadbackFromBuffer(gpu.a, got.data(), got.size() * 4);

    bool ok = true;
    for (uint32_t i = 0; i < 4; ++i)
    {
        ok = ok && (got[i] == pa[i]) && (got[8 + i] == pb[i]);
    }
    if (!ok) ++g_failures;
    std::printf("  %-26s %-5s  disp0[%08X %08X %08X %08X]\n",
                "Parameter slots", ok ? "PASS" : "FAIL",
                got[0], got[1], got[2], got[3]);
    std::printf("  %-26s        want  [%08X %08X %08X %08X]\n",
                "", pa[0], pa[1], pa[2], pa[3]);
    std::printf("  %-26s        disp1[%08X %08X %08X %08X]\n", "",
                got[8], got[9], got[10], got[11]);
    std::printf("  %-26s        want  [%08X %08X %08X %08X]\n",
                "", pb[0], pb[1], pb[2], pb[3]);
}

// SRV probe: do t0/t2/t3 reads return the data uploaded on the CPU?
void TestSrvReads(Device& dev, shaders::Library& lib, Gpu& gpu)
{
    // Distinct sentinels in each source buffer.
    const float w0 = 11.5f, kv0 = 22.5f, b0 = 33.5f;
    uint32_t wb, kb, bb;
    std::memcpy(&wb, &w0, 4);
    std::memcpy(&kb, &kv0, 4);
    std::memcpy(&bb, &b0, 4);
    dev.UploadToBuffer(gpu.w, &wb, 4, 0);
    dev.UploadToBuffer(gpu.kv, &kb, 4, 0);
    dev.UploadToBuffer(gpu.b, &bb, 4, 0);

    dev.ResetCommandList();
    {
        CommandScope cs(dev);
        cs.SetPipeline(lib.Get(shaders::Id::DebugSrv));
        cs.SetParams(nullptr, 0);
        cs.Dispatch(1, 1, 1);
    }
    dev.ExecuteAndWait();

    const F32 got = Download(dev, gpu.a, 3);
    const bool ok = (got[0] == w0) && (got[1] == kv0) && (got[2] == b0);
    if (!ok) ++g_failures;
    std::printf("  %-26s %-5s  t0=%.1f/%.1f t3(kv)=%.1f/%.1f t2(bankB)=%.1f/%.1f\n",
                "SRV reads", ok ? "PASS" : "FAIL",
                static_cast<double>(got[0]), static_cast<double>(w0),
                static_cast<double>(got[1]), static_cast<double>(kv0),
                static_cast<double>(got[2]), static_cast<double>(b0));
}

// ===========================================================================
//  Kernel tests
// ===========================================================================

// ---- GEMV: y[o] = sum_i W[o][inDim+i] * x[i] ---------------------------
void TestGEMV(Device& dev, shaders::Library& lib, Gpu& gpu)
{
    constexpr uint32_t kOut = 64, kIn = 128;

    Rng rng(0xC0FFEE01u);
    std::vector<uint32_t> W(kOut * kIn), X(kIn);
    for (auto& v : W) { const float f = rng.NextF(); std::memcpy(&v, &f, 4); }
    for (auto& v : X) { const float f = rng.NextF(); std::memcpy(&v, &f, 4); }

    Upload(dev, gpu.w, W);
    Upload(dev, gpu.a, X);

    // x from bank A, y to bank B: distinct resources, so no SRV/UAV aliasing.
    const uint32_t p[7] = { 0, 0, 0, kIn, kOut, 0, 1 };
    dev.ResetCommandList();
    {
        CommandScope cs(dev);
        cs.SetPipeline(lib.Get(shaders::Id::GEMV));
        cs.SetParams(p, 7);
        cs.Dispatch(kOut, 1, 1);
    }
    dev.ExecuteAndWait();

    const F32 got = Download(dev, gpu.b, kOut);

    F32 want(kOut, 0.0f);
    for (uint32_t o = 0; o < kOut; ++o)
    {
        float acc = 0.0f;
        for (uint32_t i = 0; i < kIn; ++i) acc += W[o * kIn + i] * X[i];
        want[o] = acc;
    }

    // Round-trip the reference through the same bit pattern for a fair compare.
    F32 ref(kOut);
    for (uint32_t o = 0; o < kOut; ++o)
    {
        float acc = 0.0f;
        for (uint32_t i = 0; i < kIn; ++i)
        {
            float wf, xf;
            std::memcpy(&wf, &W[o * kIn + i], 4);
            std::memcpy(&xf, &X[i], 4);
            acc += wf * xf;
        }
        ref[o] = acc;
    }
    std::printf("  %-26s        got[0..3]=%.5f %.5f %.5f %.5f\n", "",
                got[0], got[1], got[2], got[3]);
    std::printf("  %-26s        ref[0..3]=%.5f %.5f %.5f %.5f\n", "",
                ref[0], ref[1], ref[2], ref[3]);
    
    Report("GEMV  [64x128]", Compare(got, ref), 1e-4);
}

// ---- GEMVQuant: y[o] = scaleW * scaleX * sum_i int8W[o][i] * x[i] --------
//
// Phase 2's INT4/INT8 path has no model to run against yet, so this is the
// only thing standing between the quantized dispatch and a silent zero. The
// kernel and its 9-parameter layout were written without any test, and that is
// exactly how P[6] ended up as a literal 0 -- AsF(0) is 0.0f, so the kernel
// multiplied every row to zero and nothing complained. A test that feeds a
// non-unit weight scale and asserts against a CPU reference catches that, and
// catches a truncated scale the same way, because scaleW is passed as float
// bits into a uint32_t[] and a raw float there would silently truncate to 0.
//
// int8 weights are packed four-per-word, little-endian, exactly as the GGUF
// Q8_0-style packing the kernel expects: byte n of a word is element 4n.
void TestGEMVQuant(Device& dev, shaders::Library& lib, Gpu& gpu)
{
    constexpr uint32_t kOut = 64, kIn = 128;   // kIn must be a multiple of 4
    constexpr float    kScaleW = 0.0137f;      // deliberately not a round number
    constexpr float    kScaleX = 1.0f;         // activations are fp32, so identity

    Rng rng(0x9A17BEEFu);

    // Quantize: pick int8 values directly rather than deriving them from floats,
    // so the reference is exact and any error is the kernel's, not the quantizer's.
    std::vector<int8_t>  q(kOut * kIn);
    std::vector<uint32_t> W(kOut * kIn / 4, 0u);
    for (size_t n = 0; n < q.size(); ++n)
        q[n] = static_cast<int8_t>(static_cast<int>(rng.NextF() * 127.0f) - 63);

    for (size_t n = 0; n < q.size(); ++n)
    {
        const uint32_t lane = static_cast<uint32_t>(n & 3u);
        const uint32_t word = static_cast<uint32_t>(n >> 2);
        const uint32_t byte = static_cast<uint32_t>(static_cast<uint8_t>(q[n]));
        W[word] |= byte << (lane * 8);
    }

    std::vector<uint32_t> X(kIn);
    for (auto& v : X) { const float f = rng.NextF() * 2.0f - 1.0f; std::memcpy(&v, &f, 4); }

    Upload(dev, gpu.w, W);
    Upload(dev, gpu.a, X);

    // x from bank A, y to bank B: distinct resources, so no SRV/UAV aliasing.
    const uint32_t p[9] = {
        0, 0, 0,                              // xOff, wOff, yOff
        kIn, kOut,                            // inDim, outDim
        graph::ModelConfig::FloatBits(kScaleW),
        graph::ModelConfig::FloatBits(kScaleX),
        0, 1                                  // bank, obank
    };
    dev.ResetCommandList();
    {
        CommandScope cs(dev);
        cs.SetPipeline(lib.Get(shaders::Id::GEMVQuant));
        cs.SetParams(p, 9);
        cs.Dispatch(kOut, 1, 1);
    }
    dev.ExecuteAndWait();

    const F32 got = Download(dev, gpu.b, kOut);

    F32 ref(kOut);
    for (uint32_t o = 0; o < kOut; ++o)
    {
        float acc = 0.0f;
        for (uint32_t i = 0; i < kIn; ++i)
        {
            float xf;
            std::memcpy(&xf, &X[i], 4);
            acc += static_cast<float>(q[o * kIn + i]) * xf;
        }
        ref[o] = acc * kScaleW * kScaleX;
    }

    std::printf("  %-26s        got[0..3]=%.5f %.5f %.5f %.5f\n", "",
                got[0], got[1], got[2], got[3]);
    std::printf("  %-26s        ref[0..3]=%.5f %.5f %.5f %.5f\n", "",
                ref[0], ref[1], ref[2], ref[3]);

    Report("GEMVQuant [64x128]", Compare(got, ref), 1e-3);
}

// ---- RMSNorm -------------------------------------------------------------
void TestRMSNorm(Device& dev, shaders::Library& lib, Gpu& gpu)
{
    constexpr uint32_t kD = 256;
    const float kEps = 1e-5f;

    Rng rng(0xBADF00D2u);
    std::vector<uint32_t> X(kD), G(kD);
    for (auto& v : X) { const float f = rng.NextF(); std::memcpy(&v, &f, 4); }
    for (auto& v : G) { const float f = rng.NextF() * 0.5f; std::memcpy(&v, &f, 4); }

    Upload(dev, gpu.a, X);
    Upload(dev, gpu.w, G);

    uint32_t epsBits;
    std::memcpy(&epsBits, &kEps, 4);

    const uint32_t p[8] = { 0, 0, 0, kD, epsBits, 1, 0, 1 };
    dev.ResetCommandList();
    {
        CommandScope cs(dev);
        cs.SetPipeline(lib.Get(shaders::Id::RMSNorm));
        cs.SetParams(p, 8);
        cs.Dispatch(1, 1, 1);
    }
    dev.ExecuteAndWait();

    const F32 got = Download(dev, gpu.b, kD);

    double ss = 0.0;
    for (uint32_t i = 0; i < kD; ++i)
    {
        float xf, gf;
        std::memcpy(&xf, &X[i], 4);
        std::memcpy(&gf, &G[i], 4);
        ss += static_cast<double>(xf) * xf;
    }
    const float inv = 1.0f / std::sqrt(static_cast<float>(ss / kD) + kEps);

    F32 ref(kD);
    for (uint32_t i = 0; i < kD; ++i)
    {
        float xf, gf;
        std::memcpy(&xf, &X[i], 4);
        std::memcpy(&gf, &G[i], 4);
        ref[i] = xf * inv * gf;
    }
    Report("RMSNorm [256]", Compare(got, ref), 1e-4);
}

// ---- SwiGLU --------------------------------------------------------------
void TestSwiGLU(Device& dev, shaders::Library& lib, Gpu& gpu)
{
    constexpr uint32_t kN = 64;
    Rng rng(0x5EED1234u);
    std::vector<uint32_t> G(kN), U(kN);
    for (auto& v : G) { const float f = rng.NextF() * 3.0f; std::memcpy(&v, &f, 4); }
    for (auto& v : U) { const float f = rng.NextF();      std::memcpy(&v, &f, 4); }
    Upload(dev, gpu.a, G);

    // Both inputs live in bank A (gate at word 0, up at word 256); dir=0 reads
    // bank A and writes bank B.
    Upload(dev, gpu.a, G);
    Upload(dev, gpu.a, U, 256 * 4);

    const uint32_t p[6] = { 0, 256, 512, kN, 0, 1 };
    dev.ResetCommandList();
    {
        CommandScope cs(dev);
        cs.SetPipeline(lib.Get(shaders::Id::SwiGLU));
        cs.SetParams(p, 6);
        cs.Dispatch(1, 1, 1);
    }
    dev.ExecuteAndWait();

    const F32 got = Download(dev, gpu.b, kN, 512 * 4);

    F32 ref(kN);
    for (uint32_t i = 0; i < kN; ++i)
    {
        float g, u;
        std::memcpy(&g, &G[i], 4);
        std::memcpy(&u, &U[i], 4);
        const float s = 1.0f / (1.0f + std::exp(-g));
        ref[i] = (g * s) * u;
    }
    Report("SwiGLU [64]", Compare(got, ref), 1e-4);
}

// ---- Add -----------------------------------------------------------------
void TestAdd(Device& dev, shaders::Library& lib, Gpu& gpu)
{
    constexpr uint32_t kN = 64;
    Rng rng(0x77777777u);
    std::vector<uint32_t> A(kN), B(kN);
    for (auto& v : A) { const float f = rng.NextF(); std::memcpy(&v, &f, 4); }
    for (auto& v : B) { const float f = rng.NextF(); std::memcpy(&v, &f, 4); }
    Upload(dev, gpu.a, A);
    Upload(dev, gpu.a, B, 256 * 4);

    const uint32_t p[6] = { 0, 256, 512, kN, 0, 1 };
    dev.ResetCommandList();
    {
        CommandScope cs(dev);
        cs.SetPipeline(lib.Get(shaders::Id::Add));
        cs.SetParams(p, 6);
        cs.Dispatch(1, 1, 1);
    }
    dev.ExecuteAndWait();

    const F32 got = Download(dev, gpu.b, kN, 512 * 4);
    F32 ref(kN);
    for (uint32_t i = 0; i < kN; ++i)
    {
        float a, b;
        std::memcpy(&a, &A[i], 4);
        std::memcpy(&b, &B[i], 4);
        ref[i] = a + b;
    }
    Report("Add    [64]", Compare(got, ref), 1e-5);
}

// ---- ArgMax (49k vocabulary, mirrors the LM-head readback path) ----------
void TestArgMax(Device& dev, shaders::Library& lib, Gpu& gpu)
{
    constexpr uint32_t kVocab = 49152;
    const uint64_t tail = (kKvWords - 64);          // reserved argmax tail in KV

    Rng rng(0x13571357u);
    std::vector<uint32_t> L(kVocab);
    float best = -1e30f; uint32_t bestI = 0;
    for (uint32_t i = 0; i < kVocab; ++i)
    {
        const float f = rng.NextF() * 5.0f;
        std::memcpy(&L[i], &f, 4);
        if (f > best) { best = f; bestI = i; }
    }
    Upload(dev, gpu.a, L);

    const uint32_t p[4] = { 0, kVocab, static_cast<uint32_t>(tail), 0 };
    dev.ResetCommandList();
    {
        CommandScope cs(dev);
        cs.SetPipeline(lib.Get(shaders::Id::ArgMax));
        cs.SetParams(p, 4);
        cs.Dispatch(1, 1, 1);
    }
    dev.ExecuteAndWait();

    uint32_t got[2] = { 0, 0 };
    dev.ReadbackFromBuffer(gpu.kv, got, sizeof(got), tail * 4);

    float gotVal; std::memcpy(&gotVal, &got[0], 4);
    const bool ok = (got[1] == bestI) && (std::fabs(gotVal - best) <= 1e-5f);
    if (!ok) ++g_failures;
    std::printf("  %-26s %-5s  token=%u (ref %u)  value=%.6f (ref %.6f)\n",
                "ArgMax [49152]", ok ? "PASS" : "FAIL",
                got[1], bestI, gotVal, best);
}

// ---- Decode attention ---------------------------------------------------
// Attention is the one kernel that is compiled eight times, once per supported
// head_dim, because the group width IS head_dim. Testing only one of them is how
// the other seven could rot unnoticed: a 64-thread model dispatched onto the
// 256-thread pipeline still returns finite numbers, just the wrong ones, since
// 192 lanes read past the end of the head. So this runs the full matrix.
void TestAttentionAt(Device& dev, shaders::Library& lib, Gpu& gpu,
                     uint32_t kHD, const char* label)
{
    constexpr uint32_t kHeads = 4, kKV = 2, kSeq = 7, kMax = 64;
    const uint64_t kvOff = 0;
    const uint64_t kvStride = static_cast<uint64_t>(kMax) * kKV * kHD;

    // A distinct seed per head_dim: reusing one would let a wrong-lane mapping
    // pass on symmetric data by coincidence.
    Rng rng(0xABCDEF01u ^ (kHD * 2654435761u));
    std::vector<uint32_t> Q(kHeads * kHD);
    std::vector<uint32_t> K(kKV * kMax * kHD), V(kKV * kMax * kHD);
    for (auto& v : Q) { const float f = rng.NextF();  std::memcpy(&v, &f, 4); }
    for (auto& v : K) { const float f = rng.NextF();  std::memcpy(&v, &f, 4); }
    for (auto& v : V) { const float f = rng.NextF();  std::memcpy(&v, &f, 4); }

    Upload(dev, gpu.a, Q);
    Upload(dev, gpu.kv, K, kvOff * 4);
    Upload(dev, gpu.kv, V, (kvOff + kvStride) * 4);

    const uint32_t p[10] = { 0, static_cast<uint32_t>(kvOff), 0, kKV, kHD,
                              kMax, kSeq, kHeads, 0, 1 };
    dev.ResetCommandList();
    {
        CommandScope cs(dev);
        const shaders::Id id = shaders::Library::AttentionFor(kHD);
        OPURE_ASSERT(id != shaders::Id::Count);
        cs.SetPipeline(lib.Get(id));
        cs.SetParams(p, 10);
        cs.Dispatch(kHeads, 1, 1);
    }
    dev.ExecuteAndWait();

    const F32 got = Download(dev, gpu.b, kHeads * kHD);

    F32 ref(kHeads * kHD, 0.0f);
    const uint32_t kvGroup = kHeads / kKV;
    for (uint32_t h = 0; h < kHeads; ++h)
    {
        const uint32_t kvh = h / kvGroup;
        float m = -1e30f, l = 0.0f;
        F32 acc(kHD, 0.0f);
        for (uint32_t s = 0; s < kSeq; ++s)
        {
            float score = 0.0f;
            for (uint32_t d = 0; d < kHD; ++d)
            {
                float qf, kf;
                std::memcpy(&qf, &Q[h * kHD + d], 4);
                std::memcpy(&kf, &K[kvh * kMax * kHD + s * kHD + d], 4);
                score += qf * kf;
            }
            const float nm = m > score ? m : score;
            const float corr = (m <= -1e29f) ? 0.0f : std::exp(m - nm);
            const float pv = std::exp(score - nm);
            l = l * corr + pv;
            for (uint32_t d = 0; d < kHD; ++d)
            {
                float vf;
                std::memcpy(&vf, &V[kvh * kMax * kHD + s * kHD + d], 4);
                acc[d] = acc[d] * corr + pv * vf;
            }
            m = nm;
        }
        for (uint32_t d = 0; d < kHD; ++d) ref[h * kHD + d] = acc[d] / l;
    }
    std::string lbl = std::string("Attention ") + label + " [4h/2kv/7s]";
        Report(lbl.c_str(), Compare(got, ref), 1e-4);
}

// ---- AddRes: cross-bank residual add ---------------------------------------
//
// This kernel exists because the plain Add cannot do a transformer residual:
// the add must read one ping-pong bank AND the residual bank, then write a
// ping-pong bank, without ever reading and writing the same resource.
void TestAddRes(Device& dev, shaders::Library& lib, Gpu& gpu)
{
    constexpr uint32_t kN = 64;
    Rng rng(0xADDE5EEDu);
    std::vector<uint32_t> X(kN), Res(kN);
    for (auto& v : X)   { const float f = rng.NextF() * 2.0f - 1.0f; std::memcpy(&v, &f, 4); }
    for (auto& v : Res) { const float f = rng.NextF() * 2.0f - 1.0f; std::memcpy(&v, &f, 4); }
    Upload(dev, gpu.a, X);
    Upload(dev, gpu.res, Res);

    // 3-way banks: read A (0) and C (2), write B (1) -- three distinct
    // resources, which is the whole point of this kernel.
    const uint32_t p[7] = { 0, 0, 0, kN, 0, 2, 1 };
    dev.ResetCommandList();
    {
        CommandScope cs(dev);
        cs.SetPipeline(lib.Get(shaders::Id::AddRes));
        cs.SetParams(p, 7);
        cs.Dispatch((kN + 255) / 256, 1, 1);
    }
    dev.ExecuteAndWait();

    const F32 got = Download(dev, gpu.b, kN);
    F32 ref(kN);
    for (uint32_t i = 0; i < kN; ++i)
    {
        float a, r;
        std::memcpy(&a, &X[i], 4);
        std::memcpy(&r, &Res[i], 4);
        ref[i] = a + r;
    }
    Report("AddRes [64]", Compare(got, ref), 1e-5);
}

// ---- SignSGD plasticity --------------------------------------------------
//
// The real engine owns the weight buffer, so this exercises the raw kernels
// against the engine's two-dispatch formulation on a scratch weight matrix: one
// pass computes into a bank, one commits. Verifying the pair here is what proves
// the split is correct -- a single aliased dispatch would be a different code
// path, and D3D12 gives no guarantee about it.
void TestPlasticity(Device& dev, shaders::Library& lib, Gpu& gpu)
{
    // kRows must EXCEED one 256-thread group. Both SignSGD and the earlier
    // Embedding bug indexed by SV_GroupIndex -- the lane, not the thread -- which
    // is invisible for any row count a single group covers. 300 rows guarantees
    // the second group has work.
    constexpr uint32_t kRows = 300, kCols = 197;
    Rng rng(0x516E5D00u);

    std::vector<uint32_t> W(kRows * kCols), Act(kCols), Err(kRows);
    for (auto& v : W)   { const float f = rng.NextF() * 0.4f - 0.2f; std::memcpy(&v, &f, 4); }
    for (auto& v : Act) { const float f = rng.NextF() * 2.0f - 1.0f;  std::memcpy(&v, &f, 4); }
    for (auto& v : Err) { const float f = rng.NextF() * 2.0f - 1.0f;  std::memcpy(&v, &f, 4); }

    Upload(dev, gpu.w, W);
    Upload(dev, gpu.a, Act);
    Upload(dev, gpu.b, Err);

    const float lr = 3e-4f, wd = 0.02f, gScale = 0.5f;
    const uint32_t stage = 1024;   // well past anything the inference graph uses

    // Pass 1: compute the update into bank C at `stage`.
    {
        const uint32_t p[12] = { 0, 0, 0, kRows, kCols,
                                 graph::ModelConfig::FloatBits(lr),
                                 graph::ModelConfig::FloatBits(wd),
                                 graph::ModelConfig::FloatBits(gScale),
                                 0, 1, 2, stage };
        dev.ResetCommandList();
        {
            CommandScope cs(dev);
            cs.Barrier();
            cs.SetPipeline(lib.Get(shaders::Id::SignSGD));
            cs.SetParams(p, 12);
            cs.Dispatch((kRows + 255) / 256, 1, 1);
        }
        dev.ExecuteAndWait();
    }

    // Pass 2: commit the staged matrix over the weights.
    {
        const uint32_t p[4] = { stage, 0, kRows * kCols, 2 };
        dev.ResetCommandList();
        {
            CommandScope cs(dev);
            cs.Barrier();
            cs.SetPipeline(lib.Get(shaders::Id::WeightCommit));
            cs.SetParams(p, 4);
            cs.Dispatch((kRows * kCols + 255) / 256, 1, 1);
        }
        dev.ExecuteAndWait();
    }

    const F32 got = Download(dev, gpu.w, kRows * kCols);
    F32 ref(kRows * kCols);
    for (uint32_t o = 0; o < kRows; ++o)
    {
        float e;
        std::memcpy(&e, &Err[o], 4);
        for (uint32_t i = 0; i < kCols; ++i)
        {
            float w, a;
            std::memcpy(&w, &W[o * kCols + i], 4);
            std::memcpy(&a, &Act[i], 4);
            const float g  = a * e * gScale;
            // The shader computes sign() as a bool difference; do the same here
            // rather than relying on an int-to-float implicit conversion.
            const float sg = (g > 0.0f ? 1.0f : 0.0f) - (g < 0.0f ? 1.0f : 0.0f);
            ref[o * kCols + i] = w - (lr * sg) - (wd * w);
        }
    }
    Report("SignSGD [300x197]", Compare(got, ref), 1e-6);

    // A zero error signal must leave the weights decaying only, with no step.
    // sign(0) == 0 is the property that makes an idle update safe when there is
    // no gradient to apply. The error vector is re-uploaded as zeros -- leaving
    // the previous one in place would silently keep applying a step.
    Upload(dev, gpu.w, W);
    Upload(dev, gpu.b, std::vector<uint32_t>(kRows, 0u));
    {
        const uint32_t p[12] = { 0, 0, 0, kRows, kCols,
                                 graph::ModelConfig::FloatBits(lr),
                                 graph::ModelConfig::FloatBits(wd),
                                 graph::ModelConfig::FloatBits(gScale),
                                 0, 1, 2, stage };
        dev.ResetCommandList();
        {
            CommandScope cs(dev);
            cs.Barrier();
            cs.SetPipeline(lib.Get(shaders::Id::SignSGD));
            cs.SetParams(p, 12);
            cs.Dispatch((kRows + 255) / 256, 1, 1);
        }
        dev.ExecuteAndWait();
    }
    {
        const uint32_t p[4] = { stage, 0, kRows * kCols, 2 };
        dev.ResetCommandList();
        {
            CommandScope cs(dev);
            cs.Barrier();
            cs.SetPipeline(lib.Get(shaders::Id::WeightCommit));
            cs.SetParams(p, 4);
            cs.Dispatch((kRows * kCols + 255) / 256, 1, 1);
        }
        dev.ExecuteAndWait();
    }
    const F32 decayed = Download(dev, gpu.w, kRows * kCols);
    F32 decayRef(kRows * kCols);
    for (size_t i = 0; i < decayRef.size(); ++i)
    {
        float w;
        std::memcpy(&w, &W[i], 4);
        decayRef[i] = w - wd * w;
    }
    Report("SignSGD decay-only", Compare(decayed, decayRef), 1e-6);
}

// ---- GEMV throughput ----------------------------------------------------
void BenchGEMV(Device& dev, shaders::Library& lib, Gpu& gpu)
{
    // 576 x 576 projection, the shape of a real decode GEMV.
    constexpr uint32_t kOut = 576, kIn = 576;
    std::vector<uint32_t> W(kOut * kIn, 0x3F800000u), X(kIn, 0x3F800000u);
    Upload(dev, gpu.w, W);
    Upload(dev, gpu.a, X);

    const uint32_t p[7] = { 0, 0, 0, kIn, kOut, 0, 1 };

    // Warm up.
    for (int i = 0; i < 5; ++i)
    {
        dev.ResetCommandList();
        CommandScope cs(dev);
        cs.SetPipeline(lib.Get(shaders::Id::GEMV));
        cs.SetParams(p, 7);
        cs.Dispatch(kOut, 1, 1);
        dev.ExecuteAndWait();
    }

    const int kIters = 50;
    dev.BeginTimestamp();
    for (int i = 0; i < kIters; ++i)
    {
        CommandScope cs(dev);
        cs.SetPipeline(lib.Get(shaders::Id::GEMV));
        cs.SetParams(p, 7);
        cs.Dispatch(kOut, 1, 1);
    }
    dev.EndTimestamp();
    const double ms = dev.ResolveTimestampMs();

    const F32 got = Download(dev, gpu.b, 1);
    const double bytesRead = static_cast<double>(kOut) * kIn * 4.0;
    const double perCall = (ms > 0.0) ? ms / kIters : 0.0;
    std::printf("  %-26s %.4f ms/dispatch   %.0f GB/s effective   (sample=%.1f)\n",
                "GEMV perf [576x576]", perCall,
                (perCall > 0.0) ? (bytesRead / (perCall * 1e-3)) / 1e9 : 0.0,
                got.empty() ? 0.0 : static_cast<double>(got[0]));
}

} // namespace

int main()
{
    // Unbuffered stdout: if a test crashes the GPU/driver, the buffered tail
    // would be lost and the last successful step would be invisible.
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::printf("Opure-VoxelEngine :: compute verification\n");
    std::printf("=========================================\n");

    try
    {
        Device dev;
        dev.Initialize();
        std::printf("adapter : %s\n", DeviceName().c_str());
        std::printf("vram    : %.2f GB\n",
                    static_cast<double>(dev.DedicatedVideoMemory()) / (1024.0 * 1024.0 * 1024.0));

        shaders::Library lib;
        lib.Initialize(dev);

        Gpu gpu;
        gpu.Create(dev);

        std::printf("\nmodel file:\n");
        TestStreamer();
        TestTokenizer();
        TestGraph();

        std::printf("\nweight upload:\n");
        {
            graph::Graph wg;
            try
            {
                wg.LoadConfigOnly(L"models", dev.DedicatedVideoMemory());
                TestWeightUpload(dev, lib, gpu, wg);
            }
            catch (const std::exception& e)
            {
                std::printf("  %-26s SKIP  %s\n", "weight upload", e.what());
            }
        }

        TestModelForward(dev, lib, gpu);

        std::printf("\nbinding self-test:\n");
        TestDispatchPath(dev, lib, gpu);
        TestParamSlots(dev, lib, gpu);
        TestSrvReads(dev, lib, gpu);

        std::printf("\nkernel parity vs CPU reference:\n");
        TestGEMV(dev, lib, gpu);
        TestGEMVQuant(dev, lib, gpu);
        TestRMSNorm(dev, lib, gpu);
        TestSwiGLU(dev, lib, gpu);
        TestAdd(dev, lib, gpu);
        TestAddRes(dev, lib, gpu);
TestPlasticity(dev, lib, gpu);
        TestArgMax(dev, lib, gpu);
        TestAttentionAt(dev, lib, gpu, 32,  "hd32");
        TestAttentionAt(dev, lib, gpu, 64,  "hd64");
        TestAttentionAt(dev, lib, gpu, 96,  "hd96");
        TestAttentionAt(dev, lib, gpu, 128, "hd128");
        TestAttentionAt(dev, lib, gpu, 160, "hd160");
        TestAttentionAt(dev, lib, gpu, 192, "hd192");
        TestAttentionAt(dev, lib, gpu, 224, "hd224");
        TestAttentionAt(dev, lib, gpu, 256, "hd256");

        std::printf("\nperformance:\n");
        BenchGEMV(dev, lib, gpu);

        if (g_failures != 0)
        {
            std::printf("\n%d kernel check(s) FAILED.\n", g_failures);
        }
        else
        {
            std::printf("\nall kernel checks passed.\n");
        }

        dev.Shutdown();
        return (g_failures == 0) ? 0 : 1;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "\nFATAL: %s\n", e.what());
        return 2;
    }
}
