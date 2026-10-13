// ============================================================================
//  Opure_Graph.cpp
//  Opure-VoxelEngine -- model geometry and VRAM layout.
// ============================================================================
#include "engine/Opure_Graph.h"

#include "core/Opure_Json.h"
#include "engine/Opure_ModelSpec.h"

#include "core/Opure_Dx12.h"
#include "core/Opure_Json.h"

#include <windows.h>

#include <cstring>
#include <string>

namespace opure::graph
{

// ---------------------------------------------------------------------------
//  Small file helpers used by the graph and by tooling.
// ---------------------------------------------------------------------------

// Reads a whole small file into memory. Used for config.json only; the model
// weights go through WeightStreamer's mapping instead.
std::string ReadSmallFile(const std::wstring& path)
{
    HANDLE f = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE)
    {
        ThrowError("cannot open config.json in the model directory");
    }
    LARGE_INTEGER sz;
    ::GetFileSizeEx(f, &sz);
    std::string out(static_cast<size_t>(sz.QuadPart), '\0');
    DWORD read = 0;
    ::ReadFile(f, out.data(), static_cast<DWORD>(sz.QuadPart), &read, nullptr);
    ::CloseHandle(f);
    return out;
}

// Appends a leaf name to a directory path, inserting a backslash if needed.
std::wstring JoinPath(const std::wstring& dir, const wchar_t* leaf)
{
    if (dir.empty()) { return leaf; }
    wchar_t last = dir.back();
    if (last == L'\\' || last == L'/') { return dir + leaf; }
    return dir + L'\\' + leaf;
}

namespace
{

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

    if (const json::Value* v = root.Find("rope_scaling");
        v != nullptr && v->IsObject())
    {
        // Newer checkpoints write "rope_type"; older ones write "type".
        if (const json::Value* rt = v->Find("rope_type"); rt != nullptr && rt->IsString())
        {
            c.ropeScalingType = rt->str;
        }
        else if (const json::Value* ty = v->Find("type"); ty != nullptr && ty->IsString())
        {
            c.ropeScalingType = ty->str;
        }
        if (const json::Value* f = v->Find("factor"); f != nullptr)
        {
            c.ropeScalingFactor = f->num;
        }
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

    // Resolve the file into graph roles BEFORE planning, so a model this engine
    // cannot serve is refused with a specific reason instead of an
    // out-of-range or a name-not-found deep inside layout.
    spec_.Resolve(cfg_, stream_);
    spec_.Check(cfg_);
    if (!spec_.Supported())
    {
        ThrowError("model is not supported by this engine:\n" + spec_.Explain());
    }

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
        // The resident format is always F32, so every tensor occupies four bytes
        // per element regardless of how the file stores it; a BF16 source is
        // widened on the host at upload time. 256-byte alignment keeps every row
        // of a GEMV 64-byte-segment friendly.
        cursor += Align64(t.ElementCount() * 4, 256);
        return at / 4;   // word offset
    };

    auto placeRef = [&](const TensorRef& r) -> uint64_t
    {
        return place(stream_.Find(r.name));
    };

    plan_.embedTokens = placeRef(spec_.Embed());
    plan_.finalNorm   = placeRef(spec_.FinalNorm());
    // A tied model has no separate head; the graph points the LM head at the
    // embedding table, so no extra VRAM is reserved.
    if (spec_.LmHead().Present() && !cfg_.tieWordEmbeddings)
    {
        plan_.lmHead = placeRef(spec_.LmHead());
    }
    else
    {
        plan_.lmHead = plan_.embedTokens;
    }

    plan_.layers.resize(cfg_.numLayers);
    for (uint32_t l = 0; l < cfg_.numLayers; ++l)
    {
        const BlockTensors& b = spec_.Blocks()[l];
        VramPlan::LayerTensors& lt = plan_.layers[l];
        lt.inputNorm   = placeRef(b.inputNorm);
        lt.postAttNorm = placeRef(b.postAttNorm);
        lt.wq          = placeRef(b.wq);
        lt.wk          = placeRef(b.wk);
        lt.wv          = placeRef(b.wv);
        lt.wo          = placeRef(b.wo);
        lt.gate        = placeRef(b.gate);
        lt.up          = placeRef(b.up);
        lt.down        = placeRef(b.down);
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
    // Which precompiled attention pipeline serves this model's head_dim.
    // Keep in sync with Library::AttentionFor; this one exists so callers can
    // ask the graph rather than reaching into the shader library.
    const uint32_t hd = cfg_.HeadDim();
    switch (hd)
    {
    case 32:  return AttEntry::Hd32;
    case 64:  return AttEntry::Hd64;
    case 96:  return AttEntry::Hd96;
    case 128: return AttEntry::Hd128;
    case 160: return AttEntry::Hd160;
    case 192: return AttEntry::Hd192;
    case 224: return AttEntry::Hd224;
    case 256: return AttEntry::Hd256;
    default:
        // ModelSpec::Check refuses these at load time, so reaching here means
        // LoadConfigOnly was bypassed.
        ThrowError("head_dim " + std::to_string(hd) +
                   " has no compiled attention kernel (must be a multiple of 32, "
                   "at most 256)");
    }
}

// Copies every weight into the GPU weight buffer in planned order.
//
// The source is the memory-mapped file itself, so nothing is staged on the host
// beyond the upload buffer inside Device::UploadRanges. One command list per
// 32 MB batch, not one per tensor.
void Graph::UploadWeights(dx12::Device& dev, dx12::GpuBuffer& weights)
{
    // Two paths, because the resident format is always F32 but the file usually
    // is not.
    //
    //   F32  -> zero-copy: hand the mapping's bytes straight to the uploader.
    //   other -> widen on the host into ONE staging buffer, then upload ranges
    //            into it. A per-tensor temporary would alias, since every
    //            converted range would point at the same reused allocation.
    //
    // The previous code copied bytes verbatim regardless of dtype, so a BF16
    // model -- which is what nearly every real checkpoint ships -- would have put
    // 2-byte payloads into 4-byte slots and produced a model that loaded cleanly
    // and computed nonsense.
    struct Piece { uint64_t wordOffset; const TensorRef* ref; };

    std::vector<Piece> pieces;
    pieces.reserve(2 + plan_.layers.size() * 9);

    pieces.push_back({ plan_.embedTokens, &spec_.Embed() });
    pieces.push_back({ plan_.finalNorm,   &spec_.FinalNorm() });
    if (plan_.lmHead != plan_.embedTokens)
    {
        pieces.push_back({ plan_.lmHead, &spec_.LmHead() });
    }

    for (size_t l = 0; l < plan_.layers.size(); ++l)
    {
        const BlockTensors& b = spec_.Blocks()[l];
        const VramPlan::LayerTensors& t = plan_.layers[l];
        pieces.push_back({ t.inputNorm,   &b.inputNorm });
        pieces.push_back({ t.postAttNorm, &b.postAttNorm });
        pieces.push_back({ t.wq, &b.wq });
        pieces.push_back({ t.wk, &b.wk });
        pieces.push_back({ t.wv, &b.wv });
        pieces.push_back({ t.wo, &b.wo });
        pieces.push_back({ t.gate, &b.gate });
        pieces.push_back({ t.up,   &b.up });
        pieces.push_back({ t.down, &b.down });
    }

    // ---- pass 1: raw F32 ranges -------------------------------------------
    std::vector<dx12::Device::UploadRange> ranges;
    ranges.reserve(pieces.size());

    // ---- pass 2 staging layout (allocated only if something needs widening)
    std::vector<uint64_t> stageOffset(pieces.size(), 0);
    uint64_t stageWords = 0;
    for (size_t i = 0; i < pieces.size(); ++i)
    {
        if (pieces[i].ref->dtype == safetensors::Dtype::F32) { continue; }
        stageOffset[i] = stageWords;
        stageWords += pieces[i].ref->Elements();
    }

    std::vector<float> staging;
    if (stageWords != 0)
    {
        staging.resize(static_cast<size_t>(stageWords));
    }

    for (size_t i = 0; i < pieces.size(); ++i)
    {
        const TensorRef& r = *pieces[i].ref;
        const safetensors::Tensor& t = stream_.Find(r.name);

        if (r.dtype == safetensors::Dtype::F32)
        {
            ranges.push_back({ stream_.Data(t), t.ByteCount(), pieces[i].wordOffset * 4 });
        }
        else
        {
            safetensors::WidenToF32(r.dtype, stream_.Data(t), r.Elements(),
                                    staging.data() + stageOffset[i]);
            ranges.push_back({ staging.data() + stageOffset[i],
                               r.Elements() * 4,
                               pieces[i].wordOffset * 4 });
        }
    }

    dev.UploadRanges(weights, ranges);
}

} // namespace opure::graph