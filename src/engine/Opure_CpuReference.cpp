// ============================================================================
//  Opure_CpuReference.cpp
//  Opure-VoxelEngine -- scalar CPU reference forward pass.
// ============================================================================
#include "engine/Opure_CpuReference.h"

#include "core/Opure_Json.h"

#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace opure::cpu
{

namespace
{

std::string ReadSmallFile(const std::wstring& path)
{
    HANDLE f = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) { ThrowError("cannot open config.json"); }
    LARGE_INTEGER sz{};
    if (!::GetFileSizeEx(f, &sz) || sz.QuadPart <= 0)
    {
        ::CloseHandle(f);
        ThrowError("config.json is empty");
    }
    std::string text(static_cast<size_t>(sz.QuadPart), '\0');
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

} // namespace

// ---------------------------------------------------------------------------
Reference::~Reference() = default;

std::vector<float> Reference::Read(const safetensors::Tensor& t)
{
    const uint64_t n = t.ElementCount();
    std::vector<float> out(n);
    const uint64_t got = safetensors::WidenToF32(t.dtype, stream_.Data(t), n, out.data());
    if (got != n * 4 && t.dtype != safetensors::Dtype::U8 && t.dtype != safetensors::Dtype::I8)
    {
        ThrowError("unexpected widened size for tensor '" + t.name + "'");
    }
    return out;
}

// ---------------------------------------------------------------------------
void Reference::Load(const std::wstring& modelDir)
{
    const json::Value cfg = json::ParseObject(ReadSmallFile(JoinPath(modelDir, L"config.json")),
                                             "config.json");
    constexpr std::string_view c = "config.json";

    vocabSize_ = static_cast<uint32_t>(cfg.GetInt("vocab_size", c));
    hidden_    = static_cast<uint32_t>(cfg.GetInt("hidden_size", c));
    inter_     = static_cast<uint32_t>(cfg.GetInt("intermediate_size", c));
    layers_    = static_cast<uint32_t>(cfg.GetInt("num_hidden_layers", c));
    heads_     = static_cast<uint32_t>(cfg.GetInt("num_attention_heads", c));
    maxPos_    = static_cast<uint32_t>(cfg.GetInt("max_position_embeddings", c));
    eps_       = static_cast<float>(cfg.GetNum("rms_norm_eps", c));
    if (const json::Value* v = cfg.Find("num_key_value_heads"))
    {
        kvHeads_ = static_cast<uint32_t>(v->inum);
    }
    else
    {
        kvHeads_ = heads_;
    }
    if (const json::Value* v = cfg.Find("rope_theta")) { theta_ = v->num; }

    OPURE_CHECK(hidden_ % heads_ == 0, "hidden_size must divide by num_attention_heads");
    headDim_ = hidden_ / heads_;

    stream_.Open(JoinPath(modelDir, L"model.safetensors"));

    // The embedding table is large (49152 x 576), so it is read on demand per
    // token rather than held twice. `embed_` points straight into the mapping
    // only when the dtype is already F32, which is the common case.
    const safetensors::Tensor& emb = stream_.Find("model.embed_tokens.weight");
    OPURE_CHECK(emb.dtype == safetensors::Dtype::F32,
                "CPU reference expects F32 embeddings for direct mapping");
    embed_ = static_cast<const float*>(stream_.Data(emb));

    // Read() returns by value, so the data must be OWNED, not pointed at.
    // Assigning `.data()` straight into a raw pointer would leave it aimed at
    // a vector destroyed at the end of the statement -- a dangling pointer that
    // reads freed memory and corrupts the heap.
    finalNormBuf_ = Read(stream_.Find("model.norm.weight"));
    finalNorm_    = finalNormBuf_.data();

    lp_.resize(layers_);
    for (uint32_t l = 0; l < layers_; ++l)
    {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        LayerPtrs& P = lp_[l];
        P.inNormBuf   = Read(stream_.Find(p + "input_layernorm.weight"));
        P.postNormBuf = Read(stream_.Find(p + "post_attention_layernorm.weight"));
        P.inNorm      = P.inNormBuf.data();
        P.postNorm    = P.postNormBuf.data();
        P.wq   = static_cast<const float*>(stream_.Data(stream_.Find(p + "self_attn.q_proj.weight")));
        P.wk   = static_cast<const float*>(stream_.Data(stream_.Find(p + "self_attn.k_proj.weight")));
        P.wv   = static_cast<const float*>(stream_.Data(stream_.Find(p + "self_attn.v_proj.weight")));
        P.wo   = static_cast<const float*>(stream_.Data(stream_.Find(p + "self_attn.o_proj.weight")));
        P.gate = static_cast<const float*>(stream_.Data(stream_.Find(p + "mlp.gate_proj.weight")));
        P.up   = static_cast<const float*>(stream_.Data(stream_.Find(p + "mlp.up_proj.weight")));
        P.down = static_cast<const float*>(stream_.Data(stream_.Find(p + "mlp.down_proj.weight")));
    }

    // KV cache: [2 * layers_] planes of [kvHeads_ * maxPos_ * headDim_].
    kv_.assign(static_cast<size_t>(2) * layers_ * kvHeads_ * maxPos_ * headDim_, 0.0f);

    x_.Resize(hidden_);
    normed_.Resize(hidden_);
    n2_.Resize(hidden_);
    q_.Resize(hidden_);
    attn_.Resize(hidden_);
    proj_.Resize(hidden_);
    // tmp_ must hold the LOGITS, which are vocabSize wide because the LM head is
    // tied to the embedding matrix. Sizing it max(intermediate, hidden) instead
    // overflows it by 32x on a 135M model -- a silent heap corruption that
    // surfaces far from the cause.
    tmp_.Resize(std::max<uint64_t>({inter_, hidden_, vocabSize_}));
    k_.Resize(static_cast<size_t>(kvHeads_) * headDim_);
    v_.Resize(static_cast<size_t>(kvHeads_) * headDim_);
    g_.Resize(inter_);
    u_.Resize(inter_);
    h_.Resize(inter_);
}

// ---------------------------------------------------------------------------
void Reference::Gemv(const float* w, const float* x, float* y,
                     uint32_t outDim, uint32_t inDim)
{
    for (uint32_t o = 0; o < outDim; ++o)
    {
        double acc = 0.0;
        const float* row = w + static_cast<size_t>(o) * inDim;
        for (uint32_t i = 0; i < inDim; ++i) { acc += double(row[i]) * double(x[i]); }
        y[o] = static_cast<float>(acc);
    }
}

void Reference::RmsNorm(float* x, const float* g, uint32_t d, float eps)
{
    double ss = 0.0;
    for (uint32_t i = 0; i < d; ++i) { ss += double(x[i]) * double(x[i]); }
    const double inv = 1.0 / std::sqrt(ss / d + eps);
    for (uint32_t i = 0; i < d; ++i) { x[i] = float(double(x[i]) * inv * double(g[i])); }
}

void Reference::RoPE(float* vec, uint32_t nHeads, uint32_t headDim,
                     uint32_t pos, double theta)
{
    for (uint32_t head = 0; head < nHeads; ++head)
    {
        float* row = vec + static_cast<size_t>(head) * headDim;
        for (uint32_t i = 0; i + 1 < headDim; i += 2)
        {
            const double freq = std::pow(theta, -2.0 * double(i) / double(headDim));
            const double t = double(pos) * freq;
            const double ct = std::cos(t);
            const double st = std::sin(t);
            const double e0 = row[i];
            const double e1 = row[i + 1];
            row[i]     = float(e0 * ct - e1 * st);
            row[i + 1] = float(e0 * st + e1 * ct);
        }
    }
}

void Reference::SwiGLU(float* g, const float* u, float* h, uint32_t n)
{
    for (uint32_t i = 0; i < n; ++i)
    {
        const float s = 1.0f / (1.0f + std::exp(-g[i]));
        h[i] = (g[i] * s) * u[i];
    }
}

// ---------------------------------------------------------------------------
void Reference::Forward(const std::vector<uint32_t>& tokens, std::vector<float>& logitsOut)
{
    std::vector<std::vector<float>> trace;
    ForwardTraced(tokens, 0xFFFFFFFFu, logitsOut, trace);
}

void Reference::ForwardTraced(const std::vector<uint32_t>& tokens, uint32_t nLayers,
                              std::vector<float>& logitsOut,
                              std::vector<std::vector<float>>& layerTrace)
{
    layerTrace.clear();
    OPURE_CHECK(!tokens.empty(), "CPU reference called with no tokens");
    OPURE_CHECK(tokens.size() <= maxPos_, "prompt exceeds max_position_embeddings");

    const uint32_t kvDim = static_cast<uint32_t>(kvHeads_) * headDim_;
    const uint32_t kvPlane = kvDim * maxPos_;                 // per layer, per K/V
    const uint32_t kvGroup = heads_ / kvHeads_;

    for (size_t t = 0; t < tokens.size(); ++t)
    {
        const uint32_t tok = tokens[t];
        OPURE_CHECK(tok < vocabSize_, "token id out of vocabulary range");

        // embedding
        std::memcpy(x_.v.data(), embed_ + static_cast<size_t>(tok) * hidden_,
                    static_cast<size_t>(hidden_) * 4);

        for (uint32_t L = 0; L < layers_; ++L)
        {
            const LayerPtrs& P = lp_[L];
            (void)P;

            // ---- attention ----
            std::memcpy(normed_.v.data(), x_.v.data(), static_cast<size_t>(hidden_) * 4);
            RmsNorm(normed_.v.data(), P.inNorm, hidden_, eps_);

            Gemv(P.wq, normed_.v.data(), q_.v.data(), heads_ * headDim_, hidden_);
            Gemv(P.wk, normed_.v.data(), k_.v.data(), kvDim, hidden_);
            Gemv(P.wv, normed_.v.data(), v_.v.data(), kvDim, hidden_);

            RoPE(q_.v.data(), heads_, headDim_, static_cast<uint32_t>(t), theta_);
            RoPE(k_.v.data(), kvHeads_, headDim_, static_cast<uint32_t>(t), theta_);

            // Append K and V to the cache at position t, matching the shader's
            // kSlot/vSlot arithmetic exactly: K at [h * maxSeq + pos], V the
            // same offset within the second half of the layer's plane.
            float* kBase = kv_.data() + static_cast<size_t>(L) * 2 * kvPlane;
            float* vBase = kBase + kvPlane;
            for (uint32_t hh = 0; hh < kvHeads_; ++hh)
            {
                std::memcpy(kBase + static_cast<size_t>(hh) * maxPos_ * headDim_ +
                                static_cast<size_t>(t) * headDim_,
                            k_.v.data() + hh * headDim_, headDim_ * 4);
                std::memcpy(vBase + static_cast<size_t>(hh) * maxPos_ * headDim_ +
                                static_cast<size_t>(t) * headDim_,
                            v_.v.data() + hh * headDim_, headDim_ * 4);
            }

            // Online-softmax attention per query head.
            const uint32_t seqLen = static_cast<uint32_t>(t) + 1;
            for (uint32_t hq = 0; hq < heads_; ++hq)
            {
                const uint32_t kvh = hq / kvGroup;
                const float* qRow = q_.v.data() + static_cast<size_t>(hq) * headDim_;

                double m = -1e300;
                std::vector<double> scores(seqLen);
                for (uint32_t s = 0; s < seqLen; ++s)
                {
                    const float* kRow = kBase +
                        static_cast<size_t>(kvh) * maxPos_ * headDim_ +
                        static_cast<size_t>(s) * headDim_;
                    double sc = 0.0;
                    for (uint32_t d = 0; d < headDim_; ++d) { sc += double(qRow[d]) * double(kRow[d]); }
                    scores[s] = sc;
                    if (sc > m) { m = sc; }
                }

                double sum = 0.0;
                for (uint32_t s = 0; s < seqLen; ++s)
                {
                    const double e = std::exp(scores[s] - m);
                    scores[s] = e;
                    sum += e;
                }

                float* oRow = attn_.v.data() + static_cast<size_t>(hq) * headDim_;
                for (uint32_t d = 0; d < headDim_; ++d) { oRow[d] = 0.0f; }
                for (uint32_t s = 0; s < seqLen; ++s)
                {
                    const float* vRow = vBase +
                        static_cast<size_t>(kvh) * maxPos_ * headDim_ +
                        static_cast<size_t>(s) * headDim_;
                    const double w = scores[s];
                    for (uint32_t d = 0; d < headDim_; ++d)
                    {
                        oRow[d] = float(double(oRow[d]) + w * double(vRow[d]));
                    }
                }
                const double invSum = 1.0 / (sum > 0.0 ? sum : 1.0);
                for (uint32_t d = 0; d < headDim_; ++d)
                {
                    oRow[d] = float(double(oRow[d]) * invSum);
                }
            }

            Gemv(P.wo, attn_.v.data(), proj_.v.data(), hidden_, hidden_);
            for (uint32_t i = 0; i < hidden_; ++i)
            {
                x_[i] = x_[i] + proj_[i];
            }

            // ---- MLP ----
            std::memcpy(n2_.v.data(), x_.v.data(), static_cast<size_t>(hidden_) * 4);
            RmsNorm(n2_.v.data(), P.postNorm, hidden_, eps_);

            Gemv(P.gate, n2_.v.data(), g_.v.data(), inter_, hidden_);
            Gemv(P.up,   n2_.v.data(), u_.v.data(), inter_, hidden_);
            SwiGLU(g_.v.data(), u_.v.data(), h_.v.data(), inter_);
            Gemv(P.down, h_.v.data(), proj_.v.data(), hidden_, inter_);
            for (uint32_t i = 0; i < hidden_; ++i)
            {
                x_[i] = x_[i] + proj_[i];
            }

            // Trace the residual so the harness can find the first divergent
            // layer without bisecting by hand.
            layerTrace.push_back(x_.v);

            if (L + 1 >= nLayers)
            {
                logitsOut.assign(static_cast<size_t>(hidden_), 0.0f);
                for (uint32_t i = 0; i < hidden_; ++i) { logitsOut[i] = x_[i]; }
                return;
            }
        }

        // ---- final norm + tied LM head ----
        RmsNorm(x_.v.data(), finalNorm_, hidden_, eps_);
        Gemv(embed_, x_.v.data(), tmp_.v.data(), vocabSize_, hidden_);

        if (t + 1 == tokens.size())
        {
            logitsOut.assign(tmp_.v.begin(), tmp_.v.begin() + vocabSize_);
        }
    }
}

} // namespace opure::cpu