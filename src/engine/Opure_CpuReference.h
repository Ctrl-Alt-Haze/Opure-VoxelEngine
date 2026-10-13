// ============================================================================
//  Opure_CpuReference.h
//  Opure-VoxelEngine -- scalar CPU implementation of the same forward pass.
//
//  PURPOSE
//  ------
//  The GPU kernels are verified one at a time against small hand-written
//  references. That does NOT prove the layer GRAPH is correct: passes could be
//  sequenced wrongly, an intermediate could be routed to the wrong scratch
//  offset, or the KV cache could be addressed differently from the attention
//  kernel's expectation -- and a self-consistent GPU graph would still pass every
//  kernel test.
//
//  This class re-implements the whole forward pass independently, in plain
//  scalar C++ straight from the LLaMA equations. It shares no code with the
//  engine beyond the weight reader, so agreement between the two is real
//  evidence. It is deliberately slow and simple: clarity over speed.
//
//  Everything is computed in double and rounded to float only at the end of
//  each stage, so the reference is more accurate than the GPU, which accumulates
//  in fp32. The tolerance therefore has slack in the right direction.
// ============================================================================
#pragma once

#include "core/Opure_Platform.h"
#include "core/Opure_Streamer.h"

#include <vector>

namespace opure::cpu
{

class Reference
{
public:
    Reference() = default;
    ~Reference();
    Reference(const Reference&)            = delete;
    Reference& operator=(const Reference&) = delete;

    // Loads config.json and maps model.safetensors. Throws on failure.
    void Load(const std::wstring& modelDir);

    // Runs the full forward pass over `tokens` and writes the final-position
    // logits. `logitsOut` is resized to vocab_size.
    void Forward(const std::vector<uint32_t>& tokens, std::vector<float>& logitsOut);

    // As Forward, but stops after `nLayers` layers and appends the residual
    // stream to `layerTrace` after each one. Used to locate a divergence
    // between the GPU graph and this reference: the first layer index whose
    // residual differs is where the two implementations part company.
    void ForwardTraced(const std::vector<uint32_t>& tokens, uint32_t nLayers,
                       std::vector<float>& logitsOut,
                       std::vector<std::vector<float>>& layerTrace);

    [[nodiscard]] uint32_t VocabSize()   const { return vocabSize_; }
    [[nodiscard]] uint32_t HiddenSize()  const { return hidden_; }
    [[nodiscard]] uint32_t Layers()      const { return layers_; }

private:
    // One decoded tensor, held as floats.
    struct Tensor
    {
        std::vector<float> v;
        void Resize(size_t n) { v.assign(n, 0.0f); }
        float&       operator[](size_t i)       { return v[i]; }
        const float& operator[](size_t i) const { return v[i]; }
    };

    // y[o] = sum_i W[o * inDim + i] * x[i], matching the GEMV's weight layout.
    static void Gemv(const float* w, const float* x, float* y,
                     uint32_t outDim, uint32_t inDim);

    // y[i] = x[i] / sqrt(mean(x^2) + eps) * g[i]
    static void RmsNorm(float* x, const float* g, uint32_t d, float eps);

    // Applies RoPE in place to `vec` ([nHeads, headDim]) at `pos`.
    static void RoPE(float* vec, uint32_t nHeads, uint32_t headDim,
                     uint32_t pos, double theta);

    // in-place SwiGLU: h[i] = silu(g[i]) * u[i]
    static void SwiGLU(float* g, const float* u, float* h, uint32_t n);

    safetensors::WeightStreamer stream_;

    uint32_t vocabSize_ = 0;
    uint32_t hidden_    = 0;
    uint32_t inter_     = 0;
    uint32_t layers_    = 0;
    uint32_t heads_     = 0;
    uint32_t kvHeads_   = 0;
    uint32_t headDim_   = 0;
    uint32_t maxPos_    = 0;
    float    eps_       = 1e-5f;
    double   theta_     = 10000.0;

    // KV cache: [2][layer][kvHead][pos][headDim], K then V contiguous per layer.
    std::vector<float> kv_;

    // Per-layer parameter pointers, resolved once at load.
    struct LayerPtrs
    {
        const float* inNorm  = nullptr;
        const float* postNorm= nullptr;
        const float* wq      = nullptr;
        const float* wk      = nullptr;
        const float* wv      = nullptr;
        const float* wo      = nullptr;
        const float* gate   = nullptr;
        const float* up      = nullptr;
        const float* down   = nullptr;
        std::vector<float> inNormBuf, postNormBuf;
    };
    std::vector<LayerPtrs> lp_;
    const float* embed_ = nullptr;
    const float* finalNorm_ = nullptr;
    // Backing storage for finalNorm_; see Load() for why this must be owned.
    std::vector<float> finalNormBuf_;

    // Scratch, sized once.
    Tensor x_, normed_, q_, k_, v_, attn_, proj_, g_, u_, h_, n2_, tmp_;

    // Reads an F32 tensor into a float vector (handles BF16/F16 widening).
    std::vector<float> Read(const safetensors::Tensor& t);
};

} // namespace opure::cpu