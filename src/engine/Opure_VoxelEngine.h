// ============================================================================
//  Opure_VoxelEngine.h
//  Opure-VoxelEngine -- the transformer forward pass and generation loop.
//
//  THE GRAPH, IN ORDER
//  -------------------
//  For one token at position t:
//
//      x = embed[token]                         -> bank A[0 .. dModel)
//      for each layer L:
//          n1 = RMSNorm(x, input_layernorm)    -> bank B
//          q  = n1 @ Wq^T                      -> scratch
//          k  = n1 @ Wk^T                      -> scratch   (rotated, appended to KV)
//          v  = n1 @ Wv^T                      -> scratch
//          attn = Attention(q, KV[L], t)        -> scratch
//          x2  = attn @ Wo^T + x                -> bank A   (residual)
//          n2  = RMSNorm(x2, post_attn_norm)    -> bank B
//          g   = n2 @ Wgate^T                  -> scratch
//          u   = n2 @ Wup^T                    -> scratch
//          h   = SwiGLU(g, u)                  -> scratch
//          x3  = h @ Wdown^T + x2              -> bank A   (residual)
//      logits = finalNorm(x3) @ embed^T        -> scratch
//      next   = ArgMax(logits)
//
//  PING-PONG DISCIPLINE
//  --------------------
//  D3D12 forbids one resource being both SRV and UAV in a dispatch, so every
//  pass reads one bank and writes the other. `dir` is the direction: 0 means
//  read A write B, 1 means read B write A. The direction is flipped after each
//  pass that switches banks. Getting this wrong does not crash -- it reads
//  whatever the previous pass happened to leave -- so it is threaded explicitly
//  rather than recomputed, and the harness checks the whole model against a CPU
//  reference.
//
//  ONE TOKEN, ONE COMMAND LIST
//  ---------------------------
//  All ~200 dispatches for a token are recorded into a single command list and
//  the CPU waits once. This is what makes sub-millisecond tokens possible; a
//  wait per dispatch would cost more than the work.
// ============================================================================
#pragma once

#include "core/Opure_Dx12.h"
#include "core/Opure_Shaders.h"
#include "engine/Opure_Graph.h"
#include "model/Opure_Tokenizer.h"

namespace opure::engine
{

// ---------------------------------------------------------------------------
//  Engine options
// ---------------------------------------------------------------------------
struct EngineOptions
{
    // Enables the SignSGD plasticity pass during idle time. Off by default
    // because it MUTATES weights: a parity run must be bit-reproducible.
    bool enablePlasticity = false;

    // Hard cap on generated tokens, independent of the caller's budget.
    uint32_t maxNewTokens = 256;

    // Temperature. 0 means greedy (argmax), which is what the parity test uses.
    float    temperature   = 0.0f;
};

// ---------------------------------------------------------------------------
//  Generation result
// ---------------------------------------------------------------------------
// Describes one idle-time plasticity update.
//
// The gradient is the outer product of an input activation with a per-row error
// signal: g[o][i] = act[i] * err[o] * gradientScale. Both vectors are read from
// activation banks, so the update is a pure function of buffers the graph
// already holds -- there is no optimiser state, no shadow weight copy, and no
// CPU round trip.
struct PlasticityStep
{
    uint64_t weightOffset = 0;    // word offset of the target [rows, cols] matrix
    uint32_t rows         = 0;    // outDim
    uint32_t cols         = 0;    // inDim

    uint32_t actBank      = 0;    // bank holding act[0 .. cols)
    uint32_t actOffset    = 0;    // word offset of act
    uint32_t errBank      = 0;    // bank holding err[0 .. rows)
    uint32_t errOffset    = 0;    // word offset of err

    // Bank the updated matrix is computed into before being committed. Must
    // differ from BOTH actBank and errBank: the pass reads those through their
    // SRVs and writes this one through its UAV, and D3D12 leaves an SRV/UAV
    // alias undefined. 0xFFFFFFFF means "the third bank".
    uint32_t stageBank    = 0xFFFFFFFFu;

    float    learningRate    = 1e-4f;
    float    weightDecay     = 0.0f;
    float    gradientScale   = 1.0f;
};

struct GenerateResult
{
    std::vector<uint32_t> promptTokens;
    std::vector<uint32_t> generated;
    uint32_t              stopToken   = 0;
    bool                  hitStop     = false;
    double                msPerToken  = 0.0;
    double                lastMs       = 0.0;
    // Top-k of the final position, for the semantic sanity check.
    std::vector<std::pair<uint32_t, float>> topK;
};

// ---------------------------------------------------------------------------
//  VoxelEngine
// ---------------------------------------------------------------------------
class VoxelEngine
{
public:
    VoxelEngine() = default;
    ~VoxelEngine();
    VoxelEngine(const VoxelEngine&)            = delete;
    VoxelEngine& operator=(const VoxelEngine&) = delete;

    // Loads the model, allocates VRAM and uploads weights. Throws on failure,
    // including when the model does not fit in the reported VRAM.
    //
    // `dev` and `lib` are borrowed, not owned: the caller (the harness) creates
    // them once and outlives the engine.
    void Initialize(const std::wstring& modelDir, const EngineOptions& opts,
                    dx12::Device* dev, shaders::Library* lib);

    void Shutdown();
    [[nodiscard]] bool IsReady() const { return ready_; }

    [[nodiscard]] const graph::Graph&  GraphModel() const { return graph_; }
    [[nodiscard]] const tokenizer::BpeTokenizer& Tokenizer() const { return tokenizer_; }

    // Prefills `tokens` and generates until a stop token or the budget.
    [[nodiscard]] GenerateResult Generate(const std::vector<uint32_t>& tokens,
                                          const EngineOptions& opts);

    // Prefills `tokens` and runs exactly one forward pass, returning the
    // argmax of the final position. This is the parity entry point.
    [[nodiscard]] uint32_t ForwardArgMax(const std::vector<uint32_t>& tokens);

    // Logits of the final position, downloaded to the host. Used by the
    // model-level parity test.
    void ForwardLogits(const std::vector<uint32_t>& tokens,
                       std::vector<float>& logitsOut);

    // Runs one SignSGD plasticity step over a target projection matrix.
    //
    // Two dispatches: the update is computed into the activation banks' staging
    // tail, then copied back over the tensor. Splitting it that way is required,
    // not stylistic -- a single pass would bind the weight buffer as both SRV
    // and UAV, which D3D12 leaves undefined and this engine forbids everywhere
    // else. Net effect on the weights is an in-place, state-free update.
    void RunPlasticity(const PlasticityStep& step);

    // Convenience form: applies one step to a named tensor of `layer` using the
    // default hyperparameters. `tensor` is one of "q_proj", "k_proj", "v_proj",
    // "o_proj", "gate_proj", "up_proj", "down_proj".
    void RunPlasticity(uint32_t layer, const char* tensor,
                       float lr, float weightDecay);

    // As above, but with an explicit per-row error signal. A null `error`
    // stages a zero vector, which makes the step decay-only.
    void RunPlasticityWithError(PlasticityStep step, const float* error);

    // Reads `count` words out of the GPU weight buffer at `wordOffset`. Exposed
    // for harness diagnostics: it is the only way to tell "the upload failed"
    // apart from "the graph is wrong", which look identical from the logits.
    void ReadWeightsAt(uint64_t wordOffset, uint32_t* dst, size_t count);

    // Reads from an activation bank (0=A, 1=B, 2=C) at a word offset.
    void ReadActBank(uint32_t bank, uint64_t wordOffset, uint32_t* dst, size_t count);

    // Diagnostic only: reads the KV cache, so a broken RoPE -> attention
    // handoff can be checked against the activation it was supposed to cache.
    void ReadKv(uint64_t wordOffset, uint32_t* dst, size_t count);

    // The live weight buffer, so a caller can restore or seed weights without
    // going back through the loader.
    [[nodiscard]] dx12::GpuBuffer&       Weights()       { return weights_; }
    [[nodiscard]] const graph::VramPlan& GraphPlan() const { return graph_.Plan(); }
    [[nodiscard]] const graph::ModelConfig& Model()  const { return graph_.Config(); }

    // Records the whole forward pass for `tokenId` at `position`, leaving the
    // logits in `*logitBank`. Does NOT execute.
    void RecordForward(const uint32_t tokenId, uint32_t position, uint32_t* logitBank);

    // Records every prompt token, chunked so no single command list exceeds the
    // parameter-slot budget. Executes each chunk.
    void Prefill(const std::vector<uint32_t>& tokens, uint32_t* logitBank);

    // Appends `tokenId` at `position` and returns the argmax of the resulting
    // logits. Earlier positions come from the KV cache, so this is O(1) passes
    // per generated token rather than O(sequence length).
    uint32_t DecodeStep(uint32_t tokenId, uint32_t position);

    // Evicts the oldest KV cache entry by shifting all entries left by one.
    // Called when the sequence reaches max_position_embeddings.
    void EvictOldestKV();

    // Reduces logits already present in `logitBank` to their argmax token id.
    uint32_t ArgMaxFrom(uint32_t logitBank);

    // Records the embedding and the first `nLayers` layers, then reads the
    // residual stream back. Used by the harness to find the first layer where
    // the GPU graph and the CPU reference disagree.
    void RunLayersTraced(uint32_t tokenId, uint32_t nLayers,
                         std::vector<std::vector<float>>& trace);

    // Diagnostic only: runs the embedding and layer `layer`, stopping after
    // exactly `budget` dispatches, and returns the bank holding the residual.
    uint32_t RunPartial(uint32_t tokenId, uint32_t budget, uint32_t layer);

    // Diagnostic only: fills all three activation banks with `fill`. Exposed so
    // the harness can prove every word the graph reads was written by it.
    void PoisonBanks(float fill);

    // ---------------------------------------------------------------------------
    //  Self-monitoring / observability
    // ---------------------------------------------------------------------------
    struct MetricsSnapshot
    {
        uint64_t timestampMs = 0;
        std::vector<float> layerTimesMs;
        float kvOccupancy = 0.0f;
        float logitEntropy = 0.0f;
        float tokensPerSec = 0.0f;
        float memoryPressure = 0.0f;
        float speculativeAcceptance = -1.0f;
        float logitNorm = 0.0f;
        float maxAttnScore = 0.0f;
    };

    // Takes a snapshot of runtime metrics after the last token generation.
    // Call after each DecodeStep to track model behavior.
    MetricsSnapshot SnapshotMetrics();

    // Current sequence length (tokens written to KV cache).
    [[nodiscard]] uint32_t CurrentPosition() const { return seqLen_; }

private:
    // Borrowed from the caller; see Initialize().
    dx12::Device*          dev_   = nullptr;
    shaders::Library*      lib_   = nullptr;

    dx12::GpuBuffer        weights_, actA_, actB_, res_, kv_;
    graph::Graph           graph_;
    tokenizer::BpeTokenizer tokenizer_;

    EngineOptions          opts_{};
    bool                   ready_ = false;

    // Self-monitoring state (friend MetricsCollector for access).
    uint64_t               lastTokenTimeMs_ = 0;
    float                  lastLogitEntropy_ = 0.0f;
    float                  lastLogitNorm_ = 0.0f;
    float                  lastMaxAttnScore_ = 0.0f;

    // KV cache bookkeeping. seqLen_ counts tokens written so far.
    uint32_t               seqLen_ = 0;

    friend class MetricsCollector;
    uint32_t               kvLayerWords_ = 0;   // words of KV per layer

    // Scratch offsets inside ONE activation bank, in words.
    uint32_t               scratchQ_ = 0;
    uint32_t               scratchK_ = 0;
    uint32_t               scratchV_ = 0;
    uint32_t               scratchA_ = 0;
    uint32_t               scratchG_ = 0;
    uint32_t               scratchU_ = 0;
    uint32_t               scratchH_ = 0;
    uint32_t               scratchL_ = 0;

    // Bank holding the residual stream, rotated across layers so that every
    // residual add can read two banks and write the third.
    uint32_t               resBank_ = 0;

    // Residual bank after the most recent RecordForward, so successive
    // one-layer diagnostic calls chain correctly.
    uint32_t               curResBank_ = 0;

    // Diagnostic-only: number of layers RecordForward may record (defaults to
    // every layer), and an optional sink that, when set, stops RecordForward
    // after a single layer so the caller can execute and read the residual.
    uint32_t                              layerLimit_ = 0xFFFFFFFFu;
    std::vector<std::vector<float>>*      traceSink_  = nullptr;

    // Diagnostic-only: first layer RecordForward may record. The per-layer
    // trace runs one layer per command list and needs to request layer N, not
    // always layer 0.
    uint32_t                              layerStart_ = 0;

    // Diagnostic-only: suppress the embedding dispatch so a traced layer call
    // continues from the residual the previous layer left behind.
    bool                                 skipEmbed_  = false;

    // Diagnostic-only dispatch budget. GO() counts dispatches and RecordForward
    // returns the moment the count reaches `dispatchLimit_`, so RunPartial can
    // execute and read the banks after an exact number of passes.
    uint32_t                              dispatchLimit_ = 0xFFFFFFFFu;
    uint32_t                              dispatches_    = 0;

    // Residual bank while RecordForward is running. A member rather than a local
    // so GO() can report it when the dispatch budget is hit.
    uint32_t               rb_ = 0;

    // Two-word tail of the KV buffer where ArgMax deposits its result, so the
    // argmax dispatch never writes a bank it reads.
    uint32_t               argmaxSlot_ = 0;

    uint32_t               curToken_ = 0;
};

} // namespace opure::engine