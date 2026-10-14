// ============================================================================
//  Opure_Graph.h
//  Opure-VoxelEngine -- model geometry and VRAM layout.
//
//  DIMENSION GENERICITY
//  --------------------
//  Nothing here is hardcoded to SmolLM-135M. Every field comes from
//  config.json or from the observed tensor shapes, and the only compile-time
//  assumption is that head_dim is either 64 or 128, because exactly two
//  attention entry points are compiled. A model with head_dim 80 would fail at
//  entry-point selection with a clear message rather than compute garbage.
//
//  LAYOUT
//  ------
//  All weights live in one GPU buffer, packed in safetensors order so the file
//  can be copied in nearly verbatim. Activations ping-pong between two banks;
//  the KV cache has its own region. Offsets are 32-bit WORD indices, because
//  every range crossing the ABI is expressed in words (see Opure_Common.hlsl).
//
//  WEIGHT COPIES: 0. q/k/v are read as-is; o_proj reads its input from the
//  activation bank, so no tensor is ever both a GEMV input and output.
// ============================================================================
#pragma once

#include <bit>
#include <unordered_map>

#include "core/Opure_Platform.h"
#include "core/Opure_Streamer.h"
#include "engine/Opure_ModelSpec.h"

namespace opure::json { class Value; }

namespace opure::dx12
{
    class Device;
    class GpuBuffer;
}

namespace opure::graph
{

// ---------------------------------------------------------------------------
//  ModelConfig -- parsed from config.json.
// ---------------------------------------------------------------------------
struct ModelConfig
{
    std::string modelType      = "llama";
    std::string architecture   = "LlamaForCausalLM";
    std::string hiddenAct      = "silu";

    uint32_t vocabSize         = 0;
    uint32_t hiddenSize        = 0;     // d_model
    uint32_t intermediateSize  = 0;     // d_ff
    uint32_t numLayers         = 0;
    uint32_t numHeads          = 0;     // query heads
    uint32_t numKvHeads        = 0;     // key/value heads (GQA)
    uint32_t maxPositions      = 0;
    uint32_t rmsNormEpsBits    = 0;     // eps as an fp32 bit pattern

    double    ropeTheta        = 10000.0;
    bool      tieWordEmbeddings = true;
    bool      attentionBias    = false;
    bool      mlpBias          = false;
    std::string torchDtype;            // informational only

    // "none" when rope_scaling is absent, which is every model this engine has
    // run. Anything else (llama3, longrope, yaRN, ...) changes the RoPE kernel's
    // frequency schedule, not just its base theta, so it is reported as
    // unsupported rather than approximated -- a model that loads and quietly
    // computes the wrong positions is worse than one that refuses.
    std::string ropeScalingType  = "none";
    double      ropeScalingFactor = 1.0;

    [[nodiscard]] uint32_t HeadDim()  const { return numHeads ? hiddenSize / numHeads : 0; }
    // Number of query heads sharing one KV head. 1 for MHA, >1 for GQA.
    [[nodiscard]] uint32_t KvGroupSize() const { return numHeads / numKvHeads; }

    // Total attention working set for one token, in KV words:
    //   2 (K and V) * layers * kvHeads * maxPositions * headDim
    [[nodiscard]] uint64_t KvCacheWords() const
    {
        return 2ull * numLayers * numKvHeads * maxPositions * HeadDim();
    }

    // Weight bytes, summing exactly the tensors the graph reads.
    [[nodiscard]] uint64_t WeightBytes() const;

    // Sum of every VRAM allocation the engine needs.
    [[nodiscard]] uint64_t TotalVramBytes() const;

    // Validates internal consistency. Returns false and fills `err` on
    // mismatch. Called by LoadFrom.
    [[nodiscard]] bool Validate(std::string& err) const;

    // Parses config.json from memory.
    static ModelConfig Parse(std::string_view jsonText);

    // Float bit pattern of a float, for passing eps through the parameter block.
    // constexpr so callers can bake compile-time constants (e.g. a scale of 1.0f)
    // into static parameter arrays instead of computing them per dispatch.
    static constexpr uint32_t FloatBits(float f) { return std::bit_cast<uint32_t>(f); }
};

// ---------------------------------------------------------------------------
//  VramPlan -- addresses every buffer region, plus an admission check.
// ---------------------------------------------------------------------------
struct VramPlan
{
    // Weight buffer, packed in load order.
    uint64_t weightWords  = 0;
    uint64_t weightBytes  = 0;

    // Per-tensor word offsets into the weight buffer.
    struct LayerTensors
    {
        uint64_t inputNorm   = 0;   // [hidden]
        uint64_t postAttNorm = 0;   // [hidden]
        uint64_t wq          = 0;   // [hidden, hidden]
        uint64_t wk          = 0;   // [kvHidden, hidden]
        uint64_t wv          = 0;   // [kvHidden, hidden]
        uint64_t wo          = 0;   // [hidden, hidden]
        uint64_t gate        = 0;   // [ff, hidden]
        uint64_t up          = 0;   // [ff, hidden]
        uint64_t down        = 0;   // [hidden, ff]
    };

    std::vector<LayerTensors> layers;
    uint64_t embedTokens = 0;   // [vocab, hidden]
    uint64_t finalNorm   = 0;   // [hidden]
    // The LM head. Equal to embedTokens for tied-embedding models, which is the
    // common case and costs no extra VRAM.
    uint64_t lmHead      = 0;   // [vocab, hidden]

    // Scale factors for quantized weights (one float per quantized tensor).
    // Stored after the weight buffer.
    uint64_t scaleWords  = 0;
    uint64_t scaleBytes  = 0;

    // Activation scratch. Three banks of this size (A, B and the residual bank C);
    // see ModelConfig::TotalVramBytes for why three are required.
    uint64_t actWords   = 0;
    uint64_t actBytes   = 0;
    uint64_t act2Words  = 0;

    // Plasticity staging, in the tail of every activation bank starting at
    // `plasticityOffset`. Sized to the widest projection so a SignSGD update can
    // compute into scratch and be copied back without binding the weight buffer
    // as both SRV and UAV. The inference graph never reads this range.
    uint64_t plasticityWords  = 0;
    uint64_t plasticityOffset = 0;

    // KV cache, [2][layers][kvHeads][maxPositions][headDim].
    uint64_t kvWords    = 0;
    uint64_t kvBytes    = 0;

    uint64_t totalBytes = 0;
    uint64_t availableBytes = 0;

    // False when the plan does not fit. `headroom` is the shortfall.
    bool     fits       = false;
    uint64_t headroom   = 0;
};

// ---------------------------------------------------------------------------
//  Graph -- owns the plan and the safetensors mapping, and uploads weights.
// ---------------------------------------------------------------------------
class Graph
{
public:
    Graph() = default;
    ~Graph();
    Graph(const Graph&)            = delete;
    Graph& operator=(const Graph&) = delete;

    // Opens the model directory, parses config.json, plans the layout and
    // uploads every weight. Throws opure::Error if the model does not fit.
    void Load(const std::wstring& modelDir, uint64_t availableVramBytes);

    // Plans and validates only, without touching the GPU. Call this once to size
    // the buffers, create them, then call UploadWeights().
    void LoadConfigOnly(const std::wstring& modelDir, uint64_t availableVramBytes);

    // Copies every planned tensor into an already-created weight buffer.
    void UploadWeights(dx12::Device& dev, dx12::GpuBuffer& weights);

    void Release();

    [[nodiscard]] const ModelConfig& Config()  const { return cfg_; }
    [[nodiscard]] const VramPlan&    Plan()    const { return plan_; }
    [[nodiscard]] bool               IsLoaded() const { return loaded_; }

    // Word offset of the embedding table in the weight buffer. Needed by the
    // harness to read back what was actually uploaded.
    [[nodiscard]] uint64_t EmbedTokensWord() const { return plan_.embedTokens; }

    // Check if a tensor at the given word offset is quantized, and get its scale factor.
    //
    // GetQuantizedScale deliberately does NOT throw when the offset is absent.
    // RecordForward builds the 9-parameter quantized array for every GEMV site
    // eagerly, before branching on IsQuantized, so a strict .at() would throw
    // on the F32 tensors that the engine is running today. The returned 1.0f is
    // never consumed: the dispatch that reads P[5]/P[6] is only submitted on the
    // IsQuantized() branch.
    [[nodiscard]] bool IsQuantized(uint64_t wordOffset) const { return quantizedScales_.contains(wordOffset); }
    [[nodiscard]] float GetQuantizedScale(uint64_t wordOffset) const {
        const auto it = quantizedScales_.find(wordOffset);
        return it == quantizedScales_.end() ? 1.0f : it->second;
    }

    // Which attention entry point to use for this model's head_dim. Throws if
    // head_dim is neither 64 nor 128, because no such pipeline exists.
    enum class AttEntry : uint8_t
    {
        Hd32, Hd64, Hd96, Hd128, Hd160, Hd192, Hd224, Hd256
    };
    [[nodiscard]] AttEntry AttentionEntryPoint() const;

    // The resolved file: which tensor fills each graph role, at which dtype.
    [[nodiscard]] const ModelSpec& Spec() const { return spec_; }

private:
    void PlanLayout();

    ModelConfig      cfg_;
    VramPlan         plan_;
    ModelSpec        spec_;
    safetensors::WeightStreamer stream_;
    bool             loaded_ = false;

    // Quantized tensor info: word offset -> scale factor
    std::unordered_map<uint64_t, float> quantizedScales_;
};

// Small file helpers exposed for tooling.
[[nodiscard]] std::string ReadSmallFile(const std::wstring& path);
[[nodiscard]] std::wstring JoinPath(const std::wstring& dir, const wchar_t* leaf);

} // namespace opure::graph