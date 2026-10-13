// ============================================================================
//  Opure_ModelSpec.h
//  Opure-VoxelEngine -- resolves a .safetensors file into graph roles.
//
//  WHY THIS EXISTS
//  --------------
//  The graph does not want tensor NAMES, it wants tensor ROLES: "the query
//  projection of layer 7". Those two are not the same thing. Hugging Face has
//  shipped at least three naming conventions for the same architecture:
//
//      model.layers.7.self_attn.q_proj.weight     (Llama, Qwen, Mistral, SmolLM)
//      model.layers.7.attention.wq.weight         (Falcon, RefinedWeb)
//      transformer.h.7.attn.c_attn.wq.weight      (BLOOM / GPT-2 lineage)
//
//  and several of them use ".bias" tensors this engine does not consume. The
//  previous version hardcoded the first convention, which meant the engine could
//  only ever open a Llama-named model even though the math underneath was
//  architecture-generic.
//
//  A ModelSpec is the resolved answer: for every role, which tensor in the file
//  fills it, what dtype it is, and what shape it must have. Graph planning and
//  weight upload both read the spec instead of guessing names.
//
//  DTYPE
//  -----
//  This engine's GEMV consumes 32-bit words, so the resident format is always
//  F32. Real models overwhelmingly ship BF16 or F16, and the previous upload path
//  copied tensor bytes verbatim -- which would have put 2-byte payloads into
//  4-byte slots and produced a model that loads cleanly and computes garbage.
//  The spec records each tensor's source dtype so the upload can widen it.
//
//  SCOPE, STATED PLAINLY
//  --------------------
//  "Any model" is not achievable and this class does not pretend to be. It
//  resolves llama-family DECODER stacks: pre-norm attention, SwiGLU MLP, RoPE,
//  optional GQA, optional tied embeddings, RMSNorm. Architectures with a
//  different block structure -- MoE routing, Mamba/SSM state, multi-modal
//  towers, sliding-window or learned attention biases -- are reported as
//  unsupported with the specific reason, not silently mis-loaded.
// ============================================================================
#pragma once

#include "core/Opure_Platform.h"
#include "core/Opure_Streamer.h"

#include <string>
#include <vector>

namespace opure::graph
{

struct ModelConfig;

// ---------------------------------------------------------------------------
//  One resolved tensor, with the invariants the graph depends on.
// ---------------------------------------------------------------------------
struct TensorRef
{
    std::string              name;       // empty when absent
    safetensors::Dtype       dtype  = safetensors::Dtype::F32;
    std::vector<int64_t>     shape;      // as stored in the file
    [[nodiscard]] bool  Present()  const { return !name.empty(); }
    [[nodiscard]] uint64_t Elements() const;
    [[nodiscard]] bool  Matches(std::initializer_list<int64_t> dims) const;
};

// ---------------------------------------------------------------------------
//  The nine tensors of one decoder block, plus the bookends.
// ---------------------------------------------------------------------------
struct BlockTensors
{
    TensorRef inputNorm;      // [hidden]
    TensorRef postAttNorm;    // [hidden]
    TensorRef wq;             // [hidden,   hidden]
    TensorRef wk;             // [kvHidden, hidden]
    TensorRef wv;             // [kvHidden, hidden]
    TensorRef wo;             // [hidden,   hidden]
    TensorRef gate;           // [ff,       hidden]
    TensorRef up;             // [ff,       hidden]
    TensorRef down;           // [hidden,   ff]
};

// ---------------------------------------------------------------------------
//  A single reason a model cannot be served, with enough detail to act on.
// ---------------------------------------------------------------------------
struct SupportIssue
{
    std::string reason;   // e.g. "head_dim 96 has no compiled attention kernel"
    std::string fix;      // e.g. "add OPURE_ATT_THREADS=96 to the shader manifest"
    bool        fatal = true;
};

// ---------------------------------------------------------------------------
//  ModelSpec -- the resolved file, plus what the engine can do with it.
// ---------------------------------------------------------------------------
class ModelSpec
{
public:
    // Inspects `stream` against `cfg` WITHOUT validating shapes. Always succeeds
    // if the file parses; `Supported()` is the question to ask afterwards. This
    // is what lets an inspector tool report *why* a model is rejected instead of
    // throwing.
    void Resolve(const ModelConfig& cfg, const safetensors::WeightStreamer& stream);

    // Checks every resolved tensor against the geometry `cfg` implies, and
    // records any structural incompatibility. Call after Resolve.
    void Check(const ModelConfig& cfg);

    [[nodiscard]] bool Supported() const { return issues_.empty(); }
    [[nodiscard]] const std::vector<SupportIssue>& Issues() const { return issues_; }

    // Human-readable summary, one line per issue, for error messages and the
    // inspector tool.
    [[nodiscard]] std::string Explain() const;

    // The naming convention that matched, for diagnostics.
    [[nodiscard]] const std::string& LayerPrefix() const { return layerPrefix_; }
    [[nodiscard]] const std::string& QPrefix()   const { return qPrefix_; }
    [[nodiscard]] const std::string& KPrefix()   const { return kPrefix_; }
    [[nodiscard]] const std::string& VPrefix()   const { return vPrefix_; }
    [[nodiscard]] const std::string& OPrefix()   const { return oPrefix_; }
    [[nodiscard]] const std::string& GatePrefix() const { return gatePrefix_; }
    [[nodiscard]] const std::string& UpPrefix()  const { return upPrefix_; }
    [[nodiscard]] const std::string& DownPrefix() const { return downPrefix_; }

    // Bookends. `lmHead` is empty for tied-embedding models.
    [[nodiscard]] const TensorRef& Embed()   const { return embed_; }
    [[nodiscard]] const TensorRef& FinalNorm() const { return finalNorm_; }
    [[nodiscard]] const TensorRef& LmHead()  const { return lmHead_; }
    [[nodiscard]] const std::vector<BlockTensors>& Blocks() const { return blocks_; }

    // Every tensor the graph will read, in a stable order. Weight upload iterates
    // this rather than rebuilding names.
    [[nodiscard]] const std::vector<const TensorRef*>& ReadSet() const { return readSet_; }

    // True when at least one tensor needs CPU-side widening rather than a raw
    // byte copy. Drives which upload path runs.
    [[nodiscard]] bool NeedsWidening() const { return needsWidening_; }

private:
    void AddIssue(std::string reason, std::string fix);
    void Require(const TensorRef& t, const char* role, std::initializer_list<int64_t> dims);
    void RebuildReadSet();

    TensorRef embed_;
    TensorRef finalNorm_;
    TensorRef lmHead_;
    std::vector<BlockTensors> blocks_;

    std::string layerPrefix_, qPrefix_, kPrefix_, vPrefix_, oPrefix_;
    std::string gatePrefix_, upPrefix_, downPrefix_;
    std::string normPrefixIn_, normPrefixPost_;

    std::vector<const TensorRef*> readSet_;
    std::vector<SupportIssue>     issues_;
    bool needsWidening_ = false;
};

} // namespace opure::graph
