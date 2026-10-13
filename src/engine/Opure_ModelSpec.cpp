// ============================================================================
//  Opure_ModelSpec.cpp
//  Opure-VoxelEngine -- resolve a .safetensors file into graph roles.
// ============================================================================
#include "engine/Opure_ModelSpec.h"

#include "engine/Opure_Graph.h"

#include <algorithm>
#include <sstream>

namespace opure::graph
{

// ---------------------------------------------------------------------------
uint64_t TensorRef::Elements() const
{
    uint64_t n = 1;
    for (int64_t d : shape) { n *= static_cast<uint64_t>(d < 0 ? 0 : d); }
    return n;
}

bool TensorRef::Matches(std::initializer_list<int64_t> dims) const
{
    if (shape.size() != dims.size()) { return false; }
    size_t i = 0;
    for (int64_t d : dims)
    {
        if (shape[i++] != d) { return false; }
    }
    return true;
}

// ---------------------------------------------------------------------------
void ModelSpec::AddIssue(std::string reason, std::string fix)
{
    for (const SupportIssue& s : issues_)
    {
        if (s.reason == reason) { return; }   // one report per distinct reason
    }
    issues_.push_back({ std::move(reason), std::move(fix) });
}

std::string ModelSpec::Explain() const
{
    if (issues_.empty()) { return "supported"; }

    std::ostringstream os;
    for (size_t i = 0; i < issues_.size(); ++i)
    {
        os << (i ? "\n  - " : "  - ") << issues_[i].reason;
        if (!issues_[i].fix.empty()) { os << "\n      fix: " << issues_[i].fix; }
    }
    return os.str();
}

void ModelSpec::Require(const TensorRef& t, const char* role,
                        std::initializer_list<int64_t> dims)
{
    if (!t.Present())
    {
        AddIssue(std::string("missing tensor for role '") + role + "'",
                 "check the naming convention; the graph found no match for it");
        return;
    }

    if (!t.Matches(dims))
    {
        std::ostringstream want, got;
        for (size_t i = 0; i < dims.size(); ++i)
        {
            if (i) { want << ", "; }
            want << *(&*dims.begin() + static_cast<ptrdiff_t>(i));
        }
        for (size_t i = 0; i < t.shape.size(); ++i)
        {
            if (i) { got << ", "; }
            got << t.shape[i];
        }
        AddIssue(std::string("tensor '") + t.name + "' for role '" + role +
                 "' has shape [" + got.str() + "] but the config implies [" +
                 want.str() + "]",
                 "the weights and config.json disagree; this model is not self-consistent");
    }
}

// ---------------------------------------------------------------------------
namespace
{
    TensorRef Make(const safetensors::WeightStreamer& s, std::string_view name)
    {
        const safetensors::Tensor* t = s.TryFind(name);
        if (t == nullptr) { return TensorRef{}; }
        TensorRef r;
        r.name  = t->name;
        r.dtype = t->dtype;
        r.shape = t->shape;
        return r;
    }

    // Probes a list of candidate names and returns the first that exists.
    // The first entry is the canonical Hugging Face llama name; the rest cover
    // the other conventions seen in the wild.
    TensorRef Probe(const safetensors::WeightStreamer& s,
                    const std::vector<std::string>& candidates)
    {
        for (const std::string& c : candidates)
        {
            TensorRef r = Make(s, c);
            if (r.Present()) { return r; }
        }
        return TensorRef{};
    }

    // Chooses the per-block prefix by testing layer 0's query projection. Every
    // convention shares the same query-projection role within a block, so one
    // probe identifies all of them.
    //
    // The returned prefix deliberately STOPS BEFORE the "0." -- callers build a
    // concrete name as prefix + layerIndex + "." + role, so leaving the 0 in
    // would produce "model.layers.0.0.self_attn...".
    std::string DetectLayerPrefix(const safetensors::WeightStreamer& s)
    {
        static const char* kCandidates[] = {
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.attention.wq.weight",
            "transformer.h.0.attn.c_attn.wq.weight",
            "model.decoder.layers.0.self_attn.q_proj.weight",
            "gpt_neox.layers.0.attention.query_key_value.weight",
        };
        for (const char* c : kCandidates)
        {
            if (s.TryFind(c) != nullptr)
            {
                const std::string full(c);
                return full.substr(0, full.find("0."));
            }
        }
        return "model.layers.";
    }
}

// ---------------------------------------------------------------------------
void ModelSpec::Resolve(const ModelConfig& cfg,
                        const safetensors::WeightStreamer& stream)
{
    issues_.clear();
    readSet_.clear();
    needsWidening_ = false;

    // ---- bookends ---------------------------------------------------------
    embed_ = Probe(stream, {
        "model.embed_tokens.weight",
        "transformer.wte.weight",
        "model.decoder.embed_tokens.weight",
        "gpt_neox.embed_in.weight",
    });

    finalNorm_ = Probe(stream, {
        "model.norm.weight",
        "transformer.ln_f.weight",
        "model.final_layernorm.weight",
        "gpt_neox.final_layer_norm.weight",
    });

    // Untied models carry a separate LM head. Tied ones do not, and the graph
    // then uses the embedding table directly.
    lmHead_ = Probe(stream, {
        "lm_head.weight",
        "model.lm_head.weight",
        "transformer.lm_head.weight",
        "output.weight",
    });

    // ---- per-block prefixes ----------------------------------------------
    layerPrefix_ = DetectLayerPrefix(stream);

    auto pick = [&](const char* canon, const char* alt) -> std::string
    {
        // canon/alt are relative to layerPrefix_ with "%s" replaced by "0.".
        const std::string probe = layerPrefix_ + canon;
        if (stream.TryFind(probe) != nullptr) { return canon; }
        const std::string probeAlt = layerPrefix_ + alt;
        if (stream.TryFind(probeAlt) != nullptr) { return alt; }
        return canon;   // absent either way; Require() will report it
    };

    qPrefix_   = pick("self_attn.q_proj.weight", "attention.wq.weight");
    kPrefix_   = pick("self_attn.k_proj.weight", "attention.wk.weight");
    vPrefix_   = pick("self_attn.v_proj.weight", "attention.wv.weight");
    oPrefix_   = pick("self_attn.o_proj.weight", "attention.wo.weight");
    gatePrefix_ = pick("mlp.gate_proj.weight", "mlp.gate_proj.weight");
    upPrefix_   = pick("mlp.up_proj.weight", "mlp.up_proj.weight");
    downPrefix_ = pick("mlp.down_proj.weight", "mlp.down_proj.weight");

    // ---- blocks -----------------------------------------------------------
    blocks_.clear();
    blocks_.resize(cfg.numLayers);
    for (uint32_t l = 0; l < cfg.numLayers; ++l)
    {
        const std::string p = layerPrefix_ + std::to_string(l) + ".";
        BlockTensors& b = blocks_[l];

        b.inputNorm   = Probe(stream, { p + "input_layernorm.weight",
                                        p + "attention_norm.weight",
                                        p + "ln_1.weight",
                                        p + "attention.layer_norm.weight" });
        b.postAttNorm = Probe(stream, { p + "post_attention_layernorm.weight",
                                        p + "ffn_norm.weight",
                                        p + "ln_2.weight" });
        b.wq   = Probe(stream, { p + qPrefix_ });
        b.wk   = Probe(stream, { p + kPrefix_ });
        b.wv   = Probe(stream, { p + vPrefix_ });
        b.wo   = Probe(stream, { p + oPrefix_ });
        b.gate = Probe(stream, { p + gatePrefix_ });
        b.up   = Probe(stream, { p + upPrefix_ });
        b.down = Probe(stream, { p + downPrefix_ });
    }

    RebuildReadSet();
}

// ---------------------------------------------------------------------------
void ModelSpec::Check(const ModelConfig& cfg)
{
    const uint32_t h     = cfg.hiddenSize;
    const uint32_t ff    = cfg.intermediateSize;
    const uint32_t hd    = cfg.HeadDim();
    const uint32_t kvh   = static_cast<uint32_t>(cfg.numKvHeads) * hd;

    // ---- structural limits the compiled kernels impose ---------------------
    if (hd == 0 || hd > 256 || (hd % 32u) != 0)
    {
        AddIssue("head_dim " + std::to_string(hd) + " has no compiled attention kernel",
                 "entry points are compiled for every head_dim that is a multiple of "
                 "32 up to 256 (32/64/96/128/160/192/224/256); add the new size to "
                 "the shader manifest, to shaders::Id, and to AttentionFor()");
    }
    if (cfg.numHeads == 0 || cfg.numHeads % cfg.numKvHeads != 0)
    {
        AddIssue("num_heads " + std::to_string(cfg.numHeads) +
                 " is not a multiple of num_kv_heads " + std::to_string(cfg.numKvHeads),
                 "grouped-query attention needs an integer number of query heads "
                 "per KV head");
    }
    if (hd * cfg.numHeads != h)
    {
        AddIssue("num_heads * head_dim (" +
                 std::to_string(cfg.numHeads * hd) + ") != hidden_size (" +
                 std::to_string(h) + ")",
                 "config.json is internally inconsistent");
    }

    // ---- non-SwiGLU MLPs are a different block, not a different weight -----
    if (cfg.hiddenAct != "silu")
    {
        AddIssue("activation '" + cfg.hiddenAct + "' is not implemented (only silu/SwiGLU)",
                 "the block computes silu(gate) * up; a gelu or relu MLP needs its "
                 "own kernel and a different tensor set");
    }
    if (cfg.attentionBias || cfg.mlpBias)
    {
        AddIssue("this config declares projection biases, which the graph does not read",
                 "adding bias support means extending every GEMV and the residual add");
    }
    if (cfg.ropeScalingType != "none")
    {
        AddIssue("RoPE scaling type '" + cfg.ropeScalingType + "' is not implemented",
                 "llama3-style frequency scaling changes the RoPE kernel's theta "
                 "schedule, not just its base");
    }

    // ---- bookends ---------------------------------------------------------
    Require(embed_, "embed_tokens", { static_cast<int64_t>(cfg.vocabSize),
                                     static_cast<int64_t>(h) });
    Require(finalNorm_, "norm", { static_cast<int64_t>(h) });
    if (!cfg.tieWordEmbeddings)
    {
        Require(lmHead_, "lm_head", { static_cast<int64_t>(cfg.vocabSize),
                                      static_cast<int64_t>(h) });
    }
    else if (lmHead_.Present())
    {
        // A tied model that ALSO ships lm_head is fine; the graph ignores it.
        // Nothing to report.
    }

    // ---- per block --------------------------------------------------------
    for (size_t l = 0; l < blocks_.size(); ++l)
    {
        const BlockTensors& b = blocks_[l];
        const std::string  at = "layer " + std::to_string(l) + ": ";

        auto req = [&](const TensorRef& t, const char* r,
                       std::initializer_list<int64_t> dims)
        {
            Require(t, (at + r).c_str(), dims);
        };
        req(b.inputNorm,   "input_layernorm",   { static_cast<int64_t>(h) });
        req(b.postAttNorm, "post_attn_layernorm", { static_cast<int64_t>(h) });
        req(b.wq, "q_proj", { static_cast<int64_t>(h),     static_cast<int64_t>(h) });
        req(b.wk, "k_proj", { static_cast<int64_t>(kvh),   static_cast<int64_t>(h) });
        req(b.wv, "v_proj", { static_cast<int64_t>(kvh),   static_cast<int64_t>(h) });
        req(b.wo, "o_proj", { static_cast<int64_t>(h),     static_cast<int64_t>(h) });
        req(b.gate, "gate_proj", { static_cast<int64_t>(ff), static_cast<int64_t>(h) });
        req(b.up,   "up_proj",   { static_cast<int64_t>(ff), static_cast<int64_t>(h) });
        req(b.down, "down_proj", { static_cast<int64_t>(h),  static_cast<int64_t>(ff) });
    }
}

// ---------------------------------------------------------------------------
void ModelSpec::RebuildReadSet()
{
    readSet_.clear();
    readSet_.reserve(2 + blocks_.size() * 9);

    auto add = [this](const TensorRef& t)
    {
        if (!t.Present()) { return; }
        if (std::find(readSet_.begin(), readSet_.end(), &t) != readSet_.end()) { return; }
        readSet_.push_back(&t);
        if (t.dtype != safetensors::Dtype::F32) { needsWidening_ = true; }
    };

    add(embed_);
    add(finalNorm_);
    if (lmHead_.Present()) { add(lmHead_); }

    for (const BlockTensors& b : blocks_)
    {
        add(b.inputNorm); add(b.postAttNorm);
        add(b.wq); add(b.wk); add(b.wv); add(b.wo);
        add(b.gate); add(b.up); add(b.down);
    }
}

} // namespace opure::graph
