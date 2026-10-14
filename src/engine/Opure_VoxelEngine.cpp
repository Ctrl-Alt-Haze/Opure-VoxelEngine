// ============================================================================
//  Opure_VoxelEngine.cpp
//  Opure-VoxelEngine -- transformer forward pass and generation loop.
// ============================================================================
#include "engine/Opure_VoxelEngine.h"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace opure::engine {

namespace {

// ---------------------------------------------------------------------------
//  PER-KERNEL THREAD COUNTS
// ---------------------------------------------------------------------------
//  Each shader declares its own [numthreads]. Dispatching with the wrong divisor
//  silently under-covers the work: Embedding uses 64 threads, so dispatching
//  ceil(576 / 256) = 3 groups writes only 192 of 576 elements and the rest stay
//  zero. That produced an all-zero logit vector even though the weights were
//  uploaded perfectly. Every dispatch must divide by ITS OWN kernel's width.
constexpr uint32_t kElemThreads = 256;   // RMSNorm, SwiGLU, Add, AddRes
constexpr uint32_t kEmbedThreads = 64;    // Embedding, RoPE
constexpr uint32_t kPlasticityThreads = 256;   // SignSGD, WeightCommit

inline uint32_t GroupsFor(uint32_t n, uint32_t threads)
{
    return (n + threads - 1) / threads;
}

// The three activation banks. Index them exactly as the shaders do.
constexpr uint32_t kA = 0, kB = 1, kC = 2;

// The bank that is neither `a` nor `b`. Used to pick a destination that does
// not alias either source.
constexpr uint32_t Other(uint32_t a, uint32_t b) {
 return 3u - a - b;
}

// ---------------------------------------------------------------------------
//  BANK SCHEDULE
// ---------------------------------------------------------------------------
//  One layer's banks, as a function of the residual bank it starts with.
//
//  TWO constraints, and the second one is the one that is easy to miss:
//
//   1. SRV/UAV aliasing. D3D12 forbids a dispatch that binds one resource as both
//      SRV and UAV, so every pass must WRITE a bank it does not READ.
//
//   2. LIVENESS. The residual stream x must survive the whole layer: it is read
//      by the opening RMSNorm and by the attention residual add. A schedule that
//      satisfies (1) but writes x's bank mid-layer destroys x and produces an
//      all-zero output -- which is exactly what an earlier version did, when
//      `gate_proj` wrote the bank the residual was living in.
//
//  Hand-reasoning this does not work: an early attempt computed Other(rb, kC)
//  with rb == kC and produced the invalid bank index -1. The schedule below was
//  therefore SOLVED by enumerating assignments under both constraints, and is
//  asserted at RecordForward so a future edit cannot reintroduce the bug.
struct LayerBanks {
    uint32_t normed;   // RMSNorm(x)
    uint32_t qkv;      // q, k, v GEMVs
    uint32_t rope;     // rotated q,k
    uint32_t attn;     // attention output
    uint32_t proj;     // o_proj output
    uint32_t rb1;      // residual after the attention add
    uint32_t n2;       // second RMSNorm
    uint32_t gu;       // gate and up projections
    uint32_t hh;       // SwiGLU output
    uint32_t down;     // down_proj output
    uint32_t rb2;      // residual after the MLP add
};

constexpr LayerBanks MakeLayerBanks(uint32_t rb) {
    switch (rb) {
    //    normed qkv rope attn proj  rb1  n2  gu  hh  down rb2
    case kA: return { kB, kC, kB, kC, kB, kC, kA, kB, kA, kB, kA };
    case kB: return { kA, kC, kA, kC, kA, kC, kA, kB, kA, kB, kA };
    default:  return { kA, kB, kA, kB, kA, kB, kA, kC, kA, kC, kA };  // rb == kC
    }
}

} // namespace

// ---------------------------------------------------------------------------
VoxelEngine::~VoxelEngine() {
    Shutdown();
}

void VoxelEngine::Shutdown() {
    weights_.Destroy();
    actA_.Destroy();
    actB_.Destroy();
    res_.Destroy();
    kv_.Destroy();
    graph_.Release();
    ready_ = false;
}

// ---------------------------------------------------------------------------
void VoxelEngine::Initialize(const std::wstring& modelDir, const EngineOptions& opts,
                             dx12::Device* dev, shaders::Library* lib) {
    OPURE_CHECK(dev != nullptr && lib != nullptr, "engine needs a device and a shader library");
    dev_ = dev;
    lib_ = lib;
    opts_ = opts;

    graph_.LoadConfigOnly(modelDir, 15ull * 1024 * 1024 * 1024);
    const graph::VramPlan& plan = graph_.Plan();
    const graph::ModelConfig& cfg = graph_.Config();

    OPURE_CHECK(plan.fits,
                "model needs " + std::to_string(plan.totalBytes >> 20) +
                " MB but only " + std::to_string(plan.availableBytes >> 20) +
                " MB of VRAM is available");

    const uint32_t h  = cfg.hiddenSize;
    const uint32_t ff = cfg.intermediateSize;
    const uint32_t hd = cfg.HeadDim();
    const uint32_t kvHidden = cfg.numKvHeads * hd;

    // Scratch layout inside ONE activation bank, in words. The weights are all
    // read from the weight buffer, so a pass never needs its input and output in
    // the same bank: the source bank and destination bank always differ.
    scratchQ_ = 0;                     // q            [hidden]
    scratchK_ = scratchQ_ + h;         // k            [kvHidden]
    scratchV_ = scratchK_ + kvHidden;  // v            [kvHidden]
    scratchA_ = scratchV_ + kvHidden;  // attn out     [hidden]
    scratchG_ = scratchA_ + h;         // gate         [intermediate]
    scratchU_ = scratchG_ + ff;        // up           [intermediate]
    scratchH_ = scratchU_ + ff;        // swiglu out   [intermediate]
    scratchL_ = scratchH_ + ff;        // logits       [vocab]

    const uint64_t needWords = static_cast<uint64_t>(scratchL_) + cfg.vocabSize;
    OPURE_CHECK(needWords <= plan.actWords,
                "activation bank holds " + std::to_string(plan.actWords) +
                " words but the graph needs " + std::to_string(needWords));

    // ArgMax writes its two-word result into the KV buffer's tail, which is
    // reserved beyond the cache proper so it cannot corrupt a real KV entry.
    const uint32_t kvCacheWords = static_cast<uint32_t>(cfg.KvCacheWords());
    argmaxSlot_ = kvCacheWords;
    OPURE_ASSERT_MSG(static_cast<uint64_t>(argmaxSlot_) + 2 <= plan.kvWords,
                     "KV allocation has no room for the argmax result");

    kvLayerWords_ = static_cast<uint32_t>(cfg.maxPositions * cfg.numKvHeads * hd) * 2;

    resBank_ = kC;

    weights_.Create(*dev_, 0, plan.weightBytes, true);
    actA_   .Create(*dev_, 1, plan.actBytes, true);
    actB_   .Create(*dev_, 2, plan.actBytes, true);
    kv_     .Create(*dev_, 3, plan.kvBytes, true);
    res_    .Create(*dev_, 4, plan.actBytes, true);
    dev_->SetPersistentBuffers(&weights_, &actA_, &actB_, &kv_, &res_);

    graph_.UploadWeights(*dev_, weights_);
    tokenizer_.LoadFromFile(modelDir + L"\\tokenizer.json");

    seqLen_ = 0;
    ready_  = true;
}

// ---------------------------------------------------------------------------
//  One token, one command list.
//
//  BANK DISCIPLINE (the part that is easy to get silently wrong)
//  -------------------------------------------------------------
//  `rb` is the bank holding the residual stream x. Every pass reads some source
//  bank and writes a DIFFERENT destination bank, because D3D12 forbids a
//  dispatch that binds one resource as both SRV and UAV.
//
//  Within a layer:
//    normed = RMSNorm(x)                      rb -> nk
//    q,k,v = normed @ W                      nk -> nk2   (three GEMVs, one bank)
//    rope(q,k,v)                             nk2 -> qk   (also appends K,V to the KV cache)
//    attn = Attention(q)                     qk  -> ao
//    proj = attn @ Wo                        ao  -> pb
//    x    = x + proj                         (rb, pb) -> Other(rb, pb)
//    ... then the MLP with the same shape ...
//
//  The residual destination is Other(rb, pb), so the three banks involved in
//  each add are always distinct. Because rb rotates on the next layer, the
//  "other" bank alternates and no buffer is ever read and written by the same
//  dispatch.
// ---------------------------------------------------------------------------
void VoxelEngine::RecordForward(const uint32_t tokenId, const uint32_t position,
                                uint32_t* logitBank) {
    const graph::ModelConfig& cfg = graph_.Config();
    const graph::VramPlan&    plan = graph_.Plan();
    const uint32_t h  = cfg.hiddenSize;
    const uint32_t ff = cfg.intermediateSize;
    const uint32_t hd = cfg.HeadDim();
    const uint32_t kvHidden = cfg.numKvHeads * hd;
    const uint32_t epsBits  = cfg.rmsNormEpsBits;
    const uint32_t thetaBits = graph::ModelConfig::FloatBits(static_cast<float>(cfg.ropeTheta));

    // Activations are always fp32 in the activation banks -- only the weights
    // are quantized -- so the quantized GEMV's activation scale is the identity
    // 1.0f. It is still passed explicitly rather than left as a literal 0,
    // because AsF(0) is 0.0f and the kernel multiplies by it: a 0 here silently
    // returns an all-zero row. Weight scales likewise travel as float bits, not
    // as truncated integers.
    constexpr uint32_t kQuantActScale = graph::ModelConfig::FloatBits(1.0f);

    dispatches_ = 0;

    dx12::CommandScope cs(*dev_);

    // Every dispatch in the graph goes through GO. Besides collapsing fourteen
    // copies of the same three lines, GO is where the graph's two global rules
    // live: a UAV barrier before every dispatch, and the diagnostic dispatch
    // budget.
    //
    // The barrier is not optional. Consecutive dispatches in one command list
    // hand the SAME persistent buffers to each other: this pass writes an
    // activation bank through its UAV and the next reads it through an SRV.
    // Without an explicit UAV barrier the read is not ordered against the write,
    // so a pass occasionally sees a bank's previous contents. That surfaced as
    // q_proj returning |q| up to 3e4 for a normalised input whose max was 0.40
    // -- a dot product against activation words the graph had never written --
    // and as logits that changed from run to run. See the README bug log.
    auto GO = [&](shaders::Id id, const uint32_t* params, uint32_t count,
                  uint32_t groupsX) -> bool
    {
        cs.Barrier();
        cs.SetPipeline(lib_->Get(id));
        cs.SetParams(params, count);
        cs.Dispatch(groupsX, 1, 1);
        ++dispatches_;
        if (dispatches_ < dispatchLimit_) { return false; }
        curResBank_ = rb_;
        return true;
    };

    // ---- embedding -> residual bank --------------------------------------
    // Skippable so the per-layer diagnostic can run ONE layer against an
    // already-established residual. Re-running it each layer reset the residual
    // to the embedding and made the comparison meaningless.
    if (!skipEmbed_) {
        const uint32_t p[4] = { 0, tokenId, h, resBank_ };
        if (GO(shaders::Id::Embedding, p, 4, GroupsFor(h, kEmbedThreads))) { return; }
    }

    // A traced layer continues from the PREVIOUS layer's residual bank, so the
    // bank schedule stays consistent along the chain; a normal pass always
    // restarts at the embedding's bank. Restarting every traced layer from
    // resBank_ made layers 1..N replay the layer-0 schedule, which is why the
    // per-layer trace disagreed with the full pass.
    uint32_t rb = rb_ = (skipEmbed_ ? rb_ : resBank_);

    // Clamped to the model's layer count: `layerLimit_` is a DIAGNOSTIC cap and
    // defaults to "no limit" (0xFFFFFFFF), which must not become the loop bound
    // on its own -- that ran the loop four billion times and indexed the layer
    // tensor table off the end of the allocation.
    //
    // `layerStart_` is diagnostic too: the per-layer trace runs ONE layer per
    // command list and has to be able to ask for layer 7. With a hardcoded start
    // of 0 it replayed layer 0 six times, which the per-layer comparison then
    // reported as a divergence at layer 1.
    const uint32_t layerEnd = std::min(cfg.numLayers, layerStart_ + layerLimit_);

    for (uint32_t L = layerStart_; L < layerEnd; ++L) {
        const graph::VramPlan::LayerTensors& t = plan.layers[L];
        const uint32_t kvLayerOff = L * kvLayerWords_;
        const LayerBanks LB = MakeLayerBanks(rb);

        // Two invariants, checked at run time rather than trusted:
        //   1. no pass writes a bank it reads            (D3D12 forbids it)
        //   2. no pass writes the live residual's bank   (would destroy x)
        // Both produce plausible-looking garbage rather than an error, so they
        // are asserted here instead of left to a comment.
        const bool aliasing =
            LB.normed != rb && LB.qkv != LB.normed && LB.rope != LB.qkv &&
            LB.attn != LB.rope && LB.proj != LB.attn &&
            LB.rb1 != rb && LB.rb1 != LB.proj &&
            LB.n2 != LB.rb1 && LB.gu != LB.n2 && LB.hh != LB.gu &&
            LB.down != LB.hh && LB.rb2 != LB.rb1 && LB.rb2 != LB.down;

        const bool firstHalfClobbersRb =
            (LB.normed == rb) || (LB.qkv == rb) || (LB.rope == rb) ||
            (LB.attn == rb)  || (LB.proj == rb);

        const bool secondHalfClobbersRb1 =
            (LB.n2 == LB.rb1) || (LB.gu == LB.rb1) ||
            (LB.hh == LB.rb1) || (LB.down == LB.rb1);

        OPURE_ASSERT_MSG(aliasing,
                         "bank schedule binds one resource as both SRV and UAV, "
                         "which D3D12 forbids");
        OPURE_ASSERT_MSG(!firstHalfClobbersRb,
                         "a pass writes the residual bank while the residual is live; "
                         "x would be destroyed before the attention residual add");
        OPURE_ASSERT_MSG(!secondHalfClobbersRb1,
                         "an MLP pass writes the residual bank while the residual "
                         "is live; x would be destroyed before the MLP residual add");

        // ---- attention ----------------------------------------------------
        {
            const uint32_t p[8] = { 0, static_cast<uint32_t>(t.inputNorm), 0,
                                    h, epsBits, 1u, rb, LB.normed };
            if (GO(shaders::Id::RMSNorm, p, 8, 1)) { return; }
        }

        // q, k, v all read LB.normed and write LB.qkv, so they can share a bank.
        //
        // NOTE the argument order: GEMV's P[3] is inDim (the contracted axis) and
        // P[4] is outDim (the rows). Weights are [out, in] row-major, so a
        // projection is (inDim, outDim) = (d_in, d_out) -- k_proj/v_proj are
        // (h, kvHidden) and gate/up are (h, ff), NOT the other way round. Passing
        // them the other way makes the kernel read d_in-1 activation words past
        // the end of the input and write only min(d_in, d_out) outputs, which is
        // why the square-matrix parity test never caught it.
        {
            const uint64_t wqOff = static_cast<uint64_t>(t.wq);
            const bool wqQuant = graph_.IsQuantized(wqOff);
            const uint32_t pGEMV[7] = { 0, static_cast<uint32_t>(t.wq), scratchQ_, h, h, LB.normed, LB.qkv };
            const uint32_t pGEMVQuant[9] = { 0, static_cast<uint32_t>(t.wq), scratchQ_, h, h,
                                             graph::ModelConfig::FloatBits(graph_.GetQuantizedScale(wqOff)),
                                             kQuantActScale, LB.normed, LB.qkv };
            if (wqQuant)
            {
                if (GO(shaders::Id::GEMVQuant, pGEMVQuant, 9, h)) { return; }
            }
            else
            {
                if (GO(shaders::Id::GEMV, pGEMV, 7, h)) { return; }
            }
        }
        {
            const uint64_t wkOff = static_cast<uint64_t>(t.wk);
            const bool wkQuant = graph_.IsQuantized(wkOff);
            const uint32_t pGEMV[7] = { 0, static_cast<uint32_t>(t.wk), scratchK_, h, kvHidden, LB.normed, LB.qkv };
            const uint32_t pGEMVQuant[9] = { 0, static_cast<uint32_t>(t.wk), scratchK_, h, kvHidden,
                                             graph::ModelConfig::FloatBits(graph_.GetQuantizedScale(wkOff)),
                                             kQuantActScale, LB.normed, LB.qkv };
            if (wkQuant)
            {
                if (GO(shaders::Id::GEMVQuant, pGEMVQuant, 9, kvHidden)) { return; }
            }
            else
            {
                if (GO(shaders::Id::GEMV, pGEMV, 7, kvHidden)) { return; }
            }
        }
        {
            const uint64_t wvOff = static_cast<uint64_t>(t.wv);
            const bool wvQuant = graph_.IsQuantized(wvOff);
            const uint32_t pGEMV[7] = { 0, static_cast<uint32_t>(t.wv), scratchV_, h, kvHidden, LB.normed, LB.qkv };
            const uint32_t pGEMVQuant[9] = { 0, static_cast<uint32_t>(t.wv), scratchV_, h, kvHidden,
                                             graph::ModelConfig::FloatBits(graph_.GetQuantizedScale(wvOff)),
                                             kQuantActScale, LB.normed, LB.qkv };
            if (wvQuant)
            {
                if (GO(shaders::Id::GEMVQuant, pGEMVQuant, 9, kvHidden)) { return; }
            }
            else
            {
                if (GO(shaders::Id::GEMV, pGEMV, 7, kvHidden)) { return; }
            }
        }

        // RoPE: rotate q and k, and append K,V to the paged cache.
        {
            const uint32_t p[14] = {
                scratchQ_, scratchQ_,                 // q src, dst
                scratchK_, scratchK_,                 // k src, dst
                scratchV_, kvLayerOff,                // v src, KV base
                cfg.numHeads, cfg.numKvHeads, hd, cfg.maxPositions,
                position, thetaBits,
                LB.qkv, LB.rope
            };
            if (GO(shaders::Id::RoPE, p, 14, cfg.numHeads * (hd / 2))) { return; }
        }

        // Attention: reads the rotated q from LB.rope, writes into LB.attn.
        {
            const shaders::Id id = shaders::Library::AttentionFor(hd);
            OPURE_ASSERT(id != shaders::Id::Count);   // ModelSpec::Check refused this
            const uint32_t p[10] = {
                scratchQ_, kvLayerOff, scratchA_,
                cfg.numKvHeads, hd, cfg.maxPositions, position + 1, cfg.numHeads,
                LB.rope, LB.attn
            };
            if (GO(id, p, 10, cfg.numHeads)) { return; }
        }

        // o_proj: reads the attention output from LB.attn, writes LB.proj.
        {
            const uint64_t woOff = static_cast<uint64_t>(t.wo);
            const bool woQuant = graph_.IsQuantized(woOff);
            const uint32_t pGEMV[7] = { scratchA_, static_cast<uint32_t>(t.wo), 0, h, h, LB.attn, LB.proj };
            const uint32_t pGEMVQuant[9] = { scratchA_, static_cast<uint32_t>(t.wo), 0, h, h,
                                             graph::ModelConfig::FloatBits(graph_.GetQuantizedScale(woOff)),
                                             kQuantActScale, LB.attn, LB.proj };
            if (woQuant)
            {
                if (GO(shaders::Id::GEMVQuant, pGEMVQuant, 9, h)) { return; }
            }
            else
            {
                if (GO(shaders::Id::GEMV, pGEMV, 7, h)) { return; }
            }
        }

        // Residual: x(OLD rb) + proj(LB.proj) -> LB.rb1.
        //
        // The source bank must be captured BEFORE `rb` is advanced to the
        // destination. Passing the new `rb` as the source made the add read the
        // bank it was writing -- an SRV/UAV alias AND a read of uninitialised
        // memory, which showed up as a layer-0 residual error of 5e1 that grew
        // to 1.8e17 by layer 5.
        const uint32_t srcRes = rb;
        rb = LB.rb1;
        {
            const uint32_t p[7] = { 0, 0, 0, h, LB.proj, srcRes, rb };
            if (GO(shaders::Id::AddRes, p, 7, GroupsFor(h, kElemThreads))) { return; }
        }

        // ---- MLP ----------------------------------------------------------
        {
            const uint32_t p[8] = { 0, static_cast<uint32_t>(t.postAttNorm), 0,
                                    h, epsBits, 1u, rb, LB.n2 };
            if (GO(shaders::Id::RMSNorm, p, 8, 1)) { return; }
        }
        {
            const uint64_t gateOff = static_cast<uint64_t>(t.gate);
            const bool gateQuant = graph_.IsQuantized(gateOff);
            const uint32_t pGEMV[7] = { 0, static_cast<uint32_t>(t.gate), scratchG_, h, ff, LB.n2, LB.gu };
            const uint32_t pGEMVQuant[9] = { 0, static_cast<uint32_t>(t.gate), scratchG_, h, ff,
                                             graph::ModelConfig::FloatBits(graph_.GetQuantizedScale(gateOff)),
                                             kQuantActScale, LB.n2, LB.gu };
            if (gateQuant)
            {
                if (GO(shaders::Id::GEMVQuant, pGEMVQuant, 9, ff)) { return; }
            }
            else
            {
                if (GO(shaders::Id::GEMV, pGEMV, 7, ff)) { return; }
            }
        }
        {
            const uint64_t upOff = static_cast<uint64_t>(t.up);
            const bool upQuant = graph_.IsQuantized(upOff);
            const uint32_t pGEMV[7] = { 0, static_cast<uint32_t>(t.up), scratchU_, h, ff, LB.n2, LB.gu };
            const uint32_t pGEMVQuant[9] = { 0, static_cast<uint32_t>(t.up), scratchU_, h, ff,
                                             graph::ModelConfig::FloatBits(graph_.GetQuantizedScale(upOff)),
                                             kQuantActScale, LB.n2, LB.gu };
            if (upQuant)
            {
                if (GO(shaders::Id::GEMVQuant, pGEMVQuant, 9, ff)) { return; }
            }
            else
            {
                if (GO(shaders::Id::GEMV, pGEMV, 7, ff)) { return; }
            }
        }

        // SwiGLU: reads LB.gu, writes LB.hh.
        {
            const uint32_t p[6] = { scratchG_, scratchU_, scratchH_, ff, LB.gu, LB.hh };
            if (GO(shaders::Id::SwiGLU, p, 6, GroupsFor(ff, kElemThreads))) { return; }
        }

        // down_proj: contracts the intermediate size, so inDim = ff and
        // outDim = h -- the opposite of gate/up.
        {
            const uint64_t downOff = static_cast<uint64_t>(t.down);
            const bool downQuant = graph_.IsQuantized(downOff);
            const uint32_t pGEMV[7] = { scratchH_, static_cast<uint32_t>(t.down), scratchA_, ff, h, LB.hh, LB.down };
            const uint32_t pGEMVQuant[9] = { scratchH_, static_cast<uint32_t>(t.down), scratchA_, ff, h,
                                             graph::ModelConfig::FloatBits(graph_.GetQuantizedScale(downOff)),
                                             kQuantActScale, LB.hh, LB.down };
            if (downQuant)
            {
                if (GO(shaders::Id::GEMVQuant, pGEMVQuant, 9, h)) { return; }
            }
            else
            {
                if (GO(shaders::Id::GEMV, pGEMV, 7, h)) { return; }
            }
        }

        // Residual: x(LB.rb1) + down(LB.down) -> LB.rb2. Same capture-before-advance
        // rule as the attention add above.
        {
            const uint32_t p[7] = { scratchA_, 0, 0, h, LB.down, LB.rb1, LB.rb2 };
            if (GO(shaders::Id::AddRes, p, 7, GroupsFor(h, kElemThreads))) { return; }
        }
        rb = rb_ = LB.rb2;

        // Diagnostic mode runs exactly one layer per command list, so the caller
        // can execute and read the residual between layers. curResBank_ MUST be
        // updated here: GO only refreshes it when the dispatch budget trips, so
        // without this the caller reads whichever bank the previous step left
        // behind and every traced layer reports the same stale values.
        curResBank_ = rb;
        if (traceSink_ != nullptr) { return; }
    }

    // ---- final norm + LM head --------------------------------------------
    //
    // Skipped when the caller asked for the embedding only: the head would
    // consume a residual that no layer has produced, writing a normed vector
    // and a full logit row over banks the trace is about to read.
    if (layerLimit_ == 0)
    {
        curResBank_ = rb;
        return;
    }

    // Tied embeddings: the LM head IS the embedding matrix, so each logit is an
    // embedding row dotted with the final hidden state. It only needs two banks:
    // read x from rb, write the normed vector to any other, then the logits to
    // the third. Both must differ from each other and from rb.
    const uint32_t normBank = (rb == kA) ? kB : kA;
    const uint32_t outBank  = Other(rb, normBank);   // the third, distinct bank
    OPURE_ASSERT_MSG(normBank != rb && outBank != rb && outBank != normBank,
                     "final-norm banks alias the residual or each other");

    {
        const uint32_t p[8] = { 0, static_cast<uint32_t>(plan.finalNorm), 0,
                                h, epsBits, 1u, rb, normBank };
        if (GO(shaders::Id::RMSNorm, p, 8, 1)) { return; }
    }
    {
        const uint32_t p[7] = { 0, static_cast<uint32_t>(plan.embedTokens), scratchL_,
                                h, cfg.vocabSize, normBank, outBank };
        if (GO(shaders::Id::GEMV, p, 7, cfg.vocabSize)) { return; }
    }

    curResBank_ = rb;
    if (logitBank != nullptr) { *logitBank = outBank; }
}

// ---------------------------------------------------------------------------
//  Diagnostic path: run the embedding and the first `nLayers` layers, reading
//  the residual stream back after each one.
//
//  This executes and resets the command list after every layer, so it is slow
//  and NOT the decode path. It exists so a divergence between this graph and
//  the CPU reference can be localised to a layer index instead of bisected by
//  hand.
void VoxelEngine::RunLayersTraced(uint32_t tokenId, uint32_t nLayers,
                                  std::vector<std::vector<float>>& trace)
{
    trace.clear();
    OPURE_CHECK(ready_, "engine used before Initialize");

    const graph::ModelConfig& cfg = graph_.Config();
    const uint32_t saveLimit = layerLimit_;
    auto* saveSink = traceSink_;
    traceSink_ = &trace;
    const uint32_t saveStart = layerStart_;

    // The embedding on its own.
    dev_->ResetCommandList();
    layerLimit_ = 0;
    RecordForward(tokenId, 0, nullptr);
    dev_->ExecuteAndWait();

    // One layer per command list so the residual can be read back between them.
    // The embedding is NOT re-run: the residual must carry forward from layer to
    // layer exactly as it does in the full pass.
    layerLimit_ = 1;
    skipEmbed_  = true;
    for (uint32_t L = 0; L < nLayers; ++L)
    {
        dev_->ResetCommandList();
        layerStart_ = L;          // one command list per LAYER, not per call
        RecordForward(tokenId, L, nullptr);
        dev_->ExecuteAndWait();

        std::vector<float> row(cfg.hiddenSize);
        dev_->ReadbackFromBuffer(
            (curResBank_ == kA) ? actA_ : ((curResBank_ == kB) ? actB_ : res_),
            row.data(), static_cast<uint64_t>(cfg.hiddenSize) * 4, 0);
        trace.push_back(std::move(row));
    }

    layerLimit_ = saveLimit;
    traceSink_  = saveSink;
    skipEmbed_  = false;
    layerStart_ = saveStart;
}

// ---------------------------------------------------------------------------
//  Diagnostic path: stop after exactly `budget` dispatches of layer `layer` and
//  return the bank that holds the residual stream at that point.
//
//  Stepping the budget from 1 to 17 and reading the banks after each step walks
//  the graph one pass at a time, so a pass that introduces a NaN or reads an
//  unwritten word is identified by name rather than inferred.
uint32_t VoxelEngine::RunPartial(uint32_t tokenId, uint32_t budget, uint32_t layer)
{
    OPURE_CHECK(ready_, "engine used before Initialize");

    const uint32_t saveLimit  = layerLimit_;
    const uint32_t saveBudget = dispatchLimit_;
    layerLimit_    = layer;
    dispatchLimit_ = budget;
    skipEmbed_     = false;

    dev_->ResetCommandList();
    RecordForward(tokenId, 0, nullptr);
    dev_->ExecuteAndWait();

    const uint32_t bank = curResBank_;

    layerLimit_    = saveLimit;
    dispatchLimit_ = saveBudget;
    return bank;
}

// ---------------------------------------------------------------------------
//  Fills every activation bank with `fill`.
//
//  DIAGNOSTIC ONLY. D3D12 does not zero newly created resources, so any word
//  the graph reads without first writing yields whatever the driver left there
//  -- often a NaN bit pattern, which shows up as run-to-run non-determinism.
//  Poisoning the banks turns "the output changed between runs" into "the graph
//  reads a word it never wrote", naming the fault exactly.
void VoxelEngine::PoisonBanks(float fill)
{
    const graph::VramPlan& plan = graph_.Plan();
    const uint32_t n = static_cast<uint32_t>(plan.actWords);
    uint32_t bits;
    std::memcpy(&bits, &fill, 4);
    std::vector<uint32_t> pattern(n, bits);
    for (dx12::GpuBuffer* b : { &actA_, &actB_, &res_ })
    {
        dev_->UploadToBuffer(*b, pattern.data(), pattern.size() * 4, 0);
    }
}

// ---------------------------------------------------------------------------
//  Self-monitoring implementation
// ---------------------------------------------------------------------------
VoxelEngine::MetricsSnapshot VoxelEngine::SnapshotMetrics()
{
    MetricsSnapshot snap;
    snap.timestampMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    const auto& plan = graph_.Plan();
    const auto& cfg = graph_.Config();

    snap.kvOccupancy = cfg.maxPositions > 0
        ? static_cast<float>(seqLen_) / cfg.maxPositions
        : 0.0f;

    snap.logitEntropy = lastLogitEntropy_;
    snap.logitNorm = lastLogitNorm_;
    snap.maxAttnScore = lastMaxAttnScore_;

    snap.tokensPerSec = lastTokenTimeMs_ > 0
        ? 1000.0f / lastTokenTimeMs_
        : 0.0f;

    snap.memoryPressure = plan.availableBytes > 0
        ? static_cast<float>(plan.totalBytes) / plan.availableBytes
        : 0.0f;

    snap.speculativeAcceptance = -1.0f;  // not implemented yet
    snap.layerTimesMs.clear();  // would need per-layer timing hooks

    return snap;
}

// ---------------------------------------------------------------------------
//  Prefill
// ---------------------------------------------------------------------------
//  Records every prompt token. The tokens are split into chunks so that no
//  single command list needs more parameter slots than the device has; without
//  this a long prompt fails with a bare "parameter slot overflow" deep inside a
//  dispatch loop, which says nothing about the cause.
//
//  The chunk boundary is safe because each token's pass reads the KV cache for
//  all PRIOR positions, and those entries were written by earlier chunks.
void VoxelEngine::Prefill(const std::vector<uint32_t>& tokens, uint32_t* logitBank)
{
    OPURE_CHECK(!tokens.empty(), "Prefill called with no tokens");
    OPURE_CHECK(tokens.size() <= graph_.Config().maxPositions,
                "prompt exceeds max_position_embeddings");

    // Dispatches per token: embedding + 30 layers x 14 + final norm + LM head.
    uint32_t perToken = 3;
    for (uint32_t L = 0; L < graph_.Config().numLayers; ++L) { perToken += 14; }

    // Leave headroom so the trailing ArgMax dispatch still fits.
    const uint32_t chunk = std::max<uint32_t>(1u, (dx12::kParamSlots / 2) / perToken);

    uint32_t lb = resBank_;
    for (size_t base = 0; base < tokens.size(); base += chunk)
    {
        dev_->ResetCommandList();
        const size_t end = std::min(tokens.size(), base + chunk);
        for (size_t i = base; i < end; ++i)
        {
            RecordForward(tokens[i], static_cast<uint32_t>(i), &lb);
        }
        dev_->ExecuteAndWait();
    }

    seqLen_   = static_cast<uint32_t>(tokens.size());
    *logitBank = lb;
}

// ---------------------------------------------------------------------------
//  One decode step: append `tokenId` at `position` and reduce to the argmax.
//  Only the new token is computed; earlier positions come from the KV cache.
// ---------------------------------------------------------------------------
//  KV cache sliding-window eviction.
//  When the sequence reaches maxPositions, we evict the oldest token by
//  shifting the K/V cache left by one position. This is a memmove on the GPU
//  via a simple copy dispatch (one thread per element).
// ---------------------------------------------------------------------------
void VoxelEngine::EvictOldestKV()
{
    const auto& cfg = graph_.Config();
    const uint32_t maxPos = cfg.maxPositions;
    const uint32_t kvHidden = cfg.numKvHeads * cfg.HeadDim();
    const uint32_t kvLayerWords = maxPos * kvHidden;
    const uint32_t elementsPerLayer = 2 * kvLayerWords;  // K + V per layer
    for (uint32_t L = 0; L < cfg.numLayers; ++L)
    {
        const uint64_t kvLayerOff = L * cfg.maxPositions * cfg.numKvHeads * cfg.HeadDim();
        const uint32_t kBase = static_cast<uint32_t>(kvLayerOff);
        const uint32_t vBase = static_cast<uint32_t>(kvLayerOff + maxPos * cfg.numKvHeads * cfg.HeadDim());

        // Shift K cache: [1..maxPos-1] -> [0..maxPos-2]
        {
            dev_->ResetCommandList();
            {
                dx12::CommandScope cs(*dev_);
                const uint32_t p[4] = { kBase + kvHidden, kBase, elementsPerLayer / 2, 0 };  // src, dst, count, bank
                cs.Barrier();
                cs.SetPipeline(lib_->Get(shaders::Id::Memmove));
                cs.SetParams(p, 4);
                cs.Dispatch((elementsPerLayer / 2 + 255) / 256, 1, 1);
            }
            dev_->ExecuteAndWait();
        }

        // Shift V cache: [1..maxPos-1] -> [0..maxPos-2]
        {
            dev_->ResetCommandList();
            {
                dx12::CommandScope cs(*dev_);
                
                const uint32_t p[4] = { vBase + kvHidden, vBase, elementsPerLayer / 2, 0 };
                cs.Barrier();
                cs.SetPipeline(lib_->Get(shaders::Id::Memmove));
                cs.SetParams(p, 4);
                cs.Dispatch((elementsPerLayer / 2 + 255) / 256, 1, 1);
            }
            dev_->ExecuteAndWait();
        }
    }
}

uint32_t VoxelEngine::DecodeStep(uint32_t tokenId, uint32_t position)
{
    OPURE_CHECK(position < graph_.Config().maxPositions,
                "decode position exceeds max_position_embeddings");

    // If we're at capacity, evict oldest before writing new token.
    if (seqLen_ >= graph_.Config().maxPositions)
    {
        EvictOldestKV();
        // After eviction, logical position is maxPositions - 1.
    }

    dev_->ResetCommandList();

    uint32_t lb = resBank_;
    RecordForward(tokenId, position, &lb);

    // ArgMax reduces on the GPU so the CPU reads 8 bytes instead of 196 KB.
    {
        dx12::CommandScope cs(*dev_);
        const uint32_t p[4] = { scratchL_, graph_.Config().vocabSize, argmaxSlot_, lb };
        cs.Barrier();

        cs.SetPipeline(lib_->Get(shaders::Id::ArgMax));
        cs.SetParams(p, 4);
        cs.Dispatch(1, 1, 1);
    }
    dev_->ExecuteAndWait();
    seqLen_ = position + 1;

    uint32_t result[2] = { 0, 0 };
    dev_->ReadbackFromBuffer(kv_, result, sizeof(result),
                             static_cast<uint64_t>(argmaxSlot_) * 4);
    return result[1];
}

// ---------------------------------------------------------------------------
void VoxelEngine::ForwardLogits(const std::vector<uint32_t>& tokens,
                                std::vector<float>& logitsOut)
{
    OPURE_CHECK(ready_, "engine used before Initialize");
    OPURE_CHECK(!tokens.empty(), "ForwardLogits called with no tokens");

    const graph::ModelConfig& cfg = graph_.Config();

    uint32_t lb = 0;
    Prefill(tokens, &lb);

    // Read the logits back through the bank they were written to.
    const dx12::GpuBuffer& bank = (lb == kA) ? actA_ : ((lb == kB) ? actB_ : res_);

    logitsOut.assign(cfg.vocabSize, 0.0f);
    dev_->ReadbackFromBuffer(bank, logitsOut.data(),
                             static_cast<uint64_t>(cfg.vocabSize) * 4,
                             static_cast<uint64_t>(scratchL_) * 4);
}

uint32_t VoxelEngine::ForwardArgMax(const std::vector<uint32_t>& tokens)
{
    OPURE_CHECK(ready_, "engine used before Initialize");
    OPURE_CHECK(!tokens.empty(), "ForwardArgMax called with no tokens");

    uint32_t lb = 0;
    Prefill(tokens, &lb);

    // Identical to ArgMaxFrom, which also reopens the command list that Prefill
    // closed. Inlining that dispatch here recorded into a closed list and the
    // argmax silently never ran.
    return ArgMaxFrom(lb);
}

// ---------------------------------------------------------------------------
GenerateResult VoxelEngine::Generate(const std::vector<uint32_t>& tokens,
                                     const EngineOptions& opts)
{
    OPURE_CHECK(ready_, "engine used before Initialize");

    GenerateResult r;
    r.promptTokens = tokens;

    if (tokens.empty()) { return r; }

    const uint32_t stopId = tokenizer_.SpecialId("<|endoftext|>");

    // Prefill the prompt once, then decode ONE token per step.
    //
    // Re-running the whole sequence per generated token would be quadratic and
    // is also what previously exhausted the parameter slots: three tokens of
    // output was enough to overflow them. The KV cache exists precisely so a
    // decode step only has to compute the new token.
    uint32_t lb = 0;
    Prefill(tokens, &lb);

    // The token predicted from the last prompt position becomes the input to
    // the first decode step.
    uint32_t next = ArgMaxFrom(lb);

    const uint32_t budget = std::min(opts.maxNewTokens, opts_.maxNewTokens);
    const uint32_t maxPos  = graph_.Config().maxPositions;

    for (uint32_t i = 0; i < budget; ++i)
    {
        r.generated.push_back(next);
        if (stopId != UINT32_MAX && next == stopId)
        {
            r.stopToken = stopId;
            r.hitStop   = true;
            break;
        }
        if (seqLen_ + 1 >= maxPos) { break; }
        uint64_t stepStart = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        next = DecodeStep(next, seqLen_);
        uint64_t stepEnd = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        lastTokenTimeMs_ = static_cast<uint64_t>((stepEnd - stepStart) / 1000.0f);
    }
    return r;
}

// ---------------------------------------------------------------------------
//  Reduces the logits already sitting in `logitBank` to their argmax.
uint32_t VoxelEngine::ArgMaxFrom(uint32_t logitBank)
{
    // Prefill (and any earlier pass) has already submitted and CLOSED the
    // command list. Recording into a closed list fails every subsequent call,
    // and SetPipeline / SetParams / Dispatch do not surface the error, so the
    // argmax result was simply never written and the readback returned a stale
    // zero -- generation always emitted <|endoftext|>. A fresh list is required
    // before any dispatch that follows an ExecuteAndWait.
    dev_->ResetCommandList();

    {
        dx12::CommandScope cs(*dev_);
        const uint32_t p[4] = { scratchL_, graph_.Config().vocabSize,
                                argmaxSlot_, logitBank };
        cs.Barrier();

        cs.SetPipeline(lib_->Get(shaders::Id::ArgMax));
        cs.SetParams(p, 4);
        cs.Dispatch(1, 1, 1);
    }
    dev_->ExecuteAndWait();

    uint32_t result[2] = { 0, 0 };
    dev_->ReadbackFromBuffer(kv_, result, sizeof(result),
                             static_cast<uint64_t>(argmaxSlot_) * 4);
    return result[1];
}

// ---------------------------------------------------------------------------
void VoxelEngine::ReadWeightsAt(uint64_t wordOffset, uint32_t* dst, size_t count)
{
    dev_->ReadbackFromBuffer(weights_, dst, count * 4, wordOffset * 4);
}

// ---------------------------------------------------------------------------
void VoxelEngine::ReadActBank(uint32_t bank, uint64_t wordOffset, uint32_t* dst, size_t count)
{
    const dx12::GpuBuffer& b = (bank == kA) ? actA_ : ((bank == kB) ? actB_ : res_);
    dev_->ReadbackFromBuffer(b, dst, count * 4, wordOffset * 4);
}

// ---------------------------------------------------------------------------
void VoxelEngine::ReadKv(uint64_t wordOffset, uint32_t* dst, size_t count)
{
    dev_->ReadbackFromBuffer(kv_, dst, count * 4, wordOffset * 4);
}

// ---------------------------------------------------------------------------
//  Idle-time plasticity: one SignSGD step over a target projection.
//
//  Two dispatches, because a single pass would read O_W and write O_WU and so
//  bind the weight buffer as both SRV and UAV -- undefined in D3D12, and the one
//  rule this engine observes everywhere else. The first computes the updated
//  matrix into the staging tail of an activation bank; the second copies it back
//  over the tensor. The staging range is scratch the inference graph never reads,
//  so nothing is retained between calls and there is no optimiser state.
void VoxelEngine::RunPlasticity(const PlasticityStep& step)
{
    OPURE_CHECK(ready_, "engine used before Initialize");
    OPURE_CHECK(step.rows > 0 && step.cols > 0,
                "plasticity step has a zero dimension");

    const graph::VramPlan& plan = graph_.Plan();
    const uint64_t matrixWords = static_cast<uint64_t>(step.rows) * step.cols;

    OPURE_CHECK(matrixWords <= plan.plasticityWords,
                "plasticity target is " + std::to_string(matrixWords) +
                " words but the staging range holds " +
                std::to_string(plan.plasticityWords));
    OPURE_CHECK(step.actOffset + step.cols <= plan.actWords,
                "plasticity activation runs past the end of its bank");
    OPURE_CHECK(step.errOffset + step.rows <= plan.actWords,
                "plasticity error signal runs past the end of its bank");
    OPURE_ASSERT_MSG(step.actBank != step.errBank,
                     "activation and error signal share a bank; the SignSGD pass "
                     "would read and write one resource");

    // The staging bank must be distinct from both sources. Pointing it at the
    // activation's own bank aliases the SRV and the UAV inside one dispatch:
    // that is undefined in D3D12 and it took the device down with
    // DXGI_ERROR_DRIVER_INTERNAL_ERROR, which no debug layer was present to
    // report. All three banks carry the same tail, so any free one will do.
    const uint32_t stageBank = (step.stageBank == 0xFFFFFFFFu)
                                 ? Other(step.actBank, step.errBank)
                                 : step.stageBank;
    OPURE_ASSERT_MSG(stageBank != step.actBank && stageBank != step.errBank,
                     "plasticity staging bank aliases one of the sources");
    OPURE_ASSERT_MSG(stageBank < 3u, "plasticity staging bank is out of range");

    const uint32_t stage = static_cast<uint32_t>(plan.plasticityOffset);

    dev_->ResetCommandList();
    {
        dx12::CommandScope cs(*dev_);

        // Pass 1: compute the update into the staging tail. Reads O_W and two
        // banks through SRVs, writes the third through its UAV.
        const uint32_t p[12] = {
            static_cast<uint32_t>(step.weightOffset),
            step.actOffset,
            step.errOffset,
            step.rows,
            step.cols,
            graph::ModelConfig::FloatBits(step.learningRate),
            graph::ModelConfig::FloatBits(step.weightDecay),
            graph::ModelConfig::FloatBits(step.gradientScale),
            step.actBank,
            step.errBank,
            stageBank,
            stage
        };
        cs.Barrier();
        cs.SetPipeline(lib_->Get(shaders::Id::SignSGD));
        cs.SetParams(p, 12);
        cs.Dispatch(GroupsFor(step.rows, kPlasticityThreads), 1, 1);
    }
    {
        dx12::CommandScope cs(*dev_);

        // Pass 2: commit. Reads the staging bank, writes O_WU.
        const uint32_t q[4] = {
            stage,
            static_cast<uint32_t>(step.weightOffset),
            static_cast<uint32_t>(matrixWords),
            stageBank
        };
        cs.Barrier();
        cs.SetPipeline(lib_->Get(shaders::Id::WeightCommit));
        cs.SetParams(q, 4);
        cs.Dispatch(GroupsFor(static_cast<uint32_t>(matrixWords),
                              kPlasticityThreads), 1, 1);
    }
    dev_->ExecuteAndWait();
}

// ---------------------------------------------------------------------------
void VoxelEngine::RunPlasticity(uint32_t layer, const char* tensor,
                                float lr, float weightDecay)
{
    OPURE_CHECK(ready_, "engine used before Initialize");
    OPURE_CHECK(layer < graph_.Plan().layers.size(),
                "plasticity layer index is out of range");

    const graph::VramPlan::LayerTensors& lt = graph_.Plan().layers[layer];
    const graph::ModelConfig& cfg = graph_.Config();

    PlasticityStep s;
    s.learningRate  = lr;
    s.weightDecay   = weightDecay;
    s.gradientScale = 1.0f;
    s.actBank = kA; s.errBank = kB;

    const std::string name = (tensor != nullptr) ? tensor : "";

    if (name == "q_proj")       { s.weightOffset = lt.wq;  s.rows = cfg.hiddenSize;      s.cols = cfg.hiddenSize;      s.actOffset = 0; }
    else if (name == "k_proj")  { s.weightOffset = lt.wk;  s.rows = cfg.numKvHeads * cfg.HeadDim(); s.cols = cfg.hiddenSize; s.actOffset = 0; }
    else if (name == "v_proj")  { s.weightOffset = lt.wv;  s.rows = cfg.numKvHeads * cfg.HeadDim(); s.cols = cfg.hiddenSize; s.actOffset = 0; }
    else if (name == "o_proj")  { s.weightOffset = lt.wo;  s.rows = cfg.hiddenSize;      s.cols = cfg.hiddenSize;      s.actOffset = scratchA_; }
    else if (name == "gate_proj") { s.weightOffset = lt.gate; s.rows = cfg.intermediateSize; s.cols = cfg.hiddenSize;   s.actOffset = 0; }
    else if (name == "up_proj")   { s.weightOffset = lt.up;   s.rows = cfg.intermediateSize; s.cols = cfg.hiddenSize;   s.actOffset = 0; }
    else if (name == "down_proj") { s.weightOffset = lt.down; s.rows = cfg.hiddenSize;      s.cols = cfg.intermediateSize; s.actOffset = scratchH_; }
    else
    {
        ThrowError("unknown plasticity target tensor: " + name);
    }

    // The error signal is a caller-supplied per-row vector. With no backward
    // pass there is no principled one, so a zero vector is written: sign(0) == 0
    // makes the step decay-only, which is the honest "no gradient" behaviour and
    // is what makes the weight-decay path testable on its own.
    RunPlasticityWithError(s, nullptr);
}

// ---------------------------------------------------------------------------
void VoxelEngine::RunPlasticityWithError(PlasticityStep step, const float* error)
{
    OPURE_CHECK(ready_, "engine used before Initialize");

    // The error signal is a caller-supplied per-row vector. With no backward pass
    // there is no principled one, so a zero vector is staged when the caller has
    // none: sign(0) == 0 makes the step decay-only, which is the honest
    // "no gradient" behaviour and is what makes the weight-decay path testable
    // on its own.
    const uint64_t n = step.rows;
    std::vector<uint32_t> packed(n, 0u);
    if (error != nullptr)
    {
        for (uint64_t i = 0; i < n; ++i)
        {
            std::memcpy(&packed[i], &error[i], 4);
        }
    }

    dev_->ResetCommandList();
    dev_->UploadToBuffer(actB_, packed.data(), n * 4, step.errOffset * 4);

    RunPlasticity(step);
}

} // namespace opure::engine
