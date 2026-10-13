// ============================================================================
//  Opure_Shaders.h
//  Opure-VoxelEngine -- pipeline-state factory over the embedded DXIL blob.
//
//  DXIL is produced at build time by scripts/build_shaders.ps1 and baked into
//  the executable, so the runtime never shells out to dxc and the binary is
//  self-contained.
//
//  Every pipeline shares ONE root signature (Device::RootSig) and therefore ONE
//  descriptor layout. Shaders declare their own bindings in Opure_Common.hlsl,
//  so binding a pipeline never requires rebuilding descriptors.
// ============================================================================
#pragma once

#include "core/Opure_Dx12.h"

#include <vector>

namespace opure::shaders
{

// Identifiers matching the generated ShaderId enum in Opure_ShaderBlob.h.
enum class Id : uint32_t
{
    Debug,          // binding self-test: writes a constant, reads nothing
    DebugRootConst, // publishes root constants P[0..3]
    DebugSrv,       // copies t0/t2/t3 SRVs into bank A
    Embedding,
    RMSNorm,
    RoPE,
    GEMV,
    GEMVQuant,
    // One entry point per supported head_dim. Group width IS head_dim, so each
    // needs its own compilation rather than a runtime branch; see
    // AttentionFor() and Opure_FlashAttention_Decode.usf.
    Attention32,
    Attention64,
    Attention96,
    Attention128,
    Attention160,
    Attention192,
    Attention224,
    Attention256,

    // Speculative decode (batched verification) entry points. Same head_dim
    // specializations as attention, but with 12 root parameters for draft
    // metadata. See Opure_SpeculativeDecode.usf.
    SpecVerify32,
    SpecVerify64,
    SpecVerify96,
    SpecVerify128,
    SpecVerify160,
    SpecVerify192,
    SpecVerify224,
    SpecVerify256,

    SwiGLU,
    Add,
    AddRes,      // cross-bank residual add (residual lives in bank C)
    ArgMax,
    SignSGD,     // idle-time weight update, staged into an activation bank
    WeightCommit, // copies staged weights back over the real tensor

    // Sentinel, not a pipeline. Lets AttentionFor() return something for an
    // unsupported head_dim so callers get a diagnosable failure instead of an
    // out-of-range vector access.
    Count
};

class Library
{
public:
    // Builds every pipeline up front. A failure here is fatal: there is no point
    // proceeding with a partial graph, and the error names the offending shader.
    void Initialize(dx12::Device& dev);

    [[nodiscard]] ID3D12PipelineState* Get(Id id) const
    {
        const auto i = static_cast<size_t>(id);
        return (i < pso_.size()) ? pso_[i].Get() : nullptr;
    }

    // Returns the pipeline whose thread-group width matches the model's head
    // dimension. head_dim must be a positive multiple of the wave size; anything
    // else has no entry point, and the caller should have been refused at
    // model-load time by ModelSpec::Check.
    [[nodiscard]] static Id AttentionFor(uint32_t headDim)
    {
        switch (headDim)
        {
        case 32:  return Id::Attention32;
        case 64:  return Id::Attention64;
        case 96:  return Id::Attention96;
        case 128: return Id::Attention128;
        case 160: return Id::Attention160;
        case 192: return Id::Attention192;
        case 224: return Id::Attention224;
        case 256: return Id::Attention256;
        default:  return Id::Count;   // sentinel: caller must not dispatch this
        }
    }

    [[nodiscard]] static constexpr bool HasAttentionFor(uint32_t headDim)
    {
        return headDim != 0 && headDim <= 256 && (headDim % 32u) == 0;
    }

    [[nodiscard]] static Id SpecVerifyFor(uint32_t headDim)
    {
        switch (headDim)
        {
        case 32:  return Id::SpecVerify32;
        case 64:  return Id::SpecVerify64;
        case 96:  return Id::SpecVerify96;
        case 128: return Id::SpecVerify128;
        case 160: return Id::SpecVerify160;
        case 192: return Id::SpecVerify192;
        case 224: return Id::SpecVerify224;
        case 256: return Id::SpecVerify256;
        default:  return Id::Count;
        }
    }

private:
    std::vector<dx12::ComPtr<ID3D12PipelineState>> pso_;
};

} // namespace opure::shaders