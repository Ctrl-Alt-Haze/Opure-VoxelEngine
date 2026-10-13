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
    Attention,        // head_dim == 64  (256-thread group would idle; see .usf)
    Attention128,     // head_dim == 128
    SwiGLU,
    Add,
    AddRes,      // cross-bank residual add (residual lives in bank C)
    ArgMax,
    SignSGD,     // idle-time weight update, staged into an activation bank
    WeightCommit // copies staged weights back over the real tensor
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
    // dimension, so both head_dim 64 and 128 models get fully occupied groups.
    [[nodiscard]] Id AttentionFor(uint32_t headDim) const
    {
        return (headDim > 64) ? Id::Attention128 : Id::Attention;
    }

private:
    std::vector<dx12::ComPtr<ID3D12PipelineState>> pso_;
};

} // namespace opure::shaders