// ============================================================================
//  Opure_Shaders.cpp
// ============================================================================
#include "core/Opure_Shaders.h"

// Generated at build time into <build>/generated/Opure_ShaderBlob.h
#include "Opure_ShaderBlob.h"

namespace opure::shaders
{

void Library::Initialize(dx12::Device& dev)
{
    // The Id enum and the generated entry-point table must stay in lockstep.
    // A pinned count is what this used to assert, and it rotted silently the
    // moment a 16th entry point was compiled in. Tying the last enum value to
    // the count makes the next addition a compile error instead of a runtime
    // one, and the count assertion catches a stale generated header.
    static_assert(static_cast<uint32_t>(Id::WeightCommit) ==
                  static_cast<uint32_t>(ShaderId::CommitMain),
                  "shaders::Id has drifted out of sync with the generated "
                  "ShaderId table -- add the new pipeline to both");
    static_assert(static_cast<uint32_t>(Id::SignSGD) ==
                  static_cast<uint32_t>(ShaderId::SignSGDMain),
                  "shaders::Id has drifted out of sync with the generated "
                  "ShaderId table -- add the new pipeline to both");
    static_assert(static_cast<uint32_t>(Id::WeightCommit) + 1u ==
                  static_cast<uint32_t>(kShaderCount),
                  "kShaderCount does not match the number of pipelines in Id");
    OPURE_ASSERT(kShaderCount == 30);

    // The attention entry points are the one place where a mapping mistake is
    // invisible: a head_dim-64 model dispatched onto the 256-thread pipeline
    // would still produce finite logits, just wrong ones, because 192 of the 256
    // lanes read past the head. So the name-to-thread-count contract is asserted
    // rather than trusted.
    static_assert(static_cast<uint32_t>(Id::Attention256) ==
                  static_cast<uint32_t>(Id::Attention32) + 7u,
                  "attention entry points must be contiguous from 32 to 256");
    static_assert(static_cast<uint32_t>(Id::SpecVerify256) ==
                  static_cast<uint32_t>(Id::SpecVerify32) + 7u,
                  "speculative verify entry points must be contiguous from 32 to 256");

    // Map our Id enum onto the generated table order.
    struct Entry { Id id; const ShaderBlob* blob; const char* name; };
    const Entry table[] = {
        { Id::Debug,        &kShaderTable[static_cast<size_t>(ShaderId::DebugMain)],          "Debug"          },
        { Id::DebugRootConst,&kShaderTable[static_cast<size_t>(ShaderId::DebugRootConstMain)],"DebugRootConst" },
        { Id::DebugSrv,     &kShaderTable[static_cast<size_t>(ShaderId::DebugSrvMain)],       "DebugSrv"       },
        { Id::Embedding,    &kShaderTable[static_cast<size_t>(ShaderId::EmbeddingMain)],    "Embedding"    },
        { Id::RMSNorm,      &kShaderTable[static_cast<size_t>(ShaderId::RMSNormMain)],      "RMSNorm"      },
        { Id::RoPE,         &kShaderTable[static_cast<size_t>(ShaderId::RoPEMain)],         "RoPE"         },
        { Id::GEMV,         &kShaderTable[static_cast<size_t>(ShaderId::GEMVMain)],         "GEMV"         },
        { Id::GEMVQuant,    &kShaderTable[static_cast<size_t>(ShaderId::GEMVQuantMain)],    "GEMVQuant"    },
        { Id::Attention32,  &kShaderTable[static_cast<size_t>(ShaderId::AttentionMain32)],  "Attention32"  },
        { Id::Attention64,  &kShaderTable[static_cast<size_t>(ShaderId::AttentionMain64)],  "Attention64"  },
        { Id::Attention96,  &kShaderTable[static_cast<size_t>(ShaderId::AttentionMain96)],  "Attention96"  },
        { Id::Attention128, &kShaderTable[static_cast<size_t>(ShaderId::AttentionMain128)], "Attention128" },
        { Id::Attention160, &kShaderTable[static_cast<size_t>(ShaderId::AttentionMain160)], "Attention160" },
        { Id::Attention192, &kShaderTable[static_cast<size_t>(ShaderId::AttentionMain192)], "Attention192" },
        { Id::Attention224, &kShaderTable[static_cast<size_t>(ShaderId::AttentionMain224)], "Attention224" },
        { Id::Attention256, &kShaderTable[static_cast<size_t>(ShaderId::AttentionMain256)], "Attention256" },
        { Id::SpecVerify32,  &kShaderTable[static_cast<size_t>(ShaderId::SpeculativeVerifyMain32)],  "SpecVerify32"  },
        { Id::SpecVerify64,  &kShaderTable[static_cast<size_t>(ShaderId::SpeculativeVerifyMain64)],  "SpecVerify64"  },
        { Id::SpecVerify96,  &kShaderTable[static_cast<size_t>(ShaderId::SpeculativeVerifyMain96)],  "SpecVerify96"  },
        { Id::SpecVerify128, &kShaderTable[static_cast<size_t>(ShaderId::SpeculativeVerifyMain128)], "SpecVerify128" },
        { Id::SpecVerify160, &kShaderTable[static_cast<size_t>(ShaderId::SpeculativeVerifyMain160)], "SpecVerify160" },
        { Id::SpecVerify192, &kShaderTable[static_cast<size_t>(ShaderId::SpeculativeVerifyMain192)], "SpecVerify192" },
        { Id::SpecVerify224, &kShaderTable[static_cast<size_t>(ShaderId::SpeculativeVerifyMain224)], "SpecVerify224" },
        { Id::SpecVerify256, &kShaderTable[static_cast<size_t>(ShaderId::SpeculativeVerifyMain256)], "SpecVerify256" },
        { Id::SwiGLU,       &kShaderTable[static_cast<size_t>(ShaderId::SwiGLUMain)],       "SwiGLU"       },
        { Id::Add,          &kShaderTable[static_cast<size_t>(ShaderId::AddMain)],          "Add"          },
        { Id::AddRes,       &kShaderTable[static_cast<size_t>(ShaderId::AddResMain)],       "AddRes"       },
        { Id::ArgMax,       &kShaderTable[static_cast<size_t>(ShaderId::ArgMaxMain)],       "ArgMax"       },
        { Id::SignSGD,      &kShaderTable[static_cast<size_t>(ShaderId::SignSGDMain)],      "SignSGD"      },
        { Id::WeightCommit, &kShaderTable[static_cast<size_t>(ShaderId::CommitMain)],      "WeightCommit" },
    };

    size_t maxIndex = 0;
    for (const auto& e : table) maxIndex = std::max(maxIndex, static_cast<size_t>(e.id));
    pso_.resize(maxIndex + 1);

    for (const auto& e : table)
    {
        if (!e.blob || e.blob->code == nullptr || e.blob->size == 0)
        {
            ThrowError(std::string("shader blob for '") + e.name + "' is empty; "
                       "the DXIL blob was not generated correctly");
        }

        D3D12_SHADER_BYTECODE bc{};
        bc.pShaderBytecode = e.blob->code;
        bc.BytecodeLength  = e.blob->size;

        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
        psoDesc.pRootSignature = dev.RootSig();
        psoDesc.CS             = bc;
        psoDesc.NodeMask       = 1;
        psoDesc.Flags          = D3D12_PIPELINE_STATE_FLAG_NONE;

        ID3D12PipelineState* raw = nullptr;
        const HRESULT hr = dev.Get()->CreateComputePipelineState(&psoDesc,
                                                                 IID_PPV_ARGS(&raw));
        if (FAILED(hr))
        {
            dx12::ThrowHrWithContext(hr,
                std::string("CreateComputePipelineState(") + e.name + ")", dev.Get());
        }
        pso_[static_cast<size_t>(e.id)].Attach(raw);
    }

    OPURE_LOG_INFO("compiled " + std::to_string(sizeof(table) / sizeof(table[0])) +
                   " compute pipelines for " + dx12::DeviceName());
}

} // namespace opure::shaders