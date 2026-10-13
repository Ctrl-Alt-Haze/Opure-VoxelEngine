// ============================================================================
//  Opure_Dx12.h
//  Opure-VoxelEngine -- Direct3D 12 device layer.
//
//  Design notes
//  ------------
//  * One command queue, one command list, one root signature for the whole
//    runtime. Every pass binds the same descriptor tables and varies only its
//    root constants, which removes per-pass binding overhead entirely.
//
//  * A full token is recorded into ONE command list and submitted exactly once.
//    There is no CPU/GPU handshake inside the graph; the CPU waits once, after
//    the sampled token id has been read back.
//
//  * Every persistent buffer is created once as
//        COMMON -> GENERIC_READ (SRV) + UNORDERED_ACCESS
//    and stays there permanently. Because reads and writes coexist by design,
//    no resource barriers are needed; correctness is enforced with UAVBarrier()
//    between dependent dispatches.
//
//  * All persistent GPU storage is four large ByteAddressBuffers:
//        0 Weights    (SRV normally; UAV during the plasticity pass)
//        1 Activations
//        2 KV cache
//        3 Misc / readback / scratch
//    Individual tensors are byte ranges inside them addressed by root constants,
//    so descriptor counts stay constant as models scale.
// ============================================================================
#pragma once

#include "Opure_Platform.h"

#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#  define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include <string>

namespace opure::dx12
{

// Number of DWORDs available in the single root-constant block.
inline constexpr uint32_t kMaxRootConstants = 32;

// Parameter block: 32 DWORDs per dispatch, in its own 256-byte slot.
inline constexpr uint32_t kParamSlotDwords = 32;
inline constexpr uint32_t kParamSlotBytes  = 256;
// Parameter slots, each one dispatch's worth of constants. One token of the
// SmolLM graph is ~423 dispatches, so 1024 slots overflowed after three tokens
// of a generated sequence. 65536 slots (16 MB of UPLOAD heap, which lives in
// system memory) covers a ~150-token prefill in one command list; longer prompts
// are chunked by the engine rather than needing a bigger buffer.
inline constexpr uint32_t kParamSlots      = 65536;
inline constexpr uint32_t kParamBytes      = kParamSlotBytes * kParamSlots;

// One dispatch's worth of root constants occupies one 256-byte-aligned slot.
// A token issues on the order of 350 dispatches, so 1024 slots is ample and
// costs only 256 KiB of system memory.
inline constexpr uint32_t kRootConstSlotBytes = 256;
inline constexpr uint32_t kRootConstSlots     = 1024;
inline constexpr uint32_t kRootConstBytes     = kRootConstSlotBytes * kRootConstSlots;

// SRV/UAV slots exposed to every shader.
enum Slot : uint32_t
{
    kSlotWeights = 0,
    kSlotAct     = 1,
    kSlotKV      = 2,
    // A THIRD activation buffer, used only by the residual add.
    //
    // WHY THIS EXISTS: D3D12 forbids binding one resource as both SRV and UAV
    // in a single dispatch. The residual stream x lives in one ping-pong bank
    // while the attention output lives in the other, so by the time o_proj has
    // written its result it is necessarily in the bank opposite x -- and the
    // residual add can never legally read both and write either. A third bank
    // breaks the tie: the add reads x and the projection from two distinct
    // banks and writes the sum into the third.
    kSlotMisc    = 3,
    kSlotCount   = 4
};

std::string DeviceName();
uint64_t     DedicatedVideoMemory();

// Throws opure::Error describing `hr`, appending the device-removed reason when
// the failure was a device loss (which is otherwise indistinguishable from a
// plain allocation failure at the call site).
void ThrowHrWithContext(HRESULT hr, std::string_view what, ID3D12Device* dev);

class Device;   // GpuBuffer::Create needs the type before it is defined below

// ---------------------------------------------------------------------------
//  Scoped COM release
// ---------------------------------------------------------------------------
template <class T>
class ComPtr
{
public:
    ComPtr() = default;
    explicit ComPtr(T* p) : p_(p) {}
    ~ComPtr() { Reset(); }

    ComPtr(const ComPtr&)            = delete;
    ComPtr& operator=(const ComPtr&) = delete;
    ComPtr(ComPtr&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
    ComPtr& operator=(ComPtr&& o) noexcept
    {
        if (this != &o) { Reset(); p_ = o.p_; o.p_ = nullptr; }
        return *this;
    }

    void Reset()                 { if (p_) { p_->Release(); p_ = nullptr; } }
    T*   Get()     const        { return p_; }
    T*   operator->()  const    { return p_; }
    explicit operator bool() const { return p_ != nullptr; }

    T*   Detach()               { T* p = p_; p_ = nullptr; return p; }
    void Attach(T* p)           { Reset(); p_ = p; }

    // QI helper: `out.As(&iface)` -- `iface` is an IDXGIAdapterN-style pointer.
    template <class U>
    HRESULT As(ComPtr<U>& out) const
    {
        if (!p_) return E_POINTER;
        U* raw = nullptr;
        const HRESULT hr = p_->QueryInterface(IID_PPV_ARGS(&raw));
        out.Attach(raw);
        return hr;
    }

private:
    T* p_ = nullptr;
};

// ---------------------------------------------------------------------------
//  GpuBuffer -- a persistent buffer plus its SRV and UAV descriptors.
// ---------------------------------------------------------------------------
class GpuBuffer
{
public:
    GpuBuffer() = default;
    ~GpuBuffer() { Destroy(); }

    GpuBuffer(const GpuBuffer&)            = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;
    GpuBuffer(GpuBuffer&&)                 = default;
    GpuBuffer& operator=(GpuBuffer&&)      = default;

    void Create(Device& dev, uint32_t slot, uint64_t bytes, bool allowUav);
    void Destroy();

    [[nodiscard]] ID3D12Resource*          Resource() const { return res_.Get(); }
    [[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS GpuVa()    const { return gpuVa_; }
    [[nodiscard]] uint64_t Bytes()  const { return bytes_; }
    [[nodiscard]] bool     HasUav() const { return hasUav_; }

    [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE Srv() const { return srvH_; }
    [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE Uav() const { return uavH_; }

private:
    ComPtr<ID3D12Resource>      res_;
    D3D12_GPU_VIRTUAL_ADDRESS   gpuVa_  = 0;
    uint64_t                    bytes_  = 0;
    bool                        hasUav_ = false;
    D3D12_CPU_DESCRIPTOR_HANDLE srvH_{};
    D3D12_CPU_DESCRIPTOR_HANDLE uavH_{};
};

// ---------------------------------------------------------------------------
//  Device
// ---------------------------------------------------------------------------
class Device
{
public:
    Device() = default;
    ~Device() { Shutdown(); }
    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;

    void Initialize();
    void Shutdown();

    [[nodiscard]] ID3D12Device*            Get()     const { return dev_.Get(); }
    [[nodiscard]] ID3D12CommandQueue*     Queue()   const { return queue_.Get(); }
    [[nodiscard]] ID3D12GraphicsCommandList* CmdList() const { return list_.Get(); }
    [[nodiscard]] ID3D12RootSignature*     RootSig() const { return rootSig_.Get(); }

    // The runtime binds resources through ROOT DESCRIPTORS (GPU virtual
    // addresses), not a descriptor heap: this SDK exposes
    // SetComputeRootUnorderedAccessView / SetComputeRootShaderResourceView, which
    // take a D3D12_GPU_VIRTUAL_ADDRESS. That removes the descriptor heap entirely
    // -- no SetDescriptorHeaps, no descriptor churn, and no dependence on
    // range-offset resolution semantics.
    //
    // Root parameter map:
    //   [0..3] UAV    u0..u3   weights, actA, actB, kv
    //   [4..7] SRV    t0..t3   weights, actA, actB, kv
    //   [8]    32-bit constants (the per-dispatch parameter block)

    // Registers the four persistent buffers plus the residual buffer; called once
    // after they are created.
    void SetPersistentBuffers(GpuBuffer* w, GpuBuffer* a, GpuBuffer* b,
                              GpuBuffer* kv, GpuBuffer* res)
    {
        pW_ = w; pA_ = a; pB_ = b; pKV_ = kv; pRes_ = res;
    }

    // GPU address of the buffer bound to a logical slot (0 for unset slots).
    [[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS BufferVa(uint32_t slot) const
    {
        switch (slot)
        {
        case kSlotWeights: return pW_   ? pW_->GpuVa()   : 0;
        case kSlotAct:     return pA_   ? pA_->GpuVa()   : 0;
        case kSlotKV:      return pB_   ? pB_->GpuVa()   : 0;
        case kSlotMisc:    return pKV_  ? pKV_->GpuVa()  : 0;
        default:           return pRes_ ? pRes_->GpuVa() : 0;
        }
    }

    [[nodiscard]] uint64_t DedicatedVideoMemory() const { return dedicatedVram_; }

    // ---- memory ---------------------------------------------------------
    void     UploadToBuffer(GpuBuffer& dst, const void* src, uint64_t bytes,
                            uint64_t dstOffset = 0);
    uint64_t ReadbackFromBuffer(const GpuBuffer& src, void* dst, uint64_t bytes,
                                uint64_t srcOffset = 0);

    // Batched upload: all ranges are copied in ONE command list with a single
    // fence wait. A 272-tensor model upload via UploadToBuffer would otherwise
    // stall 272 times on the GPU.
    struct UploadRange
    {
        const void* src = nullptr;   // host memory, must outlive the call
        uint64_t    bytes    = 0;
        uint64_t    dstOffset = 0;   // byte offset into the destination
    };
    void UploadRanges(GpuBuffer& dst, std::span<const UploadRange> ranges);

    // Per-dispatch parameter block. Written directly into mapped system memory
    // at the slot the next Dispatch will bind; the host does not need to upload.
    [[nodiscard]] void* ParamSlot(uint32_t slot) const
    {
        return static_cast<uint8_t*>(paramCpu_) +
               static_cast<size_t>(slot) * kParamSlotBytes;
    }
    [[nodiscard]] D3D12_GPU_VIRTUAL_ADDRESS ParamSlotVa(uint32_t slot) const
    {
        return paramVa_ + static_cast<D3D12_GPU_VIRTUAL_ADDRESS>(slot) * kParamSlotBytes;
    }

    // Parameter-slot cursor; reset by ResetCommandList().
    [[nodiscard]] uint32_t ParamSlot() const { return paramSlot_; }
    void AdvanceParamSlot()
    {
        OPURE_CHECK(paramSlot_ < kParamSlots, "parameter slot overflow");
        ++paramSlot_;
    }

    // ---- GPU timing -----------------------------------------------------
    void   CalibrateTimestamp();
    void   BeginTimestamp();
    void   EndTimestamp();
    double ResolveTimestampMs();

    // ---- execution ------------------------------------------------------
    void ResetCommandList();     // close if open, then reopen for recording
    void ExecuteAndWait();

private:
    void CreateFence();
    void CreateQueryHeap();
    void CreateCommandObjects();
    void CreateRootSignature();
    void CreateParamBuffer();

    ComPtr<IDXGIFactory4>             factory_;
    ComPtr<ID3D12Device>              dev_;
    ComPtr<ID3D12CommandQueue>        queue_;
    ComPtr<ID3D12CommandAllocator>    alloc_;
    ComPtr<ID3D12GraphicsCommandList> list_;
    ComPtr<ID3D12RootSignature>       rootSig_;
    ComPtr<ID3D12QueryHeap>           queryHeap_;
    ComPtr<ID3D12Resource>            queryResolve_;
    ComPtr<ID3D12Resource>            queryReadback_;
    ComPtr<ID3D12Fence>               fence_;
    ComPtr<ID3D12Resource>            paramBuf_;

    void*                            paramCpu_  = nullptr;
    D3D12_GPU_VIRTUAL_ADDRESS        paramVa_   = 0;
    uint32_t                         paramSlot_ = 0;

    GpuBuffer* pW_  = nullptr;
    GpuBuffer* pA_  = nullptr;
    GpuBuffer* pB_  = nullptr;
    GpuBuffer* pKV_ = nullptr;
    GpuBuffer* pRes_ = nullptr;

    UINT64   fenceValue_     = 0;
    HANDLE   fenceEvent_     = nullptr;
    uint64_t dedicatedVram_  = 0;
    bool     listOpen_       = false;
};

// ---------------------------------------------------------------------------
//  CommandScope -- recorder over the single persistent command list.
// ---------------------------------------------------------------------------
class CommandScope
{
public:
    explicit CommandScope(Device& dev);

    void SetPipeline(ID3D12PipelineState* pso);
    void SetParams(const uint32_t* rootConstants, uint32_t numDwords);
    void Dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1);
    void Barrier();                        // UAV barrier between dependent passes
    void CopyBuffer(ID3D12Resource* dst, uint64_t dstOff,
                    ID3D12Resource* src, uint64_t srcOff, uint64_t bytes);
    void BeginMarker(const char* label);
    void EndMarker();

private:
    Device*                   dev_  = nullptr;
    ID3D12GraphicsCommandList* list_ = nullptr;
    ID3D12PipelineState*       pso_  = nullptr;
    bool                      paramsDirty_ = true;
    bool                      pipelineDirty_ = true;

    // GPU address of the persistent buffer bound to a logical slot.
    D3D12_GPU_VIRTUAL_ADDRESS BufferVa(uint32_t slot) const;
};

} // namespace opure::dx12