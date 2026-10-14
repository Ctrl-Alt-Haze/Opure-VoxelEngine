// ============================================================================
//  Opure_Dx12.cpp
//
//  NOTE ON THE D3D12 API SURFACE USED HERE
//  ----------------------------------------
//  This build targets a Windows SDK whose D3D12 binding differs from the classic
//  one in several places. All differences are handled here so the rest of the
//  engine is conventional:
//
//    * Resources are bound with ROOT DESCRIPTORS (GPU virtual address) via
//      SetComputeRootUnorderedAccessView / SetComputeRootShaderResourceView,
//      so there is no descriptor heap at all.
//    * Root constants are set with SetComputeRoot32BitConstants, which snapshots
//      the values into the command stream at record time.
//    * Lists reopen with Reset(allocator, state); CreateCommandList hands back a
//      list in the recording state.
//    * Buffer views must be structured with a 4-byte element (see GpuBuffer).
//    * EndQuery/ResolveQueryData take an explicit D3D12_QUERY_TYPE.
//    * The shader-model feature enum is D3D_SHADER_MODEL.
//    * There is no ID3D12Debug in this SDK, so the debug layer is not used.
// ============================================================================
#include "Opure_Dx12.h"

#include <cstdio>
#include <cstring>

namespace opure::dx12
{
namespace
{

std::string WideToUtf8(const wchar_t* w)
{
    if (!w || !*w) return {};
    const int n = ::WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) return {};
    std::string s(static_cast<size_t>(n - 1), '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

std::string g_adapterName;
uint64_t   g_dedicatedVram = 0;

// GPU timestamp tick frequency, calibrated once. D3D12 does not expose it.
double g_gpuTicksPerMs = 0.0;

} // namespace

std::string DeviceName()            { return g_adapterName; }
uint64_t     DedicatedVideoMemory() { return g_dedicatedVram; }

void ThrowHrWithContext(HRESULT hr, std::string_view what, ID3D12Device* dev)
{
    std::string msg(what);
    msg += " failed (hr=0x";
    msg += ToHex(static_cast<uint32_t>(hr));
    msg += ")";

    // A device loss masquerades as an ordinary allocation failure at the call
    // site; surfacing the removed reason is the difference between a
    // diagnosable failure and a mystery.
    if (dev && (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_HUNG ||
                hr == DXGI_ERROR_DEVICE_RESET))
    {
        const HRESULT reason = dev->GetDeviceRemovedReason();
        msg += " -- DEVICE LOST, GetDeviceRemovedReason()=0x";
        msg += ToHex(static_cast<uint32_t>(reason));
        switch (reason)
        {
        case DXGI_ERROR_DEVICE_HUNG:           msg += " (HUNG)";        break;
        case DXGI_ERROR_DEVICE_RESET:          msg += " (RESET)";       break;
        case DXGI_ERROR_DEVICE_REMOVED:        msg += " (REMOVED)";     break;
        case DXGI_ERROR_DRIVER_INTERNAL_ERROR: msg += " (DRIVER)";      break;
        case DXGI_ERROR_INVALID_CALL:          msg += " (INVALID_CALL)"; break;
        case E_OUTOFMEMORY:                    msg += " (OOM)";         break;
        default: break;
        }
    }
    ThrowError(msg);
}

// ---------------------------------------------------------------------------
//  GpuBuffer
//
//  VIEW BINDING CONTRACT (verified empirically against this runtime)
//  ---------------------------------------------------------------
//  Buffer views must be STRUCTURED with StructureByteStride == 4:
//    * StructureByteStride == 0   -> device removed, DXGI_ERROR_INVALID_CALL
//    * D3D12_BUFFER_SRV_FLAG_RAW  -> device removed, DXGI_ERROR_INVALID_CALL
//    * StructureByteStride == 2  -> device removed, DXGI_ERROR_INVALID_CALL
//    * StructureByteStride == 4  -> works, for both SRV and UAV
//  Shaders therefore bind these as StructuredBuffer<uint> /
//  RWStructuredBuffer<uint> addressed in 32-bit words.
// ---------------------------------------------------------------------------
void GpuBuffer::Destroy()
{
    res_.Reset();
    gpuVa_  = 0;
    bytes_  = 0;
    hasUav_ = false;
}

void GpuBuffer::Create(Device& dev, uint32_t slot, uint64_t bytes, bool allowUav)
{
    Destroy();
    // Five bindable slots: 0..3 are the named data slots and 4 is the residual
    // bank. kSlotCount is 4, so the valid RANGE is [0, kSlotCount].
    OPURE_CHECK(slot <= kSlotCount, "GpuBuffer slot out of range");
    (void)allowUav;
    if (bytes == 0) return;

    bytes = (bytes + 255ull) & ~255ull;

    D3D12_HEAP_PROPERTIES props{};
    props.Type                 = D3D12_HEAP_TYPE_DEFAULT;
    props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    props.CreationNodeMask     = 1;
    props.VisibleNodeMask      = 1;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Alignment        = 0;
    rd.Width            = bytes;
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.Format           = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.SampleDesc.Quality = 0;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    // Always request unordered-access capability: the plasticity pass writes
    // weights through u0, and a uniform resource flag set avoids surprises.
    rd.Flags            = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ID3D12Resource* raw = nullptr;
    const HRESULT hr = dev.Get()->CreateCommittedResource(
        &props, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON,
        nullptr, IID_PPV_ARGS(&raw));
    if (FAILED(hr)) ThrowHrWithContext(hr, "CreateCommittedResource(GpuBuffer)", dev.Get());
    res_.Attach(raw);

    bytes_  = bytes;
    hasUav_ = true;
    gpuVa_  = res_->GetGPUVirtualAddress();
}

// ---------------------------------------------------------------------------
//  Device
// ---------------------------------------------------------------------------
void Device::Initialize()
{
    IDXGIFactory4* rawFactory = nullptr;
    OPURE_CHECK_HRESULT(::CreateDXGIFactory2(0, IID_PPV_ARGS(&rawFactory)),
                        "CreateDXGIFactory2");
    factory_.Attach(rawFactory);

    ComPtr<IDXGIAdapter1> best;
    UINT64              bestMem  = 0;
    DXGI_ADAPTER_DESC1  bestDesc{};

    for (UINT i = 0; ; ++i)
    {
        IDXGIAdapter1* rawAdapter = nullptr;
        const HRESULT hr = factory_->EnumAdapters1(i, &rawAdapter);
        if (hr == DXGI_ERROR_NOT_FOUND || FAILED(hr)) break;

        ComPtr<IDXGIAdapter1> adapter(rawAdapter);
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc))) continue;
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;

        if (desc.DedicatedVideoMemory > bestMem)
        {
            bestMem  = desc.DedicatedVideoMemory;
            best     = std::move(adapter);
            bestDesc = desc;
        }
    }

    if (!best)
    {
        ThrowError("no hardware D3D12 adapter found (WARP-only machines are not supported)");
    }

    g_adapterName  = WideToUtf8(bestDesc.Description);
    g_dedicatedVram = bestMem;
    dedicatedVram_ = bestMem;

    ID3D12Device* rawDev = nullptr;
    OPURE_CHECK_HRESULT(D3D12CreateDevice(best.Get(), D3D_FEATURE_LEVEL_11_0,
                                          IID_PPV_ARGS(&rawDev)),
                        "D3D12CreateDevice");
    dev_.Attach(rawDev);

    D3D12_FEATURE_DATA_SHADER_MODEL sm{};
    if (FAILED(dev_->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm))))
    {
        OPURE_LOG_WARN("device declined the shader-model capability query; "
                       "continuing (PSO creation will enforce SM 6.6)");
    }
    else
    {
        OPURE_LOG_INFO("highest shader model reported by device: " +
                       std::to_string(static_cast<int>(sm.HighestShaderModel)));
    }

    // No dedicated heaps: CreateCommittedResource takes heap properties directly
    // and the runtime grows placement on demand. Pre-reserving GBs would, on a
    // card already sharing VRAM with a desktop compositor, be what tips the
    // device into DXGI_ERROR_DEVICE_REMOVED.

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type     = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    qd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
    qd.Flags    = D3D12_COMMAND_QUEUE_FLAG_NONE;
    qd.NodeMask = 1;
    ID3D12CommandQueue* rawQueue = nullptr;
    OPURE_CHECK_HRESULT(dev_->CreateCommandQueue(&qd, IID_PPV_ARGS(&rawQueue)),
                        "CreateCommandQueue");
    queue_.Attach(rawQueue);

    CreateRootSignature();
    CreateParamBuffer();
    CreateCommandObjects();
    CreateFence();
    CreateQueryHeap();
}

// The per-dispatch parameter block lives in an always-mapped UPLOAD-heap buffer
// so the host writes it straight into system memory while recording. Each
// dispatch binds its own slot by offsetting the GPU virtual address, which is
// necessary because the GPU samples this memory at EXECUTION time -- a single
// shared slot would leave every dispatch observing only the last parameters
// written during recording.
void Device::CreateParamBuffer()
{
    D3D12_HEAP_PROPERTIES props{};
    props.Type                 = D3D12_HEAP_TYPE_UPLOAD;
    props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    props.CreationNodeMask     = 1;
    props.VisibleNodeMask      = 1;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = kParamBytes;
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.Format           = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.SampleDesc.Quality = 0;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags            = D3D12_RESOURCE_FLAG_NONE;

    ID3D12Resource* raw = nullptr;
    OPURE_CHECK_HRESULT(dev_->CreateCommittedResource(
        &props, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr, IID_PPV_ARGS(&raw)), "CreateCommittedResource(param buffer)");
    paramBuf_.Attach(raw);

    OPURE_CHECK_HRESULT(paramBuf_->Map(0, nullptr, &paramCpu_), "Map(param buffer)");
    std::memset(paramCpu_, 0, static_cast<size_t>(kParamBytes));
    paramVa_ = paramBuf_->GetGPUVirtualAddress();
}

void Device::CreateRootSignature()
{
    // Twelve root parameters, all root DESCRIPTORS (no descriptor heap):
    //   rp[0..4]   UAV  u0..u4 -> weights, actA, actB, kv, residual
    //   rp[5..10]  SRV  t0..t5 -> weights, actA, actB, kv, residual, params
    //
    // Counts: kSlotCount (4) data slots + 1 residual = 5 UAVs, and 5 data SRVs
    // plus the parameter block = 6 SRVs. Total = 2 * kSlotCount + 2.
    constexpr uint32_t kUavCount = kSlotCount + 1;          // 5
    constexpr uint32_t kSrvCount = kSlotCount + 2;          // 6 (5 data + params)
    D3D12_ROOT_PARAMETER rp[kUavCount + kSrvCount]{};

    for (uint32_t i = 0; i < kUavCount; ++i)
    {
        rp[i].ParameterType              = D3D12_ROOT_PARAMETER_TYPE_UAV;
        rp[i].Descriptor.ShaderRegister  = i;
        rp[i].Descriptor.RegisterSpace   = 0;
        rp[i].ShaderVisibility           = D3D12_SHADER_VISIBILITY_ALL;
    }

    // Data SRVs at t0..t4, then the parameter block at t5.
    for (uint32_t i = 0; i < kSlotCount + 1; ++i)
    {
        const uint32_t p = kUavCount + i;
        rp[p].ParameterType              = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rp[p].Descriptor.ShaderRegister  = i;
        rp[p].Descriptor.RegisterSpace   = 0;
        rp[p].ShaderVisibility           = D3D12_SHADER_VISIBILITY_ALL;
    }
    rp[kUavCount + kSlotCount + 1].ParameterType              = D3D12_ROOT_PARAMETER_TYPE_SRV;
    rp[kUavCount + kSlotCount + 1].Descriptor.ShaderRegister  = kSlotCount + 1;
    rp[kUavCount + kSlotCount + 1].Descriptor.RegisterSpace   = 0;
    rp[kUavCount + kSlotCount + 1].ShaderVisibility           = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.NumParameters = static_cast<UINT>(sizeof(rp) / sizeof(rp[0]));
    rsd.pParameters   = rp;
    rsd.Flags         = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ID3DBlob* rawBlob = nullptr;
    ID3DBlob* rawErr  = nullptr;
    const HRESULT hr = D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1,
                                                   &rawBlob, &rawErr);
    if (FAILED(hr))
    {
        std::string msg = "D3D12SerializeRootSignature failed";
        if (rawErr)
        {
            msg += ": " + std::string(static_cast<const char*>(rawErr->GetBufferPointer()),
                                      rawErr->GetBufferSize());
            rawErr->Release();
        }
        if (rawBlob) rawBlob->Release();
        ThrowError(msg);
    }
    if (rawErr) rawErr->Release();

    ID3D12RootSignature* rawSig = nullptr;
    OPURE_CHECK_HRESULT(dev_->CreateRootSignature(
        0, rawBlob->GetBufferPointer(), rawBlob->GetBufferSize(),
        IID_PPV_ARGS(&rawSig)), "CreateRootSignature");
    rawBlob->Release();
    rootSig_.Attach(rawSig);
}

void Device::CreateCommandObjects()
{
    ID3D12CommandAllocator* rawAlloc = nullptr;
    OPURE_CHECK_HRESULT(dev_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                     IID_PPV_ARGS(&rawAlloc)),
                        "CreateCommandAllocator");
    alloc_.Attach(rawAlloc);

    ID3D12GraphicsCommandList* rawList = nullptr;
    OPURE_CHECK_HRESULT(dev_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                                                alloc_.Get(), nullptr,
                                                IID_PPV_ARGS(&rawList)),
                        "CreateCommandList");
    list_.Attach(rawList);
    // CreateCommandList returns the list in the recording (open) state.
    listOpen_ = true;
}

void Device::CreateFence()
{
    ID3D12Fence* raw = nullptr;
    OPURE_CHECK_HRESULT(dev_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&raw)),
                        "CreateFence");
    fence_.Attach(raw);

    fenceEvent_ = ::CreateEventW(nullptr, FALSE, FALSE, nullptr);
    OPURE_CHECK(fenceEvent_ != nullptr, "CreateEvent(fence) failed");
    fenceValue_ = 0;
}

void Device::CreateQueryHeap()
{
    D3D12_QUERY_HEAP_DESC qd{};
    qd.Type     = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    qd.Count    = 2;
    qd.NodeMask = 1;

    ID3D12QueryHeap* rawHeap = nullptr;
    OPURE_CHECK_HRESULT(dev_->CreateQueryHeap(&qd, IID_PPV_ARGS(&rawHeap)),
                        "CreateQueryHeap");
    queryHeap_.Attach(rawHeap);

    D3D12_HEAP_PROPERTIES props{};
    props.Type                 = D3D12_HEAP_TYPE_READBACK;
    props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    props.CreationNodeMask     = 1;
    props.VisibleNodeMask      = 1;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = 2 * sizeof(uint64_t);
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.Format           = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.SampleDesc.Quality = 0;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags            = D3D12_RESOURCE_FLAG_NONE;

    ID3D12Resource* rawResolve = nullptr;
    OPURE_CHECK_HRESULT(dev_->CreateCommittedResource(
        &props, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr, IID_PPV_ARGS(&rawResolve)), "CreateCommittedResource(queryResolve)");
    queryResolve_.Attach(rawResolve);

    ID3D12Resource* rawReadback = nullptr;
    OPURE_CHECK_HRESULT(dev_->CreateCommittedResource(
        &props, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr, IID_PPV_ARGS(&rawReadback)), "CreateCommittedResource(queryReadback)");
    queryReadback_.Attach(rawReadback);
}

void Device::Shutdown()
{
    if (queue_ && fenceValue_ != 0)
    {
        ++fenceValue_;
        if (SUCCEEDED(queue_->Signal(fence_.Get(), fenceValue_)) &&
            SUCCEEDED(fence_->SetEventOnCompletion(fenceValue_, fenceEvent_)))
        {
            ::WaitForSingleObject(fenceEvent_, 2000);
        }
    }
    if (fenceEvent_) { ::CloseHandle(fenceEvent_); fenceEvent_ = nullptr; }

    queryReadback_.Reset(); queryResolve_.Reset(); queryHeap_.Reset();
    fence_.Reset();
    list_.Reset(); alloc_.Reset(); rootSig_.Reset();
    queue_.Reset();  dev_.Reset(); factory_.Reset();
    listOpen_ = false;
}

void Device::ResetCommandList()
{
    if (listOpen_)
    {
        list_->Close();
        listOpen_ = false;
    }

    // The allocator may only be reset once everything that used it has completed.
    if (fenceValue_ != 0)
    {
        if (FAILED(fence_->SetEventOnCompletion(fenceValue_, fenceEvent_)))
        {
            ThrowError("SetEventOnCompletion failed before command-list reset");
        }
        if (::WaitForSingleObject(fenceEvent_, 30000) != WAIT_OBJECT_0)
        {
            ThrowError("timeout waiting for GPU before command-list reset");
        }
    }

    alloc_->Reset();
    OPURE_CHECK_HRESULT(list_->Reset(alloc_.Get(), nullptr),
                        "ID3D12GraphicsCommandList::Reset");
    listOpen_ = true;

    // Recycle the parameter-slot cursor. Every dispatch in a list binds its own
    // slot, so this must start over or a long-lived device exhausts kParamSlots.
    paramSlot_ = 0;
}

void Device::ExecuteAndWait()
{
    if (!listOpen_)
    {
        // Nothing has been recorded since the last submit. Submitting the list
        // again hands the queue a command list that was already closed, which
        // D3D12 rejects and the driver reports as
        // DXGI_ERROR_DRIVER_INTERNAL_ERROR plus a device hang -- with no
        // HRESULT of its own to check, because ExecuteCommandLists returns void.
        // A redundant ExecuteAndWait is an easy mistake to make (every helper
        // here already ends in one), so it is a no-op rather than a fault.
        return;
    }

    OPURE_CHECK_HRESULT(list_->Close(), "ID3D12GraphicsCommandList::Close");
    listOpen_ = false;

    ID3D12CommandList* arr[] = { list_.Get() };
    queue_->ExecuteCommandLists(1, arr);

    ++fenceValue_;
    if (FAILED(queue_->Signal(fence_.Get(), fenceValue_)))
    {
        ThrowError("queue Signal failed");
    }
    if (FAILED(fence_->SetEventOnCompletion(fenceValue_, fenceEvent_)))
    {
        ThrowError("SetEventOnCompletion failed after submit");
    }
    if (::WaitForSingleObject(fenceEvent_, 30000) != WAIT_OBJECT_0)
    {
        ThrowError("timeout waiting for GPU after submit");
    }
}

void Device::CalibrateTimestamp()
{
    if (g_gpuTicksPerMs > 0.0) return;

    LARGE_INTEGER qpf{};
    ::QueryPerformanceFrequency(&qpf);

    ResetCommandList();
    list_->EndQuery(queryHeap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    list_->ResolveQueryData(queryHeap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 1,
                            queryResolve_.Get(), 0);
    list_->CopyBufferRegion(queryReadback_.Get(), 0, queryResolve_.Get(), 0, sizeof(uint64_t));
    ExecuteAndWait();

    LARGE_INTEGER host0{}, host1{};
    ::QueryPerformanceCounter(&host0);
    ::Sleep(150);
    ::QueryPerformanceCounter(&host1);

    ResetCommandList();
    list_->EndQuery(queryHeap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    list_->ResolveQueryData(queryHeap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1, 1,
                            queryResolve_.Get(), 0);
    list_->CopyBufferRegion(queryReadback_.Get(), sizeof(uint64_t), queryResolve_.Get(), 0,
                            sizeof(uint64_t));
    ExecuteAndWait();

    uint64_t ticks[2] = {0, 0};
    void* mapped = nullptr;
    if (SUCCEEDED(queryReadback_->Map(0, nullptr, &mapped)) && mapped)
    {
        std::memcpy(ticks, mapped, sizeof(ticks));
        queryReadback_->Unmap(0, nullptr);
    }

    const uint64_t dt = (ticks[1] > ticks[0]) ? (ticks[1] - ticks[0]) : 0;
    const double   hostMs = static_cast<double>(host1.QuadPart - host0.QuadPart) * 1000.0 /
                            static_cast<double>(qpf.QuadPart);
    g_gpuTicksPerMs = (dt > 0 && hostMs > 1.0)
                        ? (static_cast<double>(dt) / hostMs)
                        : (static_cast<double>(qpf.QuadPart) / 1000.0);
}

void Device::BeginTimestamp()
{
    CalibrateTimestamp();
    ResetCommandList();
    list_->EndQuery(queryHeap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
}

void Device::EndTimestamp()
{
    list_->EndQuery(queryHeap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    list_->ResolveQueryData(queryHeap_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 2,
                            queryResolve_.Get(), 0);
    list_->CopyBufferRegion(queryReadback_.Get(), 0, queryResolve_.Get(), 0,
                            2 * sizeof(uint64_t));
    ExecuteAndWait();
}

double Device::ResolveTimestampMs()
{
    uint64_t ticks[2] = {0, 0};
    void* mapped = nullptr;
    if (FAILED(queryReadback_->Map(0, nullptr, &mapped)) || !mapped) return -1.0;
    std::memcpy(ticks, mapped, sizeof(ticks));
    queryReadback_->Unmap(0, nullptr);

    const uint64_t delta = (ticks[1] > ticks[0]) ? (ticks[1] - ticks[0]) : 0;
    if (delta == 0 || g_gpuTicksPerMs <= 0.0) return 0.0;
    return static_cast<double>(delta) / g_gpuTicksPerMs;
}

void Device::UploadToBuffer(GpuBuffer& dst, const void* src, uint64_t bytes,
                            uint64_t dstOffset)
{
    OPURE_CHECK(bytes > 0, "UploadToBuffer called with zero bytes");

    D3D12_HEAP_PROPERTIES props{};
    props.Type                 = D3D12_HEAP_TYPE_UPLOAD;
    props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    props.CreationNodeMask     = 1;
    props.VisibleNodeMask      = 1;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = bytes;
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.Format           = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.SampleDesc.Quality = 0;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags            = D3D12_RESOURCE_FLAG_NONE;

    ID3D12Resource* rawStaging = nullptr;
    const HRESULT hr = dev_->CreateCommittedResource(
        &props, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr, IID_PPV_ARGS(&rawStaging));
    if (FAILED(hr)) ThrowHrWithContext(hr, "CreateCommittedResource(upload staging)", dev_.Get());
    ComPtr<ID3D12Resource> staging(rawStaging);

    void* mapped = nullptr;
    OPURE_CHECK_HRESULT(staging->Map(0, nullptr, &mapped), "Map(upload staging)");
    std::memcpy(mapped, src, static_cast<size_t>(bytes));
    staging->Unmap(0, nullptr);

    ResetCommandList();
    list_->CopyBufferRegion(dst.Resource(), dstOffset, staging.Get(), 0, bytes);
    ExecuteAndWait();
}

// Copies many ranges in one command list with one fence wait. Ranges larger
// than the staging budget are split across several sub-copies, so a multi-GB
// upload never needs multi-GB of transient host-visible VRAM.
void Device::UploadRanges(GpuBuffer& dst, std::span<const UploadRange> ranges)
{
    if (ranges.empty()) { return; }

    // Staging budget: 32 MB is large enough to keep the copy engine busy and
    // small enough to be irrelevant next to the destination allocation.
    constexpr uint64_t kStageBudget = 32ull * 1024 * 1024;

    D3D12_HEAP_PROPERTIES props{};
    props.Type                 = D3D12_HEAP_TYPE_UPLOAD;
    props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    props.CreationNodeMask     = 1;
    props.VisibleNodeMask      = 1;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = kStageBudget;
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.Format           = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.SampleDesc.Quality = 0;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags            = D3D12_RESOURCE_FLAG_NONE;

    ID3D12Resource* rawStaging = nullptr;
    const HRESULT hr = dev_->CreateCommittedResource(
        &props, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr, IID_PPV_ARGS(&rawStaging));
    if (FAILED(hr)) ThrowHrWithContext(hr, "CreateCommittedResource(batched upload)", dev_.Get());
    ComPtr<ID3D12Resource> staging(rawStaging);

    void* mapped = nullptr;
    OPURE_CHECK_HRESULT(staging->Map(0, nullptr, &mapped), "Map(batched upload)");
    auto* base = static_cast<uint8_t*>(mapped);

    ResetCommandList();

    // The staging buffer is reused, so it must be executed before being
    // overwritten. That forces one submit per full staging batch -- ~17 for a
    // 538 MB model at a 32 MB budget, instead of one per tensor.
    uint64_t staged = 0;
    auto recycle = [&]()
    {
        ExecuteAndWait();      // runs the copies that read `staging`
        ResetCommandList();    // discards nothing; the old list is done
        staged = 0;
    };

    for (const UploadRange& r : ranges)
    {
        const auto* src = static_cast<const uint8_t*>(r.src);
        uint64_t done = 0;
        while (done < r.bytes)
        {
            if (staged + kStageBudget > kStageBudget)
            {
                recycle();   // staging is full
            }
            const uint64_t chunk = std::min(r.bytes - done, kStageBudget - staged);
            std::memcpy(base + staged, src + done, static_cast<size_t>(chunk));
            list_->CopyBufferRegion(dst.Resource(), r.dstOffset + done,
                                    staging.Get(), staged, chunk);
            staged += chunk;
            done   += chunk;
        }
    }

    staging->Unmap(0, nullptr);
    ExecuteAndWait();
}

uint64_t Device::ReadbackFromBuffer(const GpuBuffer& src, void* dst, uint64_t bytes,
                                    uint64_t srcOffset)
{
    D3D12_HEAP_PROPERTIES props{};
    props.Type                 = D3D12_HEAP_TYPE_READBACK;
    props.CPUPageProperty      = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    props.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    props.CreationNodeMask     = 1;
    props.VisibleNodeMask      = 1;

    D3D12_RESOURCE_DESC rd{};
    rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width            = bytes;
    rd.Height           = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels        = 1;
    rd.Format           = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.SampleDesc.Quality = 0;
    rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags            = D3D12_RESOURCE_FLAG_NONE;

    ID3D12Resource* rawRb = nullptr;
    const HRESULT hr = dev_->CreateCommittedResource(
        &props, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr, IID_PPV_ARGS(&rawRb));
    if (FAILED(hr)) ThrowHrWithContext(hr, "CreateCommittedResource(readback)", dev_.Get());
    ComPtr<ID3D12Resource> rb(rawRb);

    ResetCommandList();
    list_->CopyBufferRegion(rb.Get(), 0, src.Resource(), srcOffset, bytes);
    ExecuteAndWait();

    void* mapped = nullptr;
    OPURE_CHECK_HRESULT(rb->Map(0, nullptr, &mapped), "Map(readback)");
    std::memcpy(dst, mapped, static_cast<size_t>(bytes));
    rb->Unmap(0, nullptr);
    return bytes;
}

// ---------------------------------------------------------------------------
//  CommandScope
// ---------------------------------------------------------------------------
CommandScope::CommandScope(Device& dev) : dev_(&dev), list_(dev.CmdList()) {}

void CommandScope::SetPipeline(ID3D12PipelineState* pso)
{
    if (pso_ != pso || pipelineDirty_)
    {
        list_->SetPipelineState(pso);
        pso_            = pso;
        pipelineDirty_  = false;
    }
}

void CommandScope::SetParams(const uint32_t* rootConstants, uint32_t numDwords)
{
    OPURE_CHECK(numDwords <= kParamSlotDwords, "parameter count exceeds kParamSlotDwords");
    void* slot = dev_->ParamSlot(dev_->ParamSlot());
    std::memset(slot, 0, kParamSlotBytes);
    if (rootConstants && numDwords)
    {
        std::memcpy(slot, rootConstants, numDwords * sizeof(uint32_t));
    }
    paramsDirty_ = true;
}

void CommandScope::Dispatch(uint32_t x, uint32_t y, uint32_t z)
{
    list_->SetComputeRootSignature(dev_->RootSig());

    // Root descriptors: 5 UAVs at rp[0..4], data SRVs at rp[5..9], params at rp[10].
    for (uint32_t i = 0; i < kSlotCount; ++i)
    {
        list_->SetComputeRootUnorderedAccessView(i, BufferVa(i));
    }
    list_->SetComputeRootUnorderedAccessView(kSlotCount, BufferVa(kSlotCount));
    for (uint32_t i = 0; i <= kSlotCount; ++i)
    {
        list_->SetComputeRootShaderResourceView(kSlotCount + 1 + i, BufferVa(i));
    }

    // Per-dispatch parameter slot, addressed by offsetting the root descriptor's
    // virtual address so each dispatch in a long graph keeps its own values.
    list_->SetComputeRootShaderResourceView(2 * kSlotCount + 2,
                                            dev_->ParamSlotVa(dev_->ParamSlot()));
    OPURE_CHECK(dev_->ParamSlot() < kParamSlots, "parameter slot overflow");
    dev_->AdvanceParamSlot();

    list_->Dispatch(x, y, z);
}

D3D12_GPU_VIRTUAL_ADDRESS CommandScope::BufferVa(uint32_t slot) const
{
    return dev_->BufferVa(slot);
}

void CommandScope::Barrier()
{
    // Global UAV barrier: ordering for every unordered-access resource. All
    // persistent buffers are permanently in the SRV+UAV state, so this is the
    // only synchronisation primitive the graph needs. A null resource makes the
    // barrier global (this SDK's UAV barrier struct carries only pResource).
    D3D12_RESOURCE_BARRIER rb{};
    rb.Type          = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    rb.Flags         = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    rb.UAV.pResource = nullptr;
    list_->ResourceBarrier(1, &rb);
}

void CommandScope::EndMarker() { list_->EndEvent(); }

void CommandScope::CopyBuffer(ID3D12Resource* dst, uint64_t dstOff,
                              ID3D12Resource* src, uint64_t srcOff, uint64_t bytes)
{
    list_->CopyBufferRegion(dst, dstOff, src, srcOff, bytes);
}

void CommandScope::BeginMarker(const char* label)
{
    if (!label || !*label) return;
    list_->BeginEvent(0, label, static_cast<UINT>(std::strlen(label) + 1));
}

} // namespace opure::dx12