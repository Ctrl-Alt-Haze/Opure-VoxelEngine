// ============================================================================
//  Opure_Common.hlsl
//  Shared declarations for every Opure-VoxelEngine compute pass.
//
//  BINDING CONTRACT
//  ----------------
//  This runtime only accepts STRUCTURED buffer views with a 4-byte element
//  (verified empirically: stride 0, stride 2 and RAW all remove the device with
//  DXGI_ERROR_INVALID_CALL). So the four persistent buffers are bound as
//  StructuredBuffer<uint> / RWStructuredBuffer<uint> and addressed in 32-bit
//  words rather than as ByteAddressBuffer.
//
//  Every tensor is a byte range inside one of these buffers. Ranges are passed
//  as 32-bit WORD offsets (byteOffset >> 2) through the root-constant block, so
//  no descriptor churn happens as model size grows.
//
//  Root constants arrive as a cbuffer at b0 (D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS).
// ============================================================================
#ifndef OPURE_COMMON_HLSL
#define OPURE_COMMON_HLSL

// ---- bound resources ------------------------------------------------------
//
// FOUR slots: weights, activation bank A, activation bank B, KV cache. Each has
// an SRV and a UAV at the same slot.
//
// PING-PONG BANDS
// ---------------
// D3D12 forbids binding one resource as both SRV and UAV in a single dispatch,
// even at disjoint offsets. Every pass therefore READS bank A and WRITES bank B,
// or vice versa, selected by a root constant. This guarantees that no dispatch
// ever touches the same resource through both bindings. The direction is
// uniform across a dispatch, so the branch is perfectly predicted (and the
// GEMV inner loop is memory-bound regardless).
StructuredBuffer<uint>     O_W   : register(t0);   // weights      (SRV)
StructuredBuffer<uint>     O_A   : register(t1);   // activation bank A
StructuredBuffer<uint>     O_B   : register(t2);   // activation bank B
StructuredBuffer<uint>     O_KV  : register(t3);   // kv cache
StructuredBuffer<uint>     O_C   : register(t4);   // residual / third bank

RWStructuredBuffer<uint>  O_WU  : register(u0);   // weights      (UAV; plasticity)
RWStructuredBuffer<uint>  O_AU  : register(u1);   // activation bank A
RWStructuredBuffer<uint>  O_BU  : register(u2);   // activation bank B
RWStructuredBuffer<uint>  O_KVU : register(u3);   // kv cache
RWStructuredBuffer<uint>  O_CU  : register(u4);   // residual / third bank

// ---- per-dispatch parameters ---------------------------------------------
// Parameters ride an SRV rather than root constants. This runtime's
// SetComputeRoot32BitConstants path emits cbufferLoadLegacy with consecutive
// BYTE offsets (0,1,2,3) instead of dword offsets (0,4,8,12) for a uint
// array, so every P[] read back as zero. An SRV is proven to read correctly
// here, so parameters are passed that way.
//
// Each dispatch binds its own 256-byte slot by adding a slot stride to the
// buffer's GPU virtual address -- root descriptors are raw addresses, so this
// is the same mechanism already used for the other five slots. Without the
// per-dispatch offset the GPU would sample one shared slot at execution time
// and every dispatch would observe only the last parameters written.
StructuredBuffer<uint> O_PARAMS : register(t5);
#define P O_PARAMS

// Direction is passed as a root constant by every pass that touches
// activations. The two banks are always SWAPPED between read and write:
//     d == 0  ->  read A, write B
//     d == 1  ->  read B, write A
// so no dispatch ever has one resource bound as both SRV and UAV.
// Three-way bank access, by index: 0 = A, 1 = B, 2 = C (residual).
//
// WHY THIS EXISTS: D3D12 forbids a dispatch that binds one resource as both SRV
// and UAV. The transformer residual makes that unavoidable with only two
// ping-pong banks: o_proj must READ the attention output's bank and therefore
// WRITES the opposite one -- which is exactly where the residual stream already
// lives. The residual add would then have to read and write the same buffer.
//
// A third bank breaks the cycle. The residual rotates A -> C -> B -> A across
// layers, and each add reads two distinct banks and writes the third, so every
// dispatch touches three different resources.
//
// RdAct/WrAct keep the two-bank ping-pong contract used by the GEMV family:
// `dir` 0 means read A / write B, 1 means read B / write A.
inline uint RdBank(uint b, uint i)
{
    return (b == 0u) ? O_A[i] : ((b == 1u) ? O_B[i] : O_C[i]);
}
inline void WrBank(uint b, uint i, uint v)
{
    if      (b == 0u) { O_AU[i] = v; }
    else if (b == 1u) { O_BU[i] = v; }
    else              { O_CU[i] = v; }
}

inline uint RdAct(uint d, uint i) { return (d != 0u) ? O_B[i] : O_A[i]; }
inline void WrAct(uint d, uint i, uint v)
{
    if (d != 0u) { O_AU[i] = v; } else { O_BU[i] = v; }
}

#define OPURE_WAVE_SIZE 32

// Byte-offset -> word-offset. All offsets crossing the ABI are already words,
// so this exists to document intent and to catch accidental byte/word mixes.
#define OPURE_WORDS(bytes) ((uint)((bytes) >> 2))

// Bit-cast helpers. Weights and activations are stored as raw 32-bit words.
inline float AsF(uint v) { return asfloat(v); }
inline uint  AsU(float v) { return asuint(v); }

// Scalar reinterpretation without going through a resource.
inline float2  UnpackF16x2(uint packed)
{
    // low half -> first component, high half -> second
    return float2(f16tof32(packed & 0xFFFFu), f16tof32(packed >> 16));
}

inline uint PackF16x2(float a, float b)
{
    const uint lo = f32tof16(a);
    const uint hi = f32tof16(b);
    return (lo & 0xFFFFu) | (hi << 16);
}

#endif // OPURE_COMMON_HLSL