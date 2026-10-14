// ============================================================================
//  Opure_Streamer.h
//  Opure-VoxelEngine -- zero-copy .safetensors reader.
//
//  DESIGN
//  ------
//  A 135M model is 538 MB. Copying that through a staging buffer to reach a
//  mapped view costs a full extra pass over memory for no benefit, so the file
//  is mapped directly with CreateFileMappingW / MapViewOfFile and each tensor's
//  bytes are handed to the GPU as a source range.
//
//  FILE FORMAT
//  -----------
//  safetensors is deliberately trivial:
//
//      u64  header_len          (little endian)
//      u8[header_len] JSON      {"__metadata__":{...}, "name":{"dtype":..,
//                                                   "shape":[..],
//                                                   "data_offsets":[a,b]}}
//      u8[8 + header_len ...]   tensor payload, offsets relative to this point
//
//  Offsets in data_offsets are byte offsets from the START OF THE PAYLOAD, not
//  from the start of the file. Getting that wrong silently yields garbage that
//  looks like a valid tensor, so it is centralised in ByteOffset() below.
//
//  DTYPES
//  ------
//  This engine's GEMV consumes 32-bit words directly, so F32 payloads upload
//  verbatim. F16/BF16 payloads are byte-reassembled on the CPU into F32 at load
//  time (BF16 is a pure bit-shift; F16 needs real conversion). U8/I8 payloads
//  are left raw for the quantised GEMV, which unpacks four bytes per word
//  itself. I32 is rejected: nothing in the graph consumes it, so accepting it
//  would only hide a mismatch.
// ============================================================================
#pragma once

#include "Opure_Platform.h"

#include <cstring>

namespace opure::safetensors
{

// ---------------------------------------------------------------------------
//  Dtype
// ---------------------------------------------------------------------------
enum class Dtype : uint8_t
{
    F32,   // 4 bytes/elt -- uploaded verbatim
    F16,   // 2 bytes/elt -- widened to F32 on load
    BF16,  // 2 bytes/elt -- widened to F32 on load (shift left 16)
    U8,    // 1 byte/elt -- left raw, 4 elts/word, for GEMV_Quant
    I8,    // 1 byte/elt -- left raw, 4 elts/word, for GEMV_Quant
    U4,    // 0.5 byte/elt -- 2 uint4 per byte, widened to F32 on load
    I4,    // 0.5 byte/elt -- 2 int4 per byte, widened to F32 on load
    Count
};

[[nodiscard]] const char* DtypeName(Dtype d);
[[nodiscard]] uint32_t    ElementBytes(Dtype d);

// ---------------------------------------------------------------------------
//  Tensor descriptor
// ---------------------------------------------------------------------------
struct Tensor
{
    std::string          name;
    Dtype                dtype = Dtype::F32;
    std::vector<int64_t> shape;      // row-major, as stored
    uint64_t             byteBegin = 0;   // absolute, into the mapped file
    uint64_t             byteEnd   = 0;   // absolute, one past the last byte
    // Scale factor for quantized tensors (INT8/INT4). Default 1.0 for F32/F16/BF16.
    float                scale = 1.0f;

    [[nodiscard]] uint64_t ElementCount() const;
    [[nodiscard]] uint64_t ByteCount()  const { return byteEnd - byteBegin; }
    // Product of all dims except the last. 1 for a 1-D tensor.
    [[nodiscard]] uint64_t Rows() const;
    [[nodiscard]] uint64_t Cols() const;
};

// ---------------------------------------------------------------------------
//  WeightStreamer
//
//  Owns the file mapping for its lifetime. Not movable-friendly beyond a
//  pointer swap, and intentionally non-copyable.
// ---------------------------------------------------------------------------
class WeightStreamer
{
public:
    WeightStreamer() = default;
    ~WeightStreamer() { Close(); }
    WeightStreamer(const WeightStreamer&)            = delete;
    WeightStreamer& operator=(const WeightStreamer&) = delete;
    WeightStreamer(WeightStreamer&& o)            noexcept { StealFrom(o); }
    WeightStreamer& operator=(WeightStreamer&& o) noexcept;

    // Maps the file and parses its header. Throws opure::Error on any problem.
    void Open(const std::wstring& path);

    void Close();

    [[nodiscard]] bool   IsOpen()  const { return base_ != nullptr; }
    [[nodiscard]] size_t TensorCount() const { return tensors_.size(); }
    [[nodiscard]] const Tensor& TensorAt(size_t i) const { return tensors_[i]; }
    [[nodiscard]] const std::vector<Tensor>& Tensors() const { return tensors_; }

    // Exact-name lookup. Throws if absent -- a typo in a tensor name must fail
    // loudly rather than silently producing zeros.
    [[nodiscard]] const Tensor& Find(std::string_view name) const;
    [[nodiscard]] const Tensor* TryFind(std::string_view name) const;

    // Readable bytes of a tensor, straight out of the mapping. No copy.
    [[nodiscard]] const void* Data(const Tensor& t) const;

    // Byte offset of a tensor's payload within the mapping.
    [[nodiscard]] uint64_t ByteOffset(const Tensor& t) const { return t.byteBegin; }

    [[nodiscard]] uint64_t PayloadBegin() const { return payloadBegin_; }
    [[nodiscard]] uint64_t PayloadBytes() const { return payloadBytes_; }
    [[nodiscard]] uint64_t TotalBytes()  const { return totalBytes_; }

private:
    void ParseHeader(const uint8_t* header, uint64_t headerLen);
    void StealFrom(WeightStreamer& o) noexcept;

    uint8_t*             base_          = nullptr;
    size_t               mappingBytes_  = 0;
    uint64_t             payloadBegin_  = 0;
    uint64_t             payloadBytes_  = 0;
    uint64_t             totalBytes_    = 0;
    std::vector<Tensor>  tensors_;
};

// ---------------------------------------------------------------------------
//  bf16 -> f32. The exponent/mantissa split is a pure left shift, so this is
//  exact and lossless in both directions (no rounding, no denormal handling).
//  Exposed separately because the CPU reference uses it too.
// ---------------------------------------------------------------------------
[[nodiscard]] inline float Bf16ToF32(uint16_t v)
{
    const uint32_t bits = static_cast<uint32_t>(v) << 16;
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

// ===========================================================================
//  Widen a narrow dtype into an F32 word array. Returns bytes written.
//  `out` must have room for ElementCount() floats.
//  U8/I8 are passed through untouched (4 packed bytes per word).
// ===========================================================================
uint64_t WidenToF32(Dtype dt, const void* src, uint64_t elementCount, float* out);

} // namespace opure::safetensors