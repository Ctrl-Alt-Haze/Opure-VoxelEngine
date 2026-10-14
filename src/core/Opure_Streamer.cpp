// ============================================================================
//  Opure_Streamer.cpp
//  Opure-VoxelEngine -- memory-mapped .safetensors reader.
// ============================================================================
#include "Opure_Streamer.h"

#include "Opure_Json.h"

#include <windows.h>

#include <cstdio>
#include <cstring>

namespace opure::safetensors
{

namespace
{

// Reinterpret a mapped pointer as a little-endian u64. Written explicitly
// rather than memcpy'd from a uint64_t* because the mapping has no alignment
// guarantee for the 8-byte length prefix.
uint64_t ReadLE64(const uint8_t* p)
{
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
    {
        v |= static_cast<uint64_t>(p[i]) << (8 * i);
    }
    return v;
}

Dtype ParseDtype(std::string_view s)
{
    if (s == "F32")  { return Dtype::F32;  }
    if (s == "F16")  { return Dtype::F16;  }
    if (s == "BF16") { return Dtype::BF16; }
    if (s == "U8")   { return Dtype::U8;   }
    if (s == "I8")   { return Dtype::I8;   }
    if (s == "U4")   { return Dtype::U4;   }
    if (s == "I4")   { return Dtype::I4;   }
    if (s == "I8")   { return Dtype::I8;   }
    ThrowError("unsupported safetensors dtype '" + std::string(s) + "'");
}

// IEEE 754 binary16 -> binary32. Handles subnormals, which do occur in
// quantised checkpoints, so the naive "just shift the mantissa" shortcut is
// not sufficient.
float F16ToF32(uint16_t h)
{
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t man  = h & 0x3FFu;

    uint32_t bits;
    if (exp == 0u)
    {
        if (man == 0u)
        {
            bits = sign;                                  // +/- 0
        }
        else
        {
            // Subnormal: renormalise into a binary32 normal.
            uint32_t e = 0u;
            uint32_t m = man;
            while ((m & 0x400u) == 0u) { m <<= 1; ++e; }
            m &= 0x3FFu;
            bits = sign | ((127u - 15u - e) << 23) | (m << 13);
        }
    }
    else if (exp == 0x1Fu)
    {
        bits = sign | 0x7F800000u | (man << 13);           // inf / NaN
    }
    else
    {
        bits = sign | ((exp + (127u - 15u)) << 23) | (man << 13);
    }

    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
const char* DtypeName(Dtype d)
{
    switch (d)
    {
    case Dtype::F32:  return "F32";
    case Dtype::F16:  return "F16";
    case Dtype::BF16: return "BF16";
    case Dtype::U8:   return "U8";
    case Dtype::I8:   return "I8";
    case Dtype::U4:   return "U4";
    case Dtype::I4:   return "I4";
    default:          return "?";
    }
}

uint32_t ElementBytes(Dtype d)
{
    switch (d)
    {
    case Dtype::F32:
    case Dtype::F16:
    case Dtype::BF16: return 4u;   // F32 and the two 16-bit forms widen to a word
    case Dtype::U8:
    case Dtype::I8:   return 1u;
    case Dtype::U4:
    case Dtype::I4:   return 1u;   // 2 elements per byte, but we widen to F32 (4 bytes each)
    default:          return 0u;
    }
}

// ---------------------------------------------------------------------------
uint64_t Tensor::ElementCount() const
{
    uint64_t n = 1;
    for (const int64_t d : shape)
    {
        // A -1 dim would mean "inferred"; safetensors never emits one and a
        // negative extent would wrap to a huge value here, so reject it.
        OPURE_CHECK(d >= 0, "safetensors tensor '" + name + "' has a negative dimension");
        n *= static_cast<uint64_t>(d);
    }
    return n;
}

uint64_t Tensor::Rows() const
{
    uint64_t r = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i)
    {
        r *= static_cast<uint64_t>(shape[i]);
    }
    return r;
}

uint64_t Tensor::Cols() const
{
    return shape.empty() ? 1u : static_cast<uint64_t>(shape.back());
}

// ---------------------------------------------------------------------------
WeightStreamer& WeightStreamer::operator=(WeightStreamer&& o) noexcept
{
    if (this != &o)
    {
        Close();
        StealFrom(o);
    }
    return *this;
}

void WeightStreamer::StealFrom(WeightStreamer& o) noexcept
{
    base_         = o.base_;
    mappingBytes_ = o.mappingBytes_;
    payloadBegin_ = o.payloadBegin_;
    payloadBytes_ = o.payloadBytes_;
    totalBytes_   = o.totalBytes_;
    tensors_      = std::move(o.tensors_);

    o.base_          = nullptr;
    o.mappingBytes_  = 0;
    o.payloadBegin_  = 0;
    o.payloadBytes_  = 0;
    o.totalBytes_    = 0;
}

// ---------------------------------------------------------------------------
void WeightStreamer::Open(const std::wstring& path)
{
    Close();

    HANDLE file = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                nullptr, OPEN_EXISTING,
                                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
                                nullptr);
    if (file == INVALID_HANDLE_VALUE)
    {
        ThrowError("cannot open safetensors file (win32 error " +
                   ToHex(static_cast<uint32_t>(::GetLastError())) + ")");
    }

    LARGE_INTEGER size{};
    if (!::GetFileSizeEx(file, &size) || size.QuadPart <= 8)
    {
        ::CloseHandle(file);
        ThrowError("safetensors file is too small to contain a length prefix");
    }
    const auto fileBytes = static_cast<uint64_t>(size.QuadPart);

    // FILE_FLAG_RANDOM_ACCESS (the RAM/disk mapping flavour, not sequential)
    // is deliberate: we touch every tensor exactly once in a scattered order.
    HANDLE mapping = ::CreateFileMappingW(file, nullptr, PAGE_READONLY,
                                          0, 0, nullptr);
    ::CloseHandle(file);
    if (mapping == nullptr)
    {
        ThrowError("CreateFileMappingW failed (win32 error " +
                   ToHex(static_cast<uint32_t>(::GetLastError())) + ")");
    }

    auto* base = static_cast<uint8_t*>(
        ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0));
    if (base == nullptr)
    {
        const auto err = ::GetLastError();
        ::CloseHandle(mapping);
        ThrowError("MapViewOfFile failed (win32 error " + ToHex(static_cast<uint32_t>(err)) + ")");
    }

    // The mapping owns the bytes; the view stays valid after the handle closes.
    ::CloseHandle(mapping);

    base_         = base;
    mappingBytes_ = static_cast<size_t>(fileBytes);
    totalBytes_   = fileBytes;

    try
    {
        const uint64_t headerLen = ReadLE64(base_);
        OPURE_CHECK(headerLen > 0 && headerLen <= fileBytes - 8,
                    "safetensors header length is out of range");
        payloadBegin_ = 8 + headerLen;
        ParseHeader(base + 8, headerLen);
        payloadBytes_ = fileBytes - payloadBegin_;
    }
    catch (...)
    {
        Close();
        throw;
    }
}

void WeightStreamer::Close()
{
    if (base_ != nullptr)
    {
        ::UnmapViewOfFile(base_);
        base_ = nullptr;
    }
    mappingBytes_  = 0;
    payloadBegin_  = 0;
    payloadBytes_  = 0;
    totalBytes_    = 0;
    tensors_.clear();
}

// ---------------------------------------------------------------------------
void WeightStreamer::ParseHeader(const uint8_t* header, uint64_t headerLen)
{
    const std::string_view text(reinterpret_cast<const char*>(header), headerLen);
    const json::Value root = json::ParseObject(text, "safetensors header");

    // data_offsets are relative to the payload start, which is only known once
    // the header length is known -- so collect relative spans and convert after.
    struct Span { size_t index; uint64_t begin; uint64_t end; };
    std::vector<Span> spans;
    spans.reserve(root.obj.size());

    for (const auto& [name, v] : root.obj)
    {
        if (name == "__metadata__") { continue; }

        Tensor t;
        t.name = name;
        t.dtype = ParseDtype(v.At("dtype", name).str);

        // Parse scale factor for quantized tensors (default 1.0)
        if (v.IsObject())
        {
            if (const json::Value* scaleVal = v.Find("scale"); scaleVal && scaleVal->IsNumber())
            {
                t.scale = static_cast<float>(scaleVal->num);
            }
        }

        const json::Value& shape = v.At("shape", name);
        OPURE_CHECK(shape.IsArray(), "tensor '" + name + "' has a non-array shape");
        t.shape.reserve(shape.arr.size());
        for (const json::Value& d : shape.arr)
        {
            OPURE_CHECK(d.IsNumber(), "tensor '" + name + "' has a non-numeric shape entry");
            t.shape.push_back(d.inum);
        }

        const json::Value& off = v.At("data_offsets", name);
        OPURE_CHECK(off.IsArray() && off.arr.size() == 2,
                    "tensor '" + name + "' must have exactly two data_offsets");
        const uint64_t b = static_cast<uint64_t>(off.arr[0].inum);
        const uint64_t e = static_cast<uint64_t>(off.arr[1].inum);
        OPURE_CHECK(b <= e, "tensor '" + name + "' has data_offsets out of order");

        spans.push_back({ tensors_.size(), b, e });
        tensors_.push_back(std::move(t));
    }

    OPURE_CHECK(!tensors_.empty(), "safetensors header declares no tensors");

    const uint64_t payloadBytes = totalBytes_ - payloadBegin_;

    for (const Span& s : spans)
    {
        Tensor& t = tensors_[s.index];

        OPURE_CHECK(s.end <= payloadBytes,
                    "tensor '" + t.name + "' runs past the end of the payload");

        // Cross-check the declared byte span against shape x element size. A
        // mismatch means either a truncated file or a wrong dtype mapping, and
        // both produce plausible-looking garbage if allowed through.
        const uint64_t expect = t.ElementCount() * ElementBytes(t.dtype);
        OPURE_CHECK(s.end - s.begin == expect,
                    "tensor '" + t.name + "' declares " +
                    std::to_string(s.end - s.begin) + " bytes but its shape and " +
                    DtypeName(t.dtype) + " dtype imply " + std::to_string(expect));

        // The + payloadBegin_ here is the whole reason ByteOffset() exists:
        // offsets in the file are payload-relative, not file-relative.
        t.byteBegin = payloadBegin_ + s.begin;
        t.byteEnd   = payloadBegin_ + s.end;
        OPURE_CHECK(t.byteEnd <= totalBytes_,
                    "tensor '" + t.name + "' maps past the end of the file");
    }
}

// ---------------------------------------------------------------------------
const Tensor& WeightStreamer::Find(std::string_view name) const
{
    const Tensor* t = TryFind(name);
    if (t == nullptr)
    {
        ThrowError("tensor '" + std::string(name) + "' not found in the model");
    }
    return *t;
}

const Tensor* WeightStreamer::TryFind(std::string_view name) const
{
    for (const Tensor& t : tensors_)
    {
        if (t.name == name) { return &t; }
    }
    return nullptr;
}

const void* WeightStreamer::Data(const Tensor& t) const
{
    OPURE_CHECK(base_ != nullptr, "WeightStreamer::Data on a closed streamer");
    OPURE_CHECK(t.byteEnd <= totalBytes_, "tensor '" + t.name + "' is out of range");
    return base_ + t.byteBegin;
}

// ---------------------------------------------------------------------------
uint64_t WidenToF32(Dtype dt, const void* src, uint64_t elementCount, float* out)
{
    const auto* s = static_cast<const uint8_t*>(src);

    switch (dt)
    {
    case Dtype::F32:
        std::memcpy(out, s, elementCount * 4);
        return elementCount * 4;

    case Dtype::BF16:
    {
        // Read each element with memcpy rather than casting to uint16_t*: the
        // mapping has no alignment guarantee, and an unaligned 16-bit load
        // would fault or (with /W4) fail to compile.
        for (uint64_t i = 0; i < elementCount; ++i)
        {
            uint16_t h;
            std::memcpy(&h, s + i * 2, 2);
            out[i] = Bf16ToF32(h);
        }
        return elementCount * 4;
    }

    case Dtype::F16:
    {
        for (uint64_t i = 0; i < elementCount; ++i)
        {
            uint16_t h;
            std::memcpy(&h, s + i * 2, 2);
            out[i] = F16ToF32(h);
        }
        return elementCount * 4;
    }

    case Dtype::U8:
    case Dtype::I8:
        // Left packed: the quantised GEMV unpacks four bytes per word itself.
        // Copy as raw bytes; the caller treats the region as uint32 words.
        std::memcpy(out, s, elementCount);
        return elementCount;

    case Dtype::U4:
    case Dtype::I4:
    {
        // INT4 packs 2 elements per byte (high nibble first, then low nibble).
        // Each 4-bit value is sign-extended to F32.
        // For U4: values 0-15. For I4: values -8..7 (sign bit at bit 3).
        const bool isSigned = (dt == Dtype::I4);
        for (uint64_t i = 0; i < elementCount; ++i)
        {
            uint8_t byte = s[i >> 1];
            uint8_t nibble = (i & 1) ? (byte & 0x0F) : (byte >> 4);
            if (isSigned)
            {
                // Sign-extend 4-bit to 32-bit float
                int val = (nibble & 0x8) ? (int)nibble - 16 : (int)nibble;
                out[i] = static_cast<float>(val);
            }
            else
            {
                out[i] = static_cast<float>(nibble);
            }
        }
        return elementCount * 4;
    }

    default:
        ThrowError("WidenToF32 cannot widen this dtype");
    }
}

} // namespace opure::safetensors