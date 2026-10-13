// ============================================================================
//  Opure_Platform.h
//  Opure-VoxelEngine -- shared primitives, assertions, small utilities.
//
//  No third-party dependencies. MSVC / C++20. Warnings-as-errors is enabled
//  project-wide; every macro here is written to avoid MSVC-specific noise.
// ============================================================================
#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#  define NOMINMAX
#endif

#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <utility>
#include <span>

#ifndef OPURE_EXPORT
#  define OPURE_EXPORT
#endif

namespace opure
{

// ---------------------------------------------------------------------------
//  Diagnostics
// ---------------------------------------------------------------------------

enum class Severity { Info, Warn, Error, Fatal };

struct SourceLoc { const char* file; int line; };

void ReportMessage(Severity sev, SourceLoc loc, std::string_view msg);
[[noreturn]] void FailFast(SourceLoc loc, std::string_view msg);

// Runtime failures that are not bugs (OOM, missing file, device loss) surface
// as exceptions so the harness can report them cleanly instead of aborting.
struct Error : std::exception
{
    explicit Error(std::string m) : msg_(std::move(m)) {}
    [[nodiscard]] const char* what() const noexcept override { return msg_.c_str(); }
private:
    std::string msg_;
};

[[noreturn]] void ThrowError(std::string_view msg);

} // namespace opure

#define OPURE_LOC ::opure::SourceLoc{ __FILE__, __LINE__ }

#define OPURE_LOG_INFO(msg)  ::opure::ReportMessage(::opure::Severity::Info,  OPURE_LOC, (msg))
#define OPURE_LOG_WARN(msg)  ::opure::ReportMessage(::opure::Severity::Warn,  OPURE_LOC, (msg))
#define OPURE_LOG_ERROR(msg) ::opure::ReportMessage(::opure::Severity::Error, OPURE_LOC, (msg))

// Internal invariant violation -- always a bug, so it aborts.
#define OPURE_ASSERT(cond)                                                        \
    do {                                                                          \
        if (!(cond)) { ::opure::ReportMessage(::opure::Severity::Fatal,           \
                                              OPURE_LOC,                           \
                                              "assertion failed: " #cond);        \
                     ::opure::FailFast(OPURE_LOC, "assertion failed: " #cond); }   \
    } while (false)

#define OPURE_CHECK(cond, msg)                                                    \
    do {                                                                          \
        if (!(cond)) ::opure::ThrowError((msg));                                  \
    } while (false)

// Like OPURE_ASSERT but with an explanatory message, so a failure says WHY the
// invariant matters rather than just restating the condition. Unlike
// OPURE_ASSERT this throws rather than aborting, because a bank-schedule
// violation is a logic error the caller cannot meaningfully continue past.
#define OPURE_ASSERT_MSG(cond, msg)                                               \
    do {                                                                          \
        if (!(cond)) ::opure::ThrowError((msg));                                  \
    } while (false)

#define OPURE_CHECK_HRESULT(hr, what)                                             \
    do {                                                                          \
        const HRESULT _hr = (hr);                                                 \
        if (FAILED(_hr)) {                                                        \
            ::opure::ThrowError(std::string(what) + " failed (hr=0x" +            \
                ::opure::ToHex(static_cast<uint32_t>(_hr)) + ")");                \
        }                                                                         \
    } while (false)

namespace opure
{
    std::string ToHex(uint32_t v);

    // Aligned allocation for GPU-visible staging. VirtualAlloc on Windows so the
    // pointer satisfies D3D12's 64 KB logical-address alignment requirement.
    class MappedMemory
    {
    public:
        MappedMemory() = default;
        ~MappedMemory() { Release(); }
        MappedMemory(const MappedMemory&)            = delete;
        MappedMemory& operator=(const MappedMemory&) = delete;
        MappedMemory(MappedMemory&& o) noexcept { *this = std::move(o); }
        MappedMemory& operator=(MappedMemory&& o) noexcept;

        // size is rounded up to the system allocation granularity internally.
        void Allocate(size_t size, const wchar_t* name);
        void Release();

        [[nodiscard]] void*  Ptr()  const { return ptr_; }
        [[nodiscard]] size_t Size() const { return size_; }

    private:
        void*   ptr_  = nullptr;
        size_t  size_ = 0;
    };
} // namespace opure