// ============================================================================
//  Opure_Platform.cpp
// ============================================================================
#include "Opure_Platform.h"

#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace opure
{

std::string ToHex(uint32_t v)
{
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%08X", v);
    return buf;
}

static const char* SeverityTag(Severity s)
{
    switch (s)
    {
    case Severity::Info:  return "info ";
    case Severity::Warn:  return "warn ";
    case Severity::Error: return "ERROR";
    case Severity::Fatal: return "FATAL";
    }
    return "?????";
}

void ReportMessage(Severity sev, SourceLoc loc, std::string_view msg)
{
    // Shorten the path to the file name so log lines stay readable.
    std::string_view file(loc.file);
    const size_t slash = file.find_last_of("/\\");
    if (slash != std::string_view::npos) file.remove_prefix(slash + 1);

    std::fprintf(stderr, "[%s] %s:%d  %.*s\n",
                 SeverityTag(sev), file.data(), loc.line,
                 static_cast<int>(msg.size()), msg.data());
    std::fflush(stderr);
}

[[noreturn]] void FailFast(SourceLoc loc, std::string_view msg)
{
    ReportMessage(Severity::Fatal, loc, msg);
    std::abort();
}

void ThrowError(std::string_view msg)
{
    throw Error(std::string(msg));
}

MappedMemory& MappedMemory::operator=(MappedMemory&& o) noexcept
{
    if (this != &o)
    {
        Release();
        ptr_      = o.ptr_;
        size_     = o.size_;
        o.ptr_    = nullptr;
        o.size_   = 0;
    }
    return *this;
}

void MappedMemory::Allocate(size_t size, const wchar_t* name)
{
    Release();

    SYSTEM_INFO si{};
    ::GetSystemInfo(&si);
    const SIZE_T gran = si.dwAllocationGranularity;
    const size_t rounded = (size + gran - 1) / gran * gran;

    // PAGE_READWRITE is intentional: mapped file views are always shared, and we
    // want the mmap path in the streamer to be writable-capable for the same code.
    ptr_ = ::VirtualAlloc(nullptr, rounded, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!ptr_)
    {
        ThrowError("VirtualAlloc failed for staging allocation of " +
                   std::to_string(size) + " bytes");
    }
    size_ = rounded;

    if (name)
    {
        // Purely diagnostic; failure is non-fatal.
        ::VirtualAlloc(nullptr, 0, MEM_RESERVE, PAGE_READWRITE); // no-op guard
    }
}

void MappedMemory::Release()
{
    if (ptr_)
    {
        ::VirtualFree(ptr_, 0, MEM_RELEASE);
        ptr_  = nullptr;
        size_ = 0;
    }
}

} // namespace opure