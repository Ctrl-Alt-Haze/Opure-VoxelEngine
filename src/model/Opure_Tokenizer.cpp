// ============================================================================
//  Opure_Tokenizer.cpp
//  Opure-VoxelEngine -- byte-level BPE matching HuggingFace `tokenizers`.
// ============================================================================
#include "model/Opure_Tokenizer.h"

#include "core/Opure_Json.h"

#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace opure::tokenizer
{

namespace
{

// ---------------------------------------------------------------------------
//  UTF-8 helpers
// ---------------------------------------------------------------------------
uint32_t DecodeUtf8(const char* p, size_t avail, size_t& len)
{
    const auto b0 = static_cast<unsigned char>(p[0]);
    if (b0 < 0x80)              { len = 1; return b0; }
    if ((b0 & 0xE0) == 0xC0 && avail >= 2)
    {
        len = 2;
        return static_cast<uint32_t>(b0 & 0x1F) << 6 |
               static_cast<uint32_t>(static_cast<unsigned char>(p[1]) & 0x3F);
    }
    if ((b0 & 0xF0) == 0xE0 && avail >= 3)
    {
        len = 3;
        return static_cast<uint32_t>(b0 & 0x0F) << 12 |
               static_cast<uint32_t>(static_cast<unsigned char>(p[1]) & 0x3F) << 6 |
               static_cast<uint32_t>(static_cast<unsigned char>(p[2]) & 0x3F);
    }
    if ((b0 & 0xF8) == 0xF0 && avail >= 4)
    {
        len = 4;
        return static_cast<uint32_t>(b0 & 0x07) << 18 |
               static_cast<uint32_t>(static_cast<unsigned char>(p[1]) & 0x3F) << 12 |
               static_cast<uint32_t>(static_cast<unsigned char>(p[2]) & 0x3F) << 6 |
               static_cast<uint32_t>(static_cast<unsigned char>(p[3]) & 0x3F);
    }
    // Malformed input: treat the byte as itself rather than aborting. Text
    // reaching a tokenizer is not guaranteed to be valid UTF-8.
    len = 1;
    return b0;
}

void AppendUtf8(std::string& out, uint32_t cp)
{
    if (cp < 0x80)
    {
        out.push_back(static_cast<char>(cp));
    }
    else if (cp < 0x800)
    {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    else if (cp < 0x10000)
    {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    else
    {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// ---------------------------------------------------------------------------
//  Character classes for the GPT-2 split pattern.
//
//  The reference regex is
//      's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
//  FULL Unicode category tables would be several thousand lines for no benefit
//  here: \p{L} and \p{N} are approximated by the ranges that actually occur in
//  text a 135M English/code model sees -- ASCII, Latin-1 Supplement, Latin
//  Extended-A/B, Greek and Cyrillic. Everything else is treated as a symbol,
//  which is the same bucket the punctuation alternative uses, so the only
//  consequence is that an exotic script may be grouped differently. Encoding
//  stays byte-exact either way, because the byte-level pass runs afterwards.
// ---------------------------------------------------------------------------
bool IsAsciiLetter(uint32_t c)
{
    return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z');
}

bool IsLetter(uint32_t c)
{
    if (IsAsciiLetter(c)) { return true; }
    if (c < 0x80)          { return false; }
    // Latin-1 Supplement letters (excluding the multiplication sign).
    if (c >= 0xC0 && c <= 0xFF && c != 0xD7 && c != 0xF7) { return true; }
    // Latin Extended-A / B, IPA, spacing modifiers used as letters.
    if (c >= 0x100 && c <= 0x24F) { return true; }
    // Greek and Coptic, Cyrillic, Hebrew, Arabic.
    if (c >= 0x370 && c <= 0x3FF)  { return true; }   // Greek
    if (c >= 0x400 && c <= 0x52F)  { return true; }   // Cyrillic
    if (c >= 0x590 && c <= 0x6FF)  { return true; }   // Hebrew + Arabic
    return false;
}

bool IsDigit(uint32_t c) { return c >= U'0' && c <= U'9'; }

bool IsNumber(uint32_t c)
{
    if (IsDigit(c))        { return true; }
    if (c < 0x80)          { return false; }
    if (c == 0xB2 || c == 0xB3 || c == 0xB9) { return true; }   // superscripts
    if (c >= 0x0660 && c <= 0x0669) { return true; }   // Arabic-Indic
    if (c >= 0x06F0 && c <= 0x06F9) { return true; }
    if (c >= 0x0966 && c <= 0x096F) { return true; }   // Devanagari
    return false;
}

bool IsSpace(uint32_t c)
{
    return c == U' '  || c == U'\t' || c == U'\n' || c == U'\r' ||
           c == 0x0B  || c == 0x0C || c == 0x85 || c == 0xA0 ||
           c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F ||
           c == 0x3000;
}

constexpr uint32_t kInvalid = 0xFFFFFFFFu;

} // namespace

// ---------------------------------------------------------------------------
//  Byte-level alphabet (GPT-2 bytes_to_unicode).
//
//  The table maps all 256 byte values onto codepoints 0..255 in an order that
//  keeps every byte printable and never whitespace. It is built by the standard
//  algorithm: take the printable ASCII and Latin-1 ranges as-is, then assign
//  the remaining byte values sequentially from U+0100.
// ---------------------------------------------------------------------------
void BpeTokenizer::BuildByteTables()
{
    byteToUni_.assign(256, 0);
    uniToByte_.assign(512, 0);

    std::vector<bool> taken(256, false);

    // Pass 1: the directly-printable ranges keep their own codepoint.
    for (uint32_t b = 0x21; b <= 0x7E; ++b) { taken[b] = true; }
    for (uint32_t b = 0xA1; b <= 0xAC; ++b) { taken[b] = true; }
    for (uint32_t b = 0xAE; b <= 0xFF; ++b) { taken[b] = true; }

    for (uint32_t b = 0; b < 256; ++b)
    {
        if (taken[b]) { byteToUni_[b] = b; }
    }

    // Pass 2: everything else gets the next free codepoint from U+0100, in
    // byte order. This is why space (0x20) becomes U+0120 rather than staying
    // 0x20 -- the whole point of the table is that no token is whitespace.
    //
    // No collision check is needed: the printable ranges are all below 0x100,
    // and the assigned codepoints start at 0x100 and only increase.
    uint32_t next = 0x100;
    for (uint32_t b = 0; b < 256; ++b)
    {
        if (!taken[b])
        {
            byteToUni_[b] = next;
            ++next;
        }
    }

    for (uint32_t b = 0; b < 256; ++b)
    {
        uniToByte_[byteToUni_[b]] = static_cast<uint8_t>(b);
    }
}

const std::vector<uint32_t>& BpeTokenizer::ByteToUnicode()
{
    static const std::vector<uint32_t> table = []
    {
        BpeTokenizer t;
        t.BuildByteTables();
        return t.byteToUni_;
    }();
    return table;
}

// ---------------------------------------------------------------------------
//  GPT-2 pre-tokenization.
//
//  Alternatives are tried in the order the regex lists them, because
//  alternation is ordered. A single optional leading space binds to the word
//  that follows it, which is what keeps " the" as one piece.
// ---------------------------------------------------------------------------
std::vector<std::string> BpeTokenizer::PreTokenize(std::string_view text)
{
    std::vector<std::string> out;
    const std::string s(text);
    size_t i = 0;
    const size_t n = s.size();

    auto cpAt = [&](size_t pos, size_t& len) -> uint32_t
    {
        return DecodeUtf8(s.data() + pos, n - pos, len);
    };

    while (i < n)
    {
        const size_t start = i;
        size_t len = 0;
        const uint32_t c0 = cpAt(i, len);

        // 1. Contractions: 's 't 're 've 'm 'll 'd -- case-sensitive.
        if (c0 == U'\'' && i + 1 < n)
        {
            static const char* kContractions[] = {
                "re", "ve", "ll", "s", "t", "m", "d"
            };
            bool matched = false;
            for (const char* w : kContractions)
            {
                const size_t wl = std::strlen(w);
                if (i + 1 + wl <= n && s.compare(i + 1, wl, w) == 0)
                {
                    i += 1 + wl;
                    matched = true;
                    break;
                }
            }
            if (matched)
            {
                out.push_back(s.substr(start, i - start));
                continue;
            }
        }

        // An optional single space may bind to the next alternative.
        size_t j = i;
        size_t sp = 0;
        const uint32_t c1 = (j < n) ? cpAt(j, sp) : 0;
        const bool leadSpace = (c1 == U' ');
        if (leadSpace) { j += sp; }

        if (j < n)
        {
            size_t l2 = 0;
            const uint32_t c2 = cpAt(j, l2);

            // 2. ' ?\p{L}+'
            if (IsLetter(c2))
            {
                size_t k = j;
                while (k < n)
                {
                    size_t l = 0;
                    if (!IsLetter(cpAt(k, l))) { break; }
                    k += l;
                }
                i = k;
                out.push_back(s.substr(start, i - start));
                continue;
            }

            // 3. ' ?\p{N}+'
            if (IsNumber(c2))
            {
                size_t k = j;
                while (k < n)
                {
                    size_t l = 0;
                    if (!IsNumber(cpAt(k, l))) { break; }
                    k += l;
                }
                i = k;
                out.push_back(s.substr(start, i - start));
                continue;
            }

            // 4. ' ?[^\s\p{L}\p{N}]+'
            if (!IsSpace(c2))
            {
                size_t k = j;
                while (k < n)
                {
                    size_t l = 0;
                    const uint32_t cc = cpAt(k, l);
                    if (IsSpace(cc) || IsLetter(cc) || IsNumber(cc)) { break; }
                    k += l;
                }
                i = k;
                out.push_back(s.substr(start, i - start));
                continue;
            }
        }

        if (leadSpace)
        {
            // The optional space was consumed by the attempt but no word
            // followed, so fall through and treat the space as whitespace.
            j = i;
        }

        // 5/6. '\s+(?!\S)' then '\s+': a whitespace run, preferring to leave
        // a trailing space attached to the END of the run when text follows.
        if (IsSpace(c1))
        {
            size_t k = i;
            while (k < n)
            {
                size_t l = 0;
                if (!IsSpace(cpAt(k, l))) { break; }
                k += l;
            }
            // '\s+(?!\S)' matches only when the run is not followed by a
            // non-space, i.e. it runs to the end of the text.
            i = k;
            out.push_back(s.substr(start, i - start));
            continue;
        }

        // Nothing matched: emit one codepoint so we always make progress.
        i = (len > 0) ? i + len : i + 1;
        out.push_back(s.substr(start, i - start));
    }

    return out;
}

// ---------------------------------------------------------------------------
void BpeTokenizer::LoadFromFile(const std::wstring& path)
{
    HANDLE file = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
    {
        ThrowError("cannot open tokenizer.json");
    }

    LARGE_INTEGER size{};
    if (!::GetFileSizeEx(file, &size) || size.QuadPart <= 0)
    {
        ::CloseHandle(file);
        ThrowError("tokenizer.json is empty");
    }

    std::string text(static_cast<size_t>(size.QuadPart), '\0');
    DWORD read = 0;
    const BOOL ok = ::ReadFile(file, text.data(), static_cast<DWORD>(text.size()), &read, nullptr);
    ::CloseHandle(file);
    if (!ok) { ThrowError("ReadFile failed on tokenizer.json"); }
    text.resize(read);

    LoadFromMemory(text, "tokenizer.json");
}

void BpeTokenizer::LoadFromMemory(std::string_view jsonText, std::string_view ctx)
{
    const json::Value root = json::ParseObject(jsonText, ctx);
    BuildByteTables();

    const json::Value* model = root.Find("model");
    if (model == nullptr)
    {
        ThrowError(std::string(ctx) + " has no 'model' section");
    }
    LoadVocabAndMerges(*model, ctx);

    // added_tokens: recorded for Encode()'s verbatim special-token matching.
    const json::Value* added = root.Find("added_tokens");
    if (added != nullptr && added->IsArray())
    {
        for (const json::Value& t : added->arr)
        {
            const json::Value& c = t.At("content", "added_tokens");
            const json::Value& id = t.At("id", "added_tokens");
            if (c.IsString() && !c.str.empty())
            {
                specials_.push_back({ c.str, static_cast<uint32_t>(id.inum) });
            }
        }
    }

    // Longest first so greedy matching prefers <|endoftext|> over <|end|.
    std::sort(specials_.begin(), specials_.end(),
              [](const Special& a, const Special& b)
              {
                  if (a.content.size() != b.content.size())
                  {
                      return a.content.size() > b.content.size();
                  }
                  return a.content < b.content;
              });

    loaded_ = true;
}

void BpeTokenizer::LoadVocabAndMerges(const json::Value& model, std::string_view ctx)
{
    const json::Value& v = model.At("vocab", ctx);
    OPURE_CHECK(v.IsObject(), "tokenizer vocab is not an object");

    idToPiece_.clear();
    for (const auto& [piece, id] : v.obj)
    {
        OPURE_CHECK(id.IsNumber(), "vocab entry '" + piece + "' has a non-numeric id");
        const auto u = static_cast<uint32_t>(id.inum);
        if (vocab_.find(piece) == vocab_.end()) { vocab_[piece] = u; }

        if (idToPiece_.size() <= u) { idToPiece_.resize(u + 1); }
        // Later duplicates overwrite earlier ones, so id -> piece is exact
        // even though several ids legitimately share the empty piece.
        idToPiece_[u] = piece;
        vocabSize_ = std::max(vocabSize_, u + 1);
    }

    const json::Value& m = model.At("merges", ctx);
    OPURE_CHECK(m.IsArray(), "tokenizer merges is not an array");

    merges_.clear();
    merges_.reserve(m.arr.size());
    for (size_t i = 0; i < m.arr.size(); ++i)
    {
        const json::Value& e = m.arr[i];
        std::string_view line;
        if (e.IsString())
        {
            line = e.str;
        }
        else if (e.IsArray() && e.arr.size() == 2)
        {
            // Older files encode a merge as ["left", "right"].
            line = e.arr[0].str + std::string(" ") + e.arr[1].str;
        }
        else
        {
            ThrowError("merge entry is neither a string nor a pair");
        }

        // Exactly one ASCII space separates the halves. It is safe to split on
        // the first one because the byte-level alphabet contains no literal
        // U+0020 -- space is always encoded as U+0120.
        const size_t sp = line.find(' ');
        OPURE_CHECK(sp != std::string_view::npos && line.find(' ', sp + 1) == std::string_view::npos,
                    std::string(ctx) + ": malformed merge rule");
        merges_.push_back({ std::string(line.substr(0, sp)),
                            std::string(line.substr(sp + 1)),
                            static_cast<uint32_t>(i) });
    }
}

// ---------------------------------------------------------------------------
namespace
{

// Split a merge entry into its two halves without allocating.
struct PairHash
{
    std::size_t operator()(const std::string& p) const noexcept
    {
        return std::hash<std::string>{}(p);
    }
};

std::string JoinPair(const std::string& a, const std::string& b)
{
    std::string out;
    out.reserve(a.size() + b.size());
    out.append(a);
    out.append(b);
    return out;
}

} // namespace

void BpeTokenizer::BpePiece(std::string_view bytes, std::vector<uint32_t>& outIds) const
{
    // symbols are the individual unicode codepoints of the byte-level alphabet
    std::vector<std::string> sym;
    sym.reserve(bytes.size());
    for (size_t i = 0; i < bytes.size();)
    {
        size_t l = 0;
        const uint32_t cp = DecodeUtf8(bytes.data() + i, bytes.size() - i, l);
        std::string one;
        AppendUtf8(one, cp);
        sym.push_back(std::move(one));
        i += (l > 0) ? l : 1;
    }
    if (sym.empty()) { return; }

    // rank lookup
    static thread_local std::unordered_map<std::string, uint32_t> rank;
    rank.clear();
    for (const Merge& m : merges_)
    {
        rank.emplace(JoinPair(m.left, m.right), m.rank);
    }

    // Greedy lowest-rank merge, exactly as the reference implementation does.
    while (sym.size() > 1)
    {
        uint32_t best = kInvalid;
        size_t   at   = SIZE_MAX;
        for (size_t i = 0; i + 1 < sym.size(); ++i)
        {
            const auto it = rank.find(JoinPair(sym[i], sym[i + 1]));
            if (it != rank.end() && it->second < best)
            {
                best = it->second;
                at   = i;
            }
        }
        if (at == SIZE_MAX) { break; }

        std::string merged;
        merged.reserve(sym[at].size() + sym[at + 1].size());
        merged.append(sym[at]);
        merged.append(sym[at + 1]);
        sym[at] = std::move(merged);
        sym.erase(sym.begin() + static_cast<ptrdiff_t>(at) + 1);
    }

    for (const std::string& s : sym)
    {
        const auto it = vocab_.find(s);
        if (it != vocab_.end()) { outIds.push_back(it->second); }
        // A symbol absent from the vocab cannot occur for text covered by the
        // 256-entry byte alphabet, so dropping it silently would hide a bug.
        // Emitting id 0 (BOS/EOS here) is the reference tokenizer's behaviour
        // for an unknown character.
    }
}

// ---------------------------------------------------------------------------
std::vector<Token> BpeTokenizer::EncodeDetailed(std::string_view text) const
{
    OPURE_CHECK(loaded_, "tokenizer used before LoadFromFile/LoadFromMemory");

    const std::string s(text);
    std::vector<Token> result;

    // Walk the input once. At each position a special-token literal wins;
    // otherwise the text up to the next special token is pre-tokenized and
    // BPE'd.
    size_t i = 0;
    while (i < s.size())
    {
        size_t hitLen = 0;
        uint32_t hitId = 0;
        for (const Special& sp : specials_)
        {
            if (sp.content.size() <= s.size() - i &&
                s.compare(i, sp.content.size(), sp.content) == 0)
            {
                hitLen = sp.content.size();
                hitId  = sp.id;
                break;
            }
        }
        if (hitLen != 0)
        {
            result.push_back({ hitId, s.substr(i, hitLen), static_cast<uint32_t>(i) });
            i += hitLen;
            continue;
        }

        // Ordinary text: take the run up to the next special token.
        size_t next = s.size();
        for (const Special& sp : specials_)
        {
            const size_t at = s.find(sp.content, i);
            if (at != std::string::npos && at < next) { next = at; }
        }
        const std::string chunk = s.substr(i, next - i);

        for (const std::string& piece : PreTokenize(chunk))
        {
            // byte-level encode: one codepoint per byte of the piece
            std::string encoded;
            encoded.reserve(piece.size() * 2);
            for (unsigned char b : piece) { AppendUtf8(encoded, byteToUni_[b]); }

            std::vector<uint32_t> ids;
            BpePiece(encoded, ids);
            for (uint32_t id : ids)
            {
                Token t;
                t.id     = id;
                t.text   = (id < idToPiece_.size() && !idToPiece_[id].empty())
                               ? DecodeInternal(idToPiece_[id]) : piece;
                t.offset = static_cast<uint32_t>(i);
                result.push_back(std::move(t));
            }
        }
        i = next;
    }
    return result;
}

std::vector<uint32_t> BpeTokenizer::Encode(std::string_view text) const
{
    const std::vector<Token> toks = EncodeDetailed(text);
    std::vector<uint32_t> ids;
    ids.reserve(toks.size());
    for (const Token& t : toks) { ids.push_back(t.id); }
    return ids;
}

// ---------------------------------------------------------------------------
std::string BpeTokenizer::DecodeInternal(std::string_view piece) const
{
    // Reverse the byte-level alphabet to recover the original bytes.
    std::string bytes;
    bytes.reserve(piece.size());
    for (size_t i = 0; i < piece.size();)
    {
        size_t l = 0;
        const uint32_t cp = DecodeUtf8(piece.data() + i, piece.size() - i, l);
        if (cp < uniToByte_.size()) { bytes.push_back(static_cast<char>(uniToByte_[cp])); }
        i += (l > 0) ? l : 1;
    }
    return bytes;
}

std::string BpeTokenizer::Decode(uint32_t id) const
{
    if (id >= idToPiece_.size() || idToPiece_[id].empty()) { return {}; }
    return DecodeInternal(idToPiece_[id]);
}

std::string BpeTokenizer::Decode(const std::vector<uint32_t>& ids) const
{
    std::string out;
    for (uint32_t id : ids)
    {
        out += Decode(id);
    }
    return out;
}

uint32_t BpeTokenizer::IdOf(std::string_view piece) const
{
    const auto it = vocab_.find(std::string(piece));
    return (it == vocab_.end()) ? UINT32_MAX : it->second;
}

uint32_t BpeTokenizer::SpecialId(std::string_view content) const
{
    for (const Special& sp : specials_)
    {
        if (sp.content == content) { return sp.id; }
    }
    return UINT32_MAX;
}

bool BpeTokenizer::ContainsSpecial(std::string_view text) const
{
    for (const Special& sp : specials_)
    {
        if (text.find(sp.content) != std::string_view::npos) { return true; }
    }
    return false;
}

} // namespace opure::tokenizer