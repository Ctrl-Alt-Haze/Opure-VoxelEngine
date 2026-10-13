// ============================================================================
//  Opure_Json.cpp
// ============================================================================
#include "Opure_Json.h"

#include <charconv>
#include <cmath>

namespace opure::json
{

const Value* Value::Find(std::string_view key) const
{
    if (type != Type::Object) return nullptr;
    for (const auto& kv : obj)
    {
        if (kv.first == key) return &kv.second;
    }
    return nullptr;
}

const Value& Value::At(std::string_view key, std::string_view ctx) const
{
    if (type != Type::Object)
    {
        ThrowError(std::string(ctx) + ": expected a JSON object while looking up key '" +
                   std::string(key) + "'");
    }
    const Value* v = Find(key);
    if (!v)
    {
        std::string avail;
        for (const auto& kv : obj)
        {
            if (!avail.empty()) avail += ", ";
            avail += kv.first;
        }
        ThrowError(std::string(ctx) + ": missing required key '" + std::string(key) +
                   "'. Present keys: [" + avail + "]");
    }
    return *v;
}

int64_t Value::GetInt(std::string_view key, std::string_view ctx) const
{
    const Value& v = At(key, ctx);
    if (v.type != Type::Number)
    {
        ThrowError(std::string(ctx) + ": key '" + std::string(key) + "' is not a number");
    }
    return v.inum;
}

double Value::GetNum(std::string_view key, std::string_view ctx) const
{
    const Value& v = At(key, ctx);
    if (v.type != Type::Number)
    {
        ThrowError(std::string(ctx) + ": key '" + std::string(key) + "' is not a number");
    }
    return v.num;
}

std::string Value::GetStr(std::string_view key, std::string_view ctx) const
{
    const Value& v = At(key, ctx);
    if (v.type != Type::String)
    {
        ThrowError(std::string(ctx) + ": key '" + std::string(key) + "' is not a string");
    }
    return v.str;
}

bool Value::GetBool(std::string_view key, bool fallback, std::string_view ctx) const
{
    const Value* v = Find(key);
    if (!v) return fallback;
    if (v->type == Type::Bool)   return v->b;
    if (v->type == Type::Number) return (v->inum != 0);
    return fallback;
}

const Value& Value::Index(size_t i, std::string_view ctx) const
{
    if (type != Type::Array)
    {
        ThrowError(std::string(ctx) + ": expected a JSON array");
    }
    if (i >= arr.size())
    {
        ThrowError(std::string(ctx) + ": array index " + std::to_string(i) +
                   " out of range (size " + std::to_string(arr.size()) + ")");
    }
    return arr[i];
}

// ---------------------------------------------------------------------------
//  Recursive-descent parser
// ---------------------------------------------------------------------------
namespace
{

class Parser
{
public:
    Parser(std::string_view text, std::string_view ctx) : t_(text), ctx_(ctx) {}

    Value ParseDocument()
    {
        SkipWs();
        Value v = ParseValue(0);
        SkipWs();
        if (pos_ != t_.size())
        {
            Fail("trailing characters after top-level JSON value");
        }
        return v;
    }

private:
    static constexpr int kMaxDepth = 64;

    std::string_view t_;
    std::string_view ctx_;
    size_t pos_ = 0;

    [[noreturn]] void Fail(std::string_view msg) const
    {
        // Show a window of context around the error position.
        const size_t start = (pos_ > 24) ? pos_ - 24 : 0;
        const size_t end   = (pos_ + 24 < t_.size()) ? pos_ + 24 : t_.size();
        std::string around(t_.substr(start, end - start));
        for (char& c : around)
        {
            if (static_cast<unsigned char>(c) < 0x20) c = '.';
        }
        ThrowError(std::string(ctx_) + ": JSON error at byte " + std::to_string(pos_) +
                   ": " + std::string(msg) + " near \"" + around + "\"");
    }

    void SkipWs()
    {
        while (pos_ < t_.size())
        {
            const char c = t_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { ++pos_; }
            else { break; }
        }
    }

    char Peek() const { return (pos_ < t_.size()) ? t_[pos_] : '\0'; }

    bool Consume(char c)
    {
        if (pos_ < t_.size() && t_[pos_] == c) { ++pos_; return true; }
        return false;
    }

    void Expect(char c)
    {
        if (!Consume(c))
        {
            Fail(std::string("expected '") + c + "'");
        }
    }

    Value ParseValue(int depth)
    {
        if (depth > kMaxDepth) Fail("JSON nesting too deep");
        SkipWs();
        if (pos_ >= t_.size()) Fail("unexpected end of input");

        switch (Peek())
        {
        case '{': return ParseObjectBody(depth);
        case '[': return ParseArrayBody(depth);
        case '"':
        {
            Value v;
            v.type = Value::Type::String;
            v.str  = ParseString();
            return v;
        }
        case 't':
            ExpectLiteral("true");
            { Value v; v.type = Value::Type::Bool; v.b = true;  return v; }
        case 'f':
            ExpectLiteral("false");
            { Value v; v.type = Value::Type::Bool; v.b = false; return v; }
        case 'n':
            ExpectLiteral("null");
            { Value v; v.type = Value::Type::Null; return v; }
        default:
            return ParseNumber();
        }
    }

    void ExpectLiteral(std::string_view lit)
    {
        if (t_.substr(pos_, lit.size()) != lit)
        {
            Fail("invalid literal");
        }
        pos_ += lit.size();
    }

    Value ParseObjectBody(int depth)
    {
        Expect('{');
        Value v;
        v.type = Value::Type::Object;
        SkipWs();
        if (Consume('}')) return v;

        for (;;)
        {
            SkipWs();
            if (Peek() != '"') Fail("expected object key string");
            std::string key = ParseString();

            SkipWs();
            Expect(':');
            Value child = ParseValue(depth + 1);
            v.obj.emplace_back(std::move(key), std::move(child));

            SkipWs();
            if (Consume(',')) continue;
            if (Consume('}')) break;
            Fail("expected ',' or '}' in object");
        }
        return v;
    }

    Value ParseArrayBody(int depth)
    {
        Expect('[');
        Value v;
        v.type = Value::Type::Array;
        SkipWs();
        if (Consume(']')) return v;

        for (;;)
        {
            v.arr.push_back(ParseValue(depth + 1));
            SkipWs();
            if (Consume(',')) continue;
            if (Consume(']')) break;
            Fail("expected ',' or ']' in array");
        }
        return v;
    }

    // Appends a Unicode code point as UTF-8.
    static void AppendUtf8(std::string& out, uint32_t cp)
    {
        if (cp <= 0x7F)
        {
            out.push_back(static_cast<char>(cp));
        }
        else if (cp <= 0x7FF)
        {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        else if (cp <= 0xFFFF)
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

    uint32_t ParseHex4()
    {
        if (pos_ + 4 > t_.size()) Fail("truncated \\u escape");
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i)
        {
            const char c = t_[pos_ + i];
            v <<= 4;
            if      (c >= '0' && c <= '9') { v |= static_cast<uint32_t>(c - '0'); }
            else if (c >= 'a' && c <= 'f') { v |= static_cast<uint32_t>(c - 'a' + 10); }
            else if (c >= 'A' && c <= 'F') { v |= static_cast<uint32_t>(c - 'A' + 10); }
            else { Fail("invalid hex digit in \\u escape"); }
        }
        pos_ += 4;
        return v;
    }

    std::string ParseString()
    {
        Expect('"');
        std::string out;
        for (;;)
        {
            if (pos_ >= t_.size()) Fail("unterminated string");
            const char c = t_[pos_++];
            if (c == '"') break;
            if (c != '\\')
            {
                out.push_back(c);
                continue;
            }
            if (pos_ >= t_.size()) Fail("unterminated escape");
            const char e = t_[pos_++];
            switch (e)
            {
            case '"':  out.push_back('"');  break;
            case '\\': out.push_back('\\'); break;
            case '/':  out.push_back('/');  break;
            case 'b':  out.push_back('\b'); break;
            case 'f':  out.push_back('\f'); break;
            case 'n':  out.push_back('\n'); break;
            case 'r':  out.push_back('\r'); break;
            case 't':  out.push_back('\t'); break;
            case 'u':
            {
                uint32_t cp = ParseHex4();
                // Combine UTF-16 surrogate pairs.
                if (cp >= 0xD800 && cp <= 0xDBFF)
                {
                    if (pos_ + 1 < t_.size() && t_[pos_] == '\\' && t_[pos_ + 1] == 'u')
                    {
                        pos_ += 2;
                        const uint32_t lo = ParseHex4();
                        if (lo >= 0xDC00 && lo <= 0xDFFF)
                        {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        }
                        else
                        {
                            AppendUtf8(out, cp);
                            cp = lo;
                        }
                    }
                }
                AppendUtf8(out, cp);
                break;
            }
            default:
                Fail("invalid escape character");
            }
        }
        return out;
    }

    Value ParseNumber()
    {
        const size_t start = pos_;
        bool isFloat = false;

        if (Consume('-')) {}

        while (pos_ < t_.size())
        {
            const char c = t_[pos_];
            if (c >= '0' && c <= '9') { ++pos_; continue; }
            if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-')
            {
                // '+'/'-' are only part of an exponent.
                if ((c == '+' || c == '-') &&
                    !(pos_ > start && (t_[pos_ - 1] == 'e' || t_[pos_ - 1] == 'E')))
                {
                    break;
                }
                isFloat = true;
                ++pos_;
                continue;
            }
            break;
        }

        if (pos_ == start) Fail("expected a number");

        const std::string_view lit = t_.substr(start, pos_ - start);

        Value v;
        v.type = Value::Type::Number;

        if (!isFloat)
        {
            // Exact integer parse via from_chars (no locale, no double rounding).
            int64_t iv = 0;
            const auto* first = lit.data();
            const auto* last  = lit.data() + lit.size();
            const auto res = std::from_chars(first, last, iv);
            if (res.ec == std::errc{} && res.ptr == last)
            {
                v.inum = iv;
                v.num  = static_cast<double>(iv);
                return v;
            }
            // Integer literal too large for int64 -- fall through to double.
        }

        double dv = 0.0;
        const auto res = std::from_chars(lit.data(), lit.data() + lit.size(), dv);
        if (res.ec != std::errc{})
        {
            Fail("malformed number");
        }
        v.num  = dv;
        v.inum = static_cast<int64_t>(dv);
        return v;
    }
};

} // namespace

Value Parse(std::string_view text, std::string_view ctx)
{
    // Tolerate a UTF-8 BOM: some tokenizer.json exports carry one.
    if (text.size() >= 3 &&
        static_cast<unsigned char>(text[0]) == 0xEF &&
        static_cast<unsigned char>(text[1]) == 0xBB &&
        static_cast<unsigned char>(text[2]) == 0xBF)
    {
        text.remove_prefix(3);
    }
    Parser p(text, ctx);
    return p.ParseDocument();
}

Value ParseObject(std::string_view text, std::string_view ctx)
{
    Value v = Parse(text, ctx);
    if (!v.IsObject())
    {
        ThrowError(std::string(ctx) + ": top-level JSON value is not an object");
    }
    return v;
}

} // namespace opure::json