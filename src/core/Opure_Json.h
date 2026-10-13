// ============================================================================
//  Opure_Json.h
//  Opure-VoxelEngine -- minimal, allocation-conscious JSON reader.
//
//  Used for three things, all of which are large-but-simple documents:
//    * safetensors 8-byte length-prefixed header (object of tensor descriptors)
//    * tokenizer.json (nested: model.vocab object + model.merges array)
//    * config.json / generation_config.json (flat scalar fields)
//
//  Deliberately not a general-purpose library: no comments, no trailing commas,
//  no duplicate-key merging, no unicode escapes beyond \uXXXX (surrogates
//  handled, because tokenizer vocabularies contain them).
// ============================================================================
#pragma once

#include "Opure_Platform.h"

namespace opure::json
{

class Value
{
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;

    // Scalars. Numbers are kept as both double and a pre-parsed int64 because
    // safetensors offsets are integers that exceed float53 precision when a
    // model is larger than 8 TB, and we want exact 64-bit offsets.
    bool                     b = false;
    double                   num = 0.0;
    int64_t                  inum = 0;
    std::string              str;

    std::vector<Value>       arr;
    // Object members in document order (stable iteration matters for safetensors
    // header reproducibility).
    std::vector<std::pair<std::string, Value>> obj;

    [[nodiscard]] bool IsNull()   const { return type == Type::Null; }
    [[nodiscard]] bool IsObject() const { return type == Type::Object; }
    [[nodiscard]] bool IsArray()  const { return type == Type::Array; }
    [[nodiscard]] bool IsString() const { return type == Type::String; }
    [[nodiscard]] bool IsNumber() const { return type == Type::Number; }
    [[nodiscard]] bool IsBool()   const { return type == Type::Bool; }

    [[nodiscard]] const Value* Find(std::string_view key) const;

    // Typed accessors with a context string used in the thrown error.
    [[nodiscard]] const Value& At(std::string_view key, std::string_view ctx) const;
    [[nodiscard]] int64_t     GetInt(std::string_view key, std::string_view ctx) const;
    [[nodiscard]] double      GetNum(std::string_view key, std::string_view ctx) const;
    [[nodiscard]] std::string GetStr(std::string_view key, std::string_view ctx) const;
    [[nodiscard]] bool        GetBool(std::string_view key, bool fallback, std::string_view ctx) const;

    [[nodiscard]] const Value& Index(size_t i, std::string_view ctx) const;
};

// Parses a complete JSON document. Throws opure::Error with a byte offset on
// malformed input.
Value Parse(std::string_view text, std::string_view ctx);

// Parses and expects a top-level object, returning it directly.
Value ParseObject(std::string_view text, std::string_view ctx);

} // namespace opure::json