// ============================================================================
//  Opure_Tokenizer.h
//  Opure-VoxelEngine -- byte-level BPE, matching the HuggingFace
//  `tokenizers` implementation for this model's tokenizer.json.
//
//  WHAT THIS IMPLEMENTS
//  --------------------
//  The tokenizer.json in models/ declares:
//
//      "normalizer"      : null
//      "pre_tokenizer"   : Sequence[ Digits(individual_digits=true),
//                                      ByteLevel(add_prefix_space=false,
//                                                use_regex=true) ]
//      "post_processor"  : null
//      "decoder"         : ByteLevel
//      "model"           : { "vocab": {...}, "merges": ["A B", ...] }
//
//  Three things therefore have to happen to turn text into ids:
//
//    1. PRE-TOKENIZE. `use_regex: true` means the GPT-2 split pattern, which
//       separates words / numbers / punctuation / whitespace runs and keeps a
//       leading space glued to the following word. `Digits(
//       individual_digits=true)` then splits any run of digits into one piece
//       per digit BEFORE the byte-level pass, so "2024" is four tokens, not
//       one. Digits runs first in the sequence, so this is applied first.
//
//    2. BYTE-LEVEL ENCODE. Each pre-token is mapped byte-by-byte through the
//       GPT-2 `bytes_to_unicode` table, which is NOT identity: it exists so
//       that every byte becomes a printable, non-whitespace UTF-8 character.
//       Space (0x20) becomes U+0120, and bytes >= 0x21 mostly keep their
//       codepoint while 0x00-0x20, 0x7F-0xA0 and the 0x80-0xFF block are
//       shifted. Decoding is the inverse table.
//
//    3. BPE. Each pre-token starts as one symbol per byte. Merges are applied
//       in rank order (rank = index in the merges array) until no merge
//       applies. This is the standard greedy-lowest-rank loop, identical to
//       the reference implementation.
//
//  WHY NOT USE <|endoftext|> HANDLING AUTOMATICALLY
//  -------------------------------------------------
//  post_processor is null and added_tokens are not applied during encoding in
//  this configuration, so nothing is injected automatically. Special-token
//  strings are still recognised and emitted verbatim by Encode() because
//  splitting them through BPE would produce garbage ids.
// ============================================================================
#pragma once

#include "core/Opure_Platform.h"

#include <unordered_map>

namespace opure::json { class Value; }

namespace opure::tokenizer
{

// ---------------------------------------------------------------------------
//  Token
// ---------------------------------------------------------------------------
struct Token
{
    uint32_t    id   = 0;
    std::string text;      // surface form, as it appears in the input
    uint32_t    offset = 0; // byte offset into the source string
};

// ---------------------------------------------------------------------------
//  BpeTokenizer
// ---------------------------------------------------------------------------
class BpeTokenizer
{
public:
    BpeTokenizer() = default;

    // Loads tokenizer.json. Throws opure::Error on malformed or mismatched data.
    void LoadFromFile(const std::wstring& path);
    void LoadFromMemory(std::string_view json, std::string_view ctx);

    [[nodiscard]] uint32_t VocabSize()     const { return vocabSize_; }
    [[nodiscard]] uint32_t SpecialCount()  const { return static_cast<uint32_t>(specials_.size()); }

    // Encodes text to token ids. Special-token strings (e.g. <|endoftext|>)
    // present in the input are matched first and passed through verbatim.
    [[nodiscard]] std::vector<uint32_t> Encode(std::string_view text) const;

    // Encodes with surface forms and offsets, for inspection and tests.
    [[nodiscard]] std::vector<Token> EncodeDetailed(std::string_view text) const;

    // Decodes ids back to text. Byte-level inverse mapping is applied.
    [[nodiscard]] std::string Decode(const std::vector<uint32_t>& ids) const;
    [[nodiscard]] std::string Decode(uint32_t id) const;

    // Id of a single surface string, or UINT32_MAX if not a known token.
    [[nodiscard]] uint32_t IdOf(std::string_view piece) const;

    // Id of an added/special token by its literal content.
    [[nodiscard]] uint32_t SpecialId(std::string_view content) const;

    // True when the text contains a special token (used to pick a stop).
    [[nodiscard]] bool ContainsSpecial(std::string_view text) const;

    // The GPT-2 regex used by ByteLevel(use_regex=true), exposed so tests can
    // assert the split behaviour directly.
    [[nodiscard]] static std::vector<std::string> PreTokenize(std::string_view text);

    // The byte-level byte->codepoint table (identity except where it must
    // differ), exposed for tests.
    [[nodiscard]] static const std::vector<uint32_t>& ByteToUnicode();

private:
    void BuildByteTables();
    void LoadVocabAndMerges(const json::Value& modelRoot, std::string_view ctx);

    // Reverses the byte-level alphabet: a vocab piece back to raw bytes, which
    // is the actual UTF-8 text the token stands for.
    [[nodiscard]] std::string DecodeInternal(std::string_view piece) const;

    // Applies the merge loop to one already-byte-level-encoded piece.
    void BpePiece(std::string_view bytes, std::vector<uint32_t>& outIds) const;

    // vocab surface form -> id
    std::unordered_map<std::string, uint32_t> vocab_;

    // merge "left\0right" -> rank. Ordered by rank, so a flat map with the
    // rank as value and a "best rank wins" scan is both correct and fast.
    struct Merge { std::string left, right; uint32_t rank; };
    std::vector<Merge> merges_;

    // Added/special tokens, longest first so that greedy matching prefers the
    // longest literal (e.g. <|endoftext|> before <|end|).
    struct Special { std::string content; uint32_t id; };
    std::vector<Special> specials_;

    std::vector<std::string> idToPiece_;

    uint32_t vocabSize_ = 0;
    bool     loaded_    = false;

    // byte -> utf8 codepoint used to form the byte-level alphabet
    std::vector<uint32_t> byteToUni_;
    // utf8 codepoint -> byte
    std::vector<uint8_t>  uniToByte_;
};

} // namespace opure::tokenizer