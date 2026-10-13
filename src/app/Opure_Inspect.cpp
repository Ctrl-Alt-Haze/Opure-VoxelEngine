// ============================================================================
//  Opure_Inspect.cpp
//  Opure-VoxelEngine -- model directory inspector.
// ============================================================================
//  Usage: opure_inspect <model-directory>
//  Prints a single JSON object describing what the engine sees in the directory
//  and whether the model is supported, with reasons if not.
// ============================================================================

#include "engine/Opure_Graph.h"
#include "engine/Opure_ModelSpec.h"
#include "core/Opure_Platform.h"
#include "core/Opure_Json.h"

#include <cstdio>
#include <cwchar>

namespace opure {

int Inspect(int argc, const char* argv[])
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: opure_inspect <model-dir>\n");
        return 1;
    }

    // Wide path for the file APIs
    std::wstring modelDir;
    {
        const size_t len = std::strlen(argv[1]) + 1;
        std::vector<wchar_t> buf(len);
        std::mbstowcs(buf.data(), argv[1], len);
        modelDir = buf.data();
    }

    graph::ModelSpec spec;
    safetensors::WeightStreamer stream;
    graph::ModelConfig cfg;

    // Parse config.json
    std::string cfgText;
    {
        const std::wstring cfgPath = graph::JoinPath(modelDir, L"config.json");
        cfgText = graph::ReadSmallFile(cfgPath);
    }
    cfg = graph::ModelConfig::Parse(cfgText);

    // Resolve spec
    stream.Open(graph::JoinPath(modelDir, L"model.safetensors"));
    spec.Resolve(cfg, stream);
    spec.Check(cfg);

    // ---- Emit JSON --------------------------------------------------------
    auto esc = [](std::string_view s) -> std::string {
        std::string out;
        out.reserve(s.size() * 2);
        for (char c : s) {
            switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c;
            }
        }
        return out;
    };

    std::printf("{\n");
    std::printf("  \"architecture\": \"%s\",\n", esc(cfg.architecture).c_str());
    std::printf("  \"hidden_size\": %u,\n", cfg.hiddenSize);
    std::printf("  \"num_layers\": %u,\n", cfg.numLayers);
    std::printf("  \"num_heads\": %u,\n", cfg.numHeads);
    std::printf("  \"num_kv_heads\": %u,\n", cfg.numKvHeads);
    std::printf("  \"head_dim\": %u,\n", cfg.HeadDim());
    std::printf("  \"intermediate_size\": %u,\n", cfg.intermediateSize);
    std::printf("  \"vocab_size\": %u,\n", cfg.vocabSize);
    std::printf("  \"max_position_embeddings\": %u,\n", cfg.maxPositions);
    std::printf("  \"hidden_act\": \"%s\",\n", esc(cfg.hiddenAct).c_str());
    std::printf("  \"rope_theta\": %f,\n", cfg.ropeTheta);
    std::printf("  \"rope_scaling_type\": \"%s\",\n", esc(cfg.ropeScalingType).c_str());
    std::printf("  \"rope_scaling_factor\": %f,\n", cfg.ropeScalingFactor);
    std::printf("  \"tie_word_embeddings\": %s,\n", cfg.tieWordEmbeddings ? "true" : "false");
    std::printf("  \"layer_prefix\": \"%s\",\n", esc(spec.LayerPrefix()).c_str());

    // Bookends
    const auto& emb = spec.Embed();
    const auto& fn  = spec.FinalNorm();
    const auto& lmh = spec.LmHead();

    std::printf("  \"embed\": { \"name\": \"%s\", \"dtype\": \"%s\", \"shape\": [%s] },\n",
                esc(emb.name).c_str(), esc(safetensors::DtypeName(emb.dtype)).c_str(),
                [&emb]() -> std::string {
                    std::string s;
                    for (size_t i = 0; i < emb.shape.size(); ++i) {
                        if (i) s += ", ";
                        s += std::to_string(emb.shape[i]);
                    }
                    return s;
                }().c_str());

    std::printf("  \"final_norm\": { \"name\": \"%s\", \"dtype\": \"%s\", \"shape\": [%s] },\n",
                esc(fn.name).c_str(), esc(safetensors::DtypeName(fn.dtype)).c_str(),
                [&fn]() -> std::string {
                    std::string s;
                    for (size_t i = 0; i < fn.shape.size(); ++i) {
                        if (i) s += ", ";
                        s += std::to_string(fn.shape[i]);
                    }
                    return s;
                }().c_str());

    std::printf("  \"lm_head\": { \"name\": \"%s\", \"dtype\": \"%s\", \"shape\": [%s], \"tied\": %s },\n",
                esc(lmh.name).c_str(), esc(safetensors::DtypeName(lmh.dtype)).c_str(),
                [&lmh]() -> std::string {
                    std::string s;
                    for (size_t i = 0; i < lmh.shape.size(); ++i) {
                        if (i) s += ", ";
                        s += std::to_string(lmh.shape[i]);
                    }
                    return s;
                }().c_str(), cfg.tieWordEmbeddings ? "true" : "false");

    // Issues
    const auto& issues = spec.Issues();
    std::printf("  \"supported\": %s,\n", spec.Supported() ? "true" : "false");
    std::printf("  \"issues\": [\n");
    for (size_t i = 0; i < issues.size(); ++i) {
        const auto& is = issues[i];
        std::printf("    { \"reason\": \"%s\", \"fix\": \"%s\", \"fatal\": %s }%s\n",
                    esc(is.reason).c_str(), esc(is.fix).c_str(),
                    is.fatal ? "true" : "false", i + 1 == issues.size() ? "" : ",");
    }
    std::printf("  ]\n");
    std::printf("}\n");

    return spec.Supported() ? 0 : 2;
}

} // namespace opure

int wmain(int argc, const wchar_t* argv[])
{
    // Convert wargv to narrow for the implementation
    std::vector<std::string> narrow(argc);
    for (int i = 0; i < argc; ++i) {
        const size_t len = std::wcslen(argv[i]) + 1;
        std::vector<char> buf(len);
        std::wcstombs(buf.data(), argv[i], len);
        narrow[i] = buf.data();
    }
    std::vector<const char*> cargv(argc);
    for (int i = 0; i < argc; ++i) cargv[i] = narrow[i].c_str();
    return opure::Inspect(argc, cargv.data());
}