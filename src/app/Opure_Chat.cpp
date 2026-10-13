// ============================================================================
//  Opure_Chat.cpp
//  Opure-VoxelEngine -- interactive TUI chat client.
// ============================================================================
//  Usage: opure_chat [--model <path>]
//  Loads the engine directly. Streams tokens with live t/s readout.
// ---------------------------------------------------------------------------

#include "engine/Opure_VoxelEngine.h"
#include "engine/Opure_Graph.h"
#include "model/Opure_Tokenizer.h"
#include "core/Opure_Dx12.h"
#include "core/Opure_Platform.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace opure {

class ChatEngine {
public:
    bool Init(const std::wstring& modelDir) {
        dev_ = std::make_unique<dx12::Device>();
        gpuName_ = dx12::DeviceName();
        vramTotal_ = dev_->DedicatedVideoMemory();

        graph_ = std::make_unique<graph::Graph>();
        graph_->LoadConfigOnly(modelDir, vramTotal_);

        if (!graph_->Plan().fits) {
            std::cerr << "Model requires " << (graph_->Plan().totalBytes >> 20)
                      << " MiB but only " << (vramTotal_ >> 20) << " MiB available.\n";
            return false;
        }

        weights_.Create(*dev_, dx12::kSlotWeights, graph_->Plan().totalBytes, true);
        activations_.Create(*dev_, 1, graph_->Plan().actBytes, true);
        kvCache_.Create(*dev_, 3, graph_->Plan().kvBytes, true);
        params_.Create(*dev_, 5, dx12::kParamBytes, true);

        lib_ = std::make_unique<shaders::Library>();
        lib_->Initialize(*dev_);

        graph_->UploadWeights(*dev_, weights_);

        tokenizer_ = std::make_unique<tokenizer::BpeTokenizer>();
        tokenizer_->LoadFromFile(modelDir + L"\\tokenizer.json");

        engine::EngineOptions opts;
        opts.enablePlasticity = false;
        engine_.Initialize(modelDir, opts, dev_.get(), lib_.get());

        std::cout << "Loaded: " << graph_->Config().architecture << "  "
                  << graph_->Config().numLayers << "L  "
                  << graph_->Config().hiddenSize << "H  "
                  << graph_->Config().numHeads << "h/" << graph_->Config().numKvHeads << "kv  "
                  << graph_->Config().HeadDim() << "hd\n";
        std::cout << "VRAM: " << (graph_->Plan().totalBytes >> 20) << " MiB / "
                  << (vramTotal_ >> 20) << " MiB\n";
        return true;
    }

    void Chat(float temp = 0.8f, float topP = 0.95f, uint32_t topK = 50, float repPen = 1.1f) {
        std::string prompt;
        std::cout << "\nOpure Chat  (Ctrl+C to exit, /help for commands)\n";

        while (true) {
            std::cout << "\n> ";
            if (!std::getline(std::cin, prompt)) break;
            if (prompt.empty()) continue;

            if (prompt == "/help") {
                PrintHelp();
                continue;
            }
            if (prompt == "/quit" || prompt == "/exit") {
                break;
            }

            Generate(prompt);
        }
    }

private:
    std::unique_ptr<dx12::Device> dev_;
    dx12::GpuBuffer weights_;
    dx12::GpuBuffer activations_;
    dx12::GpuBuffer kvCache_;
    dx12::GpuBuffer params_;
    std::unique_ptr<shaders::Library> lib_;
    std::unique_ptr<graph::Graph> graph_;
    std::unique_ptr<tokenizer::BpeTokenizer> tokenizer_;
    engine::VoxelEngine engine_;

    std::string gpuName_;
    uint64_t vramTotal_ = 0;

    void PrintHelp() {
        std::cout << R"(
Commands:
  /help              Show this help
  /quit              Exit
)";
    }

    void Generate(const std::string& prompt) {
        std::vector<uint32_t> tokens = tokenizer_->Encode(prompt);
        if (tokens.empty()) {
            std::cerr << "Prompt tokenized to empty.\n";
            return;
        }

        engine::EngineOptions opts;
        opts.enablePlasticity = false;

        auto start = std::chrono::high_resolution_clock::now();
        auto result = engine_.Generate(tokens, opts);
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        float sec = std::chrono::duration<float>(elapsed).count();

        // Stream the generated tokens
        bool firstToken = true;
        for (uint32_t tok : result.generated) {
            if (tok == tokenizer_->SpecialId("<|endoftext|>")) break;

            std::string tokenStr = tokenizer_->Decode(tok);

            if (firstToken) {
                firstToken = false;
                std::cout << "\033[2K\r";
            }

            std::cout << tokenStr << std::flush;
        }

        std::cout << "\n";
        std::cout << "\033[90m[" << result.generated.size() << " tokens  "
                  << std::fixed << std::setprecision(1) << (sec > 0 ? result.generated.size() / sec : 0)
                  << " tok/s]\033[0m\n";
    }
};

int RunChat(int argc, char** argv) {
    std::string modelPath = ".\\models";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) modelPath = argv[++i];
        else if (arg == "--help") {
            std::cout << "Usage: opure_chat [--model <path>]\n";
            return 0;
        }
    }

    std::wstring wpath(modelPath.begin(), modelPath.end());
    ChatEngine engine;
    if (!engine.Init(wpath)) return 1;

    engine.Chat();
    return 0;
}

} // namespace opure

int main(int argc, char** argv) {
    return opure::RunChat(argc, argv);
}