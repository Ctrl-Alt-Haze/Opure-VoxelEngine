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
        // Device's constructor is defaulted; Initialize() is what enumerates the
        // adapter, creates the D3D12 device/queue/root signature and records the
        // VRAM budget. Without it dedicatedVram_ stays 0 and every model is
        // rejected by the VRAM admission gate as "0 MiB available".
        dev_->Initialize();
        gpuName_ = dx12::DeviceName();
        vramTotal_ = dev_->DedicatedVideoMemory();

        lib_ = std::make_unique<shaders::Library>();
        lib_->Initialize(*dev_);

        // VoxelEngine owns the graph, every GPU buffer, the weight upload and the
        // tokenizer. Standing up a second copy of all of that here allocated a
        // second full weight buffer (~2 GB on SmolLM) that nothing ever read, on
        // top of the engine's own -- which is most of why the TUI could not get
        // off the ground. The model description is read back off the engine's
        // graph afterwards instead of from a private one.
        engine::EngineOptions opts;
        opts.enablePlasticity = false;
        try {
            engine_.Initialize(modelDir, opts, dev_.get(), lib_.get());
        } catch (const std::exception& e) {
            std::cerr << "Engine init failed: " << e.what() << "\n";
            return false;
        }

        const auto& cfg = engine_.GraphModel().Config();
        std::cout << "Loaded: " << cfg.architecture << "  "
                  << cfg.numLayers << "L  "
                  << cfg.hiddenSize << "H  "
                  << cfg.numHeads << "h/" << cfg.numKvHeads << "kv  "
                  << cfg.HeadDim() << "hd\n";
        std::cout << "VRAM: " << (engine_.GraphModel().Plan().totalBytes >> 20) << " MiB / "
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
    std::unique_ptr<shaders::Library> lib_;
    // The engine owns the graph, the GPU buffers and the tokenizer; it exposes
    // them rather than the app duplicating them. Tokenizer() is the accessor.
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
        std::vector<uint32_t> tokens = engine_.Tokenizer().Encode(prompt);
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
            if (tok == engine_.Tokenizer().SpecialId("<|endoftext|>")) break;

            std::string tokenStr = engine_.Tokenizer().Decode(tok);

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
