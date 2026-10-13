// ============================================================================
//  Opure_Server.cpp
//  Opure-VoxelEngine -- simple HTTP server for UE5 plugin.
// ============================================================================
//  Usage: opure_server [--port N] [--model <path>]
//  Placeholder: serves /health and /generate (POST JSON) over HTTP.
//  Full implementation when UE5 integration lands.
// ---------------------------------------------------------------------------

#include "engine/Opure_VoxelEngine.h"
#include "engine/Opure_Graph.h"
#include "model/Opure_Tokenizer.h"
#include "core/Opure_Dx12.h"
#include "core/Opure_Platform.h"

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

namespace opure {

int RunServer(int argc, char** argv) {
    uint16_t port = 8080;
    std::string modelPath = ".\\models";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) port = static_cast<uint16_t>(std::stoi(argv[++i]));
        else if (arg == "--model" && i + 1 < argc) modelPath = argv[++i];
        else if (arg == "--help") {
            std::cout << "Usage: opure_server [--port N] [--model <path>]\n";
            return 0;
        }
    }

    std::cout << "Opure HTTP server placeholder.\n";
    std::cout << "Port: " << port << "  Model: " << modelPath << "\n";
    std::cout << "\n";
    std::cout << "This is a stub. Full HTTP server with /health, /generate endpoints\n";
    std::cout << "will be implemented when the UE5 plugin is developed.\n";
    std::cout << "For local inference, use: opure_chat --model <path>\n";

    // Keep alive for now
    std::cout << "\nPress Enter to exit...\n";
    std::cin.get();
    return 0;
}

} // namespace opure

int main(int argc, char** argv) {
    return opure::RunServer(argc, argv);
}