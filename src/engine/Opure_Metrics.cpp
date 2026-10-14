// ============================================================================
//  Opure_Metrics.cpp
//  Opure-VoxelEngine -- runtime self-monitoring and eval harness.
// ============================================================================

#include "engine/Opure_Metrics.h"
#include "engine/Opure_VoxelEngine.h"
#include "engine/Opure_Graph.h"
#include "core/Opure_Platform.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

namespace opure::engine {

// ---------------------------------------------------------------------------
//  MetricsCollector
// ---------------------------------------------------------------------------
MetricsCollector::MetricsCollector(VoxelEngine& engine)
    : engine_(engine)
{
    layerTimings_.resize(engine_.GraphModel().Config().numLayers, 0.0f);
    lastTokenTimeMs_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

MetricsSnapshot MetricsCollector::Snapshot()
{
    MetricsSnapshot snap;
    snap.timestampMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    // Per-layer timing (would need engine to expose this; placeholder)
    snap.layerTimesMs = layerTimings_;

    // KV occupancy
    const auto& plan = engine_.GraphModel().Plan();
    const auto& cfg = engine_.GraphModel().Config();
    snap.kvOccupancy = cfg.maxPositions > 0
        ? static_cast<float>(engine_.CurrentPosition()) / cfg.maxPositions
        : 0.0f;

    // Logit entropy & norm
    snap.logitEntropy = ComputeLogitEntropy();
    snap.logitNorm = ComputeLogitNorm();

    // Tokens/sec (instantaneous)
    uint64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    if (lastTokenTimeMs_ > 0 && now > lastTokenTimeMs_) {
        snap.tokensPerSec = 1000.0f / static_cast<float>(now - lastTokenTimeMs_);
    }
    lastTokenTimeMs_ = now;

    // Memory pressure
    snap.memoryPressure = plan.availableBytes > 0
        ? static_cast<float>(plan.totalBytes) / plan.availableBytes
        : 0.0f;

    // Speculative acceptance. -1 means "no speculative decoder attached", which
    // is the honest value: the SpeculativeVerify kernels compile but nothing
    // calls them yet. 0.0 would read as "every draft token rejected".
    snap.speculativeAcceptance = -1.0f;

    // Max attention score
    snap.maxAttnScore = ComputeMaxAttnScore();

    return snap;
}

float MetricsCollector::ComputeLogitEntropy()
{
    // Real: reads the logit vector back and derives H = -sum(p log p) over the
    // softmax. The readback is what makes this expensive, which is why the
    // engine keeps it out of DecodeStep and the snapshot pays for it on demand.
    if (engine_.LastLogitBank() != 0 || engine_.lastLogitEntropy_ < 0.0f)
        engine_.RefreshLogitStats(engine_.LastLogitBank());
    return engine_.lastLogitEntropy_;
}

float MetricsCollector::ComputeLogitNorm()
{
    // L2 norm of logits vector. Distribution shift detection.
    // RefreshLogitStats fills both this and entropy, so a snapshot costs one
    // readback for both, not one each.
    if (engine_.LastLogitBank() != 0 || engine_.lastLogitNorm_ < 0.0f)
        engine_.RefreshLogitStats(engine_.LastLogitBank());
    return engine_.lastLogitNorm_;
}

float MetricsCollector::ComputeMaxAttnScore() const
{
    // Maximum attention weight in the last attention pass.
    // Sharp attention = high confidence in specific tokens.
    return engine_.lastMaxAttnScore_;
}

// ---------------------------------------------------------------------------
//  EvalHarness
// ---------------------------------------------------------------------------
EvalHarness::EvalHarness(const EvalConfig& cfg)
    : cfg_(cfg)
{
    // Standard benchmark prompts covering different domains.
    prompts_ = {
        "The capital of France is",
        "def fibonacci(n):",
        "Once upon a time, there was a",
        "The theory of relativity states that",
        "In machine learning, overfitting occurs when",
        "The quadratic formula is",
        "Shakespeare wrote the play",
        "The mitochondria is the",
        "Quantum entanglement means that",
        "The derivative of x^2 is",
    };
}

EvalResult EvalHarness::Run(VoxelEngine& engine)
{
    EvalResult result;
    result.prompts.reserve(prompts_.size());

    // Load baseline if exists
    std::vector<float> baselineTPS;
    std::vector<float> baselineEntropy;
    if (!cfg_.baselinePath.empty()) {
        // TODO: parse JSON baseline
    }

    for (size_t i = 0; i < std::min<size_t>(prompts_.size(), cfg_.numPrompts); ++i) {
        const std::string& prompt = prompts_[i];
        
        engine::EngineOptions opts;
        opts.enablePlasticity = false;

        auto start = std::chrono::high_resolution_clock::now();
        auto genResult = engine.Generate(
            engine.Tokenizer().Encode(prompt), opts);
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        float sec = std::chrono::duration<float>(elapsed).count();
        float tps = sec > 0 ? genResult.generated.size() / sec : 0;

        EvalResult::PromptResult pr;
        pr.prompt = prompt;
        pr.tokensPerSec = tps;
        pr.logitEntropy = 0.0f; // Would need metrics collector

        if (!baselineTPS.empty() && i < baselineTPS.size()) {
            pr.drift = (tps - baselineTPS[i]) / baselineTPS[i];
            pr.regressed = pr.drift < -cfg_.speedDriftTol;
        }

        result.prompts.push_back(pr);
    }

    // Aggregate
    float sumTPS = 0, sumEnt = 0;
    for (const auto& pr : result.prompts) {
        sumTPS += pr.tokensPerSec;
        sumEnt += pr.logitEntropy;
        result.worstDrift = std::max(result.worstDrift, std::abs(pr.drift));
    }
    result.meanTokensPerSec = sumTPS / result.prompts.size();
    result.meanEntropy = sumEnt / result.prompts.size();

    result.passed = result.worstDrift < cfg_.speedDriftTol;
    result.summary = "mean_tps=" + std::to_string(result.meanTokensPerSec) +
                     " worst_drift=" + std::to_string(result.worstDrift);

    return result;
}

void EvalHarness::GenerateBaseline(VoxelEngine& engine)
{
    // Run eval and save as baseline JSON
    std::ofstream out(cfg_.outputPath);
    if (!out) return;

    out << "{\n  \"baseline\": [\n";
    for (size_t i = 0; i < std::min<size_t>(prompts_.size(), cfg_.numPrompts); ++i) {
        engine::EngineOptions opts;
        opts.enablePlasticity = false;

        auto start = std::chrono::high_resolution_clock::now();
        auto genResult = engine.Generate(
            engine.Tokenizer().Encode(prompts_[i]), opts);
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        float sec = std::chrono::duration<float>(elapsed).count();
        float tps = sec > 0 ? genResult.generated.size() / sec : 0;

        out << "    {\"prompt\": \"" << prompts_[i] << "\", \"tps\": " << tps << "}";
        if (i + 1 < cfg_.numPrompts) out << ",";
        out << "\n";
    }
    out << "  ]\n}\n";
}

} // namespace opure::engine