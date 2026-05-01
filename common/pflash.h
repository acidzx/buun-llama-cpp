#pragma once

#include "pflash-score.h"
#include <string>
#include <vector>
#include <cstdint>

struct FlashPrefillConfig;

struct pflash_config {
	std::string scorer_path;       // path to scorer GGUF (e.g., Qwen3-0.6B-BF16.gguf)
	int   min_tokens    = 8192;    // minimum prompt length to trigger PFlash
	float keep_ratio    = 0.05f;   // fraction of chunks to keep
	float alpha         = 0.12f;   // FlashPrefill block selection threshold
	int   gpu_device    = 0;       // CUDA device index
};

// Compress a long prompt via speculative prefill scoring.
// Returns compressed token IDs suitable for target prefill.
// If prompt is shorter than config.min_tokens, returns original tokens unchanged.
//
// This function handles the full pipeline:
// 1. Park target/drafter weights (if needed for memory)
// 2. Load scorer model
// 3. Run scorer forward with FlashPrefill attention
// 4. Score token importance and select top spans
// 5. Free scorer
// 6. Unpark target/drafter weights
//
// model_tgt_path, model_dft_path: paths to target/drafter GGUFs for unparking.
// Set to empty string if parking is not needed (e.g., enough VRAM).
std::vector<int32_t> pflash_compress(
	const std::vector<int32_t> & prompt_tokens,
	const pflash_config & cfg);

// Check if PFlash is enabled (scorer path is set).
bool pflash_enabled(const pflash_config & cfg);
