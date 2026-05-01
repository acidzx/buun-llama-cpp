#pragma once

#include "pflash-loader.h"
#include <vector>
#include <cstdint>

struct FlashPrefillConfig;
struct FlashPrefillBuffers;

// Run Qwen3-0.6B forward with FlashPrefill attention and extract tail scoring.
// Returns running_max[n_lookahead][seq_len] — per-token importance scores.
// Caller must free the returned vector.
struct pflash_scorer_result {
	std::vector<float> running_max; // [n_lookahead * seq_len]
	int n_lookahead;
	int seq_len;
};

// Run the scorer forward pass over the full prompt.
// token_ids: input prompt token IDs
// model: loaded pflash_model (weights on GPU)
// fp_cfg: FlashPrefill configuration
// gpu_device: CUDA device index
pflash_scorer_result pflash_score(
	const std::vector<int32_t> & token_ids,
	const pflash_model & model,
	const FlashPrefillConfig & fp_cfg,
	int gpu_device = 0);
