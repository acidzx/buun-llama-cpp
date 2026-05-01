#pragma once

#include <vector>
#include <cstdint>
#include <string>

struct pflash_score_config {
	float keep_ratio  = 0.05f;  // fraction of chunks to keep
	int   chunk_size  = 32;     // tokens per chunk for scoring
	int   pool_kernel = 13;     // avgpool smoothing window
};

struct pflash_span {
	int start; // inclusive token index
	int end;   // exclusive token index
};

// Given raw per-token importance scores from the scorer, select the most
// important spans and return them as a list of (start, end) ranges.
std::vector<pflash_span> pflash_select_spans(
	const float * scores,     // [seq_len] per-token importance
	int seq_len,
	const pflash_score_config & cfg);

// Full pipeline: scores -> smooth -> chunk -> top-K -> merge spans -> text roundtrip.
// Returns compressed token IDs ready for target prefill.
// scorer_vocab: tokenizer model path for the scorer (Qwen3-0.6B)
// target_vocab: tokenizer model path for the target (Qwen3.5/3.6)
// If both vocabs are the same (same family), skips the text roundtrip.
std::vector<int32_t> pflash_compress_tokens(
	const float * running_max,  // [n_lookahead * seq_len]
	int n_lookahead,
	int seq_len,
	const int32_t * original_ids,
	int n_original,
	const pflash_score_config & cfg,
	const std::string & delimiter = " [...] ");
