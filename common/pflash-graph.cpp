// PFlash scorer forward pass: Qwen3-0.6B with FlashPrefill attention.
// Runs the full transformer forward, then scores token importance via
// tail Q@K^T attention analysis across all layers.

#include "pflash-graph.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

// FlashPrefill CUDA interface (implemented in flashprefill.cu)
extern "C" {
#include "ggml-cuda/flashprefill.cuh"
}

static constexpr int N_LOOKAHEAD = 8;
static constexpr int CHUNK_S     = 32768; // chunk size for ggml graph eval

// RMS norm: out = x * (w / rms(x))
static ggml_tensor * build_rms_norm(
		ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, float eps = 1e-6f) {
	x = ggml_rms_norm(ctx, x, eps);
	return ggml_mul(ctx, x, w);
}

// SwiGLU FFN: down(silu(gate(x)) * up(x))
static ggml_tensor * build_ffn(
		ggml_context * ctx,
		ggml_tensor * x,
		ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down) {
	ggml_tensor * g = ggml_mul_mat(ctx, gate, x);
	ggml_tensor * u = ggml_mul_mat(ctx, up, x);
	g = ggml_silu(ctx, g);
	g = ggml_mul(ctx, g, u);
	return ggml_mul_mat(ctx, down, g);
}

pflash_scorer_result pflash_score(
		const std::vector<int32_t> & token_ids,
		const pflash_model & model,
		const FlashPrefillConfig & fp_cfg,
		int gpu_device) {

	const int S = (int)token_ids.size();
	const int n_layers   = model.n_layers;
	const int n_embd     = model.n_embd;
	const int n_heads    = model.n_heads;
	const int n_kv_heads = model.n_kv_heads;
	const int d_head     = model.d_head;
	const float scale    = 1.0f / sqrtf((float)d_head);

	pflash_scorer_result result;
	result.n_lookahead = N_LOOKAHEAD;
	result.seq_len = S;
	result.running_max.assign(N_LOOKAHEAD * S, -1e30f);

	fprintf(stderr, "pflash: scoring %d tokens across %d layers\n", S, n_layers);

	// allocate FlashPrefill scratch buffers
	FlashPrefillBuffers fp_bufs = flash_prefill_alloc(S, n_heads, n_kv_heads, fp_cfg.block_size);

	// allocate persistent K cache for all layers (needed for tail scoring)
	// K shape per layer: [S, n_kv_heads, d_head] in BF16
	const size_t k_layer_bytes = (size_t)S * n_kv_heads * d_head * sizeof(uint16_t); // bf16 = 2 bytes
	std::vector<void *> d_K_cache(n_layers, nullptr);
	for (int l = 0; l < n_layers; l++) {
		cudaMalloc(&d_K_cache[l], k_layer_bytes);
	}

	// allocate working buffers on GPU
	void * d_hidden = nullptr;  // [S, n_embd] BF16 — current hidden state
	void * d_Q = nullptr;       // [S, n_heads, d_head] BF16
	void * d_K = nullptr;       // [S, n_kv_heads, d_head] BF16 (current layer)
	void * d_V = nullptr;       // [S, n_kv_heads, d_head] BF16
	void * d_attn_out = nullptr;// [S, n_heads, d_head] BF16

	const size_t hidden_bytes = (size_t)S * n_embd * 2;
	const size_t qo_bytes     = (size_t)S * n_heads * d_head * 2;
	const size_t kv_bytes     = (size_t)S * n_kv_heads * d_head * 2;

	cudaMalloc(&d_hidden,   hidden_bytes);
	cudaMalloc(&d_Q,        qo_bytes);
	cudaMalloc(&d_K,        kv_bytes);
	cudaMalloc(&d_V,        kv_bytes);
	cudaMalloc(&d_attn_out, qo_bytes);

	// For this v1 implementation, we use ggml for all non-attention operations
	// (embedding lookup, RMS norm, Q/K/V projections, FFN) and call the
	// FlashPrefill CUDA kernels directly for attention.
	//
	// The approach: for each layer, build a ggml graph for pre-attention ops
	// (norm + Q/K/V proj + head norm + RoPE), evaluate it, then call FlashPrefill,
	// then build another graph for post-attention ops (o_proj + residual + FFN).
	//
	// This is chunked at CHUNK_S to limit ggml graph memory usage.

	// TODO: This is the scaffold. Full implementation requires:
	// 1. Embedding lookup (tok_embd)
	// 2. Per-layer: pre-attention ggml graph, FlashPrefill call, post-attention ggml graph
	// 3. Tail scoring: Q_last[N_LOOKAHEAD] @ K_cache^T -> softmax -> max-over-heads
	//
	// For now, we implement the scoring infrastructure and will fill in the ggml
	// graph building once we verify the FlashPrefill kernels work standalone.

	// --- Phase A: embedding lookup ---
	// Build minimal ggml graph: hidden = tok_embd[token_ids]
	{
		ggml_backend_t backend = ggml_backend_init_by_name("CUDA", std::to_string(gpu_device).c_str());

		size_t ctx_size = ggml_tensor_overhead() * 4 + ggml_graph_overhead();
		struct ggml_init_params params = { ctx_size, nullptr, true };
		ggml_context * ctx0 = ggml_init(params);

		ggml_tensor * ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, S);
		ggml_tensor * embd = ggml_get_rows(ctx0, model.tok_embd, ids);

		ggml_cgraph * graph = ggml_new_graph(ctx0);
		ggml_build_forward_expand(graph, embd);

		ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx0, backend);

		// set input token IDs
		ggml_backend_tensor_set(ids, token_ids.data(), 0, S * sizeof(int32_t));

		ggml_backend_graph_compute(backend, graph);

		// copy result to d_hidden
		// embd is in the model's type (BF16) — we need to ensure d_hidden gets BF16
		std::vector<char> tmp(ggml_nbytes(embd));
		ggml_backend_tensor_get(embd, tmp.data(), 0, tmp.size());
		cudaMemcpy(d_hidden, tmp.data(), tmp.size(), cudaMemcpyHostToDevice);

		ggml_backend_buffer_free(buf);
		ggml_free(ctx0);
		ggml_backend_free(backend);
	}

	// --- Phase B: per-layer forward ---
	// For each layer:
	//   1. RMS norm -> Q/K/V projection -> QK head norm -> RoPE
	//   2. FlashPrefill attention (CUDA direct)
	//   3. O projection + residual + FFN
	//   4. Save K to cache for tail scoring

	// TODO: implement the full per-layer forward loop
	// This requires building ggml graphs with the model weights, which needs
	// careful handling of the ggml backend allocation.
	// For now, this is a placeholder that will be filled in Phase 2 implementation.

	fprintf(stderr, "pflash: [TODO] per-layer forward not yet implemented — returning placeholder scores\n");

	// --- Phase C: tail scoring ---
	// Q_last = Q from last N_LOOKAHEAD positions, for each layer
	// score[j] = max over heads of softmax(Q_last @ K_cache[layer]^T)[j]
	// running_max = element-wise max across layers

	// TODO: implement tail scoring once Phase B is complete

	// cleanup
	cudaFree(d_hidden);
	cudaFree(d_Q);
	cudaFree(d_K);
	cudaFree(d_V);
	cudaFree(d_attn_out);
	for (int l = 0; l < n_layers; l++) {
		cudaFree(d_K_cache[l]);
	}
	flash_prefill_free(&fp_bufs);

	return result;
}
