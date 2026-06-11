#ifndef GLMASR_KV_POOL_H
#define GLMASR_KV_POOL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Paged KV Cache — block-based memory pool for decoder KV storage.
 *
 * Layout mirrors nano-vllm / vLLM / FlashAttention:
 *   k_cache, v_cache are pre-allocated tensors on GPU,
 *   indexed by physical block ID.
 *
 * Each block holds `block_size` consecutive tokens.
 * Per block per layer: block_size * kv_heads * head_dim * 2 bytes (bf16).
 * For GLM-ASR with block_size=128, kv_heads=4, head_dim=128:
 *   128 * 4 * 128 * 2 = 128 KB per block per layer.
 *   28 layers * 256 KB = 7 MB per block total.
 */

#define KV_DEFAULT_BLOCK_SIZE 128

typedef struct {
    void    *k_cache;           /* GPU: [num_blocks, block_size, kv_heads, head_dim] bf16 */
    void    *v_cache;           /* GPU: same layout */
    int     *d_block_tables;    /* GPU: [max_seqs, max_blocks_per_seq] int32, row-major */
    int      max_seqs;          /* maximum concurrent sequences */
    int      max_blocks_per_seq;/* maximum blocks a single sequence can occupy */

    int     *free_stack;        /* host: LIFO stack of free block IDs */
    int      free_top;          /* stack pointer (next pop index) */
    int      num_blocks;        /* total allocated blocks */

    int      block_size;        /* tokens per block (default 128) */
    int      num_layers;        /* decoder layers (28) */
    int      kv_heads;          /* 4 */
    int      head_dim;          /* 128 */
    int      kv_head_stride;    /* kv_heads * head_dim (512) */
    int      block_stride;      /* block_size * kv_head_stride */
    int      layer_stride;      /* num_blocks * block_stride */

    size_t   per_layer_bytes;   /* num_blocks * block_stride * sizeof(uint16_t) */
} KVPool;

typedef struct {
    int     *d_table;           /* GPU pointer into pool->d_block_tables at row seq_idx */
    int      max_blocks;
    int      num_blocks;        /* currently allocated blocks */
    int     *h_table;           /* CPU mirror */
    int      seq_idx;           /* which row in the pool's block_tables */
} BlockTable;

/* ---- Lifecycle ---- */

/* Initialize KV pool. Returns 0 on success. */
int kv_pool_init(KVPool *pool, int num_blocks, int block_size,
                 int num_layers, int kv_heads, int head_dim);

/* Free all GPU memory. */
void kv_pool_free(KVPool *pool);

/* Zero out all KV cache data (for reuse across batches). */
void kv_pool_zero_cache(KVPool *pool);

/* ---- Block allocation ---- */

/* Allocate one block. Returns block_id >= 0, or -1 if exhausted. */
int kv_pool_alloc_block(KVPool *pool);

/* Return a block to the free pool. */
void kv_pool_free_block(KVPool *pool, int block_id);

/* ---- Block table per sequence ---- */

/* Attach a block table row to a sequence. seq_idx < max_seqs. */
void block_table_init(BlockTable *bt, KVPool *pool, int seq_idx);

/* Allocate `n` blocks for this sequence. Returns 0 on success. */
int block_table_alloc(BlockTable *bt, KVPool *pool, int n);

/* Free all blocks for this sequence. */
void block_table_free(BlockTable *bt, KVPool *pool);

/* Upload CPU mirror to GPU. Call after any block_table_alloc/free. */
void block_table_upload(BlockTable *bt);

/* ---- Slot mapping ---- */

/*
 * Build slot_mapping: maps each token to its physical position in the cache.
 * slot_mapping[token_idx] = physical_block_id * block_size + offset_in_block
 *
 * Caller provides host arrays; this function fills them and uploads to GPU.
 *
 * block_ids: [num_blocks] — logical block IDs for this sequence
 * num_blocks: number of blocks
 * start_offset: token offset within first block (for prefix caching, 0 normally)
 * num_new_tokens: how many NEW tokens to map (from last block's free slots + new blocks)
 * h_slot_mapping: host buffer, size >= num_new_tokens
 * d_slot_mapping: GPU buffer (will be written to)
 */
void build_slot_mapping(const int *block_ids, int num_blocks,
                        int start_offset, int num_new_tokens,
                        int block_size,
                        int64_t *h_slot_mapping, int64_t *d_slot_mapping);

/* ---- Warmup ---- */

/*
 * Probe available GPU memory after model load.
 * Run a dummy forward to trigger peak allocation, then measure free memory.
 * Returns the number of KV blocks that can be allocated.
 */
int kv_pool_warmup(int block_size, int num_layers, int kv_heads, int head_dim);

#ifdef __cplusplus
}
#endif

#endif /* GLMASR_KV_POOL_H */
