#include "../include/kv_pool.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define BF16_BYTES 2ULL
#define KV_POOL_HEADROOM_BYTES (800ULL * 1024ULL * 1024ULL)
#define DEFAULT_VRAM_UTIL 0.9

static void kv_pool_reset(KVPool *pool) {
    if (pool == NULL) {
        return;
    }

    pool->k_cache = NULL;
    pool->v_cache = NULL;
    pool->d_block_tables = NULL;
    pool->free_stack = NULL;
    pool->free_top = 0;
    pool->num_blocks = 0;
    pool->block_size = 0;
    pool->num_layers = 0;
    pool->kv_heads = 0;
    pool->head_dim = 0;
    pool->kv_head_stride = 0;
    pool->block_stride = 0;
    pool->layer_stride = 0;
    pool->per_layer_bytes = 0;
}

int kv_pool_init(KVPool *pool, int num_blocks, int block_size,
                 int num_layers, int kv_heads, int head_dim) {
    int i;
    int max_seqs;
    int max_blocks_per_seq;
    size_t table_bytes;
    size_t cache_elems_per_layer;
    size_t cache_bytes_per_layer;
    size_t total_cache_bytes;
    cudaError_t err;

    if (pool == NULL || num_blocks <= 0 || block_size <= 0 ||
        num_layers <= 0 || kv_heads <= 0 || head_dim <= 0) {
        return -1;
    }

    max_seqs = pool->max_seqs > 0 ? pool->max_seqs : 1;
    max_blocks_per_seq = pool->max_blocks_per_seq > 0 ? pool->max_blocks_per_seq : num_blocks;

    pool->k_cache = NULL;
    pool->v_cache = NULL;
    pool->d_block_tables = NULL;
    pool->free_stack = NULL;

    pool->max_seqs = max_seqs;
    pool->max_blocks_per_seq = max_blocks_per_seq;
    pool->num_blocks = num_blocks;
    pool->block_size = block_size;
    pool->num_layers = num_layers;
    pool->kv_heads = kv_heads;
    pool->head_dim = head_dim;
    pool->kv_head_stride = kv_heads * head_dim;
    pool->block_stride = block_size * pool->kv_head_stride;
    pool->layer_stride = num_blocks * pool->block_stride;

    cache_elems_per_layer = (size_t) num_blocks * (size_t) block_size *
                            (size_t) kv_heads * (size_t) head_dim;
    cache_bytes_per_layer = cache_elems_per_layer * BF16_BYTES;
    total_cache_bytes = cache_bytes_per_layer * (size_t) num_layers;
    pool->per_layer_bytes = cache_bytes_per_layer;

    err = cudaMalloc(&pool->k_cache, total_cache_bytes);
    if (err != cudaSuccess) {
        kv_pool_reset(pool);
        pool->max_seqs = max_seqs;
        pool->max_blocks_per_seq = max_blocks_per_seq;
        return -1;
    }

    err = cudaMalloc(&pool->v_cache, total_cache_bytes);
    if (err != cudaSuccess) {
        kv_pool_free(pool);
        pool->max_seqs = max_seqs;
        pool->max_blocks_per_seq = max_blocks_per_seq;
        return -1;
    }

    table_bytes = (size_t) max_seqs * (size_t) max_blocks_per_seq * sizeof(int32_t);
    err = cudaMalloc((void **) &pool->d_block_tables, table_bytes);
    if (err != cudaSuccess) {
        kv_pool_free(pool);
        pool->max_seqs = max_seqs;
        pool->max_blocks_per_seq = max_blocks_per_seq;
        return -1;
    }

    pool->free_stack = (int *) malloc((size_t) num_blocks * sizeof(int));
    if (pool->free_stack == NULL) {
        kv_pool_free(pool);
        pool->max_seqs = max_seqs;
        pool->max_blocks_per_seq = max_blocks_per_seq;
        return -1;
    }

    for (i = 0; i < num_blocks; ++i) {
        pool->free_stack[i] = num_blocks - 1 - i;
    }
    pool->free_top = num_blocks;

    return 0;
}

void kv_pool_free(KVPool *pool) {
    if (pool == NULL) {
        return;
    }

    if (pool->k_cache != NULL) {
        cudaFree(pool->k_cache);
    }
    if (pool->v_cache != NULL) {
        cudaFree(pool->v_cache);
    }
    if (pool->d_block_tables != NULL) {
        cudaFree(pool->d_block_tables);
    }
    free(pool->free_stack);

    kv_pool_reset(pool);
}

void kv_pool_zero_cache(KVPool *pool) {
    if (pool == NULL) return;
    if (pool->k_cache && pool->per_layer_bytes > 0) {
        for (int l = 0; l < pool->num_layers; l++) {
            char *layer_ptr = (char *)pool->k_cache + (size_t)l * pool->layer_stride;
            cudaMemset(layer_ptr, 0, pool->per_layer_bytes);
        }
    }
    if (pool->v_cache && pool->per_layer_bytes > 0) {
        for (int l = 0; l < pool->num_layers; l++) {
            char *layer_ptr = (char *)pool->v_cache + (size_t)l * pool->layer_stride;
            cudaMemset(layer_ptr, 0, pool->per_layer_bytes);
        }
    }
    if (pool->d_block_tables && pool->max_seqs > 0 && pool->max_blocks_per_seq > 0) {
        cudaMemset(pool->d_block_tables, 0,
                   (size_t)pool->max_seqs * (size_t)pool->max_blocks_per_seq * sizeof(int));
    }
    cudaDeviceSynchronize();
}

int kv_pool_alloc_block(KVPool *pool) {
    if (pool == NULL || pool->free_top == 0) {
        return -1;
    }

    return pool->free_stack[--pool->free_top];
}

void kv_pool_free_block(KVPool *pool, int block_id) {
    if (pool == NULL || pool->free_stack == NULL) {
        return;
    }

    pool->free_stack[pool->free_top++] = block_id;
}

void block_table_init(BlockTable *bt, KVPool *pool, int seq_idx) {
    if (bt == NULL || pool == NULL) {
        return;
    }

    bt->seq_idx = seq_idx;
    bt->max_blocks = pool->max_blocks_per_seq;
    bt->num_blocks = 0;
    bt->h_table = (int *) calloc((size_t) bt->max_blocks, sizeof(int));
    bt->d_table = pool->d_block_tables + (size_t) seq_idx * (size_t) pool->max_blocks_per_seq;
}

int block_table_alloc(BlockTable *bt, KVPool *pool, int n) {
    int i;

    if (bt == NULL || pool == NULL || bt->h_table == NULL || n < 0 || n > bt->max_blocks) {
        return -1;
    }

    for (i = 0; i < n; ++i) {
        int block_id = kv_pool_alloc_block(pool);
        if (block_id < 0) {
            while (i > 0) {
                --i;
                kv_pool_free_block(pool, bt->h_table[i]);
                bt->h_table[i] = 0;
            }
            bt->num_blocks = 0;
            return -1;
        }
        bt->h_table[i] = block_id;
    }

    bt->num_blocks = n;
    return 0;
}

void block_table_free(BlockTable *bt, KVPool *pool) {
    int i;

    if (bt == NULL || pool == NULL || bt->h_table == NULL) {
        return;
    }

    for (i = 0; i < bt->num_blocks; ++i) {
        kv_pool_free_block(pool, bt->h_table[i]);
        bt->h_table[i] = 0;
    }
    bt->num_blocks = 0;
}

void block_table_upload(BlockTable *bt) {
    if (bt == NULL || bt->h_table == NULL || bt->d_table == NULL || bt->max_blocks <= 0) {
        return;
    }

    cudaMemcpy(bt->d_table, bt->h_table,
               (size_t) bt->max_blocks * sizeof(int),
               cudaMemcpyHostToDevice);
}

void build_slot_mapping(const int *block_ids, int num_blocks,
                        int start_offset, int num_new_tokens,
                        int block_size,
                        int64_t *h_slot_mapping, int64_t *d_slot_mapping) {
    int i;

    if (block_ids == NULL || h_slot_mapping == NULL || d_slot_mapping == NULL ||
        num_blocks < 0 || num_new_tokens < 0 || block_size <= 0) {
        return;
    }

    for (i = 0; i < num_new_tokens; ++i) {
        int token_idx = start_offset + i;
        int block_idx = token_idx / block_size;
        int offset = token_idx % block_size;

        if (block_idx < 0 || block_idx >= num_blocks) {
            return;
        }

        h_slot_mapping[i] = (int64_t) block_ids[block_idx] * (int64_t) block_size + (int64_t) offset;
    }

    cudaMemcpy(d_slot_mapping, h_slot_mapping,
               (size_t) num_new_tokens * sizeof(int64_t),
               cudaMemcpyHostToDevice);
}

int kv_pool_warmup(int block_size, int num_layers, int kv_heads, int head_dim) {
    size_t free_bytes;
    size_t total_bytes;
    size_t usable_bytes;
    size_t max_allowed;
    size_t already_used;
    size_t per_block_per_layer;
    size_t per_block_total;
    double vram_util;
    const char *env;

    if (block_size <= 0 || num_layers <= 0 || kv_heads <= 0 || head_dim <= 0) {
        return 0;
    }

    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return 0;
    }

    /* Read VRAM utilization ratio from env var (default 0.9 = 90% of total) */
    vram_util = DEFAULT_VRAM_UTIL;
    env = getenv("GLMASR_VRAM_UTIL");
    if (env && env[0] != '\0') {
        double val = atof(env);
        if (val > 0.01 && val <= 1.0) {
            vram_util = val;
        }
    }

    /*
     * Cap total GPU memory usage to vram_util fraction of total VRAM.
     * already_used = model weights + workspace + any prior allocations
     * max_allowed  = total_bytes * vram_util
     * available_for_kv = max_allowed - already_used - headroom
     */
    already_used = total_bytes - free_bytes;
    max_allowed = (size_t)(total_bytes * vram_util);

    if (max_allowed <= already_used + KV_POOL_HEADROOM_BYTES) {
        fprintf(stderr,
                "WARNING: VRAM cap %.0f%% (%zu MB) already exceeded by "
                "existing allocations (%zu MB) plus headroom (%zu MB). "
                "KV pool will be minimal.\n",
                vram_util * 100.0,
                (size_t)(max_allowed / (1024ULL * 1024ULL)),
                (size_t)(already_used / (1024ULL * 1024ULL)),
                (size_t)(KV_POOL_HEADROOM_BYTES / (1024ULL * 1024ULL)));
        return 0;
    }

    usable_bytes = max_allowed - already_used - KV_POOL_HEADROOM_BYTES;

    if (usable_bytes > free_bytes) {
        usable_bytes = free_bytes;
    }

    per_block_per_layer = (size_t) block_size * (size_t) kv_heads * (size_t) head_dim * BF16_BYTES;
    per_block_total = (size_t) num_layers * per_block_per_layer * 2ULL;
    if (per_block_per_layer == 0 || per_block_total == 0) {
        return 0;
    }

    {
        int num_blocks = (int)(usable_bytes / per_block_total);
        if (num_blocks < 4) num_blocks = 4;

        fprintf(stderr,
                "VRAM: total=%zu MB, used=%zu MB, util_cap=%.0f%% (%zu MB), "
                "headroom=%zu MB, kv_budget=%zu MB → %d blocks\n",
                (size_t)(total_bytes / (1024ULL * 1024ULL)),
                (size_t)(already_used / (1024ULL * 1024ULL)),
                vram_util * 100.0,
                (size_t)(max_allowed / (1024ULL * 1024ULL)),
                (size_t)(KV_POOL_HEADROOM_BYTES / (1024ULL * 1024ULL)),
                (size_t)(usable_bytes / (1024ULL * 1024ULL)),
                num_blocks);
        return num_blocks;
    }
}
