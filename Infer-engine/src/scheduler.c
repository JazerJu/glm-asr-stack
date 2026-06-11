#include "../include/scheduler.h"

#include <stdlib.h>
#include <string.h>

static int scheduler_prefix_tokens(const Sequence *seq) {
    int prompt_tokens;
    int replay_tokens;

    if (seq == NULL) {
        return 0;
    }

    prompt_tokens = seq->num_prompt_tokens > 0 ?
                    seq->num_prompt_tokens : seq->estimated_prompt_tokens;
    replay_tokens = prompt_tokens;
    if (seq->total_seq_len > 0 && seq->n_generated > 0) {
        replay_tokens += seq->n_generated;
    }
    if (seq->total_seq_len > replay_tokens) {
        return seq->total_seq_len;
    }
    return replay_tokens;
}

static int scheduler_required_blocks(const Scheduler *sch, const Sequence *seq) {
    int prefix_tokens;

    if (sch == NULL || seq == NULL || sch->block_size <= 0) {
        return 0;
    }

    prefix_tokens = scheduler_prefix_tokens(seq);
    return (prefix_tokens + sch->block_size - 1) / sch->block_size;
}

static int scheduler_prefill_tokens(const Sequence *seq) {
    if (seq == NULL) {
        return 0;
    }

    return scheduler_prefix_tokens(seq);
}

static void scheduler_prepare_block_table(Sequence *seq, KVPool *pool) {
    if (seq == NULL || pool == NULL) {
        return;
    }

    if (seq->bt.h_table == NULL) {
        block_table_init(&seq->bt, pool, seq->seq_idx);
        return;
    }

    seq->bt.seq_idx = seq->seq_idx;
    seq->bt.max_blocks = pool->max_blocks_per_seq;
    seq->bt.num_blocks = 0;
    seq->bt.d_table = pool->d_block_tables + (size_t) seq->seq_idx * (size_t) pool->max_blocks_per_seq;
    memset(seq->bt.h_table, 0, (size_t) seq->bt.max_blocks * sizeof(*seq->bt.h_table));
}

static void scheduler_waiting_remove_front(Scheduler *sch) {
    if (sch == NULL || sch->waiting_len <= 0) {
        return;
    }

    if (sch->waiting_len > 1) {
        memmove(sch->waiting, sch->waiting + 1,
                (size_t) (sch->waiting_len - 1) * sizeof(*sch->waiting));
    }
    sch->waiting_len--;
}

static void scheduler_waiting_push_front(Scheduler *sch, Sequence *seq) {
    if (sch == NULL || seq == NULL || sch->waiting_len >= sch->max_seqs) {
        return;
    }

    if (sch->waiting_len > 0) {
        memmove(sch->waiting + 1, sch->waiting,
                (size_t) sch->waiting_len * sizeof(*sch->waiting));
    }
    sch->waiting[0] = seq;
    sch->waiting_len++;
}

static int scheduler_running_remove_at(Scheduler *sch, int idx) {
    if (sch == NULL || idx < 0 || idx >= sch->running_len) {
        return 0;
    }

    if (idx + 1 < sch->running_len) {
        memmove(sch->running + idx, sch->running + idx + 1,
                (size_t) (sch->running_len - idx - 1) * sizeof(*sch->running));
    }
    sch->running_len--;
    sch->running[sch->running_len] = NULL;
    return 1;
}

static int scheduler_preempt_sequence(Scheduler *sch, Sequence *victim) {
    int i;

    if (sch == NULL || victim == NULL) {
        return 0;
    }

    for (i = 0; i < sch->running_len; ++i) {
        if (sch->running[i] == victim) {
            scheduler_running_remove_at(sch, i);
            block_table_free(&victim->bt, sch->pool);
            block_table_upload(&victim->bt);
            victim->status = SEQ_WAITING;
            scheduler_waiting_push_front(sch, victim);
            sch->preemptions++;
            return 1;
        }
    }

    return 0;
}

static int scheduler_is_already_scheduled(Sequence *seq,
                                          Sequence **scheduled,
                                          int num_scheduled) {
    int i;

    if (seq == NULL || scheduled == NULL || num_scheduled <= 0) {
        return 0;
    }

    for (i = 0; i < num_scheduled; ++i) {
        if (scheduled[i] == seq) {
            return 1;
        }
    }

    return 0;
}

static int scheduler_preempt_one_except(Scheduler *sch,
                                        Sequence *exclude,
                                        Sequence **scheduled,
                                        int num_scheduled) {
    Sequence *victim;
    int i;

    if (sch == NULL || sch->running_len <= 0) {
        return 0;
    }

    for (i = sch->running_len - 1; i >= 0; --i) {
        victim = sch->running[i];
        if (victim != NULL &&
            victim != exclude &&
            !scheduler_is_already_scheduled(victim, scheduled, num_scheduled)) {
            return scheduler_preempt_sequence(sch, victim);
        }
    }

    return 0;
}

static int scheduler_append_block(Sequence *seq, KVPool *pool) {
    int block_id;

    if (seq == NULL || pool == NULL || seq->bt.h_table == NULL ||
        seq->bt.num_blocks >= seq->bt.max_blocks) {
        return -1;
    }

    block_id = kv_pool_alloc_block(pool);
    if (block_id < 0) {
        return -1;
    }

    seq->bt.h_table[seq->bt.num_blocks++] = block_id;
    block_table_upload(&seq->bt);
    return 0;
}

void scheduler_init(Scheduler *sch, KVPool *pool,
                    int max_seqs, int max_batched_tokens) {
    if (sch == NULL) {
        return;
    }

    memset(sch, 0, sizeof(*sch));
    sch->waiting = (Sequence **) calloc((size_t) max_seqs, sizeof(*sch->waiting));
    sch->running = (Sequence **) calloc((size_t) max_seqs, sizeof(*sch->running));
    sch->pool = pool;
    sch->max_seqs = max_seqs;
    sch->max_batched_tokens = max_batched_tokens;
    sch->block_size = (pool != NULL) ? pool->block_size : 0;
    sch->estimated_blocks_in_use = 0;
}

void scheduler_free(Scheduler *sch) {
    if (sch == NULL) {
        return;
    }

    free(sch->waiting);
    free(sch->running);
    memset(sch, 0, sizeof(*sch));
}

void scheduler_add(Scheduler *sch, Sequence *seq) {
    if (sch == NULL || seq == NULL || sch->waiting_len >= sch->max_seqs) {
        return;
    }

    seq->status = SEQ_WAITING;
    if (seq->estimated_total_blocks > 0) {
        sch->estimated_blocks_in_use += seq->estimated_total_blocks;
    }
    sch->waiting[sch->waiting_len++] = seq;
}

void scheduler_reset(Scheduler *sch) {
    if (!sch) return;
    sch->waiting_len = 0;
    sch->running_len = 0;
    sch->estimated_blocks_in_use = 0;
    sch->preemptions = 0;
    sch->self_preemptions = 0;
    sch->appended_blocks = 0;
}

/*
 * Mixed batch scheduler strategy:
 * 1. First, add prefill sequences from waiting queue (up to budget)
 * 2. Then, fill remaining slots with decode sequences from running queue
 * 3. Always leave some room for decode to prevent starvation
 */
#define MIXED_BATCH_PREFILL_FRACTION 0.5f  /* Max 50% of batch for prefill */

ScheduleResult *scheduler_schedule(Scheduler *sch) {
    static ScheduleResult result;
    static Sequence **result_seqs = NULL;
    static int *result_input_lens = NULL;
    static int *result_cached_lens = NULL;
    static int result_capacity = 0;
    int num_batched;
    int max_prefill_slots;
    int i;

    if (sch == NULL) {
        return NULL;
    }

    /* Ensure capacity for all arrays */
    if (result_capacity < sch->max_seqs) {
        Sequence **new_seqs = (Sequence **) realloc(result_seqs,
                                                    (size_t) sch->max_seqs * sizeof(*result_seqs));
        int *new_input_lens = (int *) realloc(result_input_lens,
                                               (size_t) sch->max_seqs * sizeof(*result_input_lens));
        int *new_cached_lens = (int *) realloc(result_cached_lens,
                                                (size_t) sch->max_seqs * sizeof(*result_cached_lens));
        if (new_seqs == NULL || new_input_lens == NULL || new_cached_lens == NULL) {
            return NULL;
        }
        result_seqs = new_seqs;
        result_input_lens = new_input_lens;
        result_cached_lens = new_cached_lens;
        result_capacity = sch->max_seqs;
    }

    result.seqs = result_seqs;
    result.seq_input_lens = result_input_lens;
    result.seq_cached_lens = result_cached_lens;
    result.num_seqs = 0;
    result.type = STEP_DECODE;  /* Default to decode for mixed batches */
    num_batched = 0;

    /* Calculate max prefill slots (leave room for decode) */
    max_prefill_slots = (int)(sch->max_seqs * MIXED_BATCH_PREFILL_FRACTION);
    max_prefill_slots = max_prefill_slots < 1 ? 1 : max_prefill_slots;

    /* Phase 1: Add prefill sequences from waiting queue */
    int prefill_count = 0;
    while (sch->waiting_len > 0 && 
           sch->running_len < sch->max_seqs &&
           prefill_count < max_prefill_slots) {
        Sequence *seq = sch->waiting[0];
        int needed;
        int prefill_tokens;

        if (seq == NULL) {
            scheduler_waiting_remove_front(sch);
            continue;
        }

        prefill_tokens = scheduler_prefill_tokens(seq);
        if (sch->max_batched_tokens > 0 &&
            num_batched + prefill_tokens > sch->max_batched_tokens) {
            break;
        }

        needed = scheduler_required_blocks(sch, seq);
        if (sch->pool == NULL || needed > sch->pool->free_top) {
            break;
        }

        scheduler_prepare_block_table(seq, sch->pool);
        if (block_table_alloc(&seq->bt, sch->pool, needed) != 0) {
            if (seq->bt.h_table != NULL) {
                memset(seq->bt.h_table, 0,
                       (size_t) seq->bt.max_blocks * sizeof(*seq->bt.h_table));
            }
            seq->bt.num_blocks = 0;
            break;
        }
        block_table_upload(&seq->bt);

        seq->status = SEQ_RUNNING;
        scheduler_waiting_remove_front(sch);
        sch->running[sch->running_len++] = seq;
        
        /* Add to result */
        result.seqs[result.num_seqs] = seq;
        result.seq_input_lens[result.num_seqs] = prefill_tokens;
        result.seq_cached_lens[result.num_seqs] = 0;  /* Prefill starts from 0 */
        result.num_seqs++;
        
        num_batched += prefill_tokens;
        prefill_count++;
    }

    /* Keep prefill and decode in separate scheduler steps. Mixing them reuses
     * shared decoder workspaces in ways that can perturb prefill sampling for
     * tail requests in long daemon RUNs. */
    if (prefill_count > 0) {
        result.type = STEP_PREFILL;
        return &result;
    }

    /* Phase 2: Fill remaining slots with decode sequences */
    int remaining_slots = sch->max_seqs - result.num_seqs;
    int decode_count = 0;
    
    for (i = 0; i < sch->running_len && decode_count < remaining_slots; ++i) {
        Sequence *seq = sch->running[i];
        int needed_blocks;

        if (seq == NULL || sch->block_size <= 0) {
            continue;
        }

        /* Skip sequences that were just added for prefill in this batch */
        int already_in_batch = 0;
        for (int j = 0; j < result.num_seqs; j++) {
            if (result.seqs[j] == seq) {
                already_in_batch = 1;
                break;
            }
        }
        if (already_in_batch) {
            continue;
        }

        /* Check token budget */
        if (sch->max_batched_tokens > 0 &&
            num_batched + 1 > sch->max_batched_tokens) {
            break;
        }

        /* Ensure enough KV blocks. Decode may preempt another running
         * sequence to free a block; if there is no victim, preempt itself
         * and skip this round. Prefill never preempts. */
        needed_blocks = (seq->total_seq_len + 1 + sch->block_size - 1) / sch->block_size;
        while (seq->bt.num_blocks < needed_blocks) {
            if (sch->pool != NULL && sch->pool->free_top <= 0) {
                if (!scheduler_preempt_one_except(sch, seq, result.seqs, result.num_seqs)) {
                    if (scheduler_preempt_sequence(sch, seq)) {
                        sch->self_preemptions++;
                    }
                    break;
                }
            }
            if (scheduler_append_block(seq, sch->pool) != 0) {
                break;
            }
            sch->appended_blocks++;
        }

        if (seq->status != SEQ_RUNNING || seq->bt.num_blocks < needed_blocks) {
            i--;
            continue;
        }

        /* Add decode sequence to batch */
        result.seqs[result.num_seqs] = seq;
        result.seq_input_lens[result.num_seqs] = 1;  /* Decode always 1 token */
        result.seq_cached_lens[result.num_seqs] = seq->total_seq_len;
        result.num_seqs++;
        
        num_batched += 1;
        decode_count++;
    }

    return &result;
}

void scheduler_postprocess(Scheduler *sch, ScheduleResult *result) {
    int i;

    (void) sch;

    if (result == NULL) {
        return;
    }

    for (i = 0; i < result->num_seqs; ++i) {
        Sequence *seq = result->seqs[i];
        if (seq == NULL) {
            continue;
        }
        
        /* Update based on actual tokens processed */
        int input_len = result->seq_input_lens ? result->seq_input_lens[i] : 1;
        
        if (result->seq_cached_lens && result->seq_cached_lens[i] == 0) {
            /* This was a prefill sequence */
            seq->total_seq_len = input_len;
        } else {
            /* This was a decode sequence */
            seq->total_seq_len++;
            seq->n_generated++;
        }
    }
}

void scheduler_release(Scheduler *sch, Sequence *seq) {
    if (sch == NULL || seq == NULL) {
        return;
    }

    sch->estimated_blocks_in_use -= seq->estimated_total_blocks;
    if (sch->estimated_blocks_in_use < 0) {
        sch->estimated_blocks_in_use = 0;
    }
}

int scheduler_is_finished(const Scheduler *sch) {
    if (sch == NULL) {
        return 1;
    }

    return sch->waiting_len == 0 && sch->running_len == 0;
}
