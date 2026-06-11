#ifndef GLMASR_SCHEDULER_H
#define GLMASR_SCHEDULER_H

#include <stdint.h>

#include "sequence.h"
#include "kv_pool.h"

/*
 * Prefill-first scheduler (nano-vllm style).
 * schedule() → which sequences to process next + prefill or decode.
 * postprocess() → update sequence state after GPU step.
 */

typedef enum {
    STEP_PREFILL,
    STEP_DECODE
} StepType;

typedef struct {
    Sequence  **waiting;
    int         waiting_len;

    Sequence  **running;
    int         running_len;

    KVPool     *pool;

    int         max_seqs;
    int         max_batched_tokens;
    int         block_size;
    int         estimated_blocks_in_use;
    uint64_t    preemptions;
    uint64_t    self_preemptions;
    uint64_t    appended_blocks;
} Scheduler;

typedef struct {
    Sequence **seqs;
    int        num_seqs;
    StepType   type;
    int       *seq_input_lens;    /* [num_seqs] input tokens for this step */
    int       *seq_cached_lens;   /* [num_seqs] cached KV length (for attention) */
} ScheduleResult;

void scheduler_init(Scheduler *sch, KVPool *pool,
                    int max_seqs, int max_batched_tokens);

void scheduler_free(Scheduler *sch);

void scheduler_reset(Scheduler *sch);

void scheduler_add(Scheduler *sch, Sequence *seq);

/*
 * Decide next step. Returns sequences to process and step type.
 * Caller owns the returned pointer (valid until next schedule() call).
 */
ScheduleResult *scheduler_schedule(Scheduler *sch);

/*
 * Update sequence state after a step.
 * For decode: seq->total_seq_len++, seq->n_generated++.
 * For prefill: seq->total_seq_len = seq->num_prompt_tokens.
 */
void scheduler_postprocess(Scheduler *sch, ScheduleResult *result);

void scheduler_release(Scheduler *sch, Sequence *seq);

int scheduler_is_finished(const Scheduler *sch);

#endif
