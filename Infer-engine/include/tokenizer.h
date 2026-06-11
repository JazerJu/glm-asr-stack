#ifndef GLMASR_TOKENIZER_H
#define GLMASR_TOKENIZER_H

#include <stdint.h>

typedef struct Tokenizer Tokenizer;

Tokenizer *tokenizer_load(const char *path);
void       tokenizer_free(Tokenizer *tok);

/*
 * Decode token IDs to UTF-8 string.
 * Caller must free() the returned buffer.
 * Handles ByteLevel BPE: GPT-2 byte mapping + Ġ → space.
 */
char *tokenizer_decode(const Tokenizer *tok, const int *ids, int n);

int   tokenizer_encode_piece(const Tokenizer *tok, const char *text);

#endif
