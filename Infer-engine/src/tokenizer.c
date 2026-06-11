#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "tokenizer.h"

#define VOCAB_BASE_SIZE 59246
#define VOCAB_TOTAL_SIZE 59264

struct Tokenizer {
    char **id_to_piece;
    int vocab_size;
};

static const char *skip_ws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

static int parse_hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static char decode_unicode_escape(const char *s, char *out) {
    if (s[0] != 'u') return 0;

    int d1 = parse_hex_digit(s[1]);
    int d2 = parse_hex_digit(s[2]);
    int d3 = parse_hex_digit(s[3]);
    int d4 = parse_hex_digit(s[4]);
    if (d1 < 0 || d2 < 0 || d3 < 0 || d4 < 0) return 0;

    uint16_t code = (d1 << 12) | (d2 << 8) | (d3 << 4) | d4;

    if (code < 0x80) {
        out[0] = (char)code;
        return 1;
    } else if (code < 0x800) {
        out[0] = 0xC0 | (code >> 6);
        out[1] = 0x80 | (code & 0x3F);
        return 2;
    } else if (code < 0x10000) {
        out[0] = 0xE0 | (code >> 12);
        out[1] = 0x80 | ((code >> 6) & 0x3F);
        out[2] = 0x80 | (code & 0x3F);
        return 3;
    }
    return 0;
}

static char *parse_json_string(const char *p, const char **end_out) {
    if (*p != '"') return NULL;
    p++;

    char *buf = malloc(256);
    int buflen = 256;
    int len = 0;

    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            if (*p == 'u') {
                char utf8[4];
                int n = decode_unicode_escape(p, utf8);
                if (n > 0) {
                    if (len + n >= buflen) {
                        buflen *= 2;
                        buf = realloc(buf, buflen);
                    }
                    memcpy(buf + len, utf8, n);
                    len += n;
                    p += 5;
                    continue;
                }
            }
            if (*p == 'n') { buf[len++] = '\n'; }
            else if (*p == 't') { buf[len++] = '\t'; }
            else if (*p == 'r') { buf[len++] = '\r'; }
            else if (*p == '"') { buf[len++] = '"'; }
            else if (*p == '\\') { buf[len++] = '\\'; }
            else { buf[len++] = *p; }
        } else {
            if (len + 1 >= buflen) {
                buflen *= 2;
                buf = realloc(buf, buflen);
            }
            buf[len++] = *p;
        }
        p++;
    }

    if (*p == '"') p++;
    buf[len] = '\0';
    *end_out = p;
    return buf;
}

static int64_t parse_json_int(const char *p, const char **end_out) {
    int64_t val = 0;
    int neg = 0;

    if (*p == '-') { neg = 1; p++; }
    while (*p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        p++;
    }
    *end_out = p;
    return neg ? -val : val;
}

static int parse_vocab(const char *p, const char *end, char ***id_to_piece_out) {
    char **id_to_piece = *id_to_piece_out;
    int loaded = 0;

    while (p < end) {
        p = skip_ws(p);
        if (*p == ',') { p++; p = skip_ws(p); }
        if (*p != '"') break;

        char *piece = parse_json_string(p, &p);
        if (!piece) break;

        p = skip_ws(p);
        if (*p != ':') { free(piece); break; }
        p++;
        p = skip_ws(p);

        int64_t id = parse_json_int(p, &p);
        if (id >= 0 && id < VOCAB_TOTAL_SIZE) {
            if (id_to_piece[id]) free(id_to_piece[id]);
            id_to_piece[id] = piece;
            loaded++;
        } else {
            free(piece);
        }
    }
    return 0;
}

static int parse_added_tokens(const char *p, char ***id_to_piece) {
    while (*p && *p != '[') p++;
    if (*p != '[') return -1;
    p++;

    while (*p) {
        p = skip_ws(p);
        if (*p != '{') break;
        p++;

        int64_t id = -1;
        char *content = NULL;

        while (*p && *p != '}') {
            p = skip_ws(p);
            if (*p != '"') break;

            char *key = parse_json_string(p, &p);
            if (!key) break;

            p = skip_ws(p);
            if (*p != ':') {
                free(key);
                break;
            }
            p++;
            p = skip_ws(p);

            if (strcmp(key, "id") == 0) {
                id = parse_json_int(p, &p);
            } else if (strcmp(key, "content") == 0) {
                content = parse_json_string(p, &p);
            } else {
                if (*p == '"') {
                    char *tmp = parse_json_string(p, &p);
                    if (tmp) free(tmp);
                } else {
                    while (*p && *p != ',' && *p != '}') p++;
                }
            }
            free(key);

            p = skip_ws(p);
            if (*p == ',') p++;
        }

        if (*p == '}') p++;
        p = skip_ws(p);
        if (*p == ',') p++;

        if (id >= 0 && content && id >= VOCAB_BASE_SIZE && id < VOCAB_TOTAL_SIZE) {
            (*id_to_piece)[id] = content;
        } else {
            if (content) free(content);
        }

        if (*p == ']') break;
    }

    return 0;
}

Tokenizer *tokenizer_load(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", path);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json = malloc(len + 1);
    if (fread(json, 1, len, f) != (size_t)len) {
        fprintf(stderr, "ERROR: Failed to read tokenizer.json\n");
        free(json);
        fclose(f);
        return NULL;
    }
    json[len] = '\0';
    fclose(f);

    Tokenizer *tok = malloc(sizeof(Tokenizer));
    tok->vocab_size = VOCAB_TOTAL_SIZE;
    tok->id_to_piece = malloc(VOCAB_TOTAL_SIZE * sizeof(char *));
    for (int i = 0; i < VOCAB_TOTAL_SIZE; i++) {
        tok->id_to_piece[i] = NULL;
    }

    const char *vocab_pos = strstr(json, "\"vocab\":");
    if (vocab_pos) {
        const char *p = vocab_pos + 8;
        p = skip_ws(p);
        if (*p == '{') {
            p++;
            const char *vocab_end = p;
            int brace_depth = 1;
            while (*vocab_end && brace_depth > 0) {
                if (*vocab_end == '"') {
                    vocab_end++;
                    while (*vocab_end && *vocab_end != '"') {
                        if (*vocab_end == '\\') vocab_end++;
                        vocab_end++;
                    }
                }
                if (*vocab_end == '{') brace_depth++;
                else if (*vocab_end == '}') brace_depth--;
                vocab_end++;
            }
            parse_vocab(p, vocab_end - 1, &tok->id_to_piece);
        }
    }

    const char *added_pos = strstr(json, "\"added_tokens\":");
    if (added_pos) {
        const char *p = added_pos + 15;
        parse_added_tokens(p, &tok->id_to_piece);
    }

    free(json);

    for (int i = 0; i < tok->vocab_size; i++) {
        if (!tok->id_to_piece[i]) {
            fprintf(stderr, "WARNING: Missing piece for id %d\n", i);
        }
    }

    return tok;
}

void tokenizer_free(Tokenizer *tok) {
    if (!tok) return;
    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->id_to_piece[i]) {
            free(tok->id_to_piece[i]);
        }
    }
    free(tok->id_to_piece);
    free(tok);
}

static int utf8_to_codepoint(const char *s, const char **end_out, unsigned int *cp) {
    const unsigned char *u = (const unsigned char *)s;

    if (u[0] < 0x80) {
        *cp = u[0];
        *end_out = s + 1;
        return 1;
    } else if ((u[0] & 0xE0) == 0xC0) {
        *cp = ((u[0] & 0x1F) << 6) | (u[1] & 0x3F);
        *end_out = s + 2;
        return 2;
    } else if ((u[0] & 0xF0) == 0xE0) {
        *cp = ((u[0] & 0x0F) << 12) | ((u[1] & 0x3F) << 6) | (u[2] & 0x3F);
        *end_out = s + 3;
        return 3;
    } else if ((u[0] & 0xF8) == 0xF0) {
        *cp = ((u[0] & 0x07) << 18) | ((u[1] & 0x3F) << 12) |
              ((u[2] & 0x3F) << 6) | (u[3] & 0x3F);
        *end_out = s + 4;
        return 4;
    }
    return 0;
}

/*
 * GPT-2 ByteLevel: exact inverse of Python bytes_to_unicode().
 *
 * Forward (byte → codepoint):
 *   bytes 33-126      → codepoints 33-126      (printable ASCII)
 *   bytes 161-172     → codepoints 161-172     (¡-¬)
 *   bytes 174-255     → codepoints 174-255     (®-ÿ)
 *   byte 0            → codepoint 256
 *   byte 1            → codepoint 257
 *   ...
 *   byte 32           → codepoint 288          (space)
 *   byte 127          → codepoint 289
 *   bytes 128-160     → codepoints 290-322
 *   byte 173          → codepoint 323
 *
 * We build a lookup table indexed by codepoint (0-323) → original byte.
 * Codepoints outside this range are invalid and mapped to 0xFF sentinel.
 */
#define BYTELEVEL_MAX_CP 324

static unsigned char g_bytelevel_inv[BYTELEVEL_MAX_CP];
static int g_bytelevel_ready = 0;

static void bytelevel_init(void) {
    memset(g_bytelevel_inv, 0xFF, sizeof(g_bytelevel_inv));

    /* Identity ranges: codepoint == byte value */
    for (int b = 33; b <= 126; b++)  g_bytelevel_inv[b] = (unsigned char)b;
    for (int b = 161; b <= 172; b++) g_bytelevel_inv[b] = (unsigned char)b;
    for (int b = 174; b <= 255; b++) g_bytelevel_inv[b] = (unsigned char)b;

    /* Non-identity: remaining bytes in iteration order 0..255 */
    int n = 0;
    for (int b = 0;   b <= 32;  b++) g_bytelevel_inv[256 + n++] = (unsigned char)b;
    g_bytelevel_inv[256 + n++] = 127;       /* DEL */
    for (int b = 128; b <= 160; b++) g_bytelevel_inv[256 + n++] = (unsigned char)b;
    g_bytelevel_inv[256 + n++] = 173;       /* soft hyphen */

    g_bytelevel_ready = 1;
}

static void bytelevel_decode(const char *piece, char *output) {
    if (!g_bytelevel_ready) bytelevel_init();

    const char *p = piece;
    char *out = output;

    while (*p) {
        unsigned int cp = 0;
        const char *next = p;
        int len = utf8_to_codepoint(p, &next, &cp);

        if (len == 0) {
            p++;
            continue;
        }

        if (cp < BYTELEVEL_MAX_CP) {
            unsigned char byte_val = g_bytelevel_inv[cp];
            if (byte_val != 0xFF) {
                *out++ = (char)byte_val;
            } else {
                memcpy(out, p, len);
                out += len;
            }
        } else {
            memcpy(out, p, len);
            out += len;
        }

        p = next;
    }
    *out = '\0';
}

char *tokenizer_decode(const Tokenizer *tok, const int *ids, int n) {
    if (!tok || !ids || n <= 0) {
        char *empty = malloc(1);
        empty[0] = '\0';
        return empty;
    }

    char **decoded = malloc(n * sizeof(char *));
    size_t total_len = 0;

    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id >= 0 && id < tok->vocab_size && tok->id_to_piece[id]) {
            char *piece = tok->id_to_piece[id];
            char *decoded_piece = malloc(strlen(piece) * 4 + 1);
            bytelevel_decode(piece, decoded_piece);
            decoded[i] = decoded_piece;
            total_len += strlen(decoded_piece);
        } else {
            decoded[i] = malloc(1);
            decoded[i][0] = '\0';
        }
    }

    char *result = malloc(total_len + 1);
    result[0] = '\0';

    for (int i = 0; i < n; i++) {
        strcat(result, decoded[i]);
        free(decoded[i]);
    }
    free(decoded);

    return result;
}

int tokenizer_encode_piece(const Tokenizer *tok, const char *text) {
    if (!tok || !text) return -1;

    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->id_to_piece[i] && strcmp(tok->id_to_piece[i], text) == 0) {
            return i;
        }
    }
    return -1;
}
