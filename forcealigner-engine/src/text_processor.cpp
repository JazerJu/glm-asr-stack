#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "text_processor.h"

static int is_cjk(uint32_t code) {
    return (code >= 0x4E00 && code <= 0x9FFF)
        || (code >= 0x3400 && code <= 0x4DBF)
        || (code >= 0x20000 && code <= 0x2A6DF)
        || (code >= 0xF900 && code <= 0xFAFF);
}

static int is_kept_char(uint32_t code) {
    if (code == '\'') return 1;
    int cat_major = (code >> 8);
    /* Letters (Lu, Ll, Lt, Lm, Lo) start around 0x4x in unicode cat */
    /* Simplified: keep ASCII letters, digits, and apostrophe */
    if ((code >= 'A' && code <= 'Z') || (code >= 'a' && code <= 'z')) return 1;
    if (code >= '0' && code <= '9') return 1;
    if (is_cjk(code)) return 1;
    return 0;
}

static uint32_t utf8_decode(const char *s, int *bytes) {
    uint8_t c = (uint8_t)s[0];
    if (c < 0x80) { *bytes = 1; return c; }
    if ((c & 0xE0) == 0xC0) { *bytes = 2; return ((c & 0x1F) << 6) | (s[1] & 0x3F); }
    if ((c & 0xF0) == 0xE0) {
        *bytes = 3;
        return ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F);
    }
    if ((c & 0xF8) == 0xF0) {
        *bytes = 4;
        return ((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
    }
    *bytes = 1;
    return c;
}

int tokenize_for_align(const char *text, const char *language,
                       char words[][256], int max_words) {
    int count = 0;
    int len = (int)strlen(text);
    int is_chinese = (strcmp(language, "Chinese") == 0 || strcmp(language, "chinese") == 0
                  || strcmp(language, "Cantonese") == 0 || strcmp(language, "Japanese") == 0
                  || strcmp(language, "Korean") == 0);

    if (is_chinese) {
        int i = 0;
        char latin_buf[256];
        int latin_len = 0;

        auto flush_latin = [&]() {
            if (latin_len > 0) {
                latin_buf[latin_len] = '\0';
                if (count < max_words) {
                    strncpy(words[count], latin_buf, 255);
                    words[count][255] = '\0';
                    count++;
                }
                latin_len = 0;
            }
        };

        while (i < len) {
            int bytes;
            uint32_t cp = utf8_decode(text + i, &bytes);
            if (is_cjk(cp)) {
                flush_latin();
                if (count < max_words) {
                    int written = (bytes < 255) ? bytes : 255;
                    memcpy(words[count], text + i, written);
                    words[count][written] = '\0';
                    count++;
                }
            } else if (is_kept_char(cp)) {
                if (latin_len + bytes < 255) {
                    memcpy(latin_buf + latin_len, text + i, bytes);
                    latin_len += bytes;
                }
            } else {
                flush_latin();
            }
            i += bytes;
        }
        flush_latin();
    } else {
        int i = 0;
        while (i < len && count < max_words) {
            while (i < len && !is_kept_char((uint8_t)text[i]) && (uint8_t)text[i] < 0x80) i++;
            if (i >= len) break;

            int wlen = 0;
            while (i < len && wlen < 255) {
                int bytes;
                uint32_t cp = utf8_decode(text + i, &bytes);
                if (is_cjk(cp)) {
                    if (wlen > 0) break;
                    memcpy(words[count] + wlen, text + i, bytes);
                    wlen += bytes;
                    i += bytes;
                    break;
                }
                if (!is_kept_char(cp)) break;
                memcpy(words[count] + wlen, text + i, bytes);
                wlen += bytes;
                i += bytes;
            }
            if (wlen > 0) {
                words[count][wlen] = '\0';
                count++;
            }
        }
    }
    return count;
}

int build_alignment_prompt(const int *word_token_ids, const int *word_token_counts,
                          int num_words, int num_audio_tokens,
                          const AlignerConfig *cfg,
                          int *out_ids, int max_ids) {
    int pos = 0;

    if (pos >= max_ids) return -1;
    out_ids[pos++] = cfg->audio_start_id;

    for (int i = 0; i < num_audio_tokens && pos < max_ids; i++) {
        out_ids[pos++] = cfg->audio_pad_id;
    }

    if (pos >= max_ids) return -1;
    out_ids[pos++] = cfg->audio_end_id;

    for (int w = 0; w < num_words; w++) {
        for (int t = 0; t < word_token_counts[w] && pos < max_ids; t++) {
            out_ids[pos++] = word_token_ids[w * 64 + t];
        }
        if (pos + 1 >= max_ids) return -1;
        out_ids[pos++] = cfg->timestamp_id;
        out_ids[pos++] = cfg->timestamp_id;
    }

    return pos;
}
