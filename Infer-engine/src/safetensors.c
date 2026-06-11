#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "safetensors.h"
#include "types.h"

static const char *skip_ws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

static char *parse_str(const char *p, const char **out) {
    if (*p != '"') return NULL;
    p++;
    int cap = 256, len = 0;
    char *buf = malloc(cap);
    while (*p && *p != '"') {
        if (*p == '\\') { p++; buf[len++] = *p; }
        else buf[len++] = *p;
        if (len >= cap - 1) { cap *= 2; buf = realloc(buf, cap); }
        p++;
    }
    if (*p == '"') p++;
    buf[len] = 0;
    *out = p;
    return buf;
}

static int64_t parse_int(const char *p, const char **out) {
    int64_t v = 0; int neg = 0;
    if (*p == '-') { neg = 1; p++; }
    while (*p >= '0' && *p <= '9') { v = v * 10 + (*p - '0'); p++; }
    *out = p;
    return neg ? -v : v;
}

static void skip_value(const char *p, const char **out) {
    p = skip_ws(p);
    if (*p == '"') { free(parse_str(p, &p)); }
    else if (*p == '{') { int d=1; p++; while (*p && d) { if(*p=='{')d++; if(*p=='}')d--; p++; } }
    else if (*p == '[') { int d=1; p++; while (*p && d) { if(*p=='[')d++; if(*p==']')d--; p++; } }
    else { while (*p && *p != ',' && *p != '}' && *p != ']') p++; }
    *out = p;
}

static int dtype_from_str(const char *s) {
    if (strcmp(s, "BF16") == 0) return DTYPE_BF16;
    if (strcmp(s, "F32") == 0)  return DTYPE_FP32;
    if (strcmp(s, "F16") == 0)  return DTYPE_FP16;
    if (strcmp(s, "U8") == 0)   return DTYPE_UINT8;
    if (strcmp(s, "F8_E4M3") == 0) return DTYPE_FP8;
    return -1;
}

int safetensors_load(const char *path, WeightStore *ws) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return -1; }

    uint64_t header_size;
    fread(&header_size, 8, 1, f);

    char *json = malloc(header_size + 1);
    fread(json, 1, header_size, f);
    json[header_size] = 0;

    int64_t data_start = 8 + (int64_t)header_size;
    ws->count = 0;

    const char *p = json;
    p = skip_ws(p);
    if (*p == '{') p++;

    while (*p && *p != '}' && ws->count < MAX_WEIGHTS) {
        p = skip_ws(p);
        if (*p == ',') { p++; p = skip_ws(p); }
        if (*p == '}' || !*p) break;

        char *key = parse_str(p, &p);
        if (!key) break;

        p = skip_ws(p);
        if (*p == ':') p++;
        p = skip_ws(p);

        if (strcmp(key, "__metadata__") == 0) {
            free(key);
            skip_value(p, &p);
            continue;
        }

        if (*p != '{') { free(key); skip_value(p, &p); continue; }
        p++;

        char *dtype_str = NULL;
        int shape[MAX_DIMS] = {0};
        int ndim = 0;
        int64_t off0 = 0, off1 = 0;

        while (*p && *p != '}') {
            p = skip_ws(p);
            if (*p == ',') { p++; p = skip_ws(p); }
            if (*p == '}') break;

            char *field = parse_str(p, &p);
            if (!field) break;
            p = skip_ws(p);
            if (*p == ':') p++;
            p = skip_ws(p);

            if (strcmp(field, "dtype") == 0) {
                dtype_str = parse_str(p, &p);
            } else if (strcmp(field, "shape") == 0) {
                if (*p == '[') {
                    p++;
                    p = skip_ws(p);
                    while (*p != ']' && ndim < MAX_DIMS) {
                        shape[ndim++] = (int)parse_int(p, &p);
                        p = skip_ws(p);
                        if (*p == ',') { p++; p = skip_ws(p); }
                    }
                    if (*p == ']') p++;
                }
            } else if (strcmp(field, "data_offsets") == 0) {
                if (*p == '[') {
                    p++;
                    p = skip_ws(p);
                    off0 = parse_int(p, &p);
                    p = skip_ws(p);
                    if (*p == ',') p++;
                    p = skip_ws(p);
                    off1 = parse_int(p, &p);
                    p = skip_ws(p);
                    if (*p == ']') p++;
                }
            } else {
                skip_value(p, &p);
            }
            free(field);
        }
        if (*p == '}') p++;

        int dtype = dtype_str ? dtype_from_str(dtype_str) : -1;
        if (dtype_str) free(dtype_str);

        if (dtype < 0) { free(key); continue; }

        size_t nbytes = (size_t)(off1 - off0);
        if (nbytes == 0 && ndim == 0) {
            nbytes = dtype_size(dtype);
        }

        void *d_data = NULL;
        cudaError_t err = cudaMalloc(&d_data, nbytes > 0 ? nbytes : 4);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for %s: %s\n", key, cudaGetErrorString(err));
            free(key);
            continue;
        }

        void *h_buf = malloc(nbytes > 0 ? nbytes : 4);
        fseek(f, data_start + off0, SEEK_SET);
        fread(h_buf, 1, nbytes > 0 ? nbytes : 4, f);
        cudaMemcpy(d_data, h_buf, nbytes > 0 ? nbytes : 4, cudaMemcpyHostToDevice);
        free(h_buf);

        int idx = ws->count;
        ws->names[idx] = key;
        ws->tensors[idx].data = d_data;
        ws->tensors[idx].dtype = dtype;
        ws->tensors[idx].ndim = ndim > 0 ? ndim : 0;
        for (int j = 0; j < ndim; j++) ws->tensors[idx].shape[j] = shape[j];
        ws->tensors[idx].nbytes = nbytes;
        ws->count++;
    }

    free(json);
    fclose(f);
    return 0;
}

void safetensors_free(WeightStore *ws) {
    for (int i = 0; i < ws->count; i++) {
        if (ws->tensors[i].data) cudaFree(ws->tensors[i].data);
        if (ws->names[i]) free(ws->names[i]);
    }
    ws->count = 0;
}
