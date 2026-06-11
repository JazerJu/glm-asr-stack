#include "../include/safetensors.h"

#include <cuda_runtime_api.h>

#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error [%s:%d]: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)
#endif

namespace {

void skip_ws(const char *&p, const char *end) {
    while (p < end && std::isspace(static_cast<unsigned char>(*p))) {
        ++p;
    }
}

bool parse_string(const char *&p, const char *end, std::string &out) {
    if (p >= end || *p != '"') {
        return false;
    }
    ++p;
    out.clear();
    while (p < end) {
        char ch = *p++;
        if (ch == '"') {
            return true;
        }
        if (ch == '\\') {
            if (p >= end) {
                return false;
            }
            char esc = *p++;
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                default:
                    return false;
            }
            continue;
        }
        out.push_back(ch);
    }
    return false;
}

const char *match_enclosed(const char *p, const char *end, char open_ch, char close_ch) {
    if (p >= end || *p != open_ch) {
        return nullptr;
    }
    int depth = 0;
    bool in_string = false;
    bool escape = false;
    for (const char *it = p; it < end; ++it) {
        char ch = *it;
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (ch == '\\') {
                escape = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }
        if (ch == '"') {
            in_string = true;
            continue;
        }
        if (ch == open_ch) {
            ++depth;
        } else if (ch == close_ch) {
            --depth;
            if (depth == 0) {
                return it;
            }
        }
    }
    return nullptr;
}

bool find_key_value(const char *obj, size_t obj_len, const char *key,
                    const char *&value_ptr, size_t &value_len) {
    std::string pattern = std::string("\"") + key + "\"";
    std::string object(obj, obj_len);
    size_t key_pos = object.find(pattern);
    if (key_pos == std::string::npos) {
        return false;
    }
    size_t colon = object.find(':', key_pos + pattern.size());
    if (colon == std::string::npos) {
        return false;
    }
    size_t start = colon + 1;
    while (start < obj_len && std::isspace(static_cast<unsigned char>(object[start]))) {
        ++start;
    }
    if (start >= obj_len) {
        return false;
    }
    if (object[start] == '"') {
        size_t pos = start + 1;
        bool escape = false;
        while (pos < obj_len) {
            char ch = object[pos];
            if (escape) {
                escape = false;
            } else if (ch == '\\') {
                escape = true;
            } else if (ch == '"') {
                value_ptr = obj + start;
                value_len = pos - start + 1;
                return true;
            }
            ++pos;
        }
        return false;
    }
    if (object[start] == '[') {
        const char *begin = obj + start;
        const char *close = match_enclosed(begin, obj + obj_len, '[', ']');
        if (!close) {
            return false;
        }
        value_ptr = begin;
        value_len = static_cast<size_t>(close - begin + 1);
        return true;
    }
    if (object[start] == '{') {
        const char *begin = obj + start;
        const char *close = match_enclosed(begin, obj + obj_len, '{', '}');
        if (!close) {
            return false;
        }
        value_ptr = begin;
        value_len = static_cast<size_t>(close - begin + 1);
        return true;
    }

    size_t pos = start;
    while (pos < obj_len && object[pos] != ',' && object[pos] != '}') {
        ++pos;
    }
    size_t finish = pos;
    while (finish > start && std::isspace(static_cast<unsigned char>(object[finish - 1]))) {
        --finish;
    }
    value_ptr = obj + start;
    value_len = finish - start;
    return true;
}

bool parse_dtype(const char *obj, size_t obj_len, int &dtype) {
    const char *value_ptr = nullptr;
    size_t value_len = 0;
    if (!find_key_value(obj, obj_len, "dtype", value_ptr, value_len) || value_len < 2 ||
        value_ptr[0] != '"' || value_ptr[value_len - 1] != '"') {
        return false;
    }
    std::string inner(value_ptr + 1, value_len - 2);
    if (inner == "BF16" || inner == "F16_BF16") {
        dtype = DTYPE_BF16;
        return true;
    }
    if (inner == "F32") {
        dtype = DTYPE_FP32;
        return true;
    }
    return false;
}

bool parse_shape(const char *obj, size_t obj_len, Tensor &tensor) {
    const char *value_ptr = nullptr;
    size_t value_len = 0;
    if (!find_key_value(obj, obj_len, "shape", value_ptr, value_len) || value_len < 2 ||
        value_ptr[0] != '[' || value_ptr[value_len - 1] != ']') {
        return false;
    }

    tensor.ndim = 0;
    const char *p = value_ptr + 1;
    const char *end = value_ptr + value_len - 1;
    skip_ws(p, end);
    if (p >= end) {
        return false;
    }
    while (p < end) {
        if (tensor.ndim >= MAX_DIMS) {
            return false;
        }
        char *next = nullptr;
        long long dim = std::strtoll(p, &next, 10);
        if (next == p) {
            return false;
        }
        tensor.shape[tensor.ndim++] = static_cast<int64_t>(dim);
        p = next;
        skip_ws(p, end);
        if (p < end && *p == ',') {
            ++p;
            skip_ws(p, end);
        }
    }
    return tensor.ndim > 0;
}

bool parse_offsets(const char *obj, size_t obj_len, size_t &start, size_t &finish) {
    const char *value_ptr = nullptr;
    size_t value_len = 0;
    if (!find_key_value(obj, obj_len, "data_offsets", value_ptr, value_len) || value_len < 2 ||
        value_ptr[0] != '[' || value_ptr[value_len - 1] != ']') {
        return false;
    }

    const char *p = value_ptr + 1;
    const char *end = value_ptr + value_len - 1;
    skip_ws(p, end);
    char *next = nullptr;
    unsigned long long s = std::strtoull(p, &next, 10);
    if (next == p) {
        return false;
    }
    p = next;
    skip_ws(p, end);
    if (p >= end || *p != ',') {
        return false;
    }
    ++p;
    skip_ws(p, end);
    unsigned long long e = std::strtoull(p, &next, 10);
    if (next == p) {
        return false;
    }
    start = static_cast<size_t>(s);
    finish = static_cast<size_t>(e);
    return finish >= start;
}

}  // namespace

int safetensors_load(const char *path, WeightStore *ws) {
    if (!path || !ws) {
        fprintf(stderr, "safetensors_load: invalid argument\n");
        return -1;
    }

    std::memset(ws, 0, sizeof(*ws));

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "safetensors_load: failed to open %s: %s\n", path, std::strerror(errno));
        return -1;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "safetensors_load: fstat failed for %s: %s\n", path, std::strerror(errno));
        close(fd);
        return -1;
    }
    if (st.st_size < 8) {
        fprintf(stderr, "safetensors_load: file too small: %s\n", path);
        close(fd);
        return -1;
    }

    uint64_t header_len = 0;
    ssize_t got = pread(fd, &header_len, sizeof(header_len), 0);
    if (got != static_cast<ssize_t>(sizeof(header_len))) {
        fprintf(stderr, "safetensors_load: failed to read header length from %s\n", path);
        close(fd);
        return -1;
    }

    const uint64_t data_offset = sizeof(uint64_t) + header_len;
    if (data_offset > static_cast<uint64_t>(st.st_size)) {
        fprintf(stderr, "safetensors_load: invalid header length in %s\n", path);
        close(fd);
        return -1;
    }

    std::string header(static_cast<size_t>(header_len), '\0');
    if (header_len > 0) {
        got = pread(fd, header.data(), static_cast<size_t>(header_len), sizeof(uint64_t));
        if (got != static_cast<ssize_t>(header_len)) {
            fprintf(stderr, "safetensors_load: failed to read JSON header from %s\n", path);
            close(fd);
            return -1;
        }
    }

    size_t data_size = static_cast<size_t>(st.st_size - static_cast<off_t>(data_offset));
    long page_size = sysconf(_SC_PAGESIZE);
    off_t aligned_offset = (static_cast<off_t>(data_offset) / page_size) * page_size;
    size_t extra = static_cast<size_t>(data_offset - static_cast<uint64_t>(aligned_offset));
    size_t mapped_size = data_size + extra;
    void *mapped = mmap(nullptr, mapped_size, PROT_READ, MAP_PRIVATE, fd, aligned_offset);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "safetensors_load: mmap failed for %s: %s\n", path, std::strerror(errno));
        close(fd);
        return -1;
    }
    const uint8_t *data_base = static_cast<const uint8_t *>(mapped) + extra;

    const char *p = header.data();
    const char *end = header.data() + header.size();
    skip_ws(p, end);
    if (p >= end || *p != '{') {
        fprintf(stderr, "safetensors_load: invalid JSON header in %s\n", path);
        munmap(mapped, data_size);
        close(fd);
        return -1;
    }
    ++p;

    while (true) {
        skip_ws(p, end);
        if (p >= end) {
            fprintf(stderr, "safetensors_load: truncated JSON header in %s\n", path);
            safetensors_free(ws);
            munmap(mapped, data_size);
            close(fd);
            return -1;
        }
        if (*p == '}') {
            ++p;
            break;
        }

        std::string name;
        if (!parse_string(p, end, name)) {
            fprintf(stderr, "safetensors_load: failed to parse tensor name in %s\n", path);
            safetensors_free(ws);
            munmap(mapped, data_size);
            close(fd);
            return -1;
        }
        skip_ws(p, end);
        if (p >= end || *p != ':') {
            fprintf(stderr, "safetensors_load: missing ':' after tensor name %s\n", name.c_str());
            safetensors_free(ws);
            munmap(mapped, data_size);
            close(fd);
            return -1;
        }
        ++p;
        skip_ws(p, end);

        if (p >= end || *p != '{') {
            fprintf(stderr, "safetensors_load: expected object for tensor %s\n", name.c_str());
            safetensors_free(ws);
            munmap(mapped, data_size);
            close(fd);
            return -1;
        }
        const char *obj_begin = p;
        const char *obj_end = match_enclosed(obj_begin, end, '{', '}');
        if (!obj_end) {
            fprintf(stderr, "safetensors_load: malformed object for tensor %s\n", name.c_str());
            safetensors_free(ws);
            munmap(mapped, data_size);
            close(fd);
            return -1;
        }
        size_t obj_len = static_cast<size_t>(obj_end - obj_begin + 1);
        p = obj_end + 1;

        if (name != "__metadata__") {
            if (ws->count >= MAX_WEIGHTS) {
                fprintf(stderr, "safetensors_load: exceeded MAX_WEIGHTS (%d)\n", MAX_WEIGHTS);
                safetensors_free(ws);
                munmap(mapped, data_size);
                close(fd);
                return -1;
            }

            Tensor tensor{};
            size_t start = 0;
            size_t finish = 0;
            if (!parse_dtype(obj_begin, obj_len, tensor.dtype)) {
                fprintf(stderr, "safetensors_load: unsupported or missing dtype for %s\n", name.c_str());
                safetensors_free(ws);
                munmap(mapped, data_size);
                close(fd);
                return -1;
            }
            if (!parse_shape(obj_begin, obj_len, tensor)) {
                fprintf(stderr, "safetensors_load: invalid shape for %s\n", name.c_str());
                safetensors_free(ws);
                munmap(mapped, data_size);
                close(fd);
                return -1;
            }
            if (!parse_offsets(obj_begin, obj_len, start, finish)) {
                fprintf(stderr, "safetensors_load: invalid data_offsets for %s\n", name.c_str());
                safetensors_free(ws);
                munmap(mapped, data_size);
                close(fd);
                return -1;
            }
            if (finish > data_size) {
                fprintf(stderr, "safetensors_load: data range out of bounds for %s\n", name.c_str());
                safetensors_free(ws);
                munmap(mapped, data_size);
                close(fd);
                return -1;
            }

            size_t elem_size = dtype_size(tensor.dtype);
            if (elem_size == 0) {
                fprintf(stderr, "safetensors_load: bad dtype size for %s\n", name.c_str());
                safetensors_free(ws);
                munmap(mapped, data_size);
                close(fd);
                return -1;
            }

            int64_t numel = tensor_numel(&tensor);
            if (numel <= 0) {
                fprintf(stderr, "safetensors_load: invalid tensor numel for %s\n", name.c_str());
                safetensors_free(ws);
                munmap(mapped, data_size);
                close(fd);
                return -1;
            }

            tensor.nbytes = finish - start;
            size_t expected_nbytes = static_cast<size_t>(numel) * elem_size;
            if (tensor.nbytes != expected_nbytes) {
                fprintf(stderr, "safetensors_load: size mismatch for %s (got %zu, expected %zu)\n",
                        name.c_str(), tensor.nbytes, expected_nbytes);
                safetensors_free(ws);
                munmap(mapped, data_size);
                close(fd);
                return -1;
            }

            CUDA_CHECK(cudaMalloc(&tensor.data, tensor.nbytes));
            CUDA_CHECK(cudaMemcpy(tensor.data,
                                  data_base + start,
                                  tensor.nbytes,
                                  cudaMemcpyHostToDevice));

            char *stored_name = static_cast<char *>(std::malloc(name.size() + 1));
            if (!stored_name) {
                fprintf(stderr, "safetensors_load: failed to allocate name for %s\n", name.c_str());
                CUDA_CHECK(cudaFree(tensor.data));
                safetensors_free(ws);
                munmap(mapped, data_size);
                close(fd);
                return -1;
            }
            std::memcpy(stored_name, name.c_str(), name.size() + 1);

            ws->names[ws->count] = stored_name;
            ws->tensors[ws->count] = tensor;
            ++ws->count;
        }

        skip_ws(p, end);
        if (p < end && *p == ',') {
            ++p;
            continue;
        }
        if (p < end && *p == '}') {
            ++p;
            break;
        }
    }

    munmap(mapped, mapped_size);
    close(fd);
    return 0;
}

void safetensors_free(WeightStore *ws) {
    if (!ws) {
        return;
    }
    for (int i = 0; i < ws->count; ++i) {
        if (ws->tensors[i].data) {
            CUDA_CHECK(cudaFree(ws->tensors[i].data));
        }
        if (ws->names[i]) {
            std::free(ws->names[i]);
        }
        ws->names[i] = nullptr;
        std::memset(&ws->tensors[i], 0, sizeof(ws->tensors[i]));
    }
    ws->count = 0;
}
