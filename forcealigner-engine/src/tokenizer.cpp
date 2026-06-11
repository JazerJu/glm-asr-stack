#include "tokenizer.h"

#include <climits>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct TokenizerInternal {
    std::unordered_map<std::string, int> vocab_lookup;
    std::unordered_map<std::string, int> merge_priority;
};

static std::unordered_map<const Tokenizer *, TokenizerInternal> g_tokenizer_data;

static char *dup_cstr(const std::string &s) {
    char *out = static_cast<char *>(std::malloc(s.size() + 1));
    if (!out) {
        return nullptr;
    }
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

static std::string join_path(const char *dir, const char *name) {
    std::string path = dir ? dir : "";
    if (!path.empty() && path.back() != '/') {
        path.push_back('/');
    }
    path += name;
    return path;
}

static bool read_file(const std::string &path, std::string *out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    out->assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return true;
}

static void append_utf8(std::string *out, unsigned int codepoint) {
    if (codepoint <= 0x7F) {
        out->push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FF) {
        out->push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        out->push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else if (codepoint <= 0xFFFF) {
        out->push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        out->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    } else {
        out->push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
        out->push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
}

static int byte_level_unicode_to_byte(int cp) {
    static const int direct_ranges[][2] = {{33,126},{161,172},{174,255}};
    for (auto &r : direct_ranges) {
        if (cp >= r[0] && cp <= r[1]) return cp;
    }
    if (cp < 256) return -1;
    int n = cp - 256;
    static const int other_bytes[] = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
        127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,
        173
    };
    if (n < 0 || n >= 68) return -1;
    return other_bytes[n];
}

static std::string byte_level_decode(const std::string &encoded) {
    std::string raw_bytes;
    raw_bytes.reserve(encoded.size());
    size_t i = 0;
    while (i < encoded.size()) {
        int cp = static_cast<unsigned char>(encoded[i]);
        i++;
        if (cp >= 0xF0) {
            cp = ((cp & 0x07) << 18) | ((static_cast<unsigned char>(encoded[i]) & 0x3F) << 12)
               | ((static_cast<unsigned char>(encoded[i+1]) & 0x3F) << 6) | (static_cast<unsigned char>(encoded[i+2]) & 0x3F);
            i += 3;
        } else if (cp >= 0xE0) {
            cp = ((cp & 0x0F) << 12) | ((static_cast<unsigned char>(encoded[i]) & 0x3F) << 6) | (static_cast<unsigned char>(encoded[i+1]) & 0x3F);
            i += 2;
        } else if (cp >= 0xC0) {
            cp = ((cp & 0x1F) << 6) | (static_cast<unsigned char>(encoded[i]) & 0x3F);
            i += 1;
        }
        int byte_val = byte_level_unicode_to_byte(cp);
        if (byte_val >= 0) {
            raw_bytes.push_back(static_cast<char>(byte_val));
        }
    }
    return raw_bytes;
}

static void skip_ws(const std::string &s, size_t *pos) {
    while (*pos < s.size() && std::isspace(static_cast<unsigned char>(s[*pos]))) {
        (*pos)++;
    }
}

static bool parse_json_string(const std::string &s, size_t *pos, std::string *out) {
    skip_ws(s, pos);
    if (*pos >= s.size() || s[*pos] != '"') {
        return false;
    }

    (*pos)++;
    out->clear();
    while (*pos < s.size()) {
        char ch = s[*pos];
        (*pos)++;
        if (ch == '"') {
            return true;
        }
        if (ch != '\\') {
            out->push_back(ch);
            continue;
        }
        if (*pos >= s.size()) {
            return false;
        }

        char esc = s[*pos];
        (*pos)++;
        switch (esc) {
            case '"': out->push_back('"'); break;
            case '\\': out->push_back('\\'); break;
            case '/': out->push_back('/'); break;
            case 'b': out->push_back('\b'); break;
            case 'f': out->push_back('\f'); break;
            case 'n': out->push_back('\n'); break;
            case 'r': out->push_back('\r'); break;
            case 't': out->push_back('\t'); break;
            case 'u': {
                if (*pos + 4 > s.size()) {
                    return false;
                }
                unsigned int codepoint = 0;
                for (int i = 0; i < 4; ++i) {
                    char hex = s[*pos + i];
                    codepoint <<= 4;
                    if (hex >= '0' && hex <= '9') {
                        codepoint |= static_cast<unsigned int>(hex - '0');
                    } else if (hex >= 'a' && hex <= 'f') {
                        codepoint |= static_cast<unsigned int>(hex - 'a' + 10);
                    } else if (hex >= 'A' && hex <= 'F') {
                        codepoint |= static_cast<unsigned int>(hex - 'A' + 10);
                    } else {
                        return false;
                    }
                }
                *pos += 4;
                append_utf8(out, codepoint);
                break;
            }
            default:
                return false;
        }
    }

    return false;
}

static bool parse_json_int(const std::string &s, size_t *pos, int *out) {
    skip_ws(s, pos);
    if (*pos >= s.size()) {
        return false;
    }

    char *end = nullptr;
    long value = std::strtol(s.c_str() + *pos, &end, 10);
    if (end == s.c_str() + *pos) {
        return false;
    }
    *pos = static_cast<size_t>(end - s.c_str());
    *out = static_cast<int>(value);
    return true;
}

static bool parse_vocab_json(const std::string &content,
                            std::vector<std::pair<std::string, int>> *entries,
                            int *max_id) {
    size_t pos = 0;
    *max_id = -1;
    entries->clear();
    skip_ws(content, &pos);
    if (pos >= content.size() || content[pos] != '{') {
        return false;
    }
    pos++;

    while (true) {
        skip_ws(content, &pos);
        if (pos >= content.size()) {
            return false;
        }
        if (content[pos] == '}') {
            pos++;
            return true;
        }

        std::string key;
        int value = -1;
        if (!parse_json_string(content, &pos, &key)) {
            return false;
        }
        skip_ws(content, &pos);
        if (pos >= content.size() || content[pos] != ':') {
            return false;
        }
        pos++;
        if (!parse_json_int(content, &pos, &value)) {
            return false;
        }
        entries->push_back(std::make_pair(key, value));
        if (value > *max_id) {
            *max_id = value;
        }

        skip_ws(content, &pos);
        if (pos >= content.size()) {
            return false;
        }
        if (content[pos] == ',') {
            pos++;
            continue;
        }
        if (content[pos] == '}') {
            pos++;
            return true;
        }
        return false;
    }
}

static bool parse_added_token_ids(const std::string &content,
                                  int *audio_start_id,
                                  int *audio_end_id,
                                  int *audio_pad_id,
                                  int *timestamp_id) {
    *audio_start_id = -1;
    *audio_end_id = -1;
    *audio_pad_id = -1;
    *timestamp_id = -1;

    size_t pos = content.find("\"added_tokens_decoder\"");
    if (pos == std::string::npos) {
        return false;
    }
    pos = content.find('{', pos);
    if (pos == std::string::npos) {
        return false;
    }
    pos++;

    while (true) {
        skip_ws(content, &pos);
        if (pos >= content.size()) {
            return false;
        }
        if (content[pos] == '}') {
            return true;
        }

        std::string id_text;
        if (!parse_json_string(content, &pos, &id_text)) {
            return false;
        }
        skip_ws(content, &pos);
        if (pos >= content.size() || content[pos] != ':') {
            return false;
        }
        pos++;
        skip_ws(content, &pos);
        if (pos >= content.size() || content[pos] != '{') {
            return false;
        }
        pos++;

        int id = std::atoi(id_text.c_str());
        std::string token_content;

        while (true) {
            skip_ws(content, &pos);
            if (pos >= content.size()) {
                return false;
            }
            if (content[pos] == '}') {
                pos++;
                break;
            }

            std::string field_name;
            if (!parse_json_string(content, &pos, &field_name)) {
                return false;
            }
            skip_ws(content, &pos);
            if (pos >= content.size() || content[pos] != ':') {
                return false;
            }
            pos++;

            if (field_name == "content") {
                if (!parse_json_string(content, &pos, &token_content)) {
                    return false;
                }
            } else {
                skip_ws(content, &pos);
                if (pos >= content.size()) {
                    return false;
                }
                if (content[pos] == '"') {
                    std::string ignored;
                    if (!parse_json_string(content, &pos, &ignored)) {
                        return false;
                    }
                } else if (content[pos] == '{') {
                    int depth = 0;
                    do {
                        if (content[pos] == '{') {
                            depth++;
                        } else if (content[pos] == '}') {
                            depth--;
                        }
                        pos++;
                        if (pos > content.size()) {
                            return false;
                        }
                    } while (depth > 0);
                } else if (content[pos] == '[') {
                    int depth = 0;
                    do {
                        if (content[pos] == '[') {
                            depth++;
                        } else if (content[pos] == ']') {
                            depth--;
                        }
                        pos++;
                        if (pos > content.size()) {
                            return false;
                        }
                    } while (depth > 0);
                } else {
                    while (pos < content.size() && content[pos] != ',' && content[pos] != '}') {
                        pos++;
                    }
                }
            }

            skip_ws(content, &pos);
            if (pos >= content.size()) {
                return false;
            }
            if (content[pos] == ',') {
                pos++;
                continue;
            }
            if (content[pos] == '}') {
                pos++;
                break;
            }
            return false;
        }

        if (token_content == "<|audio_start|>") {
            *audio_start_id = id;
        } else if (token_content == "<|audio_end|>") {
            *audio_end_id = id;
        } else if (token_content == "<|audio_pad|>") {
            *audio_pad_id = id;
        } else if (token_content == "<timestamp>") {
            *timestamp_id = id;
        }

        skip_ws(content, &pos);
        if (pos >= content.size()) {
            return false;
        }
        if (content[pos] == ',') {
            pos++;
            continue;
        }
        if (content[pos] == '}') {
            return true;
        }
        return false;
    }
}

static const std::vector<std::string> &byte_to_unicode_table() {
    static std::vector<std::string> table;
    if (!table.empty()) {
        return table;
    }

    std::vector<int> bs;
    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) {
        bs.push_back(b);
    }
    for (int b = 0xA1; b <= 0xAC; ++b) {
        bs.push_back(b);
    }
    for (int b = 0xAE; b <= 0xFF; ++b) {
        bs.push_back(b);
    }

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool found = false;
        for (size_t i = 0; i < bs.size(); ++i) {
            if (bs[i] == b) {
                found = true;
                break;
            }
        }
        if (!found) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    table.resize(256);
    for (size_t i = 0; i < bs.size(); ++i) {
        std::string utf8;
        append_utf8(&utf8, static_cast<unsigned int>(cs[i]));
        table[bs[i]] = utf8;
    }
    return table;
}

static std::vector<std::string> byte_level_symbols(const char *text) {
    const std::vector<std::string> &table = byte_to_unicode_table();
    std::vector<std::string> symbols;
    if (!text || text[0] == '\0') {
        return symbols;
    }

    const unsigned char *ptr = reinterpret_cast<const unsigned char *>(text);
    while (*ptr != 0) {
        symbols.push_back(table[*ptr]);
        ptr++;
    }
    return symbols;
}

static std::vector<std::vector<std::string>> split_words(const std::vector<std::string> &symbols) {
    const std::string &space_marker = byte_to_unicode_table()[static_cast<unsigned char>(' ')];
    std::vector<std::vector<std::string>> words;
    std::vector<std::string> current;

    for (size_t i = 0; i < symbols.size(); ++i) {
        if (symbols[i] == space_marker && !current.empty()) {
            words.push_back(current);
            current.clear();
        }
        current.push_back(symbols[i]);
    }

    if (!current.empty()) {
        words.push_back(current);
    }
    return words;
}

static std::vector<std::string> apply_bpe(const std::vector<std::string> &word,
                                          const std::unordered_map<std::string, int> &merge_priority) {
    std::vector<std::string> pieces = word;
    while (pieces.size() > 1) {
        int best_rank = INT_MAX;
        int best_pos = -1;

        for (size_t i = 0; i + 1 < pieces.size(); ++i) {
            std::string key = pieces[i];
            key.push_back(' ');
            key += pieces[i + 1];
            std::unordered_map<std::string, int>::const_iterator it = merge_priority.find(key);
            if (it != merge_priority.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = static_cast<int>(i);
            }
        }

        if (best_pos < 0) {
            break;
        }

        pieces[best_pos] += pieces[best_pos + 1];
        pieces.erase(pieces.begin() + best_pos + 1);
    }
    return pieces;
}

static bool load_vocab(Tokenizer *tok, TokenizerInternal *internal, const std::string &path) {
    std::string content;
    if (!read_file(path, &content)) {
        std::fprintf(stderr, "Failed to read vocab: %s\n", path.c_str());
        return false;
    }

    std::vector<std::pair<std::string, int>> entries;
    int max_id = -1;
    if (!parse_vocab_json(content, &entries, &max_id)) {
        std::fprintf(stderr, "Failed to parse vocab json: %s\n", path.c_str());
        return false;
    }
    if (entries.size() > MAX_VOCAB) {
        std::fprintf(stderr, "Vocab too large: %zu > %d\n", entries.size(), MAX_VOCAB);
        return false;
    }

    tok->vocab_count = static_cast<int>(entries.size());
    tok->max_id = max_id;
    tok->vocab_str = static_cast<char **>(std::calloc(entries.size(), sizeof(char *)));
    tok->vocab_id = static_cast<int *>(std::calloc(entries.size(), sizeof(int)));
    tok->id_to_str = static_cast<char **>(std::calloc(static_cast<size_t>(max_id + 1), sizeof(char *)));
    if (!tok->vocab_str || !tok->vocab_id || !tok->id_to_str) {
        std::fprintf(stderr, "Failed to allocate tokenizer vocab tables\n");
        return false;
    }

    internal->vocab_lookup.reserve(entries.size() * 2);
    for (size_t i = 0; i < entries.size(); ++i) {
        tok->vocab_str[i] = dup_cstr(entries[i].first);
        if (!tok->vocab_str[i]) {
            std::fprintf(stderr, "Failed to allocate vocab string\n");
            return false;
        }
        tok->vocab_id[i] = entries[i].second;
        if (entries[i].second >= 0 && entries[i].second <= tok->max_id) {
            tok->id_to_str[entries[i].second] = tok->vocab_str[i];
        }
        internal->vocab_lookup[entries[i].first] = entries[i].second;
    }
    return true;
}

static bool load_merges(Tokenizer *tok, TokenizerInternal *internal, const std::string &path) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "Failed to open merges: %s\n", path.c_str());
        return false;
    }

    std::vector<std::pair<std::string, std::string>> merges;
    merges.reserve(160000);

    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }

        size_t split = line.find(' ');
        if (split == std::string::npos || split == 0 || split + 1 >= line.size()) {
            continue;
        }

        std::string left = line.substr(0, split);
        std::string right = line.substr(split + 1);
        if (left.size() >= 256 || right.size() >= 256) {
            std::fprintf(stderr, "Merge token exceeds fixed buffer width\n");
            return false;
        }

        merges.push_back(std::make_pair(left, right));
        if (merges.size() > MAX_MERGES) {
            std::fprintf(stderr, "Too many merges: %zu > %d\n", merges.size(), MAX_MERGES);
            return false;
        }
    }

    tok->merge_count = static_cast<int>(merges.size());
    tok->merge_left = new char[merges.size()][256];
    tok->merge_right = new char[merges.size()][256];
    internal->merge_priority.reserve(merges.size() * 2);
    for (size_t i = 0; i < merges.size(); ++i) {
        std::snprintf(tok->merge_left[i], 256, "%s", merges[i].first.c_str());
        std::snprintf(tok->merge_right[i], 256, "%s", merges[i].second.c_str());
        std::string key = merges[i].first;
        key.push_back(' ');
        key += merges[i].second;
        internal->merge_priority[key] = static_cast<int>(i);
    }
    return true;
}

static bool load_special_ids(Tokenizer *tok, const std::string &path) {
    std::string content;
    if (!read_file(path, &content)) {
        std::fprintf(stderr, "Failed to read tokenizer config: %s\n", path.c_str());
        return false;
    }
    if (!parse_added_token_ids(content,
                               &tok->audio_start_id,
                               &tok->audio_end_id,
                               &tok->audio_pad_id,
                               &tok->timestamp_id)) {
        std::fprintf(stderr, "Failed to parse tokenizer config: %s\n", path.c_str());
        return false;
    }
    if (tok->audio_start_id < 0 || tok->audio_end_id < 0 ||
        tok->audio_pad_id < 0 || tok->timestamp_id < 0) {
        std::fprintf(stderr, "Missing required special token IDs in: %s\n", path.c_str());
        return false;
    }
    return true;
}

}  // namespace

int tokenizer_load(Tokenizer *tok, const char *model_dir) {
    if (!tok || !model_dir) {
        return -1;
    }

    std::memset(tok, 0, sizeof(*tok));
    tok->audio_start_id = -1;
    tok->audio_end_id = -1;
    tok->audio_pad_id = -1;
    tok->timestamp_id = -1;

    TokenizerInternal internal;
    if (!load_vocab(tok, &internal, join_path(model_dir, "vocab.json")) ||
        !load_merges(tok, &internal, join_path(model_dir, "merges.txt")) ||
        !load_special_ids(tok, join_path(model_dir, "tokenizer_config.json"))) {
        tokenizer_free(tok);
        return -1;
    }

    g_tokenizer_data[tok] = std::move(internal);
    return 0;
}

int tokenizer_encode(const Tokenizer *tok, const char *text, int *out_ids, int max_ids) {
    if (!tok || !text || !out_ids || max_ids <= 0) {
        return -1;
    }
    if (text[0] == '\0') {
        return 0;
    }

    std::unordered_map<const Tokenizer *, TokenizerInternal>::const_iterator data_it = g_tokenizer_data.find(tok);
    if (data_it == g_tokenizer_data.end()) {
        return -1;
    }

    std::vector<std::string> symbols = byte_level_symbols(text);
    std::vector<std::vector<std::string>> words = split_words(symbols);

    int out_count = 0;
    for (size_t i = 0; i < words.size(); ++i) {
        std::vector<std::string> pieces = apply_bpe(words[i], data_it->second.merge_priority);
        for (size_t j = 0; j < pieces.size(); ++j) {
            std::unordered_map<std::string, int>::const_iterator vocab_it = data_it->second.vocab_lookup.find(pieces[j]);
            if (vocab_it == data_it->second.vocab_lookup.end()) {
                std::fprintf(stderr, "Missing vocab token during BPE encode\n");
                return -1;
            }
            if (out_count >= max_ids || out_count >= MAX_TOKENS) {
                return -1;
            }
            out_ids[out_count++] = vocab_it->second;
        }
    }

    return out_count;
}

const char *tokenizer_id_to_str(const Tokenizer *tok, int token_id) {
    if (!tok || !tok->id_to_str || token_id < 0 || token_id > tok->max_id) {
        return "?";
    }
    const char *s = tok->id_to_str[token_id];
    return s ? s : "?";
}

const char *tokenizer_id_to_display_str(const Tokenizer *tok, int token_id) {
    static std::unordered_map<int, std::string> cache;
    auto it = cache.find(token_id);
    if (it != cache.end()) return it->second.c_str();
    const char *raw = tokenizer_id_to_str(tok, token_id);
    if (!raw || raw[0] == '?') { cache[token_id] = "?"; return "?"; }
    std::string decoded = byte_level_decode(raw);
    cache[token_id] = decoded;
    return cache[token_id].c_str();
}

int tokenizer_token_starts_word(const Tokenizer *tok, int token_id) {
    const char *s = tokenizer_id_to_str(tok, token_id);
    if (!s || s[0] == '?') return 0;
    unsigned char c = (unsigned char)s[0];
    return c == 0xC4 && s[1] == (char)0xA0;
}

void tokenizer_free(Tokenizer *tok) {
    if (!tok) {
        return;
    }

    g_tokenizer_data.erase(tok);

    if (tok->vocab_str) {
        for (int i = 0; i < tok->vocab_count; ++i) {
            std::free(tok->vocab_str[i]);
        }
        std::free(tok->vocab_str);
    }
    std::free(tok->vocab_id);
    std::free(tok->id_to_str);
    delete[] tok->merge_left;
    delete[] tok->merge_right;

    tok->vocab_str = nullptr;
    tok->vocab_id = nullptr;
    tok->vocab_count = 0;
    tok->merge_left = nullptr;
    tok->merge_right = nullptr;
    tok->merge_count = 0;
    tok->id_to_str = nullptr;
    tok->max_id = 0;
    tok->audio_start_id = -1;
    tok->audio_end_id = -1;
    tok->audio_pad_id = -1;
    tok->timestamp_id = -1;
}
