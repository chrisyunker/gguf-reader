#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <dirent.h>
#include <pwd.h>
#include <sys/stat.h>
#include <unistd.h>

static const uint32_t GGUF_MAGIC = 0x46554747; // "GGUF" little-endian

enum GgufValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

template<typename T>
static T read_val(FILE* f) {
    T v;
    if (fread(&v, sizeof(T), 1, f) != 1) {
        fprintf(stderr, "Error: unexpected end of file\n");
        exit(1);
    }
    return v;
}

static std::string read_string(FILE* f) {
    uint64_t len = read_val<uint64_t>(f);
    std::string s(len, '\0');
    if (fread(s.data(), 1, len, f) != len) {
        fprintf(stderr, "Error: unexpected end of file reading string\n");
        exit(1);
    }
    return s;
}

static const char* token_type_name(int32_t t) {
    switch (t) {
        case 0: return "NORMAL";
        case 1: return "UNKNOWN";
        case 2: return "CONTROL";
        case 3: return "USER_DEFINED";
        case 4: return "UNUSED";
        case 5: return "BYTE";
        default: return "?";
    }
}

static const char* file_type_name(int32_t t) {
    switch (t) {
        case  0: return "F32";
        case  1: return "F16";
        case  2: return "Q4_0";
        case  3: return "Q4_1";
        case  6: return "Q5_0";
        case  7: return "Q5_1";
        case  8: return "Q8_0";
        case  9: return "Q8_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K";
        case 12: return "Q4_K";
        case 13: return "Q5_K";
        case 14: return "Q6_K";
        case 15: return "Q8_K";
        case 16: return "IQ2_XXS";
        case 17: return "IQ2_XS";
        case 18: return "IQ3_XXS";
        case 19: return "IQ1_S";
        case 20: return "IQ4_NL";
        case 21: return "IQ3_S";
        case 22: return "IQ2_S";
        case 23: return "IQ4_XS";
        case 24: return "I8";
        case 25: return "I16";
        case 26: return "I32";
        case 27: return "I64";
        case 28: return "F64";
        case 29: return "IQ1_M";
        case 30: return "BF16";
        default: return "?";
    }
}

static void skip_value(FILE* f, GgufValueType type) {
    switch (type) {
        case UINT8:  case INT8:  case BOOL:    fseek(f, 1, SEEK_CUR); break;
        case UINT16: case INT16:               fseek(f, 2, SEEK_CUR); break;
        case UINT32: case INT32: case FLOAT32: fseek(f, 4, SEEK_CUR); break;
        case UINT64: case INT64: case FLOAT64: fseek(f, 8, SEEK_CUR); break;
        case STRING: {
            uint64_t len = read_val<uint64_t>(f);
            fseek(f, (long)len, SEEK_CUR);
            break;
        }
        case ARRAY: {
            auto elem_type = read_val<GgufValueType>(f);
            uint64_t count = read_val<uint64_t>(f);
            for (uint64_t i = 0; i < count; i++) skip_value(f, elem_type);
            break;
        }
        default: break;
    }
}

static void print_scalar(FILE* f, GgufValueType type) {
    switch (type) {
        case UINT8:   printf("%u",   read_val<uint8_t>(f));  break;
        case INT8:    printf("%d",   read_val<int8_t>(f));   break;
        case UINT16:  printf("%u",   read_val<uint16_t>(f)); break;
        case INT16:   printf("%d",   read_val<int16_t>(f));  break;
        case UINT32:  printf("%u",   read_val<uint32_t>(f)); break;
        case INT32:   printf("%d",   read_val<int32_t>(f));  break;
        case FLOAT32: printf("%g",   read_val<float>(f));    break;
        case BOOL:    printf("%s",   read_val<uint8_t>(f) ? "true" : "false"); break;
        case STRING:  printf("\"%s\"", read_string(f).c_str()); break;
        case UINT64:  printf("%llu", (unsigned long long)read_val<uint64_t>(f)); break;
        case INT64:   printf("%lld", (long long)read_val<int64_t>(f));           break;
        case FLOAT64: printf("%g",   read_val<double>(f));   break;
        default:      printf("<unknown type %u>", type);     break;
    }
}

static void print_value(FILE* f, GgufValueType type) {
    if (type != ARRAY) {
        print_scalar(f, type);
        return;
    }

    static const uint64_t MAX_PRINT = 10;
    auto elem_type  = read_val<GgufValueType>(f);
    uint64_t count  = read_val<uint64_t>(f);

    printf("[");
    for (uint64_t i = 0; i < count; i++) {
        if (i == MAX_PRINT) {
            printf("... %llu more", (unsigned long long)(count - MAX_PRINT));
            for (uint64_t j = i; j < count; j++) skip_value(f, elem_type);
            break;
        }
        if (i > 0) printf(", ");
        print_scalar(f, elem_type);
    }
    printf("] (count=%llu)", (unsigned long long)count);
}

static std::string home_dir() {
    const char* h = getenv("HOME");
    if (h) return h;
    struct passwd* pw = getpwuid(getuid());
    return pw ? pw->pw_dir : "";
}

// Returns the resolved .gguf path for a HuggingFace model ID (e.g. "org/model").
// Exits with an error message if the model or snapshot can't be found.
// If multiple .gguf files exist, lists them and exits so the user can pick.
static std::string resolve_hf_path(const char* model_id) {
    // Build cache dir: ~/.cache/huggingface/hub/models--org--model
    std::string repo;
    for (const char* p = model_id; *p; p++)
        repo += (*p == '/') ? "--" : std::string(1, *p);
    std::string dir = home_dir() + "/.cache/huggingface/hub/models--" + repo + "/snapshots";

    // Find snapshot subdirs, pick the most recently modified
    DIR* d = opendir(dir.c_str());
    if (!d) {
        fprintf(stderr, "Error: no HuggingFace cache found for '%s'\n  (looked in %s)\n", model_id, dir.c_str());
        exit(1);
    }
    std::string best_snap;
    time_t best_mtime = 0;
    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        if (ent->d_name[0] == '.') continue;
        std::string snap = dir + "/" + ent->d_name;
        struct stat st;
        if (stat(snap.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            if (st.st_mtime > best_mtime) { best_mtime = st.st_mtime; best_snap = snap; }
        }
    }
    closedir(d);

    if (best_snap.empty()) {
        fprintf(stderr, "Error: no snapshots found for '%s'\n", model_id);
        exit(1);
    }

    // Collect .gguf files in the snapshot dir
    std::vector<std::string> gguf_files;
    d = opendir(best_snap.c_str());
    if (d) {
        while ((ent = readdir(d)) != nullptr) {
            std::string name = ent->d_name;
            if (name.size() > 5 && name.substr(name.size() - 5) == ".gguf")
                gguf_files.push_back(best_snap + "/" + name);
        }
        closedir(d);
    }

    if (gguf_files.empty()) {
        fprintf(stderr, "Error: no .gguf files found for '%s'\n  (looked in %s)\n", model_id, best_snap.c_str());
        exit(1);
    }
    if (gguf_files.size() == 1) return gguf_files[0];

    // Multiple files: list them for the user to choose
    fprintf(stderr, "Multiple .gguf files found for '%s':\n", model_id);
    for (const auto& p : gguf_files) fprintf(stderr, "  %s\n", p.c_str());
    fprintf(stderr, "Pass the full path directly to select one.\n");
    exit(1);
}

static void print_tokens(FILE* f, uint64_t metadata_count) {
    fpos_t after_header;
    fgetpos(f, &after_header);
    std::vector<int32_t> token_types;
    for (uint64_t i = 0; i < metadata_count; i++) {
        std::string key = read_string(f);
        auto val_type = read_val<GgufValueType>(f);
        if (key == "tokenizer.ggml.token_type" && val_type == ARRAY) {
            auto elem_type = read_val<GgufValueType>(f);
            uint64_t count = read_val<uint64_t>(f);
            token_types.reserve(count);
            for (uint64_t j = 0; j < count; j++) {
                if (elem_type == INT32) {
                    token_types.push_back(read_val<int32_t>(f));
                } else {
                    skip_value(f, elem_type);
                }
            }
            break;
        }
        skip_value(f, val_type);
    }

    fsetpos(f, &after_header);
    for (uint64_t i = 0; i < metadata_count; i++) {
        std::string key = read_string(f);
        auto val_type = read_val<GgufValueType>(f);
        if (key == "tokenizer.ggml.tokens" && val_type == ARRAY) {
            auto elem_type = read_val<GgufValueType>(f);
            uint64_t count = read_val<uint64_t>(f);
            int width = (count > 0) ? snprintf(nullptr, 0, "%llu", (unsigned long long)(count - 1)) : 1;
            for (uint64_t j = 0; j < count; j++) {
                if (elem_type == STRING) {
                    const char* tname = j < token_types.size() ? token_type_name(token_types[j]) : "";
                    printf("%*llu: %-12s %s\n", width, (unsigned long long)j, tname, read_string(f).c_str());
                } else {
                    skip_value(f, elem_type);
                }
            }
            break;
        }
        skip_value(f, val_type);
    }
}

static void print_merges(FILE* f, uint64_t metadata_count) {
    for (uint64_t i = 0; i < metadata_count; i++) {
        std::string key = read_string(f);
        auto val_type = read_val<GgufValueType>(f);
        if (key == "tokenizer.ggml.merges" && val_type == ARRAY) {
            auto elem_type = read_val<GgufValueType>(f);
            uint64_t count = read_val<uint64_t>(f);
            int width = (count > 0) ? snprintf(nullptr, 0, "%llu", (unsigned long long)(count - 1)) : 1;
            for (uint64_t j = 0; j < count; j++) {
                if (elem_type == STRING) {
                    printf("%*llu: %s\n", width, (unsigned long long)j, read_string(f).c_str());
                } else {
                    skip_value(f, elem_type);
                }
            }
            break;
        }
        skip_value(f, val_type);
    }
}

static void print_metadata(FILE* f, uint64_t metadata_count) {
    for (uint64_t i = 0; i < metadata_count; i++) {
        std::string key = read_string(f);
        auto val_type = read_val<GgufValueType>(f);

        printf("%-48s = ", key.c_str());
        if (key == "general.file_type" && (val_type == INT32 || val_type == UINT32)) {
            printf("%s", file_type_name((int32_t)read_val<uint32_t>(f)));
        } else {
            print_value(f, val_type);
        }
        printf("\n");        
    }
}

static void print_tensors(FILE* f, uint64_t tensor_count) {
    if (tensor_count == 0) {
        return;
    }

    std::map<std::string, uint64_t> type_counts;
    std::map<std::string, uint64_t> shape_counts;
    for (uint64_t i = 0; i < tensor_count; i++) {
        read_string(f);                          // name
        uint32_t n_dims = read_val<uint32_t>(f);
        std::string shape = "[";
        for (uint32_t d = 0; d < n_dims; d++) {
            uint64_t dim = read_val<uint64_t>(f);
            if (d > 0) shape += ", ";
            char buf[32];
            snprintf(buf, sizeof(buf), "%llu", (unsigned long long)dim);
            shape += buf;
        }
        shape += "]";
        uint32_t ttype = read_val<uint32_t>(f);
        read_val<uint64_t>(f);                   // offset
        type_counts[file_type_name((int32_t)ttype)]++;
        shape_counts[shape]++; 
    }

    auto make_sorted = [](const std::map<std::string, uint64_t>& m) {
        std::vector<std::pair<uint64_t, std::string>> v;
        for (const auto& kv : m) v.push_back({kv.second, kv.first});
        std::sort(v.rbegin(), v.rend());
        return v;
    };

    printf("\nTensor types:\n");
    for (const auto& p : make_sorted(type_counts))
        printf("  %-8s: %llu\n", p.second.c_str(), (unsigned long long)p.first);

    printf("\nTensor shapes:\n");
    for (const auto& p : make_sorted(shape_counts))
        printf("  %-24s: %llu\n", p.second.c_str(), (unsigned long long)p.first);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s [--tokens] [--merges] [--hf <model-id>] <file.gguf>\n", argv[0]);
        return 1;
    }

    bool dump_tokens = false;
    bool dump_merges = false;
    const char* hf_model  = nullptr;
    const char* filename  = nullptr;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--tokens") == 0)
            dump_tokens = true;
        else if (strcmp(argv[i], "--merges") == 0)
            dump_merges = true;
        else if ((strcmp(argv[i], "--hf") == 0 || strcmp(argv[i], "-hf") == 0) && i + 1 < argc)
            hf_model = argv[++i];
        else
            filename = argv[i];
    }

    std::string resolved_path;
    if (hf_model) {
        resolved_path = resolve_hf_path(hf_model);
        filename = resolved_path.c_str();
    }

    if (!filename) {
        fprintf(stderr, "Usage: %s [--tokens] [--merges] [--hf <model-id>] <file.gguf>\n", argv[0]);
        return 1;
    }

    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open file '%s'\n", filename);
        return 1;
    }

    uint32_t magic = read_val<uint32_t>(f);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "Error: not a GGUF file (bad magic)\n");
        fclose(f);
        return 1;
    }

    uint32_t version        = read_val<uint32_t>(f);
    uint64_t tensor_count   = read_val<uint64_t>(f);
    uint64_t metadata_count = read_val<uint64_t>(f);

    if (!dump_tokens && !dump_merges) {
        printf("GGUF version:   %u\n", version);
        printf("Tensor count:   %llu\n", (unsigned long long)tensor_count);
        printf("Metadata count: %llu\n\n", (unsigned long long)metadata_count);
    }

    if (dump_tokens) {
        print_tokens(f, metadata_count);
    } else if (dump_merges) {
        print_merges(f, metadata_count);
    } else {
        print_metadata(f, metadata_count);
        print_tensors(f, tensor_count);
    }

    fclose(f);
    return 0;
}
