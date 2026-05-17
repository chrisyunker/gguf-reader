#include <cstdint>
#include <cstdio>
#include <cstring>
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

// Forward declaration
static void skip_value(FILE* f, GgufValueType type);

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

    fpos_t after_header;
    fgetpos(f, &after_header);

    if (dump_tokens) {
        // First pass: collect token types
        std::vector<int32_t> token_types;
        for (uint64_t i = 0; i < metadata_count; i++) {
            std::string key = read_string(f);
            auto val_type   = read_val<GgufValueType>(f);
            if (key == "tokenizer.ggml.token_type" && val_type == ARRAY) {
                auto elem_type = read_val<GgufValueType>(f);
                uint64_t count = read_val<uint64_t>(f);
                token_types.reserve(count);
                for (uint64_t j = 0; j < count; j++) {
                    if (elem_type == INT32)
                        token_types.push_back(read_val<int32_t>(f));
                    else
                        skip_value(f, elem_type);
                }
                break;
            }
            skip_value(f, val_type);
        }

        // Second pass: print tokens with types
        fsetpos(f, &after_header);
        for (uint64_t i = 0; i < metadata_count; i++) {
            std::string key = read_string(f);
            auto val_type   = read_val<GgufValueType>(f);
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
    } else if (dump_merges) {
        for (uint64_t i = 0; i < metadata_count; i++) {
            std::string key = read_string(f);
            auto val_type   = read_val<GgufValueType>(f);
            if (key == "tokenizer.ggml.merges" && val_type == ARRAY) {
                auto elem_type = read_val<GgufValueType>(f);
                uint64_t count = read_val<uint64_t>(f);
                int width = (count > 0) ? snprintf(nullptr, 0, "%llu", (unsigned long long)(count - 1)) : 1;
                for (uint64_t j = 0; j < count; j++) {
                    if (elem_type == STRING)
                        printf("%*llu: %s\n", width, (unsigned long long)j, read_string(f).c_str());
                    else
                        skip_value(f, elem_type);
                }
                break;
            }
            skip_value(f, val_type);
        }
    } else {
        for (uint64_t i = 0; i < metadata_count; i++) {
            std::string key = read_string(f);
            auto val_type   = read_val<GgufValueType>(f);
            printf("%-48s = ", key.c_str());
            print_value(f, val_type);
            printf("\n");
        }
    }

    fclose(f);
    return 0;
}
