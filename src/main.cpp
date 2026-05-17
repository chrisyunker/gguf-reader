#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

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

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <file.gguf> [--tokens]\n", argv[0]);
        return 1;
    }

    bool dump_tokens = (argc == 3 && strcmp(argv[2], "--tokens") == 0);
    if (argc == 3 && !dump_tokens) {
        fprintf(stderr, "Usage: %s <file.gguf> [--tokens]\n", argv[0]);
        return 1;
    }

    FILE* f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open file '%s'\n", argv[1]);
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

    if (!dump_tokens) {
        printf("GGUF version:   %u\n", version);
        printf("Tensor count:   %llu\n", (unsigned long long)tensor_count);
        printf("Metadata count: %llu\n\n", (unsigned long long)metadata_count);
    }

    for (uint64_t i = 0; i < metadata_count; i++) {
        std::string key = read_string(f);
        auto val_type   = read_val<GgufValueType>(f);

        if (dump_tokens) {
            if (key == "tokenizer.ggml.tokens" && val_type == ARRAY) {
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
        } else {
            printf("%-48s = ", key.c_str());
            print_value(f, val_type);
            printf("\n");
        }
    }

    fclose(f);
    return 0;
}
