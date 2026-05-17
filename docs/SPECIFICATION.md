# GGUF Reader - Specification

## Overview

A C++ command-line tool that reads a GGUF file and extracts/displays its metadata.

## GGUF File Format Summary

GGUF (GGML Unified Format) is a binary format for storing LLM model weights and metadata.

### Header (28 bytes)
| Field           | Type    | Size  |
|-----------------|---------|-------|
| Magic           | char[4] | 4     |
| Version         | uint32  | 4     |
| Tensor count    | uint64  | 8     |
| Metadata count  | uint64  | 8     |

### Metadata Key-Value Pairs
Each pair:
1. Key: uint64 length + UTF-8 bytes
2. Value type: uint32 enum
3. Value: encoded per type

### Metadata Value Types
| Code | Type    |
|------|---------|
| 0    | UINT8   |
| 1    | INT8    |
| 2    | UINT16  |
| 3    | INT16   |
| 4    | UINT32  |
| 5    | INT32   |
| 6    | FLOAT32 |
| 7    | BOOL    |
| 8    | STRING  |
| 9    | ARRAY   |
| 10   | UINT64  |
| 11   | INT64   |
| 12   | FLOAT64 |

## Build System

CMake-based build. Target: `gguf-reader` executable.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Usage

```
./build/gguf-reader <path-to-gguf-file>
```

Outputs all metadata key-value pairs to stdout. Arrays longer than 10 elements are truncated with a count shown.
