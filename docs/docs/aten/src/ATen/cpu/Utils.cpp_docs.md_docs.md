# Documentation: `docs/aten/src/ATen/cpu/Utils.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/cpu/Utils.cpp_docs.md`
- **Size**: 5,101 bytes (4.98 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/cpu/Utils.cpp`

## File Metadata

- **Path**: `aten/src/ATen/cpu/Utils.cpp`
- **Size**: 3,010 bytes (2.94 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <cassert>
#include <ATen/cpu/Utils.h>
#if !defined(__s390x__ ) && !defined(__powerpc__)
#include <cpuinfo.h>
#endif
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace at::cpu {
bool is_avx2_supported() {
#if !defined(__s390x__) && !defined(__powerpc__)
  return cpuinfo_initialize() && cpuinfo_has_x86_avx2();
#else
  return false;
#endif
}

bool is_avx512_supported() {
#if !defined(__s390x__) && !defined(__powerpc__)
  return cpuinfo_initialize() && cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512vl() && cpuinfo_has_x86_avx512bw() && cpuinfo_has_x86_avx512dq();
#else
  return false;
#endif
}

bool is_avx512_vnni_supported() {
#if !defined(__s390x__) && !defined(__powerpc__)
  return cpuinfo_initialize() && cpuinfo_has_x86_avx512vnni();
#else
  return false;
#endif
}

bool is_avx512_bf16_supported() {
#if !defined(__s390x__) && !defined(__powerpc__)
  return cpuinfo_initialize() && cpuinfo_has_x86_avx512bf16();
#else
  return false;
#endif
}

bool is_amx_tile_supported() {
#if !defined(__s390x__) && !defined(__powerpc__)
  return cpuinfo_initialize() && cpuinfo_has_x86_amx_tile();
#else
  return false;
#endif
}

bool is_amx_fp16_supported() {
#if !defined(__s390x__) && !defined(__powerpc__)
  return is_amx_tile_supported() && cpuinfo_has_x86_amx_fp16();
#else
  return false;
#endif
}

bool init_amx() {
  if (!is_amx_tile_supported()) {
    return false;
  }

#if defined(__linux__) && !defined(__ANDROID__) && defined(__x86_64__)
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

  unsigned long bitmask = 0;
  // Request permission to use AMX instructions
  long rc = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
  if (rc) {
      return false;
  }
  // Check if the system supports AMX instructions
  rc = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
  if (rc) {
      return false;
  }
  if (bitmask & XFEATURE_MASK_XTILE) {
      return true;
  }
  return false;
#else
  return true;
#endif
}

static uint32_t get_cache_size(int level) {
#if !defined(__s390x__) && !defined(__powerpc__)
  if (!cpuinfo_initialize()) {
    return 0;
  }
  const struct cpuinfo_processor* processors = cpuinfo_get_processors();
  if (!processors) {
    return 0;
  }
  const struct cpuinfo_cache* cache = nullptr;
  switch (level) {
    case 1:
      cache = processors[0].cache.l1d;
      break;
    case 2:
      cache = processors[0].cache.l2;
      break;
    default:
      assert(false && "Unsupported cache level");
  }

  if (!cache) {
    return 0;
  }
  return cache->size;
#else
  return 0;
#endif
}

uint32_t L1d_cache_size() {
  return get_cache_size(1);
}

uint32_t L2_cache_size() {
  return get_cache_size(2);
}

} // namespace at::cpu

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 23 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `cpuinfo_processor`, `cpuinfo_cache`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `cassert`
- `ATen/cpu/Utils.h`
- `cpuinfo.h`
- `sys/syscall.h`
- `unistd.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/cpu`):

- [`vml.h_docs.md`](./vml.h_docs.md)
- [`FlushDenormal.cpp_docs.md`](./FlushDenormal.cpp_docs.md)
- [`Utils.h_docs.md`](./Utils.h_docs.md)
- [`FlushDenormal.h_docs.md`](./FlushDenormal.h_docs.md)


## Cross-References

- **File Documentation**: `Utils.cpp_docs.md`
- **Keyword Index**: `Utils.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/cpu`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/aten/src/ATen/cpu`):

- [`FlushDenormal.cpp_docs.md_docs.md`](./FlushDenormal.cpp_docs.md_docs.md)
- [`FlushDenormal.cpp_kw.md_docs.md`](./FlushDenormal.cpp_kw.md_docs.md)
- [`FlushDenormal.h_docs.md_docs.md`](./FlushDenormal.h_docs.md_docs.md)
- [`vml.h_kw.md_docs.md`](./vml.h_kw.md_docs.md)
- [`Utils.cpp_kw.md_docs.md`](./Utils.cpp_kw.md_docs.md)
- [`FlushDenormal.h_kw.md_docs.md`](./FlushDenormal.h_kw.md_docs.md)
- [`Utils.h_kw.md_docs.md`](./Utils.h_kw.md_docs.md)
- [`Utils.h_docs.md_docs.md`](./Utils.h_docs.md_docs.md)
- [`vml.h_docs.md_docs.md`](./vml.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `Utils.cpp_docs.md_docs.md`
- **Keyword Index**: `Utils.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
