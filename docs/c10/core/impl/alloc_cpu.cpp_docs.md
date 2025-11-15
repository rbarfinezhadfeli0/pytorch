# Documentation: `c10/core/impl/alloc_cpu.cpp`

## File Metadata

- **Path**: `c10/core/impl/alloc_cpu.cpp`
- **Size**: 5,469 bytes (5.34 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/core/impl/alloc_cpu.h>

#include <c10/core/alignment.h>
#include <c10/util/Flags.h>
#include <c10/util/Logging.h>
#include <c10/util/env.h>
#include <c10/util/error.h>
#include <c10/util/irange.h>
#include <c10/util/numa.h>
#include <cstring>

#ifdef USE_MIMALLOC
#include <mimalloc.h>
#endif

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

// TODO: rename flags to C10
// NOLINTNEXTLINE(misc-use-internal-linkage)
C10_DEFINE_bool(
    caffe2_cpu_allocator_do_zero_fill,
    false,
    "If set, do memory zerofilling when allocating on CPU")

// NOLINTNEXTLINE(misc-use-internal-linkage)
C10_DEFINE_bool(
    caffe2_cpu_allocator_do_junk_fill,
    false,
    "If set, fill memory with deterministic junk when allocating on CPU")

namespace c10 {

namespace {

// Fill the data memory region of num bytes with a particular garbage pattern.
// The garbage value is chosen to be NaN if interpreted as floating point value,
// or a very large integer.
void memset_junk(void* data, size_t num) {
  // This garbage pattern is NaN when interpreted as floating point values,
  // or as very large integer values.
  static constexpr int32_t kJunkPattern = 0x7fedbeef;
  static constexpr int64_t kJunkPattern64 =
      static_cast<int64_t>(kJunkPattern) << 32 | kJunkPattern;
  auto int64_count = num / sizeof(kJunkPattern64);
  auto remaining_bytes = num % sizeof(kJunkPattern64);
  int64_t* data_i64 = reinterpret_cast<int64_t*>(data);
  for (const auto i : c10::irange(int64_count)) {
    data_i64[i] = kJunkPattern64;
  }
  if (remaining_bytes > 0) {
    memcpy(data_i64 + int64_count, &kJunkPattern64, remaining_bytes);
  }
}

#if defined(__linux__) && !defined(__ANDROID__)
inline bool is_thp_alloc_enabled() {
  static bool value = [&] {
    auto env = c10::utils::check_env("THP_MEM_ALLOC_ENABLE");
    return env.has_value() ? env.value() : 0;
  }();
  return value;
}

inline bool is_thp_alloc(size_t nbytes) {
  // enable thp (transparent huge pages) for larger buffers
  return (is_thp_alloc_enabled() && (nbytes >= gAlloc_threshold_thp));
}

#elif !defined(__ANDROID__) && !defined(_MSC_VER)
constexpr size_t c10_compute_alignment(size_t /*nbytes*/) {
  return gAlignment;
}

constexpr bool is_thp_alloc([[maybe_unused]] size_t nbytes) {
  return false;
}
#endif
} // namespace

#if defined(__linux__) && !defined(__ANDROID__)
size_t c10_compute_alignment(size_t nbytes) {
  static const auto pagesize = sysconf(_SC_PAGESIZE);
  // for kernels that don't provide page size, default it to 4K
  const size_t thp_alignment = (pagesize < 0 ? gPagesize : pagesize);
  return (is_thp_alloc(nbytes) ? thp_alignment : gAlignment);
}
#endif

void* alloc_cpu(size_t nbytes) {
  if (nbytes == 0) {
    return nullptr;
  }
  // We might have clowny upstream code that tries to alloc a negative number
  // of bytes. Let's catch it early.
  CAFFE_ENFORCE(
      ((ptrdiff_t)nbytes) >= 0,
      "alloc_cpu() seems to have been called with negative number: ",
      nbytes);

  void* data = nullptr;
#ifdef __ANDROID__
  data = memalign(gAlignment, nbytes);
  CAFFE_ENFORCE(
      data,
      "DefaultCPUAllocator: not enough memory: you tried to allocate ",
      nbytes,
      " bytes.");
#elif defined(USE_MIMALLOC)
  data = mi_malloc_aligned(nbytes, gAlignment);
  CAFFE_ENFORCE(
      data,
      "DefaultCPUAllocator: not enough memory: you tried to allocate ",
      nbytes,
      " bytes.");
#elif defined(_MSC_VER)
  data = _aligned_malloc(nbytes, gAlignment);
  CAFFE_ENFORCE(
      data,
      "DefaultCPUAllocator: not enough memory: you tried to allocate ",
      nbytes,
      " bytes.");
#else
  int err = posix_memalign(&data, c10_compute_alignment(nbytes), nbytes);
  CAFFE_ENFORCE(
      err == 0,
      "DefaultCPUAllocator: can't allocate memory: you tried to allocate ",
      nbytes,
      " bytes. Error code ",
      err,
      " (",
      c10::utils::str_error(err),
      ")");
  if (is_thp_alloc(nbytes)) {
#ifdef __linux__
    // MADV_HUGEPAGE advise is available only for linux.
    // general posix compliant systems can check POSIX_MADV_SEQUENTIAL advise.
    int ret = madvise(data, nbytes, MADV_HUGEPAGE);
    if (ret != 0) {
      TORCH_WARN_ONCE(
          "thp madvise for HUGEPAGE failed with ",
          c10::utils::str_error(errno));
    }
#endif
  }
#endif

  // move data to a thread's NUMA node
  NUMAMove(data, nbytes, GetCurrentNUMANode());
  CHECK(
      !FLAGS_caffe2_cpu_allocator_do_zero_fill ||
      !FLAGS_caffe2_cpu_allocator_do_junk_fill)
      << "Cannot request both zero-fill and junk-fill at the same time";
  if (FLAGS_caffe2_cpu_allocator_do_zero_fill) {
    memset(data, 0, nbytes);
  } else if (FLAGS_caffe2_cpu_allocator_do_junk_fill) {
    memset_junk(data, nbytes);
  }

  return data;
}

void free_cpu(void* data) {
#ifdef USE_MIMALLOC
  mi_free(data);
#elif defined(_MSC_VER)
  _aligned_free(data);
#else
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  free(data);
#endif
}

#ifdef USE_MIMALLOC_ON_MKL
namespace mi_malloc_wrapper {
void* c10_mi_malloc(size_t size) {
  return mi_malloc(size);
}

void* c10_mi_calloc(size_t count, size_t size) {
  return mi_calloc(count, size);
}

void* c10_mi_realloc(void* p, size_t newsize) {
  return mi_realloc(p, newsize);
}

void* c10_mi_malloc_aligned(size_t size, size_t alignment) {
  return mi_malloc_aligned(size, alignment);
}

void c10_mi_free(void* p) {
  mi_free(p);
}
} // namespace mi_malloc_wrapper
#endif
} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 22 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `mi_malloc_wrapper`, `c10`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/core/impl`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/impl/alloc_cpu.h`
- `c10/core/alignment.h`
- `c10/util/Flags.h`
- `c10/util/Logging.h`
- `c10/util/env.h`
- `c10/util/error.h`
- `c10/util/irange.h`
- `c10/util/numa.h`
- `cstring`
- `mimalloc.h`
- `sys/mman.h`
- `unistd.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`c10/core/impl`):

- [`LocalDispatchKeySet.h_docs.md`](./LocalDispatchKeySet.h_docs.md)
- [`TorchDispatchModeTLS.cpp_docs.md`](./TorchDispatchModeTLS.cpp_docs.md)
- [`PythonDispatcherTLS.h_docs.md`](./PythonDispatcherTLS.h_docs.md)
- [`PyInterpreter.h_docs.md`](./PyInterpreter.h_docs.md)
- [`alloc_cpu.h_docs.md`](./alloc_cpu.h_docs.md)
- [`PythonDispatcherTLS.cpp_docs.md`](./PythonDispatcherTLS.cpp_docs.md)
- [`InlineEvent.h_docs.md`](./InlineEvent.h_docs.md)
- [`PyInterpreterHooks.h_docs.md`](./PyInterpreterHooks.h_docs.md)
- [`LocalDispatchKeySet.cpp_docs.md`](./LocalDispatchKeySet.cpp_docs.md)
- [`DeviceGuardImplInterface.cpp_docs.md`](./DeviceGuardImplInterface.cpp_docs.md)


## Cross-References

- **File Documentation**: `alloc_cpu.cpp_docs.md`
- **Keyword Index**: `alloc_cpu.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
