# Documentation: `c10/util/numa.cpp`

## File Metadata

- **Path**: `c10/util/numa.cpp`
- **Size**: 2,882 bytes (2.81 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/Exception.h>
#include <c10/util/numa.h>

C10_DEFINE_bool(caffe2_cpu_numa_enabled, false, "Use NUMA whenever possible.")

#if defined(__linux__) && defined(C10_USE_NUMA) && !defined(C10_MOBILE)
#include <numa.h>
#include <numaif.h>
#include <unistd.h>
#define C10_ENABLE_NUMA
#endif

// This code used to have a lot of VLOGs. However, because allocation might be
// triggered during static initialization, it's unsafe to invoke VLOG here

namespace c10 {

#ifdef C10_ENABLE_NUMA
bool IsNUMAEnabled() {
  return FLAGS_caffe2_cpu_numa_enabled && numa_available() >= 0;
}

void NUMABind(int numa_node_id) {
  if (numa_node_id < 0) {
    return;
  }
  if (!IsNUMAEnabled()) {
    return;
  }

  TORCH_CHECK(
      numa_node_id <= numa_max_node(),
      "NUMA node id ",
      numa_node_id,
      " is unavailable");

  auto bm = numa_allocate_nodemask();
  numa_bitmask_setbit(bm, numa_node_id);
  numa_bind(bm);
  numa_bitmask_free(bm);
}

int GetNUMANode(const void* ptr) {
  if (!IsNUMAEnabled()) {
    return -1;
  }
  AT_ASSERT(ptr);

  int numa_node = -1;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  TORCH_CHECK(
      get_mempolicy(
          &numa_node,
          nullptr,
          0,
          const_cast<void*>(ptr),
          MPOL_F_NODE | MPOL_F_ADDR) == 0,
      "Unable to get memory policy, errno:",
      errno);
  return numa_node;
}

int GetNumNUMANodes() {
  if (!IsNUMAEnabled()) {
    return -1;
  }

  return numa_num_configured_nodes();
}

void NUMAMove(void* ptr, size_t size, int numa_node_id) {
  if (numa_node_id < 0) {
    return;
  }
  if (!IsNUMAEnabled()) {
    return;
  }
  AT_ASSERT(ptr);

  uintptr_t page_start_ptr =
      ((reinterpret_cast<uintptr_t>(ptr)) & ~(getpagesize() - 1));
  // NOLINTNEXTLINE(*-conversions)
  ptrdiff_t offset = reinterpret_cast<uintptr_t>(ptr) - page_start_ptr;
  // Avoid extra dynamic allocation and NUMA api calls
  AT_ASSERT(
      numa_node_id >= 0 &&
      static_cast<unsigned>(numa_node_id) < sizeof(unsigned long) * 8);
  unsigned long mask = 1UL << numa_node_id;
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  TORCH_CHECK(
      mbind(
          reinterpret_cast<void*>(page_start_ptr),
          size + offset,
          MPOL_BIND,
          &mask,
          sizeof(mask) * 8,
          MPOL_MF_MOVE | MPOL_MF_STRICT) == 0,
      "Could not move memory to a NUMA node");
}

int GetCurrentNUMANode() {
  if (!IsNUMAEnabled()) {
    return -1;
  }

  auto n = numa_node_of_cpu(sched_getcpu());
  return n;
}

#else // C10_ENABLE_NUMA

bool IsNUMAEnabled() {
  return false;
}

void NUMABind(int /*numa_node_id*/) {}

int GetNUMANode(const void* /*ptr*/) {
  return -1;
}

int GetNumNUMANodes() {
  return -1;
}

void NUMAMove(void* /*ptr*/, size_t /*size*/, int /*numa_node_id*/) {}

int GetCurrentNUMANode() {
  return -1;
}

#endif // C10_NUMA_ENABLED

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/Exception.h`
- `c10/util/numa.h`
- `numa.h`
- `numaif.h`
- `unistd.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`c10/util`):

- [`CallOnce.h_docs.md`](./CallOnce.h_docs.md)
- [`Unicode.cpp_docs.md`](./Unicode.cpp_docs.md)
- [`logging_is_not_google_glog.h_docs.md`](./logging_is_not_google_glog.h_docs.md)
- [`Array.h_docs.md`](./Array.h_docs.md)
- [`complex_math.h_docs.md`](./complex_math.h_docs.md)
- [`order_preserving_flat_hash_map.h_docs.md`](./order_preserving_flat_hash_map.h_docs.md)
- [`flags_use_gflags.cpp_docs.md`](./flags_use_gflags.cpp_docs.md)
- [`flags_use_no_gflags.cpp_docs.md`](./flags_use_no_gflags.cpp_docs.md)
- [`Float8_e4m3fnuz.h_docs.md`](./Float8_e4m3fnuz.h_docs.md)
- [`typeid.cpp_docs.md`](./typeid.cpp_docs.md)


## Cross-References

- **File Documentation**: `numa.cpp_docs.md`
- **Keyword Index**: `numa.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
