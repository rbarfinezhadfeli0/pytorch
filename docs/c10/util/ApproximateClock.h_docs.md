# Documentation: `c10/util/ApproximateClock.h`

## File Metadata

- **Path**: `c10/util/ApproximateClock.h`
- **Size**: 3,482 bytes (3.40 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
// Copyright 2023-present Facebook. All Rights Reserved.

#pragma once

#include <c10/macros/Export.h>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <functional>
#include <type_traits>

#if defined(C10_IOS) && defined(C10_MOBILE)
#include <sys/time.h> // for gettimeofday()
#endif

#if defined(__i386__) || defined(__x86_64__) || defined(__amd64__)
#define C10_RDTSC
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__CUDACC__) || defined(__HIPCC__)
#undef C10_RDTSC
#elif defined(__clang__)
// `__rdtsc` is available by default.
// NB: This has to be first, because Clang will also define `__GNUC__`
#elif defined(__GNUC__)
#include <x86intrin.h>
#else
#undef C10_RDTSC
#endif
#endif

namespace c10 {

using time_t = int64_t;
using steady_clock_t = std::conditional_t<
    std::chrono::high_resolution_clock::is_steady,
    std::chrono::high_resolution_clock,
    std::chrono::steady_clock>;

inline time_t getTimeSinceEpoch() {
  auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}

inline time_t getTime(bool allow_monotonic = false) {
#if defined(C10_IOS) && defined(C10_MOBILE)
  // clock_gettime is only available on iOS 10.0 or newer. Unlike OS X, iOS
  // can't rely on CLOCK_REALTIME, as it is defined no matter if clock_gettime
  // is implemented or not
  struct timeval now;
  gettimeofday(&now, NULL);
  return static_cast<time_t>(now.tv_sec) * 1000000000 +
      static_cast<time_t>(now.tv_usec) * 1000;
#elif defined(_WIN32) || defined(__MACH__)
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             steady_clock_t::now().time_since_epoch())
      .count();
#else
  // clock_gettime is *much* faster than std::chrono implementation on Linux
  struct timespec t{};
  auto mode = CLOCK_REALTIME;
  if (allow_monotonic) {
    mode = CLOCK_MONOTONIC;
  }
  clock_gettime(mode, &t);
  return static_cast<time_t>(t.tv_sec) * 1000000000 +
      static_cast<time_t>(t.tv_nsec);
#endif
}

// We often do not need to capture true wall times. If a fast mechanism such
// as TSC is available we can use that instead and convert back to epoch time
// during post processing. This greatly reduce the clock's contribution to
// profiling.
//   http://btorpey.github.io/blog/2014/02/18/clock-sources-in-linux/
//   https://quick-bench.com/q/r8opkkGZSJMu9wM_XTbDouq-0Io
// TODO: We should use
// `https://github.com/google/benchmark/blob/main/src/cycleclock.h`
inline auto getApproximateTime() {
#if defined(C10_RDTSC)
  return static_cast<uint64_t>(__rdtsc());
#else
  return getTime();
#endif
}

using approx_time_t = decltype(getApproximateTime());
static_assert(
    std::is_same_v<approx_time_t, int64_t> ||
        std::is_same_v<approx_time_t, uint64_t>,
    "Expected either int64_t (`getTime`) or uint64_t (some TSC reads).");

// Convert `getCount` results to Nanoseconds since unix epoch.
class C10_API ApproximateClockToUnixTimeConverter final {
 public:
  ApproximateClockToUnixTimeConverter();
  std::function<time_t(approx_time_t)> makeConverter();

  struct UnixAndApproximateTimePair {
    time_t t_;
    approx_time_t approx_t_;
  };
  static UnixAndApproximateTimePair measurePair();

 private:
  static constexpr size_t replicates = 1001;
  using time_pairs = std::array<UnixAndApproximateTimePair, replicates>;
  time_pairs measurePairs();

  time_pairs start_times_;
};

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 18 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `timeval`, `timespec`, `C10_API`, `UnixAndApproximateTimePair`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/macros/Export.h`
- `array`
- `chrono`
- `cstddef`
- `cstdint`
- `ctime`
- `functional`
- `type_traits`
- `sys/time.h`
- `intrin.h`
- `x86intrin.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

- **File Documentation**: `ApproximateClock.h_docs.md`
- **Keyword Index**: `ApproximateClock.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
