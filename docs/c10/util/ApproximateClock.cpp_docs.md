# Documentation: `c10/util/ApproximateClock.cpp`

## File Metadata

- **Path**: `c10/util/ApproximateClock.cpp`
- **Size**: 3,203 bytes (3.13 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/ApproximateClock.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

namespace c10 {

ApproximateClockToUnixTimeConverter::ApproximateClockToUnixTimeConverter()
    : start_times_(measurePairs()) {}

ApproximateClockToUnixTimeConverter::UnixAndApproximateTimePair
ApproximateClockToUnixTimeConverter::measurePair() {
  // Take a measurement on either side to avoid an ordering bias.
  auto fast_0 = getApproximateTime();
  auto wall = std::chrono::system_clock::now();
  auto fast_1 = getApproximateTime();

  TORCH_INTERNAL_ASSERT(fast_1 >= fast_0, "getCount is non-monotonic.");
  auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(
      wall.time_since_epoch());

  // `x + (y - x) / 2` is a more numerically stable average than `(x + y) / 2`.
  return {t.count(), fast_0 + (fast_1 - fast_0) / 2};
}

ApproximateClockToUnixTimeConverter::time_pairs
ApproximateClockToUnixTimeConverter::measurePairs() {
  static constexpr auto n_warmup = 5;
  for ([[maybe_unused]] const auto _ : c10::irange(n_warmup)) {
    getApproximateTime();
    static_cast<void>(steady_clock_t::now());
  }

  time_pairs out;
  for (const auto i : c10::irange(out.size())) {
    out[i] = measurePair();
  }
  return out;
}

std::function<time_t(approx_time_t)> ApproximateClockToUnixTimeConverter::
    makeConverter() {
  auto end_times = measurePairs();

  // Compute the real time that passes for each tick of the approximate clock.
  std::array<long double, replicates> scale_factors{};
  for (const auto i : c10::irange(replicates)) {
    auto delta_ns = end_times[i].t_ - start_times_[i].t_;
    auto delta_approx = end_times[i].approx_t_ - start_times_[i].approx_t_;
    scale_factors[i] =
        static_cast<double>(delta_ns) / static_cast<double>(delta_approx);
  }
  std::sort(scale_factors.begin(), scale_factors.end());
  long double scale_factor = scale_factors[replicates / 2 + 1];

  // We shift all times by `t0` for better numerics. Double precision only has
  // 16 decimal digits of accuracy, so if we blindly multiply times by
  // `scale_factor` we may suffer from precision loss. The choice of `t0` is
  // mostly arbitrary; we just need a factor that is the correct order of
  // magnitude to bring the intermediate values closer to zero. We are not,
  // however, guaranteed that `t0_approx` is *exactly* the getApproximateTime
  // equivalent of `t0`; it is only an estimate that we have to fine tune.
  auto t0 = start_times_[0].t_;
  auto t0_approx = start_times_[0].approx_t_;
  std::array<double, replicates> t0_correction{};
  for (const auto i : c10::irange(replicates)) {
    auto dt = start_times_[i].t_ - t0;
    auto dt_approx =
        static_cast<double>(start_times_[i].approx_t_ - t0_approx) *
        scale_factor;
    t0_correction[i] = dt - (time_t)dt_approx; // NOLINT
  }
  t0 += t0_correction[t0_correction.size() / 2 + 1]; // NOLINT

  return [=](approx_time_t t_approx) {
    // See above for why this is more stable than `A * t_approx + B`.
    return t_approx > t0_approx
        ? static_cast<time_t>(
              static_cast<double>(t_approx - t0_approx) * scale_factor) +
            t0
        : 0;
  };
}

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

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

- `c10/util/ApproximateClock.h`
- `c10/util/Exception.h`
- `c10/util/irange.h`


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

- **File Documentation**: `ApproximateClock.cpp_docs.md`
- **Keyword Index**: `ApproximateClock.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
