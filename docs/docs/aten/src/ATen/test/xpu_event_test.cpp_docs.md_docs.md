# Documentation: `docs/aten/src/ATen/test/xpu_event_test.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/test/xpu_event_test.cpp_docs.md`
- **Size**: 5,055 bytes (4.94 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/test/xpu_event_test.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/xpu_event_test.cpp`
- **Size**: 2,493 bytes (2.43 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <ATen/xpu/XPUEvent.h>
#include <c10/util/irange.h>
#include <c10/xpu/test/impl/XPUTest.h>

TEST(XpuEventTest, testXPUEventBehavior) {
  if (!at::xpu::is_available()) {
    return;
  }
  auto stream = c10::xpu::getStreamFromPool();
  at::xpu::XPUEvent event;

  EXPECT_TRUE(event.query());
  EXPECT_TRUE(!event.isCreated());

  event.recordOnce(stream);
  EXPECT_TRUE(event.isCreated());

  auto wait_stream0 = c10::xpu::getStreamFromPool();
  auto wait_stream1 = c10::xpu::getStreamFromPool();

  event.block(wait_stream0);
  event.block(wait_stream1);

  wait_stream0.synchronize();
  EXPECT_TRUE(event.query());
}

TEST(XpuEventTest, testXPUEventCrossDevice) {
  if (at::xpu::device_count() <= 1) {
    return;
  }

  const auto stream0 = at::xpu::getStreamFromPool();
  at::xpu::XPUEvent event0;

  const auto stream1 = at::xpu::getStreamFromPool(false, 1);
  at::xpu::XPUEvent event1;

  event0.record(stream0);
  event1.record(stream1);

  event0 = std::move(event1);

  EXPECT_EQ(event0.device(), at::Device(at::kXPU, 1));

  event0.block(stream0);

  stream0.synchronize();
  ASSERT_TRUE(event0.query());
}

void eventSync(sycl::event& event) {
  event.wait();
}

TEST(XpuEventTest, testXPUEventFunction) {
  if (!at::xpu::is_available()) {
    return;
  }

  constexpr int numel = 1024;
  int hostData[numel];
  initHostData(hostData, numel);

  auto stream = c10::xpu::getStreamFromPool();
  int* deviceData = sycl::malloc_device<int>(numel, stream);

  // H2D
  stream.queue().memcpy(deviceData, hostData, sizeof(int) * numel);
  at::xpu::XPUEvent event;
  event.record(stream);
  // To validate the implicit conversion of an XPUEvent to sycl::event.
  eventSync(event);
  EXPECT_TRUE(event.query());

  clearHostData(hostData, numel);

  // D2H
  stream.queue().memcpy(hostData, deviceData, sizeof(int) * numel);
  event.record(stream);
  event.synchronize();

  validateHostData(hostData, numel);

  clearHostData(hostData, numel);
  // D2H
  stream.queue().memcpy(hostData, deviceData, sizeof(int) * numel);
  // The event has already been created, so there will be no recording of the
  // stream via recordOnce() here.
  event.recordOnce(stream);
  EXPECT_TRUE(event.query());

  stream.synchronize();
  sycl::free(deviceData, c10::xpu::get_device_context());

  if (at::xpu::device_count() <= 1) {
    return;
  }
  c10::xpu::set_device(1);
  auto stream1 = c10::xpu::getStreamFromPool();
  ASSERT_THROW(event.record(stream1), c10::Error);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `ATen/xpu/XPUEvent.h`
- `c10/util/irange.h`
- `c10/xpu/test/impl/XPUTest.h`


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

This is a test file. Run it with:

```bash
python aten/src/ATen/test/xpu_event_test.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/test`):

- [`operators_test.cpp_docs.md`](./operators_test.cpp_docs.md)
- [`xpu_generator_test.cpp_docs.md`](./xpu_generator_test.cpp_docs.md)
- [`native_test.cpp_docs.md`](./native_test.cpp_docs.md)
- [`reportMemoryUsage.h_docs.md`](./reportMemoryUsage.h_docs.md)
- [`tensor_iterator_test.cpp_docs.md`](./tensor_iterator_test.cpp_docs.md)
- [`memory_overlapping_test.cpp_docs.md`](./memory_overlapping_test.cpp_docs.md)
- [`operator_name_test.cpp_docs.md`](./operator_name_test.cpp_docs.md)
- [`cuda_distributions_test.cu_docs.md`](./cuda_distributions_test.cu_docs.md)
- [`type_test.cpp_docs.md`](./type_test.cpp_docs.md)
- [`allocator_clone_test.h_docs.md`](./allocator_clone_test.h_docs.md)


## Cross-References

- **File Documentation**: `xpu_event_test.cpp_docs.md`
- **Keyword Index**: `xpu_event_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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

This is a test file. Run it with:

```bash
python docs/aten/src/ATen/test/xpu_event_test.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/aten/src/ATen/test`):

- [`cuda_dlconvertor_test.cpp_kw.md_docs.md`](./cuda_dlconvertor_test.cpp_kw.md_docs.md)
- [`cuda_atomic_ops_test.cu_kw.md_docs.md`](./cuda_atomic_ops_test.cu_kw.md_docs.md)
- [`ivalue_test.cpp_kw.md_docs.md`](./ivalue_test.cpp_kw.md_docs.md)
- [`mobile_memory_cleanup.cpp_kw.md_docs.md`](./mobile_memory_cleanup.cpp_kw.md_docs.md)
- [`reportMemoryUsage_test.cpp_docs.md_docs.md`](./reportMemoryUsage_test.cpp_docs.md_docs.md)
- [`cpu_rng_test.cpp_kw.md_docs.md`](./cpu_rng_test.cpp_kw.md_docs.md)
- [`lazy_tensor_test.cpp_kw.md_docs.md`](./lazy_tensor_test.cpp_kw.md_docs.md)
- [`cuda_allocator_test.cpp_docs.md_docs.md`](./cuda_allocator_test.cpp_docs.md_docs.md)
- [`MaybeOwned_test.cpp_docs.md_docs.md`](./MaybeOwned_test.cpp_docs.md_docs.md)
- [`dlconvertor_test.cpp_kw.md_docs.md`](./dlconvertor_test.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `xpu_event_test.cpp_docs.md_docs.md`
- **Keyword Index**: `xpu_event_test.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
