# Documentation: `aten/src/ATen/test/xpu_caching_host_allocator_test.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/xpu_caching_host_allocator_test.cpp`
- **Size**: 5,992 bytes (5.85 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>
#include <ATen/xpu/CachingHostAllocator.h>
#include <ATen/xpu/XPUContext.h>
#include <ATen/xpu/XPUEvent.h>
#include <c10/core/ScalarType.h>
#include <c10/xpu/XPUStream.h>

constexpr int64_t N = 100;

TEST(CachingHostAllocatorTest, testPinnedAliasSlice) {
  if (!at::xpu::is_available()) {
    return;
  }

  // Check a standard pinned tensor can be correctly recorded.
  auto pinned_tensor =
      at::empty({N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
  // TODO: Uncomment this line when op `pin_memory` is supported on XPU.
  // ASSERT_TRUE(pinned_tensor.is_pinned());
  ASSERT_TRUE(at::getHostAllocator(at::kXPU)->record_event(
      pinned_tensor.data_ptr(),
      pinned_tensor.storage().data_ptr().get_context(),
      at::xpu::getCurrentXPUStream().unwrap()));

  // Check an tensor constructed with from_blob can be correctly recorded (via
  // the shared data_ptr)
  auto alias_tensor = at::from_blob(
      pinned_tensor.data_ptr(), pinned_tensor.sizes(), pinned_tensor.options());
  // ASSERT_TRUE(alias_tensor.is_pinned());

  ASSERT_FALSE(
      alias_tensor.storage().data_ptr().get_context() ==
      pinned_tensor.storage().data_ptr().get_context());
  ASSERT_EQ(alias_tensor.data_ptr(), pinned_tensor.data_ptr());
  ASSERT_TRUE(at::getHostAllocator(at::kXPU)->record_event(
      alias_tensor.data_ptr(),
      alias_tensor.storage().data_ptr().get_context(),
      at::xpu::getCurrentXPUStream().unwrap()));

  // Check an tensor constructed with slicing can be correctly recorded (via
  // the shared context)
  auto slice_tensor =
      pinned_tensor.index({at::indexing::Slice(1, at::indexing::None, 2)});
  ASSERT_EQ(
      slice_tensor.storage().data_ptr().get_context(),
      pinned_tensor.storage().data_ptr().get_context());
  ASSERT_NE(slice_tensor.data_ptr(), pinned_tensor.data_ptr());
  ASSERT_TRUE(at::getHostAllocator(at::kXPU)->record_event(
      slice_tensor.data_ptr(),
      slice_tensor.storage().data_ptr().get_context(),
      at::xpu::getCurrentXPUStream().unwrap()));

  // Check a tensor that has neither a matching context nor data_ptr cannot be
  // recorded.
  auto alias_slice_tensor = at::from_blob(
      slice_tensor.data_ptr(), slice_tensor.sizes(), slice_tensor.options());
  // ASSERT_TRUE(alias_slice_tensor.is_pinned());
  ASSERT_FALSE(at::getHostAllocator(at::kXPU)->record_event(
      alias_slice_tensor.data_ptr(),
      alias_slice_tensor.storage().data_ptr().get_context(),
      at::xpu::getCurrentXPUStream().unwrap()));
  ASSERT_NE(
      alias_slice_tensor.storage().data_ptr().get(),
      slice_tensor.storage().data_ptr().get());
}

TEST(CachingHostAllocatorTest, testRawAllocation) {
  if (!at::xpu::is_available()) {
    return;
  }

  auto data_ptr = at::getHostAllocator(at::kXPU)->allocate(N);
  class UserDataDeleter {
   public:
    explicit UserDataDeleter(std::unique_ptr<void, c10::DeleterFnPtr> ptr)
        : ptr_(std::move(ptr)) {}

   private:
    std::unique_ptr<void, c10::DeleterFnPtr> ptr_;
  };
  auto* user_data_deleter = new UserDataDeleter(data_ptr.move_context());

  struct IOBuf {
    explicit IOBuf(void* buf, void* ctx, std::function<void(void*)> deleter)
        : buf_(buf), ctx_(ctx), deleter_(std::move(deleter)) {}
    void* buf_;
    void* ctx_;
    std::function<void(void*)> deleter_;
    ~IOBuf() {
      deleter_(ctx_);
    }
  };
  auto iobuf =
      std::make_unique<IOBuf>(data_ptr.get(), user_data_deleter, [](void* ctx) {
        delete static_cast<UserDataDeleter*>(ctx);
      });
  auto pinned_tensor =
      at::for_blob(iobuf->buf_, {N})
          .context(
              iobuf.release(),
              [](void* ctx) { delete static_cast<IOBuf*>(ctx); })
          .make_tensor();

  // ASSERT_TRUE(pinned_tensor.is_pinned());
  ASSERT_TRUE(at::getHostAllocator(at::kXPU)->record_event(
      pinned_tensor.data_ptr(),
      pinned_tensor.storage().data_ptr().get_context(),
      at::xpu::getCurrentXPUStream().unwrap()));
}

TEST(CachingHostAllocatorTest, testUnknownTensor) {
  if (!at::xpu::is_available()) {
    return;
  }

  auto unpinned_tensor =
      at::empty({N}, at::TensorOptions().dtype(at::kByte).pinned_memory(false));

  ASSERT_FALSE(at::getHostAllocator(at::kXPU)->record_event(
      unpinned_tensor.data_ptr(),
      unpinned_tensor.storage().data_ptr().get_context(),
      at::xpu::getCurrentXPUStream().unwrap()));
}

TEST(CachingHostAllocatorTest, testEmptyCache) {
  if (!at::xpu::is_available()) {
    return;
  }

  void* ptr{nullptr};
  void* ctx{nullptr};
  {
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    ptr = pinned_tensor.data_ptr();
    ctx = pinned_tensor.storage().data_ptr().get_context();
    ASSERT_TRUE(at::getHostAllocator(at::kXPU)->record_event(
        ptr, ctx, at::xpu::getCurrentXPUStream().unwrap()));
  }

  {
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    at::xpu::syncStreamsOnDevice();
  }

  at::getHostAllocator(at::kXPU)->empty_cache();
  ASSERT_FALSE(at::getHostAllocator(at::kXPU)->record_event(
      ptr, ctx, at::xpu::getCurrentXPUStream().unwrap()));
}

TEST(CachingHostAllocatorTest, testReuse) {
  if (!at::xpu::is_available()) {
    return;
  }

  void* ptr{nullptr};
  void* ctx{nullptr};
  {
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    ptr = pinned_tensor.data_ptr();
    ctx = pinned_tensor.storage().data_ptr().get_context();
  }
  // Ensure we reuse the allocation.
  {
    auto pinned_tensor = at::empty(
        {N}, at::TensorOptions().dtype(at::kByte).pinned_memory(true));
    ASSERT_EQ(ptr, pinned_tensor.data_ptr());
    ASSERT_EQ(ctx, pinned_tensor.storage().data_ptr().get_context());
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `UserDataDeleter`, `IOBuf`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `ATen/ATen.h`
- `ATen/TensorIndexing.h`
- `ATen/xpu/CachingHostAllocator.h`
- `ATen/xpu/XPUContext.h`
- `ATen/xpu/XPUEvent.h`
- `c10/core/ScalarType.h`
- `c10/xpu/XPUStream.h`


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

This is a test file. Run it with:

```bash
python aten/src/ATen/test/xpu_caching_host_allocator_test.cpp
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

- **File Documentation**: `xpu_caching_host_allocator_test.cpp_docs.md`
- **Keyword Index**: `xpu_caching_host_allocator_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
