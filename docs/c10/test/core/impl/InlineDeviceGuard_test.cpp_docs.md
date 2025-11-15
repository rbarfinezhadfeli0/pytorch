# Documentation: `c10/test/core/impl/InlineDeviceGuard_test.cpp`

## File Metadata

- **Path**: `c10/test/core/impl/InlineDeviceGuard_test.cpp`
- **Size**: 5,862 bytes (5.72 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>
#include <initializer_list>

#include <c10/core/impl/FakeGuardImpl.h>
#include <c10/core/impl/InlineDeviceGuard.h>

using namespace c10;
using namespace c10::impl;

constexpr auto TestDeviceType = DeviceType::CUDA;
using TestGuardImpl = FakeGuardImpl<TestDeviceType>;

static Device dev(DeviceIndex index) {
  return Device(TestDeviceType, index);
}

// -- InlineDeviceGuard -------------------------------------------------------

using TestGuard = InlineDeviceGuard<TestGuardImpl>;

TEST(InlineDeviceGuard, Constructor) {
  for (DeviceIndex i : std::initializer_list<DeviceIndex>{-1, 0, 1}) {
    DeviceIndex init_i = 0;
    TestGuardImpl::setDeviceIndex(init_i);
    auto test_body = [&](TestGuard& g) -> void {
      ASSERT_EQ(g.original_device(), dev(init_i));
      ASSERT_EQ(g.current_device(), dev(i == -1 ? init_i : i));
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i == -1 ? init_i : i);
      // Test un-bracketed write to device index
      TestGuardImpl::setDeviceIndex(4);
    };
    {
      // Index constructor
      TestGuard g(i);
      test_body(g);
    }
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
    {
      // Device constructor
      TestGuard g(dev(i));
      test_body(g);
    }
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
    /*
    {
      // Optional constructor
      TestGuard g(dev(i));
      test_body(g);
    }
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
    */
  }
}

TEST(InlineDeviceGuard, ConstructorError) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_ANY_THROW(InlineDeviceGuard<FakeGuardImpl<DeviceType::CUDA>> g(
      Device(DeviceType::HIP, 1)));
}

TEST(InlineDeviceGuard, SetDevice) {
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  DeviceIndex i = 1;
  TestGuard g(i);
  DeviceIndex i2 = 2;
  g.set_device(dev(i2));
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i2));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
  g.set_device(dev(i2));
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i2));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
}

TEST(InlineDeviceGuard, ResetDevice) {
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  DeviceIndex i = 1;
  TestGuard g(i);
  DeviceIndex i2 = 2;
  g.reset_device(dev(i2));
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i2));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
  g.reset_device(dev(i2));
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i2));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
}

TEST(InlineDeviceGuard, SetIndex) {
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  DeviceIndex i = 1;
  TestGuard g(i);
  DeviceIndex i2 = 2;
  g.set_index(i2);
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i2));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
  g.set_index(i2);
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i2));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i2);
}

// -- InlineOptionalDeviceGuard
// --------------------------------------------------

using MaybeTestGuard = InlineOptionalDeviceGuard<TestGuardImpl>;

TEST(InlineOptionalDeviceGuard, Constructor) {
  for (DeviceIndex i : std::initializer_list<DeviceIndex>{-1, 0, 1}) {
    DeviceIndex init_i = 0;
    TestGuardImpl::setDeviceIndex(init_i);
    auto test_body = [&](MaybeTestGuard& g) -> void {
      ASSERT_EQ(g.original_device(), dev(init_i));
      ASSERT_EQ(g.current_device(), dev(i == -1 ? init_i : i));
      ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i == -1 ? init_i : i);
      // Test un-bracketed write to device index
      TestGuardImpl::setDeviceIndex(4);
    };
    {
      // Index constructor
      MaybeTestGuard g(i);
      test_body(g);
    }
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
    {
      // Device constructor
      MaybeTestGuard g(dev(i));
      test_body(g);
    }
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
    {
      // Optional constructor
      MaybeTestGuard g(dev(i));
      test_body(g);
    }
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
  }
}

TEST(InlineOptionalDeviceGuard, NullaryConstructor) {
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  auto test_body = [&](MaybeTestGuard& g) -> void {
    ASSERT_EQ(g.original_device(), std::nullopt);
    ASSERT_EQ(g.current_device(), std::nullopt);
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), init_i);
  };
  {
    MaybeTestGuard g;
    test_body(g);
  }
  {
    // If you want nullopt directly to work, define a nullopt_t
    // overload.  But I don't really see why you'd want this lol.
    std::optional<Device> dev_opt = std::nullopt;
    MaybeTestGuard g(dev_opt);
    test_body(g);
  }
}

TEST(InlineOptionalDeviceGuard, SetDevice) {
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  MaybeTestGuard g;
  DeviceIndex i = 1;
  g.set_device(dev(i));
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);
  g.set_device(dev(i));
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);
}

TEST(InlineOptionalDeviceGuard, SetIndex) {
  DeviceIndex init_i = 0;
  TestGuardImpl::setDeviceIndex(init_i);
  DeviceIndex i = 1;
  MaybeTestGuard g;
  g.set_index(i);
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);
  g.set_index(i);
  ASSERT_EQ(g.original_device(), dev(init_i));
  ASSERT_EQ(g.current_device(), dev(i));
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), i);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/test/core/impl`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `initializer_list`
- `c10/core/impl/FakeGuardImpl.h`
- `c10/core/impl/InlineDeviceGuard.h`


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

This is a test file. Run it with:

```bash
python c10/test/core/impl/InlineDeviceGuard_test.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`c10/test/core/impl`):

- [`InlineStreamGuard_test.cpp_docs.md`](./InlineStreamGuard_test.cpp_docs.md)
- [`SizesAndStrides_test.cpp_docs.md`](./SizesAndStrides_test.cpp_docs.md)
- [`cow_test.cpp_docs.md`](./cow_test.cpp_docs.md)


## Cross-References

- **File Documentation**: `InlineDeviceGuard_test.cpp_docs.md`
- **Keyword Index**: `InlineDeviceGuard_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
