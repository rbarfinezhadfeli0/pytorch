# Documentation: `c10/test/core/impl/InlineStreamGuard_test.cpp`

## File Metadata

- **Path**: `c10/test/core/impl/InlineStreamGuard_test.cpp`
- **Size**: 8,010 bytes (7.82 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <gtest/gtest.h>

#include <c10/core/impl/FakeGuardImpl.h>
#include <c10/core/impl/InlineStreamGuard.h>
#include <vector>

using namespace c10;
using namespace c10::impl;

constexpr auto TestDeviceType = DeviceType::CUDA;
using TestGuardImpl = FakeGuardImpl<TestDeviceType>;

static Device dev(DeviceIndex index) {
  return Device{TestDeviceType, index};
}

static Stream stream(DeviceIndex index, StreamId sid) {
  return Stream(Stream::UNSAFE, dev(index), sid);
}

// -- InlineStreamGuard -------------------------------------------------------

using TestGuard = InlineStreamGuard<TestGuardImpl>;

TEST(InlineStreamGuard, Constructor) {
  TestGuardImpl::setDeviceIndex(0);
  TestGuardImpl::resetStreams();
  {
    TestGuard g(stream(1, 2));
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 1);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 2);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    ASSERT_EQ(g.original_stream(), stream(0, 0));
    ASSERT_EQ(g.current_stream(), stream(1, 2));
    ASSERT_EQ(g.original_device(), dev(0));
    ASSERT_EQ(g.current_device(), dev(1));
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
}

TEST(InlineStreamGuard, ResetStreamSameSameDevice) {
  TestGuardImpl::setDeviceIndex(0);
  TestGuardImpl::resetStreams();
  {
    TestGuard g(stream(0, 2));
    g.reset_stream(stream(0, 3));
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 3);
    ASSERT_EQ(g.original_stream(), stream(0, 0));
    ASSERT_EQ(g.current_stream(), stream(0, 3));
    ASSERT_EQ(g.original_device(), dev(0));
    ASSERT_EQ(g.current_device(), dev(0));
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
}

TEST(InlineStreamGuard, ResetStreamDifferentSameDevice) {
  TestGuardImpl::setDeviceIndex(0);
  TestGuardImpl::resetStreams();
  {
    TestGuard g(stream(1, 2));
    g.reset_stream(stream(1, 3));
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 1);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 3);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    ASSERT_EQ(g.original_stream(), stream(0, 0));
    ASSERT_EQ(g.current_stream(), stream(1, 3));
    ASSERT_EQ(g.original_device(), dev(0));
    ASSERT_EQ(g.current_device(), dev(1));
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
}

TEST(InlineStreamGuard, ResetStreamDifferentDevice) {
  TestGuardImpl::setDeviceIndex(0);
  TestGuardImpl::resetStreams();
  {
    TestGuard g(stream(1, 2));
    g.reset_stream(stream(2, 3));
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 2);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(2), 3);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    ASSERT_EQ(g.original_stream(), stream(0, 0));
    ASSERT_EQ(g.current_stream(), stream(2, 3));
    ASSERT_EQ(g.original_device(), dev(0));
    ASSERT_EQ(g.current_device(), dev(2));
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(2), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
}

// -- OptionalInlineStreamGuard
// -------------------------------------------------------

using OptionalTestGuard = InlineOptionalStreamGuard<TestGuardImpl>;

TEST(InlineOptionalStreamGuard, Constructor) {
  TestGuardImpl::setDeviceIndex(0);
  TestGuardImpl::resetStreams();
  {
    OptionalTestGuard g(stream(1, 2));
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 1);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 2);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    ASSERT_EQ(g.original_stream(), stream(0, 0));
    ASSERT_EQ(g.current_stream(), stream(1, 2));
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
  {
    OptionalTestGuard g(stream(1, 2));
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 1);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 2);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    ASSERT_EQ(g.original_stream(), stream(0, 0));
    ASSERT_EQ(g.current_stream(), stream(1, 2));
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
  {
    OptionalTestGuard g;
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
}

TEST(InlineOptionalStreamGuard, ResetStreamSameDevice) {
  TestGuardImpl::setDeviceIndex(0);
  TestGuardImpl::resetStreams();
  {
    OptionalTestGuard g;
    g.reset_stream(stream(1, 3));
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 1);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 3);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    ASSERT_EQ(g.original_stream(), stream(0, 0));
    ASSERT_EQ(g.current_stream(), stream(1, 3));
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
}

TEST(InlineOptionalStreamGuard, ResetStreamDifferentDevice) {
  TestGuardImpl::setDeviceIndex(0);
  TestGuardImpl::resetStreams();
  {
    OptionalTestGuard g;
    g.reset_stream(stream(2, 3));
    ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 2);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(2), 3);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    ASSERT_EQ(g.original_stream(), stream(0, 0));
    ASSERT_EQ(g.current_stream(), stream(2, 3));
  }
  ASSERT_EQ(TestGuardImpl::getDeviceIndex(), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(2), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
}

// -- InlineMultiStreamGuard
// -------------------------------------------------------

using MultiTestGuard = InlineMultiStreamGuard<TestGuardImpl>;

TEST(InlineMultiStreamGuard, Constructor) {
  TestGuardImpl::resetStreams();
  {
    std::vector<Stream> streams;
    MultiTestGuard g(streams);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  }
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  {
    std::vector<Stream> streams = {stream(0, 2)};
    MultiTestGuard g(streams);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 2);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  }
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  {
    std::vector<Stream> streams = {stream(1, 3)};
    MultiTestGuard g(streams);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 3);
  }
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
  {
    std::vector<Stream> streams = {stream(0, 2), stream(1, 3)};
    MultiTestGuard g(streams);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 2);
    ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 3);
  }
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(0), 0);
  ASSERT_EQ(TestGuardImpl::getCurrentStreamIdFor(1), 0);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 13 function(s).

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

- `c10/core/Device.h`
- `c10/core/DeviceType.h`
- `c10/core/Stream.h`
- `gtest/gtest.h`
- `c10/core/impl/FakeGuardImpl.h`
- `c10/core/impl/InlineStreamGuard.h`
- `vector`


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
python c10/test/core/impl/InlineStreamGuard_test.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`c10/test/core/impl`):

- [`InlineDeviceGuard_test.cpp_docs.md`](./InlineDeviceGuard_test.cpp_docs.md)
- [`SizesAndStrides_test.cpp_docs.md`](./SizesAndStrides_test.cpp_docs.md)
- [`cow_test.cpp_docs.md`](./cow_test.cpp_docs.md)


## Cross-References

- **File Documentation**: `InlineStreamGuard_test.cpp_docs.md`
- **Keyword Index**: `InlineStreamGuard_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
