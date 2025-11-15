# Documentation: `test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegStream.h`

## File Metadata

- **Path**: `test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegStream.h`
- **Size**: 3,903 bytes (3.81 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```c
#pragma once

#include <include/openreg.h>

#include "OpenRegException.h"
#include "OpenRegFunctions.h"

#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/util/Exception.h>

namespace c10::openreg {

static constexpr int max_compile_time_stream_priorities = 1;

class OpenRegStream {
 public:
  enum Unchecked { UNCHECKED };

  explicit OpenRegStream(Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == DeviceType::PrivateUse1);
  }

  explicit OpenRegStream(Unchecked, Stream stream) : stream_(stream) {}

  bool operator==(const OpenRegStream& other) const noexcept {
    return unwrap() == other.unwrap();
  }

  bool operator!=(const OpenRegStream& other) const noexcept {
    return unwrap() != other.unwrap();
  }

  operator orStream_t() const {
    return stream();
  }

  operator Stream() const {
    return unwrap();
  }

  DeviceType device_type() const {
    return DeviceType::PrivateUse1;
  }

  DeviceIndex device_index() const {
    return stream_.device_index();
  }

  Device device() const {
    return Device(DeviceType::PrivateUse1, device_index());
  }

  StreamId id() const {
    return stream_.id();
  }

  bool query() const {
    DeviceGuard guard{stream_.device()};

    if (orStreamQuery(stream()) == orSuccess) {
      return true;
    }

    return false;
  }

  void synchronize() const {
    DeviceGuard guard{stream_.device()};
    OPENREG_CHECK(orStreamSynchronize(stream()));
  }

  int priority() const {
    DeviceGuard guard{stream_.device()};
    int priority = 0;
    OPENREG_CHECK(orStreamGetPriority(stream(), &priority));
    return priority;
  }

  orStream_t stream() const;

  Stream unwrap() const {
    return stream_;
  }

  struct c10::StreamData3 pack3() const {
    return stream_.pack3();
  }

  static OpenRegStream unpack3(
      StreamId stream_id,
      DeviceIndex device_index,
      DeviceType device_type) {
    return OpenRegStream(Stream::unpack3(stream_id, device_index, device_type));
  }

 private:
  Stream stream_;
};

/*
 * Get a stream from the pool in a round-robin fashion.
 *
 * You can request a stream from the highest priority pool by setting
 * isHighPriority to true for a specific device.
 */
OPENREG_EXPORT OpenRegStream
getStreamFromPool(const bool isHighPriority = false, DeviceIndex device = -1);

/*
 * Get a stream from the pool in a round-robin fashion.
 *
 * You can request a stream by setting a priority value for a specific device.
 * The priority number lower, the priority higher.
 */
OPENREG_EXPORT OpenRegStream
getStreamFromPool(const int priority, DeviceIndex device = -1);

/*
 * Get a OpenRegStream from a externally allocated one.
 *
 * This is mainly for interoperability with different libraries where we
 * want to operate on a non-torch allocated stream for data exchange or similar
 * purposes
 */
OPENREG_EXPORT OpenRegStream
getStreamFromExternal(orStream_t ext_stream, DeviceIndex device_index);

/*
 * Get the default OpenReg stream, for the passed OpenReg device, or for the
 * current device if no device index is passed.
 */
OPENREG_EXPORT OpenRegStream
getDefaultOpenRegStream(DeviceIndex device_index = -1);

/*
 * Get the current OpenReg stream, for the passed OpenReg device, or for the
 * current device if no device index is passed.
 */
OPENREG_EXPORT OpenRegStream
getCurrentOpenRegStream(DeviceIndex device_index = -1);

/*
 * Set the current stream on the device of the passed in stream to be the passed
 * in stream.
 */
OPENREG_EXPORT void setCurrentOpenRegStream(OpenRegStream stream);

OPENREG_EXPORT std::ostream& operator<<(
    std::ostream& stream,
    const OpenRegStream& s);

} // namespace c10::openreg

namespace std {
template <>
struct hash<c10::openreg::OpenRegStream> {
  size_t operator()(c10::openreg::OpenRegStream s) const noexcept {
    return std::hash<c10::Stream>{}(s.unwrap());
  }
};
} // namespace std

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 28 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `std`, `c10`

**Classes/Structs**: `OpenRegStream`, `c10`, `hash`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `include/openreg.h`
- `OpenRegException.h`
- `OpenRegFunctions.h`
- `c10/core/DeviceGuard.h`
- `c10/core/Stream.h`
- `c10/util/Exception.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegStream.h
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime`):

- [`OpenRegHostAllocator.cpp_docs.md`](./OpenRegHostAllocator.cpp_docs.md)
- [`OpenRegStream.cpp_docs.md`](./OpenRegStream.cpp_docs.md)
- [`OpenRegGenerator.cpp_docs.md`](./OpenRegGenerator.cpp_docs.md)
- [`OpenRegHostAllocator.h_docs.md`](./OpenRegHostAllocator.h_docs.md)
- [`OpenRegHooks.cpp_docs.md`](./OpenRegHooks.cpp_docs.md)
- [`OpenRegSerialization.cpp_docs.md`](./OpenRegSerialization.cpp_docs.md)
- [`OpenRegFunctions.cpp_docs.md`](./OpenRegFunctions.cpp_docs.md)
- [`OpenRegGuard.h_docs.md`](./OpenRegGuard.h_docs.md)
- [`OpenRegDeviceAllocator.h_docs.md`](./OpenRegDeviceAllocator.h_docs.md)
- [`OpenRegException.cpp_docs.md`](./OpenRegException.cpp_docs.md)


## Cross-References

- **File Documentation**: `OpenRegStream.h_docs.md`
- **Keyword Index**: `OpenRegStream.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
