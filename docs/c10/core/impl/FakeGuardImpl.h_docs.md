# Documentation: `c10/core/impl/FakeGuardImpl.h`

## File Metadata

- **Path**: `c10/core/impl/FakeGuardImpl.h`
- **Size**: 3,182 bytes (3.11 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <array>

namespace c10::impl {

// FakeGuardImpl is hardcoded to have eight devices.  Not for
// any good reason, just to simplify code.
constexpr DeviceIndex kFakeGuardImplMaxDevices = 8;

/**
 * A fake implementation of DeviceGuardImplInterface suitable for testing.
 * The current device is modeled as a mutable field in the guard implementation
 * class.  See DeviceGuard_test.cpp for an example use.
 */
template <DeviceType T>
struct FakeGuardImpl final : public DeviceGuardImplInterface {
  static constexpr DeviceType static_type = T;
  // Runtime device type is not used
  FakeGuardImpl(DeviceType /*unused*/) {}
  FakeGuardImpl() = default;
  DeviceType type() const override {
    return T;
  }
  Device exchangeDevice(Device d) const override {
    AT_ASSERT(d.type() == type());
    AT_ASSERT(d.index() < kFakeGuardImplMaxDevices);
    Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      current_device_ = d.index();
    }
    return old_device;
  }
  Device getDevice() const override {
    return Device(type(), current_device_);
  }
  void setDevice(Device d) const override {
    AT_ASSERT(d.type() == type());
    AT_ASSERT(d.index() >= 0);
    AT_ASSERT(d.index() < kFakeGuardImplMaxDevices);
    current_device_ = d.index();
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    current_device_ = d.index();
  }
  Stream getStream(Device d) const noexcept override {
    return Stream(Stream::UNSAFE, d, current_streams_[d.index()]);
  }
  Stream exchangeStream(Stream s) const noexcept override {
    auto old_id = current_streams_[s.device_index()];
    current_streams_[s.device_index()] = s.id();
    return Stream(Stream::UNSAFE, s.device(), old_id);
  }
  DeviceIndex deviceCount() const noexcept override {
    return kFakeGuardImplMaxDevices;
  }

  // Event-related functions
  void record(
      void** /*event*/,
      const Stream& /*stream*/,
      const DeviceIndex /*device_index*/,
      const EventFlag /*flag*/) const override {}
  void block(void* /*event*/, const Stream& /*stream*/) const override {}
  bool queryEvent(void* /*event*/) const override {
    return true;
  }
  void destroyEvent(void* /*event*/, const DeviceIndex /*device_index*/)
      const noexcept override {}

  // Convenience methods for testing
  static DeviceIndex getDeviceIndex() {
    return current_device_;
  }
  static void setDeviceIndex(DeviceIndex i) {
    AT_ASSERT(i >= 0);
    AT_ASSERT(i < kFakeGuardImplMaxDevices);
    current_device_ = i;
  }
  static StreamId getCurrentStreamIdFor(DeviceIndex i) {
    return current_streams_.at(i);
  }
  static void resetStreams() {
    current_streams_.fill(0);
  }

 private:
  thread_local static DeviceIndex current_device_;
  thread_local static std::array<StreamId, kFakeGuardImplMaxDevices>
      current_streams_;
};

template <DeviceType T>
thread_local DeviceIndex FakeGuardImpl<T>::current_device_ = 0;

template <DeviceType T>
thread_local std::array<StreamId, kFakeGuardImplMaxDevices>
    FakeGuardImpl<T>::current_streams_ = {0, 0, 0, 0, 0, 0, 0, 0};

} // namespace c10::impl

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 20 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `FakeGuardImpl`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/core/impl`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/impl/DeviceGuardImplInterface.h`
- `array`


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

- **File Documentation**: `FakeGuardImpl.h_docs.md`
- **Keyword Index**: `FakeGuardImpl.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
