# Documentation: `test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.h`

## File Metadata

- **Path**: `test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.h`
- **Size**: 8,511 bytes (8.31 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```c
#pragma once

#include <c10/core/Device.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <include/openreg.h>

#include "OpenRegEvent.h"
#include "OpenRegFunctions.h"
#include "OpenRegStream.h"

namespace c10::openreg {

// LITERALINCLUDE START: OPENREG DEVICE MGMT GUARD IMPL EXAMPLE
struct OpenRegGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = c10::DeviceType::PrivateUse1;

  OpenRegGuardImpl() = default;

  explicit OpenRegGuardImpl(c10::DeviceType t) {
    TORCH_CHECK(
        t == static_type,
        "OpenRegGuardImpl initialized with non-PrivateUse1 DeviceType: ",
        t);
  }

  /**
   * Return the type of device managed by this guard implementation.
   */
  c10::DeviceType type() const override {
    return static_type;
  }

  /**
   * Set the current device to Device, and return the previous c10::Device.
   */
  c10::Device exchangeDevice(c10::Device d) const override {
    TORCH_CHECK(
        d.is_privateuseone(), "Excepted a PrivateUse1 device, but got ", d);

    auto old_device_index = ExchangeDevice(d.index());
    return c10::Device(static_type, old_device_index);
  }

  /**
   * Get the current device.
   */
  c10::Device getDevice() const override {
    int device_index = current_device();
    return c10::Device(static_type, device_index);
  }

  /**
   * Set the current device to c10::Device.
   */
  void setDevice(c10::Device d) const override {
    TORCH_CHECK(
        d.is_privateuseone(), "Excepted a PrivateUse1 device, but got ", d);

    set_device(d.index());
  }
// LITERALINCLUDE END: OPENREG DEVICE MGMT GUARD IMPL EXAMPLE

  /**
   * Set the current device to c10::Device, without checking for errors
   * (so, e.g., this can be called from a destructor).
   */
  void uncheckedSetDevice(c10::Device d) const noexcept override {
    set_device(d.index());
  }

  /**
   * Get the current stream for a given device.
   */
  c10::Stream getStream(c10::Device d) const noexcept override {
    return getCurrentOpenRegStream(d.index()).unwrap();
  }

  /**
   * Get the default stream for a given device.
   */
  c10::Stream getDefaultStream(c10::Device d) const override {
    return getDefaultOpenRegStream(d.index());
  }

  /**
   * Return a new stream for a given device and priority. The stream will be
   * copied and shared around, device backend should be able to correctly handle
   * the lifetime of the stream.
   */
  Stream getNewStream(Device d, int priority = 0) const override {
    return getStreamFromPool(priority, d.index());
  }

  /**
   * Get a stream from the global pool for a given device.
   */
  Stream getStreamFromGlobalPool(Device d, bool isHighPriority = false)
      const override {
    return getStreamFromPool(isHighPriority, d.index());
  }

  /**
   * Set a stream to be the thread local current stream for its device.
   * Return the previous stream for that device. You are NOT required
   * to set the current device to match the device of this stream.
   */
  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    const OpenRegStream stream(s);
    const auto old_stream = getCurrentOpenRegStream(s.device().index());
    setCurrentOpenRegStream(stream);
    return old_stream.unwrap();
  }

  /**
   * Get the number of devices.
   *
   * WARNING: This is REQUIRED to not raise an exception.
   * If there is some sort of problem, e.g., driver error,
   * you should report that there are zero available devices.
   */
  DeviceIndex deviceCount() const noexcept override {
    return device_count();
  }

  /**
   * Destroys the given event.
   */
  void destroyEvent(void* event, const c10::DeviceIndex device_index)
      const noexcept override {
    if (!event)
      return;

    auto or_event = static_cast<orEvent_t>(event);
    auto orig_device = current_device();
    set_device(device_index);
    orEventDestroy(or_event);
    set_device(orig_device);
  }

  /**
   * Increments the event's version and enqueues a job with this version
   * in the stream's work queue. When the stream process that job
   * it notifies all streams waiting on / blocked by that version of the
   * event to continue and marks that version as recorded.
   * */
  void record(
      void** event,
      const c10::Stream& stream,
      const c10::DeviceIndex device_index,
      const c10::EventFlag flag) const override {
    TORCH_CHECK(
        device_index == -1 || device_index == stream.device_index(),
        "Event device index ",
        device_index,
        " does not match recording stream's device index ",
        stream.device_index(),
        ".");

    orEvent_t or_event = static_cast<orEvent_t>(*event);
    OpenRegStream or_stream{stream};

    const auto orig_device = current_device();
    set_device(stream.device().index());

    if (!or_event) {
      auto or_flag = orEventDisableTiming;
      switch (flag) {
        case EventFlag::PYTORCH_DEFAULT:
          or_flag = orEventDisableTiming;
          break;
        case EventFlag::BACKEND_DEFAULT:
          or_flag = orEventEnableTiming;
          break;
        default:
          TORCH_CHECK(false, "Received unknown flag");
      }

      orEventCreateWithFlags(&or_event, or_flag);
    }

    orEventRecord(or_event, or_stream);
    *event = or_event;

    set_device(orig_device);
  }

  /**
   * Does nothing if the event has not been scheduled to be recorded.
   * If the event was previously enqueued to be recorded, a command
   * to wait for the version of the event that exists at the time of this call
   * is inserted in the stream's work queue.
   * When the stream reaches this command it will stop processing
   * additional commands until that version of the event is marked as recorded.
   */
  void block(void* event, const c10::Stream& stream) const override {
    if (!event)
      return;

    orEvent_t or_event = static_cast<orEvent_t>(event);
    OpenRegStream or_stream{stream};
    const auto orig_device = current_device();
    set_device(stream.device().index());
    orStreamWaitEvent(or_stream, or_event, 0);
    set_device(orig_device);
  }

  /**
   * Returns true if (and only if)
   *  (1) the event has never been scheduled to be recorded
   *  (2) the current version is marked as recorded.
   * Returns false otherwise.
   */
  bool queryEvent(void* event) const override {
    if (!event)
      return true;

    orEvent_t or_event = static_cast<orEvent_t>(event);
    const orError_t err = orEventQuery(or_event);

    return err == orSuccess ? true : false;
  }

  /**
   * Return true if all the work previously enqueued on the stream for
   * asynchronous execution has completed running on the device.
   */
  bool queryStream(const c10::Stream& stream) const override {
    OpenRegStream or_stream{stream};
    return or_stream.query();
  }

  /**
   * Wait (by blocking the calling thread) until all the work previously
   * enqueued on the stream has completed running on the device.
   */
  void synchronizeStream(const c10::Stream& stream) const override {
    OpenRegStream or_stream{stream};
    or_stream.synchronize();
  }

  /**
   * Wait (by blocking the calling thread) until all the work previously
   * recorded on the event has completed running on the device.
   */
  void synchronizeEvent(void* event) const override {
    if (!event)
      return;

    orEvent_t or_event = static_cast<orEvent_t>(event);
    orEventSynchronize(or_event);
  }

  /**
   * Wait (by blocking the calling thread) until all the work has
   * completed running on the device.
   */
  void synchronizeDevice(const c10::DeviceIndex device_index) const override {
    DeviceIndex orig_device{-1};
    auto orig_devicec = current_device();
    set_device(device_index);
    orDeviceSynchronize();
    set_device(orig_device);
  }

  /**
   * Fetch the elapsed time between two recorded events.
   */
  double elapsedTime(
      void* event1,
      void* event2,
      const c10::DeviceIndex device_index) const override {
    TORCH_CHECK(
        event1 && event2,
        "Both events must be recorded before calculating elapsed time.");
    auto orig_device = current_device();
    set_device(device_index);

    orEvent_t or_event1 = static_cast<orEvent_t>(event1);
    orEvent_t or_event2 = static_cast<orEvent_t>(event2);
    float time_ms = 0;
    orEventElapsedTime(&time_ms, or_event1, or_event2);

    set_device(orig_device);

    return static_cast<double>(time_ms);
  }
};

} // namespace c10::openreg

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 28 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `OpenRegGuardImpl`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/Device.h`
- `c10/core/impl/DeviceGuardImplInterface.h`
- `include/openreg.h`
- `OpenRegEvent.h`
- `OpenRegFunctions.h`
- `OpenRegStream.h`


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
python test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.h
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
- [`OpenRegDeviceAllocator.h_docs.md`](./OpenRegDeviceAllocator.h_docs.md)
- [`OpenRegException.cpp_docs.md`](./OpenRegException.cpp_docs.md)


## Cross-References

- **File Documentation**: `OpenRegGuard.h_docs.md`
- **Keyword Index**: `OpenRegGuard.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
