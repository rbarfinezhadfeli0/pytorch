# Documentation: `c10/core/Event.h`

## File Metadata

- **Path**: `c10/core/Event.h`
- **Size**: 4,456 bytes (4.35 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/InlineEvent.h>
#include <c10/core/impl/VirtualGuardImpl.h>

namespace c10 {

/**
 * A backend-generic movable, not copyable, not thread-safe event.
 *
 * The design of this event follows that of CUDA and HIP events. These events
 * are recorded and waited on by streams and can be rerecorded to,
 * each rerecording essentially creating a new version of the event.
 * For example, if (in CPU time), stream X is asked to record E,
 * stream Y waits on E, and stream X is asked to record E again, then Y will
 * wait for X to finish the first call to record and not the second, because
 * it's waiting on the first version of event E, not the second.
 * Querying an event only returns the status of its most recent version.
 *
 * Backend-generic events are implemented by this class and
 * impl::InlineEvent. In addition to these events there are also
 * some backend-specific events, like ATen's CUDAEvent. Each of these
 * classes has its own use.
 *
 * impl::InlineEvent<...> or a backend-specific event should be
 * preferred when the backend is known at compile time and known to
 * be compiled. Backend-specific events may have additional functionality.
 *
 * This Event should be used if a particular backend may not be available,
 * or the backend required is not known at compile time.
 *
 * These generic events are built on top of DeviceGuardImpls, analogous
 * to DeviceGuard and InlineDeviceGuard. The name "DeviceGuardImpls,"
 * is no longer entirely accurate, as these classes implement the
 * backend-specific logic for a generic backend interface.
 *
 * See DeviceGuardImplInterface.h for a list of all supported flags.
 */

struct Event final {
  // Constructors
  Event() = delete;
  Event(
      const DeviceType _device_type,
      const EventFlag _flag = EventFlag::PYTORCH_DEFAULT)
      : impl_{_device_type, _flag} {}

  // Copy constructor and copy assignment operator (deleted)
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;

  // Move constructor and move assignment operator
  Event(Event&&) noexcept = default;
  Event& operator=(Event&&) noexcept = default;

  // Destructor
  ~Event() = default;

  // Getters
  Device device() const noexcept {
    return Device(device_type(), device_index());
  }
  DeviceType device_type() const noexcept {
    return impl_.device_type();
  }
  DeviceIndex device_index() const noexcept {
    return impl_.device_index();
  }
  EventFlag flag() const noexcept {
    return impl_.flag();
  }
  bool was_marked_for_recording() const noexcept {
    return impl_.was_marked_for_recording();
  }

  /**
   * Calls record() if and only if record() has never been called for this
   * event. Note: because Event is not thread-safe recordOnce() may call
   * record() multiple times if called from multiple threads.
   */
  void recordOnce(const Stream& stream) {
    impl_.recordOnce(stream);
  }

  /**
   * Increments the event's version and enqueues a job with this version
   * in the stream's work queue. When the stream process that job
   * it notifies all streams waiting on / blocked by that version of the
   * event to continue and marks that version as recorded.
   * */
  void record(const Stream& stream) {
    impl_.record(stream);
  }

  /**
   * Does nothing if the event has not been scheduled to be recorded.
   * If the event was previously enqueued to be recorded, a command
   * to wait for the version of the event that exists at the time of this call
   * is inserted in the stream's work queue.
   * When the stream reaches this command it will stop processing
   * additional commands until that version of the event is marked as recorded.
   */
  void block(const Stream& stream) const {
    impl_.block(stream);
  }

  /**
   * Returns true if (and only if)
   *  (1) the event has never been scheduled to be recorded
   *  (2) the current version is marked as recorded.
   * Returns false otherwise.
   */
  bool query() const {
    return impl_.query();
  }

  double elapsedTime(const Event& event) const {
    return impl_.elapsedTime(event.impl_);
  }

  void* eventId() const {
    return impl_.eventId();
  }

  void synchronize() const {
    impl_.synchronize();
  }

 private:
  impl::InlineEvent<impl::VirtualGuardImpl> impl_;
};

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 19 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `and`, `Event`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/core`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/Device.h`
- `c10/core/DeviceType.h`
- `c10/core/Stream.h`
- `c10/core/impl/DeviceGuardImplInterface.h`
- `c10/core/impl/InlineEvent.h`
- `c10/core/impl/VirtualGuardImpl.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

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

Files in the same folder (`c10/core`):

- [`DispatchKey.cpp_docs.md`](./DispatchKey.cpp_docs.md)
- [`CopyBytes.h_docs.md`](./CopyBytes.h_docs.md)
- [`OptionalRef.h_docs.md`](./OptionalRef.h_docs.md)
- [`TensorOptions.h_docs.md`](./TensorOptions.h_docs.md)
- [`MemoryFormat.h_docs.md`](./MemoryFormat.h_docs.md)
- [`SafePyObject.cpp_docs.md`](./SafePyObject.cpp_docs.md)
- [`DeviceType.cpp_docs.md`](./DeviceType.cpp_docs.md)
- [`SymBool.cpp_docs.md`](./SymBool.cpp_docs.md)
- [`Allocator.cpp_docs.md`](./Allocator.cpp_docs.md)
- [`SymbolicShapeMeta.cpp_docs.md`](./SymbolicShapeMeta.cpp_docs.md)


## Cross-References

- **File Documentation**: `Event.h_docs.md`
- **Keyword Index**: `Event.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
