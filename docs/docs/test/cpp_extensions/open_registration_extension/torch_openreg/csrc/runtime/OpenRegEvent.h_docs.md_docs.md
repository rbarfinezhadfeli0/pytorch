# Documentation: `docs/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegEvent.h_docs.md`

## File Metadata

- **Path**: `docs/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegEvent.h_docs.md`
- **Size**: 6,186 bytes (6.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegEvent.h`

## File Metadata

- **Path**: `test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegEvent.h`
- **Size**: 3,415 bytes (3.33 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```c
#pragma once

#include <include/openreg.h>

#include "OpenRegException.h"
#include "OpenRegStream.h"

namespace c10::openreg {

struct OpenRegEvent {
  OpenRegEvent(bool enable_timing) noexcept : enable_timing_{enable_timing} {}

  ~OpenRegEvent() {
    if (is_created_) {
      OPENREG_CHECK(orEventDestroy(event_));
    }
  }

  OpenRegEvent(const OpenRegEvent&) = delete;
  OpenRegEvent& operator=(const OpenRegEvent&) = delete;

  OpenRegEvent(OpenRegEvent&& other) noexcept {
    moveHelper(std::move(other));
  }
  OpenRegEvent& operator=(OpenRegEvent&& other) noexcept {
    if (this != &other) {
      moveHelper(std::move(other));
    }
    return *this;
  }

  operator orEvent_t() const {
    return event();
  }

  std::optional<at::Device> device() const {
    if (is_created_) {
      return at::Device(at::kPrivateUse1, device_index_);
    } else {
      return std::nullopt;
    }
  }

  bool isCreated() const {
    return is_created_;
  }

  DeviceIndex device_index() const {
    return device_index_;
  }

  orEvent_t event() const {
    return event_;
  }

  bool query() const {
    if (!is_created_) {
      return true;
    }

    orError_t err = orEventQuery(event_);
    if (err == orSuccess) {
      return true;
    }

    return false;
  }

  void record() {
    record(getCurrentOpenRegStream());
  }

  void recordOnce(const OpenRegStream& stream) {
    if (!was_recorded_)
      record(stream);
  }

  void record(const OpenRegStream& stream) {
    if (!is_created_) {
      createEvent(stream.device_index());
    }

    TORCH_CHECK(
        device_index_ == stream.device_index(),
        "Event device ",
        device_index_,
        " does not match recording stream's device ",
        stream.device_index(),
        ".");

    OPENREG_CHECK(orEventRecord(event_, stream));
    was_recorded_ = true;
  }

  void block(const OpenRegStream& stream) {
    if (is_created_) {
      OPENREG_CHECK(orStreamWaitEvent(stream, event_, 0));
    }
  }

  float elapsed_time(const OpenRegEvent& other) const {
    TORCH_CHECK_VALUE(
        !(enable_timing_ & orEventDisableTiming) &&
            !(other.enable_timing_ & orEventDisableTiming),
        "Both events must be created with argument 'enable_timing=True'.");
    TORCH_CHECK_VALUE(
        is_created_ && other.isCreated(),
        "Both events must be recorded before calculating elapsed time.");
    TORCH_CHECK(
        query() && other.query(),
        "Both events must be completed before calculating elapsed time.");

    float time_ms = 0;
    OPENREG_CHECK(orEventElapsedTime(&time_ms, event_, other.event_));
    return time_ms;
  }

  void synchronize() const {
    if (is_created_) {
      OPENREG_CHECK(orEventSynchronize(event_));
    }
  }

 private:
  unsigned int enable_timing_{orEventDisableTiming};
  bool is_created_{false};
  bool was_recorded_{false};
  DeviceIndex device_index_{-1};
  orEvent_t event_{};

  void createEvent(DeviceIndex device_index) {
    device_index_ = device_index;
    OPENREG_CHECK(orEventCreateWithFlags(&event_, enable_timing_));
    is_created_ = true;
  }

  void moveHelper(OpenRegEvent&& other) {
    std::swap(enable_timing_, other.enable_timing_);
    std::swap(is_created_, other.is_created_);
    std::swap(was_recorded_, other.was_recorded_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
  }
};

} // namespace c10::openreg

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 14 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `OpenRegEvent`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `include/openreg.h`
- `OpenRegException.h`
- `OpenRegStream.h`


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
python test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegEvent.h
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

- **File Documentation**: `OpenRegEvent.h_docs.md`
- **Keyword Index**: `OpenRegEvent.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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
python docs/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegEvent.h_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime`):

- [`OpenRegStream.cpp_kw.md_docs.md`](./OpenRegStream.cpp_kw.md_docs.md)
- [`OpenRegEvent.h_kw.md_docs.md`](./OpenRegEvent.h_kw.md_docs.md)
- [`OpenRegHooks.h_docs.md_docs.md`](./OpenRegHooks.h_docs.md_docs.md)
- [`OpenRegException.cpp_docs.md_docs.md`](./OpenRegException.cpp_docs.md_docs.md)
- [`OpenRegStream.h_kw.md_docs.md`](./OpenRegStream.h_kw.md_docs.md)
- [`OpenRegSerialization.h_kw.md_docs.md`](./OpenRegSerialization.h_kw.md_docs.md)
- [`OpenRegHostAllocator.cpp_docs.md_docs.md`](./OpenRegHostAllocator.cpp_docs.md_docs.md)
- [`OpenRegGenerator.cpp_kw.md_docs.md`](./OpenRegGenerator.cpp_kw.md_docs.md)
- [`OpenRegFunctions.h_kw.md_docs.md`](./OpenRegFunctions.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `OpenRegEvent.h_docs.md_docs.md`
- **Keyword Index**: `OpenRegEvent.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
