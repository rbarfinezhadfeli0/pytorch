# Documentation: `c10/cuda/CUDAGuard.h`

## File Metadata

- **Path**: `c10/cuda/CUDAGuard.h`
- **Size**: 11,345 bytes (11.08 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/cuda/impl/CUDAGuardImpl.h>

namespace c10::cuda {

// This code is kind of boilerplatey.  See Note [Whither the DeviceGuard
// boilerplate]

/// A variant of DeviceGuard that is specialized for CUDA.  It accepts
/// integer indices (interpreting them as CUDA devices) and is a little
/// more efficient than DeviceGuard (it compiles to straight line
/// cudaSetDevice/cudaGetDevice calls); however, it can only be used
/// from code that links against CUDA directly.
struct CUDAGuard {
  /// No default constructor; see Note [Omitted default constructor from RAII]
  explicit CUDAGuard() = delete;

  /// Set the current CUDA device to the passed device index.
  explicit CUDAGuard(DeviceIndex device_index) : guard_(device_index) {}

  /// Sets the current CUDA device to the passed device.  Errors if the passed
  /// device is not a CUDA device.
  explicit CUDAGuard(Device device) : guard_(device) {}

  // Copy is not allowed
  CUDAGuard(const CUDAGuard&) = delete;
  CUDAGuard& operator=(const CUDAGuard&) = delete;

  // Move is not allowed (there is no uninitialized state)
  CUDAGuard(CUDAGuard&& other) = delete;
  CUDAGuard& operator=(CUDAGuard&& other) = delete;
  ~CUDAGuard() = default;

  /// Sets the CUDA device to the given device.  Errors if the given device
  /// is not a CUDA device.
  void set_device(Device device) {
    guard_.set_device(device);
  }

  /// Sets the CUDA device to the given device.  Errors if the given device
  /// is not a CUDA device.  (This method is provided for uniformity with
  /// DeviceGuard).
  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  /// Sets the CUDA device to the given device index.
  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set upon construction of the guard
  Device original_device() const {
    return guard_.original_device();
  }

  /// Returns the last device that was set via `set_device`, if any, otherwise
  /// the device passed during construction.
  Device current_device() const {
    return guard_.current_device();
  }

 private:
  /// The guard for the current device.
  c10::impl::InlineDeviceGuard<impl::CUDAGuardImpl> guard_;
};

/// A variant of OptionalDeviceGuard that is specialized for CUDA.  See
/// CUDAGuard for when you can use this.
struct OptionalCUDAGuard {
  /// Create an uninitialized OptionalCUDAGuard.
  explicit OptionalCUDAGuard() = default;

  /// Set the current CUDA device to the passed Device, if it is not nullopt.
  explicit OptionalCUDAGuard(std::optional<Device> device_opt)
      : guard_(device_opt) {}

  /// Set the current CUDA device to the passed device index, if it is not
  /// nullopt
  explicit OptionalCUDAGuard(std::optional<DeviceIndex> device_index_opt)
      : guard_(device_index_opt) {}

  // Copy is not allowed
  OptionalCUDAGuard(const OptionalCUDAGuard&) = delete;
  OptionalCUDAGuard& operator=(const OptionalCUDAGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalCUDAGuard(OptionalCUDAGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalCUDAGuard& operator=(OptionalCUDAGuard&& other) = delete;
  ~OptionalCUDAGuard() = default;

  /// Sets the CUDA device to the given device, initializing the guard if it
  /// is not already initialized.  Errors if the given device is not a CUDA
  /// device.
  void set_device(Device device) {
    guard_.set_device(device);
  }

  /// Sets the CUDA device to the given device, initializing the guard if it is
  /// not already initialized.  Errors if the given device is not a CUDA device.
  /// (This method is provided for uniformity with OptionalDeviceGuard).
  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  /// Sets the CUDA device to the given device index, initializing the guard if
  /// it is not already initialized.
  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set immediately prior to initialization of the
  /// guard, or nullopt if the guard is uninitialized.
  std::optional<Device> original_device() const {
    return guard_.original_device();
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device, if the guard is initialized,
  /// or nullopt if the guard is uninitialized.
  std::optional<Device> current_device() const {
    return guard_.current_device();
  }

  /// Restore the original CUDA device, resetting this guard to uninitialized
  /// state.
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalDeviceGuard<impl::CUDAGuardImpl> guard_;
};

/// A variant of StreamGuard that is specialized for CUDA.  See CUDAGuard
/// for when you can use this.
struct CUDAStreamGuard {
  /// No default constructor, see Note [Omitted default constructor from RAII]
  explicit CUDAStreamGuard() = delete;

  /// Set the current CUDA device to the device associated with the passed
  /// stream, and set the current CUDA stream on that device to the passed
  /// stream. Errors if the Stream is not a CUDA stream.
  explicit CUDAStreamGuard(Stream stream) : guard_(stream) {}
  ~CUDAStreamGuard() = default;

  /// Copy is disallowed
  CUDAStreamGuard(const CUDAStreamGuard&) = delete;
  CUDAStreamGuard& operator=(const CUDAStreamGuard&) = delete;

  /// Move is disallowed, as CUDAStreamGuard does not have an uninitialized
  /// state, which is required for moves on types with nontrivial destructors.
  CUDAStreamGuard(CUDAStreamGuard&& other) = delete;
  CUDAStreamGuard& operator=(CUDAStreamGuard&& other) = delete;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Errors if the stream passed is not a CUDA stream.
  ///
  /// NOTE: this implementation may skip some stream/device setting if
  /// it can prove that it is unnecessary.
  ///
  /// WARNING: reset_stream does NOT preserve previously set streams on
  /// different devices.  If you need to set streams on multiple devices
  /// on CUDA, use CUDAMultiStreamGuard instead.
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the CUDA stream that was set at the time the guard was
  /// constructed.
  CUDAStream original_stream() const {
    return CUDAStream(CUDAStream::UNCHECKED, guard_.original_stream());
  }

  /// Returns the most recent CUDA stream that was set using this device guard,
  /// either from construction, or via set_stream.
  CUDAStream current_stream() const {
    return CUDAStream(CUDAStream::UNCHECKED, guard_.current_stream());
  }

  /// Returns the most recent CUDA device that was set using this device guard,
  /// either from construction, or via set_device/reset_device/set_index.
  Device current_device() const {
    return guard_.current_device();
  }

  /// Returns the CUDA device that was set at the most recent reset_stream(),
  /// or otherwise the device at construction time.
  Device original_device() const {
    return guard_.original_device();
  }

 private:
  c10::impl::InlineStreamGuard<impl::CUDAGuardImpl> guard_;
};

/// A variant of OptionalStreamGuard that is specialized for CUDA.  See
/// CUDAGuard for when you can use this.
struct OptionalCUDAStreamGuard {
  /// Create an uninitialized guard.
  explicit OptionalCUDAStreamGuard() = default;

  /// Set the current CUDA device to the device associated with the passed
  /// stream, and set the current CUDA stream on that device to the passed
  /// stream. Errors if the Stream is not a CUDA stream.
  explicit OptionalCUDAStreamGuard(Stream stream) : guard_(stream) {}

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream,
  /// if the passed stream is not nullopt.
  explicit OptionalCUDAStreamGuard(std::optional<Stream> stream_opt)
      : guard_(stream_opt) {}

  /// Copy is disallowed
  OptionalCUDAStreamGuard(const OptionalCUDAStreamGuard&) = delete;
  OptionalCUDAStreamGuard& operator=(const OptionalCUDAStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalCUDAStreamGuard(OptionalCUDAStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalCUDAStreamGuard& operator=(OptionalCUDAStreamGuard&& other) = delete;
  ~OptionalCUDAStreamGuard() = default;

  /// Resets the currently set CUDA stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Initializes the guard if it was not previously initialized.
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the CUDA stream that was set at the time the guard was most
  /// recently initialized, or nullopt if the guard is uninitialized.
  std::optional<CUDAStream> original_stream() const {
    auto r = guard_.original_stream();
    if (r.has_value()) {
      return CUDAStream(CUDAStream::UNCHECKED, r.value());
    } else {
      return std::nullopt;
    }
  }

  /// Returns the most recent CUDA stream that was set using this stream guard,
  /// either from construction, or via reset_stream, if the guard is
  /// initialized, or nullopt if the guard is uninitialized.
  std::optional<CUDAStream> current_stream() const {
    auto r = guard_.current_stream();
    if (r.has_value()) {
      return CUDAStream(CUDAStream::UNCHECKED, r.value());
    } else {
      return std::nullopt;
    }
  }

  /// Restore the original CUDA device and stream, resetting this guard to
  /// uninitialized state.
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalStreamGuard<impl::CUDAGuardImpl> guard_;
};

/// A variant of MultiStreamGuard that is specialized for CUDA.
struct CUDAMultiStreamGuard {
  explicit CUDAMultiStreamGuard(ArrayRef<CUDAStream> streams)
      : guard_(unwrapStreams(streams)) {}

  /// Copy is disallowed
  CUDAMultiStreamGuard(const CUDAMultiStreamGuard&) = delete;
  CUDAMultiStreamGuard& operator=(const CUDAMultiStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  CUDAMultiStreamGuard(CUDAMultiStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  CUDAMultiStreamGuard& operator=(CUDAMultiStreamGuard&& other) = delete;
  ~CUDAMultiStreamGuard() = default;

 private:
  c10::impl::InlineMultiStreamGuard<impl::CUDAGuardImpl> guard_;

  static std::vector<Stream> unwrapStreams(ArrayRef<CUDAStream> cudaStreams) {
    std::vector<Stream> streams;
    streams.reserve(cudaStreams.size());
    for (const CUDAStream& cudaStream : cudaStreams) {
      streams.push_back(cudaStream);
    }
    return streams;
  }
};

} // namespace c10::cuda

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 41 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `CUDAGuard`, `OptionalCUDAGuard`, `CUDAStreamGuard`, `OptionalCUDAStreamGuard`, `CUDAMultiStreamGuard`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/cuda`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/DeviceType.h`
- `c10/core/impl/InlineDeviceGuard.h`
- `c10/core/impl/InlineStreamGuard.h`
- `c10/cuda/CUDAMacros.h`
- `c10/cuda/impl/CUDAGuardImpl.h`


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

Files in the same folder (`c10/cuda`):

- [`build.bzl_docs.md`](./build.bzl_docs.md)
- [`CUDACachingAllocator.h_docs.md`](./CUDACachingAllocator.h_docs.md)
- [`CUDAAlgorithm.h_docs.md`](./CUDAAlgorithm.h_docs.md)
- [`CUDAFunctions.h_docs.md`](./CUDAFunctions.h_docs.md)
- [`CUDAAllocatorConfig.cpp_docs.md`](./CUDAAllocatorConfig.cpp_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`CUDAMallocAsyncAllocator.cpp_docs.md`](./CUDAMallocAsyncAllocator.cpp_docs.md)
- [`BUILD.bazel_docs.md`](./BUILD.bazel_docs.md)
- [`CUDACachingAllocator.cpp_docs.md`](./CUDACachingAllocator.cpp_docs.md)
- [`CUDAException.h_docs.md`](./CUDAException.h_docs.md)


## Cross-References

- **File Documentation**: `CUDAGuard.h_docs.md`
- **Keyword Index**: `CUDAGuard.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
