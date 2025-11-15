# Documentation: `c10/core/GeneratorImpl.cpp`

## File Metadata

- **Path**: `c10/core/GeneratorImpl.cpp`
- **Size**: 3,300 bytes (3.22 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/core/GeneratorImpl.h>
#include <random>

#if defined(__SGX_ENABLED__)
#include <sgx_trts.h>
#endif

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#else
#include <chrono>
#endif

namespace c10 {

/**
 * GeneratorImpl class implementation
 */
GeneratorImpl::GeneratorImpl(Device device_in, DispatchKeySet key_set)
    : device_{device_in}, key_set_(key_set) {}

/**
 * Clone this generator. Note that clone() is the only
 * method for copying for Generators in ATen.
 */
c10::intrusive_ptr<GeneratorImpl> GeneratorImpl::clone() const {
  auto res = this->clone_impl();
  c10::raw::intrusive_ptr::incref(res);
  c10::raw::weak_intrusive_ptr::incref(res);
  return c10::intrusive_ptr<GeneratorImpl>::reclaim(res);
}

void GeneratorImpl::graphsafe_set_state(
    const c10::intrusive_ptr<c10::GeneratorImpl>& /*state*/) {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "graphsafe_set_state is not supported in this Generator");
}

c10::intrusive_ptr<c10::GeneratorImpl> GeneratorImpl::graphsafe_get_state()
    const {
  TORCH_CHECK_NOT_IMPLEMENTED(
      false, "graphsafe_get_state is not supported in this Generator");
}

/**
 * Gets the device of a generator.
 */
Device GeneratorImpl::device() const {
  return device_;
}

namespace detail {

/**
 * Gets a random number for /dev/urandom
 * Note this is a legacy method (from THRandom.cpp)
 * FIXME: use std::random_device with entropy information
 */
#if !defined(_WIN32)
static uint64_t readURandomLong() {
  int randDev = open("/dev/urandom", O_RDONLY);
  TORCH_CHECK(randDev >= 0, "Unable to open /dev/urandom");
  uint64_t randValue{};
  ssize_t readBytes = read(randDev, &randValue, sizeof(randValue));
  close(randDev);
  TORCH_CHECK(
      readBytes >= (ssize_t)sizeof(randValue),
      "Unable to read from /dev/urandom");
  return randValue;
}
#endif // _WIN32

/**
 * Gets a non deterministic random number number from either the
 * /dev/urandom or the current time. For CUDA, gets random from
 * std::random_device and adds a transformation on it. For Intel SGX
 * platform use sgx_read_rand as reading from /dev/urandom is
 * prohibited on that platform.
 *
 * FIXME: The behavior in this function is from legacy code
 * (THRandom_seed/THCRandom_seed) and is probably not the right thing to do,
 * even though our tests pass. Figure out if tests get perturbed
 * - when the same algorithm is used for all backends. Note that the current
 * behavior is different for CPU, CUDA and Windows CPU.
 * - when using C++11 std objects, such as std::random_device
 * - when constructing a 64 bit seed properly, rather than static casting
 *   a 32 bit number to 64 bit.
 */
uint64_t getNonDeterministicRandom(bool is_cuda) {
  uint64_t s = 0;
  if (!is_cuda) {
#ifdef _WIN32
    s = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
#elif defined(__SGX_ENABLED__)
    TORCH_CHECK(
        sgx_read_rand(reinterpret_cast<uint8_t*>(&s), sizeof(s)) == SGX_SUCCESS,
        "Could not generate random number with sgx_read_rand.");
#else
    s = readURandomLong();
#endif
  } else {
    std::random_device rd;
    // limit to 53 bits to ensure unique representation in double
    s = (((static_cast<uint64_t>(rd())) << 32) + rd()) & 0x1FFFFFFFFFFFFF;
  }
  return s;
}

} // namespace detail
} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `detail`, `c10`

**Classes/Structs**: `implementation`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/core`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/GeneratorImpl.h`
- `random`
- `sgx_trts.h`
- `fcntl.h`
- `unistd.h`
- `chrono`


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

- **File Documentation**: `GeneratorImpl.cpp_docs.md`
- **Keyword Index**: `GeneratorImpl.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
