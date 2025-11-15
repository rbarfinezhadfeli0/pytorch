# Documentation: `c10/core/GeneratorImpl.h`

## File Metadata

- **Path**: `c10/core/GeneratorImpl.h`
- **Size**: 3,937 bytes (3.84 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <cstdint>
#include <mutex>

#include <c10/core/Device.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/TensorImpl.h>
#include <c10/macros/Export.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/python_stub.h>

/**
 * Note [Generator]
 * ~~~~~~~~~~~~~~~~
 * A Pseudo Random Number Generator (PRNG) is an engine that uses an algorithm
 * to generate a seemingly random sequence of numbers, that may be later be used
 * in creating a random distribution. Such an engine almost always maintains a
 * state and requires a seed to start off the creation of random numbers. Often
 * times, users have found it beneficial to be able to explicitly create,
 * retain, and destroy PRNG states and also be able to have control over the
 * seed value.
 *
 * A Generator in ATen gives users the ability to read, write and modify a PRNG
 * engine. For instance, it does so by letting users seed a PRNG engine, fork
 * the state of the engine, etc.
 *
 * By default, there is one generator per device, and a device's generator is
 * lazily created. A user can use the torch.Generator() api to create their own
 * generator. Currently torch.Generator() can only create a CPUGeneratorImpl.
 */

/**
 * Note [Acquire lock when using random generators]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Generator and its derived classes are NOT thread-safe. Please note that most
 * of the places where we have inserted locking for generators are historically
 * based, and we haven't actually checked that everything is truly thread safe
 * (and it probably isn't). Please use the public mutex_ when using any methods
 * from these classes, except for the read-only methods. You can learn about the
 * usage by looking into the unittests (aten/src/ATen/cpu_generator_test.cpp)
 * and other places where we have used lock_guard.
 *
 * TODO: Look into changing the threading semantics of Generators in ATen (e.g.,
 * making them non-thread safe and instead making the generator state
 * splittable, to accommodate forks into other threads).
 */

namespace c10 {

// The default seed is selected to be a large number
// with good distribution of 0s and 1s in bit representation
constexpr uint64_t default_rng_seed_val = 67280421310721;

struct C10_API GeneratorImpl : public c10::intrusive_ptr_target {
  // Constructors
  GeneratorImpl(Device device_in, DispatchKeySet key_set);

  // Delete all copy and move assignment in favor of clone()
  // method
  GeneratorImpl(const GeneratorImpl& other) = delete;
  GeneratorImpl(GeneratorImpl&& other) = delete;
  GeneratorImpl& operator=(const GeneratorImpl& other) = delete;
  GeneratorImpl& operator=(GeneratorImpl&& other) = delete;

  ~GeneratorImpl() override = default;
  c10::intrusive_ptr<GeneratorImpl> clone() const;

  // Common methods for all generators
  virtual void set_current_seed(uint64_t seed) = 0;
  virtual void set_offset(uint64_t offset) = 0;
  virtual uint64_t get_offset() const = 0;
  virtual uint64_t current_seed() const = 0;
  virtual uint64_t seed() = 0;
  virtual void set_state(const c10::TensorImpl& new_state) = 0;
  virtual c10::intrusive_ptr<c10::TensorImpl> get_state() const = 0;
  virtual void graphsafe_set_state(
      const c10::intrusive_ptr<c10::GeneratorImpl>& new_state);
  virtual c10::intrusive_ptr<c10::GeneratorImpl> graphsafe_get_state() const;
  Device device() const;

  // See Note [Acquire lock when using random generators]
  std::mutex mutex_;

  DispatchKeySet key_set() const {
    return key_set_;
  }

  inline void set_pyobj(PyObject* pyobj) noexcept {
    pyobj_ = pyobj;
  }

  inline PyObject* pyobj() const noexcept {
    return pyobj_;
  }

 protected:
  Device device_;
  DispatchKeySet key_set_;
  PyObject* pyobj_ = nullptr;

  virtual GeneratorImpl* clone_impl() const = 0;
};

namespace detail {

C10_API uint64_t getNonDeterministicRandom(bool is_cuda = false);

} // namespace detail

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `detail`, `c10`

**Classes/Structs**: `C10_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/core`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `cstdint`
- `mutex`
- `c10/core/Device.h`
- `c10/core/DispatchKeySet.h`
- `c10/core/TensorImpl.h`
- `c10/macros/Export.h`
- `c10/util/intrusive_ptr.h`
- `c10/util/python_stub.h`


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

- **File Documentation**: `GeneratorImpl.h_docs.md`
- **Keyword Index**: `GeneratorImpl.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
