# Documentation: `torch/csrc/profiler/stubs/base.cpp`

## File Metadata

- **Path**: `torch/csrc/profiler/stubs/base.cpp`
- **Size**: 3,082 bytes (3.01 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <torch/csrc/profiler/stubs/base.h>
#include <cstdint>
#include <functional>

namespace torch::profiler::impl {

namespace {
struct DefaultStubs : public ProfilerStubs {
  explicit DefaultStubs(const char* name) : name_{name} {}

  void record(
      c10::DeviceIndex* /*device*/,
      ProfilerVoidEventStub* /*event*/,
      int64_t* /*cpu_ns*/) const override {
    fail();
  }
  float elapsed(
      const ProfilerVoidEventStub* /*event*/,
      const ProfilerVoidEventStub* /*event2*/) const override {
    fail();
    return 0.F;
  }
  void mark(const char* /*name*/) const override {
    fail();
  }
  void rangePush(const char* /*name*/) const override {
    fail();
  }
  void rangePop() const override {
    fail();
  }
  bool enabled() const override {
    return false;
  }
  void onEachDevice(std::function<void(int)> /*op*/) const override {
    fail();
  }
  void synchronize() const override {
    fail();
  }
  ~DefaultStubs() override = default;

 private:
  void fail() const {
    TORCH_CHECK(false, name_, " used in profiler but not enabled.");
  }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const char* const name_;
};
} // namespace

#define REGISTER_DEFAULT(name, upper_name)                                   \
  namespace {                                                                \
  const DefaultStubs default_##name##_stubs{#upper_name};                    \
  constexpr const DefaultStubs* default_##name##_stubs_addr =                \
      &default_##name##_stubs;                                               \
                                                                             \
  /* Constant initialization, so it is guaranteed to be initialized before*/ \
  /* static initialization calls which may invoke register<name>Methods*/    \
  inline const ProfilerStubs*& name##_stubs() {                              \
    static const ProfilerStubs* stubs_ =                                     \
        static_cast<const ProfilerStubs*>(default_##name##_stubs_addr);      \
    return stubs_;                                                           \
  }                                                                          \
  } /*namespace*/                                                            \
                                                                             \
  const ProfilerStubs* name##Stubs() {                                       \
    return name##_stubs();                                                   \
  }                                                                          \
                                                                             \
  void register##upper_name##Methods(ProfilerStubs* stubs) {                 \
    name##_stubs() = stubs;                                                  \
  }

REGISTER_DEFAULT(cuda, CUDA)
REGISTER_DEFAULT(itt, ITT)
REGISTER_DEFAULT(privateuse1, PrivateUse1)
#undef REGISTER_DEFAULT

} // namespace torch::profiler::impl

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `DefaultStubs`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/profiler/stubs`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/Device.h`
- `c10/util/Exception.h`
- `torch/csrc/profiler/stubs/base.h`
- `cstdint`
- `functional`


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

Files in the same folder (`torch/csrc/profiler/stubs`):

- [`base.h_docs.md`](./base.h_docs.md)
- [`cuda.cpp_docs.md`](./cuda.cpp_docs.md)
- [`itt.cpp_docs.md`](./itt.cpp_docs.md)


## Cross-References

- **File Documentation**: `base.cpp_docs.md`
- **Keyword Index**: `base.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
