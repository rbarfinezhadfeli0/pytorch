# Documentation: `test/cpp_extensions/rng_extension.cpp`

## File Metadata

- **Path**: `test/cpp_extensions/rng_extension.cpp`
- **Size**: 2,715 bytes (2.65 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/cpu/DistributionTemplates.h>
#include <memory>

using namespace at;

static size_t instance_count = 0;

struct TestCPUGenerator : public c10::GeneratorImpl {
  TestCPUGenerator(uint64_t value) : c10::GeneratorImpl{Device(DeviceType::CPU), DispatchKeySet(DispatchKey::CustomRNGKeyId)}, value_(value) {
    ++instance_count;
  }
  ~TestCPUGenerator() {
    --instance_count;
  }
  uint32_t random() { return static_cast<uint32_t>(value_); }
  uint64_t random64() { return value_; }
  void set_current_seed(uint64_t seed) override { throw std::runtime_error("not implemented"); }
  void set_offset(uint64_t offset) override { throw std::runtime_error("not implemented"); }
  uint64_t get_offset() const override { throw std::runtime_error("not implemented"); }
  uint64_t current_seed() const override { throw std::runtime_error("not implemented"); }
  uint64_t seed() override { throw std::runtime_error("not implemented"); }
  void set_state(const c10::TensorImpl& new_state) override { throw std::runtime_error("not implemented"); }
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override { throw std::runtime_error("not implemented"); }
  TestCPUGenerator* clone_impl() const override { throw std::runtime_error("not implemented"); }

  static DeviceType device_type() { return DeviceType::CPU; }

  uint64_t value_;
};

Tensor& random_(Tensor& self, std::optional<Generator> generator) {
  return at::native::templates::random_impl<native::templates::cpu::RandomKernel, TestCPUGenerator>(self, generator);
}

Tensor& random_from_to(Tensor& self, int64_t from, optional<int64_t> to, std::optional<Generator> generator) {
  return at::native::templates::random_from_to_impl<native::templates::cpu::RandomFromToKernel, TestCPUGenerator>(self, from, to, generator);
}

Tensor& random_to(Tensor& self, int64_t to, std::optional<Generator> generator) {
  return random_from_to(self, 0, to, generator);
}

Generator createTestCPUGenerator(uint64_t value) {
  return at::make_generator<TestCPUGenerator>(value);
}

Generator identity(Generator g) {
  return g;
}

size_t getInstanceCount() {
  return instance_count;
}

TORCH_LIBRARY_IMPL(aten, CustomRNGKeyId, m) {
  m.impl("aten::random_.from",                 random_from_to);
  m.impl("aten::random_.to",                   random_to);
  m.impl("aten::random_",                      random_);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createTestCPUGenerator", &createTestCPUGenerator);
  m.def("getInstanceCount", &getInstanceCount);
  m.def("identity", &identity);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `TestCPUGenerator`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp_extensions`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/extension.h`
- `torch/library.h`
- `ATen/Generator.h`
- `ATen/Tensor.h`
- `ATen/native/DistributionTemplates.h`
- `ATen/native/cpu/DistributionTemplates.h`
- `memory`


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
python test/cpp_extensions/rng_extension.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp_extensions`):

- [`cpp_frontend_extension.cpp_docs.md`](./cpp_frontend_extension.cpp_docs.md)
- [`extension.cpp_docs.md`](./extension.cpp_docs.md)
- [`identity.cpp_docs.md`](./identity.cpp_docs.md)
- [`doubler.h_docs.md`](./doubler.h_docs.md)
- [`open_registration_extension.cpp_docs.md`](./open_registration_extension.cpp_docs.md)
- [`setup.py_docs.md`](./setup.py_docs.md)
- [`cusolver_extension.cpp_docs.md`](./cusolver_extension.cpp_docs.md)
- [`cuda_dlink_extension.cpp_docs.md`](./cuda_dlink_extension.cpp_docs.md)
- [`cuda_dlink_extension_add.cu_docs.md`](./cuda_dlink_extension_add.cu_docs.md)


## Cross-References

- **File Documentation**: `rng_extension.cpp_docs.md`
- **Keyword Index**: `rng_extension.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
