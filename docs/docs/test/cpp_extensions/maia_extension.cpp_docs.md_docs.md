# Documentation: `docs/test/cpp_extensions/maia_extension.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp_extensions/maia_extension.cpp_docs.md`
- **Size**: 7,566 bytes (7.39 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/cpp_extensions/maia_extension.cpp`

## File Metadata

- **Path**: `test/cpp_extensions/maia_extension.cpp`
- **Size**: 5,021 bytes (4.90 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
#include <torch/extension.h>
#include <torch/library.h>

using namespace at;

static int test_int;

Tensor get_tensor(caffe2::TypeMeta dtype, IntArrayRef size) {
  auto tensor_impl = c10::make_intrusive<TensorImpl, UndefinedTensorImpl>(
      Storage(
          Storage::use_byte_size_t(),
          0,
          at::DataPtr(nullptr, Device(DeviceType::MAIA, 0)),
          nullptr,
          false),
      DispatchKey::MAIA,
      dtype);
  // This is a hack to workaround the shape checks in _convolution.
  tensor_impl->set_sizes_contiguous(size);
  return Tensor(std::move(tensor_impl));
}

Tensor empty_override(IntArrayRef size, std::optional<ScalarType> dtype, std::optional<Layout> layout, std::optional<Device> device,
                      std::optional<bool> pin_memory, std::optional<c10::MemoryFormat> optional_memory_format) {
  test_int = 0;
  return get_tensor(scalarTypeToTypeMeta(dtype_or_default(dtype)), size);
}

Tensor& add_out_override(const Tensor & a, const Tensor & b , const Scalar& c, Tensor & out) {
  test_int = 1;
  return out;
}

Tensor fake_convolution(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation,
    bool transposed, IntArrayRef output_padding, int64_t groups) {
  test_int = 2;
  // Only the first 2 dimension of output shape is correct.
  return get_tensor(input.dtype(), {input.size(0), weight.size(0), input.size(2), input.size(3)});
}

std::tuple<Tensor,Tensor,Tensor> fake_convolution_backward(
        const Tensor & grad_output, const Tensor & input, const Tensor & weight,
        IntArrayRef stride, IntArrayRef padding,
        IntArrayRef dilation, bool transposed, IntArrayRef output_padding,
        int64_t groups, std::array<bool,3> output_mask) {
    test_int = 3;
    return std::tuple<Tensor, Tensor, Tensor>(
            get_tensor(input.dtype(), input.sizes()),
            get_tensor(weight.dtype(), weight.sizes()),
            get_tensor(input.dtype(), {}));
}

at::Tensor maia_to_dtype_override(
  const at::Tensor & self, at::ScalarType dtype, bool non_blocking,
  bool copy, ::std::optional<at::MemoryFormat> memory_format
) {
  return get_tensor(scalarTypeToTypeMeta(dtype), self.sizes());
}

at::Tensor maia_matmul_override(const at::Tensor & self, const at::Tensor & other) {
  AT_ASSERT(self.dim() == 2);
  AT_ASSERT(other.dim() == 2);
  AT_ASSERT(self.dtype() == other.dtype());
  AT_ASSERT(self.device() == other.device());
  return get_tensor(self.dtype(), {self.size(0), other.size(1)});
}

TORCH_LIBRARY_IMPL(aten, MAIA, m) {
  m.impl("empty.memory_format",                empty_override);
  m.impl("add.out",                            add_out_override);
  m.impl("convolution_overrideable",           fake_convolution);
  m.impl("convolution_backward_overrideable",  fake_convolution_backward);
  m.impl("to.dtype",                           maia_to_dtype_override);
  m.impl("matmul",                             maia_matmul_override);
}

// TODO: Extend this to exercise multi-device setting.  In that case,
// we need to add a thread local variable to track the current device.
struct MAIAGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::MAIA;
  MAIAGuardImpl() {}
  MAIAGuardImpl(DeviceType t) {
    AT_ASSERT(t == DeviceType::MAIA);
  }
  DeviceType type() const override {
    return DeviceType::MAIA;
  }
  Device exchangeDevice(Device d) const override {
    AT_ASSERT(d.type() == DeviceType::MAIA);
    AT_ASSERT(d.index() == 0);
    return d;
  }
  Device getDevice() const override {
    return Device(DeviceType::MAIA, 0);
  }
  void setDevice(Device d) const override {
    AT_ASSERT(d.type() == DeviceType::MAIA);
    AT_ASSERT(d.index() == 0);
  }
  void uncheckedSetDevice(Device d) const noexcept override {
  }
  Stream getStream(Device d) const noexcept override {
    return Stream(Stream::DEFAULT, Device(DeviceType::MAIA, 0));
  }
  Stream exchangeStream(Stream s) const noexcept override {
    return Stream(Stream::DEFAULT, Device(DeviceType::MAIA, 0));
  }
  DeviceIndex deviceCount() const noexcept override {
    return 1;
  }

  // Event-related functions
  void record(void** event,
    const Stream& stream,
    const DeviceIndex device_index,
    const EventFlag flag) const override {
    TORCH_CHECK(false, "MAIA backend doesn't support events.");
  }
  void block(
    void* event,
    const Stream& stream) const override {
    TORCH_CHECK(false, "MAIA backend doesn't support events.");
  }
  bool queryEvent(void* event) const override {
    TORCH_CHECK(false, "MAIA backend doesn't support events.");
  }
  void destroyEvent(
    void* event,
    const DeviceIndex device_index) const noexcept override { }
};

constexpr DeviceType MAIAGuardImpl::static_type;
C10_REGISTER_GUARD_IMPL(MAIA, MAIAGuardImpl);

int get_test_int() {
  return test_int;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_test_int", &get_test_int);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 26 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `MAIAGuardImpl`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp_extensions`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/extension.h`
- `torch/library.h`


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
python test/cpp_extensions/maia_extension.cpp
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
- [`rng_extension.cpp_docs.md`](./rng_extension.cpp_docs.md)
- [`cusolver_extension.cpp_docs.md`](./cusolver_extension.cpp_docs.md)
- [`cuda_dlink_extension.cpp_docs.md`](./cuda_dlink_extension.cpp_docs.md)
- [`cuda_dlink_extension_add.cu_docs.md`](./cuda_dlink_extension_add.cu_docs.md)


## Cross-References

- **File Documentation**: `maia_extension.cpp_docs.md`
- **Keyword Index**: `maia_extension.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp_extensions`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp_extensions`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python docs/test/cpp_extensions/maia_extension.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp_extensions`):

- [`cpp_frontend_extension.cpp_docs.md_docs.md`](./cpp_frontend_extension.cpp_docs.md_docs.md)
- [`cuda_dlink_extension_add.cu_docs.md_docs.md`](./cuda_dlink_extension_add.cu_docs.md_docs.md)
- [`cpp_c10d_extension.cpp_docs.md_docs.md`](./cpp_c10d_extension.cpp_docs.md_docs.md)
- [`setup.py_kw.md_docs.md`](./setup.py_kw.md_docs.md)
- [`extension.cpp_kw.md_docs.md`](./extension.cpp_kw.md_docs.md)
- [`jit_extension.cpp_docs.md_docs.md`](./jit_extension.cpp_docs.md_docs.md)
- [`cuda_dlink_extension_kernel.cu_kw.md_docs.md`](./cuda_dlink_extension_kernel.cu_kw.md_docs.md)
- [`cuda_extension_kernel2.cu_kw.md_docs.md`](./cuda_extension_kernel2.cu_kw.md_docs.md)
- [`mtia_extension.cpp_kw.md_docs.md`](./mtia_extension.cpp_kw.md_docs.md)
- [`setup.py_docs.md_docs.md`](./setup.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `maia_extension.cpp_docs.md_docs.md`
- **Keyword Index**: `maia_extension.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
