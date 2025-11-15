# Documentation: `aten/src/ATen/native/metal/MetalPrepackOpRegister.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/metal/MetalPrepackOpRegister.cpp`
- **Size**: 4,736 bytes (4.62 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/metal/MetalPrepackOpContext.h>
#include <c10/util/accumulate.h>

namespace at::native::metal {

static c10::intrusive_ptr<Conv2dOpContext> unpack(
    Tensor&& weight,
    std::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  auto packedWeight = weight.contiguous(MemoryFormat::ChannelsLast);
  return c10::make_intrusive<Conv2dOpContext>(
      std::move(packedWeight),
      std::move(bias),
      stride,
      padding,
      dilation,
      groups,
      output_min,
      output_max);
}

static c10::intrusive_ptr<LinearOpContext> unpack(
    Tensor&& weight,
    std::optional<Tensor>&& bias,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  TORCH_CHECK(weight.dim() == 2);
  // Don't need to do `weight.t()`
  auto packedWeight = weight.view({weight.size(0), weight.size(1), 1, 1})
                          .contiguous(MemoryFormat::ChannelsLast);
  return c10::make_intrusive<LinearOpContext>(
      std::move(packedWeight), std::move(bias), output_min, output_max);
}

TORCH_LIBRARY(metal, m) {
  m.class_<Conv2dOpContext>("Conv2dOpContext")
      .def_pickle(
          [](const c10::intrusive_ptr<Conv2dOpContext>& op_context)
              -> SerializationTypeConv2dPrePack { // __getstate__
            return op_context->pack();
          },
          [](SerializationTypeConv2dPrePack state)
              -> c10::intrusive_ptr<Conv2dOpContext> { // __setstate__
            return unpack(
                std::move(std::get<0>(state)),
                std::move(std::get<1>(state)),
                std::move(std::get<2>(state)),
                std::move(std::get<3>(state)),
                std::move(std::get<4>(state)),
                std::move(std::get<5>(state)),
                std::move(std::get<6>(state)),
                std::move(std::get<7>(state)));
          });
  m.class_<LinearOpContext>("LinearOpContext")
      .def_pickle(
          [](const c10::intrusive_ptr<LinearOpContext>& op_context)
              -> SerializationTypeLinearPrePack { // __getstate__
            return op_context->pack();
          },
          [](SerializationTypeLinearPrePack state)
              -> c10::intrusive_ptr<LinearOpContext> { // __setstate__
            return unpack(
                std::move(std::get<0>(state)),
                std::move(std::get<1>(state)),
                std::get<2>(state),
                std::get<3>(state));
          });
  m.def("copy_to_host(Tensor X) -> Tensor Y");
}

TORCH_LIBRARY(metal_prepack, m) {
  m.def(
      TORCH_SELECTIVE_SCHEMA("metal_prepack::conv2d_prepack(Tensor W, Tensor? B, int[2] stride, "
      "int[2] padding, int[2] dilation, int groups, "
      "Scalar? output_min=None, Scalar? output_max=None) "
      "-> __torch__.torch.classes.metal.Conv2dOpContext"));
  m.def(
      TORCH_SELECTIVE_SCHEMA("metal_prepack::conv2d_run(Tensor X, "
      "__torch__.torch.classes.metal.Conv2dOpContext W_prepack) -> Tensor Y"));

  m.def(
      TORCH_SELECTIVE_SCHEMA("metal_prepack::linear_prepack(Tensor W, Tensor? B, Scalar? output_min=None, Scalar? output_max=None) -> __torch__.torch.classes.metal.LinearOpContext"));

  m.def(
      TORCH_SELECTIVE_SCHEMA("metal_prepack::linear_run(Tensor X, __torch__.torch.classes.metal.LinearOpContext W_prepack) -> Tensor Y"));
}

static c10::intrusive_ptr<Conv2dOpContext> conv2d_prepack(
    Tensor&& weight,
    std::optional<Tensor>&& bias,
    std::vector<int64_t>&& stride,
    std::vector<int64_t>&& padding,
    std::vector<int64_t>&& dilation,
    int64_t groups,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  TORCH_CHECK(weight.dim() == 4);
  return c10::make_intrusive<Conv2dOpContext>(
      std::move(weight),
      std::move(bias),
      stride,
      padding,
      dilation,
      groups,
      output_min,
      output_max);
}

static c10::intrusive_ptr<LinearOpContext> linear_prepack(
    Tensor&& weight,
    std::optional<Tensor>&& bias,
    const std::optional<Scalar>& output_min,
    const std::optional<Scalar>& output_max) {
  return c10::make_intrusive<LinearOpContext>(
      std::move(weight), std::move(bias), output_min, output_max);
}

TORCH_LIBRARY_IMPL(metal_prepack, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("metal_prepack::conv2d_prepack"), TORCH_FN(conv2d_prepack));
  m.impl(TORCH_SELECTIVE_NAME("metal_prepack::linear_prepack"), TORCH_FN(linear_prepack));
}

} // namespace at::native::metal

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/metal`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `ATen/core/op_registration/op_registration.h`
- `ATen/native/metal/MetalPrepackOpContext.h`
- `c10/util/accumulate.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/native/metal`):

- [`MetalShaders.h_docs.md`](./MetalShaders.h_docs.md)
- [`MetalDevice.h_docs.md`](./MetalDevice.h_docs.md)
- [`MetalGuardImpl.cpp_docs.md`](./MetalGuardImpl.cpp_docs.md)
- [`MetalNeuronType.h_docs.md`](./MetalNeuronType.h_docs.md)
- [`MetalTensorImplStorage.h_docs.md`](./MetalTensorImplStorage.h_docs.md)
- [`MetalConvParams.h_docs.md`](./MetalConvParams.h_docs.md)
- [`MetalCommandBuffer.h_docs.md`](./MetalCommandBuffer.h_docs.md)
- [`MetalTensorUtils.h_docs.md`](./MetalTensorUtils.h_docs.md)
- [`MetalTensorImpl.h_docs.md`](./MetalTensorImpl.h_docs.md)


## Cross-References

- **File Documentation**: `MetalPrepackOpRegister.cpp_docs.md`
- **Keyword Index**: `MetalPrepackOpRegister.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
