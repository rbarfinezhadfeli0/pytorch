# Documentation: `aten/src/ATen/native/xnnpack/RegisterOpContextClass.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/xnnpack/RegisterOpContextClass.cpp`
- **Size**: 5,219 bytes (5.10 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#ifdef USE_XNNPACK

#include <torch/library.h>
#include <ATen/native/xnnpack/Convolution.h>
#include <ATen/native/xnnpack/Linear.h>
#include <ATen/native/xnnpack/OpContext.h>
#include <torch/custom_class.h>

namespace at::native::xnnpack {

using internal::linear::createLinearClampPrePackOpContext;
using internal::convolution2d::createConv2dClampPrePackOpContext;
using internal::convolution2d::createConv2dTransposeClampPrePackOpContext;

TORCH_LIBRARY(xnnpack, m) {
  m.class_<LinearOpContext>(TORCH_SELECTIVE_CLASS("LinearOpContext"))
    .def_pickle(
        [](const c10::intrusive_ptr<LinearOpContext>& op_context)
            -> SerializationTypeLinearPrePack { // __getstate__
          return op_context->unpack();
        },
        [](SerializationTypeLinearPrePack state)
            -> c10::intrusive_ptr<LinearOpContext> { // __setstate__
          return createLinearClampPrePackOpContext(
              std::get<0>(state),
              std::get<1>(state),
              std::get<2>(state),
              std::get<3>(state));
        })
    .def("unpack", &LinearOpContext::unpack);

  m.class_<Conv2dOpContext>(TORCH_SELECTIVE_CLASS("Conv2dOpContext"))
    .def_pickle(
        [](const c10::intrusive_ptr<Conv2dOpContext>& op_context)
            -> SerializationTypeConv2dPrePack { // __getstate__
          return op_context->unpack();
        },
        [](SerializationTypeConv2dPrePack state)
            -> c10::intrusive_ptr<Conv2dOpContext> { // __setstate__
          return createConv2dClampPrePackOpContext(
              std::get<0>(state),
              std::get<1>(state),
              std::get<2>(state),
              std::get<3>(state),
              std::get<4>(state),
              std::get<5>(state),
              std::get<6>(state),
              std::get<7>(state));
        })
    .def("unpack", &Conv2dOpContext::unpack);

  m.class_<TransposeConv2dOpContext>(TORCH_SELECTIVE_CLASS("TransposeConv2dOpContext"))
    .def_pickle(
        [](const c10::intrusive_ptr<TransposeConv2dOpContext>& op_context)
            -> SerializationTypeTransposeConv2dPrePack { // __getstate__
          return op_context->unpack();
        },
        [](SerializationTypeTransposeConv2dPrePack state)
            -> c10::intrusive_ptr<TransposeConv2dOpContext> { // __setstate__
          return createConv2dTransposeClampPrePackOpContext(
              std::get<0>(state),
              std::get<1>(state),
              std::get<2>(state),
              std::get<3>(state),
              std::get<4>(state),
              std::get<5>(state),
              std::get<6>(state),
              std::get<7>(state),
              std::get<8>(state));
        });

}

// Registration using the TORCH_LIBRARY def gives dispatching errors when there is no tensor input
TORCH_LIBRARY(prepacked, m) {
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::unpack_prepacked_sizes_conv2d(Any W_prepack) -> (Any)"), [](const IValue& inp) { return internal::convolution2d::unpack_prepacked_sizes_conv2d(inp);});
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::unpack_prepacked_sizes_linear(Any W_prepack) -> (Any)"), [](const IValue& inp) { return internal::linear::unpack_prepacked_sizes_linear(inp);});
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::linear_clamp_prepack(Tensor W, Tensor? B=None, Scalar? output_min=None, Scalar? output_max=None) -> __torch__.torch.classes.xnnpack.LinearOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::linear_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.LinearOpContext W_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::conv2d_clamp_prepack(Tensor W, Tensor? B, int[2] stride, int[2] padding, int[2] dilation, int groups, Scalar? output_min=None, Scalar? output_max=None) -> __torch__.torch.classes.xnnpack.Conv2dOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::conv2d_transpose_clamp_prepack(Tensor W, Tensor? B, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation, int groups, Scalar? output_min=None, Scalar? output_max=None) -> __torch__.torch.classes.xnnpack.TransposeConv2dOpContext"));
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::conv2d_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.Conv2dOpContext W_prepack) -> Tensor Y"));
  m.def(TORCH_SELECTIVE_SCHEMA("prepacked::conv2d_transpose_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.TransposeConv2dOpContext W_prepack) -> Tensor Y"));
}

TORCH_LIBRARY_IMPL(prepacked, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("prepacked::linear_clamp_prepack"), TORCH_FN(createLinearClampPrePackOpContext));
  m.impl(TORCH_SELECTIVE_NAME("prepacked::linear_clamp_run"), TORCH_FN(internal::linear::linear_clamp_run));
  m.impl(TORCH_SELECTIVE_NAME("prepacked::conv2d_clamp_prepack"), TORCH_FN(createConv2dClampPrePackOpContext));
  m.impl(TORCH_SELECTIVE_NAME("prepacked::conv2d_transpose_clamp_prepack"), TORCH_FN(createConv2dTransposeClampPrePackOpContext));
  m.impl(TORCH_SELECTIVE_NAME("prepacked::conv2d_clamp_run"), TORCH_FN(internal::convolution2d::conv2d_clamp_run));
  m.impl(TORCH_SELECTIVE_NAME("prepacked::conv2d_transpose_clamp_run"), TORCH_FN(internal::convolution2d::conv2d_transpose_clamp_run));
}

} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/xnnpack`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `torch/library.h`
- `ATen/native/xnnpack/Convolution.h`
- `ATen/native/xnnpack/Linear.h`
- `ATen/native/xnnpack/OpContext.h`
- `torch/custom_class.h`


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

Files in the same folder (`aten/src/ATen/native/xnnpack`):

- [`Engine.h_docs.md`](./Engine.h_docs.md)
- [`Linear.cpp_docs.md`](./Linear.cpp_docs.md)
- [`ChannelShuffle.cpp_docs.md`](./ChannelShuffle.cpp_docs.md)
- [`Convolution.h_docs.md`](./Convolution.h_docs.md)
- [`Common.h_docs.md`](./Common.h_docs.md)
- [`Convolution.cpp_docs.md`](./Convolution.cpp_docs.md)
- [`Activation.cpp_docs.md`](./Activation.cpp_docs.md)
- [`Linear.h_docs.md`](./Linear.h_docs.md)
- [`Shim.cpp_docs.md`](./Shim.cpp_docs.md)


## Cross-References

- **File Documentation**: `RegisterOpContextClass.cpp_docs.md`
- **Keyword Index**: `RegisterOpContextClass.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
