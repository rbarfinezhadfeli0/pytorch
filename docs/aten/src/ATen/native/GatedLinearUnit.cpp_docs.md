# Documentation: `aten/src/ATen/native/GatedLinearUnit.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/GatedLinearUnit.cpp`
- **Size**: 5,699 bytes (5.57 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/native/Activation.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/glu_backward_native.h>
#include <ATen/ops/glu_backward_jvp_native.h>
#include <ATen/ops/glu_jvp_native.h>
#include <ATen/ops/glu_native.h>
#include <ATen/ops/sigmoid.h>
#endif

namespace at::meta {

TORCH_META_FUNC(glu) (
    const Tensor& self, int64_t dim
) {
  // this can't pass anyway because a 0-dimensional tensor has "size" 1, which
  // can't be evenly halved, but give a nicer error message here.
  TORCH_CHECK(self.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  const int64_t nIn = self.size(wrap_dim);
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
              wrap_dim, " is size ", nIn);

  // size output to half of input
  const int64_t selfSize = nIn / 2;
  Tensor firstHalf = self.narrow(wrap_dim, 0, selfSize);
  Tensor secondHalf = self.narrow(wrap_dim, selfSize, selfSize);
  build_borrowing_binary_op(maybe_get_output(), firstHalf, secondHalf);
}
} // namespace at::meta

namespace at::native {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(glu_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(glu_backward_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(glu_jvp_stub);

TORCH_IMPL_FUNC(glu_out) (const Tensor& self, int64_t dim, const Tensor& out) {
  glu_stub(device_type(), *this);
}

Tensor& glu_backward_cpu_out(const Tensor& grad_output, const Tensor& input,
                             int64_t dim, Tensor& grad_input) {
  TORCH_CHECK(input.dim() > 0, "glu does not support 0-dimensional tensors");
  auto wrap_dim = maybe_wrap_dim(dim, input.dim());
  const int64_t nIn = input.size(wrap_dim);
  TORCH_CHECK(nIn % 2 == 0, "Halving dimension must be even, but dimension ",
              wrap_dim, " is size ", nIn);

  grad_input.resize_as_(input);
  const int64_t inputSize = nIn / 2;
  // half tensor
  Tensor firstHalf = input.narrow(wrap_dim, 0, inputSize);
  Tensor secondHalf = input.narrow(wrap_dim, inputSize, inputSize);
  Tensor gradInputfirstHalf = grad_input.narrow(wrap_dim, 0, inputSize);
  Tensor gradInputsecondHalf = grad_input.narrow(wrap_dim, inputSize, inputSize);

  at::sigmoid_out(gradInputfirstHalf, secondHalf);
  // for second gradinput half, can get a better performance by fusion
  auto iter = at::TensorIteratorConfig()
    .add_output(gradInputsecondHalf)
    .add_const_input(gradInputfirstHalf)
    .add_const_input(firstHalf)
    .add_const_input(grad_output)
    .build();
  glu_backward_stub(iter.device_type(), iter);
  gradInputfirstHalf.mul_(grad_output);
  return grad_input;
}

Tensor glu_backward_cpu(const Tensor& grad_output, const Tensor& input, int64_t dim) {
  auto grad_input = at::empty({0}, input.options());
  return glu_backward_cpu_out(grad_output, input, dim, grad_input);
}

Tensor glu_jvp(
    const Tensor& glu,
    const Tensor& x,
    const Tensor& dx,
    int64_t dim
) {
  dim = maybe_wrap_dim(dim, x.dim());
  const auto glu_size = glu.size(dim);
  const auto b = x.narrow(dim, glu_size, glu_size);
  const auto da = dx.narrow(dim, 0, glu_size);
  const auto db = dx.narrow(dim, glu_size, glu_size);
  auto dglu = at::empty_like(glu);
  auto iter = at::TensorIteratorConfig()
    .add_output(dglu)
    .add_const_input(glu)
    .add_const_input(b)
    .add_const_input(da)
    .add_const_input(db)
    .build();
  glu_jvp_stub(iter.device_type(), iter);
  return dglu;
}

Tensor glu_backward_jvp(
    const Tensor& grad_x,
    const Tensor& grad_glu,
    const Tensor& x,
    const Tensor& dgrad_glu,
    const Tensor& dx,
    int64_t dim
) {
  dim = maybe_wrap_dim(dim, x.dim());
  const auto glu_size = grad_glu.size(dim);
  const auto a = x.narrow(dim, 0, glu_size);
  const auto b = x.narrow(dim, glu_size, glu_size);
  const auto da = dx.narrow(dim, 0, glu_size);
  const auto db = dx.narrow(dim, glu_size, glu_size);
  // grad_x_a = grad_glu * sigmoid(b)
  const auto grad_x_a = grad_x.narrow(dim, 0, glu_size);
  // grad_x_b = grad_x_a * a * (1 - sigmoid(b))
  const auto grad_x_b = grad_x.narrow(dim, glu_size, glu_size);

  const auto sig_b = at::sigmoid(b);
  // TODO: use glu from forward.
  // TODO: fuse kernels.
  const auto glu = a * sig_b;
  const auto db_neg_sig_b = db - db * sig_b;

  // dgrad_x_a = d(grad_glu * sigmoid(b))
  //           = dgrad_glu * sigmoid(b) + grad_glu * sigmoid(b) * (1 - sigmoid(b)) * db
  //           = dgrad_glu * sig_b + grad_x_a * (db - db * sig_b)
  //           = dgrad_glu * sig_b + grad_x_a * db_neg_sig_b
  const auto dgrad_x_a = dgrad_glu * sig_b + grad_x_a * db_neg_sig_b;

  // dgrad_x_b = d(grad_glu * sigmoid(b) * a * (1 - sigmoid(b))
  //           =  d(grad_glu * sigmoid(b)) * a * (1 - sigmoid(b))
  //            + grad_glu * sigmoid(b) * da * (1 - sigmoid(b))
  //            - grad_glu * sigmoid(b) * a * sigmoid(b) * (1 - sigmoid(b)) * db
  //          =   dgrad_x_a * a * (1 - sigmoid(b))
  //           + (grad_glu * sigmoid(b)) * (da * (1 - sigmoid(b)) - a * sigmoid(b) * (1 - sigmoid(b)) * db)
  //          = dgrad_x_a * (a - glu) + grad_x_a * (da - da * sig_b - glu * db_neg_sig_b
  const auto dgrad_x_b = dgrad_x_a * (a - glu) + grad_x_a * (da - da * sig_b - glu * db_neg_sig_b);

  return at::cat({dgrad_x_a, dgrad_x_b}, dim);
}


} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/TensorIterator.h`
- `ATen/TensorOperators.h`
- `ATen/native/Activation.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/cat.h`
- `ATen/ops/empty.h`
- `ATen/ops/glu_backward_native.h`
- `ATen/ops/glu_backward_jvp_native.h`
- `ATen/ops/glu_jvp_native.h`
- `ATen/ops/glu_native.h`
- `ATen/ops/sigmoid.h`


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/native`):

- [`LossMulti.h_docs.md`](./LossMulti.h_docs.md)
- [`NaiveConvolutionTranspose3d.cpp_docs.md`](./NaiveConvolutionTranspose3d.cpp_docs.md)
- [`UnaryOps.cpp_docs.md`](./UnaryOps.cpp_docs.md)
- [`ResizeCommon.h_docs.md`](./ResizeCommon.h_docs.md)
- [`FusedAdagrad.cpp_docs.md`](./FusedAdagrad.cpp_docs.md)
- [`SharedReduceOps.h_docs.md`](./SharedReduceOps.h_docs.md)
- [`SpectralOpsUtils.h_docs.md`](./SpectralOpsUtils.h_docs.md)
- [`FractionalMaxPooling.h_docs.md`](./FractionalMaxPooling.h_docs.md)
- [`TensorDimApply.h_docs.md`](./TensorDimApply.h_docs.md)
- [`Lerp.cpp_docs.md`](./Lerp.cpp_docs.md)


## Cross-References

- **File Documentation**: `GatedLinearUnit.cpp_docs.md`
- **Keyword Index**: `GatedLinearUnit.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
