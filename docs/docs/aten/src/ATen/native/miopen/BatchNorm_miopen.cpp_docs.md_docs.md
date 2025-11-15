# Documentation: `docs/aten/src/ATen/native/miopen/BatchNorm_miopen.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/miopen/BatchNorm_miopen.cpp_docs.md`
- **Size**: 11,743 bytes (11.47 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/miopen/BatchNorm_miopen.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/miopen/BatchNorm_miopen.cpp`
- **Size**: 9,329 bytes (9.11 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Config.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/miopen_batch_norm_native.h>
#include <ATen/ops/miopen_batch_norm_backward_native.h>
#endif

// TODO: Remove the condition on AT_ROCM_ENABLED entirely,
// don't build this file as part of CPU build.
#include <ATen/cuda/CUDAConfig.h>

#if !AT_ROCM_ENABLED()

namespace at::native {

// See Note [ATen preprocessor philosophy]

std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm(
    const Tensor& input, const Tensor& weight, const std::optional<Tensor>& bias_opt, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
    bool training, double exponential_average_factor, double epsilon) {
  TORCH_CHECK(false, "miopen_batch_norm: ATen not compiled with MIOpen support");
}

std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm_backward(
    const Tensor& input, const Tensor& grad_output, const Tensor& weight, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt, const std::optional<Tensor>& save_mean_opt, const std::optional<Tensor>& save_var_opt,
    double epsilon) {
  TORCH_CHECK(false, "miopen_batch_norm_backward: ATen not compiled with MIOpen support");
}

}  // namespace at::native

#else // AT_ROCM_ENABLED

#include <ATen/miopen/Descriptors.h>
#include <ATen/miopen/Types.h>
#include <ATen/miopen/Utils.h>

#include <ATen/TensorUtils.h>

namespace at { namespace native {

namespace {

Tensor expandScale(const Tensor& t, int64_t dim) {
  std::vector<int64_t> size{ 1, t.numel() };
  while (static_cast<int64_t>(size.size()) < dim) {
    size.emplace_back(1);
  }
  return t.view(size);
}

}  // namespace

std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm(
    const Tensor& input_t, const Tensor& weight_t, const std::optional<Tensor>& bias_t_opt, const std::optional<Tensor>& running_mean_t_opt, const std::optional<Tensor>& running_var_t_opt,
    bool training, double exponential_average_factor, double epsilon)
{
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> bias_t_maybe_owned = at::borrow_from_optional_tensor(bias_t_opt);
  const Tensor& bias_t = *bias_t_maybe_owned;
  const Tensor& running_mean_t = running_mean_t_opt.value_or(Tensor());
  const Tensor& running_var_t = running_var_t_opt.value_or(Tensor());

  TensorArg input{ input_t, "input", 1 },
            weight{ weight_t, "weight", 2 },
            bias{ bias_t, "bias", 3 },
            running_mean{ running_mean_t, "running_mean", 4 },
            running_var{ running_var_t, "running_var", 5 };
  CheckedFrom c = "miopen_batch_norm";

  checkAllDefined(c, {input, weight, bias});
  if (!training) {
    checkAllDefined(c, {running_mean, running_var});
  }
  checkAllSameGPU(c, {input, weight, bias, running_mean, running_var});
  if (input->scalar_type() == ScalarType::Half || input->scalar_type() == ScalarType::BFloat16) {
    checkScalarType(c, weight, ScalarType::Float);
  } else {
    checkAllSameType(c, {input, weight});
  }
  checkAllSameType(c, {weight, bias, running_mean, running_var});
  checkAllContiguous(c, {weight, bias, running_mean, running_var});
  TORCH_CHECK(input->is_contiguous(input->suggest_memory_format()));
  checkDimRange(c, input, 2, 6 /* exclusive */);
  auto num_features = input->size(1);
  for (auto t : {weight, bias, running_mean, running_var}) {
    if (t->defined()) {
      checkNumel(c, t, num_features);
    }
  }

  miopenBatchNormMode_t mode;
  if (input->dim() == 2) {
    mode = miopenBNPerActivation;
  } else {
    mode = miopenBNSpatial;
  }

  auto output_t = at::empty_like(input_t, input_t.options(), input_t.suggest_memory_format());
  TensorArg output{ output_t, "output", 0 };

  auto handle = getMiopenHandle();
  auto dataType = getMiopenDataType(*input);
  TensorDescriptor idesc{ *input, 4 };  // input descriptor
  TensorDescriptor wdesc{ expandScale(*weight, input->dim()), 4 };  // descriptor for weight, bias, running_mean, etc.

  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  Tensor save_mean, save_var;

  if (training) {
    int64_t num_features = input_t.size(1);
    save_mean = at::empty({ num_features }, weight_t.options());
    save_var = at::empty({ num_features }, weight_t.options());
    MIOPEN_CHECK(miopenBatchNormalizationForwardTraining(
      handle, mode, &one, &zero,
      idesc.desc(), input->const_data_ptr(),
      idesc.desc(), output->data_ptr(),
      wdesc.desc(),
      // NOTE: MIOpen docs say that the bnScale and bnBias args are only inputs,
      // not outputs. However, unfortunately the function signature only takes
      // non-const pointers, presumably by accident
      const_cast<void*>(weight->const_data_ptr()),
      const_cast<void*>(bias->const_data_ptr()),
      exponential_average_factor,
      at::maybe_data_ptr(running_mean),
      at::maybe_data_ptr(running_var),
      epsilon,
      save_mean.mutable_data_ptr(),
      save_var.mutable_data_ptr()));
  } else {
    save_mean = at::empty({0}, weight_t.options());
    save_var = at::empty({0}, weight_t.options());
    MIOPEN_CHECK(miopenBatchNormalizationForwardInference(
      handle, mode, &one, &zero,
      idesc.desc(), input->const_data_ptr(),
      idesc.desc(), output->data_ptr(),
      wdesc.desc(),
      // NOTE: MIOpen docs say that the bnScale and bnBias args are only inputs,
      // not outputs. However, unfortunately the function signature only takes
      // non-const pointers, presumably by accident
      const_cast<void*>(weight->const_data_ptr()),
      const_cast<void*>(bias->const_data_ptr()),
      running_mean->data_ptr(),
      running_var->data_ptr(),
      epsilon));
  }

  // save_mean and save_var can be undefined
  // If this causes problems, we can initialize them to empty tensors
  // of the correct type
  return std::tuple<Tensor, Tensor, Tensor>{output_t, save_mean, save_var};
}

std::tuple<Tensor, Tensor, Tensor> miopen_batch_norm_backward(
    const Tensor& input_t,
    const Tensor& grad_output_t,
    const Tensor& weight_t,
    // Unused: but we require them to be passed so that double backwards
    // has access
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    const std::optional<Tensor>& save_mean_t_opt,
    const std::optional<Tensor>& save_var_t_opt,
    double epsilon) {
  // See [Note: hacky wrapper removal for optional tensor]
  const Tensor& save_mean_t = save_mean_t_opt.value_or(Tensor());
  const Tensor& save_var_t = save_var_t_opt.value_or(Tensor());

  auto grad_output_contig =
      grad_output_t.contiguous(input_t.suggest_memory_format());
  TensorArg input{input_t, "input", 1},
      grad_output{grad_output_contig, "grad_output", 2},
      weight{weight_t, "weight", 3}, save_mean{save_mean_t, "save_mean", 4},
      save_var{save_var_t, "save_var", 5};
  CheckedFrom c = "miopen_batch_norm_backward";

  checkAllDefined(c, {input, grad_output, weight, save_mean, save_var});
  checkAllSameGPU(c, {input, grad_output, weight, save_mean, save_var});
  if (input->scalar_type() == ScalarType::Half || input->scalar_type() == ScalarType::BFloat16) {
    checkScalarType(c, weight, ScalarType::Float);
  } else {
    checkAllSameType(c, {input, weight});
  }
  checkAllSameType(c, {input, grad_output});
  checkAllSameType(c, {weight, save_mean, save_var});
  // TODO: is weight required to be contiguous?
  checkAllContiguous(c, {save_mean, save_var});
  // TODO: TensorArg check should start handle memory format
  TORCH_CHECK(input->is_contiguous(input->suggest_memory_format()));
  TORCH_CHECK(grad_output->is_contiguous(input->suggest_memory_format()));
  checkDimRange(c, input, 2, 6 /* exclusive */);
  checkSameSize(c, input, grad_output);
  auto num_features = input->size(1);
  for (auto t : {weight, save_mean, save_var}) {
    checkNumel(c, t, num_features);
  }

  miopenBatchNormMode_t mode;
  if (input->dim() == 2) {
    mode = miopenBNPerActivation;
  } else {
    mode = miopenBNSpatial;
  }

  auto grad_input_t  = at::empty(input->sizes(), input->options(), input->suggest_memory_format());
  auto grad_weight_t = at::empty(weight->sizes(), weight->options());
  auto grad_bias_t   = at::empty(weight->sizes(), weight->options());

  auto handle = getMiopenHandle();
  auto dataType = getMiopenDataType(*input);

  TensorDescriptor idesc{ *input, 4 };  // input, output, grad_output descriptor
  TensorDescriptor wdesc{ expandScale(*weight, input->dim()), 4 };  // descriptor for weight, bias, save_mean, etc.

  Constant one(dataType, 1);
  Constant zero(dataType, 0);

  MIOPEN_CHECK(miopenBatchNormalizationBackward(
    handle, mode, &one, &zero, &one, &zero,
    idesc.desc(), input->const_data_ptr(),
    idesc.desc(), grad_output->const_data_ptr(),
    idesc.desc(), grad_input_t.data_ptr(),
    wdesc.desc(), weight->const_data_ptr(),
    grad_weight_t.data_ptr(),
    grad_bias_t.data_ptr(),
    epsilon,
    save_mean->const_data_ptr(),
    save_var->const_data_ptr()));

  return std::tuple<Tensor,Tensor,Tensor>{grad_input_t, grad_weight_t, grad_bias_t};
}

}}  // namespace native

#endif

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 6 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `native`, `std`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/miopen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Config.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/empty.h`
- `ATen/ops/empty_like.h`
- `ATen/ops/miopen_batch_norm_native.h`
- `ATen/ops/miopen_batch_norm_backward_native.h`
- `ATen/cuda/CUDAConfig.h`
- `ATen/miopen/Descriptors.h`
- `ATen/miopen/Types.h`
- `ATen/miopen/Utils.h`
- `ATen/TensorUtils.h`


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

Files in the same folder (`aten/src/ATen/native/miopen`):

- [`Conv_miopen.cpp_docs.md`](./Conv_miopen.cpp_docs.md)
- [`RNN_miopen.cpp_docs.md`](./RNN_miopen.cpp_docs.md)


## Cross-References

- **File Documentation**: `BatchNorm_miopen.cpp_docs.md`
- **Keyword Index**: `BatchNorm_miopen.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/miopen`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/miopen`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/aten/src/ATen/native/miopen`):

- [`Conv_miopen.cpp_kw.md_docs.md`](./Conv_miopen.cpp_kw.md_docs.md)
- [`RNN_miopen.cpp_kw.md_docs.md`](./RNN_miopen.cpp_kw.md_docs.md)
- [`Conv_miopen.cpp_docs.md_docs.md`](./Conv_miopen.cpp_docs.md_docs.md)
- [`BatchNorm_miopen.cpp_kw.md_docs.md`](./BatchNorm_miopen.cpp_kw.md_docs.md)
- [`RNN_miopen.cpp_docs.md_docs.md`](./RNN_miopen.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `BatchNorm_miopen.cpp_docs.md_docs.md`
- **Keyword Index**: `BatchNorm_miopen.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
