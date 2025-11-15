# Documentation: `aten/src/ATen/native/nested/NestedTensorTransformerFunctions.h`

## File Metadata

- **Path**: `aten/src/ATen/native/nested/NestedTensorTransformerFunctions.h`
- **Size**: 2,832 bytes (2.77 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
/**
 * Transformer-specific NestedTensor utility functions.
 *
 * Not co-located with NestedTensor core code yet because they only
 * support specific cases needed in transformers.
 */
#pragma once

#include <vector>

#include <c10/macros/Macros.h>
#include <optional>

namespace c10 {
class Scalar;
} // namespace c10

namespace at {
class Tensor;
namespace native {
struct NestedTensorImpl;

// Requires that self is a contiguous NestedTensor, other is not a
// NestedTensor, self.dim() == 3, and other.dim() == 2. Also, self
// must have a consistent last dimension across its included Tensors
// and that dimension must match other.size(0).
Tensor NestedTensor_matmul(const Tensor& self, const Tensor& other);

// Requires that mat1 is a contiguous NestedTensor, self & mat2 are
// not NestedTensors, mat1.dim() == 3, mat2.dim() == 2, and that mat1
// has a consistent last dimension across its included Tensors that
// matches mat2.size(0).
Tensor NestedTensor_times_Tensor_plus_Tensor_addmm(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const c10::Scalar& beta,
    const c10::Scalar& alpha,
    std::optional<bool> use_gelu = std::nullopt);

Tensor NestedTensor_add_NestedTensor_in_place(
    const Tensor& self,
    const Tensor& other);

TORCH_API Tensor NestedTensor_batch_offsets_from_size_tensor(
    const Tensor& sizes,
    int64_t extra_elements);

Tensor NestedTensor_from_padded_tensor_cpu(
    const Tensor& padded,
    const NestedTensorImpl& nt);

TORCH_API Tensor NestedTensor_to_mask(const Tensor& nt, std::optional<int64_t> mask_dim, std::optional<int64_t> mask_dim_length);

template <typename T>
void remove_padding_kernelLauncher(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int64_t output_dim,
    const int64_t batch_size);

template <typename T>
void remove_padding_transform0213_kernelLauncher(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int64_t output_dim,
    const int64_t batch_size);

template <typename T>
void add_padding_kernelLauncher(
    T* input,
    T* output,
    T padding_value,
    const int* offsets,
    const int* input_sizes,
    int input_dim,
    const std::vector<int64_t>& output_sizes,
    const int batch_size,
    const int output_batch_size);

TORCH_API Tensor flash_attention_helper(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool need_attn_weights,
    bool is_causal);

TORCH_API std::tuple<Tensor, Tensor> mem_efficient_helper_nested_unpacked(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool need_attn_weights,
    bool is_causal);
} // namespace native
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`, `native`, `c10`

**Classes/Structs**: `Scalar`, `Tensor`, `NestedTensorImpl`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/nested`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `vector`
- `c10/macros/Macros.h`
- `optional`


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

Files in the same folder (`aten/src/ATen/native/nested`):

- [`NestedTensorBinaryOps.cpp_docs.md`](./NestedTensorBinaryOps.cpp_docs.md)
- [`NestedTensorUtils.cpp_docs.md`](./NestedTensorUtils.cpp_docs.md)
- [`NestedTensorUnaryOps.cpp_docs.md`](./NestedTensorUnaryOps.cpp_docs.md)
- [`NestedTensorBinaryOps.h_docs.md`](./NestedTensorBinaryOps.h_docs.md)
- [`NestedTensorFactories.cpp_docs.md`](./NestedTensorFactories.cpp_docs.md)
- [`NestedTensorBackward.cpp_docs.md`](./NestedTensorBackward.cpp_docs.md)
- [`NestedTensorMatmul.cpp_docs.md`](./NestedTensorMatmul.cpp_docs.md)
- [`NestedTensorMath.cpp_docs.md`](./NestedTensorMath.cpp_docs.md)
- [`NestedTensorTransformerUtils.h_docs.md`](./NestedTensorTransformerUtils.h_docs.md)


## Cross-References

- **File Documentation**: `NestedTensorTransformerFunctions.h_docs.md`
- **Keyword Index**: `NestedTensorTransformerFunctions.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
