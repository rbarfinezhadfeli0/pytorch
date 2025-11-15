# Documentation: `docs/aten/src/ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h_docs.md`
- **Size**: 5,554 bytes (5.42 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h`

## File Metadata

- **Path**: `aten/src/ATen/native/ao_sparse/quantized/cpu/fbgemm_utils.h`
- **Size**: 2,980 bytes (2.91 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/Tensor.h>
#include <c10/core/QScheme.h>

#ifdef USE_FBGEMM
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wextra-semi")
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmSparse.h>
#include <ATen/native/ao_sparse/quantized/cpu/packed_params.h>
C10_DIAGNOSTIC_POP()


namespace ao::sparse {

struct TORCH_API PackedLinearWeight
    : public LinearPackedParamsBase {
  PackedLinearWeight(std::unique_ptr<fbgemm::BCSRMatrix<int8_t>> w,
                     std::optional<at::Tensor> bias,
                     std::vector<int32_t> col_offsets,
                     std::vector<float> w_scale,
                     std::vector<int32_t> w_zp,
                     c10::QScheme q_scheme,
                     const int64_t out_features_block_size /* block sparsity size across output_features */,
                     const int64_t in_features_block_size /* block sparsity size across input_features */)
      : LinearPackedParamsBase(
            out_features_block_size,
            in_features_block_size),
        w(std::move(w)),
        bias_(std::move(bias)),
        col_offsets(std::move(col_offsets)),
        w_scale(std::move(w_scale)),
        w_zp(std::move(w_zp)),
        q_scheme(q_scheme) {}
  std::unique_ptr<fbgemm::BCSRMatrix<int8_t>> w;
  std::optional<at::Tensor> bias_;
  std::vector<int32_t> col_offsets;
  std::vector<float> w_scale;
  std::vector<int32_t> w_zp;
  c10::QScheme q_scheme;

  at::Tensor apply(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;
  at::Tensor apply_relu(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point) override;

  at::Tensor apply_dynamic(const at::Tensor& input) override {
    TORCH_INTERNAL_ASSERT(
        false,
        "Sparse quantized dynamic linear with fused relu is not yet "
        "supported on qnnpack backend.");
    return at::Tensor();
  }
  at::Tensor apply_dynamic_relu(const at::Tensor& input) override {
    TORCH_INTERNAL_ASSERT(
        false,
        "Sparse quantized dynamic linear with fused relu is not yet "
        "supported on qnnpack backend.");
    return at::Tensor();
  }

  LinearPackedSerializationType unpack() override;

  BCSRSerializationType serialize() override;

  static c10::intrusive_ptr<LinearPackedParamsBase> deserialize(
      const BCSRSerializationType& serialized);

  std::optional<at::Tensor> bias() override {
    return bias_;
  }

  static c10::intrusive_ptr<LinearPackedParamsBase> prepack(
      const at::Tensor& weight,
      const std::optional<at::Tensor>& bias,
      const int64_t out_features_block_size,
      const int64_t in_features_block_size);

 private:
  template <bool ReluFused>
  at::Tensor apply_impl(
      const at::Tensor& input,
      double output_scale,
      int64_t output_zero_point);
};

} // namespace ao::sparse

#endif // USE_FBGEMM

namespace ao::sparse {
int register_linear_params();
}  // namespace ao::sparse

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `ao`

**Classes/Structs**: `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/ao_sparse/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Tensor.h`
- `c10/core/QScheme.h`
- `fbgemm/Fbgemm.h`
- `fbgemm/FbgemmSparse.h`
- `ATen/native/ao_sparse/quantized/cpu/packed_params.h`


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

Files in the same folder (`aten/src/ATen/native/ao_sparse/quantized/cpu`):

- [`packed_params.h_docs.md`](./packed_params.h_docs.md)
- [`qlinear_serialize.cpp_docs.md`](./qlinear_serialize.cpp_docs.md)
- [`qlinear_dynamic.cpp_docs.md`](./qlinear_dynamic.cpp_docs.md)
- [`qlinear_unpack.cpp_docs.md`](./qlinear_unpack.cpp_docs.md)
- [`fbgemm_utils.cpp_docs.md`](./fbgemm_utils.cpp_docs.md)
- [`qlinear_deserialize.cpp_docs.md`](./qlinear_deserialize.cpp_docs.md)
- [`qlinear_prepack.cpp_docs.md`](./qlinear_prepack.cpp_docs.md)
- [`qnnpack_utils.h_docs.md`](./qnnpack_utils.h_docs.md)
- [`qlinear.cpp_docs.md`](./qlinear.cpp_docs.md)


## Cross-References

- **File Documentation**: `fbgemm_utils.h_docs.md`
- **Keyword Index**: `fbgemm_utils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/ao_sparse/quantized/cpu`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/ao_sparse/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/aten/src/ATen/native/ao_sparse/quantized/cpu`):

- [`qnnpack_utils.h_kw.md_docs.md`](./qnnpack_utils.h_kw.md_docs.md)
- [`fbgemm_utils.cpp_kw.md_docs.md`](./fbgemm_utils.cpp_kw.md_docs.md)
- [`qlinear_dynamic.cpp_kw.md_docs.md`](./qlinear_dynamic.cpp_kw.md_docs.md)
- [`packed_params.h_kw.md_docs.md`](./packed_params.h_kw.md_docs.md)
- [`qlinear_deserialize.cpp_kw.md_docs.md`](./qlinear_deserialize.cpp_kw.md_docs.md)
- [`qlinear_prepack.cpp_docs.md_docs.md`](./qlinear_prepack.cpp_docs.md_docs.md)
- [`qlinear_serialize.cpp_docs.md_docs.md`](./qlinear_serialize.cpp_docs.md_docs.md)
- [`qnnpack_utils.h_docs.md_docs.md`](./qnnpack_utils.h_docs.md_docs.md)
- [`packed_params.h_docs.md_docs.md`](./packed_params.h_docs.md_docs.md)
- [`qlinear_serialize.cpp_kw.md_docs.md`](./qlinear_serialize.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `fbgemm_utils.h_docs.md_docs.md`
- **Keyword Index**: `fbgemm_utils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
