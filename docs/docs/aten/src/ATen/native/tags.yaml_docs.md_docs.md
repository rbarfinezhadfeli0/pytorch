# Documentation: `docs/aten/src/ATen/native/tags.yaml_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/tags.yaml_docs.md`
- **Size**: 8,059 bytes (7.87 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/tags.yaml`

## File Metadata

- **Path**: `aten/src/ATen/native/tags.yaml`
- **Size**: 5,508 bytes (5.38 KB)
- **Type**: YAML Configuration
- **Extension**: `.yaml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
# This yaml file contains all the possible tags that can be defined in `tags` in `native_functions.yaml`

- tag: inplace_view
  desc: |
          This tag indicates if an operator *only* modifies the tensor metadata
- tag: pt2_compliant_tag
  desc: |
          This tag indicates if the operator is guaranteed to
          work with the PT2 compilation APIs (torch.compile,
          torch.export, etc). If you add this tag to an
          operator, please use
          `torch.testing._internal.optest.opcheck` to test that
          the operator has been registered correctly and
          works with torch.compile
- tag: view_copy
  desc: |
          This tag indicates operators that are *_copy* variants
          of view/aliasing operators. If an operator has a view_copy tag,
          then it should have the name {op}_copy, where {op} is a view operator.
- tag: dynamic_output_shape
  desc: |
          This tag indicates if an operator's output's shape depends on input Tensor
          data.
- tag: data_dependent_output
  desc: |
          Operator has a non-Tensor output whose value is dependent on the data
          of Tensor inputs.  Among other things, this implies that this operator
          cannot be run with meta tensor (since data is not available), nor
          can it be symbolically traced.
- tag: generated
  desc: |
          This tag indicates that the operator doesn't have an explicit entry in
          native_functions.yaml, and instead was generated automatically by the codegen.
- tag: nondeterministic_seeded
  desc: |
          This tag indicates if an operator is nondeterministically seeded
          (i.e., is random) such that the operator intentionally produces
          different results when run twice on the same inputs, but this randomness
          is controlled by a Generator which, if reseeded would give you the
          same result.
- tag: nondeterministic_bitwise
  desc: |
          This tag indicates if an operator doesn't guarantee bitwise equivalence
          across different runs of an operator with identical inputs.
- tag: needs_exact_strides
  desc: |
          This tag indicates that the operator should be passed Tensors following
          the same strides as observed in eager when compiled in inductor.
          Only one of {needs_exact_strides, needs_contiguous_strides, needs_fixed_stride_order, flexible_layout}
          can apply; if multiple are assigned then we assume the most restrictive one.
- tag: needs_contiguous_strides
  desc: |
          This tag indicates that the operator should be passed contiguous Tensors.
          Failure to do so will result in undefined behavior.
- tag: needs_fixed_stride_order
  desc: |
          This tag indicates that the operator should be passed Tensors following
          the same stride permutation as observed in eager when compiled in inductor.
          Only one of {needs_exact_strides, needs_contiguous_strides, needs_fixed_stride_order, flexible_layout}
          can apply; if multiple are assigned then we assume the most restrictive one.
- tag: flexible_layout
  desc: |
          This tag indicates that the custom operator can accept inputs with varying
          strides/storage_offset and that when compiled, Inductor is allowed to change
          the strides/storage_offset of inputs to the custom operator.
          Only one of {needs_exact_strides, needs_contiguous_strides, needs_fixed_stride_order, flexible_layout}
          can apply; if multiple are assigned then we assume the most restrictive one.

# NOTE [Core ATen Ops]
- tag: core
  desc: |
          Core aten ops is a subset of aten ops that remains after aten-to-aten decomposition and
          functionalization pass. Core aten ops are fully functional and adhere to single static
          assignment (SSA): this implies there will be no `inplace` or `_out` variants in this opset.
          This opset is designed to serve as the functional IR to interface with compiler backends.
          In contrast to primTorch, core aten opset doesn't decompose ops into explicit
          type promotion and broadcasting ops.
          Core aten ops is also effectively the opset produced by torchdynamo.export(aten_graph=True),
          and thus can be used as an opset for export purpose.
- tag: pointwise
  desc: |
          Pointwise operators are operators where each element of the output is computed only by accessing
          the corresponding element of all the broadcasted inputs. The output shape will be the broadcasted
          shape of the inputs.
- tag: maybe_aliasing_or_mutating
  desc: |
          For some ops, we can't statically determine whether the op is functional or not. Note that this is only
          relevant to CIA ops that decompose before functionalization/autograd. It is useful to
          know this information for export as we would want to decompose these ops as they are unsafe to be
          preserved.
- tag: cudagraph_unsafe
  desc: |
          This operator does not support cudagraphs. The presence of this tag on an operator will cause
          Inductor to split the graph around this operator. Note that operators without this tag may still
          not support CUDAGraphs. Inductor may have other hardcoded lists around that.
- tag: reduction
  desc: |
          This tag indicates that an operator performs a reduction operation, computing aggregate values
          (sum, mean, max, min, etc.) across one or more dimensions of the input tensor(s).

```



## High-Level Overview

This file is part of the PyTorch framework located at `aten/src/ATen/native`.

## Detailed Analysis

### Code Structure

This is a configuration file. See the original source for structure.


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

- **File Documentation**: `tags.yaml_docs.md`
- **Keyword Index**: `tags.yaml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native`):

- [`AdaptiveMaxPooling2d.cpp_docs.md_docs.md`](./AdaptiveMaxPooling2d.cpp_docs.md_docs.md)
- [`Distributions.cpp_docs.md_docs.md`](./Distributions.cpp_docs.md_docs.md)
- [`im2col_shape_check.h_docs.md_docs.md`](./im2col_shape_check.h_docs.md_docs.md)
- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`Lerp.cpp_kw.md_docs.md`](./Lerp.cpp_kw.md_docs.md)
- [`CPUFallback.h_docs.md_docs.md`](./CPUFallback.h_docs.md_docs.md)
- [`MetaTensor.cpp_docs.md_docs.md`](./MetaTensor.cpp_docs.md_docs.md)
- [`Correlation.cpp_kw.md_docs.md`](./Correlation.cpp_kw.md_docs.md)
- [`im2col_shape_check.h_kw.md_docs.md`](./im2col_shape_check.h_kw.md_docs.md)
- [`UpSampleNearest2d.cpp_kw.md_docs.md`](./UpSampleNearest2d.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `tags.yaml_docs.md_docs.md`
- **Keyword Index**: `tags.yaml_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
