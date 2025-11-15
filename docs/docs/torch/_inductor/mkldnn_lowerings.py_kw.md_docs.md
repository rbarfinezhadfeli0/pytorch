# Documentation: `docs/torch/_inductor/mkldnn_lowerings.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_inductor/mkldnn_lowerings.py_kw.md`
- **Size**: 4,939 bytes (4.82 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_inductor/mkldnn_lowerings.py`

## File Information

- **Original File**: [torch/_inductor/mkldnn_lowerings.py](../../../torch/_inductor/mkldnn_lowerings.py)
- **Documentation**: [`mkldnn_lowerings.py_docs.md`](./mkldnn_lowerings.py_docs.md)
- **Folder**: `torch/_inductor`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Functions

- **`codegen_int8_gemm_template_compensation`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`convolution_binary`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`convolution_binary_inplace`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`convolution_transpose_unary`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`convolution_unary`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`create_int8_compensation`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`epilogue_creator`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`grouped_gemm_lowering`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`inner_fn`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`inner_fn_cast_output_to_bf16`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`inner_fn_requant`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`linear_binary`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`linear_unary`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`mkl_packed_linear`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`mkldnn_rnn_layer`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`qconvolution_binary`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`qconvolution_unary`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`qlinear_binary`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`qlinear_unary`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`register_onednn_fusion_ops`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)

### Imports

- **`.`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`.codegen.cpp_gemm_template`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`.codegen.cpp_grouped_gemm_template`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`.codegen.cpp_utils`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`.ir`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`.lowering`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`.select_algorithm`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`.utils`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`.virtualized`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`CppGemmTemplate`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`CppGroupedGemmTemplate`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`Optional`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`TensorBox`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`_create_constants`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`config`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`create_epilogue_with_attr`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`functools`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`mkldnn_ir`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`mm_args`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`ops`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`torch`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`torch._inductor.kernel.mm_common`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`torch.utils._pytree`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`typing`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)
- **`use_aten_gemm_kernels`**: [mkldnn_lowerings.py_docs.md](./mkldnn_lowerings.py_docs.md)


## Keyword → Section Map

The following sections in the documentation cover these topics:

- **File Metadata**: Basic file information
- **Original Source**: Complete source code
- **High-Level Overview**: Purpose and role
- **Detailed Analysis**: In-depth code analysis
- **Architecture & Design**: Design patterns and structure
- **Dependencies**: Related modules and imports
- **Performance Considerations**: Efficiency and optimization
- **Security & Safety**: Security analysis
- **Testing & Usage**: How to use and test

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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

Files in the same folder (`docs/torch/_inductor`):

- [`freezing.py_docs.md_docs.md`](./freezing.py_docs.md_docs.md)
- [`lowering.py_kw.md_docs.md`](./lowering.py_kw.md_docs.md)
- [`quantized_lowerings.py_docs.md_docs.md`](./quantized_lowerings.py_docs.md_docs.md)
- [`select_algorithm.py_docs.md_docs.md`](./select_algorithm.py_docs.md_docs.md)
- [`kernel_inputs.py_kw.md_docs.md`](./kernel_inputs.py_kw.md_docs.md)
- [`compile_fx_ext.py_kw.md_docs.md`](./compile_fx_ext.py_kw.md_docs.md)
- [`extern_node_serializer.py_docs.md_docs.md`](./extern_node_serializer.py_docs.md_docs.md)
- [`ops_handler.py_docs.md_docs.md`](./ops_handler.py_docs.md_docs.md)
- [`test_operators.py_docs.md_docs.md`](./test_operators.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `mkldnn_lowerings.py_kw.md_docs.md`
- **Keyword Index**: `mkldnn_lowerings.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
