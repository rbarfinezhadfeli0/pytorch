# Documentation: `docs/torch/_inductor/codegen/rocm/ck_tile_universal_gemm_template.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_inductor/codegen/rocm/ck_tile_universal_gemm_template.py_kw.md`
- **Size**: 8,104 bytes (7.91 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_inductor/codegen/rocm/ck_tile_universal_gemm_template.py`

## File Information

- **Original File**: [torch/_inductor/codegen/rocm/ck_tile_universal_gemm_template.py](../../../../../torch/_inductor/codegen/rocm/ck_tile_universal_gemm_template.py)
- **Documentation**: [`ck_tile_universal_gemm_template.py_docs.md`](./ck_tile_universal_gemm_template.py_docs.md)
- **Folder**: `torch/_inductor/codegen/rocm`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`CKTileGemmTemplate`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`class`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`from`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`is`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)

### Functions

- **`__init__`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`add_choices`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`check`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`check_alignments`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`check_block_tile_size`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`check_block_tiles`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`check_dtypes`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`check_layouts`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`check_warp_tiles`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`dict_items`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`dtype_repr`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`emit_ck_instance`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`filter_op`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`gen_ops`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`get_gemm_problem_size`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`get_runtime_arg_info`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`get_runtime_arg_values`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`globals`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`header`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`is_static_int`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`k_batch_choices`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`layout_repr`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`max_alignment`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`name`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`ops`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`render`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`render_dispatch`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`render_epilogue`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`render_pipeline`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`render_scheduler`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`size_args`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`tile_sizes`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`torch_layout_to_ck_layout`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)

### Imports

- **`...utils`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`Any`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`ArgInfo`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`Buffer`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`CKTileTemplate`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`IndentedBuffer`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`OrderedSet`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`ROCmTemplateKernel`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`asdict`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`config`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`dataclasses`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`functools`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`itertools`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`logging`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`random`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`sympy`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`torch`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`torch._inductor`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`torch._inductor.codegen.rocm.ck_tile_template`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`torch._inductor.codegen.rocm.rocm_kernel`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`torch._inductor.codegen.rocm.rocm_template`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`torch._inductor.ir`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`torch.utils._ordered_set`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)
- **`typing`**: [ck_tile_universal_gemm_template.py_docs.md](./ck_tile_universal_gemm_template.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/_inductor/codegen/rocm`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/codegen/rocm`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/_inductor/codegen/rocm`):

- [`ck_template.py_docs.md_docs.md`](./ck_template.py_docs.md_docs.md)
- [`rocm_template.py_kw.md_docs.md`](./rocm_template.py_kw.md_docs.md)
- [`ck_tile_universal_gemm_template.py_docs.md_docs.md`](./ck_tile_universal_gemm_template.py_docs.md_docs.md)
- [`ck_tile_template.py_kw.md_docs.md`](./ck_tile_template.py_kw.md_docs.md)
- [`rocm_template_buffer.py_kw.md_docs.md`](./rocm_template_buffer.py_kw.md_docs.md)
- [`rocm_utils.py_kw.md_docs.md`](./rocm_utils.py_kw.md_docs.md)
- [`ck_universal_gemm_template.py_docs.md_docs.md`](./ck_universal_gemm_template.py_docs.md_docs.md)
- [`ck_conv_template.py_docs.md_docs.md`](./ck_conv_template.py_docs.md_docs.md)
- [`rocm_template.py_docs.md_docs.md`](./rocm_template.py_docs.md_docs.md)
- [`rocm_kernel.py_docs.md_docs.md`](./rocm_kernel.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `ck_tile_universal_gemm_template.py_kw.md_docs.md`
- **Keyword Index**: `ck_tile_universal_gemm_template.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
