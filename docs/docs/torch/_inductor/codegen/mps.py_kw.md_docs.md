# Documentation: `docs/torch/_inductor/codegen/mps.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_inductor/codegen/mps.py_kw.md`
- **Size**: 7,547 bytes (7.37 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_inductor/codegen/mps.py`

## File Information

- **Original File**: [torch/_inductor/codegen/mps.py](../../../../torch/_inductor/codegen/mps.py)
- **Documentation**: [`mps.py_docs.md`](./mps.py_docs.md)
- **Folder**: `torch/_inductor/codegen`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`MetalExprPrinter`**: [mps.py_docs.md](./mps.py_docs.md)
- **`MetalKernel`**: [mps.py_docs.md](./mps.py_docs.md)
- **`MetalOverrides`**: [mps.py_docs.md](./mps.py_docs.md)
- **`MetalScheduling`**: [mps.py_docs.md](./mps.py_docs.md)
- **`emits`**: [mps.py_docs.md](./mps.py_docs.md)

### Functions

- **`__init__`**: [mps.py_docs.md](./mps.py_docs.md)
- **`_initialize_special_ops`**: [mps.py_docs.md](./mps.py_docs.md)
- **`_new_idxvar`**: [mps.py_docs.md](./mps.py_docs.md)
- **`_print_Abs`**: [mps.py_docs.md](./mps.py_docs.md)
- **`_print_Float`**: [mps.py_docs.md](./mps.py_docs.md)
- **`_print_FloorDiv`**: [mps.py_docs.md](./mps.py_docs.md)
- **`_print_FloorToInt`**: [mps.py_docs.md](./mps.py_docs.md)
- **`_print_IntTrueDiv`**: [mps.py_docs.md](./mps.py_docs.md)
- **`_print_Max`**: [mps.py_docs.md](./mps.py_docs.md)
- **`_print_Min`**: [mps.py_docs.md](./mps.py_docs.md)
- **`_print_ModularIndexing`**: [mps.py_docs.md](./mps.py_docs.md)
- **`_print_OpaqueUnaryFn_log2`**: [mps.py_docs.md](./mps.py_docs.md)
- **`_print_PowByNatural`**: [mps.py_docs.md](./mps.py_docs.md)
- **`_print_RoundDecimal`**: [mps.py_docs.md](./mps.py_docs.md)
- **`_print_RoundToInt`**: [mps.py_docs.md](./mps.py_docs.md)
- **`_print_ToFloat`**: [mps.py_docs.md](./mps.py_docs.md)
- **`_print_TruncToInt`**: [mps.py_docs.md](./mps.py_docs.md)
- **`_reduction_nocache`**: [mps.py_docs.md](./mps.py_docs.md)
- **`_special_binary`**: [mps.py_docs.md](./mps.py_docs.md)
- **`_special_unary`**: [mps.py_docs.md](./mps.py_docs.md)
- **`_unwrap_helper`**: [mps.py_docs.md](./mps.py_docs.md)
- **`abs`**: [mps.py_docs.md](./mps.py_docs.md)
- **`acos`**: [mps.py_docs.md](./mps.py_docs.md)
- **`asin`**: [mps.py_docs.md](./mps.py_docs.md)
- **`atan`**: [mps.py_docs.md](./mps.py_docs.md)
- **`atan2`**: [mps.py_docs.md](./mps.py_docs.md)
- **`atanh`**: [mps.py_docs.md](./mps.py_docs.md)
- **`call_kernel`**: [mps.py_docs.md](./mps.py_docs.md)
- **`ceil`**: [mps.py_docs.md](./mps.py_docs.md)
- **`check_bounds`**: [mps.py_docs.md](./mps.py_docs.md)
- **`codegen_body`**: [mps.py_docs.md](./mps.py_docs.md)
- **`codegen_iteration_ranges_entry`**: [mps.py_docs.md](./mps.py_docs.md)
- **`codegen_kernel`**: [mps.py_docs.md](./mps.py_docs.md)
- **`constant`**: [mps.py_docs.md](./mps.py_docs.md)
- **`cos`**: [mps.py_docs.md](./mps.py_docs.md)
- **`define_kernel`**: [mps.py_docs.md](./mps.py_docs.md)
- **`dtype_to_str`**: [mps.py_docs.md](./mps.py_docs.md)
- **`exp`**: [mps.py_docs.md](./mps.py_docs.md)
- **`floor`**: [mps.py_docs.md](./mps.py_docs.md)
- **`floordiv`**: [mps.py_docs.md](./mps.py_docs.md)
- **`fmod`**: [mps.py_docs.md](./mps.py_docs.md)
- **`format_threads`**: [mps.py_docs.md](./mps.py_docs.md)
- **`index_expr`**: [mps.py_docs.md](./mps.py_docs.md)
- **`isinf`**: [mps.py_docs.md](./mps.py_docs.md)
- **`isnan`**: [mps.py_docs.md](./mps.py_docs.md)
- **`load`**: [mps.py_docs.md](./mps.py_docs.md)
- **`log`**: [mps.py_docs.md](./mps.py_docs.md)
- **`logical_and`**: [mps.py_docs.md](./mps.py_docs.md)
- **`logical_or`**: [mps.py_docs.md](./mps.py_docs.md)
- **`masked`**: [mps.py_docs.md](./mps.py_docs.md)
- **`maximum`**: [mps.py_docs.md](./mps.py_docs.md)
- **`minimum`**: [mps.py_docs.md](./mps.py_docs.md)
- **`neg`**: [mps.py_docs.md](./mps.py_docs.md)
- **`pow`**: [mps.py_docs.md](./mps.py_docs.md)
- **`rand`**: [mps.py_docs.md](./mps.py_docs.md)
- **`randint64`**: [mps.py_docs.md](./mps.py_docs.md)
- **`randn`**: [mps.py_docs.md](./mps.py_docs.md)
- **`reduction`**: [mps.py_docs.md](./mps.py_docs.md)
- **`remainder`**: [mps.py_docs.md](./mps.py_docs.md)
- **`round`**: [mps.py_docs.md](./mps.py_docs.md)
- **`rsqrt`**: [mps.py_docs.md](./mps.py_docs.md)
- **`sign`**: [mps.py_docs.md](./mps.py_docs.md)
- **`signbit`**: [mps.py_docs.md](./mps.py_docs.md)
- **`sin`**: [mps.py_docs.md](./mps.py_docs.md)
- **`sinc`**: [mps.py_docs.md](./mps.py_docs.md)
- **`sqrt`**: [mps.py_docs.md](./mps.py_docs.md)
- **`store`**: [mps.py_docs.md](./mps.py_docs.md)
- **`store_reduction`**: [mps.py_docs.md](./mps.py_docs.md)
- **`tan`**: [mps.py_docs.md](./mps.py_docs.md)
- **`tanh`**: [mps.py_docs.md](./mps.py_docs.md)
- **`to_dtype`**: [mps.py_docs.md](./mps.py_docs.md)
- **`to_dtype_bitcast`**: [mps.py_docs.md](./mps.py_docs.md)
- **`trunc`**: [mps.py_docs.md](./mps.py_docs.md)
- **`truncdiv`**: [mps.py_docs.md](./mps.py_docs.md)
- **`value_to_metal`**: [mps.py_docs.md](./mps.py_docs.md)
- **`where`**: [mps.py_docs.md](./mps.py_docs.md)

### Imports

- **`..ops_handler`**: [mps.py_docs.md](./mps.py_docs.md)
- **`..scheduler`**: [mps.py_docs.md](./mps.py_docs.md)
- **`..utils`**: [mps.py_docs.md](./mps.py_docs.md)
- **`..virtualized`**: [mps.py_docs.md](./mps.py_docs.md)
- **`.common`**: [mps.py_docs.md](./mps.py_docs.md)
- **`.simd`**: [mps.py_docs.md](./mps.py_docs.md)
- **`Any`**: [mps.py_docs.md](./mps.py_docs.md)
- **`CppPrinter`**: [mps.py_docs.md](./mps.py_docs.md)
- **`IterationRangesEntry`**: [mps.py_docs.md](./mps.py_docs.md)
- **`OpVarT`**: [mps.py_docs.md](./mps.py_docs.md)
- **`OrderedSet`**: [mps.py_docs.md](./mps.py_docs.md)
- **`PRECEDENCE`**: [mps.py_docs.md](./mps.py_docs.md)
- **`Path`**: [mps.py_docs.md](./mps.py_docs.md)
- **`ReductionType`**: [mps.py_docs.md](./mps.py_docs.md)
- **`Scheduler`**: [mps.py_docs.md](./mps.py_docs.md)
- **`Union`**: [mps.py_docs.md](./mps.py_docs.md)
- **`ValueRanges`**: [mps.py_docs.md](./mps.py_docs.md)
- **`__future__`**: [mps.py_docs.md](./mps.py_docs.md)
- **`_embed_headers`**: [mps.py_docs.md](./mps.py_docs.md)
- **`annotations`**: [mps.py_docs.md](./mps.py_docs.md)
- **`ceildiv`**: [mps.py_docs.md](./mps.py_docs.md)
- **`compile_mps_shader`**: [mps.py_docs.md](./mps.py_docs.md)
- **`functools`**: [mps.py_docs.md](./mps.py_docs.md)
- **`itertools`**: [mps.py_docs.md](./mps.py_docs.md)
- **`logging`**: [mps.py_docs.md](./mps.py_docs.md)
- **`math`**: [mps.py_docs.md](./mps.py_docs.md)
- **`ops`**: [mps.py_docs.md](./mps.py_docs.md)
- **`pathlib`**: [mps.py_docs.md](./mps.py_docs.md)
- **`sympy`**: [mps.py_docs.md](./mps.py_docs.md)
- **`sympy.printing.precedence`**: [mps.py_docs.md](./mps.py_docs.md)
- **`torch`**: [mps.py_docs.md](./mps.py_docs.md)
- **`torch._inductor.runtime.runtime_utils`**: [mps.py_docs.md](./mps.py_docs.md)
- **`torch.utils._cpp_embed_headers`**: [mps.py_docs.md](./mps.py_docs.md)
- **`torch.utils._ordered_set`**: [mps.py_docs.md](./mps.py_docs.md)
- **`torch.utils._sympy.printers`**: [mps.py_docs.md](./mps.py_docs.md)
- **`torch.utils._sympy.value_ranges`**: [mps.py_docs.md](./mps.py_docs.md)
- **`typing`**: [mps.py_docs.md](./mps.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/_inductor/codegen`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/codegen`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/_inductor/codegen`):

- [`wrapper_fxir.py_kw.md_docs.md`](./wrapper_fxir.py_kw.md_docs.md)
- [`simd.py_docs.md_docs.md`](./simd.py_docs.md_docs.md)
- [`mps_device_op_overrides.py_docs.md_docs.md`](./mps_device_op_overrides.py_docs.md_docs.md)
- [`simd_kernel_features.py_docs.md_docs.md`](./simd_kernel_features.py_docs.md_docs.md)
- [`segmented_tree.py_docs.md_docs.md`](./segmented_tree.py_docs.md_docs.md)
- [`triton_utils.py_docs.md_docs.md`](./triton_utils.py_docs.md_docs.md)
- [`wrapper.py_kw.md_docs.md`](./wrapper.py_kw.md_docs.md)
- [`cpu_device_op_overrides.py_kw.md_docs.md`](./cpu_device_op_overrides.py_kw.md_docs.md)
- [`cpp_gemm_template.py_kw.md_docs.md`](./cpp_gemm_template.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `mps.py_kw.md_docs.md`
- **Keyword Index**: `mps.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
