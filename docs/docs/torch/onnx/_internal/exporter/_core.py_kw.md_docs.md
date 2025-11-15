# Documentation: `docs/torch/onnx/_internal/exporter/_core.py_kw.md`

## File Metadata

- **Path**: `docs/torch/onnx/_internal/exporter/_core.py_kw.md`
- **Size**: 5,757 bytes (5.62 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/onnx/_internal/exporter/_core.py`

## File Information

- **Original File**: [torch/onnx/_internal/exporter/_core.py](../../../../../torch/onnx/_internal/exporter/_core.py)
- **Documentation**: [`_core.py_docs.md`](./_core.py_docs.md)
- **Folder**: `torch/onnx/_internal/exporter`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`GraphModule`**: [_core.py_docs.md](./_core.py_docs.md)
- **`InputKind`**: [_core.py_docs.md](./_core.py_docs.md)
- **`OutputKind`**: [_core.py_docs.md](./_core.py_docs.md)
- **`TorchTensor`**: [_core.py_docs.md](./_core.py_docs.md)
- **`in`**: [_core.py_docs.md](./_core.py_docs.md)
- **`is`**: [_core.py_docs.md](./_core.py_docs.md)
- **`will`**: [_core.py_docs.md](./_core.py_docs.md)

### Functions

- **`__array__`**: [_core.py_docs.md](./_core.py_docs.md)
- **`__init__`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_convert_fx_arg_to_onnx_arg`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_exported_program_to_onnx_program`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_format_exception`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_format_exceptions_for_all_strategies`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_get_cbytes`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_get_inputs_and_attributes`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_get_node_namespace`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_get_onnxscript_opset`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_get_qualified_module_name`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_get_scope_name`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_handle_call_function_node`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_handle_call_function_node_with_lowering`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_handle_get_attr_node`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_handle_getitem_node`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_handle_output_node`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_handle_placeholder_node`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_is_onnx_op`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_maybe_start_profiler`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_maybe_stop_profiler_and_get_result`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_parse_onnx_op`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_prepare_exported_program_for_export`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_set_node_metadata`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_set_shape_type`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_set_shape_types`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_summarize_exception_stack`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_translate_fx_graph`**: [_core.py_docs.md](./_core.py_docs.md)
- **`_verbose_printer`**: [_core.py_docs.md](./_core.py_docs.md)
- **`export`**: [_core.py_docs.md](./_core.py_docs.md)
- **`exported_program_to_ir`**: [_core.py_docs.md](./_core.py_docs.md)
- **`forward`**: [_core.py_docs.md](./_core.py_docs.md)
- **`numpy`**: [_core.py_docs.md](./_core.py_docs.md)
- **`tobytes`**: [_core.py_docs.md](./_core.py_docs.md)
- **`tofile`**: [_core.py_docs.md](./_core.py_docs.md)
- **`torch_dtype_to_onnx_dtype`**: [_core.py_docs.md](./_core.py_docs.md)

### Imports

- **`Any`**: [_core.py_docs.md](./_core.py_docs.md)
- **`Callable`**: [_core.py_docs.md](./_core.py_docs.md)
- **`__future__`**: [_core.py_docs.md](./_core.py_docs.md)
- **`annotations`**: [_core.py_docs.md](./_core.py_docs.md)
- **`collections.abc`**: [_core.py_docs.md](./_core.py_docs.md)
- **`convenience`**: [_core.py_docs.md](./_core.py_docs.md)
- **`ctypes`**: [_core.py_docs.md](./_core.py_docs.md)
- **`datetime`**: [_core.py_docs.md](./_core.py_docs.md)
- **`graph_signature`**: [_core.py_docs.md](./_core.py_docs.md)
- **`import`**: [_core.py_docs.md](./_core.py_docs.md)
- **`inspect`**: [_core.py_docs.md](./_core.py_docs.md)
- **`ir`**: [_core.py_docs.md](./_core.py_docs.md)
- **`itertools`**: [_core.py_docs.md](./_core.py_docs.md)
- **`logging`**: [_core.py_docs.md](./_core.py_docs.md)
- **`numpy.typing`**: [_core.py_docs.md](./_core.py_docs.md)
- **`onnxscript`**: [_core.py_docs.md](./_core.py_docs.md)
- **`onnxscript.evaluator`**: [_core.py_docs.md](./_core.py_docs.md)
- **`onnxscript.ir`**: [_core.py_docs.md](./_core.py_docs.md)
- **`operator`**: [_core.py_docs.md](./_core.py_docs.md)
- **`os`**: [_core.py_docs.md](./_core.py_docs.md)
- **`pathlib`**: [_core.py_docs.md](./_core.py_docs.md)
- **`pyinstrument`**: [_core.py_docs.md](./_core.py_docs.md)
- **`textwrap`**: [_core.py_docs.md](./_core.py_docs.md)
- **`torch`**: [_core.py_docs.md](./_core.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [_core.py_docs.md](./_core.py_docs.md)
- **`torch.export`**: [_core.py_docs.md](./_core.py_docs.md)
- **`torch.fx`**: [_core.py_docs.md](./_core.py_docs.md)
- **`torch.onnx._internal._lazy_import`**: [_core.py_docs.md](./_core.py_docs.md)
- **`torch.onnx._internal.exporter`**: [_core.py_docs.md](./_core.py_docs.md)
- **`traceback`**: [_core.py_docs.md](./_core.py_docs.md)
- **`typing`**: [_core.py_docs.md](./_core.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/onnx/_internal/exporter`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/onnx/_internal/exporter`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/onnx/_internal/exporter`):

- [`_onnx_program.py_docs.md_docs.md`](./_onnx_program.py_docs.md_docs.md)
- [`_decomp.py_docs.md_docs.md`](./_decomp.py_docs.md_docs.md)
- [`_testing.py_docs.md_docs.md`](./_testing.py_docs.md_docs.md)
- [`_flags.py_docs.md_docs.md`](./_flags.py_docs.md_docs.md)
- [`_verification.py_docs.md_docs.md`](./_verification.py_docs.md_docs.md)
- [`_dispatching.py_docs.md_docs.md`](./_dispatching.py_docs.md_docs.md)
- [`_errors.py_kw.md_docs.md`](./_errors.py_kw.md_docs.md)
- [`_schemas.py_kw.md_docs.md`](./_schemas.py_kw.md_docs.md)
- [`_ir_passes.py_kw.md_docs.md`](./_ir_passes.py_kw.md_docs.md)
- [`_compat.py_kw.md_docs.md`](./_compat.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_core.py_kw.md_docs.md`
- **Keyword Index**: `_core.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
