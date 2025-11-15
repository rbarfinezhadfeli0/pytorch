# Documentation: `docs/torch/ao/quantization/pt2e/_numeric_debugger.py_kw.md`

## File Metadata

- **Path**: `docs/torch/ao/quantization/pt2e/_numeric_debugger.py_kw.md`
- **Size**: 4,844 bytes (4.73 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/ao/quantization/pt2e/_numeric_debugger.py`

## File Information

- **Original File**: [torch/ao/quantization/pt2e/_numeric_debugger.py](../../../../../torch/ao/quantization/pt2e/_numeric_debugger.py)
- **Documentation**: [`_numeric_debugger.py_docs.md`](./_numeric_debugger.py_docs.md)
- **Folder**: `torch/ao/quantization/pt2e`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`NodeAccuracySummary`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`OutputLogger`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`QuantizationComparisonResult`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`for`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`import`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)

### Functions

- **`__extra_repr__`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`__init__`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`__post_init__`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`__repr__`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`_assign_debug_handle`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`_detach`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`_find_max_id`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`_insert_logger`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`_loss_fn`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`_module_stack_to_str`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`_tensor_shape_equals`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`compare_results`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`extract_results_from_loggers`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`forward`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`generate_numeric_debug_handle`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`loss`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`mse_loss`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`prepare_for_propagation_comparison`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`sqnr`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)

### Imports

- **`Callable`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`ExportedProgram`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`GraphModule`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`bfs_trace_with_node_process`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`collections.abc`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`compute_sqnr`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`copy`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`dataclass`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`dataclasses`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`functional`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`get_new_attr_name_with_prefix`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`logging`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`torch`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`torch.ao.ns.fx.utils`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`torch.ao.quantization.fx.utils`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`torch.ao.quantization.pt2e.graph_utils`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`torch.export`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`torch.fx`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)
- **`torch.nn`**: [_numeric_debugger.py_docs.md](./_numeric_debugger.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/ao/quantization/pt2e`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/ao/quantization/pt2e`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`docs/torch/ao/quantization/pt2e`):

- [`lowering.py_kw.md_docs.md`](./lowering.py_kw.md_docs.md)
- [`duplicate_dq_pass.py_docs.md_docs.md`](./duplicate_dq_pass.py_docs.md_docs.md)
- [`prepare.py_kw.md_docs.md`](./prepare.py_kw.md_docs.md)
- [`qat_utils.py_docs.md_docs.md`](./qat_utils.py_docs.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`graph_utils.py_docs.md_docs.md`](./graph_utils.py_docs.md_docs.md)
- [`export_utils.py_docs.md_docs.md`](./export_utils.py_docs.md_docs.md)
- [`lowering.py_docs.md_docs.md`](./lowering.py_docs.md_docs.md)
- [`export_utils.py_kw.md_docs.md`](./export_utils.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `_numeric_debugger.py_kw.md_docs.md`
- **Keyword Index**: `_numeric_debugger.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
