# Documentation: `docs/torch/_higher_order_ops/scan.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_higher_order_ops/scan.py_kw.md`
- **Size**: 4,962 bytes (4.85 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_higher_order_ops/scan.py`

## File Information

- **Original File**: [torch/_higher_order_ops/scan.py](../../../torch/_higher_order_ops/scan.py)
- **Documentation**: [`scan.py_docs.md`](./scan.py_docs.md)
- **Folder**: `torch/_higher_order_ops`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ScanAutogradImpl`**: [scan.py_docs.md](./scan.py_docs.md)
- **`ScanAutogradOp`**: [scan.py_docs.md](./scan.py_docs.md)
- **`ScanForwardIntermediatesHandlingPolicy`**: [scan.py_docs.md](./scan.py_docs.md)
- **`ScanOp`**: [scan.py_docs.md](./scan.py_docs.md)

### Functions

- **`__call__`**: [scan.py_docs.md](./scan.py_docs.md)
- **`__init__`**: [scan.py_docs.md](./scan.py_docs.md)
- **`_extract_carry_and_out`**: [scan.py_docs.md](./scan.py_docs.md)
- **`_fake_scan`**: [scan.py_docs.md](./scan.py_docs.md)
- **`_insert_clone`**: [scan.py_docs.md](./scan.py_docs.md)
- **`_optimize_forward_intermediates`**: [scan.py_docs.md](./scan.py_docs.md)
- **`_scan`**: [scan.py_docs.md](./scan.py_docs.md)
- **`_validate_input`**: [scan.py_docs.md](./scan.py_docs.md)
- **`add`**: [scan.py_docs.md](./scan.py_docs.md)
- **`backward`**: [scan.py_docs.md](./scan.py_docs.md)
- **`bw_single_step_wrapper`**: [scan.py_docs.md](./scan.py_docs.md)
- **`call_backward`**: [scan.py_docs.md](./scan.py_docs.md)
- **`call_forward`**: [scan.py_docs.md](./scan.py_docs.md)
- **`call_operator`**: [scan.py_docs.md](./scan.py_docs.md)
- **`combine_fn`**: [scan.py_docs.md](./scan.py_docs.md)
- **`forward`**: [scan.py_docs.md](./scan.py_docs.md)
- **`gen_schema`**: [scan.py_docs.md](./scan.py_docs.md)
- **`generic_scan`**: [scan.py_docs.md](./scan.py_docs.md)
- **`run_flattened_scan`**: [scan.py_docs.md](./scan.py_docs.md)
- **`scan`**: [scan.py_docs.md](./scan.py_docs.md)
- **`scan_autograd`**: [scan.py_docs.md](./scan.py_docs.md)
- **`scan_batch_rule`**: [scan.py_docs.md](./scan.py_docs.md)
- **`scan_fake_tensor_mode`**: [scan.py_docs.md](./scan.py_docs.md)
- **`scan_functionalize`**: [scan.py_docs.md](./scan.py_docs.md)
- **`scan_op_dense`**: [scan.py_docs.md](./scan.py_docs.md)
- **`scan_proxy_mode`**: [scan.py_docs.md](./scan.py_docs.md)
- **`stack_y`**: [scan.py_docs.md](./scan.py_docs.md)
- **`store_out_in_outs`**: [scan.py_docs.md](./scan.py_docs.md)
- **`trace_scan`**: [scan.py_docs.md](./scan.py_docs.md)
- **`wrap_combine_fn_flat`**: [scan.py_docs.md](./scan.py_docs.md)
- **`wrapper`**: [scan.py_docs.md](./scan.py_docs.md)

### Imports

- **`Any`**: [scan.py_docs.md](./scan.py_docs.md)
- **`Callable`**: [scan.py_docs.md](./scan.py_docs.md)
- **`DispatchKey`**: [scan.py_docs.md](./scan.py_docs.md)
- **`FakeTensorMode`**: [scan.py_docs.md](./scan.py_docs.md)
- **`HigherOrderOperator`**: [scan.py_docs.md](./scan.py_docs.md)
- **`HopSchemaGenerator`**: [scan.py_docs.md](./scan.py_docs.md)
- **`_get_current_dispatch_mode`**: [scan.py_docs.md](./scan.py_docs.md)
- **`clone_input`**: [scan.py_docs.md](./scan.py_docs.md)
- **`collections.abc`**: [scan.py_docs.md](./scan.py_docs.md)
- **`enum`**: [scan.py_docs.md](./scan.py_docs.md)
- **`functools`**: [scan.py_docs.md](./scan.py_docs.md)
- **`itertools`**: [scan.py_docs.md](./scan.py_docs.md)
- **`logging`**: [scan.py_docs.md](./scan.py_docs.md)
- **`materialize_as_graph`**: [scan.py_docs.md](./scan.py_docs.md)
- **`restore_vmap`**: [scan.py_docs.md](./scan.py_docs.md)
- **`torch`**: [scan.py_docs.md](./scan.py_docs.md)
- **`torch._C`**: [scan.py_docs.md](./scan.py_docs.md)
- **`torch._dynamo.utils`**: [scan.py_docs.md](./scan.py_docs.md)
- **`torch._functorch.vmap`**: [scan.py_docs.md](./scan.py_docs.md)
- **`torch._higher_order_ops.partitioner`**: [scan.py_docs.md](./scan.py_docs.md)
- **`torch._higher_order_ops.schema`**: [scan.py_docs.md](./scan.py_docs.md)
- **`torch._higher_order_ops.utils`**: [scan.py_docs.md](./scan.py_docs.md)
- **`torch._ops`**: [scan.py_docs.md](./scan.py_docs.md)
- **`torch._prims_common`**: [scan.py_docs.md](./scan.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [scan.py_docs.md](./scan.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [scan.py_docs.md](./scan.py_docs.md)
- **`torch.utils._python_dispatch`**: [scan.py_docs.md](./scan.py_docs.md)
- **`torch.utils._pytree`**: [scan.py_docs.md](./scan.py_docs.md)
- **`typing`**: [scan.py_docs.md](./scan.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/_higher_order_ops`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_higher_order_ops`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/_higher_order_ops`):

- [`schema.py_docs.md_docs.md`](./schema.py_docs.md_docs.md)
- [`run_const_graph.py_docs.md_docs.md`](./run_const_graph.py_docs.md_docs.md)
- [`effects.py_kw.md_docs.md`](./effects.py_kw.md_docs.md)
- [`partitioner.py_docs.md_docs.md`](./partitioner.py_docs.md_docs.md)
- [`strict_mode.py_docs.md_docs.md`](./strict_mode.py_docs.md_docs.md)
- [`out_dtype.py_kw.md_docs.md`](./out_dtype.py_kw.md_docs.md)
- [`wrap.py_docs.md_docs.md`](./wrap.py_docs.md_docs.md)
- [`while_loop.py_kw.md_docs.md`](./while_loop.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`invoke_subgraph.py_docs.md_docs.md`](./invoke_subgraph.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `scan.py_kw.md_docs.md`
- **Keyword Index**: `scan.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
