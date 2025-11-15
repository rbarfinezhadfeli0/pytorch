# Documentation: `docs/torch/_higher_order_ops/auto_functionalize.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_higher_order_ops/auto_functionalize.py_kw.md`
- **Size**: 7,583 bytes (7.41 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_higher_order_ops/auto_functionalize.py`

## File Information

- **Original File**: [torch/_higher_order_ops/auto_functionalize.py](../../../torch/_higher_order_ops/auto_functionalize.py)
- **Documentation**: [`auto_functionalize.py_docs.md`](./auto_functionalize.py_docs.md)
- **Folder**: `torch/_higher_order_ops`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`AutoFunctionalized`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`AutoFunctionalizedV2`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`FunctionalCallableWithEpilogue`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`SchemaHolder`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`ViewInfo`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`class`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`from`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)

### Functions

- **`__call__`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`__eq__`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`__hash__`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`__init__`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`_functionalize_callable`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`_generate_new_op_kwargs_from_bases`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`_maybe_register_subgraph`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`auto_functionalized_dense`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`auto_functionalized_fake`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`auto_functionalized_func`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`auto_functionalized_proxy`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`auto_functionalized_v2_dense`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`auto_functionalized_v2_fake`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`auto_functionalized_v2_func`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`auto_functionalized_v2_proxy`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`can_auto_functionalize`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`do_auto_functionalize`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`do_auto_functionalize_v2`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`from_tree_spec`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`get_arg`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`get_base`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`get_mutable_args`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`get_mutable_args_from_schema`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`is_alias`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`maybe_copy`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`read_single_view`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`read_view_information_from_args`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`regenerate_view`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`set_result`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`sync_update`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`try_use_slice`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`update_dict`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`use_alias`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`use_as_strided`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`use_slice`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`write_single_view`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`write_view_information_to_args`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)

### Imports

- **`ABC`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`Any`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`Callable`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`DispatchKey`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`FakeTensorMode`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`HigherOrderOperator`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`PythonFunctionalizeAPI`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`Tensor`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`abc`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`clone_preserve_strides`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`collections.abc`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`dataclass`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`dataclasses`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`statically_known_true`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`torch`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`torch._C`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`torch._higher_order_ops.utils`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`torch._library.utils`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`torch._ops`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`torch._prims_common`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`torch._subclasses.functional_tensor`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`torch.utils._pytree`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`typing`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)
- **`warnings`**: [auto_functionalize.py_docs.md](./auto_functionalize.py_docs.md)


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

- **Abstract Base Classes**: Defines abstract interfaces


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

- **File Documentation**: `auto_functionalize.py_kw.md_docs.md`
- **Keyword Index**: `auto_functionalize.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
