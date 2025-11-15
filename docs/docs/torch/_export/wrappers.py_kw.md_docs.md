# Documentation: `docs/torch/_export/wrappers.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_export/wrappers.py_kw.md`
- **Size**: 5,192 bytes (5.07 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_export/wrappers.py`

## File Information

- **Original File**: [torch/_export/wrappers.py](../../../torch/_export/wrappers.py)
- **Documentation**: [`wrappers.py_docs.md`](./wrappers.py_docs.md)
- **Folder**: `torch/_export`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`ExportTracepoint`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`FooTensor`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`MyCoolCustomAutogradFunc`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`constructors`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`instance`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`spec`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`tensor`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`to`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`traceble`**: [wrappers.py_docs.md](./wrappers.py_docs.md)

### Functions

- **`__call__`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`__init__`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`__new__`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`_emit_flat_apply_call`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`_is_init`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`_mark_strict_experimental`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`_register_func_spec_proxy_in_tracer`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`_wrap_submodule`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`_wrap_submodules`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`allow_in_pre_dispatch_graph`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`apply`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`call`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`check_flattened`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`export_tracepoint_cpu`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`export_tracepoint_dispatch_mode`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`export_tracepoint_fake_tensor_mode`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`export_tracepoint_functional`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`mark_subclass_constructor_exportable_experimental`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`post_hook`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`pre_hook`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`update_module_call_signatures`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`wrapper`**: [wrappers.py_docs.md](./wrappers.py_docs.md)

### Imports

- **`DispatchKey`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`FakeTensorMode`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`HigherOrderOperator`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`_get_dispatch_mode_pre_dispatch`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`_maybe_find_pre_dispatch_tf_mode_for_export`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`_pytree`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`autograd_not_implemented`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`contextlib`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`contextmanager`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`functools`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`inspect`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`is_traceable_wrapper_subclass_type`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`strict_mode`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch._C`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch._custom_ops`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch._export.utils`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch._higher_order_ops.flat_apply`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch._higher_order_ops.strict_mode`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch._higher_order_ops.utils`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch._ops`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch.export.custom_ops`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch.fx.experimental.proxy_tensor`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch.utils`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`torch.utils._python_dispatch`**: [wrappers.py_docs.md](./wrappers.py_docs.md)
- **`wraps`**: [wrappers.py_docs.md](./wrappers.py_docs.md)


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

This file is part of the PyTorch framework located at `docs/torch/_export`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_export`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/_export`):

- [`error.py_kw.md_docs.md`](./error.py_kw.md_docs.md)
- [`converter.py_kw.md_docs.md`](./converter.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`pass_base.py_kw.md_docs.md`](./pass_base.py_kw.md_docs.md)
- [`wrappers.py_docs.md_docs.md`](./wrappers.py_docs.md_docs.md)
- [`converter.py_docs.md_docs.md`](./converter.py_docs.md_docs.md)
- [`config.py_kw.md_docs.md`](./config.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`verifier.py_kw.md_docs.md`](./verifier.py_kw.md_docs.md)
- [`verifier.py_docs.md_docs.md`](./verifier.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `wrappers.py_kw.md_docs.md`
- **Keyword Index**: `wrappers.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
