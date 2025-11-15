# Documentation: `docs/torch/_export/non_strict_utils.py_kw.md`

## File Metadata

- **Path**: `docs/torch/_export/non_strict_utils.py_kw.md`
- **Size**: 8,547 bytes (8.35 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Keyword Index: `torch/_export/non_strict_utils.py`

## File Information

- **Original File**: [torch/_export/non_strict_utils.py](../../../torch/_export/non_strict_utils.py)
- **Documentation**: [`non_strict_utils.py_docs.md`](./non_strict_utils.py_docs.md)
- **Folder**: `torch/_export`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Classs

- **`_KeyPath`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_KeyPathTrie`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_NonStrictTorchFunctionHandler`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`recursion`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)

### Functions

- **`__init__`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`__torch_function__`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_clean_dynamic_markers`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_constrain_user_specified_dimhint_range`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_create_symbolic_context_for_tensor`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_enable_graph_inputs_of_type_nn_module`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_enter_enable_graph_inputs_of_type_nn_module`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_exit_enable_graph_inputs_of_type_nn_module`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_fakify_module_inputs`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_fakify_script_objects`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_flatten_dynamic_shapes`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_gather_constant_attrs`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_get_graph_inputs_of_type_nn_module`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_is_constant_argument`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_is_unbacked_symint`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_leaf_mod_and_attr`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_maybe_fakify_obj`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_override`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_override_builtin_ops`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_tensor_min_max`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_tree_map_helper`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`add`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`fakify`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`get`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`inner`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`is_int`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`key_path_to_source`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`make_constraints`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`make_fake_inputs`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`make_sourced_prefixes`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`produce_guards_and_solve_constraints`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`rewrite`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`run`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)

### Imports

- **`Any`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`AttrSource`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`Callable`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`ConstantAttrMap`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`Constraint`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`CustomObjArgument`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`FakeScriptObject`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`FakeTensorMode`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`Source`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`Symbol`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`TrackedFake`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_config`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`_fakify_params_buffers`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`builtins`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`collections`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`collections.abc`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`contextlib`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`contextmanager`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`defaultdict`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`functools`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`inspect`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`int_oo`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`is_opaque_type`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`is_traceable_wrapper_subclass`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`logging`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`math`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`sympy`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`sys`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`torch`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`torch._dynamo.source`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`torch._dynamo.variables.builder`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`torch._export.passes.lift_constants_pass`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`torch._export.utils`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`torch._functorch.config`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`torch._guards`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`torch._library.fake_class_registry`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`torch._library.opaque_object`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`torch._subclasses.fake_tensor`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`torch.distributed._functional_collectives`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`torch.export`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`torch.export.dynamic_shapes`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`torch.export.graph_signature`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`torch.fx.experimental`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`torch.fx.experimental.symbolic_shapes`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`torch.utils._python_dispatch`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`torch.utils._pytree`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`torch.utils._sympy.numbers`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)
- **`typing`**: [non_strict_utils.py_docs.md](./non_strict_utils.py_docs.md)


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

- **File Documentation**: `non_strict_utils.py_kw.md_docs.md`
- **Keyword Index**: `non_strict_utils.py_kw.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
