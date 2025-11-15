# Documentation: `docs/source/fx.experimental.md`

## File Metadata

- **Path**: `docs/source/fx.experimental.md`
- **Size**: 3,643 bytes (3.56 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
```{eval-rst}
.. currentmodule:: torch.fx.experimental
```

# torch.fx.experimental

:::{warning}
These APIs are experimental and subject to change without notice.
:::

```{eval-rst}
.. autoclass:: torch.fx.experimental.sym_node.DynamicInt
```

## torch.fx.experimental.sym_node

```{eval-rst}
.. currentmodule:: torch.fx.experimental.sym_node
```

```{eval-rst}
.. automodule:: torch.fx.experimental.sym_node
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    is_channels_last_contiguous_2d
    is_channels_last_contiguous_3d
    is_channels_last_strides_2d
    is_channels_last_strides_3d
    is_contiguous
    is_non_overlapping_and_dense_indicator
    method_to_operator
    sympy_is_channels_last_contiguous_2d
    sympy_is_channels_last_contiguous_3d
    sympy_is_channels_last_strides_2d
    sympy_is_channels_last_strides_3d
    sympy_is_channels_last_strides_generic
    sympy_is_contiguous
    sympy_is_contiguous_generic
```

## torch.fx.experimental.symbolic_shapes

```{eval-rst}
.. currentmodule:: torch.fx.experimental.symbolic_shapes
```

```{eval-rst}
.. automodule:: torch.fx.experimental.symbolic_shapes
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    ShapeEnv
    DimDynamic
    StrictMinMaxConstraint
    RelaxedUnspecConstraint
    EqualityConstraint
    SymbolicContext
    StatelessSymbolicContext
    StatefulSymbolicContext
    SubclassSymbolicContext
    DimConstraints
    ShapeEnvSettings
    ConvertIntKey
    CallMethodKey
    PropagateUnbackedSymInts
    DivideByKey
    InnerTensorKey
    Specialization

    hint_int
    is_concrete_int
    is_concrete_bool
    is_concrete_float
    has_free_symbols
    has_free_unbacked_symbols
    guard_or_true
    guard_or_false
    guard_size_oblivious
    sym_and
    sym_eq
    sym_or
    constrain_range
    constrain_unify
    canonicalize_bool_expr
    statically_known_true
    statically_known_false
    has_static_value
    lru_cache
    check_consistent
    compute_unbacked_bindings
    rebind_unbacked
    resolve_unbacked_bindings
    is_accessor_node
    cast_symbool_to_symint_guardless
    create_contiguous
    error
    eval_guards
    eval_is_non_overlapping_and_dense
    find_symbol_binding_fx_nodes
    free_symbols
    free_unbacked_symbols
    fx_placeholder_targets
    fx_placeholder_vals
    guard_bool
    guard_float
    guard_int
    guard_scalar
    has_hint
    has_symbolic_sizes_strides
    is_nested_int
    is_symbol_binding_fx_node
    is_symbolic
```

## torch.fx.experimental.proxy_tensor

```{eval-rst}
.. currentmodule:: torch.fx.experimental.proxy_tensor
```

```{eval-rst}
.. automodule:: torch.fx.experimental.proxy_tensor
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    make_fx
    handle_sym_dispatch
    get_proxy_mode
    maybe_enable_thunkify
    maybe_disable_thunkify
    decompose
    disable_autocast_cache
    disable_proxy_modes_tracing
    extract_val
    fake_signature
    fetch_object_proxy
    fetch_sym_proxy
    has_proxy_slot
    is_sym_node
    maybe_handle_decomp
    proxy_call
    set_meta
    set_original_aten_op
    set_proxy_slot
    snapshot_fake
```

## torch.fx.experimental.unification.unification_tools

```{eval-rst}
.. currentmodule:: torch.fx.experimental.unification.unification_tools
```

```{eval-rst}
.. automodule:: torch.fx.experimental.unification.unification_tools
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    assoc
    assoc_in
    dissoc
    first
    keyfilter
    keymap
    merge
    merge_with
    update_in
    valfilter
    valmap

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/source`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/source`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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

Files in the same folder (`docs/source`):

- [`torch.compiler_troubleshooting.md_docs.md`](./torch.compiler_troubleshooting.md_docs.md)
- [`torch.compiler_aot_inductor_debugging_guide.md_docs.md`](./torch.compiler_aot_inductor_debugging_guide.md_docs.md)
- [`mtia.memory.md_docs.md`](./mtia.memory.md_docs.md)
- [`torch.compiler_get_started.md_docs.md`](./torch.compiler_get_started.md_docs.md)
- [`torch.compiler_dynamo_deepdive.md_docs.md`](./torch.compiler_dynamo_deepdive.md_docs.md)
- [`mtia.mtia_graph.md_docs.md`](./mtia.mtia_graph.md_docs.md)
- [`hub.md_docs.md`](./hub.md_docs.md)
- [`torch_nccl_environment_variables.md_docs.md`](./torch_nccl_environment_variables.md_docs.md)
- [`optim.md_docs.md`](./optim.md_docs.md)
- [`torch.compiler_aot_inductor.md_docs.md`](./torch.compiler_aot_inductor.md_docs.md)


## Cross-References

- **File Documentation**: `fx.experimental.md_docs.md`
- **Keyword Index**: `fx.experimental.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
