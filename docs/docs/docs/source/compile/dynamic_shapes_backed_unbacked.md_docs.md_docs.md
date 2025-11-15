# Documentation: `docs/docs/source/compile/dynamic_shapes_backed_unbacked.md_docs.md`

## File Metadata

- **Path**: `docs/docs/source/compile/dynamic_shapes_backed_unbacked.md_docs.md`
- **Size**: 4,879 bytes (4.76 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `docs/source/compile/dynamic_shapes_backed_unbacked.md`

## File Metadata

- **Path**: `docs/source/compile/dynamic_shapes_backed_unbacked.md`
- **Size**: 2,012 bytes (1.96 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
(backed-vs-unbacked-symints)=
# Backed vs Unbacked Symints

Backed `SymInts` are symbolic integers that have a concrete value or "hint"
associated with them. This means that torch can use these values to make
decisions about control flow, such as determining which branch of code
to execute. They are typically derived from operations where the size or
value is known or can be inferred.

Unbacked `SymInts` are symbolic integers that do not have a concrete value or
hint. They often arise from data-dependent operations, such as `.nonzero()`
or `.item()`, where the size or value cannot be determined at compile time.
Since they lack a concrete value, they cannot be used for control flow
decisions, and attempting to do so requires a graph break.

Unbacked `SymInts` use *oblivious-size reasoning* which is particularly
useful when you are dealing with
{ref}`0/1 specialization recompilation problem <zero-one-specialization>`.

In summary, backed `SymInts` have known values that can be used for
decision-making, while unbacked `SymInts` do not, requiring special handling
to avoid graph breaks.

Unbacked symbolic integers can be too restrictive, causing most PyTorch programs
to fail. To address this, you can use the following methods and APIs as
workaround:

* Use higher-level APIs like `empty` instead of `empty_strided` to create tensors.
This ensures the tensor is non-overlapping and dense, avoiding unnecessary stride
sorting and guard creation.to avoid unnecessary recomputation of these properties.

* Modify your code to make precomputed properties *lazy*. This ensures that
guards on unbacked symbolic integers are only applied when necessary,
reducing computational overhead.

## How to use unbacked
To use unbacked APIs, replace `mark_dynamic` with `mark_unbacked` and
`TORCH_COMPILE_DYNAMIC_SOURCES` with `TORCH_COMPILE_UNBACKED_SOURCES`.
This tells the compiler to treat an input as unbacked.

```{seealso}
* {ref}`dynamic_shapes`
* {ref}`torch.export`
* {ref}`what_is_a_specialization`
```

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/source/compile`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/source/compile`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/source/compile`):

- [`dynamic_shapes_troubleshooting_guardon_errors.md_docs.md`](./dynamic_shapes_troubleshooting_guardon_errors.md_docs.md)
- [`dynamic_shapes_core_concepts.md_docs.md`](./dynamic_shapes_core_concepts.md_docs.md)
- [`programming_model.error_on_graph_break.md_docs.md`](./programming_model.error_on_graph_break.md_docs.md)
- [`dynamic_shapes_troubleshooting.md_docs.md`](./dynamic_shapes_troubleshooting.md_docs.md)
- [`programming_model.recompilation.md_docs.md`](./programming_model.recompilation.md_docs.md)
- [`dynamic_shapes_zero_one_specialization.md_docs.md`](./dynamic_shapes_zero_one_specialization.md_docs.md)
- [`dynamic_shapes_beyond_the_basics.md_docs.md`](./dynamic_shapes_beyond_the_basics.md_docs.md)
- [`programming_model.graph_breaks_index.md_docs.md`](./programming_model.graph_breaks_index.md_docs.md)
- [`programming_model.md_docs.md`](./programming_model.md_docs.md)
- [`dynamic_shapes_debugging_tlparse_torch_logs.md_docs.md`](./dynamic_shapes_debugging_tlparse_torch_logs.md_docs.md)


## Cross-References

- **File Documentation**: `dynamic_shapes_backed_unbacked.md_docs.md`
- **Keyword Index**: `dynamic_shapes_backed_unbacked.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/docs/source/compile`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/docs/source/compile`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/docs/source/compile`):

- [`dynamic_shapes_advanced_control_options.md_docs.md_docs.md`](./dynamic_shapes_advanced_control_options.md_docs.md_docs.md)
- [`programming_model.recompilation.md_docs.md_docs.md`](./programming_model.recompilation.md_docs.md_docs.md)
- [`programming_model.dynamo_core_concepts.md_docs.md_docs.md`](./programming_model.dynamo_core_concepts.md_docs.md_docs.md)
- [`programming_model.nested_graph_breaks.md_kw.md_docs.md`](./programming_model.nested_graph_breaks.md_kw.md_docs.md)
- [`programming_model.where_to_apply_compile.md_docs.md_docs.md`](./programming_model.where_to_apply_compile.md_docs.md_docs.md)
- [`programming_model.graph_breaks_index.md_docs.md_docs.md`](./programming_model.graph_breaks_index.md_docs.md_docs.md)
- [`programming_model.nested_graph_breaks.md_docs.md_docs.md`](./programming_model.nested_graph_breaks.md_docs.md_docs.md)
- [`programming_model.fullgraph_true.md_docs.md_docs.md`](./programming_model.fullgraph_true.md_docs.md_docs.md)
- [`dynamic_shapes_zero_one_specialization.md_docs.md_docs.md`](./dynamic_shapes_zero_one_specialization.md_docs.md_docs.md)
- [`programming_model.dynamo_nonstrict_trace.md_kw.md_docs.md`](./programming_model.dynamo_nonstrict_trace.md_kw.md_docs.md)


## Cross-References

- **File Documentation**: `dynamic_shapes_backed_unbacked.md_docs.md_docs.md`
- **Keyword Index**: `dynamic_shapes_backed_unbacked.md_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
