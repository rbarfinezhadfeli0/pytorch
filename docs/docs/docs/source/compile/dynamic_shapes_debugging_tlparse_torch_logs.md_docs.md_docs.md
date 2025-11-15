# Documentation: `docs/docs/source/compile/dynamic_shapes_debugging_tlparse_torch_logs.md_docs.md`

## File Metadata

- **Path**: `docs/docs/source/compile/dynamic_shapes_debugging_tlparse_torch_logs.md_docs.md`
- **Size**: 6,128 bytes (5.98 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `docs/source/compile/dynamic_shapes_debugging_tlparse_torch_logs.md`

## File Metadata

- **Path**: `docs/source/compile/dynamic_shapes_debugging_tlparse_torch_logs.md`
- **Size**: 3,328 bytes (3.25 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
(debugging-tlparse-torch-logs)=
# Debugging with `tlparse` and `TORCH_LOGS=dynamic`

`tlparse` is a tool used for analyzing and understanding the compilation
process in PyTorch, particularly when dealing with dynamic shapes. It helps
identify where guards and specializations occur in your code.

`TORCH_LOGS=dynamic` is an environment variable setting that enables detailed
logging of dynamic shape operations, providing insights into how symbolic
shapes are handled during execution.

This section will guide you through using `tlparse` and `TORCH_LOGS=dynamic` to
troubleshoot dynamic shape issues in your code, including debugging
specialization, guards, and more.

# Debugging Specialization

In the following example, `x.shape[0]` is dynamic but becomes specialized due to multiplication:

```python
import torch

@torch.compile
def fn(x, y):
    return x * y

x = torch.randn(5)
y = torch.randn(5)
torch._dynamo.decorators.mark_dynamic(x, 0)

fn(x, y)
```

By using `TORCH_LOGS=dynamic`, you can observe this specialization in the logs:

```xml
TORCH_LOGS=dynamic python tl.py
I0721 11:10:00.950000 845259 torch/fx/experimental/symbolic_shapes.py:3776] [0/0] create_env
I0721 11:10:01.030000 845259 torch/fx/experimental/symbolic_shapes.py:5117] [0/0] create_symbol s77 = 5 for L['x'].size()[0] [2, int_oo] return x * y  # tl.py:5 in fn (_dynamo/variables/builder.py:3466 in <lambda>), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="s77" or to suppress this message run with TORCHDYNAMO_EXTENDED_ADVICE="0"
I0721 11:10:01.038000 845259 torch/fx/experimental/symbolic_shapes.py:7211] [0/0] eval Eq(s77, 5) [guard added] return x * y  # tl.py:5 in fn (_subclasses/fake_impls.py:922 in infer_size), for more info run with TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED="Eq(s77, 5)"
```

The line `eval Eq(s77, 5) [guard added] return x * y # tl.py:5` indicates the specialization.

## Debugging Guards

Consider the following code, which may cause recompilations due to dynamic
shapes:

```python
import torch

@torch.compile
def fn(x, y):
    if x.shape[0] < 10:
        return x * y

x = torch.randn(5)
y = torch.randn(5)
torch._dynamo.decorators.mark_dynamic(x, 0)
torch._dynamo.decorators.mark_dynamic(y, 0)

fn(x, y)
```

To identify where dynamic shape guards originate, use `tlparse`. Here is an example tlparse output:

```{image} ../_static/img/dynamic_shapes/tlparse9_debugging_guards.png
```

By clicking on the `dynamo_cpp_guards` link, you can view all guards from the compilation, including the symbolic shape guard `L['x'].size()[0] <= 9`.

Astute readers will notice the 0/1 specialization where we guard on `L['x'].size()[0] >= 2`. By modifying the code to use unbacked symbols, this guard is removed:

```python
import torch

@torch.compile
def fn(x, y):
    # Necessary runtime assert since we can't guard on unbacked
    torch._check(x.shape[0] < 10)
    if x.shape[0] < 10:
        return x * y

x = torch.randn(5)
y = torch.randn(5)
torch._dynamo.decorators.mark_unbacked(x, 0)
torch._dynamo.decorators.mark_unbacked(y, 0)

fn(x, y)
```

Now, this compiled region can be used for inputs of size 0 and 1:

```{image} ../_static/img/dynamic_shapes/tlparse10_debugging_guards_unbacked.png
```

```{seealso}
* {ref}`dynamic_shapes`
* {ref}`troubleshooting_guardondatadependentsymnode_errors`
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


## Cross-References

- **File Documentation**: `dynamic_shapes_debugging_tlparse_torch_logs.md_docs.md`
- **Keyword Index**: `dynamic_shapes_debugging_tlparse_torch_logs.md_kw.md`
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

- **File Documentation**: `dynamic_shapes_debugging_tlparse_torch_logs.md_docs.md_docs.md`
- **Keyword Index**: `dynamic_shapes_debugging_tlparse_torch_logs.md_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
