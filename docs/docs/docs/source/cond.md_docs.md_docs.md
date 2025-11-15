# Documentation: `docs/docs/source/cond.md_docs.md`

## File Metadata

- **Path**: `docs/docs/source/cond.md_docs.md`
- **Size**: 8,474 bytes (8.28 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `docs/source/cond.md`

## File Metadata

- **Path**: `docs/source/cond.md`
- **Size**: 5,962 bytes (5.82 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
(cond)=

# Control Flow - Cond

`torch.cond` is a structured control flow operator. It can be used to specify if-else like control flow
and can logically be seen as implemented as follows.

```python
def cond(
    pred: Union[bool, torch.Tensor],
    true_fn: Callable,
    false_fn: Callable,
    operands: Tuple[torch.Tensor]
):
    if pred:
        return true_fn(*operands)
    else:
        return false_fn(*operands)
```

Its unique power lies in its ability of expressing **data-dependent control flow**: it lowers to a conditional
operator (`torch.ops.higher_order.cond`), which preserves predicate, true function and false functions.
This unlocks great flexibility in writing and deploying models that change model architecture based on
the **value** or **shape** of inputs or intermediate outputs of tensor operations.

```{warning}
`torch.cond` is a prototype feature in PyTorch. It has limited support for input and output types and
doesn't support training currently. Please look forward to a more stable implementation in a future version of PyTorch.
Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype
```

## Examples

Below is an example that uses cond to branch based on input shape:

```python
import torch

def true_fn(x: torch.Tensor):
    return x.cos() + x.sin()

def false_fn(x: torch.Tensor):
    return x.sin()

class DynamicShapeCondPredicate(torch.nn.Module):
    """
    A basic usage of cond based on dynamic shape predicate.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def true_fn(x: torch.Tensor):
            return x.cos()

        def false_fn(x: torch.Tensor):
            return x.sin()

        return torch.cond(x.shape[0] > 4, true_fn, false_fn, (x,))

dyn_shape_mod = DynamicShapeCondPredicate()
```

We can eagerly run the model and expect the results vary based on input shape:

```python
inp = torch.randn(3)
inp2 = torch.randn(5)
assert torch.equal(dyn_shape_mod(inp), false_fn(inp))
assert torch.equal(dyn_shape_mod(inp2), true_fn(inp2))
```

We can export the model for further transformations and deployment:

```python
inp = torch.randn(4, 3)
dim_batch = torch.export.Dim("batch", min=2)
ep = torch.export.export(DynamicShapeCondPredicate(), (inp,), {}, dynamic_shapes={"x": {0: dim_batch}})
print(ep)
```

This gives us an exported program as shown below:

```
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: f32[s0, 3]):
        sym_size: Sym(s0) = torch.ops.aten.sym_size.int(arg0_1, 0)
        gt: Sym(s0 > 4) = sym_size > 4;  sym_size = None
        true_graph_0 = self.true_graph_0
        false_graph_0 = self.false_graph_0
        conditional: f32[s0, 3] = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, [arg0_1]);  gt = true_graph_0 = false_graph_0 = arg0_1 = None
        return (conditional,)

    class <lambda>(torch.nn.Module):
        def forward(self, arg0_1: f32[s0, 3]):
            cos: f32[s0, 3] = torch.ops.aten.cos.default(arg0_1)
            sin: f32[s0, 3] = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
            add: f32[s0, 3] = torch.ops.aten.add.Tensor(cos, sin);  cos = sin = None
            return add

    class <lambda>(torch.nn.Module):
        def forward(self, arg0_1: f32[s0, 3]):
            sin: f32[s0, 3] = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
            return sin
```

Notice that `torch.cond` is lowered to `torch.ops.higher_order.cond`, its predicate becomes a Symbolic expression over the shape of input,
and branch functions becomes two sub-graph attributes of the top level graph module.

Here is another example that showcases how to express a data-dependent control flow:

```python
class DataDependentCondPredicate(torch.nn.Module):
    """
    A basic usage of cond based on data dependent predicate.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cond(x.sum() > 4.0, true_fn, false_fn, (x,))
```

The exported program we get after export:

```
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: f32[s0, 3]):
        sum_1: f32[] = torch.ops.aten.sum.default(arg0_1)
        gt: b8[] = torch.ops.aten.gt.Scalar(sum_1, 4.0);  sum_1 = None

        true_graph_0 = self.true_graph_0
        false_graph_0 = self.false_graph_0
        conditional: f32[s0, 3] = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, [arg0_1]);  gt = true_graph_0 = false_graph_0 = arg0_1 = None
        return (conditional,)

    class <lambda>(torch.nn.Module):
        def forward(self, arg0_1: f32[s0, 3]):
            cos: f32[s0, 3] = torch.ops.aten.cos.default(arg0_1)
            sin: f32[s0, 3] = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
            add: f32[s0, 3] = torch.ops.aten.add.Tensor(cos, sin);  cos = sin = None
            return add

    class <lambda>(torch.nn.Module):
        def forward(self, arg0_1: f32[s0, 3]):
            sin: f32[s0, 3] = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
            return sin
```

## Invariants of torch.ops.higher_order.cond

There are several useful invariants for `torch.ops.higher_order.cond`:

- For predicate:
    - Dynamicness of predicate is preserved (e.g. `gt` shown in the above example)
    - If the predicate in user-program is constant (e.g. a python bool constant), the `pred` of the operator will be a constant.

- For branches:
    - The input and output signature will be a flattened tuple.
    - They are `torch.fx.GraphModule`.
    - Closures in original function becomes explicit inputs. No closures.
    - No mutations on inputs or globals are allowed.

- For operands:
    - It will also be a flat tuple.

- Nesting of `torch.cond` in user program becomes nested graph modules.

## API Reference

```{eval-rst}
.. autofunction:: torch._higher_order_ops.cond.cond
```

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

- **Object-Oriented Design**: Uses classes and constructors
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

- **File Documentation**: `cond.md_docs.md`
- **Keyword Index**: `cond.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/docs/source`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/docs/source`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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

Files in the same folder (`docs/docs/source`):

- [`distributions.md_docs.md_docs.md`](./distributions.md_docs.md_docs.md)
- [`distributed.optim.md_docs.md_docs.md`](./distributed.optim.md_docs.md_docs.md)
- [`torch.compiler_dynamic_shapes.md_kw.md_docs.md`](./torch.compiler_dynamic_shapes.md_kw.md_docs.md)
- [`tensor_attributes.rst_docs.md_docs.md`](./tensor_attributes.rst_docs.md_docs.md)
- [`tensor_attributes.rst_kw.md_docs.md`](./tensor_attributes.rst_kw.md_docs.md)
- [`torch.compiler_dynamo_overview.md_docs.md_docs.md`](./torch.compiler_dynamo_overview.md_docs.md_docs.md)
- [`mtia.memory.md_kw.md_docs.md`](./mtia.memory.md_kw.md_docs.md)
- [`nn.attention.varlen.md_kw.md_docs.md`](./nn.attention.varlen.md_kw.md_docs.md)
- [`cpu.rst_kw.md_docs.md`](./cpu.rst_kw.md_docs.md)
- [`torch.compiler_faq.md_docs.md_docs.md`](./torch.compiler_faq.md_docs.md_docs.md)


## Cross-References

- **File Documentation**: `cond.md_docs.md_docs.md`
- **Keyword Index**: `cond.md_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
