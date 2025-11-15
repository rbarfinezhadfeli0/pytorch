# Documentation: `docs/source/func.whirlwind_tour.md`

## File Metadata

- **Path**: `docs/source/func.whirlwind_tour.md`
- **Size**: 5,826 bytes (5.69 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# torch.func Whirlwind Tour

## What is torch.func?

```{eval-rst}
.. currentmodule:: torch.func
```

torch.func, previously known as functorch, is a library for
[JAX](https://github.com/google/jax)-like composable function transforms in
PyTorch.

- A "function transform" is a higher-order function that accepts a numerical
  function and returns a new function that computes a different quantity.
- torch.func has auto-differentiation transforms (`grad(f)` returns a function
  that computes the gradient of `f`), a vectorization/batching transform
  (`vmap(f)` returns a function that computes `f` over batches of inputs),
  and others.
- These function transforms can compose with each other arbitrarily. For
  example, composing `vmap(grad(f))` computes a quantity called
  per-sample-gradients that stock PyTorch cannot efficiently compute today.

## Why composable function transforms?

There are a number of use cases that are tricky to do in PyTorch today:

- computing per-sample-gradients (or other per-sample quantities)
- running ensembles of models on a single machine
- efficiently batching together tasks in the inner-loop of MAML
- efficiently computing Jacobians and Hessians
- efficiently computing batched Jacobians and Hessians

Composing {func}`vmap`, {func}`grad`, {func}`vjp`, and {func}`jvp` transforms
allows us to express the above without designing a separate subsystem for each.

## What are the transforms?

### {func}`grad` (gradient computation)

`grad(func)` is our gradient computation transform. It returns a new function
that computes the gradients of `func`. It assumes `func` returns a single-element
Tensor and by default it computes the gradients of the output of `func` w.r.t.
to the first input.

```python
import torch
from torch.func import grad
x = torch.randn([])
cos_x = grad(lambda x: torch.sin(x))(x)
assert torch.allclose(cos_x, x.cos())

# Second-order gradients
neg_sin_x = grad(grad(lambda x: torch.sin(x)))(x)
assert torch.allclose(neg_sin_x, -x.sin())
```

### {func}`vmap` (auto-vectorization)

Note: {func}`vmap` imposes restrictions on the code that it can be used on. For more
details, please see {ref}`ux-limitations`.

`vmap(func)(*inputs)` is a transform that adds a dimension to all Tensor
operations in `func`. `vmap(func)` returns a new function that maps `func`
over some dimension (default: 0) of each Tensor in inputs.

vmap is useful for hiding batch dimensions: one can write a function func that
runs on examples and then lift it to a function that can take batches of
examples with `vmap(func)`, leading to a simpler modeling experience:

```python
import torch
from torch.func import vmap
batch_size, feature_size = 3, 5
weights = torch.randn(feature_size, requires_grad=True)

def model(feature_vec):
    # Very simple linear model with activation
    assert feature_vec.dim() == 1
    return feature_vec.dot(weights).relu()

examples = torch.randn(batch_size, feature_size)
result = vmap(model)(examples)
```

When composed with {func}`grad`, {func}`vmap` can be used to compute per-sample-gradients:

```python
from torch.func import vmap
batch_size, feature_size = 3, 5

def model(weights,feature_vec):
    # Very simple linear model with activation
    assert feature_vec.dim() == 1
    return feature_vec.dot(weights).relu()

def compute_loss(weights, example, target):
    y = model(weights, example)
    return ((y - target) ** 2).mean()  # MSELoss

weights = torch.randn(feature_size, requires_grad=True)
examples = torch.randn(batch_size, feature_size)
targets = torch.randn(batch_size)
inputs = (weights,examples, targets)
grad_weight_per_example = vmap(grad(compute_loss), in_dims=(None, 0, 0))(*inputs)
```

### {func}`vjp` (vector-Jacobian product)

The {func}`vjp` transform applies `func` to `inputs` and returns a new function
that computes the vector-Jacobian product (vjp) given some `cotangents` Tensors.

```python
from torch.func import vjp

inputs = torch.randn(3)
func = torch.sin
cotangents = (torch.randn(3),)

outputs, vjp_fn = vjp(func, inputs); vjps = vjp_fn(*cotangents)
```

### {func}`jvp` (Jacobian-vector product)

The {func}`jvp` transforms computes Jacobian-vector-products and is also known as
"forward-mode AD". It is not a higher-order function unlike most other transforms,
but it returns the outputs of `func(inputs)` as well as the jvps.

```python
from torch.func import jvp
x = torch.randn(5)
y = torch.randn(5)
f = lambda x, y: (x * y)
_, out_tangent = jvp(f, (x, y), (torch.ones(5), torch.ones(5)))
assert torch.allclose(out_tangent, x + y)
```

### {func}`jacrev`, {func}`jacfwd`, and {func}`hessian`

The {func}`jacrev` transform returns a new function that takes in `x` and returns
the Jacobian of the function with respect to `x` using reverse-mode AD.

```python
from torch.func import jacrev
x = torch.randn(5)
jacobian = jacrev(torch.sin)(x)
expected = torch.diag(torch.cos(x))
assert torch.allclose(jacobian, expected)
```

{func}`jacrev` can be composed with {func}`vmap` to produce batched jacobians:

```python
x = torch.randn(64, 5)
jacobian = vmap(jacrev(torch.sin))(x)
assert jacobian.shape == (64, 5, 5)
```

{func}`jacfwd` is a drop-in replacement for jacrev that computes Jacobians using
forward-mode AD:

```python
from torch.func import jacfwd
x = torch.randn(5)
jacobian = jacfwd(torch.sin)(x)
expected = torch.diag(torch.cos(x))
assert torch.allclose(jacobian, expected)
```

Composing {func}`jacrev` with itself or {func}`jacfwd` can produce hessians:

```python
def f(x):
    return x.sin().sum()

x = torch.randn(5)
hessian0 = jacrev(jacrev(f))(x)
hessian1 = jacfwd(jacrev(f))(x)
```

{func}`hessian` is a convenience function that combines jacfwd and jacrev:

```python
from torch.func import hessian

def f(x):
    return x.sin().sum()

x = torch.randn(5)
hess = hessian(f)(x)
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

- **File Documentation**: `func.whirlwind_tour.md_docs.md`
- **Keyword Index**: `func.whirlwind_tour.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
