# Documentation: `docs/torch/testing/_internal/optests/make_fx.py_docs.md`

## File Metadata

- **Path**: `docs/torch/testing/_internal/optests/make_fx.py_docs.md`
- **Size**: 5,682 bytes (5.55 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/testing/_internal/optests/make_fx.py`

## File Metadata

- **Path**: `torch/testing/_internal/optests/make_fx.py`
- **Size**: 3,265 bytes (3.19 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
# mypy: ignore-errors

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._utils import wrapper_set_seed
import torch.utils._pytree as pytree


def make_fx_check(
    func,
    args,
    kwargs,
    tracing_mode,
    assert_close=torch.testing.assert_close,
    randomize_data=False,
):
    f, *new_args = handle_sizes_for_dynamic_shapes(func, args, kwargs)

    def run(f, *args, **kwargs):
        return wrapper_set_seed(f, *args, **kwargs)

    traced_f = make_fx(f, tracing_mode=tracing_mode)(*new_args)

    msg = (
        "op(*args, **kwargs) and make_fx(op)(*args, **kwargs) produced different "
        "values. This could mean that your abstract impls (meta/FakeTensor impls) "
        "are incorrect, that your operator is not completely traceable (e.g., "
        "it relies on some global state), or that there is a bug in make_fx. "
        "Note that if you passed a python function (and not an operator) to "
        "make_fx_check, it is still possible that the python function will still "
        "work with torch.compile because it handles capturing pieces of "
        "your python code to compile."
    )

    # Randomize the data and run the traced graph with it, to catch bugs
    # where we may have baked in Tensor data into the trace.
    # This is not guaranteed to succeed, because `f` might have preconditions
    # on the values of the inputs, so we just ignore if we used
    # random data and it fails.
    if randomize_data:
        new_args = randomize(new_args)
    try:
        expected = run(f, *new_args)
    except Exception:
        if randomize_data:
            return
        raise
    result = run(traced_f, *new_args)
    assert_close(result, expected, msg=msg)


# Arguably we should make make_fx promote torch.Size() objects to symbolic shapes.
# Absent that, here is our strategy:
#
# If any argument is a torch.Size(), maybe get dynamic shapes for it by:
# - Create a temporary Tensor whose size is the torch.Size() we want. Note that
#   we use an expanded Tensor as we cannot pass "meta" Tensors to make_fx.
# - Pass it to make_fx such that it is converted to a proxy Tensor
# - Unpack the size in the wrapper to get a torch.Size with dynamic shapes (in
#   symbolic mode, a no-op otherwise)
def handle_sizes_for_dynamic_shapes(func, args, kwargs):
    def f(args, kwargs, extra_args, extra_kwargs):
        if extra_args:
            for i, t in extra_args:
                args[i] = t.size()
        if extra_kwargs:
            for k, t in extra_kwargs.items():
                kwargs[k] = t.size()

        return func(*args, **kwargs)

    extra_args = []
    extra_kwargs = {}
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Size):
            extra_args.append((i, torch.empty(arg, device="cpu")))
    for key, value in kwargs.items():
        if isinstance(value, torch.Size):
            extra_kwargs[key] = torch.empty(value, device="cpu")

    return f, args, kwargs, extra_args, extra_kwargs


def randomize(args):
    def transform(x):
        if not x.dtype.is_floating_point:
            return x
        return x.detach().clone().uniform_(0, 1).requires_grad_(x.requires_grad)
    return pytree.tree_map_only(torch.Tensor, transform, args)

```



## High-Level Overview


This Python file contains 0 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `make_fx_check`, `run`, `handle_sizes_for_dynamic_shapes`, `f`, `randomize`, `transform`

**Key imports**: torch, make_fx, wrapper_set_seed, torch.utils._pytree as pytree


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/testing/_internal/optests`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.fx.experimental.proxy_tensor`: make_fx
- `torch.testing._utils`: wrapper_set_seed
- `torch.utils._pytree as pytree`


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

This is a test file. Run it with:

```bash
python torch/testing/_internal/optests/make_fx.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/testing/_internal/optests`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`autograd_registration.py_docs.md`](./autograd_registration.py_docs.md)
- [`generate_tests.py_docs.md`](./generate_tests.py_docs.md)
- [`fake_tensor.py_docs.md`](./fake_tensor.py_docs.md)
- [`aot_autograd.py_docs.md`](./aot_autograd.py_docs.md)


## Cross-References

- **File Documentation**: `make_fx.py_docs.md`
- **Keyword Index**: `make_fx.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/testing/_internal/optests`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/testing/_internal/optests`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

This is a test file. Run it with:

```bash
python docs/torch/testing/_internal/optests/make_fx.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/testing/_internal/optests`):

- [`generate_tests.py_kw.md_docs.md`](./generate_tests.py_kw.md_docs.md)
- [`aot_autograd.py_kw.md_docs.md`](./aot_autograd.py_kw.md_docs.md)
- [`make_fx.py_kw.md_docs.md`](./make_fx.py_kw.md_docs.md)
- [`generate_tests.py_docs.md_docs.md`](./generate_tests.py_docs.md_docs.md)
- [`fake_tensor.py_docs.md_docs.md`](./fake_tensor.py_docs.md_docs.md)
- [`autograd_registration.py_kw.md_docs.md`](./autograd_registration.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`aot_autograd.py_docs.md_docs.md`](./aot_autograd.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `make_fx.py_docs.md_docs.md`
- **Keyword Index**: `make_fx.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
