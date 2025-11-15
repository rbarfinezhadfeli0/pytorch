# Documentation: `docs/torch/fx/experimental/shape_inference/infer_shape.py_docs.md`

## File Metadata

- **Path**: `docs/torch/fx/experimental/shape_inference/infer_shape.py_docs.md`
- **Size**: 5,584 bytes (5.45 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/fx/experimental/shape_inference/infer_shape.py`

## File Metadata

- **Path**: `torch/fx/experimental/shape_inference/infer_shape.py`
- **Size**: 3,224 bytes (3.15 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import copy
from collections import defaultdict

import torch
from torch._dynamo.source import LocalSource
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.shape_inference.infer_symbol_values import (
    infer_symbol_values,
)
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.utils import _pytree


"""
This is the function that runs shape inference. It will modify the input graph module so that shapes are annotated.
"""


def infer_shape(gm, input_tensors):
    # Prepare environments
    shape_env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True)

    flatten_inputs, spec = _pytree.tree_flatten(input_tensors)
    dim_count = 1
    for input_tensor in flatten_inputs:
        dim_count += input_tensor.dim() - 1

    sample = {f"s{i}": 2 for i in range(dim_count)}
    init_symints = [
        mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC)
        for k, v in sample.items()
    ]
    symints = copy.deepcopy(init_symints)
    symbol_to_idx_dict = {f"s{i}": i for i in range(dim_count)}
    padding_constraints = defaultdict(list)  # type: ignore[var-annotated]

    complete_flag = False
    allowed_try_times = dim_count * 2

    while not complete_flag and allowed_try_times > 0:
        # Create symbolic input tensors
        with fake_mode:
            sym_tensors = []
            i = 1
            for input_tensor in flatten_inputs:
                curr_dim = input_tensor.dim()
                desired_size = [symints[0]] + [
                    symints[ii] for ii in range(i, i + curr_dim - 1)
                ]
                sym_tensor = torch.randn(desired_size)
                sym_tensors.append(sym_tensor)
                i += curr_dim - 1
            sym_tensors = _pytree.tree_unflatten(sym_tensors, spec)
        try:
            with fake_mode:
                make_fx(
                    gm,
                    tracing_mode="symbolic",
                    _allow_non_fake_inputs=True,
                    pre_dispatch=True,
                    _allow_fake_constant=True,
                )(*sym_tensors)
            complete_flag = True
            return (gm, input_tensors, fake_mode, symints[0])
        except RuntimeError as e:
            if e:
                infer_symbol_values(
                    symints,
                    init_symints,
                    symbol_to_idx_dict,
                    padding_constraints,
                    str(e),
                )
                allowed_try_times -= 1
        except ValueError as e:
            if e:
                infer_symbol_values(
                    symints,
                    init_symints,
                    symbol_to_idx_dict,
                    padding_constraints,
                    str(e),
                )
                allowed_try_times -= 1


def mksym(shape_env, value, source, dynamic_dim):
    return shape_env.create_symintnode(
        shape_env.create_symbol(
            value,
            source=source,
            dynamic_dim=dynamic_dim,
        ),
        hint=value,
        source=source,
    )

```



## High-Level Overview

"""This is the function that runs shape inference. It will modify the input graph module so that shapes are annotated.

This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `infer_shape`, `mksym`

**Key imports**: copy, defaultdict, torch, LocalSource, FakeTensorMode, make_fx, DimDynamic, ShapeEnv, _pytree


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx/experimental/shape_inference`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `collections`: defaultdict
- `torch`
- `torch._dynamo.source`: LocalSource
- `torch._subclasses`: FakeTensorMode
- `torch.fx.experimental.proxy_tensor`: make_fx
- `torch.fx.experimental.symbolic_shapes`: DimDynamic, ShapeEnv
- `torch.utils`: _pytree


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

Files in the same folder (`torch/fx/experimental/shape_inference`):

- [`infer_symbol_values.py_docs.md`](./infer_symbol_values.py_docs.md)


## Cross-References

- **File Documentation**: `infer_shape.py_docs.md`
- **Keyword Index**: `infer_shape.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/fx/experimental/shape_inference`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/fx/experimental/shape_inference`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/fx/experimental/shape_inference`):

- [`infer_symbol_values.py_docs.md_docs.md`](./infer_symbol_values.py_docs.md_docs.md)
- [`infer_shape.py_kw.md_docs.md`](./infer_shape.py_kw.md_docs.md)
- [`infer_symbol_values.py_kw.md_docs.md`](./infer_symbol_values.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `infer_shape.py_docs.md_docs.md`
- **Keyword Index**: `infer_shape.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
