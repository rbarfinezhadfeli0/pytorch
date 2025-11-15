# Documentation: `docs/torch/fx/experimental/normalize.py_docs.md`

## File Metadata

- **Path**: `docs/torch/fx/experimental/normalize.py_docs.md`
- **Size**: 8,967 bytes (8.76 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/fx/experimental/normalize.py`

## File Metadata

- **Path**: `torch/fx/experimental/normalize.py`
- **Size**: 5,491 bytes (5.36 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import operator
from collections.abc import Callable
from typing import Any, Optional

import torch
import torch.fx
import torch.fx as fx
from torch.fx import Proxy, Transformer
from torch.fx.node import Argument, map_aggregate, Node, Target
from torch.fx.operator_schemas import (
    create_type_hint,
    normalize_function,
    normalize_module,
)

from .schema_type_annotation import AnnotateTypesWithSchema


class NormalizeArgs(Transformer):
    """
    Normalize arguments to Python targets. This means that
    `args/kwargs` will be matched up to the module/functional's
    signature and rewritten to exclusively kwargs in positional order
    if `normalize_to_only_use_kwargs` is true. Also populates default
    values. Does not support positional-only parameters or varargs
    parameters (*args, **kwargs).

    If the nodes have 'type' metadata, it will use it to disambiguate
    overloads. Otherwise, it will throw an error.

    Example usage:
        m = torchvision.models.resnet18()
        traced = torch.fx.symbolic_trace(m)
        traced = NormalizeArgs(traced).transform()
    """

    def __init__(
        self, module: torch.fx.GraphModule, normalize_to_only_use_kwargs: bool = True
    ):
        super().__init__(module)
        self.node_map: dict[Proxy, Node] = {}
        self.normalize_to_only_use_kwargs = normalize_to_only_use_kwargs

    def run_node(self, n: Node) -> Any:
        args, kwargs = self.fetch_args_kwargs_from_env(n)

        def get_type(arg):
            if isinstance(arg, fx.Node):
                return n.meta.get("type")
            return type(arg)

        arg_types = map_aggregate(n.args, get_type)
        assert isinstance(arg_types, tuple)
        arg_types = tuple(create_type_hint(i) for i in arg_types)
        kwarg_types = {k: get_type(v) for k, v in kwargs.items()}
        if n.op == "call_function":
            out = self.call_function(n.target, args, kwargs, arg_types, kwarg_types)
        else:
            out = super().run_node(n)
        if n.op != "output":
            self.node_map[out] = n
            out.node.meta = n.meta
            out.node.type = n.type
        return out

    def call_function(
        self,
        target: Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Any],
        arg_types: Optional[tuple[Any, ...]] = None,
        kwarg_types: Optional[dict[str, Any]] = None,
    ):
        assert callable(target)
        new_args_and_kwargs = normalize_function(
            target,
            args,  # type: ignore[arg-type]
            kwargs,
            arg_types,  # type: ignore[arg-type]
            kwarg_types,
            self.normalize_to_only_use_kwargs,
        )
        if new_args_and_kwargs:
            new_args, new_kwargs = new_args_and_kwargs
            return self.tracer.create_proxy(
                "call_function", target, new_args, new_kwargs
            )
        else:
            return super().call_function(target, args, kwargs)

    def call_module(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ):
        assert isinstance(target, str)
        new_args_and_kwargs = normalize_module(
            self.module,
            target,
            args,  # type: ignore[arg-type]
            kwargs,
            self.normalize_to_only_use_kwargs,
        )
        if new_args_and_kwargs:
            new_args, new_kwargs = new_args_and_kwargs
            return super().call_module(target, new_args, new_kwargs)
        else:
            return super().call_module(target, args, kwargs)


class NormalizeOperators(AnnotateTypesWithSchema):
    """
    Normalize callsites that are different ways of "spelling" the same
    invocation into a single, canonical call. Currently supports:

    1. Normalize operators (e.g. operator.add) to the `torch` ops they
       ultimately invoke (e.g. torch.add) when it is possible to statically
       reason that

    Example usage:

        m = torchvision.models.resnet18()

        traced = torch.fx.symbolic_trace(m)

        traced = NormalizeOperators(traced).transform()
    """

    binary_magic_method_remap: dict[
        Callable[[Any, Any], Any], Callable[[Any, Any], Any]
    ] = {
        torch.add: operator.add,
        torch.mul: operator.mul,
        torch.sub: operator.sub,
        torch.div: operator.truediv,
        torch.floor_divide: operator.floordiv,
        torch.remainder: operator.mod,
        torch.eq: operator.eq,
        torch.ne: operator.ne,
        torch.lt: operator.lt,
        torch.le: operator.le,
        torch.gt: operator.gt,
        torch.ge: operator.ge,
    }

    def call_function(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ):
        # Normalize operators according to the magic methods implemented on tensors here:
        # https://github.com/pytorch/pytorch/blob/28c5d90b679c6b38bf4183ec99f16d933c2f1bcd/tools/autograd/templates/python_variable_methods.cpp#L1137 # noqa: B950

        assert callable(target)

        if target in self.binary_magic_method_remap:
            if len(args) != 2:
                return super().call_function(target, args, kwargs)
            lhs, rhs = args

            return super().call_function(
                target=self.binary_magic_method_remap[target],
                args=(lhs, rhs),
                kwargs={},
            )

        return super().call_function(target, args, kwargs)

```



## High-Level Overview

"""    Normalize arguments to Python targets. This means that    `args/kwargs` will be matched up to the module/functional's    signature and rewritten to exclusively kwargs in positional order    if `normalize_to_only_use_kwargs` is true. Also populates default    values. Does not support positional-only parameters or varargs    parameters (*args, **kwargs).    If the nodes have 'type' metadata, it will use it to disambiguate    overloads. Otherwise, it will throw an error.    Example usage:        m = torchvision.models.resnet18()        traced = torch.fx.symbolic_trace(m)        traced = NormalizeArgs(traced).transform()

This Python file contains 2 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `NormalizeArgs`, `NormalizeOperators`

**Functions defined**: `__init__`, `run_node`, `get_type`, `call_function`, `call_module`, `call_function`

**Key imports**: operator, Callable, Any, Optional, torch, torch.fx, torch.fx as fx, Proxy, Transformer, Argument, map_aggregate, Node, Target, AnnotateTypesWithSchema


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx/experimental`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `operator`
- `collections.abc`: Callable
- `typing`: Any, Optional
- `torch`
- `torch.fx`
- `torch.fx as fx`
- `torch.fx.node`: Argument, map_aggregate, Node, Target
- `.schema_type_annotation`: AnnotateTypesWithSchema


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`torch/fx/experimental`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`graph_gradual_typechecker.py_docs.md`](./graph_gradual_typechecker.py_docs.md)
- [`validator.py_docs.md`](./validator.py_docs.md)
- [`accelerator_partitioner.py_docs.md`](./accelerator_partitioner.py_docs.md)
- [`unify_refinements.py_docs.md`](./unify_refinements.py_docs.md)
- [`debug.py_docs.md`](./debug.py_docs.md)
- [`const_fold.py_docs.md`](./const_fold.py_docs.md)
- [`merge_matmul.py_docs.md`](./merge_matmul.py_docs.md)
- [`rewriter.py_docs.md`](./rewriter.py_docs.md)
- [`partitioner_utils.py_docs.md`](./partitioner_utils.py_docs.md)


## Cross-References

- **File Documentation**: `normalize.py_docs.md`
- **Keyword Index**: `normalize.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/fx/experimental`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/fx/experimental`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`docs/torch/fx/experimental`):

- [`schema_type_annotation.py_kw.md_docs.md`](./schema_type_annotation.py_kw.md_docs.md)
- [`proxy_tensor.py_kw.md_docs.md`](./proxy_tensor.py_kw.md_docs.md)
- [`partitioner_utils.py_docs.md_docs.md`](./partitioner_utils.py_docs.md_docs.md)
- [`recording.py_docs.md_docs.md`](./recording.py_docs.md_docs.md)
- [`validator.py_kw.md_docs.md`](./validator.py_kw.md_docs.md)
- [`recording.py_kw.md_docs.md`](./recording.py_kw.md_docs.md)
- [`accelerator_partitioner.py_kw.md_docs.md`](./accelerator_partitioner.py_kw.md_docs.md)
- [`optimization.py_kw.md_docs.md`](./optimization.py_kw.md_docs.md)
- [`graph_gradual_typechecker.py_docs.md_docs.md`](./graph_gradual_typechecker.py_docs.md_docs.md)
- [`_dynamism.py_kw.md_docs.md`](./_dynamism.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `normalize.py_docs.md_docs.md`
- **Keyword Index**: `normalize.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
