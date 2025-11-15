# Documentation: `torch/fx/passes/operator_support.py`

## File Metadata

- **Path**: `torch/fx/passes/operator_support.py`
- **Size**: 7,632 bytes (7.45 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import abc
import typing as t

import torch
import torch.fx
from torch.fx._compatibility import compatibility

from .shape_prop import TensorMetadata
from .tools_common import CALLABLE_NODE_OPS, get_node_target


__all__ = [
    "OperatorSupportBase",
    "OperatorSupport",
    "create_op_support",
    "chain",
    "OpSupports",
    "any_chain",
]

# fx.Node.target typename, as returned by `get_node_target()`
TargetTypeName = str

# Arguments' dtypes for a given node, see `OperatorSupport`
SupportedArgumentDTypes = t.Optional[
    tuple[
        t.Sequence[t.Sequence[torch.dtype]],
        dict[str, t.Sequence[torch.dtype]],
    ]
]

SupportDict = t.Mapping[TargetTypeName, SupportedArgumentDTypes]


@compatibility(is_backward_compatible=False)
class OperatorSupportBase(abc.ABC):
    """Interface for determining if a fx.Node is supported by a backend"""

    @abc.abstractmethod
    def is_node_supported(
        self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        raise NotImplementedError


@compatibility(is_backward_compatible=False)
class OperatorSupport(OperatorSupportBase):
    """
    `_support_dict` maps node.target typename to supported inputs dtypes.

    node.target typename is retrieved using helper function `get_node_target()`

    If supported inputs dtypes is None, it means any dtype is supported, else
    we should see a tuple like (([dtypes], ...), {"name":[dtypes], ...}).

    The first tuple ([dtypes], ...) indicates what dtypes are supported for
    inputs in node.args and the second dict {"name": [dtypes], ...} indicates
    what dtypes are supported for inputs in node.kwargs.

    For inputs in args, if we don't want to check it, we can put None there,
    e.g. (None, [torch.float]) indicates that we don't care about the type of
    the first input in args. And for inputs in kwargs, if not listed, will not
    be checked.
    """

    _support_dict: SupportDict

    def __init__(self, support_dict: t.Optional[SupportDict] = None):
        self._support_dict = support_dict or {}

    def is_node_supported(
        self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        """
        Args:
            `submodules`: mapping from module name to the module. This can be
                          retrieved by calling model.named_modules().

            `node`: a Fx node that we want to determine whether it's supported.

        Returns:
            `is_supported`: whether the arg `node` is supported.
        """
        if node.op not in CALLABLE_NODE_OPS:
            return True

        target = get_node_target(submodules, node)

        # Target not found in _support_dict meaning that we don't support this op at all
        if target not in self._support_dict:
            return False

        # The rule for target is None meaning that we accept any dtype
        if self._support_dict[target] is None:
            return True

        args_dtypes, kwargs_dtypes = self._support_dict[target]  # type: ignore[misc]

        # Check args dtypes
        for i, dtypes in enumerate(args_dtypes):
            if len(node.args) <= i:
                break

            # None indicates we don't care about the dtype of args[i]
            if dtypes is None:
                continue

            # If arg is not a node then we don't check it
            if not isinstance(node.args[i], torch.fx.Node):
                continue

            arg_dtype = _get_arg_dtype(node.args[i])  # type: ignore[arg-type]
            if arg_dtype not in dtypes:
                return False

        # Check kwargs dtypes
        for k, dtypes in kwargs_dtypes.items():
            if k not in node.kwargs:
                continue

            # If arg is not a node then we don't check it
            if not isinstance(node.kwargs[k], torch.fx.Node):
                continue

            kwarg_dtype = _get_arg_dtype(node.kwargs[k])  # type: ignore[arg-type]
            if kwarg_dtype not in dtypes:
                return False

        return True


# ======================================================================
# Functional interfaces and utils for defining basic operator support logic
# and composing them into more complex ones
# ======================================================================

IsNodeSupported = t.Callable[[t.Mapping[str, torch.nn.Module], torch.fx.Node], bool]


@compatibility(is_backward_compatible=False)
def create_op_support(is_node_supported: IsNodeSupported) -> OperatorSupportBase:
    """Wraps a `IsNodeSupported` function into an `OperatorSupportBase` instance

    `IsNodeSupported` has the same call signature as
    `OperatorSupportBase.is_node_supported`
    """

    class FunctionalOperatorSupport(OperatorSupportBase):
        def is_node_supported(
            self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node
        ) -> bool:
            return is_node_supported(submodules, node)

    return FunctionalOperatorSupport()


@compatibility(is_backward_compatible=False)
def chain(*op_support: OperatorSupportBase) -> OperatorSupportBase:
    """Combines a sequence of `OperatorSupportBase` instances to form a single `OperatorSupportBase`
    instance by evaluating each input `OperatorSupportBase` instance, and returns False if
    any of it reports False.
    """

    def _chain(submods, node) -> bool:
        return all(x.is_node_supported(submods, node) for x in op_support)

    return create_op_support(_chain)


@compatibility(is_backward_compatible=False)
def any_chain(*op_support: OperatorSupportBase) -> OperatorSupportBase:
    """Combines a sequence of `OperatorSupportBase` instances to form a single `OperatorSupportBase`
    instance by evaluating each input `OperatorSupportBase` instance, and returns True if
    any of it reports True.
    """

    def _any_chain(submods, node) -> bool:
        return any(x.is_node_supported(submods, node) for x in op_support)

    return create_op_support(_any_chain)


@compatibility(is_backward_compatible=False)
class OpSupports:
    """A set of atomic `OperatorSupportBase` instances that can be combined together
    to form more complex operator support logic.
    """

    @classmethod
    def decline_if_input_dtype(cls, dtype: torch.dtype) -> OperatorSupportBase:
        """Report a node as non-supported, if any of its arguments is of dtype"""

        def _decline_if_input_dtype(
            submodules: t.Mapping[str, torch.nn.Module],
            node: torch.fx.Node,
        ) -> bool:
            for arg in node.all_input_nodes:
                arg_dtype = _get_arg_dtype(arg)
                if arg_dtype == dtype:
                    return False
            return True

        return create_op_support(_decline_if_input_dtype)

    @classmethod
    def decline_if_node_in_names(cls, disallow_set: set[str]) -> OperatorSupportBase:
        """
        If a node has a name that is in the disallow set, reported it as non-supported.
        """

        def _decline_if_node_in_names(
            submodules: t.Mapping[str, torch.nn.Module],
            node: torch.fx.Node,
        ) -> bool:
            return node.name not in disallow_set

        return create_op_support(_decline_if_node_in_names)


def _get_arg_dtype(arg: torch.fx.Node) -> t.Any:
    assert isinstance(arg, torch.fx.Node)
    tensor_meta = arg.meta.get("tensor_meta")  # type: ignore[union-attr]
    dtype = (
        tensor_meta.dtype
        if isinstance(tensor_meta, TensorMetadata)
        else arg.meta["type"]
    )
    return dtype

```



## High-Level Overview

"""Interface for determining if a fx.Node is supported by a backend"""    @abc.abstractmethod    def is_node_supported(        self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node    ) -> bool:        raise NotImplementedError@compatibility(is_backward_compatible=False)class OperatorSupport(OperatorSupportBase):

This Python file contains 4 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `OperatorSupportBase`, `OperatorSupport`, `FunctionalOperatorSupport`, `OpSupports`

**Functions defined**: `is_node_supported`, `__init__`, `is_node_supported`, `create_op_support`, `is_node_supported`, `chain`, `_chain`, `any_chain`, `_any_chain`, `decline_if_input_dtype`, `_decline_if_input_dtype`, `decline_if_node_in_names`, `_decline_if_node_in_names`, `_get_arg_dtype`

**Key imports**: abc, typing as t, torch, torch.fx, compatibility, TensorMetadata, CALLABLE_NODE_OPS, get_node_target


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `abc`
- `typing as t`
- `torch`
- `torch.fx`
- `torch.fx._compatibility`: compatibility
- `.shape_prop`: TensorMetadata
- `.tools_common`: CALLABLE_NODE_OPS, get_node_target


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
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

Files in the same folder (`torch/fx/passes`):

- [`reinplace.py_docs.md`](./reinplace.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`graph_drawer.py_docs.md`](./graph_drawer.py_docs.md)
- [`shape_prop.py_docs.md`](./shape_prop.py_docs.md)
- [`split_utils.py_docs.md`](./split_utils.py_docs.md)
- [`runtime_assert.py_docs.md`](./runtime_assert.py_docs.md)
- [`splitter_base.py_docs.md`](./splitter_base.py_docs.md)
- [`graph_transform_observer.py_docs.md`](./graph_transform_observer.py_docs.md)
- [`fake_tensor_prop.py_docs.md`](./fake_tensor_prop.py_docs.md)


## Cross-References

- **File Documentation**: `operator_support.py_docs.md`
- **Keyword Index**: `operator_support.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
