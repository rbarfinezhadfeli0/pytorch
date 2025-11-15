# Documentation: `torch/fx/tensor_type.py`

## File Metadata

- **Path**: `torch/fx/tensor_type.py`
- **Size**: 3,009 bytes (2.94 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
from torch.fx.experimental.unification import Var  # type: ignore[attr-defined]

from ._compatibility import compatibility


@compatibility(is_backward_compatible=False)
class TensorType:
    """
    TensorType defines a type for tensors, which consists of a list of dimensions.
    Example:
        class M(torch.nn.Module):
            def forward(self, x:TensorType((1,2,3, Dyn)), y:TensorType((1,2,3, Dyn))):
                return torch.add(x, y)
    """

    def __init__(self, dim):
        self.__origin__ = TensorType
        self.__args__ = dim

    def __repr__(self):
        return f"TensorType[{self.__args__}]"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return list(self.__args__) == list(other.__args__)
        else:
            return False

    @staticmethod
    def __class_getitem__(*args):
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0]
        return TensorType(tuple(args))


class _DynType:
    """
    _DynType defines a type which stands for the absence of type information.
    """

    def __init__(self) -> None:
        self.__name__ = "_DynType"

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __str__(self):
        return "Dyn"

    def __repr__(self):
        return "Dyn"


Dyn = _DynType()


@compatibility(is_backward_compatible=False)
def is_consistent(t1, t2):
    """
    A binary relation denoted by ~ that determines if t1 is consistent with t2.
    The relation is reflexive, symmetric but not transitive.
    returns True if t1 and t2 are consistent and False otherwise.
    Example:
        Dyn ~ TensorType((1,2,3))
        int ~ Dyn
        int ~ int
        TensorType((1,Dyn,3)) ~ TensorType((1,2,3))
    """

    if t1 == t2:
        return True

    if t1 == Dyn or t2 == Dyn or isinstance(t1, Var) or isinstance(t2, Var):
        return True

    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        return len(t1.__args__) == len(t2.__args__) and all(
            is_consistent(elem1, elem2)
            for elem1, elem2 in zip(t1.__args__, t2.__args__)
        )
    else:
        return False


@compatibility(is_backward_compatible=False)
def is_more_precise(t1, t2):
    """
    A binary relation denoted by <= that determines if t1 is more precise than t2.
    The relation is reflexive and transitive.
    returns True if t1 is more precise than t2 and False otherwise.
    Example:
        Dyn >= TensorType((1,2,3))
        int >= Dyn
        int >= int
        TensorType((1,Dyn,3)) <= TensorType((1,2,3))
    """
    if t1 == t2:
        return True

    if isinstance(t2, _DynType):
        return True

    if isinstance(t1, TensorType) and isinstance(t2, TensorType):
        return len(t1.__args__) == len(t2.__args__) and all(
            is_more_precise(elem1, elem2)
            for elem1, elem2 in zip(t1.__args__, t2.__args__)
        )

    else:
        return False

```



## High-Level Overview

"""    TensorType defines a type for tensors, which consists of a list of dimensions.    Example:        class M(torch.nn.Module):            def forward(self, x:TensorType((1,2,3, Dyn)), y:TensorType((1,2,3, Dyn))):                return torch.add(x, y)

This Python file contains 3 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TensorType`, `M`, `_DynType`

**Functions defined**: `forward`, `__init__`, `__repr__`, `__eq__`, `__class_getitem__`, `__init__`, `__eq__`, `__str__`, `__repr__`, `is_consistent`, `is_more_precise`

**Key imports**: Var  , compatibility


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/fx`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch.fx.experimental.unification`: Var  
- `._compatibility`: compatibility


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

Files in the same folder (`torch/fx`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`traceback.py_docs.md`](./traceback.py_docs.md)
- [`_symbolic_trace.py_docs.md`](./_symbolic_trace.py_docs.md)
- [`graph.py_docs.md`](./graph.py_docs.md)
- [`node.py_docs.md`](./node.py_docs.md)
- [`annotate.py_docs.md`](./annotate.py_docs.md)
- [`config.py_docs.md`](./config.py_docs.md)
- [`_utils.py_docs.md`](./_utils.py_docs.md)
- [`subgraph_rewriter.py_docs.md`](./subgraph_rewriter.py_docs.md)


## Cross-References

- **File Documentation**: `tensor_type.py_docs.md`
- **Keyword Index**: `tensor_type.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
