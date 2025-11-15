# Documentation: _ordered_set.py

## File Metadata
- **Path**: `torch/utils/_ordered_set.py`
- **Size**: 5658 bytes
- **Lines**: 179
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
from __future__ import annotations

from collections.abc import (
    Hashable,
    Iterable,
    Iterator,
    MutableSet,
    Reversible,
    Set as AbstractSet,
)
from typing import Any, cast, TypeVar


T = TypeVar("T", bound=Hashable)
T_co = TypeVar("T_co", bound=Hashable, covariant=True)

__all__ = ["OrderedSet"]


class OrderedSet(MutableSet[T], Reversible[T]):
    """
    Insertion ordered set, similar to OrderedDict.
    """

    __slots__ = ("_dict",)

    def __init__(self, iterable: Iterable[T] | None = None) -> None:
        self._dict = dict.fromkeys(iterable, None) if iterable is not None else {}

    @staticmethod
    def _from_dict(dict_inp: dict[T, None]) -> OrderedSet[T]:
        s: OrderedSet[T] = OrderedSet()
        s._dict = dict_inp
        return s

    #
    # Required overridden abstract methods
    #
    def __contains__(self, elem: object) -> bool:
        return elem in self._dict

    def __iter__(self) -> Iterator[T]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __reversed__(self) -> Iterator[T]:
        return reversed(self._dict)

    def add(self, elem: T) -> None:
        self._dict[elem] = None

    def discard(self, elem: T) -> None:
        self._dict.pop(elem, None)

    def clear(self) -> None:
        # overridden because MutableSet impl is slow
        self._dict.clear()

    # Unimplemented set() methods in _collections_abc.MutableSet

    @classmethod
    def _wrap_iter_in_set(cls, other: Any) -> Any:
        """
        Wrap non-Set Iterables in OrderedSets

        Some of the magic methods are more strict on input types than
        the public apis, so we need to wrap inputs in sets.
        """

        if not isinstance(other, AbstractSet) and isinstance(other, Iterable):
            return cls(other)
        else:
            return other

    def pop(self) -> T:
        if not self:
            raise KeyError("pop from an empty set")
        # pyrefly: ignore [bad-return]
        return self._dict.popitem()[0]

    def copy(self) -> OrderedSet[T]:
        return OrderedSet._from_dict(self._dict.copy())

    def difference(self, *others: Iterable[T]) -> OrderedSet[T]:
        res = self.copy()
        res.difference_update(*others)
        return res

    def difference_update(self, *others: Iterable[T]) -> None:
        for other in others:
            self -= other  # type: ignore[arg-type]

    def update(self, *others: Iterable[T]) -> None:
        for other in others:
            self |= other

    def intersection(self, *others: Iterable[T]) -> OrderedSet[T]:
        res = self.copy()
        for other in others:
            if other is not self:
                res &= other  # type: ignore[arg-type]
        return res

    def intersection_update(self, *others: Iterable[T]) -> None:
        for other in others:
            self &= other  # type: ignore[arg-type]

    def issubset(self, other: Iterable[T]) -> bool:
        return self <= self._wrap_iter_in_set(other)

    def issuperset(self, other: Iterable[T]) -> bool:
        return self >= self._wrap_iter_in_set(other)

    def symmetric_difference(self, other: Iterable[T]) -> OrderedSet[T]:
        return self ^ other  # type: ignore[operator]

    def symmetric_difference_update(self, other: Iterable[T]) -> None:
        self ^= other  # type: ignore[arg-type]

    def union(self, *others: Iterable[T]) -> OrderedSet[T]:
        res = self.copy()
        for other in others:
            if other is self:
                continue
            res |= other
        return res

    # Specify here for correct type inference, otherwise would
    # return AbstractSet[T]
    def __sub__(self, other: AbstractSet[T_co]) -> OrderedSet[T]:
        # following cpython set impl optimization
        if isinstance(other, OrderedSet) and (len(self) * 4) > len(other):
            out = self.copy()
            out -= other
            return out
        return cast(OrderedSet[T], super().__sub__(other))

    def __ior__(self, other: Iterable[T]) -> OrderedSet[T]:  # type: ignore[misc, override]   # noqa: PYI034
        if isinstance(other, OrderedSet):
            self._dict.update(other._dict)
            return self
        return super().__ior__(other)  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OrderedSet):
            return self._dict == other._dict
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        if isinstance(other, OrderedSet):
            return self._dict != other._dict
        return super().__ne__(other)

    def __or__(self, other: AbstractSet[T_co]) -> OrderedSet[T]:
        return cast(OrderedSet[T], super().__or__(other))

    def __and__(self, other: AbstractSet[T_co]) -> OrderedSet[T]:
        # MutableSet impl will iterate over other, iter over smaller of two sets
        if isinstance(other, OrderedSet) and len(self) < len(other):
            # pyrefly: ignore [unsupported-operation, bad-return]
            return other & self
        return cast(OrderedSet[T], super().__and__(other))

    def __xor__(self, other: AbstractSet[T_co]) -> OrderedSet[T]:
        return cast(OrderedSet[T], super().__xor__(other))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({list(self)})"

    def __getstate__(self) -> list[T]:
        return list(self._dict.keys())

    def __setstate__(self, state: list[T]) -> None:
        self._dict = dict.fromkeys(state, None)

    def __reduce__(self) -> tuple[type[OrderedSet[T]], tuple[list[T]]]:
        return (OrderedSet, (list(self),))

```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 1 class(es): OrderedSet

### Functions
This file defines 33 function(s): __init__, _from_dict, __contains__, __iter__, __len__, __reversed__, add, discard, clear, _wrap_iter_in_set, pop, copy, difference, difference_update, update, intersection, intersection_update, issubset, issuperset, symmetric_difference, symmetric_difference_update, union, __sub__, __ior__, __eq__, __ne__, __or__, __and__, __xor__, __repr__


## Key Components

The file contains 565 words across 179 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 5658 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
