# Documentation: `test/jit/test_dataclasses.py`

## File Metadata

- **Path**: `test/jit/test_dataclasses.py`
- **Size**: 4,644 bytes (4.54 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

from dataclasses import dataclass, field, InitVar
from enum import Enum
from typing import List, Optional

from hypothesis import given, settings, strategies as st

import torch
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


# Example jittable dataclass
@dataclass(order=True)
class Point:
    x: float
    y: float
    norm: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.norm = (torch.tensor(self.x) ** 2 + torch.tensor(self.y) ** 2) ** 0.5


class MixupScheme(Enum):
    INPUT = ["input"]

    MANIFOLD = [
        "input",
        "before_fusion_projection",
        "after_fusion_projection",
        "after_classifier_projection",
    ]


@dataclass
class MixupParams:
    def __init__(self, alpha: float = 0.125, scheme: MixupScheme = MixupScheme.INPUT):
        self.alpha = alpha
        self.scheme = scheme


class MixupScheme2(Enum):
    A = 1
    B = 2


@dataclass
class MixupParams2:
    def __init__(self, alpha: float = 0.125, scheme: MixupScheme2 = MixupScheme2.A):
        self.alpha = alpha
        self.scheme = scheme


@dataclass
class MixupParams3:
    def __init__(self, alpha: float = 0.125, scheme: MixupScheme2 = MixupScheme2.A):
        self.alpha = alpha
        self.scheme = scheme


# Make sure the Meta internal tooling doesn't raise an overflow error
NonHugeFloats = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False)


class TestDataclasses(JitTestCase):
    @classmethod
    def tearDownClass(cls):
        torch._C._jit_clear_class_registry()

    def test_init_vars(self):
        @torch.jit.script
        @dataclass(order=True)
        class Point2:
            x: float
            y: float
            norm_p: InitVar[int] = 2
            norm: Optional[torch.Tensor] = None

            def __post_init__(self, norm_p: int):
                self.norm = (
                    torch.tensor(self.x) ** norm_p + torch.tensor(self.y) ** norm_p
                ) ** (1 / norm_p)

        def fn(x: float, y: float, p: int):
            pt = Point2(x, y, p)
            return pt.norm

        self.checkScript(fn, (1.0, 2.0, 3))

    # Sort of tests both __post_init__ and optional fields
    @settings(deadline=None)
    @given(NonHugeFloats, NonHugeFloats)
    def test__post_init__(self, x, y):
        P = torch.jit.script(Point)

        def fn(x: float, y: float):
            pt = P(x, y)
            return pt.norm

        self.checkScript(fn, [x, y])

    @settings(deadline=None)
    @given(
        st.tuples(NonHugeFloats, NonHugeFloats), st.tuples(NonHugeFloats, NonHugeFloats)
    )
    def test_comparators(self, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        P = torch.jit.script(Point)

        def compare(x1: float, y1: float, x2: float, y2: float):
            pt1 = P(x1, y1)
            pt2 = P(x2, y2)
            return (
                pt1 == pt2,
                # pt1 != pt2,   # TODO: Modify interpreter to auto-resolve (a != b) to not (a == b) when there's no __ne__
                pt1 < pt2,
                pt1 <= pt2,
                pt1 > pt2,
                pt1 >= pt2,
            )

        self.checkScript(compare, [x1, y1, x2, y2])

    def test_default_factories(self):
        @dataclass
        class Foo(object):
            x: List[int] = field(default_factory=list)

        with self.assertRaises(NotImplementedError):
            torch.jit.script(Foo)

            def fn():
                foo = Foo()
                return foo.x

            torch.jit.script(fn)()

    # The user should be able to write their own __eq__ implementation
    # without us overriding it.
    def test_custom__eq__(self):
        @torch.jit.script
        @dataclass
        class CustomEq:
            a: int
            b: int

            def __eq__(self, other: "CustomEq") -> bool:
                return self.a == other.a  # ignore the b field

        def fn(a: int, b1: int, b2: int):
            pt1 = CustomEq(a, b1)
            pt2 = CustomEq(a, b2)
            return pt1 == pt2

        self.checkScript(fn, [1, 2, 3])

    def test_no_source(self):
        with self.assertRaises(RuntimeError):
            # uses list in Enum is not supported
            torch.jit.script(MixupParams)

        torch.jit.script(MixupParams2)  # don't throw

    def test_use_unregistered_dataclass_raises(self):
        def f(a: MixupParams3):
            return 0

        with self.assertRaises(OSError):
            torch.jit.script(f)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview


This Python file contains 10 class(es) and 20 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Point`, `MixupScheme`, `MixupParams`, `MixupScheme2`, `MixupParams2`, `MixupParams3`, `TestDataclasses`, `Point2`, `Foo`, `CustomEq`

**Functions defined**: `__post_init__`, `__init__`, `__init__`, `__init__`, `tearDownClass`, `test_init_vars`, `__post_init__`, `fn`, `test__post_init__`, `fn`, `test_comparators`, `compare`, `test_default_factories`, `fn`, `test_custom__eq__`, `__eq__`, `fn`, `test_no_source`, `test_use_unregistered_dataclass_raises`, `f`

**Key imports**: dataclass, field, InitVar, Enum, List, Optional, given, settings, strategies as st, torch, raise_on_run_directly, JitTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `dataclasses`: dataclass, field, InitVar
- `enum`: Enum
- `typing`: List, Optional
- `hypothesis`: given, settings, strategies as st
- `torch`
- `torch.testing._internal.common_utils`: raise_on_run_directly
- `torch.testing._internal.jit_utils`: JitTestCase


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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
python test/jit/test_dataclasses.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/jit`):

- [`test_recursive_script.py_docs.md`](./test_recursive_script.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_python_builtins.py_docs.md`](./test_python_builtins.py_docs.md)
- [`test_functional_blocks.py_docs.md`](./test_functional_blocks.py_docs.md)
- [`test_hooks_modules.py_docs.md`](./test_hooks_modules.py_docs.md)
- [`mydecorator.py_docs.md`](./mydecorator.py_docs.md)
- [`test_union.py_docs.md`](./test_union.py_docs.md)
- [`test_python_bindings.py_docs.md`](./test_python_bindings.py_docs.md)
- [`test_parametrization.py_docs.md`](./test_parametrization.py_docs.md)


## Cross-References

- **File Documentation**: `test_dataclasses.py_docs.md`
- **Keyword Index**: `test_dataclasses.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
