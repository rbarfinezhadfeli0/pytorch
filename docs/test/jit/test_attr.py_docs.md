# Documentation: `test/jit/test_attr.py`

## File Metadata

- **Path**: `test/jit/test_attr.py`
- **Size**: 2,151 bytes (2.10 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

from typing import NamedTuple, Tuple

import torch
from torch.testing import FileCheck
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestGetDefaultAttr(JitTestCase):
    def test_getattr_with_default(self):
        class A(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.init_attr_val = 1.0

            def forward(self, x):
                y = getattr(self, "init_attr_val")  # noqa: B009
                w: list[float] = [1.0]
                z = getattr(self, "missing", w)  # noqa: B009
                z.append(y)
                return z

        result = A().forward(0.0)
        self.assertEqual(2, len(result))
        graph = torch.jit.script(A()).graph

        # The "init_attr_val" attribute exists
        FileCheck().check('prim::GetAttr[name="init_attr_val"]').run(graph)
        # The "missing" attribute does not exist, so there should be no corresponding GetAttr in AST
        FileCheck().check_not("missing").run(graph)
        # instead the getattr call will emit the default value, which is a list with one float element
        FileCheck().check("float[] = prim::ListConstruct").run(graph)

    def test_getattr_named_tuple(self):
        global MyTuple

        class MyTuple(NamedTuple):
            x: str
            y: torch.Tensor

        def fn(x: MyTuple) -> Tuple[str, torch.Tensor, int]:
            return (
                getattr(x, "x", "fdsa"),
                getattr(x, "y", torch.ones((3, 3))),
                getattr(x, "z", 7),
            )

        inp = MyTuple(x="test", y=torch.ones(3, 3) * 2)
        ref = fn(inp)
        fn_s = torch.jit.script(fn)
        res = fn_s(inp)
        self.assertEqual(res, ref)

    def test_getattr_tuple(self):
        def fn(x: Tuple[str, int]) -> int:
            return getattr(x, "x", 2)

        with self.assertRaisesRegex(RuntimeError, "but got a normal Tuple"):
            torch.jit.script(fn)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview


This Python file contains 3 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestGetDefaultAttr`, `A`, `MyTuple`

**Functions defined**: `test_getattr_with_default`, `__init__`, `forward`, `test_getattr_named_tuple`, `fn`, `test_getattr_tuple`, `fn`

**Key imports**: NamedTuple, Tuple, torch, FileCheck, raise_on_run_directly, JitTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: NamedTuple, Tuple
- `torch`
- `torch.testing`: FileCheck
- `torch.testing._internal.common_utils`: raise_on_run_directly
- `torch.testing._internal.jit_utils`: JitTestCase


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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
python test/jit/test_attr.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/jit`):

- [`test_dataclasses.py_docs.md`](./test_dataclasses.py_docs.md)
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

- **File Documentation**: `test_attr.py_docs.md`
- **Keyword Index**: `test_attr.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
