# Documentation: `test/jit/test_dce.py`

## File Metadata

- **Path**: `test/jit/test_dce.py`
- **Size**: 2,181 bytes (2.13 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import torch
from torch.testing import FileCheck
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase, make_global


class TestDCE(JitTestCase):
    def test_setattr_no_aliasdb(self):
        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = torch.empty([2, 2])

            def forward(self):
                x = torch.rand([3, 3])
                self.x = x

        net = torch.jit.script(Net())

        FileCheck().check("prim::SetAttr").run(net.graph)

    def test_setattr_removed(self):
        @torch.jit.script
        class Thing1:
            def __init__(self) -> None:
                self.x = torch.zeros([2, 2])

        make_global(Thing1)

        class Thing2(torch.nn.Module):
            def forward(self):
                x = torch.rand([2, 2])
                y = torch.rand([2, 2])
                t1 = Thing1()
                t1.x = x
                return y

        unscripted = Thing2()

        t2 = torch.jit.script(unscripted)
        t2.eval()

        # freezing inlines t1.__init__(), after which DCE can occur.
        t2 = torch.jit.freeze(t2)
        FileCheck().check_not("prim::SetAttr").run(t2.graph)

    def test_mutated_simple(self):
        def fn(x: torch.Tensor):
            y = x.sin()
            y_slice = y[::2]
            y_slice.add_(x[::2])
            z = y.cos()
            return z

        fn_s = torch.jit.script(fn)
        torch._C._jit_pass_dce_graph(fn_s.graph)

        FileCheck().check("aten::add_").run(fn_s.graph)

    def test_mutated_loop(self):
        def fn(x: torch.Tensor):
            y = x.sin()
            y_slice = y[::2]
            y_slice.add_(x[::2])
            for _ in range(2):
                y_slice = y[::2]
                y = y.repeat(2)
            z = y.cos()
            return z

        fn_s = torch.jit.script(fn)
        torch._C._jit_pass_dce_graph(fn_s.graph)

        FileCheck().check("aten::add_").run(fn_s.graph)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview


This Python file contains 4 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestDCE`, `Net`, `Thing1`, `Thing2`

**Functions defined**: `test_setattr_no_aliasdb`, `__init__`, `forward`, `test_setattr_removed`, `__init__`, `forward`, `test_mutated_simple`, `fn`, `test_mutated_loop`, `fn`

**Key imports**: torch, FileCheck, raise_on_run_directly, JitTestCase, make_global


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.testing`: FileCheck
- `torch.testing._internal.common_utils`: raise_on_run_directly
- `torch.testing._internal.jit_utils`: JitTestCase, make_global


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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/jit/test_dce.py
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

- **File Documentation**: `test_dce.py_docs.md`
- **Keyword Index**: `test_dce.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
