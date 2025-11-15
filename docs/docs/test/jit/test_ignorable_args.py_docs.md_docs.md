# Documentation: `docs/test/jit/test_ignorable_args.py_docs.md`

## File Metadata

- **Path**: `docs/test/jit/test_ignorable_args.py_docs.md`
- **Size**: 6,011 bytes (5.87 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/jit/test_ignorable_args.py`

## File Metadata

- **Path**: `test/jit/test_ignorable_args.py`
- **Size**: 2,334 bytes (2.28 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import os
import sys

import torch
from torch._C import parse_ir
from torch.testing import FileCheck


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


# Tests that Python slice class is supported in TorchScript
class TestIgnorableArgs(JitTestCase):
    def test_slice_ignorable_args_for_slice(self):
        graph_str = """graph():
            %13 : int = prim::Constant[value=0]()
            %10 : bool = prim::Constant[value=0]()
            %8 : NoneType = prim::Constant()
            %0 : int = prim::Constant[value=1]()
            %1 : int = prim::Constant[value=2]()
            %2 : int = prim::Constant[value=3]()
            %3 : int = prim::Constant[value=4]()
            %4 : int = prim::Constant[value=9]()
            %5 : int[] = prim::ListConstruct(%0, %1, %2, %3, %4, %4)
            %6 : int[] = prim::ListConstruct(%0, %1, %2, %3, %4, %4)
            %7 : int[][] = prim::ListConstruct(%5, %6)
            %val.1 : Tensor = aten::tensor(%7, %8, %8, %10)
            %16 : Tensor = aten::slice(%val.1, %13, %1, %8, %0)
            %20 : Tensor = aten::slice(%16, %0, %8, %0, %0)
            return (%20)"""
        graph = parse_ir(graph_str)
        function = self.createFunctionFromGraph(graph)
        function_copy = self.getExportImportCopy(function)
        src = str(function.code)
        # For a signature:
        # aten::slice(Tensor self, int dim, int start, int end, int step) -> Tensor
        # We ignore trailing arguments after start=2 for dim 0
        # and after end=1 for dim 1
        # because in %16, %15 and %0 are default values for the schema.
        FileCheck().check(
            "torch.slice(torch.slice(torch.tensor(_0), 0, 2), 1, None, 1)"
        ).run(src)
        self.assertEqual(function(), function_copy())

    def test_add_out_ignorable_args(self):
        @torch.jit.script
        def fn(x: torch.Tensor, y: torch.Tensor):
            torch.add(x, y, out=y)

        FileCheck().check("torch.add(x, y, out=y)").run(fn.code)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview

graph_str = """graph():            %13 : int = prim::Constant[value=0]()            %10 : bool = prim::Constant[value=0]()            %8 : NoneType = prim::Constant()            %0 : int = prim::Constant[value=1]()            %1 : int = prim::Constant[value=2]()            %2 : int = prim::Constant[value=3]()            %3 : int = prim::Constant[value=4]()            %4 : int = prim::Constant[value=9]()            %5 : int[] = prim::ListConstruct(%0, %1, %2, %3, %4, %4)            %6 : int[] = prim::ListConstruct(%0, %1, %2, %3, %4, %4)            %7 : int[][] = prim::ListConstruct(%5, %6)            %val.1 : Tensor = aten::tensor(%7, %8, %8, %10)            %16 : Tensor = aten::slice(%val.1, %13, %1, %8, %0)            %20 : Tensor = aten::slice(%16, %0, %8, %0, %0)

This Python file contains 2 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestIgnorableArgs`

**Functions defined**: `test_slice_ignorable_args_for_slice`, `test_add_out_ignorable_args`, `fn`

**Key imports**: os, sys, torch, parse_ir, FileCheck, raise_on_run_directly, JitTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `sys`
- `torch`
- `torch._C`: parse_ir
- `torch.testing`: FileCheck
- `torch.testing._internal.common_utils`: raise_on_run_directly
- `torch.testing._internal.jit_utils`: JitTestCase


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python test/jit/test_ignorable_args.py
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

- **File Documentation**: `test_ignorable_args.py_docs.md`
- **Keyword Index**: `test_ignorable_args.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/jit`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python docs/test/jit/test_ignorable_args.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/jit`):

- [`test_attr.py_kw.md_docs.md`](./test_attr.py_kw.md_docs.md)
- [`test_parametrization.py_kw.md_docs.md`](./test_parametrization.py_kw.md_docs.md)
- [`test_hooks.py_kw.md_docs.md`](./test_hooks.py_kw.md_docs.md)
- [`test_dataclasses.py_docs.md_docs.md`](./test_dataclasses.py_docs.md_docs.md)
- [`test_aten_pow.py_kw.md_docs.md`](./test_aten_pow.py_kw.md_docs.md)
- [`test_misc.py_docs.md_docs.md`](./test_misc.py_docs.md_docs.md)
- [`test_graph_rewrite_passes.py_kw.md_docs.md`](./test_graph_rewrite_passes.py_kw.md_docs.md)
- [`test_module_containers.py_kw.md_docs.md`](./test_module_containers.py_kw.md_docs.md)
- [`test_complex.py_kw.md_docs.md`](./test_complex.py_kw.md_docs.md)
- [`test_types.py_kw.md_docs.md`](./test_types.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_ignorable_args.py_docs.md_docs.md`
- **Keyword Index**: `test_ignorable_args.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
