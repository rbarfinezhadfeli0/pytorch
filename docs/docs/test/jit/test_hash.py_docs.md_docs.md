# Documentation: `docs/test/jit/test_hash.py_docs.md`

## File Metadata

- **Path**: `docs/test/jit/test_hash.py_docs.md`
- **Size**: 6,852 bytes (6.69 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/jit/test_hash.py`

## File Metadata

- **Path**: `test/jit/test_hash.py`
- **Size**: 3,409 bytes (3.33 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import os
import sys
from typing import List, Tuple

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestHash(JitTestCase):
    def test_hash_tuple(self):
        def fn(t1: Tuple[int, int], t2: Tuple[int, int]) -> bool:
            return hash(t1) == hash(t2)

        self.checkScript(fn, ((1, 2), (1, 2)))
        self.checkScript(fn, ((1, 2), (3, 4)))
        self.checkScript(fn, ((1, 2), (2, 1)))

    def test_hash_tuple_nested_unhashable_type(self):
        # Tuples may contain unhashable types like `list`, check that we error
        # properly in that case.
        @torch.jit.script
        def fn_unhashable(t1: Tuple[int, List[int]]):
            return hash(t1)

        with self.assertRaisesRegexWithHighlight(RuntimeError, "unhashable", "hash"):
            fn_unhashable((1, [1]))

    def test_hash_tensor(self):
        """Tensors should hash by identity"""

        def fn(t1, t2):
            return hash(t1) == hash(t2)

        tensor1 = torch.tensor(1)
        tensor1_clone = torch.tensor(1)
        tensor2 = torch.tensor(2)

        self.checkScript(fn, (tensor1, tensor1))
        self.checkScript(fn, (tensor1, tensor1_clone))
        self.checkScript(fn, (tensor1, tensor2))

    def test_hash_none(self):
        def fn():
            n1 = None
            n2 = None
            return hash(n1) == hash(n2)

        self.checkScript(fn, ())

    def test_hash_bool(self):
        def fn(b1: bool, b2: bool):
            return hash(b1) == hash(b2)

        self.checkScript(fn, (True, False))
        self.checkScript(fn, (True, True))
        self.checkScript(fn, (False, True))
        self.checkScript(fn, (False, False))

    def test_hash_float(self):
        def fn(f1: float, f2: float):
            return hash(f1) == hash(f2)

        self.checkScript(fn, (1.2345, 1.2345))
        self.checkScript(fn, (1.2345, 6.789))
        self.checkScript(fn, (1.2345, float("inf")))
        self.checkScript(fn, (float("inf"), float("inf")))
        self.checkScript(fn, (1.2345, float("nan")))
        self.checkScript(fn, (float("nan"), float("inf")))

    def test_hash_int(self):
        def fn(i1: int, i2: int):
            return hash(i1) == hash(i2)

        self.checkScript(fn, (123, 456))
        self.checkScript(fn, (123, 123))
        self.checkScript(fn, (123, -123))
        self.checkScript(fn, (-123, -123))
        self.checkScript(fn, (123, 0))

    def test_hash_string(self):
        def fn(s1: str, s2: str):
            return hash(s1) == hash(s2)

        self.checkScript(fn, ("foo", "foo"))
        self.checkScript(fn, ("foo", "bar"))
        self.checkScript(fn, ("foo", ""))

    def test_hash_device(self):
        def fn(d1: torch.device, d2: torch.device):
            return hash(d1) == hash(d2)

        gpu0 = torch.device("cuda:0")
        gpu1 = torch.device("cuda:1")
        cpu = torch.device("cpu")
        self.checkScript(fn, (gpu0, gpu0))
        self.checkScript(fn, (gpu0, gpu1))
        self.checkScript(fn, (gpu0, cpu))
        self.checkScript(fn, (cpu, cpu))


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview

"""Tensors should hash by identity"""        def fn(t1, t2):            return hash(t1) == hash(t2)        tensor1 = torch.tensor(1)        tensor1_clone = torch.tensor(1)        tensor2 = torch.tensor(2)        self.checkScript(fn, (tensor1, tensor1))        self.checkScript(fn, (tensor1, tensor1_clone))        self.checkScript(fn, (tensor1, tensor2))    def test_hash_none(self):

This Python file contains 1 class(es) and 18 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestHash`

**Functions defined**: `test_hash_tuple`, `fn`, `test_hash_tuple_nested_unhashable_type`, `fn_unhashable`, `test_hash_tensor`, `fn`, `test_hash_none`, `fn`, `test_hash_bool`, `fn`, `test_hash_float`, `fn`, `test_hash_int`, `fn`, `test_hash_string`, `fn`, `test_hash_device`, `fn`

**Key imports**: os, sys, List, Tuple, torch, raise_on_run_directly, JitTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `sys`
- `typing`: List, Tuple
- `torch`
- `torch.testing._internal.common_utils`: raise_on_run_directly
- `torch.testing._internal.jit_utils`: JitTestCase


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python test/jit/test_hash.py
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

- **File Documentation**: `test_hash.py_docs.md`
- **Keyword Index**: `test_hash.py_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
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
python docs/test/jit/test_hash.py_docs.md
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

- **File Documentation**: `test_hash.py_docs.md_docs.md`
- **Keyword Index**: `test_hash.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
