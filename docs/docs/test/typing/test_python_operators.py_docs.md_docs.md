# Documentation: `docs/test/typing/test_python_operators.py_docs.md`

## File Metadata

- **Path**: `docs/test/typing/test_python_operators.py_docs.md`
- **Size**: 7,731 bytes (7.55 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/typing/test_python_operators.py`

## File Metadata

- **Path**: `test/typing/test_python_operators.py`
- **Size**: 5,584 bytes (5.45 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# mypy: ignore-errors
# Owner(s): ["module: unknown"]
import token
from itertools import product
from pathlib import Path

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


MM = "@"

BINARY_RETURNS_BOOL = "!=", "<", "<=", "==", ">", ">="
BINARY_ACCEPTS_FLOAT_OR_INT = "%", "*", "**", "+", "-", "/", "//"
BINARY_ACCEPTS_INT_ONLY = "&", "<<", ">>", "^", "|"
BINARY_OPS = (
    *BINARY_RETURNS_BOOL,
    *BINARY_ACCEPTS_FLOAT_OR_INT,
    *BINARY_ACCEPTS_INT_ONLY,
    MM,
)

BINARY_RETURNS_FLOAT = ("/",)

UNARY_ACCEPTS_FLOAT_OR_INT = "+", "-"
UNARY_ACCEPTS_INT_ONLY = ("~",)
UNARY_OPS = *UNARY_ACCEPTS_FLOAT_OR_INT, *UNARY_ACCEPTS_INT_ONLY

PUNCTUATION = ",", ";"

OPERATORS = *UNARY_OPS, *BINARY_OPS, *PUNCTUATION

FLOATS = 1.5, torch.tensor((2.5, 3.5))
INTS = 3, torch.tensor((1, 2))
ALL = *FLOATS, *INTS

TYPE_TEST_FILE = Path(__file__).parent / "pass/arithmetic_ops.py"


class TestPythonOperators(TestCase):
    # Prove that UNARY_OPS, BINARY_OPS, and OPERATORS are correct and complete
    def test_operators_are_correct_and_complete(self):
        self.assertFalse(set(OPERATORS).difference(token.EXACT_TOKEN_TYPES))

        unary, binary, punctuation = {}, {}, {}

        for op in token.EXACT_TOKEN_TYPES:
            if op in PUNCTUATION:
                punctuation[op] = True
            else:
                try:
                    unary[op] = compile(f"{op}1 ; {op}a", op, "single")
                except SyntaxError:
                    pass
                try:
                    binary[op] = compile(f"2 {op} 3 ; a {op} b", op, "single")
                except SyntaxError:
                    pass

        self.assertEqual(sorted(unary), sorted(UNARY_OPS))
        self.assertEqual(sorted(binary), sorted(BINARY_OPS))
        self.assertEqual(sorted(punctuation), sorted(PUNCTUATION))

    def test_type_tests_are_complete(self):
        binary, unary = {}, []

        with TYPE_TEST_FILE.open() as fp:
            # Looking for lines like:  assert_type(TENSOR ^ BOOL, Tensor)
            # But not:                 assert_type(BOOL ^ BINARY, Binary)
            lines = (i for i in fp if "TENSOR" in i)
            for line in lines:
                if expr := line.partition("assert_type(")[2].partition(",")[0]:
                    if expr[0].isalpha():
                        # ** formats differently from all other operators
                        a, op, b = expr.replace("**", " ** ").split()
                        binary.setdefault(op, []).append((a, b))
                    else:
                        unary.append(expr[0])

        self.assertEqual(sorted(unary), sorted(UNARY_OPS))
        self.assertEqual(sorted(binary), sorted(BINARY_OPS))
        value, *values = binary.values()
        self.assertEqual(values, [value] * len(values))

    @parametrize("a, op, b", product(ALL, BINARY_OPS, ALL))
    def test_binary(self, a, op, b):
        try:
            r = eval(f"a {op} b")
        except Exception as e:
            r = e

        any_tensor = isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor)
        any_float = _any_float(a, b)
        returns_float = any_float or op in BINARY_RETURNS_FLOAT

        if op == MM:
            if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
                self.assertIsInstance(r, TypeError)
            elif a is b:
                self.assertIsInstance(r, torch.Tensor)
            else:
                self.assertIsInstance(r, RuntimeError)

        elif any_tensor:
            if op in BINARY_ACCEPTS_INT_ONLY and any_float:
                # See https://github.com/pytorch/pytorch/issues/15754
                self.assertIsInstance(r, NotImplementedError)
            else:
                self.assertIsInstance(r, torch.Tensor)

                if op in BINARY_RETURNS_BOOL:
                    self.assertEqual(r.dtype, torch.bool)
                elif op in BINARY_ACCEPTS_INT_ONLY:
                    self.assertFalse(r.dtype.is_floating_point)
                elif op in BINARY_ACCEPTS_FLOAT_OR_INT:
                    self.assertEqual(r.dtype.is_floating_point, returns_float)
                else:
                    self.assertFalse("Logic error")

        elif op in BINARY_RETURNS_BOOL:
            self.assertIsInstance(r, bool)

        elif op in BINARY_ACCEPTS_INT_ONLY:
            if any_float:
                self.assertIsInstance(r, TypeError)
            else:
                self.assertIsInstance(r, int)

        elif returns_float:
            self.assertIsInstance(r, float)

        else:
            self.assertIsInstance(r, int)

    @parametrize("op, a", product(UNARY_OPS, ALL))
    def test_unary(self, op, a):
        try:
            r = eval(f"{op} a")
        except Exception as e:
            r = e

        if op in UNARY_ACCEPTS_INT_ONLY and _any_float(a):
            self.assertIsInstance(r, TypeError)
        elif isinstance(a, torch.Tensor):
            self.assertIsInstance(r, torch.Tensor)
        elif op in UNARY_ACCEPTS_INT_ONLY:
            self.assertIsInstance(r, int)
        elif isinstance(a, float):
            self.assertIsInstance(r, float)
        else:
            self.assertIsInstance(r, int)


def _any_float(*x):
    for i in x:
        if isinstance(i, float) or (
            isinstance(i, torch.Tensor) and i.dtype.is_floating_point
        ):
            return True
    return False


instantiate_parametrized_tests(TestPythonOperators)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestPythonOperators`

**Functions defined**: `test_operators_are_correct_and_complete`, `test_type_tests_are_complete`, `test_binary`, `test_unary`, `_any_float`

**Key imports**: token, product, Path, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/typing`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `token`
- `itertools`: product
- `pathlib`: Path
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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
python test/typing/test_python_operators.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/typing`):



## Cross-References

- **File Documentation**: `test_python_operators.py_docs.md`
- **Keyword Index**: `test_python_operators.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/typing`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/typing`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/typing/test_python_operators.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/typing`):

- [`test_python_operators.py_kw.md_docs.md`](./test_python_operators.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_python_operators.py_docs.md_docs.md`
- **Keyword Index**: `test_python_operators.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
