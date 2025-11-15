# Documentation: `docs/test/export/test_tools.py_docs.md`

## File Metadata

- **Path**: `docs/test/export/test_tools.py_docs.md`
- **Size**: 4,773 bytes (4.66 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/export/test_tools.py`

## File Metadata

- **Path**: `test/export/test_tools.py`
- **Size**: 1,889 bytes (1.84 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: export"]

import torch
from torch._dynamo.test_case import TestCase
from torch._export.tools import report_exportability
from torch.testing._internal.common_utils import run_tests


torch.library.define(
    "testlib::op_missing_meta",
    "(Tensor(a!) x, Tensor(b!) z) -> Tensor",
    tags=torch.Tag.pt2_compliant_tag,
)


@torch.library.impl("testlib::op_missing_meta", "cpu")
@torch._dynamo.disable
def op_missing_meta(x, z):
    x.add_(5)
    z.add_(5)
    return x + z


class TestExportTools(TestCase):
    def test_report_exportability_basic(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                return x[0] + y

        f = Module()
        inp = ([torch.ones(1, 3)], torch.ones(1, 3))

        report = report_exportability(f, inp)
        self.assertTrue(len(report) == 1)
        self.assertTrue(report[""] is None)

    def test_report_exportability_with_issues(self):
        class Unsupported(torch.nn.Module):
            def forward(self, x):
                return torch.ops.testlib.op_missing_meta(x, x.cos())

        class Supported(torch.nn.Module):
            def forward(self, x):
                return x.sin()

        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.unsupported = Unsupported()
                self.supported = Supported()

            def forward(self, x):
                y = torch.nonzero(x)
                return self.unsupported(y) + self.supported(y)

        f = Module()
        inp = (torch.ones(4, 4),)

        report = report_exportability(f, inp, strict=False, pre_dispatch=True)

        self.assertTrue(report[""] is not None)
        self.assertTrue(report["unsupported"] is not None)
        self.assertTrue(report["supported"] is None)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 5 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestExportTools`, `Module`, `Unsupported`, `Supported`, `Module`

**Functions defined**: `op_missing_meta`, `test_report_exportability_basic`, `forward`, `test_report_exportability_with_issues`, `forward`, `forward`, `__init__`, `forward`

**Key imports**: torch, TestCase, report_exportability, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/export`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._dynamo.test_case`: TestCase
- `torch._export.tools`: report_exportability
- `torch.testing._internal.common_utils`: run_tests


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

This is a test file. Run it with:

```bash
python test/export/test_tools.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/export`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_schema.py_docs.md`](./test_schema.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_cpp_serdes.py_docs.md`](./test_cpp_serdes.py_docs.md)
- [`test_export_opinfo.py_docs.md`](./test_export_opinfo.py_docs.md)
- [`test_lift_unlift.py_docs.md`](./test_lift_unlift.py_docs.md)
- [`test_retraceability.py_docs.md`](./test_retraceability.py_docs.md)
- [`test_converter.py_docs.md`](./test_converter.py_docs.md)
- [`test_nativert.py_docs.md`](./test_nativert.py_docs.md)
- [`test_export.py_docs.md`](./test_export.py_docs.md)


## Cross-References

- **File Documentation**: `test_tools.py_docs.md`
- **Keyword Index**: `test_tools.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/export`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/export`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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

This is a test file. Run it with:

```bash
python docs/test/export/test_tools.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/export`):

- [`test_serialize.py_docs.md_docs.md`](./test_serialize.py_docs.md_docs.md)
- [`test_verifier.py_kw.md_docs.md`](./test_verifier.py_kw.md_docs.md)
- [`test_upgrader.py_kw.md_docs.md`](./test_upgrader.py_kw.md_docs.md)
- [`test_db.py_docs.md_docs.md`](./test_db.py_docs.md_docs.md)
- [`test_export.py_docs.md_docs.md`](./test_export.py_docs.md_docs.md)
- [`test_dynamic_shapes.py_kw.md_docs.md`](./test_dynamic_shapes.py_kw.md_docs.md)
- [`test_passes.py_kw.md_docs.md`](./test_passes.py_kw.md_docs.md)
- [`test_unflatten.py_docs.md_docs.md`](./test_unflatten.py_docs.md_docs.md)
- [`test_pass_infra.py_kw.md_docs.md`](./test_pass_infra.py_kw.md_docs.md)
- [`test_functionalized_assertions.py_kw.md_docs.md`](./test_functionalized_assertions.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_tools.py_docs.md_docs.md`
- **Keyword Index**: `test_tools.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
