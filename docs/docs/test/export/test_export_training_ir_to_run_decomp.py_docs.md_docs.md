# Documentation: `docs/test/export/test_export_training_ir_to_run_decomp.py_docs.md`

## File Metadata

- **Path**: `docs/test/export/test_export_training_ir_to_run_decomp.py_docs.md`
- **Size**: 4,710 bytes (4.60 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/export/test_export_training_ir_to_run_decomp.py`

## File Metadata

- **Path**: `test/export/test_export_training_ir_to_run_decomp.py`
- **Size**: 1,909 bytes (1.86 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: export"]
import torch


try:
    from . import test_export, testing
except ImportError:
    import test_export  # @manual=fbcode//caffe2/test:test_export-library

    import testing  # @manual=fbcode//caffe2/test:test_export-library


test_classes = {}


def mocked_training_ir_to_run_decomp_export_strict(*args, **kwargs):
    if "strict" in kwargs:
        ep = torch.export.export(*args, **kwargs)
    else:
        ep = torch.export.export(*args, **kwargs, strict=True)
    return ep.run_decompositions({})


def mocked_training_ir_to_run_decomp_export_non_strict(*args, **kwargs):
    ep = torch.export.export(*args, **kwargs)

    return ep.run_decompositions({})


def make_dynamic_cls(cls, strict):
    if strict:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            "TrainingIRToRunDecompExport",
            test_export.TRAINING_IR_DECOMP_STRICT_SUFFIX,
            mocked_training_ir_to_run_decomp_export_strict,
            xfail_prop="_expected_failure_training_ir_to_run_decomp",
        )
    else:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            "TrainingIRToRunDecompExportNonStrict",
            test_export.TRAINING_IR_DECOMP_NON_STRICT_SUFFIX,
            mocked_training_ir_to_run_decomp_export_non_strict,
            xfail_prop="_expected_failure_training_ir_to_run_decomp_non_strict",
        )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


tests = [
    test_export.TestDynamismExpression,
    test_export.TestExport,
]
for test in tests:
    make_dynamic_cls(test, True)
    make_dynamic_cls(test, False)
del test

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `mocked_training_ir_to_run_decomp_export_strict`, `mocked_training_ir_to_run_decomp_export_non_strict`, `make_dynamic_cls`

**Key imports**: torch, test_export, testing, test_export  , testing  , run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/export`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `.`: test_export, testing
- `test_export  `
- `testing  `
- `torch._dynamo.test_case`: run_tests


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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
python test/export/test_export_training_ir_to_run_decomp.py
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

- **File Documentation**: `test_export_training_ir_to_run_decomp.py_docs.md`
- **Keyword Index**: `test_export_training_ir_to_run_decomp.py_kw.md`
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

- **Error Handling**: Includes exception handling


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
python docs/test/export/test_export_training_ir_to_run_decomp.py_docs.md
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

- **File Documentation**: `test_export_training_ir_to_run_decomp.py_docs.md_docs.md`
- **Keyword Index**: `test_export_training_ir_to_run_decomp.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
