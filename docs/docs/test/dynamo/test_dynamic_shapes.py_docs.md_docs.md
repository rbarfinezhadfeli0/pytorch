# Documentation: `docs/test/dynamo/test_dynamic_shapes.py_docs.md`

## File Metadata

- **Path**: `docs/test/dynamo/test_dynamic_shapes.py_docs.md`
- **Size**: 6,814 bytes (6.65 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/dynamo/test_dynamic_shapes.py`

## File Metadata

- **Path**: `test/dynamo/test_dynamic_shapes.py`
- **Size**: 3,695 bytes (3.61 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]
import unittest
import warnings

from torch._dynamo import config
from torch._dynamo.testing import make_test_cls_with_patches
from torch.fx.experimental import _config as fx_config
from torch.testing._internal.common_utils import slowTest, TEST_Z3


try:
    from . import (
        test_aot_autograd,
        test_ctx_manager,
        test_export,
        test_functions,
        test_higher_order_ops,
        test_misc,
        test_modules,
        test_repros,
        test_sdpa,
        test_subgraphs,
    )
except ImportError:
    import test_aot_autograd
    import test_ctx_manager
    import test_export
    import test_functions
    import test_higher_order_ops
    import test_misc

    import test_modules
    import test_repros
    import test_sdpa
    import test_subgraphs


test_classes = {}


def make_dynamic_cls(cls):
    suffix = "_dynamic_shapes"

    cls_prefix = "DynamicShapes"

    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        (config, "assume_static_by_default", False),
        (config, "specialize_int", False),
        # When we unspecialize float, we wobble tests by changing
        # the op count since previously we would just specialize and constant
        # fold floats into the graph, whereas when we unspecialize we will have
        # ops for item, add, and all other tensorified operations. Since these
        # tests really aren't testing that, we purposely specialize floats here.
        (config, "specialize_float", True),
        (fx_config, "translation_validation", TEST_Z3),
        (fx_config, "check_shape_env_recorded_events", True),
        (fx_config, "validate_shape_env_version_key", True),
        xfail_prop="_expected_failure_dynamic",
    )

    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


tests = [
    test_ctx_manager.CtxManagerTests,
    test_functions.FunctionTests,
    test_misc.MiscTests,
    test_repros.ReproTests,
    test_modules.NNModuleTests,
    test_export.ExportTests,
    test_subgraphs.SubGraphTests,
    test_higher_order_ops.HigherOrderOpTests,
    test_higher_order_ops.FuncTorchHigherOrderOpTests,
    test_aot_autograd.AotAutogradFallbackTests,
    test_sdpa.TestSDPA,
]
for test in tests:
    make_dynamic_cls(test)
del test

if TEST_Z3:
    if not config.inline_inbuilt_nn_modules:
        # TODO model is somehow not being freed when z3 is available
        unittest.expectedFailure(
            DynamicShapesMiscTests.test_parameter_free_dynamic_shapes  # noqa: F821
        )

# Test takes too long ~700s as of 414a1fd29f04d06e41b7f895368dd1f83a4be29d
DynamicShapesExportTests.test_retracibility_dynamic_shapes = slowTest(  # noqa: F821
    DynamicShapesExportTests.test_retracibility_dynamic_shapes  # noqa: F821
)
# Also take more than 30m as of 15cc9f2e7e7b2b175f24755925dc38d4d430905d
DynamicShapesExportTests.test_retracibility_dict_container_inp_out_dynamic_shapes = slowTest(  # noqa: F821
    DynamicShapesExportTests.test_retracibility_dict_container_inp_out_dynamic_shapes  # noqa: F821
)
DynamicShapesExportTests.test_retracibility_nested_list_out_dynamic_shapes = slowTest(  # noqa: F821
    DynamicShapesExportTests.test_retracibility_nested_list_out_dynamic_shapes  # noqa: F821
)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if not TEST_Z3:
        warnings.warn(
            "translation validation is off. "
            "Testing with translation validation requires Z3."
        )

    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `make_dynamic_cls`

**Key imports**: unittest, warnings, config, make_test_cls_with_patches, _config as fx_config, slowTest, TEST_Z3, test_aot_autograd, test_ctx_manager, test_export, test_functions


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `warnings`
- `torch._dynamo`: config
- `torch._dynamo.testing`: make_test_cls_with_patches
- `torch.fx.experimental`: _config as fx_config
- `torch.testing._internal.common_utils`: slowTest, TEST_Z3
- `test_aot_autograd`
- `test_ctx_manager`
- `test_export`
- `test_functions`
- `test_higher_order_ops`
- `test_misc`
- `test_modules`
- `test_repros`
- `test_sdpa`
- `test_subgraphs`
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
python test/dynamo/test_dynamic_shapes.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/dynamo`):

- [`test_guard_serialization.py_docs.md`](./test_guard_serialization.py_docs.md)
- [`test_subgraphs.py_docs.md`](./test_subgraphs.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_unspec.py_docs.md`](./test_unspec.py_docs.md)
- [`test_trace_rules.py_docs.md`](./test_trace_rules.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_pre_dispatch.py_docs.md`](./test_pre_dispatch.py_docs.md)
- [`test_autograd_function.py_docs.md`](./test_autograd_function.py_docs.md)
- [`test_optimizers.py_docs.md`](./test_optimizers.py_docs.md)
- [`test_callback.py_docs.md`](./test_callback.py_docs.md)


## Cross-References

- **File Documentation**: `test_dynamic_shapes.py_docs.md`
- **Keyword Index**: `test_dynamic_shapes.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/dynamo`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/dynamo`, which is part of the **testing infrastructure**.



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
python docs/test/dynamo/test_dynamic_shapes.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/dynamo`):

- [`test_error_messages.py_docs.md_docs.md`](./test_error_messages.py_docs.md_docs.md)
- [`test_hooks.py_kw.md_docs.md`](./test_hooks.py_kw.md_docs.md)
- [`test_unittest.py_docs.md_docs.md`](./test_unittest.py_docs.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)
- [`test_aot_autograd.py_kw.md_docs.md`](./test_aot_autograd.py_kw.md_docs.md)
- [`test_einops.py_docs.md_docs.md`](./test_einops.py_docs.md_docs.md)
- [`test_compile.py_kw.md_docs.md`](./test_compile.py_kw.md_docs.md)
- [`test_misc.py_docs.md_docs.md`](./test_misc.py_docs.md_docs.md)
- [`test_buffers_override.py_kw.md_docs.md`](./test_buffers_override.py_kw.md_docs.md)
- [`test_frame_init.py_docs.md_docs.md`](./test_frame_init.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_dynamic_shapes.py_docs.md_docs.md`
- **Keyword Index**: `test_dynamic_shapes.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
