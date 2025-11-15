# Documentation: `test/jit/test_logging.py`

## File Metadata

- **Path**: `test/jit/test_logging.py`
- **Size**: 4,263 bytes (4.16 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]
# ruff: noqa: F841

import os
import sys

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestLogging(JitTestCase):
    def test_bump_numeric_counter(self):
        class ModuleThatLogs(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                for _ in range(x.size(0)):
                    x += 1.0
                    torch.jit._logging.add_stat_value("foo", 1)

                if bool(x.sum() > 0.0):
                    torch.jit._logging.add_stat_value("positive", 1)
                else:
                    torch.jit._logging.add_stat_value("negative", 1)
                return x

        logger = torch.jit._logging.LockingLogger()
        old_logger = torch.jit._logging.set_logger(logger)
        try:
            mtl = ModuleThatLogs()
            for _ in range(5):
                mtl(torch.rand(3, 4, 5))

            self.assertEqual(logger.get_counter_val("foo"), 15)
            self.assertEqual(logger.get_counter_val("positive"), 5)
        finally:
            torch.jit._logging.set_logger(old_logger)

    def test_trace_numeric_counter(self):
        def foo(x):
            torch.jit._logging.add_stat_value("foo", 1)
            return x + 1.0

        traced = torch.jit.trace(foo, torch.rand(3, 4))
        logger = torch.jit._logging.LockingLogger()
        old_logger = torch.jit._logging.set_logger(logger)
        try:
            traced(torch.rand(3, 4))

            self.assertEqual(logger.get_counter_val("foo"), 1)
        finally:
            torch.jit._logging.set_logger(old_logger)

    def test_time_measurement_counter(self):
        class ModuleThatTimes(torch.jit.ScriptModule):
            def forward(self, x):
                tp_start = torch.jit._logging.time_point()
                for _ in range(30):
                    x += 1.0
                tp_end = torch.jit._logging.time_point()
                torch.jit._logging.add_stat_value("mytimer", tp_end - tp_start)
                return x

        mtm = ModuleThatTimes()
        logger = torch.jit._logging.LockingLogger()
        old_logger = torch.jit._logging.set_logger(logger)
        try:
            mtm(torch.rand(3, 4))
            self.assertGreater(logger.get_counter_val("mytimer"), 0)
        finally:
            torch.jit._logging.set_logger(old_logger)

    def test_time_measurement_counter_script(self):
        class ModuleThatTimes(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                tp_start = torch.jit._logging.time_point()
                for _ in range(30):
                    x += 1.0
                tp_end = torch.jit._logging.time_point()
                torch.jit._logging.add_stat_value("mytimer", tp_end - tp_start)
                return x

        mtm = ModuleThatTimes()
        logger = torch.jit._logging.LockingLogger()
        old_logger = torch.jit._logging.set_logger(logger)
        try:
            mtm(torch.rand(3, 4))
            self.assertGreater(logger.get_counter_val("mytimer"), 0)
        finally:
            torch.jit._logging.set_logger(old_logger)

    def test_counter_aggregation(self):
        def foo(x):
            for _ in range(3):
                torch.jit._logging.add_stat_value("foo", 1)
            return x + 1.0

        traced = torch.jit.trace(foo, torch.rand(3, 4))
        logger = torch.jit._logging.LockingLogger()
        logger.set_aggregation_type("foo", torch.jit._logging.AggregationType.AVG)
        old_logger = torch.jit._logging.set_logger(logger)
        try:
            traced(torch.rand(3, 4))

            self.assertEqual(logger.get_counter_val("foo"), 1)
        finally:
            torch.jit._logging.set_logger(old_logger)

    def test_logging_levels_set(self):
        torch._C._jit_set_logging_option("foo")
        self.assertEqual("foo", torch._C._jit_get_logging_option())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview


This Python file contains 4 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestLogging`, `ModuleThatLogs`, `ModuleThatTimes`, `ModuleThatTimes`

**Functions defined**: `test_bump_numeric_counter`, `forward`, `test_trace_numeric_counter`, `foo`, `test_time_measurement_counter`, `forward`, `test_time_measurement_counter_script`, `forward`, `test_counter_aggregation`, `foo`, `test_logging_levels_set`

**Key imports**: os, sys, torch, raise_on_run_directly, JitTestCase


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
python test/jit/test_logging.py
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

- **File Documentation**: `test_logging.py_docs.md`
- **Keyword Index**: `test_logging.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
