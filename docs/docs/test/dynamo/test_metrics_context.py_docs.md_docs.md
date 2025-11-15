# Documentation: `docs/test/dynamo/test_metrics_context.py_docs.md`

## File Metadata

- **Path**: `docs/test/dynamo/test_metrics_context.py_docs.md`
- **Size**: 6,980 bytes (6.82 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/dynamo/test_metrics_context.py`

## File Metadata

- **Path**: `test/dynamo/test_metrics_context.py`
- **Size**: 4,044 bytes (3.95 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]

from torch._dynamo.metrics_context import MetricsContext, TopN
from torch._dynamo.test_case import run_tests, TestCase


class TestMetricsContext(TestCase):
    def setUp(self):
        super().setUp()
        self.metrics = {}

    def _on_exit(self, start_ns, end_ns, metrics, exc_type, exc_value):
        # Save away the metrics to be validated in the test.
        self.metrics = metrics.copy()

    def test_context_exists(self):
        """
        Setting a value without entering the context should raise.
        """
        context = MetricsContext(self._on_exit)
        with self.assertRaisesRegex(RuntimeError, "outside of a MetricsContext"):
            context.increment("m", 1)

        with self.assertRaisesRegex(RuntimeError, "outside of a MetricsContext"):
            context.set("m", 1)

        with self.assertRaisesRegex(RuntimeError, "outside of a MetricsContext"):
            context.update({"m", 1})

    def test_nested_context(self):
        """
        Only the outermost context should get an on_exit call, and it should
        include everything.
        """
        context = MetricsContext(self._on_exit)
        with context:
            with context:
                context.set("m1", 1)
            self.assertEqual(self.metrics, {})
            context.set("m2", 2)
        self.assertEqual(self.metrics, {"m1": 1, "m2": 2})

    def test_set(self):
        """
        Validate various ways to set metrics.
        """
        with MetricsContext(self._on_exit) as context:
            context.set("m1", 1)
            context.set("m2", 2)
            context.update({"m3": 3, "m4": 4})

        self.assertEqual(self.metrics, {"m1": 1, "m2": 2, "m3": 3, "m4": 4})

    def test_set_disallow_overwrite(self):
        """
        Validate set won't overwrite.
        """
        with MetricsContext(self._on_exit) as context:
            context.set("m1", 1)
            with self.assertRaisesRegex(RuntimeError, "already been set"):
                context.set("m1", 2)

        self.assertEqual(self.metrics, {"m1": 1})

    def test_update_disallow_overwrite(self):
        """
        Validate update won't overwrite.
        """
        with MetricsContext(self._on_exit) as context:
            context.update({"m1": 1, "m2": 2})
            with self.assertRaisesRegex(RuntimeError, "already been set"):
                context.update({"m1": 7, "m3": 3})

    def test_update_allow_overwrite(self):
        """
        Validate update will overwrite when given param.
        """
        with MetricsContext(self._on_exit) as context:
            context.update({"m1": 1, "m2": 2})
            context.update({"m1": 7, "m3": 3}, overwrite=True)

        self.assertEqual(self.metrics, {"m1": 7, "m2": 2, "m3": 3})

    def test_add_to_set(self):
        """
        Validate add_to_set.
        """
        with MetricsContext(self._on_exit) as context:
            context.add_to_set("m1", 1)
            context.add_to_set("m1", 2)
            context.add_to_set("m2", 3)
            context.add_to_set("m2", 4)

        self.assertEqual(self.metrics, {"m1": {1, 2}, "m2": {3, 4}})
        self.assertTrue(isinstance(self.metrics["m1"], set))
        self.assertTrue(isinstance(self.metrics["m2"], set))

    def test_set_key_value(self):
        with MetricsContext(self._on_exit) as context:
            context.set_key_value("feature_usage", "k", True)
            # Overrides allowed
            context.set_key_value("feature_usage", "k2", True)
            context.set_key_value("feature_usage", "k2", False)

        self.assertEqual(self.metrics, {"feature_usage": {"k": True, "k2": False}})

    def test_top_n(self):
        top_n = TopN(3)
        for k, v in (("seven", 7), ("four", 4), ("five", 5), ("six", 6), ("eight", 8)):
            top_n.add(k, v)

        self.assertEqual(len(top_n), 3)
        print(list(top_n))
        self.assertEqual(list(top_n), [("eight", 8), ("seven", 7), ("six", 6)])


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""        Setting a value without entering the context should raise.

This Python file contains 1 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestMetricsContext`

**Functions defined**: `setUp`, `_on_exit`, `test_context_exists`, `test_nested_context`, `test_set`, `test_set_disallow_overwrite`, `test_update_disallow_overwrite`, `test_update_allow_overwrite`, `test_add_to_set`, `test_set_key_value`, `test_top_n`

**Key imports**: MetricsContext, TopN, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch._dynamo.metrics_context`: MetricsContext, TopN
- `torch._dynamo.test_case`: run_tests, TestCase


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python test/dynamo/test_metrics_context.py
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

- **File Documentation**: `test_metrics_context.py_docs.md`
- **Keyword Index**: `test_metrics_context.py_kw.md`
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

*No specific patterns automatically detected.*


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
python docs/test/dynamo/test_metrics_context.py_docs.md
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

- **File Documentation**: `test_metrics_context.py_docs.md_docs.md`
- **Keyword Index**: `test_metrics_context.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
