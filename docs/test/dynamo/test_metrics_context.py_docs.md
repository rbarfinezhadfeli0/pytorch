# Documentation: test_metrics_context.py

## File Metadata
- **Path**: `test/dynamo/test_metrics_context.py`
- **Size**: 4044 bytes
- **Lines**: 118
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 1 class(es): TestMetricsContext

### Functions
This file defines 11 function(s): setUp, _on_exit, test_context_exists, test_nested_context, test_set, test_set_disallow_overwrite, test_update_disallow_overwrite, test_update_allow_overwrite, test_add_to_set, test_set_key_value, test_top_n


## Key Components

The file contains 319 words across 118 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4044 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
