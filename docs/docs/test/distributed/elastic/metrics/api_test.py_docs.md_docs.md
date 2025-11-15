# Documentation: `docs/test/distributed/elastic/metrics/api_test.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/elastic/metrics/api_test.py_docs.md`
- **Size**: 5,880 bytes (5.74 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/elastic/metrics/api_test.py`

## File Metadata

- **Path**: `test/distributed/elastic/metrics/api_test.py`
- **Size**: 3,400 bytes (3.32 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.abs
import abc
import unittest.mock as mock

from torch.distributed.elastic.metrics.api import (
    _get_metric_name,
    MetricData,
    MetricHandler,
    MetricStream,
    prof,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def foo_1():
    pass


class TestMetricsHandler(MetricHandler):
    def __init__(self) -> None:
        self.metric_data = {}

    def emit(self, metric_data: MetricData):
        self.metric_data[metric_data.name] = metric_data


class Parent(abc.ABC):
    @abc.abstractmethod
    def func(self):
        raise NotImplementedError

    def base_func(self):
        self.func()


class Child(Parent):
    # need to decorate the implementation not the abstract method!
    @prof
    def func(self):
        pass


class MetricsApiTest(TestCase):
    def foo_2(self):
        pass

    @prof
    def bar(self):
        pass

    @prof
    def throw(self):
        raise RuntimeError

    @prof(group="torchelastic")
    def bar2(self):
        pass

    def test_get_metric_name(self):
        # Note: since pytorch uses main method to launch tests,
        # the module will be different between fb and oss, this
        # allows keeping the module name consistent.
        foo_1.__module__ = "api_test"
        self.assertEqual("api_test.foo_1", _get_metric_name(foo_1))
        self.assertEqual("MetricsApiTest.foo_2", _get_metric_name(self.foo_2))

    def test_profile(self):
        handler = TestMetricsHandler()
        stream = MetricStream("torchelastic", handler)
        # patch instead of configure to avoid conflicts when running tests in parallel
        with mock.patch(
            "torch.distributed.elastic.metrics.api.getStream", return_value=stream
        ):
            self.bar()

            self.assertEqual(1, handler.metric_data["MetricsApiTest.bar.success"].value)
            self.assertNotIn("MetricsApiTest.bar.failure", handler.metric_data)
            self.assertIn("MetricsApiTest.bar.duration.ms", handler.metric_data)

            with self.assertRaises(RuntimeError):
                self.throw()

            self.assertEqual(
                1, handler.metric_data["MetricsApiTest.throw.failure"].value
            )
            self.assertNotIn("MetricsApiTest.bar_raise.success", handler.metric_data)
            self.assertIn("MetricsApiTest.throw.duration.ms", handler.metric_data)

            self.bar2()
            self.assertEqual(
                "torchelastic",
                handler.metric_data["MetricsApiTest.bar2.success"].group_name,
            )

    def test_inheritance(self):
        handler = TestMetricsHandler()
        stream = MetricStream("torchelastic", handler)
        # patch instead of configure to avoid conflicts when running tests in parallel
        with mock.patch(
            "torch.distributed.elastic.metrics.api.getStream", return_value=stream
        ):
            c = Child()
            c.base_func()

            self.assertEqual(1, handler.metric_data["Child.func.success"].value)
            self.assertIn("Child.func.duration.ms", handler.metric_data)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 4 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestMetricsHandler`, `Parent`, `Child`, `MetricsApiTest`

**Functions defined**: `foo_1`, `__init__`, `emit`, `func`, `base_func`, `func`, `foo_2`, `bar`, `throw`, `bar2`, `test_get_metric_name`, `test_profile`, `test_inheritance`

**Key imports**: abc, unittest.mock as mock, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/elastic/metrics`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `abc`
- `unittest.mock as mock`
- `torch.testing._internal.common_utils`: run_tests, TestCase


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/elastic/metrics/api_test.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/elastic/metrics`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)


## Cross-References

- **File Documentation**: `api_test.py_docs.md`
- **Keyword Index**: `api_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/elastic/metrics`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/elastic/metrics`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python docs/test/distributed/elastic/metrics/api_test.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/elastic/metrics`):

- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`api_test.py_kw.md_docs.md`](./api_test.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `api_test.py_docs.md_docs.md`
- **Keyword Index**: `api_test.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
