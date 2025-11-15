# Documentation: `tools/test/heuristics/test_heuristics.py`

## File Metadata

- **Path**: `tools/test/heuristics/test_heuristics.py`
- **Size**: 6,486 bytes (6.33 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# For testing specific heuristics
from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from tools.test.heuristics.test_interface import TestTD
from tools.testing.target_determination.heuristics.filepath import (
    file_matches_keyword,
    get_keywords,
)
from tools.testing.target_determination.heuristics.historical_class_failure_correlation import (
    HistoricalClassFailurCorrelation,
)
from tools.testing.target_determination.heuristics.interface import TestPrioritizations
from tools.testing.target_determination.heuristics.previously_failed_in_pr import (
    get_previous_failures,
)
from tools.testing.test_run import TestRun


sys.path.remove(str(REPO_ROOT))

HEURISTIC_CLASS = "tools.testing.target_determination.heuristics.historical_class_failure_correlation."


def mocked_file(contents: dict[Any, Any]) -> io.IOBase:
    file_object = io.StringIO()
    json.dump(contents, file_object)
    file_object.seek(0)
    return file_object


def gen_historical_class_failures() -> dict[str, dict[str, float]]:
    return {
        "file1": {
            "test1::classA": 0.5,
            "test2::classA": 0.2,
            "test5::classB": 0.1,
        },
        "file2": {
            "test1::classB": 0.3,
            "test3::classA": 0.2,
            "test5::classA": 1.5,
            "test7::classC": 0.1,
        },
        "file3": {
            "test1::classC": 0.4,
            "test4::classA": 0.2,
            "test7::classC": 1.5,
            "test8::classC": 0.1,
        },
    }


ALL_TESTS = [
    "test1",
    "test2",
    "test3",
    "test4",
    "test5",
    "test6",
    "test7",
    "test8",
]


class TestHistoricalClassFailureCorrelation(TestTD):
    @mock.patch(
        HEURISTIC_CLASS + "_get_historical_test_class_correlations",
        return_value=gen_historical_class_failures(),
    )
    @mock.patch(
        HEURISTIC_CLASS + "query_changed_files",
        return_value=["file1"],
    )
    def test_get_prediction_confidence(
        self,
        historical_class_failures: dict[str, dict[str, float]],
        changed_files: list[str],
    ) -> None:
        tests_to_prioritize = ALL_TESTS

        heuristic = HistoricalClassFailurCorrelation()
        test_prioritizations = heuristic.get_prediction_confidence(tests_to_prioritize)

        expected = TestPrioritizations(
            tests_to_prioritize,
            {
                TestRun("test1::classA"): 0.25,
                TestRun("test2::classA"): 0.1,
                TestRun("test5::classB"): 0.05,
                TestRun("test1", excluded=["classA"]): 0.0,
                TestRun("test2", excluded=["classA"]): 0.0,
                TestRun("test3"): 0.0,
                TestRun("test4"): 0.0,
                TestRun("test5", excluded=["classB"]): 0.0,
                TestRun("test6"): 0.0,
                TestRun("test7"): 0.0,
                TestRun("test8"): 0.0,
            },
        )

        self.assert_test_scores_almost_equal(
            test_prioritizations._test_scores, expected._test_scores
        )


class TestParsePrevTests(TestTD):
    @mock.patch("os.path.exists", return_value=False)
    def test_cache_does_not_exist(self, mock_exists: Any) -> None:
        expected_failing_test_files: set[str] = set()

        found_tests = get_previous_failures()

        self.assertSetEqual(expected_failing_test_files, found_tests)

    @mock.patch("os.path.exists", return_value=True)
    @mock.patch("builtins.open", return_value=mocked_file({"": True}))
    def test_empty_cache(self, mock_exists: Any, mock_open: Any) -> None:
        expected_failing_test_files: set[str] = set()

        found_tests = get_previous_failures()

        self.assertSetEqual(expected_failing_test_files, found_tests)
        mock_open.assert_called()

    lastfailed_with_multiple_tests_per_file = {
        "test/test_car.py::TestCar::test_num[17]": True,
        "test/test_car.py::TestBar::test_num[25]": True,
        "test/test_far.py::TestFar::test_fun_copy[17]": True,
        "test/test_bar.py::TestBar::test_fun_copy[25]": True,
    }

    @mock.patch("os.path.exists", return_value=True)
    @mock.patch(
        "builtins.open",
        return_value=mocked_file(lastfailed_with_multiple_tests_per_file),
    )
    def test_dedupes_failing_test_files(self, mock_exists: Any, mock_open: Any) -> None:
        expected_failing_test_files = {"test_car", "test_bar", "test_far"}
        found_tests = get_previous_failures()

        self.assertSetEqual(expected_failing_test_files, found_tests)


class TestFilePath(TestTD):
    def test_get_keywords(self) -> None:
        self.assertEqual(get_keywords("test/test_car.py"), ["car"])
        self.assertEqual(get_keywords("test/nn/test_amp.py"), ["nn", "amp"])
        self.assertEqual(get_keywords("torch/nn/test_amp.py"), ["nn", "amp"])
        self.assertEqual(
            get_keywords("torch/nn/mixed_precision/test_something.py"),
            ["nn", "amp", "something"],
        )

    def test_match_keywords(self) -> None:
        self.assertTrue(file_matches_keyword("test/quantization/test_car.py", "quant"))
        self.assertTrue(file_matches_keyword("test/test_quantization.py", "quant"))
        self.assertTrue(file_matches_keyword("test/nn/test_amp.py", "nn"))
        self.assertTrue(file_matches_keyword("test/nn/test_amp.py", "amp"))
        self.assertTrue(file_matches_keyword("test/test_onnx.py", "onnx"))
        self.assertFalse(file_matches_keyword("test/test_onnx.py", "nn"))

    def test_get_keywords_match(self) -> None:
        def helper(test_file: str, changed_file: str) -> bool:
            return any(
                file_matches_keyword(test_file, x) for x in get_keywords(changed_file)
            )

        self.assertTrue(helper("test/quantization/test_car.py", "quantize/t.py"))
        self.assertFalse(helper("test/onnx/test_car.py", "nn/t.py"))
        self.assertTrue(helper("test/nn/test_car.py", "nn/t.py"))
        self.assertFalse(helper("test/nn/test_car.py", "test/b.py"))
        self.assertTrue(helper("test/test_mixed_precision.py", "torch/amp/t.py"))
        self.assertTrue(helper("test/test_amp.py", "torch/mixed_precision/t.py"))
        self.assertTrue(helper("test/idk/other/random.py", "torch/idk/t.py"))


if __name__ == "__main__":
    unittest.main()

```



## High-Level Overview


This Python file contains 3 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestHistoricalClassFailureCorrelation`, `TestParsePrevTests`, `TestFilePath`

**Functions defined**: `mocked_file`, `gen_historical_class_failures`, `test_get_prediction_confidence`, `test_cache_does_not_exist`, `test_empty_cache`, `test_dedupes_failing_test_files`, `test_get_keywords`, `test_match_keywords`, `test_get_keywords_match`, `helper`

**Key imports**: annotations, io, json, sys, unittest, Path, Any, mock, TestTD, TestPrioritizations


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/test/heuristics`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `io`
- `json`
- `sys`
- `unittest`
- `pathlib`: Path
- `typing`: Any
- `tools.test.heuristics.test_interface`: TestTD
- `tools.testing.target_determination.heuristics.interface`: TestPrioritizations
- `tools.testing.test_run`: TestRun


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python tools/test/heuristics/test_heuristics.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/test/heuristics`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_interface.py_docs.md`](./test_interface.py_docs.md)
- [`test_utils.py_docs.md`](./test_utils.py_docs.md)


## Cross-References

- **File Documentation**: `test_heuristics.py_docs.md`
- **Keyword Index**: `test_heuristics.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
