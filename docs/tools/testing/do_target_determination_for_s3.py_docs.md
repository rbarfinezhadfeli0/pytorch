# Documentation: `tools/testing/do_target_determination_for_s3.py`

## File Metadata

- **Path**: `tools/testing/do_target_determination_for_s3.py`
- **Size**: 2,375 bytes (2.32 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.stats.import_test_stats import (
    copy_additional_previous_failures,
    copy_pytest_cache,
    get_td_heuristic_historial_edited_files_json,
    get_td_heuristic_profiling_json,
    get_test_class_ratings,
    get_test_class_times,
    get_test_file_ratings,
    get_test_times,
)
from tools.stats.upload_metrics import emit_metric
from tools.testing.discover_tests import TESTS
from tools.testing.target_determination.determinator import get_test_prioritizations
from tools.testing.target_determination.heuristics.interface import (
    AggregatedHeuristics,
    TestPrioritizations,
)


sys.path.remove(str(REPO_ROOT))


def import_results() -> TestPrioritizations:
    if not (REPO_ROOT / ".additional_ci_files/td_results.json").exists():
        print("No TD results found")
        return TestPrioritizations([], {})
    with open(REPO_ROOT / ".additional_ci_files/td_results.json") as f:
        td_results = json.load(f)
        tp = TestPrioritizations.from_json(td_results)

    return tp


def main() -> None:
    selected_tests = TESTS

    aggregated_heuristics: AggregatedHeuristics = AggregatedHeuristics(selected_tests)

    get_test_times()
    get_test_class_times()
    get_test_file_ratings()
    get_test_class_ratings()
    get_td_heuristic_historial_edited_files_json()
    get_td_heuristic_profiling_json()
    copy_pytest_cache()
    copy_additional_previous_failures()

    aggregated_heuristics = get_test_prioritizations(selected_tests)

    test_prioritizations = aggregated_heuristics.get_aggregated_priorities()

    print("Aggregated Heuristics")
    print(test_prioritizations.get_info_str(verbose=False))

    if os.getenv("CI") == "true":
        print("Emitting metrics")
        # Split into 3 due to size constraints
        emit_metric(
            "td_results_final_test_prioritizations",
            {"test_prioritizations": test_prioritizations.to_json()},
        )
        emit_metric(
            "td_results_aggregated_heuristics",
            {"aggregated_heuristics": aggregated_heuristics.to_json()},
        )

    with open(REPO_ROOT / "td_results.json", "w") as f:
        f.write(json.dumps(test_prioritizations.to_json()))


if __name__ == "__main__":
    main()

```



## High-Level Overview


This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `import_results`, `main`

**Key imports**: json, os, sys, Path, emit_metric, TESTS, get_test_prioritizations


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/testing`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `json`
- `os`
- `sys`
- `pathlib`: Path
- `tools.stats.upload_metrics`: emit_metric
- `tools.testing.discover_tests`: TESTS
- `tools.testing.target_determination.determinator`: get_test_prioritizations


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
python tools/testing/do_target_determination_for_s3.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/testing`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_run.py_docs.md`](./test_run.py_docs.md)
- [`explicit_ci_jobs.py_docs.md`](./explicit_ci_jobs.py_docs.md)
- [`test_selections.py_docs.md`](./test_selections.py_docs.md)
- [`clickhouse.py_docs.md`](./clickhouse.py_docs.md)
- [`update_slow_tests.py_docs.md`](./update_slow_tests.py_docs.md)
- [`upload_artifacts.py_docs.md`](./upload_artifacts.py_docs.md)
- [`modulefinder_determinator.py_docs.md`](./modulefinder_determinator.py_docs.md)
- [`discover_tests.py_docs.md`](./discover_tests.py_docs.md)


## Cross-References

- **File Documentation**: `do_target_determination_for_s3.py_docs.md`
- **Keyword Index**: `do_target_determination_for_s3.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
