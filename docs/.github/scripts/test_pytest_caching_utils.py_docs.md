# Documentation: `.github/scripts/test_pytest_caching_utils.py`

## File Metadata

- **Path**: `.github/scripts/test_pytest_caching_utils.py`
- **Size**: 3,157 bytes (3.08 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
from unittest import main, TestCase

from pytest_caching_utils import _merged_lastfailed_content


class TestPytestCachingUtils(TestCase):
    def test_merged_lastfailed_content_with_overlap(self) -> None:
        last_failed_source = {
            "tools/tests/test_foo.py::test_num1": True,
            "tools/tests/test_foo.py::test_num2": True,
            "tools/tests/test_bar.py::test_num1": True,
        }
        last_failed_dest = {
            "tools/tests/test_foo.py::test_num1": True,
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }
        last_failed_merged = {
            "tools/tests/test_foo.py::test_num1": True,
            "tools/tests/test_foo.py::test_num2": True,
            "tools/tests/test_bar.py::test_num1": True,
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }

        merged = _merged_lastfailed_content(last_failed_source, last_failed_dest)
        self.assertEqual(merged, last_failed_merged)

    def test_merged_lastfailed_content_without_overlap(self) -> None:
        last_failed_source = {
            "tools/tests/test_foo.py::test_num1": True,
            "tools/tests/test_foo.py::test_num2": True,
            "tools/tests/test_bar.py::test_num1": True,
        }
        last_failed_dest = {
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }
        last_failed_merged = {
            "tools/tests/test_foo.py::test_num1": True,
            "tools/tests/test_foo.py::test_num2": True,
            "tools/tests/test_bar.py::test_num1": True,
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }

        merged = _merged_lastfailed_content(last_failed_source, last_failed_dest)
        self.assertEqual(merged, last_failed_merged)

    def test_merged_lastfailed_content_with_empty_source(self) -> None:
        last_failed_source = {
            "": True,
        }
        last_failed_dest = {
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }
        last_failed_merged = {
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }

        merged = _merged_lastfailed_content(last_failed_source, last_failed_dest)
        self.assertEqual(merged, last_failed_merged)

    def test_merged_lastfailed_content_with_empty_dest(self) -> None:
        last_failed_source = {
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }
        last_failed_dest = {
            "": True,
        }
        last_failed_merged = {
            "tools/tests/test_car.py::test_num1": True,
            "tools/tests/test_car.py::test_num2": True,
        }

        merged = _merged_lastfailed_content(last_failed_source, last_failed_dest)
        self.assertEqual(merged, last_failed_merged)


if __name__ == "__main__":
    main()

```



## High-Level Overview


This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestPytestCachingUtils`

**Functions defined**: `test_merged_lastfailed_content_with_overlap`, `test_merged_lastfailed_content_without_overlap`, `test_merged_lastfailed_content_with_empty_source`, `test_merged_lastfailed_content_with_empty_dest`

**Key imports**: main, TestCase, _merged_lastfailed_content


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.github/scripts`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`: main, TestCase
- `pytest_caching_utils`: _merged_lastfailed_content


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
python .github/scripts/test_pytest_caching_utils.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`.github/scripts`):

- [`convert_lintrunner_annotations_to_github.py_docs.md`](./convert_lintrunner_annotations_to_github.py_docs.md)
- [`gitutils.py_docs.md`](./gitutils.py_docs.md)
- [`collect_ciflow_labels.py_docs.md`](./collect_ciflow_labels.py_docs.md)
- [`generate_docker_release_matrix.py_docs.md`](./generate_docker_release_matrix.py_docs.md)
- [`github_utils.py_docs.md`](./github_utils.py_docs.md)
- [`filter_test_configs.py_docs.md`](./filter_test_configs.py_docs.md)
- [`test_runner_determinator.py_docs.md`](./test_runner_determinator.py_docs.md)
- [`trymerge.py_docs.md`](./trymerge.py_docs.md)
- [`comment_on_pr.py_docs.md`](./comment_on_pr.py_docs.md)
- [`generate_binary_build_matrix.py_docs.md`](./generate_binary_build_matrix.py_docs.md)


## Cross-References

- **File Documentation**: `test_pytest_caching_utils.py_docs.md`
- **Keyword Index**: `test_pytest_caching_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
