# Documentation: `tools/test/test_test_run.py`

## File Metadata

- **Path**: `tools/test/test_test_run.py`
- **Size**: 5,739 bytes (5.60 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    # using tools/ to optimize test run.
    sys.path.append(str(REPO_ROOT))
    from tools.testing.test_run import ShardedTest, TestRun
except ModuleNotFoundError:
    print("Can't import required modules, exiting")
    sys.exit(1)


class TestTestRun(unittest.TestCase):
    def test_union_with_full_run(self) -> None:
        run1 = TestRun("foo")
        run2 = TestRun("foo::bar")

        self.assertEqual(run1 | run2, run1)
        self.assertEqual(run2 | run1, run1)

    def test_union_with_inclusions(self) -> None:
        run1 = TestRun("foo::bar")
        run2 = TestRun("foo::baz")

        expected = TestRun("foo", included=["bar", "baz"])

        self.assertEqual(run1 | run2, expected)
        self.assertEqual(run2 | run1, expected)

    def test_union_with_non_overlapping_exclusions(self) -> None:
        run1 = TestRun("foo", excluded=["bar"])
        run2 = TestRun("foo", excluded=["baz"])

        expected = TestRun("foo")

        self.assertEqual(run1 | run2, expected)
        self.assertEqual(run2 | run1, expected)

    def test_union_with_overlapping_exclusions(self) -> None:
        run1 = TestRun("foo", excluded=["bar", "car"])
        run2 = TestRun("foo", excluded=["bar", "caz"])

        expected = TestRun("foo", excluded=["bar"])

        self.assertEqual(run1 | run2, expected)
        self.assertEqual(run2 | run1, expected)

    def test_union_with_mixed_inclusion_exclusions(self) -> None:
        run1 = TestRun("foo", excluded=["baz", "car"])
        run2 = TestRun("foo", included=["baz"])

        expected = TestRun("foo", excluded=["car"])

        self.assertEqual(run1 | run2, expected)
        self.assertEqual(run2 | run1, expected)

    def test_union_with_mixed_files_fails(self) -> None:
        run1 = TestRun("foo")
        run2 = TestRun("bar")

        with self.assertRaises(AssertionError):
            run1 | run2

    def test_union_with_empty_file_yields_orig_file(self) -> None:
        run1 = TestRun("foo")
        run2 = TestRun.empty()

        self.assertEqual(run1 | run2, run1)
        self.assertEqual(run2 | run1, run1)

    def test_subtracting_full_run_fails(self) -> None:
        run1 = TestRun("foo::bar")
        run2 = TestRun("foo")

        self.assertEqual(run1 - run2, TestRun.empty())

    def test_subtracting_empty_file_yields_orig_file(self) -> None:
        run1 = TestRun("foo")
        run2 = TestRun.empty()

        self.assertEqual(run1 - run2, run1)
        self.assertEqual(run2 - run1, TestRun.empty())

    def test_empty_is_falsey(self) -> None:
        self.assertFalse(TestRun.empty())

    def test_subtracting_inclusion_from_full_run(self) -> None:
        run1 = TestRun("foo")
        run2 = TestRun("foo::bar")

        expected = TestRun("foo", excluded=["bar"])

        self.assertEqual(run1 - run2, expected)

    def test_subtracting_inclusion_from_overlapping_inclusion(self) -> None:
        run1 = TestRun("foo", included=["bar", "baz"])
        run2 = TestRun("foo::baz")

        self.assertEqual(run1 - run2, TestRun("foo", included=["bar"]))

    def test_subtracting_inclusion_from_nonoverlapping_inclusion(self) -> None:
        run1 = TestRun("foo", included=["bar", "baz"])
        run2 = TestRun("foo", included=["car"])

        self.assertEqual(run1 - run2, TestRun("foo", included=["bar", "baz"]))

    def test_subtracting_exclusion_from_full_run(self) -> None:
        run1 = TestRun("foo")
        run2 = TestRun("foo", excluded=["bar"])

        self.assertEqual(run1 - run2, TestRun("foo", included=["bar"]))

    def test_subtracting_exclusion_from_superset_exclusion(self) -> None:
        run1 = TestRun("foo", excluded=["bar", "baz"])
        run2 = TestRun("foo", excluded=["baz"])

        self.assertEqual(run1 - run2, TestRun.empty())
        self.assertEqual(run2 - run1, TestRun("foo", included=["bar"]))

    def test_subtracting_exclusion_from_nonoverlapping_exclusion(self) -> None:
        run1 = TestRun("foo", excluded=["bar", "baz"])
        run2 = TestRun("foo", excluded=["car"])

        self.assertEqual(run1 - run2, TestRun("foo", included=["car"]))
        self.assertEqual(run2 - run1, TestRun("foo", included=["bar", "baz"]))

    def test_subtracting_inclusion_from_exclusion_without_overlaps(self) -> None:
        run1 = TestRun("foo", excluded=["bar", "baz"])
        run2 = TestRun("foo", included=["bar"])

        self.assertEqual(run1 - run2, run1)
        self.assertEqual(run2 - run1, run2)

    def test_subtracting_inclusion_from_exclusion_with_overlaps(self) -> None:
        run1 = TestRun("foo", excluded=["bar", "baz"])
        run2 = TestRun("foo", included=["bar", "car"])

        self.assertEqual(run1 - run2, TestRun("foo", excluded=["bar", "baz", "car"]))
        self.assertEqual(run2 - run1, TestRun("foo", included=["bar"]))

    def test_and(self) -> None:
        run1 = TestRun("foo", included=["bar", "baz"])
        run2 = TestRun("foo", included=["bar", "car"])

        self.assertEqual(run1 & run2, TestRun("foo", included=["bar"]))

    def test_and_exclusions(self) -> None:
        run1 = TestRun("foo", excluded=["bar", "baz"])
        run2 = TestRun("foo", excluded=["bar", "car"])

        self.assertEqual(run1 & run2, TestRun("foo", excluded=["bar", "baz", "car"]))


class TestShardedTest(unittest.TestCase):
    def test_get_pytest_args(self) -> None:
        test = TestRun("foo", included=["bar", "baz"])
        sharded_test = ShardedTest(test, 1, 1)

        expected_args = ["-k", "bar or baz"]

        self.assertListEqual(sharded_test.get_pytest_args(), expected_args)


if __name__ == "__main__":
    unittest.main()

```



## High-Level Overview


This Python file contains 2 class(es) and 21 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestTestRun`, `TestShardedTest`

**Functions defined**: `test_union_with_full_run`, `test_union_with_inclusions`, `test_union_with_non_overlapping_exclusions`, `test_union_with_overlapping_exclusions`, `test_union_with_mixed_inclusion_exclusions`, `test_union_with_mixed_files_fails`, `test_union_with_empty_file_yields_orig_file`, `test_subtracting_full_run_fails`, `test_subtracting_empty_file_yields_orig_file`, `test_empty_is_falsey`, `test_subtracting_inclusion_from_full_run`, `test_subtracting_inclusion_from_overlapping_inclusion`, `test_subtracting_inclusion_from_nonoverlapping_inclusion`, `test_subtracting_exclusion_from_full_run`, `test_subtracting_exclusion_from_superset_exclusion`, `test_subtracting_exclusion_from_nonoverlapping_exclusion`, `test_subtracting_inclusion_from_exclusion_without_overlaps`, `test_subtracting_inclusion_from_exclusion_with_overlaps`, `test_and`, `test_and_exclusions`

**Key imports**: sys, unittest, Path, ShardedTest, TestRun, required modules, exiting


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `unittest`
- `pathlib`: Path
- `tools.testing.test_run`: ShardedTest, TestRun
- `required modules, exiting`


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
python tools/test/test_test_run.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/test`):

- [`test_upload_stats_lib.py_docs.md`](./test_upload_stats_lib.py_docs.md)
- [`test_codegen.py_docs.md`](./test_codegen.py_docs.md)
- [`linter_test_case.py_docs.md`](./linter_test_case.py_docs.md)
- [`test_upload_gate.py_docs.md`](./test_upload_gate.py_docs.md)
- [`test_gen_backend_stubs.py_docs.md`](./test_gen_backend_stubs.py_docs.md)
- [`test_gb_registry_linter.py_docs.md`](./test_gb_registry_linter.py_docs.md)
- [`test_utils.py_docs.md`](./test_utils.py_docs.md)
- [`test_set_linter.py_docs.md`](./test_set_linter.py_docs.md)
- [`gen_oplist_test.py_docs.md`](./gen_oplist_test.py_docs.md)
- [`test_upload_test_stats.py_docs.md`](./test_upload_test_stats.py_docs.md)


## Cross-References

- **File Documentation**: `test_test_run.py_docs.md`
- **Keyword Index**: `test_test_run.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
