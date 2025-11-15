# Documentation: `scripts/compile_tests/passrate.py`

## File Metadata

- **Path**: `scripts/compile_tests/passrate.py`
- **Size**: 3,070 bytes (3.00 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
import argparse

from common import (
    get_excluded_testcases,
    get_passed_testcases,
    get_testcases,
    key,
    open_test_results,
)
from download_reports import download_reports


"""
Usage: passrate.py commit_sha

Parses test reports to measure the passrate. The passrate is defined as:

A) Take the number of tests that pass under eager mode, excluding
CUDA, OpInfo, and ModuleInfo tests
B) Of those tests, count the number of tests that pass under Dynamo
C) Take B/A.

You'll need to provide the commit_sha for a commit on the main branch,
from which we will pull CI test results.

This script requires the `gh` cli. You'll need to install it and then
authenticate with it via `gh auth login` before using this script.
https://docs.github.com/en/github-cli/github-cli/quickstart
"""


def testcases_by_time(xmls):
    testcases = get_testcases(xmls)
    testcases.sort(reverse=True, key=lambda x: float(x.attrib["time"]))
    return testcases


def should_exclude(key):
    test_file = key.split("::")[0]
    # C++ tests
    if test_file == "UNKNOWN":
        return True
    # Policy: "pass rate" does not include inductor, export, or dynamo tests.
    return test_file.startswith(("inductor/", "export/", "dynamo/"))


def compute_pass_rate(eager_dir, dynamo_dir):
    print("parsing xmls")
    eager_xmls = open_test_results(eager_dir)
    dynamo_xmls = open_test_results(dynamo_dir)

    print("computing pass rate")
    eager_passed = get_passed_testcases(eager_xmls)
    dynamo_passed = get_passed_testcases(dynamo_xmls)
    dynamo_pass_keys = {key(testcase) for testcase in dynamo_passed}
    dynamo_pass_keys = {key_ for key_ in dynamo_pass_keys if not should_exclude(key_)}
    tmp_eager_pass_keys = {key(testcase) for testcase in eager_passed}
    tmp_eager_pass_keys = {
        key_ for key_ in tmp_eager_pass_keys if not should_exclude(key_)
    }
    excluded = [key(t) for t in get_excluded_testcases(dynamo_xmls)]
    eager_pass_keys = tmp_eager_pass_keys - set(excluded)

    subset = eager_pass_keys.intersection(dynamo_pass_keys)
    total_subset = len(subset)
    total_tests = len(eager_pass_keys)
    print("pass rate", total_subset / total_tests, total_subset, total_tests)

    dynamo_testcases = get_testcases(dynamo_xmls)
    tc = {key(t): t for t in dynamo_testcases}

    # Useful for debugging
    not_there_keys = set()
    for key_ in eager_pass_keys:
        if key_ not in tc:
            not_there_keys.add(key_)

    fail_keys = eager_pass_keys - subset
    return fail_keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="passrate", description="Computes the Dynamo unittest pass rate"
    )
    parser.add_argument(
        "commit",
        help=(
            "The commit sha for the latest commit on a PR from which we will "
            "pull CI test results, e.g. 7e5f597aeeba30c390c05f7d316829b3798064a5"
        ),
    )
    args = parser.parse_args()
    dynamo311, eager311 = download_reports(args.commit, ("dynamo311", "eager311"))
    compute_pass_rate(eager311, dynamo311)

```



## High-Level Overview

"""Usage: passrate.py commit_shaParses test reports to measure the passrate. The passrate is defined as:A) Take the number of tests that pass under eager mode, excludingCUDA, OpInfo, and ModuleInfo testsB) Of those tests, count the number of tests that pass under DynamoC) Take B/A.You'll need to provide the commit_sha for a commit on the main branch,from which we will pull CI test results.This script requires the `gh` cli. You'll need to install it and thenauthenticate with it via `gh auth login` before using this script.https://docs.github.com/en/github-cli/github-cli/quickstart

This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `testcases_by_time`, `should_exclude`, `compute_pass_rate`

**Key imports**: argparse, download_reports


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `scripts/compile_tests`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `download_reports`: download_reports


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python scripts/compile_tests/passrate.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`scripts/compile_tests`):

- [`update_failures.py_docs.md`](./update_failures.py_docs.md)
- [`failures_histogram.py_docs.md`](./failures_histogram.py_docs.md)
- [`common.py_docs.md`](./common.py_docs.md)
- [`download_reports.py_docs.md`](./download_reports.py_docs.md)


## Cross-References

- **File Documentation**: `passrate.py_docs.md`
- **Keyword Index**: `passrate.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
