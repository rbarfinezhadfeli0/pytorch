# Documentation: `scripts/compile_tests/common.py`

## File Metadata

- **Path**: `scripts/compile_tests/common.py`
- **Size**: 3,536 bytes (3.45 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**.

## Original Source

```python
import functools
import os
import warnings


try:
    import lxml.etree

    p = lxml.etree.XMLParser(huge_tree=True)
    parse = functools.partial(lxml.etree.parse, parser=p)
except ImportError:
    import xml.etree.ElementTree as ET

    parse = ET.parse
    warnings.warn(
        "lxml was not found. `pip install lxml` to make this script run much faster"
    )


def open_test_results(directory):
    xmls = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".xml"):
                tree = parse(f"{root}/{file}")
                xmls.append(tree)
    return xmls


def get_testcases(xmls):
    testcases = []
    for xml in xmls:
        root = xml.getroot()
        testcases.extend(list(root.iter("testcase")))
    return testcases


def find(testcase, condition):
    children = list(testcase.iter())
    assert children[0] is testcase
    children = children[1:]
    return condition(children)


def skipped_test(testcase):
    def condition(children):
        return "skipped" in {child.tag for child in children}

    return find(testcase, condition)


def passed_test(testcase):
    def condition(children):
        if len(children) == 0:
            return True
        tags = {child.tag for child in children}
        return "skipped" not in tags and "failed" not in tags

    return find(testcase, condition)


def key(testcase):
    file = testcase.attrib.get("file", "UNKNOWN")
    classname = testcase.attrib["classname"]
    name = testcase.attrib["name"]
    return "::".join([file, classname, name])


def get_passed_testcases(xmls):
    testcases = get_testcases(xmls)
    passed_testcases = [testcase for testcase in testcases if passed_test(testcase)]
    return passed_testcases


def get_excluded_testcases(xmls):
    testcases = get_testcases(xmls)
    excluded_testcases = [t for t in testcases if excluded_testcase(t)]
    return excluded_testcases


def excluded_testcase(testcase):
    def condition(children):
        for child in children:
            if child.tag == "skipped":
                if "Policy: we don't run" in child.attrib["message"]:
                    return True
        return False

    return find(testcase, condition)


def is_unexpected_success(testcase):
    def condition(children):
        for child in children:
            if child.tag != "failure":
                continue
            is_unexpected_success = (
                "unexpected success" in child.attrib["message"].lower()
            )
            if is_unexpected_success:
                return True
        return False

    return find(testcase, condition)


MSG = "This test passed, maybe we can remove the skip from dynamo_test_failures.py"


def is_passing_skipped_test(testcase):
    def condition(children):
        for child in children:
            if child.tag != "skipped":
                continue
            has_passing_skipped_test_msg = MSG in child.attrib["message"]
            if has_passing_skipped_test_msg:
                return True
        return False

    return find(testcase, condition)


# NB: not an unexpected success
def is_failure(testcase):
    def condition(children):
        for child in children:
            if child.tag != "failure":
                continue
            is_unexpected_success = (
                "unexpected success" in child.attrib["message"].lower()
            )
            if not is_unexpected_success:
                return True
        return False

    return find(testcase, condition)

```



## High-Level Overview


This Python file contains 0 class(es) and 18 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `open_test_results`, `get_testcases`, `find`, `skipped_test`, `condition`, `passed_test`, `condition`, `key`, `get_passed_testcases`, `get_excluded_testcases`, `excluded_testcase`, `condition`, `is_unexpected_success`, `condition`, `is_passing_skipped_test`, `condition`, `is_failure`, `condition`

**Key imports**: functools, os, warnings, lxml.etree, xml.etree.ElementTree as ET


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `scripts/compile_tests`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `os`
- `warnings`
- `lxml.etree`
- `xml.etree.ElementTree as ET`


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
python scripts/compile_tests/common.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`scripts/compile_tests`):

- [`update_failures.py_docs.md`](./update_failures.py_docs.md)
- [`failures_histogram.py_docs.md`](./failures_histogram.py_docs.md)
- [`passrate.py_docs.md`](./passrate.py_docs.md)
- [`download_reports.py_docs.md`](./download_reports.py_docs.md)


## Cross-References

- **File Documentation**: `common.py_docs.md`
- **Keyword Index**: `common.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
