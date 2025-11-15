# Documentation: `tools/linter/adapters/testowners_linter.py`

## File Metadata

- **Path**: `tools/linter/adapters/testowners_linter.py`
- **Size**: 4,996 bytes (4.88 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3
"""
Test ownership was introduced in https://github.com/pytorch/pytorch/issues/66232.

This lint verifies that every Python test file (file that matches test_*.py or *_test.py in the test folder)
has valid ownership information in a comment header. Valid means:
  - The format of the header follows the pattern "# Owner(s): ["list", "of owner", "labels"]
  - Each owner label actually exists in PyTorch
  - Each owner label starts with "module: " or "oncall: " or is in ACCEPTABLE_OWNER_LABELS
"""

from __future__ import annotations

import argparse
import json
import urllib.error
from enum import Enum
from typing import Any, NamedTuple
from urllib.request import urlopen


LINTER_CODE = "TESTOWNERS"


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


def get_pytorch_labels() -> Any:
    url = "https://ossci-metrics.s3.amazonaws.com/pytorch_labels.json"
    try:
        labels = urlopen(url).read().decode("utf-8")
    except urllib.error.URLError:
        # This is an FB-only hack, if the json isn't available we may
        # need to use a forwarding proxy to get out
        proxy_url = "http://fwdproxy:8080"
        proxy_handler = urllib.request.ProxyHandler(
            {"http": proxy_url, "https": proxy_url}
        )
        context = urllib.request.build_opener(proxy_handler)
        labels = context.open(url).read().decode("utf-8")
    return json.loads(labels)


PYTORCH_LABELS = get_pytorch_labels()
# Team/owner labels usually start with "module: " or "oncall: ", but the following are acceptable exceptions
ACCEPTABLE_OWNER_LABELS = ["NNC", "high priority"]
OWNERS_PREFIX = "# Owner(s): "
GLOB_EXCEPTIONS = ["**/test/run_test.py"]


def check_labels(
    labels: list[str], filename: str, line_number: int
) -> list[LintMessage]:
    lint_messages = []
    for label in labels:
        if label not in PYTORCH_LABELS:
            lint_messages.append(
                LintMessage(
                    path=filename,
                    line=line_number,
                    char=None,
                    code=LINTER_CODE,
                    severity=LintSeverity.ERROR,
                    name="[invalid-label]",
                    original=None,
                    replacement=None,
                    description=(
                        f"{label} is not a PyTorch label "
                        "(please choose from https://github.com/pytorch/pytorch/labels)"
                    ),
                )
            )

        if label.startswith(("module:", "oncall:")) or label in ACCEPTABLE_OWNER_LABELS:
            continue

        lint_messages.append(
            LintMessage(
                path=filename,
                line=line_number,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="[invalid-owner]",
                original=None,
                replacement=None,
                description=(
                    f"{label} is not an acceptable owner "
                    "(please update to another label or edit ACCEPTABLE_OWNERS_LABELS "
                    "in tools/linters/adapters/testowners_linter.py)"
                ),
            )
        )

    return lint_messages


def check_file(filename: str) -> list[LintMessage]:
    lint_messages = []
    has_ownership_info = False

    with open(filename) as f:
        for idx, line in enumerate(f):
            if not line.startswith(OWNERS_PREFIX):
                continue

            has_ownership_info = True
            labels = json.loads(line[len(OWNERS_PREFIX) :])
            lint_messages.extend(check_labels(labels, filename, idx + 1))

    if has_ownership_info is False:
        lint_messages.append(
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.ERROR,
                name="[no-owner-info]",
                original=None,
                replacement=None,
                description="Missing a comment header with ownership information.",
            )
        )

    return lint_messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="test ownership linter",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )

    args = parser.parse_args()
    lint_messages = []

    for filename in args.filenames:
        lint_messages.extend(check_file(filename))

    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()

```



## High-Level Overview

"""Test ownership was introduced in https://github.com/pytorch/pytorch/issues/66232.This lint verifies that every Python test file (file that matches test_*.py or *_test.py in the test folder)has valid ownership information in a comment header. Valid means:  - The format of the header follows the pattern "# Owner(s): ["list", "of owner", "labels"]  - Each owner label actually exists in PyTorch  - Each owner label starts with "module: " or "oncall: " or is in ACCEPTABLE_OWNER_LABELS

This Python file contains 2 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LintSeverity`, `LintMessage`

**Functions defined**: `get_pytorch_labels`, `check_labels`, `check_file`, `main`

**Key imports**: annotations, argparse, json, urllib.error, Enum, Any, NamedTuple, urlopen


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `tools/linter/adapters`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `__future__`: annotations
- `argparse`
- `json`
- `urllib.error`
- `enum`: Enum
- `typing`: Any, NamedTuple
- `urllib.request`: urlopen


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
python tools/linter/adapters/testowners_linter.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`tools/linter/adapters`):

- [`grep_linter.py_docs.md`](./grep_linter.py_docs.md)
- [`import_linter.py_docs.md`](./import_linter.py_docs.md)
- [`gha_linter.py_docs.md`](./gha_linter.py_docs.md)
- [`actionlint_linter.py_docs.md`](./actionlint_linter.py_docs.md)
- [`pyfmt_linter.py_docs.md`](./pyfmt_linter.py_docs.md)
- [`mypy_linter.py_docs.md`](./mypy_linter.py_docs.md)
- [`no_merge_conflict_csv_linter.py_docs.md`](./no_merge_conflict_csv_linter.py_docs.md)
- [`no_workflows_on_fork.py_docs.md`](./no_workflows_on_fork.py_docs.md)
- [`bazel_linter.py_docs.md`](./bazel_linter.py_docs.md)
- [`test_device_bias_linter.py_docs.md`](./test_device_bias_linter.py_docs.md)


## Cross-References

- **File Documentation**: `testowners_linter.py_docs.md`
- **Keyword Index**: `testowners_linter.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
