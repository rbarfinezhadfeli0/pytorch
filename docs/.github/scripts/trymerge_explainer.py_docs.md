# Documentation: `.github/scripts/trymerge_explainer.py`

## File Metadata

- **Path**: `.github/scripts/trymerge_explainer.py`
- **Size**: 3,259 bytes (3.18 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**.

## Original Source

```python
import os
import re
from re import Pattern
from typing import Optional


BOT_COMMANDS_WIKI = "https://github.com/pytorch/pytorch/wiki/Bot-commands"

CIFLOW_LABEL = re.compile(r"^ciflow/.+")
CIFLOW_TRUNK_LABEL = re.compile(r"^ciflow/trunk")

OFFICE_HOURS_LINK = "https://github.com/pytorch/pytorch/wiki/Dev-Infra-Office-Hours"
CONTACT_US = f"Questions? Feedback? Please reach out to the [PyTorch DevX Team]({OFFICE_HOURS_LINK})"
ALTERNATIVES = f"Learn more about merging in the [wiki]({BOT_COMMANDS_WIKI})."


def has_label(labels: list[str], pattern: Pattern[str] = CIFLOW_LABEL) -> bool:
    return len(list(filter(pattern.match, labels))) > 0


class TryMergeExplainer:
    force: bool
    labels: list[str]
    pr_num: int
    org: str
    project: str
    ignore_current: bool

    has_trunk_label: bool
    has_ciflow_label: bool

    def __init__(
        self,
        force: bool,
        labels: list[str],
        pr_num: int,
        org: str,
        project: str,
        ignore_current: bool,
    ):
        self.force = force
        self.labels = labels
        self.pr_num = pr_num
        self.org = org
        self.project = project
        self.ignore_current = ignore_current

    def _get_flag_msg(
        self,
        ignore_current_checks: Optional[
            list[tuple[str, Optional[str], Optional[int]]]
        ] = None,
    ) -> str:
        if self.force:
            return (
                "Your change will be merged immediately since you used the force (-f) flag, "
                + "**bypassing any CI checks** (ETA: 1-5 minutes).  "
                + "Please use `-f` as last resort and instead consider `-i/--ignore-current` "
                + "to continue the merge ignoring current failures.  This will allow "
                + "currently pending tests to finish and report signal before the merge."
            )
        elif self.ignore_current and ignore_current_checks is not None:
            msg = f"Your change will be merged while ignoring the following {len(ignore_current_checks)} checks: "
            msg += ", ".join(f"[{x[0]}]({x[1]})" for x in ignore_current_checks)
            return msg
        else:
            return "Your change will be merged once all checks pass (ETA 0-4 Hours)."

    def get_merge_message(
        self,
        ignore_current_checks: Optional[
            list[tuple[str, Optional[str], Optional[int]]]
        ] = None,
    ) -> str:
        title = "### Merge started"
        main_message = self._get_flag_msg(ignore_current_checks)

        advanced_debugging = "\n".join(
            (
                "<details><summary>Advanced Debugging</summary>",
                "Check the merge workflow status ",
                f'<a href="{os.getenv("GH_RUN_URL")}">here</a>',
                "</details>",
            )
        )

        msg = title + "\n"
        msg += main_message + "\n\n"
        msg += ALTERNATIVES + "\n\n"
        msg += CONTACT_US
        msg += advanced_debugging
        return msg


def get_revert_message(org: str, project: str, pr_num: int) -> str:
    msg = (
        "@pytorchbot successfully started a revert job."
        + f" Check the current status [here]({os.getenv('GH_RUN_URL')}).\n"
    )
    msg += CONTACT_US
    return msg

```



## High-Level Overview


This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TryMergeExplainer`

**Functions defined**: `has_label`, `__init__`, `_get_flag_msg`, `get_merge_message`, `get_revert_message`

**Key imports**: os, re, Pattern, Optional


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.github/scripts`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `re`
- `typing`: Optional


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

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

- **File Documentation**: `trymerge_explainer.py_docs.md`
- **Keyword Index**: `trymerge_explainer.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
