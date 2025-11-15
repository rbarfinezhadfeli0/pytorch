# Documentation: `.github/scripts/ensure_actions_will_cancel.py`

## File Metadata

- **Path**: `.github/scripts/ensure_actions_will_cancel.py`
- **Size**: 2,515 bytes (2.46 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3

import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
EXPECTED_GROUP_PREFIX = (
    "${{ github.workflow }}-${{ github.event.pull_request.number || github.sha }}"
)
EXPECTED_GROUP = (
    EXPECTED_GROUP_PREFIX + "-${{ github.event_name == 'workflow_dispatch' }}"
)


def should_check(filename: Path) -> bool:
    with open(filename) as f:
        content = f.read()

    data = yaml.safe_load(content)
    on = data.get("on", data.get(True, {}))
    return "pull_request" in on


if __name__ == "__main__":
    errors_found = False
    files = [f for f in WORKFLOWS.glob("*.yml") if should_check(f)]
    names = set()
    for filename in files:
        with open(filename) as f:
            data = yaml.safe_load(f)

        name = data.get("name")
        if name is not None and name in names:
            print("ERROR: duplicate workflow name:", name, file=sys.stderr)
            errors_found = True
        names.add(name)
        actual = data.get("concurrency", {})
        if filename.name == "create_release.yml":
            if not actual.get("group", "").startswith(EXPECTED_GROUP_PREFIX):
                print(
                    f"'concurrency' incorrect or not found in '{filename.relative_to(REPO_ROOT)}'",
                    file=sys.stderr,
                )
                print(
                    f"concurrency group should start with {EXPECTED_GROUP_PREFIX} but found {actual.get('group', None)}",
                    file=sys.stderr,
                )
                errors_found = True
        elif not actual.get("group", "").startswith(EXPECTED_GROUP):
            print(
                f"'concurrency' incorrect or not found in '{filename.relative_to(REPO_ROOT)}'",
                file=sys.stderr,
            )
            print(
                f"concurrency group should start with {EXPECTED_GROUP} but found {actual.get('group', None)}",
                file=sys.stderr,
            )
            errors_found = True
        if not actual.get("cancel-in-progress", False):
            print(
                f"'concurrency' incorrect or not found in '{filename.relative_to(REPO_ROOT)}'",
                file=sys.stderr,
            )
            print(
                f"concurrency cancel-in-progress should be True but found {actual.get('cancel-in-progress', None)}",
                file=sys.stderr,
            )

    if errors_found:
        sys.exit(1)

```



## High-Level Overview


This Python file contains 0 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `should_check`

**Key imports**: sys, Path, yaml


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.github/scripts`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `pathlib`: Path
- `yaml`


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

- **File Documentation**: `ensure_actions_will_cancel.py_docs.md`
- **Keyword Index**: `ensure_actions_will_cancel.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
