# Documentation: `.github/scripts/collect_ciflow_labels.py`

## File Metadata

- **Path**: `.github/scripts/collect_ciflow_labels.py`
- **Size**: 2,510 bytes (2.45 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3

import sys
from pathlib import Path
from typing import Any, cast

import yaml


GITHUB_DIR = Path(__file__).parent.parent


def get_workflows_push_tags() -> set[str]:
    "Extract all known push tags from workflows"
    rc: set[str] = set()
    for fname in (GITHUB_DIR / "workflows").glob("*.yml"):
        with fname.open("r") as f:
            wf_yml = yaml.safe_load(f)
        # "on" is alias to True in yaml
        on_tag = wf_yml.get(True, None)
        push_tag = on_tag.get("push", None) if isinstance(on_tag, dict) else None
        tags_tag = push_tag.get("tags", None) if isinstance(push_tag, dict) else None
        if isinstance(tags_tag, list):
            rc.update(tags_tag)
    return rc


def filter_ciflow_tags(tags: set[str]) -> list[str]:
    "Return sorted list of ciflow tags"
    return sorted(
        tag[:-2] for tag in tags if tag.startswith("ciflow/") and tag.endswith("/*")
    )


def read_probot_config() -> dict[str, Any]:
    with (GITHUB_DIR / "pytorch-probot.yml").open("r") as f:
        return cast(dict[str, Any], yaml.safe_load(f))


def update_probot_config(labels: set[str]) -> None:
    orig = read_probot_config()
    orig["ciflow_push_tags"] = filter_ciflow_tags(labels)
    with (GITHUB_DIR / "pytorch-probot.yml").open("w") as f:
        yaml.dump(orig, f, indent=4, sort_keys=False)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser("Validate or update list of tags")
    parser.add_argument("--validate-tags", action="store_true")
    args = parser.parse_args()
    pushtags = get_workflows_push_tags()
    if args.validate_tags:
        config = read_probot_config()
        ciflow_tags = set(filter_ciflow_tags(pushtags))
        config_tags = set(config["ciflow_push_tags"])
        if config_tags != ciflow_tags:
            print("Tags mismatch!")
            if ciflow_tags.difference(config_tags):
                print(
                    "Reference in workflows but not in config",
                    ciflow_tags.difference(config_tags),
                )
            if config_tags.difference(ciflow_tags):
                print(
                    "Reference in config, but not in workflows",
                    config_tags.difference(ciflow_tags),
                )
            print(f"Please run {__file__} to remediate the difference")
            sys.exit(-1)
        print("All tags are listed in pytorch-probot.yml")
    else:
        update_probot_config(pushtags)

```



## High-Level Overview


This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `get_workflows_push_tags`, `filter_ciflow_tags`, `read_probot_config`, `update_probot_config`

**Key imports**: sys, Path, Any, cast, yaml, ArgumentParser


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.github/scripts`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `pathlib`: Path
- `typing`: Any, cast
- `yaml`
- `argparse`: ArgumentParser


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
- [`generate_docker_release_matrix.py_docs.md`](./generate_docker_release_matrix.py_docs.md)
- [`github_utils.py_docs.md`](./github_utils.py_docs.md)
- [`filter_test_configs.py_docs.md`](./filter_test_configs.py_docs.md)
- [`test_runner_determinator.py_docs.md`](./test_runner_determinator.py_docs.md)
- [`trymerge.py_docs.md`](./trymerge.py_docs.md)
- [`comment_on_pr.py_docs.md`](./comment_on_pr.py_docs.md)
- [`generate_binary_build_matrix.py_docs.md`](./generate_binary_build_matrix.py_docs.md)


## Cross-References

- **File Documentation**: `collect_ciflow_labels.py_docs.md`
- **Keyword Index**: `collect_ciflow_labels.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
