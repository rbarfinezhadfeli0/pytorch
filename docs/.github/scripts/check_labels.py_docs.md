# Documentation: `.github/scripts/check_labels.py`

## File Metadata

- **Path**: `.github/scripts/check_labels.py`
- **Size**: 1,958 bytes (1.91 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3
"""Check whether a PR has required labels."""

import sys
from typing import Any

from github_utils import gh_delete_comment, gh_post_pr_comment
from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo
from label_utils import has_required_labels, is_label_err_comment, LABEL_ERR_MSG
from trymerge import GitHubPR


def delete_all_label_err_comments(pr: "GitHubPR") -> None:
    for comment in pr.get_comments():
        if is_label_err_comment(comment):
            gh_delete_comment(pr.org, pr.project, comment.database_id)


def add_label_err_comment(pr: "GitHubPR") -> None:
    # Only make a comment if one doesn't exist already
    if not any(is_label_err_comment(comment) for comment in pr.get_comments()):
        gh_post_pr_comment(pr.org, pr.project, pr.pr_num, LABEL_ERR_MSG)


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser("Check PR labels")
    parser.add_argument("pr_num", type=int)
    # add a flag to return a non-zero exit code if the PR does not have the required labels
    parser.add_argument(
        "--exit-non-zero",
        action="store_true",
        help="Return a non-zero exit code if the PR does not have the required labels",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
    org, project = repo.gh_owner_and_name()
    pr = GitHubPR(org, project, args.pr_num)

    try:
        if not has_required_labels(pr):
            print(LABEL_ERR_MSG, flush=True)
            add_label_err_comment(pr)
            if args.exit_non_zero:
                raise RuntimeError("PR does not have required labels")
        else:
            delete_all_label_err_comments(pr)
    except Exception as e:
        if args.exit_non_zero:
            raise RuntimeError(f"Error checking labels: {e}") from e

    sys.exit(0)


if __name__ == "__main__":
    main()

```



## High-Level Overview

"""Check whether a PR has required labels."""import sysfrom typing import Anyfrom github_utils import gh_delete_comment, gh_post_pr_commentfrom gitutils import get_git_remote_name, get_git_repo_dir, GitRepofrom label_utils import has_required_labels, is_label_err_comment, LABEL_ERR_MSGfrom trymerge import GitHubPRdef delete_all_label_err_comments(pr: "GitHubPR") -> None:    for comment in pr.get_comments():        if is_label_err_comment(comment):            gh_delete_comment(pr.org, pr.project, comment.database_id)def add_label_err_comment(pr: "GitHubPR") -> None:    # Only make a comment if one doesn't exist already    if not any(is_label_err_comment(comment) for comment in pr.get_comments()):        gh_post_pr_comment(pr.org, pr.project, pr.pr_num, LABEL_ERR_MSG)def parse_args() -> Any:    from argparse import ArgumentParser    parser = ArgumentParser("Check PR labels")    parser.add_argument("pr_num", type=int)    # add a flag to return a non-zero exit code if the PR does not have the required labels    parser.add_argument(        "--exit-non-zero",        action="store_true",        help="Return a non-zero exit code if the PR does not have the required labels",    )    return parser.parse_args()def main() -> None:    args = parse_args()    repo = GitRepo(get_git_repo_dir(), get_git_remote_name())    org, project = repo.gh_owner_and_name()    pr = GitHubPR(org, project, args.pr_num)    try:        if not has_required_labels(pr):            print(LABEL_ERR_MSG, flush=True)            add_label_err_comment(pr)            if args.exit_non_zero:

This Python file contains 0 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `delete_all_label_err_comments`, `add_label_err_comment`, `parse_args`, `main`

**Key imports**: sys, Any, gh_delete_comment, gh_post_pr_comment, get_git_remote_name, get_git_repo_dir, GitRepo, has_required_labels, is_label_err_comment, LABEL_ERR_MSG, GitHubPR, ArgumentParser


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.github/scripts`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `typing`: Any
- `github_utils`: gh_delete_comment, gh_post_pr_comment
- `gitutils`: get_git_remote_name, get_git_repo_dir, GitRepo
- `label_utils`: has_required_labels, is_label_err_comment, LABEL_ERR_MSG
- `trymerge`: GitHubPR
- `argparse`: ArgumentParser


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

- **File Documentation**: `check_labels.py_docs.md`
- **Keyword Index**: `check_labels.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
