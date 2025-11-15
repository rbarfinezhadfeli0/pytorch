# Documentation: `.github/scripts/test_check_labels.py`

## File Metadata

- **Path**: `.github/scripts/test_check_labels.py`
- **Size**: 5,241 bytes (5.12 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
"""test_check_labels.py"""

from typing import Any
from unittest import main, mock, TestCase

from check_labels import (
    add_label_err_comment,
    delete_all_label_err_comments,
    main as check_labels_main,
)
from github_utils import GitHubComment
from label_utils import BOT_AUTHORS, LABEL_ERR_MSG_TITLE
from test_trymerge import mock_gh_get_info, mocked_gh_graphql
from trymerge import GitHubPR


def mock_parse_args() -> object:
    class Object:
        def __init__(self) -> None:
            self.pr_num = 76123
            self.exit_non_zero = False

    return Object()


def mock_add_label_err_comment(pr: "GitHubPR") -> None:
    pass


def mock_delete_all_label_err_comments(pr: "GitHubPR") -> None:
    pass


def mock_get_comments() -> list[GitHubComment]:
    return [
        # Case 1 - a non label err comment
        GitHubComment(
            body_text="mock_body_text",
            created_at="",
            author_login="",
            author_url=None,
            author_association="",
            editor_login=None,
            database_id=1,
            url="",
        ),
        # Case 2 - a label err comment
        GitHubComment(
            body_text=" #" + LABEL_ERR_MSG_TITLE.replace("`", ""),
            created_at="",
            author_login=BOT_AUTHORS[1],
            author_url=None,
            author_association="",
            editor_login=None,
            database_id=2,
            url="",
        ),
    ]


class TestCheckLabels(TestCase):
    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch("trymerge.GitHubPR.get_comments", return_value=[mock_get_comments()[0]])
    @mock.patch("check_labels.gh_post_pr_comment")
    def test_correctly_add_label_err_comment(
        self, mock_gh_post_pr_comment: Any, mock_get_comments: Any, mock_gh_grphql: Any
    ) -> None:
        "Test add label err comment when similar comments don't exist."
        pr = GitHubPR("pytorch", "pytorch", 75095)
        add_label_err_comment(pr)
        mock_gh_post_pr_comment.assert_called_once()

    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch("trymerge.GitHubPR.get_comments", return_value=[mock_get_comments()[1]])
    @mock.patch("check_labels.gh_post_pr_comment")
    def test_not_add_label_err_comment(
        self, mock_gh_post_pr_comment: Any, mock_get_comments: Any, mock_gh_grphql: Any
    ) -> None:
        "Test not add label err comment when similar comments exist."
        pr = GitHubPR("pytorch", "pytorch", 75095)
        add_label_err_comment(pr)
        mock_gh_post_pr_comment.assert_not_called()

    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch("trymerge.GitHubPR.get_comments", return_value=mock_get_comments())
    @mock.patch("check_labels.gh_delete_comment")
    def test_correctly_delete_all_label_err_comments(
        self, mock_gh_delete_comment: Any, mock_get_comments: Any, mock_gh_grphql: Any
    ) -> None:
        "Test only delete label err comment."
        pr = GitHubPR("pytorch", "pytorch", 75095)
        delete_all_label_err_comments(pr)
        mock_gh_delete_comment.assert_called_once_with("pytorch", "pytorch", 2)

    @mock.patch("trymerge.gh_get_pr_info", return_value=mock_gh_get_info())
    @mock.patch("check_labels.parse_args", return_value=mock_parse_args())
    @mock.patch("check_labels.has_required_labels", return_value=False)
    @mock.patch(
        "check_labels.delete_all_label_err_comments",
        side_effect=mock_delete_all_label_err_comments,
    )
    @mock.patch(
        "check_labels.add_label_err_comment", side_effect=mock_add_label_err_comment
    )
    def test_ci_comments_and_exit0_without_required_labels(
        self,
        mock_add_label_err_comment: Any,
        mock_delete_all_label_err_comments: Any,
        mock_has_required_labels: Any,
        mock_parse_args: Any,
        mock_gh_get_info: Any,
    ) -> None:
        with self.assertRaises(SystemExit) as sys_exit:
            check_labels_main()
        self.assertEqual(str(sys_exit.exception), "0")
        mock_add_label_err_comment.assert_called_once()
        mock_delete_all_label_err_comments.assert_not_called()

    @mock.patch("trymerge.gh_get_pr_info", return_value=mock_gh_get_info())
    @mock.patch("check_labels.parse_args", return_value=mock_parse_args())
    @mock.patch("check_labels.has_required_labels", return_value=True)
    @mock.patch(
        "check_labels.delete_all_label_err_comments",
        side_effect=mock_delete_all_label_err_comments,
    )
    @mock.patch(
        "check_labels.add_label_err_comment", side_effect=mock_add_label_err_comment
    )
    def test_ci_exit0_with_required_labels(
        self,
        mock_add_label_err_comment: Any,
        mock_delete_all_label_err_comments: Any,
        mock_has_required_labels: Any,
        mock_parse_args: Any,
        mock_gh_get_info: Any,
    ) -> None:
        with self.assertRaises(SystemExit) as sys_exit:
            check_labels_main()
        self.assertEqual(str(sys_exit.exception), "0")
        mock_add_label_err_comment.assert_not_called()
        mock_delete_all_label_err_comments.assert_called_once()


if __name__ == "__main__":
    main()

```



## High-Level Overview

"""test_check_labels.py"""from typing import Anyfrom unittest import main, mock, TestCasefrom check_labels import (    add_label_err_comment,    delete_all_label_err_comments,    main as check_labels_main,)from github_utils import GitHubCommentfrom label_utils import BOT_AUTHORS, LABEL_ERR_MSG_TITLEfrom test_trymerge import mock_gh_get_info, mocked_gh_graphqlfrom trymerge import GitHubPRdef mock_parse_args() -> object:    class Object:        def __init__(self) -> None:            self.pr_num = 76123            self.exit_non_zero = False    return Object()def mock_add_label_err_comment(pr: "GitHubPR") -> None:    passdef mock_delete_all_label_err_comments(pr: "GitHubPR") -> None:    passdef mock_get_comments() -> list[GitHubComment]:    return [        # Case 1 - a non label err comment        GitHubComment(            body_text="mock_body_text",            created_at="",            author_login="",            author_url=None,            author_association="",            editor_login=None,            database_id=1,            url="",        ),        # Case 2 - a label err comment        GitHubComment(            body_text=" #" + LABEL_ERR_MSG_TITLE.replace("`", ""),            created_at="",

This Python file contains 2 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Object`, `TestCheckLabels`

**Functions defined**: `mock_parse_args`, `__init__`, `mock_add_label_err_comment`, `mock_delete_all_label_err_comments`, `mock_get_comments`, `test_correctly_add_label_err_comment`, `test_not_add_label_err_comment`, `test_correctly_delete_all_label_err_comments`, `test_ci_comments_and_exit0_without_required_labels`, `test_ci_exit0_with_required_labels`

**Key imports**: Any, main, mock, TestCase, GitHubComment, BOT_AUTHORS, LABEL_ERR_MSG_TITLE, mock_gh_get_info, mocked_gh_graphql, GitHubPR


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.github/scripts`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any
- `unittest`: main, mock, TestCase
- `github_utils`: GitHubComment
- `label_utils`: BOT_AUTHORS, LABEL_ERR_MSG_TITLE
- `test_trymerge`: mock_gh_get_info, mocked_gh_graphql
- `trymerge`: GitHubPR


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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
python .github/scripts/test_check_labels.py
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

- **File Documentation**: `test_check_labels.py_docs.md`
- **Keyword Index**: `test_check_labels.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
