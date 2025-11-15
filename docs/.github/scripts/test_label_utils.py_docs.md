# Documentation: `.github/scripts/test_label_utils.py`

## File Metadata

- **Path**: `.github/scripts/test_label_utils.py`
- **Size**: 3,360 bytes (3.28 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
from typing import Any
from unittest import main, mock, TestCase

from label_utils import (
    get_last_page_num_from_header,
    gh_get_labels,
    has_required_labels,
)
from test_trymerge import mocked_gh_graphql
from trymerge import GitHubPR


release_notes_labels = [
    "release notes: nn",
]


class TestLabelUtils(TestCase):
    MOCK_HEADER_LINKS_TO_PAGE_NUMS = {
        1: {
            "link": "<https://api.github.com/dummy/labels?per_page=10&page=1>; rel='last'"
        },
        2: {"link": "<https://api.github.com/dummy/labels?per_page=1&page=2>;"},
        3: {"link": "<https://api.github.com/dummy/labels?per_page=1&page=2&page=3>;"},
    }

    def test_get_last_page_num_from_header(self) -> None:
        for (
            expected_page_num,
            mock_header,
        ) in self.MOCK_HEADER_LINKS_TO_PAGE_NUMS.items():
            self.assertEqual(
                get_last_page_num_from_header(mock_header), expected_page_num
            )

    MOCK_LABEL_INFO = '[{"name": "foo"}]'

    @mock.patch("label_utils.get_last_page_num_from_header", return_value=3)
    @mock.patch("label_utils.request_for_labels", return_value=(None, MOCK_LABEL_INFO))
    def test_gh_get_labels(
        self,
        mock_request_for_labels: Any,
        mock_get_last_page_num_from_header: Any,
    ) -> None:
        res = gh_get_labels("mock_org", "mock_repo")
        mock_get_last_page_num_from_header.assert_called_once()
        self.assertEqual(res, ["foo"] * 3)

    @mock.patch("label_utils.get_last_page_num_from_header", return_value=0)
    @mock.patch("label_utils.request_for_labels", return_value=(None, MOCK_LABEL_INFO))
    def test_gh_get_labels_raises_with_no_pages(
        self,
        mock_request_for_labels: Any,
        get_last_page_num_from_header: Any,
    ) -> None:
        with self.assertRaises(AssertionError) as err:
            gh_get_labels("foo", "bar")
        self.assertIn("number of pages of labels", str(err.exception))

    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch(
        "label_utils.get_release_notes_labels", return_value=release_notes_labels
    )
    def test_pr_with_missing_labels(
        self, mocked_rn_labels: Any, mocked_gql: Any
    ) -> None:
        "Test PR with no 'release notes:' label or 'topic: not user facing' label"
        pr = GitHubPR("pytorch", "pytorch", 82169)
        self.assertFalse(has_required_labels(pr))

    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch(
        "label_utils.get_release_notes_labels", return_value=release_notes_labels
    )
    def test_pr_with_release_notes_label(
        self, mocked_rn_labels: Any, mocked_gql: Any
    ) -> None:
        "Test PR with 'release notes: nn' label"
        pr = GitHubPR("pytorch", "pytorch", 71759)
        self.assertTrue(has_required_labels(pr))

    @mock.patch("trymerge.gh_graphql", side_effect=mocked_gh_graphql)
    @mock.patch(
        "label_utils.get_release_notes_labels", return_value=release_notes_labels
    )
    def test_pr_with_not_user_facing_label(
        self, mocked_rn_labels: Any, mocked_gql: Any
    ) -> None:
        "Test PR with 'topic: not user facing' label"
        pr = GitHubPR("pytorch", "pytorch", 75095)
        self.assertTrue(has_required_labels(pr))


if __name__ == "__main__":
    main()

```



## High-Level Overview


This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestLabelUtils`

**Functions defined**: `test_get_last_page_num_from_header`, `test_gh_get_labels`, `test_gh_get_labels_raises_with_no_pages`, `test_pr_with_missing_labels`, `test_pr_with_release_notes_label`, `test_pr_with_not_user_facing_label`

**Key imports**: Any, main, mock, TestCase, mocked_gh_graphql, GitHubPR


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.github/scripts`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Any
- `unittest`: main, mock, TestCase
- `test_trymerge`: mocked_gh_graphql
- `trymerge`: GitHubPR


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
python .github/scripts/test_label_utils.py
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

- **File Documentation**: `test_label_utils.py_docs.md`
- **Keyword Index**: `test_label_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
