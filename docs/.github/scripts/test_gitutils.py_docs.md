# Documentation: `.github/scripts/test_gitutils.py`

## File Metadata

- **Path**: `.github/scripts/test_gitutils.py`
- **Size**: 3,121 bytes (3.05 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is a **utility or tool script**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3
from pathlib import Path
from unittest import main, SkipTest, TestCase

from gitutils import (
    _shasum,
    are_ghstack_branches_in_sync,
    GitRepo,
    patterns_to_regex,
    PeekableIterator,
    retries_decorator,
)


BASE_DIR = Path(__file__).parent


class TestPeekableIterator(TestCase):
    def test_iterator(self, input_: str = "abcdef") -> None:
        iter_ = PeekableIterator(input_)
        for idx, c in enumerate(iter_):
            self.assertEqual(c, input_[idx])

    def test_is_iterable(self) -> None:
        from collections.abc import Iterator

        iter_ = PeekableIterator("")
        self.assertTrue(isinstance(iter_, Iterator))

    def test_peek(self, input_: str = "abcdef") -> None:
        iter_ = PeekableIterator(input_)
        for idx, c in enumerate(iter_):
            if idx + 1 < len(input_):
                self.assertEqual(iter_.peek(), input_[idx + 1])
            else:
                self.assertTrue(iter_.peek() is None)


class TestPattern(TestCase):
    def test_double_asterisks(self) -> None:
        allowed_patterns = [
            "aten/src/ATen/native/**LinearAlgebra*",
        ]
        patterns_re = patterns_to_regex(allowed_patterns)
        fnames = [
            "aten/src/ATen/native/LinearAlgebra.cpp",
            "aten/src/ATen/native/cpu/LinearAlgebraKernel.cpp",
        ]
        for filename in fnames:
            self.assertTrue(patterns_re.match(filename))


class TestRetriesDecorator(TestCase):
    def test_simple(self) -> None:
        @retries_decorator()
        def foo(x: int, y: int) -> int:
            return x + y

        self.assertEqual(foo(3, 4), 7)

    def test_fails(self) -> None:
        @retries_decorator(rc=0)
        def foo(x: int, y: int) -> int:
            return x + y

        self.assertEqual(foo("a", 4), 0)


class TestGitRepo(TestCase):
    def setUp(self) -> None:
        repo_dir = BASE_DIR.absolute().parent.parent
        if not (repo_dir / ".git").is_dir():
            raise SkipTest(
                "Can't find git directory, make sure to run this test on real repo checkout"
            )
        self.repo = GitRepo(str(repo_dir))

    def _skip_if_ref_does_not_exist(self, ref: str) -> None:
        """Skip test if ref is missing as stale branches are deleted with time"""
        try:
            self.repo.show_ref(ref)
        except RuntimeError as e:
            raise SkipTest(f"Can't find head ref {ref} due to {str(e)}") from e

    def test_compute_diff(self) -> None:
        diff = self.repo.diff("HEAD")
        sha = _shasum(diff)
        self.assertEqual(len(sha), 64)

    def test_ghstack_branches_in_sync(self) -> None:
        head_ref = "gh/SS-JIA/206/head"
        self._skip_if_ref_does_not_exist(head_ref)
        self.assertTrue(are_ghstack_branches_in_sync(self.repo, head_ref))

    def test_ghstack_branches_not_in_sync(self) -> None:
        head_ref = "gh/clee2000/1/head"
        self._skip_if_ref_does_not_exist(head_ref)
        self.assertFalse(are_ghstack_branches_in_sync(self.repo, head_ref))


if __name__ == "__main__":
    main()

```



## High-Level Overview


This Python file contains 4 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestPeekableIterator`, `TestPattern`, `TestRetriesDecorator`, `TestGitRepo`

**Functions defined**: `test_iterator`, `test_is_iterable`, `test_peek`, `test_double_asterisks`, `test_simple`, `foo`, `test_fails`, `foo`, `setUp`, `_skip_if_ref_does_not_exist`, `test_compute_diff`, `test_ghstack_branches_in_sync`, `test_ghstack_branches_not_in_sync`

**Key imports**: Path, main, SkipTest, TestCase, Iterator


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.github/scripts`, which contains **development tools and scripts**.



## Dependencies

### Import Dependencies

This file imports:

- `pathlib`: Path
- `unittest`: main, SkipTest, TestCase
- `collections.abc`: Iterator


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
python .github/scripts/test_gitutils.py
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

- **File Documentation**: `test_gitutils.py_docs.md`
- **Keyword Index**: `test_gitutils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
