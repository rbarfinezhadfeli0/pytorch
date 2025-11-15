# Documentation: `test/package/test_glob_group.py`

## File Metadata

- **Path**: `test/package/test_glob_group.py`
- **Size**: 4,374 bytes (4.27 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: package/deploy"]

from collections.abc import Iterable

from torch.package import GlobGroup
from torch.testing._internal.common_utils import run_tests


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestGlobGroup(PackageTestCase):
    def assertMatchesGlob(self, glob: GlobGroup, candidates: Iterable[str]):
        for candidate in candidates:
            self.assertTrue(glob.matches(candidate))

    def assertNotMatchesGlob(self, glob: GlobGroup, candidates: Iterable[str]):
        for candidate in candidates:
            self.assertFalse(glob.matches(candidate))

    def test_one_star(self):
        glob_group = GlobGroup("torch.*")
        self.assertMatchesGlob(glob_group, ["torch.foo", "torch.bar"])
        self.assertNotMatchesGlob(glob_group, ["tor.foo", "torch.foo.bar", "torch"])

    def test_one_star_middle(self):
        glob_group = GlobGroup("foo.*.bar")
        self.assertMatchesGlob(glob_group, ["foo.q.bar", "foo.foo.bar"])
        self.assertNotMatchesGlob(
            glob_group,
            [
                "foo.bar",
                "foo.foo",
                "outer.foo.inner.bar",
                "foo.q.bar.more",
                "foo.one.two.bar",
            ],
        )

    def test_one_star_partial(self):
        glob_group = GlobGroup("fo*.bar")
        self.assertMatchesGlob(glob_group, ["fo.bar", "foo.bar", "foobar.bar"])
        self.assertNotMatchesGlob(glob_group, ["oij.bar", "f.bar", "foo"])

    def test_one_star_multiple_in_component(self):
        glob_group = GlobGroup("foo/a*.htm*", separator="/")
        self.assertMatchesGlob(glob_group, ["foo/a.html", "foo/a.htm", "foo/abc.html"])

    def test_one_star_partial_extension(self):
        glob_group = GlobGroup("foo/*.txt", separator="/")
        self.assertMatchesGlob(
            glob_group, ["foo/hello.txt", "foo/goodbye.txt", "foo/.txt"]
        )
        self.assertNotMatchesGlob(
            glob_group, ["foo/bar/hello.txt", "bar/foo/hello.txt"]
        )

    def test_two_star(self):
        glob_group = GlobGroup("torch.**")
        self.assertMatchesGlob(
            glob_group, ["torch.foo", "torch.bar", "torch.foo.bar", "torch"]
        )
        self.assertNotMatchesGlob(glob_group, ["what.torch", "torchvision"])

    def test_two_star_end(self):
        glob_group = GlobGroup("**.torch")
        self.assertMatchesGlob(glob_group, ["torch", "bar.torch"])
        self.assertNotMatchesGlob(glob_group, ["visiontorch"])

    def test_two_star_middle(self):
        glob_group = GlobGroup("foo.**.baz")
        self.assertMatchesGlob(
            glob_group, ["foo.baz", "foo.bar.baz", "foo.bar1.bar2.baz"]
        )
        self.assertNotMatchesGlob(glob_group, ["foobaz", "foo.bar.baz.z"])

    def test_two_star_multiple(self):
        glob_group = GlobGroup("**/bar/**/*.txt", separator="/")
        self.assertMatchesGlob(
            glob_group, ["bar/baz.txt", "a/bar/b.txt", "bar/foo/c.txt"]
        )
        self.assertNotMatchesGlob(glob_group, ["baz.txt", "a/b.txt"])

    def test_raw_two_star(self):
        glob_group = GlobGroup("**")
        self.assertMatchesGlob(glob_group, ["bar", "foo.bar", "ab.c.d.e"])
        self.assertNotMatchesGlob(glob_group, [""])

    def test_invalid_raw(self):
        with self.assertRaises(ValueError):
            GlobGroup("a.**b")

    def test_exclude(self):
        glob_group = GlobGroup("torch.**", exclude=["torch.**.foo"])
        self.assertMatchesGlob(
            glob_group,
            ["torch", "torch.bar", "torch.barfoo"],
        )
        self.assertNotMatchesGlob(
            glob_group,
            ["torch.foo", "torch.some.foo"],
        )

    def test_exclude_from_all(self):
        glob_group = GlobGroup("**", exclude=["foo.**", "bar.**"])
        self.assertMatchesGlob(glob_group, ["a", "hello", "anything.really"])
        self.assertNotMatchesGlob(glob_group, ["foo.bar", "foo.bar.baz"])

    def test_list_include_exclude(self):
        glob_group = GlobGroup(["foo", "bar.**"], exclude=["bar.baz", "bar.qux"])
        self.assertMatchesGlob(glob_group, ["foo", "bar.other", "bar.bazother"])
        self.assertNotMatchesGlob(glob_group, ["bar.baz", "bar.qux"])


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 16 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestGlobGroup`

**Functions defined**: `assertMatchesGlob`, `assertNotMatchesGlob`, `test_one_star`, `test_one_star_middle`, `test_one_star_partial`, `test_one_star_multiple_in_component`, `test_one_star_partial_extension`, `test_two_star`, `test_two_star_end`, `test_two_star_middle`, `test_two_star_multiple`, `test_raw_two_star`, `test_invalid_raw`, `test_exclude`, `test_exclude_from_all`, `test_list_include_exclude`

**Key imports**: Iterable, GlobGroup, run_tests, PackageTestCase, PackageTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/package`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Iterable
- `torch.package`: GlobGroup
- `torch.testing._internal.common_utils`: run_tests
- `.common`: PackageTestCase
- `common`: PackageTestCase


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
python test/package/test_glob_group.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/package`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_directory_reader.py_docs.md`](./test_directory_reader.py_docs.md)
- [`test_digraph.py_docs.md`](./test_digraph.py_docs.md)
- [`test_dependency_api.py_docs.md`](./test_dependency_api.py_docs.md)
- [`module_a.py_docs.md`](./module_a.py_docs.md)
- [`test_model.py_docs.md`](./test_model.py_docs.md)
- [`module_a_remapped_path.py_docs.md`](./module_a_remapped_path.py_docs.md)
- [`test_load_bc_packages.py_docs.md`](./test_load_bc_packages.py_docs.md)
- [`test_mangling.py_docs.md`](./test_mangling.py_docs.md)


## Cross-References

- **File Documentation**: `test_glob_group.py_docs.md`
- **Keyword Index**: `test_glob_group.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
