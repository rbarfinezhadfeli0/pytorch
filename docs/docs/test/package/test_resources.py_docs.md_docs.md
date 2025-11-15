# Documentation: `docs/test/package/test_resources.py_docs.md`

## File Metadata

- **Path**: `docs/test/package/test_resources.py_docs.md`
- **Size**: 8,635 bytes (8.43 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/package/test_resources.py`

## File Metadata

- **Path**: `test/package/test_resources.py`
- **Size**: 5,447 bytes (5.32 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: package/deploy"]

from io import BytesIO
from sys import version_info
from textwrap import dedent
from unittest import skipIf

from torch.package import PackageExporter, PackageImporter
from torch.testing._internal.common_utils import run_tests


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestResources(PackageTestCase):
    """Tests for access APIs for packaged resources."""

    def test_resource_reader(self):
        """Test compliance with the get_resource_reader importlib API."""
        buffer = BytesIO()
        with PackageExporter(buffer) as pe:
            # Layout looks like:
            #    package
            #    |-- one/
            #    |   |-- a.txt
            #    |   |-- b.txt
            #    |   |-- c.txt
            #    |   +-- three/
            #    |       |-- d.txt
            #    |       +-- e.txt
            #    +-- two/
            #       |-- f.txt
            #       +-- g.txt
            pe.save_text("one", "a.txt", "hello, a!")
            pe.save_text("one", "b.txt", "hello, b!")
            pe.save_text("one", "c.txt", "hello, c!")

            pe.save_text("one.three", "d.txt", "hello, d!")
            pe.save_text("one.three", "e.txt", "hello, e!")

            pe.save_text("two", "f.txt", "hello, f!")
            pe.save_text("two", "g.txt", "hello, g!")

        buffer.seek(0)
        importer = PackageImporter(buffer)

        reader_one = importer.get_resource_reader("one")
        with self.assertRaises(FileNotFoundError):
            reader_one.resource_path("a.txt")

        self.assertTrue(reader_one.is_resource("a.txt"))
        self.assertEqual(reader_one.open_resource("a.txt").getbuffer(), b"hello, a!")
        self.assertFalse(reader_one.is_resource("three"))
        reader_one_contents = list(reader_one.contents())
        self.assertSequenceEqual(
            reader_one_contents, ["a.txt", "b.txt", "c.txt", "three"]
        )

        reader_two = importer.get_resource_reader("two")
        self.assertTrue(reader_two.is_resource("f.txt"))
        self.assertEqual(reader_two.open_resource("f.txt").getbuffer(), b"hello, f!")
        reader_two_contents = list(reader_two.contents())
        self.assertSequenceEqual(reader_two_contents, ["f.txt", "g.txt"])

        reader_one_three = importer.get_resource_reader("one.three")
        self.assertTrue(reader_one_three.is_resource("d.txt"))
        self.assertEqual(
            reader_one_three.open_resource("d.txt").getbuffer(), b"hello, d!"
        )
        reader_one_three_contenst = list(reader_one_three.contents())
        self.assertSequenceEqual(reader_one_three_contenst, ["d.txt", "e.txt"])

        self.assertIsNone(importer.get_resource_reader("nonexistent_package"))

    @skipIf(version_info >= (3, 13), "https://github.com/python/cpython/issues/127012")
    def test_package_resource_access(self):
        """Packaged modules should be able to use the importlib.resources API to access
        resources saved in the package.
        """
        mod_src = dedent(
            """\
            import importlib.resources
            import my_cool_resources

            def secret_message():
                return importlib.resources.read_text(my_cool_resources, 'sekrit.txt')
            """
        )
        buffer = BytesIO()
        with PackageExporter(buffer) as pe:
            pe.save_source_string("foo.bar", mod_src)
            pe.save_text("my_cool_resources", "sekrit.txt", "my sekrit plays")

        buffer.seek(0)
        importer = PackageImporter(buffer)
        self.assertEqual(
            importer.import_module("foo.bar").secret_message(), "my sekrit plays"
        )

    def test_importer_access(self):
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.save_text("main", "main", "my string")
            he.save_binary("main", "main_binary", b"my string")
            src = dedent(
                """\
                import importlib
                import torch_package_importer as resources

                t = resources.load_text('main', 'main')
                b = resources.load_binary('main', 'main_binary')
                """
            )
            he.save_source_string("main", src, is_package=True)
        buffer.seek(0)
        hi = PackageImporter(buffer)
        m = hi.import_module("main")
        self.assertEqual(m.t, "my string")
        self.assertEqual(m.b, b"my string")

    def test_resource_access_by_path(self):
        """
        Tests that packaged code can used importlib.resources.path.
        """
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.save_binary("string_module", "my_string", b"my string")
            src = dedent(
                """\
                import importlib.resources
                import string_module

                with importlib.resources.path(string_module, 'my_string') as path:
                    with open(path, mode='r', encoding='utf-8') as f:
                        s = f.read()
                """
            )
            he.save_source_string("main", src, is_package=True)
        buffer.seek(0)
        hi = PackageImporter(buffer)
        m = hi.import_module("main")
        self.assertEqual(m.s, "my string")


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Tests for access APIs for packaged resources."""    def test_resource_reader(self):

This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestResources`

**Functions defined**: `test_resource_reader`, `test_package_resource_access`, `secret_message`, `test_importer_access`, `test_resource_access_by_path`

**Key imports**: BytesIO, version_info, dedent, skipIf, PackageExporter, PackageImporter, run_tests, PackageTestCase, PackageTestCase, importlib.resources, my_cool_resources


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/package`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `io`: BytesIO
- `sys`: version_info
- `textwrap`: dedent
- `unittest`: skipIf
- `torch.package`: PackageExporter, PackageImporter
- `torch.testing._internal.common_utils`: run_tests
- `.common`: PackageTestCase
- `common`: PackageTestCase
- `importlib.resources`
- `my_cool_resources`
- `importlib`
- `torch_package_importer as resources`
- `string_module`


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
python test/package/test_resources.py
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
- [`test_glob_group.py_docs.md`](./test_glob_group.py_docs.md)
- [`test_load_bc_packages.py_docs.md`](./test_load_bc_packages.py_docs.md)
- [`test_mangling.py_docs.md`](./test_mangling.py_docs.md)


## Cross-References

- **File Documentation**: `test_resources.py_docs.md`
- **Keyword Index**: `test_resources.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/package`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/package`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/package/test_resources.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/package`):

- [`test_mangling.py_docs.md_docs.md`](./test_mangling.py_docs.md_docs.md)
- [`test_analyze.py_docs.md_docs.md`](./test_analyze.py_docs.md_docs.md)
- [`test_misc.py_docs.md_docs.md`](./test_misc.py_docs.md_docs.md)
- [`test_resources.py_kw.md_docs.md`](./test_resources.py_kw.md_docs.md)
- [`generate_bc_packages.py_kw.md_docs.md`](./generate_bc_packages.py_kw.md_docs.md)
- [`test_package_fx.py_docs.md_docs.md`](./test_package_fx.py_docs.md_docs.md)
- [`test_repackage.py_kw.md_docs.md`](./test_repackage.py_kw.md_docs.md)
- [`test_importer.py_kw.md_docs.md`](./test_importer.py_kw.md_docs.md)
- [`test_repackage.py_docs.md_docs.md`](./test_repackage.py_docs.md_docs.md)
- [`common.py_docs.md_docs.md`](./common.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_resources.py_docs.md_docs.md`
- **Keyword Index**: `test_resources.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
