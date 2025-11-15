# Documentation: `docs/test/package/test_dependency_api.py_docs.md`

## File Metadata

- **Path**: `docs/test/package/test_dependency_api.py_docs.md`
- **Size**: 16,995 bytes (16.60 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/package/test_dependency_api.py`

## File Metadata

- **Path**: `test/package/test_dependency_api.py`
- **Size**: 13,286 bytes (12.97 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: package/deploy"]

import importlib
from io import BytesIO
from textwrap import dedent
from unittest import skipIf

import torch.nn
from torch.package import EmptyMatchError, Importer, PackageExporter, PackageImporter
from torch.package.package_exporter import PackagingError
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestDependencyAPI(PackageTestCase):
    """Dependency management API tests.
    - mock()
    - extern()
    - deny()
    """

    def test_extern(self):
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.extern(["package_a.subpackage", "module_a"])
            he.save_source_string("foo", "import package_a.subpackage; import module_a")
        buffer.seek(0)
        hi = PackageImporter(buffer)
        import module_a
        import package_a.subpackage

        module_a_im = hi.import_module("module_a")
        hi.import_module("package_a.subpackage")
        package_a_im = hi.import_module("package_a")

        self.assertIs(module_a, module_a_im)
        self.assertIsNot(package_a, package_a_im)
        self.assertIs(package_a.subpackage, package_a_im.subpackage)

    def test_extern_glob(self):
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.extern(["package_a.*", "module_*"])
            he.save_module("package_a")
            he.save_source_string(
                "test_module",
                dedent(
                    """\
                    import package_a.subpackage
                    import module_a
                    """
                ),
            )
        buffer.seek(0)
        hi = PackageImporter(buffer)
        import module_a
        import package_a.subpackage

        module_a_im = hi.import_module("module_a")
        hi.import_module("package_a.subpackage")
        package_a_im = hi.import_module("package_a")

        self.assertIs(module_a, module_a_im)
        self.assertIsNot(package_a, package_a_im)
        self.assertIs(package_a.subpackage, package_a_im.subpackage)

    def test_extern_glob_allow_empty(self):
        """
        Test that an error is thrown when a extern glob is specified with allow_empty=True
        and no matching module is required during packaging.
        """
        import package_a.subpackage  # noqa: F401

        buffer = BytesIO()
        with self.assertRaisesRegex(EmptyMatchError, r"did not match any modules"):
            with PackageExporter(buffer) as exporter:
                exporter.extern(include=["package_b.*"], allow_empty=False)
                exporter.save_module("package_a.subpackage")

    def test_deny(self):
        """
        Test marking packages as "deny" during export.
        """
        buffer = BytesIO()

        with self.assertRaisesRegex(PackagingError, "denied"):
            with PackageExporter(buffer) as exporter:
                exporter.deny(["package_a.subpackage", "module_a"])
                exporter.save_source_string("foo", "import package_a.subpackage")

    def test_deny_glob(self):
        """
        Test marking packages as "deny" using globs instead of package names.
        """
        buffer = BytesIO()
        with self.assertRaises(PackagingError):
            with PackageExporter(buffer) as exporter:
                exporter.deny(["package_a.*", "module_*"])
                exporter.save_source_string(
                    "test_module",
                    dedent(
                        """\
                        import package_a.subpackage
                        import module_a
                        """
                    ),
                )

    def test_mock(self):
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.mock(["package_a.subpackage", "module_a"])
            # Import something that dependso n package_a.subpackage
            he.save_source_string("foo", "import package_a.subpackage")
        buffer.seek(0)
        hi = PackageImporter(buffer)
        import package_a.subpackage

        _ = package_a.subpackage
        import module_a

        _ = module_a

        m = hi.import_module("package_a.subpackage")
        r = m.result
        with self.assertRaisesRegex(NotImplementedError, "was mocked out"):
            r()

    def test_mock_glob(self):
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.mock(["package_a.*", "module*"])
            he.save_module("package_a")
            he.save_source_string(
                "test_module",
                dedent(
                    """\
                    import package_a.subpackage
                    import module_a
                    """
                ),
            )
        buffer.seek(0)
        hi = PackageImporter(buffer)
        import package_a.subpackage

        _ = package_a.subpackage
        import module_a

        _ = module_a

        m = hi.import_module("package_a.subpackage")
        r = m.result
        with self.assertRaisesRegex(NotImplementedError, "was mocked out"):
            r()

    def test_mock_glob_allow_empty(self):
        """
        Test that an error is thrown when a mock glob is specified with allow_empty=True
        and no matching module is required during packaging.
        """
        import package_a.subpackage  # noqa: F401

        buffer = BytesIO()
        with self.assertRaisesRegex(EmptyMatchError, r"did not match any modules"):
            with PackageExporter(buffer) as exporter:
                exporter.mock(include=["package_b.*"], allow_empty=False)
                exporter.save_module("package_a.subpackage")

    def test_pickle_mocked(self):
        import package_a.subpackage

        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)

        buffer = BytesIO()
        with self.assertRaises(PackagingError):
            with PackageExporter(buffer) as he:
                he.mock(include="package_a.subpackage")
                he.intern("**")
                he.save_pickle("obj", "obj.pkl", obj2)

    def test_pickle_mocked_all(self):
        import package_a.subpackage

        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)

        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.intern(include="package_a.**")
            he.mock("**")
            he.save_pickle("obj", "obj.pkl", obj2)

    def test_allow_empty_with_error(self):
        """If an error occurs during packaging, it should not be shadowed by the allow_empty error."""
        buffer = BytesIO()
        with self.assertRaises(ModuleNotFoundError):
            with PackageExporter(buffer) as pe:
                # Even though we did not extern a module that matches this
                # pattern, we want to show the save_module error, not the allow_empty error.

                pe.extern("foo", allow_empty=False)
                pe.save_module("aodoifjodisfj")  # will error

                # we never get here, so technically the allow_empty check
                # should raise an error. However, the error above is more
                # informative to what's actually going wrong with packaging.
                pe.save_source_string("bar", "import foo\n")

    def test_implicit_intern(self):
        """The save_module APIs should implicitly intern the module being saved."""
        import package_a  # noqa: F401

        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.save_module("package_a")

    def test_intern_error(self):
        """Failure to handle all dependencies should lead to an error."""
        import package_a.subpackage

        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)

        buffer = BytesIO()

        with self.assertRaises(PackagingError) as e:
            with PackageExporter(buffer) as he:
                he.save_pickle("obj", "obj.pkl", obj2)

        self.assertEqual(
            str(e.exception),
            dedent(
                """
                * Module did not match against any action pattern. Extern, mock, or intern it.
                    package_a
                    package_a.subpackage

                Set debug=True when invoking PackageExporter for a visualization of where broken modules are coming from!
                """
            ),
        )

        # Interning all dependencies should work
        with PackageExporter(buffer) as he:
            he.intern(["package_a", "package_a.subpackage"])
            he.save_pickle("obj", "obj.pkl", obj2)

    @skipIf(IS_WINDOWS, "extension modules have a different file extension on windows")
    def test_broken_dependency(self):
        """A unpackageable dependency should raise a PackagingError."""

        def create_module(name):
            spec = importlib.machinery.ModuleSpec(name, self, is_package=False)  # type: ignore[arg-type]
            module = importlib.util.module_from_spec(spec)
            ns = module.__dict__
            ns["__spec__"] = spec
            ns["__loader__"] = self
            ns["__file__"] = f"{name}.so"
            ns["__cached__"] = None
            return module

        class BrokenImporter(Importer):
            def __init__(self) -> None:
                self.modules = {
                    "foo": create_module("foo"),
                    "bar": create_module("bar"),
                }

            def import_module(self, module_name):
                return self.modules[module_name]

        buffer = BytesIO()

        with self.assertRaises(PackagingError) as e:
            with PackageExporter(buffer, importer=BrokenImporter()) as exporter:
                exporter.intern(["foo", "bar"])
                exporter.save_source_string("my_module", "import foo; import bar")

        self.assertEqual(
            str(e.exception),
            dedent(
                """
                * Module is a C extension module. torch.package supports Python modules only.
                    foo
                    bar

                Set debug=True when invoking PackageExporter for a visualization of where broken modules are coming from!
                """
            ),
        )

    def test_invalid_import(self):
        """An incorrectly-formed import should raise a PackagingError."""
        buffer = BytesIO()
        with self.assertRaises(PackagingError) as e:
            with PackageExporter(buffer) as exporter:
                # This import will fail to load.
                exporter.save_source_string("foo", "from ........ import lol")

        self.assertEqual(
            str(e.exception),
            dedent(
                """
                * Dependency resolution failed.
                    foo
                      Context: attempted relative import beyond top-level package

                Set debug=True when invoking PackageExporter for a visualization of where broken modules are coming from!
                """
            ),
        )

    def test_repackage_mocked_module(self):
        """Re-packaging a package that contains a mocked module should work correctly."""
        buffer = BytesIO()
        with PackageExporter(buffer) as exporter:
            exporter.mock("package_a")
            exporter.save_source_string("foo", "import package_a")

        buffer.seek(0)
        importer = PackageImporter(buffer)
        foo = importer.import_module("foo")

        # "package_a" should be mocked out.
        with self.assertRaises(NotImplementedError):
            foo.package_a.get_something()

        # Re-package the model, but intern the previously-mocked module and mock
        # everything else.
        buffer2 = BytesIO()
        with PackageExporter(buffer2, importer=importer) as exporter:
            exporter.intern("package_a")
            exporter.mock("**")
            exporter.save_source_string("foo", "import package_a")

        buffer2.seek(0)
        importer2 = PackageImporter(buffer2)
        foo2 = importer2.import_module("foo")

        # "package_a" should still be mocked out.
        with self.assertRaises(NotImplementedError):
            foo2.package_a.get_something()

    def test_externing_c_extension(self):
        """Externing c extensions modules should allow us to still access them especially those found in torch._C."""

        buffer = BytesIO()
        # The C extension module in question is F.gelu which comes from torch._C._nn
        model = torch.nn.TransformerEncoderLayer(
            d_model=64,
            nhead=2,
            dim_feedforward=64,
            dropout=1.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        with PackageExporter(buffer) as e:
            e.extern("torch.**")
            e.intern("**")

            e.save_pickle("model", "model.pkl", model)
        buffer.seek(0)
        imp = PackageImporter(buffer)
        imp.load_pickle("model", "model.pkl")


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Dependency management API tests.    - mock()    - extern()    - deny()

This Python file contains 2 class(es) and 20 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestDependencyAPI`, `BrokenImporter`

**Functions defined**: `test_extern`, `test_extern_glob`, `test_extern_glob_allow_empty`, `test_deny`, `test_deny_glob`, `test_mock`, `test_mock_glob`, `test_mock_glob_allow_empty`, `test_pickle_mocked`, `test_pickle_mocked_all`, `test_allow_empty_with_error`, `test_implicit_intern`, `test_intern_error`, `test_broken_dependency`, `create_module`, `__init__`, `import_module`, `test_invalid_import`, `test_repackage_mocked_module`, `test_externing_c_extension`

**Key imports**: importlib, BytesIO, dedent, skipIf, torch.nn, EmptyMatchError, Importer, PackageExporter, PackageImporter, PackagingError, IS_WINDOWS, run_tests, PackageTestCase, PackageTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/package`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `importlib`
- `io`: BytesIO
- `textwrap`: dedent
- `unittest`: skipIf
- `torch.nn`
- `torch.package`: EmptyMatchError, Importer, PackageExporter, PackageImporter
- `torch.package.package_exporter`: PackagingError
- `torch.testing._internal.common_utils`: IS_WINDOWS, run_tests
- `.common`: PackageTestCase
- `common`: PackageTestCase
- `package_a.subpackage`
- `module_a`
- `package_a.subpackage  `


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/package/test_dependency_api.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/package`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_directory_reader.py_docs.md`](./test_directory_reader.py_docs.md)
- [`test_digraph.py_docs.md`](./test_digraph.py_docs.md)
- [`module_a.py_docs.md`](./module_a.py_docs.md)
- [`test_model.py_docs.md`](./test_model.py_docs.md)
- [`module_a_remapped_path.py_docs.md`](./module_a_remapped_path.py_docs.md)
- [`test_glob_group.py_docs.md`](./test_glob_group.py_docs.md)
- [`test_load_bc_packages.py_docs.md`](./test_load_bc_packages.py_docs.md)
- [`test_mangling.py_docs.md`](./test_mangling.py_docs.md)


## Cross-References

- **File Documentation**: `test_dependency_api.py_docs.md`
- **Keyword Index**: `test_dependency_api.py_kw.md`
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

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/package/test_dependency_api.py_docs.md
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

- **File Documentation**: `test_dependency_api.py_docs.md_docs.md`
- **Keyword Index**: `test_dependency_api.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
