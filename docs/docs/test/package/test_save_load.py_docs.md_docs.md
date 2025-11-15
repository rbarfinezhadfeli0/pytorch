# Documentation: `docs/test/package/test_save_load.py_docs.md`

## File Metadata

- **Path**: `docs/test/package/test_save_load.py_docs.md`
- **Size**: 14,116 bytes (13.79 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/package/test_save_load.py`

## File Metadata

- **Path**: `test/package/test_save_load.py`
- **Size**: 10,042 bytes (9.81 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: package/deploy"]

import pickle
from io import BytesIO
from sys import version_info
from textwrap import dedent
from unittest import skipIf

import torch
from torch.package import PackageExporter, PackageImporter, sys_importer
from torch.testing._internal.common_utils import run_tests


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase

from pathlib import Path


packaging_directory = Path(__file__).parent


class TestSaveLoad(PackageTestCase):
    """Core save_* and loading API tests."""

    def test_saving_source(self):
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.save_source_file("foo", str(packaging_directory / "module_a.py"))
            he.save_source_file("foodir", str(packaging_directory / "package_a"))
        buffer.seek(0)
        hi = PackageImporter(buffer)
        foo = hi.import_module("foo")
        s = hi.import_module("foodir.subpackage")
        self.assertEqual(foo.result, "module_a")
        self.assertEqual(s.result, "package_a.subpackage")

    def test_saving_string(self):
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            src = dedent(
                """\
                import math
                the_math = math
                """
            )
            he.save_source_string("my_mod", src)
        buffer.seek(0)
        hi = PackageImporter(buffer)
        m = hi.import_module("math")
        import math

        self.assertIs(m, math)
        my_mod = hi.import_module("my_mod")
        self.assertIs(my_mod.math, math)

    def test_save_module(self):
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            import module_a
            import package_a

            he.save_module(module_a.__name__)
            he.save_module(package_a.__name__)
        buffer.seek(0)
        hi = PackageImporter(buffer)
        module_a_i = hi.import_module("module_a")
        self.assertEqual(module_a_i.result, "module_a")
        self.assertIsNot(module_a, module_a_i)
        package_a_i = hi.import_module("package_a")
        self.assertEqual(package_a_i.result, "package_a")
        self.assertIsNot(package_a_i, package_a)

    def test_dunder_imports(self):
        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            import package_b

            obj = package_b.PackageBObject
            he.intern("**")
            he.save_pickle("res", "obj.pkl", obj)

        buffer.seek(0)
        hi = PackageImporter(buffer)
        hi.load_pickle("res", "obj.pkl")

        package_b = hi.import_module("package_b")
        self.assertEqual(package_b.result, "package_b")

        math = hi.import_module("math")
        self.assertEqual(math.__name__, "math")

        xml_sub_sub_package = hi.import_module("xml.sax.xmlreader")
        self.assertEqual(xml_sub_sub_package.__name__, "xml.sax.xmlreader")

        subpackage_1 = hi.import_module("package_b.subpackage_1")
        self.assertEqual(subpackage_1.result, "subpackage_1")

        subpackage_2 = hi.import_module("package_b.subpackage_2")
        self.assertEqual(subpackage_2.result, "subpackage_2")

        subsubpackage_0 = hi.import_module("package_b.subpackage_0.subsubpackage_0")
        self.assertEqual(subsubpackage_0.result, "subsubpackage_0")

    def test_bad_dunder_imports(self):
        """Test to ensure bad __imports__ don't cause PackageExporter to fail."""
        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.save_source_string(
                "m", '__import__(these, unresolvable, "things", wont, crash, me)'
            )

    def test_save_module_binary(self):
        f = BytesIO()
        with PackageExporter(f) as he:
            import module_a
            import package_a

            he.save_module(module_a.__name__)
            he.save_module(package_a.__name__)
        f.seek(0)
        hi = PackageImporter(f)
        module_a_i = hi.import_module("module_a")
        self.assertEqual(module_a_i.result, "module_a")
        self.assertIsNot(module_a, module_a_i)
        package_a_i = hi.import_module("package_a")
        self.assertEqual(package_a_i.result, "package_a")
        self.assertIsNot(package_a_i, package_a)

    def test_pickle(self):
        import package_a.subpackage

        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)

        buffer = BytesIO()
        with PackageExporter(buffer) as he:
            he.intern("**")
            he.save_pickle("obj", "obj.pkl", obj2)
        buffer.seek(0)
        hi = PackageImporter(buffer)

        # check we got dependencies
        sp = hi.import_module("package_a.subpackage")
        # check we didn't get other stuff
        with self.assertRaises(ImportError):
            hi.import_module("module_a")

        obj_loaded = hi.load_pickle("obj", "obj.pkl")
        self.assertIsNot(obj2, obj_loaded)
        self.assertIsInstance(obj_loaded.obj, sp.PackageASubpackageObject)
        self.assertIsNot(
            package_a.subpackage.PackageASubpackageObject, sp.PackageASubpackageObject
        )

    def test_pickle_long_name_with_protocol_4(self):
        import package_a.long_name

        container = []

        # Indirectly grab the function to avoid pasting a 256 character
        # function into the test
        package_a.long_name.add_function(container)

        buffer = BytesIO()
        with PackageExporter(buffer) as exporter:
            exporter.intern("**")
            exporter.save_pickle(
                "container", "container.pkl", container, pickle_protocol=4
            )

        buffer.seek(0)
        importer = PackageImporter(buffer)
        unpickled_container = importer.load_pickle("container", "container.pkl")
        self.assertIsNot(container, unpickled_container)
        self.assertEqual(len(unpickled_container), 1)
        self.assertEqual(container[0](), unpickled_container[0]())

    def test_exporting_mismatched_code(self):
        """
        If an object with the same qualified name is loaded from different
        packages, the user should get an error if they try to re-save the
        object with the wrong package's source code.
        """
        import package_a.subpackage

        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)

        b1 = BytesIO()
        with PackageExporter(b1) as pe:
            pe.intern("**")
            pe.save_pickle("obj", "obj.pkl", obj2)

        b1.seek(0)
        importer1 = PackageImporter(b1)
        loaded1 = importer1.load_pickle("obj", "obj.pkl")

        b1.seek(0)
        importer2 = PackageImporter(b1)
        loaded2 = importer2.load_pickle("obj", "obj.pkl")

        def make_exporter():
            pe = PackageExporter(BytesIO(), importer=[importer1, sys_importer])
            # Ensure that the importer finds the 'PackageAObject' defined in 'importer1' first.
            return pe

        # This succeeds because OrderedImporter.get_name() properly
        # falls back to sys_importer which can find the original PackageAObject
        pe = make_exporter()
        pe.save_pickle("obj", "obj.pkl", obj2)

        # This should also fail. The 'PackageAObject' type defined from 'importer1'
        # is not necessarily the same as the one defined from 'importer2'
        pe = make_exporter()
        with self.assertRaises(pickle.PicklingError):
            pe.save_pickle("obj", "obj.pkl", loaded2)

        # This should succeed. The 'PackageAObject' type defined from
        # 'importer1' is a match for the one used by loaded1.
        pe = make_exporter()
        pe.save_pickle("obj", "obj.pkl", loaded1)

    def test_save_imported_module(self):
        """Saving a module that came from another PackageImporter should work."""
        import package_a.subpackage

        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)

        buffer = BytesIO()
        with PackageExporter(buffer) as exporter:
            exporter.intern("**")
            exporter.save_pickle("model", "model.pkl", obj2)

        buffer.seek(0)

        importer = PackageImporter(buffer)
        imported_obj2 = importer.load_pickle("model", "model.pkl")
        imported_obj2_module = imported_obj2.__class__.__module__

        # Should export without error.
        buffer2 = BytesIO()
        with PackageExporter(buffer2, importer=(importer, sys_importer)) as exporter:
            exporter.intern("**")
            exporter.save_module(imported_obj2_module)

    def test_save_imported_module_using_package_importer(self):
        """Exercise a corner case: re-packaging a module that uses `torch_package_importer`"""
        import package_a.use_torch_package_importer  # noqa: F401

        buffer = BytesIO()
        with PackageExporter(buffer) as exporter:
            exporter.intern("**")
            exporter.save_module("package_a.use_torch_package_importer")

        buffer.seek(0)

        importer = PackageImporter(buffer)

        # Should export without error.
        buffer2 = BytesIO()
        with PackageExporter(buffer2, importer=(importer, sys_importer)) as exporter:
            exporter.intern("**")
            exporter.save_module("package_a.use_torch_package_importer")

    @skipIf(version_info >= (3, 13), "https://github.com/pytorch/pytorch/issues/142170")
    def test_save_load_fp8(self):
        tensor = torch.rand(20, 20).to(torch.float8_e4m3fn)

        buffer = BytesIO()
        with PackageExporter(buffer) as exporter:
            exporter.save_pickle("fp8_model", "model.pkl", tensor)

        buffer.seek(0)

        importer = PackageImporter(buffer)
        loaded_tensor = importer.load_pickle("fp8_model", "model.pkl")
        self.assertTrue(torch.equal(tensor, loaded_tensor))


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Core save_* and loading API tests."""    def test_saving_source(self):        buffer = BytesIO()        with PackageExporter(buffer) as he:            he.save_source_file("foo", str(packaging_directory / "module_a.py"))            he.save_source_file("foodir", str(packaging_directory / "package_a"))        buffer.seek(0)        hi = PackageImporter(buffer)        foo = hi.import_module("foo")        s = hi.import_module("foodir.subpackage")        self.assertEqual(foo.result, "module_a")        self.assertEqual(s.result, "package_a.subpackage")    def test_saving_string(self):        buffer = BytesIO()        with PackageExporter(buffer) as he:            src = dedent(

This Python file contains 1 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestSaveLoad`

**Functions defined**: `test_saving_source`, `test_saving_string`, `test_save_module`, `test_dunder_imports`, `test_bad_dunder_imports`, `test_save_module_binary`, `test_pickle`, `test_pickle_long_name_with_protocol_4`, `test_exporting_mismatched_code`, `make_exporter`, `test_save_imported_module`, `test_save_imported_module_using_package_importer`, `test_save_load_fp8`

**Key imports**: pickle, BytesIO, version_info, dedent, skipIf, torch, PackageExporter, PackageImporter, sys_importer, run_tests, PackageTestCase, PackageTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/package`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `pickle`
- `io`: BytesIO
- `sys`: version_info
- `textwrap`: dedent
- `unittest`: skipIf
- `torch`
- `torch.package`: PackageExporter, PackageImporter, sys_importer
- `torch.testing._internal.common_utils`: run_tests
- `.common`: PackageTestCase
- `common`: PackageTestCase
- `pathlib`: Path
- `math`
- `module_a`
- `package_a`
- `package_b`
- `package_a.subpackage`
- `package_a.long_name`
- `package_a.use_torch_package_importer  `


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/package/test_save_load.py
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

- **File Documentation**: `test_save_load.py_docs.md`
- **Keyword Index**: `test_save_load.py_kw.md`
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

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/package/test_save_load.py_docs.md
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

- **File Documentation**: `test_save_load.py_docs.md_docs.md`
- **Keyword Index**: `test_save_load.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
