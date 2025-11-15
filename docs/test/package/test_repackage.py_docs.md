# Documentation: `test/package/test_repackage.py`

## File Metadata

- **Path**: `test/package/test_repackage.py`
- **Size**: 1,347 bytes (1.32 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: package/deploy"]

from io import BytesIO

from torch.package import PackageExporter, PackageImporter, sys_importer
from torch.testing._internal.common_utils import run_tests


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestRepackage(PackageTestCase):
    """Tests for repackaging."""

    def test_repackage_import_indirectly_via_parent_module(self):
        from package_d.imports_directly import ImportsDirectlyFromSubSubPackage
        from package_d.imports_indirectly import ImportsIndirectlyFromSubPackage

        model_a = ImportsDirectlyFromSubSubPackage()
        buffer = BytesIO()
        with PackageExporter(buffer) as pe:
            pe.intern("**")
            pe.save_pickle("default", "model.py", model_a)

        buffer.seek(0)
        pi = PackageImporter(buffer)
        pi.load_pickle("default", "model.py")

        model_b = ImportsIndirectlyFromSubPackage()
        buffer = BytesIO()
        with PackageExporter(
            buffer,
            importer=(
                pi,
                sys_importer,
            ),
        ) as pe:
            pe.intern("**")
            pe.save_pickle("default", "model_b.py", model_b)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Tests for repackaging."""    def test_repackage_import_indirectly_via_parent_module(self):        from package_d.imports_directly import ImportsDirectlyFromSubSubPackage        from package_d.imports_indirectly import ImportsIndirectlyFromSubPackage        model_a = ImportsDirectlyFromSubSubPackage()        buffer = BytesIO()        with PackageExporter(buffer) as pe:            pe.intern("**")            pe.save_pickle("default", "model.py", model_a)        buffer.seek(0)        pi = PackageImporter(buffer)        pi.load_pickle("default", "model.py")        model_b = ImportsIndirectlyFromSubPackage()        buffer = BytesIO()        with PackageExporter(            buffer,            importer=(                pi,                sys_importer,            ),        ) as pe:            pe.intern("**")            pe.save_pickle("default", "model_b.py", model_b)if __name__ == "__main__":    run_tests()

This Python file contains 1 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestRepackage`

**Functions defined**: `test_repackage_import_indirectly_via_parent_module`

**Key imports**: BytesIO, PackageExporter, PackageImporter, sys_importer, run_tests, PackageTestCase, PackageTestCase, ImportsDirectlyFromSubSubPackage, ImportsIndirectlyFromSubPackage


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/package`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `io`: BytesIO
- `torch.package`: PackageExporter, PackageImporter, sys_importer
- `torch.testing._internal.common_utils`: run_tests
- `.common`: PackageTestCase
- `common`: PackageTestCase
- `package_d.imports_directly`: ImportsDirectlyFromSubSubPackage
- `package_d.imports_indirectly`: ImportsIndirectlyFromSubPackage


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
python test/package/test_repackage.py
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

- **File Documentation**: `test_repackage.py_docs.md`
- **Keyword Index**: `test_repackage.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
