# Documentation: `test/package/test_dependency_hooks.py`

## File Metadata

- **Path**: `test/package/test_dependency_hooks.py`
- **Size**: 3,970 bytes (3.88 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: package/deploy"]

from io import BytesIO

from torch.package import PackageExporter
from torch.testing._internal.common_utils import run_tests


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestDependencyHooks(PackageTestCase):
    """Dependency management hooks API tests.
    - register_mock_hook()
    - register_extern_hook()
    """

    def test_single_hook(self):
        buffer = BytesIO()

        my_externs = set()

        def my_extern_hook(package_exporter, module_name):
            my_externs.add(module_name)

        with PackageExporter(buffer) as exporter:
            exporter.extern(["package_a.subpackage", "module_a"])
            exporter.register_extern_hook(my_extern_hook)
            exporter.save_source_string("foo", "import module_a")

        self.assertEqual(my_externs, {"module_a"})

    def test_multiple_extern_hooks(self):
        buffer = BytesIO()

        my_externs = set()

        def my_extern_hook(package_exporter, module_name):
            my_externs.add(module_name)

        # This also checks ordering, since `remove()` will fail if the value is not in the set.
        def my_extern_hook2(package_exporter, module_name):
            my_externs.remove(module_name)

        with PackageExporter(buffer) as exporter:
            exporter.extern(["package_a.subpackage", "module_a"])
            exporter.register_extern_hook(my_extern_hook)
            exporter.register_extern_hook(my_extern_hook2)
            exporter.save_source_string("foo", "import module_a")

        self.assertEqual(my_externs, set())

    def test_multiple_mock_hooks(self):
        buffer = BytesIO()

        my_mocks = set()

        def my_mock_hook(package_exporter, module_name):
            my_mocks.add(module_name)

        # This also checks ordering, since `remove()` will fail if the value is not in the set.
        def my_mock_hook2(package_exporter, module_name):
            my_mocks.remove(module_name)

        with PackageExporter(buffer) as exporter:
            exporter.mock(["package_a.subpackage", "module_a"])
            exporter.register_mock_hook(my_mock_hook)
            exporter.register_mock_hook(my_mock_hook2)
            exporter.save_source_string("foo", "import module_a")

        self.assertEqual(my_mocks, set())

    def test_remove_hooks(self):
        buffer = BytesIO()

        my_externs = set()
        my_externs2 = set()

        def my_extern_hook(package_exporter, module_name):
            my_externs.add(module_name)

        def my_extern_hook2(package_exporter, module_name):
            my_externs2.add(module_name)

        with PackageExporter(buffer) as exporter:
            exporter.extern(["package_a.subpackage", "module_a"])
            handle = exporter.register_extern_hook(my_extern_hook)
            exporter.register_extern_hook(my_extern_hook2)
            handle.remove()
            exporter.save_source_string("foo", "import module_a")

        self.assertEqual(my_externs, set())
        self.assertEqual(my_externs2, {"module_a"})

    def test_extern_and_mock_hook(self):
        buffer = BytesIO()

        my_externs = set()
        my_mocks = set()

        def my_extern_hook(package_exporter, module_name):
            my_externs.add(module_name)

        def my_mock_hook(package_exporter, module_name):
            my_mocks.add(module_name)

        with PackageExporter(buffer) as exporter:
            exporter.extern("module_a")
            exporter.mock("package_a")
            exporter.register_extern_hook(my_extern_hook)
            exporter.register_mock_hook(my_mock_hook)
            exporter.save_source_string("foo", "import module_a; import package_a")

        self.assertEqual(my_externs, {"module_a"})
        self.assertEqual(my_mocks, {"package_a"})


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Dependency management hooks API tests.    - register_mock_hook()    - register_extern_hook()

This Python file contains 1 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestDependencyHooks`

**Functions defined**: `test_single_hook`, `my_extern_hook`, `test_multiple_extern_hooks`, `my_extern_hook`, `my_extern_hook2`, `test_multiple_mock_hooks`, `my_mock_hook`, `my_mock_hook2`, `test_remove_hooks`, `my_extern_hook`, `my_extern_hook2`, `test_extern_and_mock_hook`, `my_extern_hook`, `my_mock_hook`

**Key imports**: BytesIO, PackageExporter, run_tests, PackageTestCase, PackageTestCase, module_a, module_a, module_a, module_a, module_a


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/package`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `io`: BytesIO
- `torch.package`: PackageExporter
- `torch.testing._internal.common_utils`: run_tests
- `.common`: PackageTestCase
- `common`: PackageTestCase
- `module_a`
- `package_a`


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
python test/package/test_dependency_hooks.py
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

- **File Documentation**: `test_dependency_hooks.py_docs.md`
- **Keyword Index**: `test_dependency_hooks.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
