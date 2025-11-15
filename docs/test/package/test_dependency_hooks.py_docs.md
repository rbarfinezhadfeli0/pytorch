# Documentation: test_dependency_hooks.py

## File Metadata
- **Path**: `test/package/test_dependency_hooks.py`
- **Size**: 3970 bytes
- **Lines**: 123
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 1 class(es): TestDependencyHooks

### Functions
This file defines 14 function(s): test_single_hook, my_extern_hook, test_multiple_extern_hooks, my_extern_hook, my_extern_hook2, test_multiple_mock_hooks, my_mock_hook, my_mock_hook2, test_remove_hooks, my_extern_hook, my_extern_hook2, test_extern_and_mock_hook, my_extern_hook, my_mock_hook


## Key Components

The file contains 243 words across 123 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 3970 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
