# Documentation: `docs/test/cpp_extensions/python_agnostic_extension/test/test_python_agnostic.py_docs.md`

## File Metadata

- **Path**: `docs/test/cpp_extensions/python_agnostic_extension/test/test_python_agnostic.py_docs.md`
- **Size**: 5,413 bytes (5.29 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp_extensions/python_agnostic_extension/test/test_python_agnostic.py`

## File Metadata

- **Path**: `test/cpp_extensions/python_agnostic_extension/test/test_python_agnostic.py`
- **Size**: 2,892 bytes (2.82 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: cpp"]

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    IS_LINUX,
    run_tests,
    shell,
    TEST_XPU,
    TestCase,
)


class TestPythonAgnostic(TestCase):
    @classmethod
    def setUpClass(cls):
        # Wipe the dist dir if it exists
        cls.extension_root = Path(__file__).parent.parent
        cls.dist_dir = os.path.join(cls.extension_root, "dist")
        if os.path.exists(cls.dist_dir):
            shutil.rmtree(cls.dist_dir)

        # Build the wheel
        wheel_cmd = [sys.executable, "-m", "build", "--wheel", "--no-isolation"]
        return_code = shell(wheel_cmd, cwd=cls.extension_root, env=os.environ)
        if return_code != 0:
            raise RuntimeError("python_agnostic bdist_wheel failed to build")

    @unittest.skipIf(
        not (TEST_CUDA or TEST_XPU),
        "test requires CUDA or XPU",
    )
    @unittest.skipIf(not IS_LINUX, "test requires linux tools ldd and nm")
    def test_extension_is_python_agnostic(self, device):
        # For this test, run_test.py will call `python -m build --wheel --no-isolation` in the
        # cpp_extensions/python_agnostic_extension folder, where the extension and
        # setup calls specify py_limited_api to `True`. To approximate that the
        # extension is indeed python agnostic, we test
        #   a. The extension wheel name contains "cp39-abi3", meaning the wheel
        # should be runnable for any Python 3 version after and including 3.9
        #   b. The produced shared library does not have libtorch_python.so as a
        # dependency from the output of "ldd _C.so"
        #   c. The .so does not need any python related symbols. We approximate
        # this by running "nm -u _C.so" and grepping that nothing starts with "Py"

        matches = list(Path(self.dist_dir).glob("*.whl"))
        self.assertEqual(len(matches), 1, msg=str(matches))
        whl_file = matches[0]
        self.assertRegex(str(whl_file), r".*python_agnostic-0\.0-cp39-abi3-.*\.whl")

        build_dir = os.path.join(self.extension_root, "build")
        matches = list(Path(build_dir).glob("**/*.so"))
        self.assertEqual(len(matches), 1, msg=str(matches))
        so_file = matches[0]
        lddtree = subprocess.check_output(["ldd", so_file]).decode("utf-8")
        self.assertFalse("torch_python" in lddtree)

        missing_symbols = subprocess.check_output(["nm", "-u", so_file]).decode("utf-8")
        self.assertFalse("Py" in missing_symbols)


devices = ("cuda", "xpu")
instantiate_device_type_tests(
    TestPythonAgnostic, globals(), only_for=devices, allow_xpu=True
)

if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestPythonAgnostic`

**Functions defined**: `setUpClass`, `test_extension_is_python_agnostic`

**Key imports**: os, shutil, subprocess, sys, unittest, Path, TEST_CUDA, instantiate_device_type_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp_extensions/python_agnostic_extension/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `shutil`
- `subprocess`
- `sys`
- `unittest`
- `pathlib`: Path
- `torch.testing._internal.common_cuda`: TEST_CUDA
- `torch.testing._internal.common_device_type`: instantiate_device_type_tests


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp_extensions/python_agnostic_extension/test/test_python_agnostic.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp_extensions/python_agnostic_extension/test`):



## Cross-References

- **File Documentation**: `test_python_agnostic.py_docs.md`
- **Keyword Index**: `test_python_agnostic.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp_extensions/python_agnostic_extension/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp_extensions/python_agnostic_extension/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/cpp_extensions/python_agnostic_extension/test/test_python_agnostic.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp_extensions/python_agnostic_extension/test`):

- [`test_python_agnostic.py_kw.md_docs.md`](./test_python_agnostic.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_python_agnostic.py_docs.md_docs.md`
- **Keyword Index**: `test_python_agnostic.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
