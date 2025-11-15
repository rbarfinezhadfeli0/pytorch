# Documentation: `test/cpp_extensions/setup.py`

## File Metadata

- **Path**: `test/cpp_extensions/setup.py`
- **Size**: 3,699 bytes (3.61 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file handles **configuration or setup**.

## Original Source

```python
import os
import sys

from setuptools import setup

import torch.cuda
from torch.testing._internal.common_utils import IS_WINDOWS
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDA_HOME,
    CUDAExtension,
    ROCM_HOME,
    SyclExtension,
)


if sys.platform == "win32":
    vc_version = os.getenv("VCToolsVersion", "")
    if vc_version.startswith("14.16."):
        CXX_FLAGS = ["/sdl"]
    else:
        CXX_FLAGS = ["/sdl", "/permissive-"]
else:
    CXX_FLAGS = ["-g"]

USE_NINJA = os.getenv("USE_NINJA") == "1"

ext_modules = [
    CppExtension(
        "torch_test_cpp_extension.cpp", ["extension.cpp"], extra_compile_args=CXX_FLAGS
    ),
    CppExtension(
        "torch_test_cpp_extension.maia",
        ["maia_extension.cpp"],
        extra_compile_args=CXX_FLAGS,
    ),
    CppExtension(
        "torch_test_cpp_extension.rng",
        ["rng_extension.cpp"],
        extra_compile_args=CXX_FLAGS,
    ),
]

if torch.cuda.is_available() and (CUDA_HOME is not None or ROCM_HOME is not None):
    extension = CUDAExtension(
        "torch_test_cpp_extension.cuda",
        [
            "cuda_extension.cpp",
            "cuda_extension_kernel.cu",
            "cuda_extension_kernel2.cu",
        ],
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": ["-O2"]},
    )
    ext_modules.append(extension)

if torch.cuda.is_available() and (CUDA_HOME is not None or ROCM_HOME is not None):
    extension = CUDAExtension(
        "torch_test_cpp_extension.torch_library",
        ["torch_library.cu"],
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": ["-O2"]},
    )
    ext_modules.append(extension)

if torch.backends.mps.is_available():
    extension = CppExtension(
        "torch_test_cpp_extension.mps",
        ["mps_extension.mm"],
        extra_compile_args=CXX_FLAGS,
    )
    ext_modules.append(extension)

if torch.xpu.is_available() and USE_NINJA:
    extension = SyclExtension(
        "torch_test_cpp_extension.sycl",
        ["xpu_extension.sycl"],
        extra_compile_args={"cxx": CXX_FLAGS, "sycl": ["-O2"]},
    )
    ext_modules.append(extension)


# todo(mkozuki): Figure out the root cause
if (not IS_WINDOWS) and torch.cuda.is_available() and CUDA_HOME is not None:
    # malfet: One should not assume that PyTorch re-exports CUDA dependencies
    cublas_extension = CUDAExtension(
        name="torch_test_cpp_extension.cublas_extension",
        sources=["cublas_extension.cpp"],
        libraries=["cublas"] if torch.version.hip is None else [],
    )
    ext_modules.append(cublas_extension)

    cusolver_extension = CUDAExtension(
        name="torch_test_cpp_extension.cusolver_extension",
        sources=["cusolver_extension.cpp"],
        libraries=["cusolver"] if torch.version.hip is None else [],
    )
    ext_modules.append(cusolver_extension)

if (
    USE_NINJA
    and (not IS_WINDOWS)
    and torch.cuda.is_available()
    and CUDA_HOME is not None
):
    extension = CUDAExtension(
        name="torch_test_cpp_extension.cuda_dlink",
        sources=[
            "cuda_dlink_extension.cpp",
            "cuda_dlink_extension_kernel.cu",
            "cuda_dlink_extension_add.cu",
        ],
        dlink=True,
        extra_compile_args={"cxx": CXX_FLAGS, "nvcc": ["-O2", "-dc"]},
    )
    ext_modules.append(extension)

setup(
    name="torch_test_cpp_extension",
    packages=["torch_test_cpp_extension"],
    ext_modules=ext_modules,
    include_dirs="self_compiler_include_dirs_test",
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=USE_NINJA)},
    entry_points={
        "torch.backends": [
            "device_backend = torch_test_cpp_extension:_autoload",
        ],
    },
)

```



## High-Level Overview


This Python file contains 0 class(es) and 0 function(s).

## Detailed Analysis

### Code Structure

**Key imports**: os, sys, setup, torch.cuda, IS_WINDOWS


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp_extensions`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `sys`
- `setuptools`: setup
- `torch.cuda`
- `torch.testing._internal.common_utils`: IS_WINDOWS


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp_extensions/setup.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp_extensions`):

- [`cpp_frontend_extension.cpp_docs.md`](./cpp_frontend_extension.cpp_docs.md)
- [`extension.cpp_docs.md`](./extension.cpp_docs.md)
- [`identity.cpp_docs.md`](./identity.cpp_docs.md)
- [`doubler.h_docs.md`](./doubler.h_docs.md)
- [`open_registration_extension.cpp_docs.md`](./open_registration_extension.cpp_docs.md)
- [`rng_extension.cpp_docs.md`](./rng_extension.cpp_docs.md)
- [`cusolver_extension.cpp_docs.md`](./cusolver_extension.cpp_docs.md)
- [`cuda_dlink_extension.cpp_docs.md`](./cuda_dlink_extension.cpp_docs.md)
- [`cuda_dlink_extension_add.cu_docs.md`](./cuda_dlink_extension_add.cu_docs.md)


## Cross-References

- **File Documentation**: `setup.py_docs.md`
- **Keyword Index**: `setup.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
