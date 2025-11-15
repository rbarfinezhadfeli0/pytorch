# Documentation: `docs/test/cpp_extensions/open_registration_extension/torch_openreg/setup.py_docs.md`

## File Metadata

- **Path**: `docs/test/cpp_extensions/open_registration_extension/torch_openreg/setup.py_docs.md`
- **Size**: 7,084 bytes (6.92 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This file handles **configuration or setup**.

## Original Source

```markdown
# Documentation: `test/cpp_extensions/open_registration_extension/torch_openreg/setup.py`

## File Metadata

- **Path**: `test/cpp_extensions/open_registration_extension/torch_openreg/setup.py`
- **Size**: 4,451 bytes (4.35 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file handles **configuration or setup**. Can be **executed as a standalone script**.

## Original Source

```python
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from distutils.command.clean import clean

from setuptools import Extension, find_packages, setup


# Env Variables
IS_DARWIN = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
RUN_BUILD_DEPS = any(arg in {"clean", "dist_info"} for arg in sys.argv)


def make_relative_rpath_args(path):
    if IS_DARWIN:
        return ["-Wl,-rpath,@loader_path/" + path]
    elif IS_WINDOWS:
        return []
    else:
        return ["-Wl,-rpath,$ORIGIN/" + path]


def get_pytorch_dir():
    # Disable autoload of the accelerator

    # We must do this for two reasons:
    # We only need to get the PyTorch installation directory, so whether the accelerator is loaded or not is irrelevant
    # If the accelerator has been previously built and not uninstalled, importing torch will cause a circular import error
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch

    return os.path.dirname(os.path.realpath(torch.__file__))


def build_deps():
    build_dir = os.path.join(BASE_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    cmake_args = [
        "-DCMAKE_INSTALL_PREFIX="
        + os.path.realpath(os.path.join(BASE_DIR, "torch_openreg")),
        "-DPYTHON_INCLUDE_DIR=" + sysconfig.get_paths().get("include"),
        "-DPYTORCH_INSTALL_DIR=" + get_pytorch_dir(),
    ]

    subprocess.check_call(
        ["cmake", BASE_DIR] + cmake_args, cwd=build_dir, env=os.environ
    )

    build_args = [
        "--build",
        ".",
        "--target",
        "install",
        "--config",  # For multi-config generators
        "Release",
        "--",
    ]

    if IS_WINDOWS:
        build_args += ["/m:" + str(multiprocessing.cpu_count())]
    else:
        build_args += ["-j", str(multiprocessing.cpu_count())]

    command = ["cmake"] + build_args
    subprocess.check_call(command, cwd=build_dir, env=os.environ)


class BuildClean(clean):
    def run(self):
        for i in ["build", "install", "torch_openreg/lib"]:
            dirs = os.path.join(BASE_DIR, i)
            if os.path.exists(dirs) and os.path.isdir(dirs):
                shutil.rmtree(dirs)

        for dirpath, _, filenames in os.walk(os.path.join(BASE_DIR, "torch_openreg")):
            for filename in filenames:
                if filename.endswith(".so"):
                    os.remove(os.path.join(dirpath, filename))


def main():
    if not RUN_BUILD_DEPS:
        build_deps()

    if IS_WINDOWS:
        # /NODEFAULTLIB makes sure we only link to DLL runtime
        # and matches the flags set for protobuf and ONNX
        extra_link_args: list[str] = ["/NODEFAULTLIB:LIBCMT.LIB"] + [
            *make_relative_rpath_args("lib")
        ]
        # /MD links against DLL runtime
        # and matches the flags set for protobuf and ONNX
        # /EHsc is about standard C++ exception handling
        extra_compile_args: list[str] = ["/MD", "/FS", "/EHsc"]
    else:
        extra_link_args = [*make_relative_rpath_args("lib")]
        extra_compile_args = [
            "-Wall",
            "-Wextra",
            "-Wno-strict-overflow",
            "-Wno-unused-parameter",
            "-Wno-missing-field-initializers",
            "-Wno-unknown-pragmas",
            "-fno-strict-aliasing",
        ]

    ext_modules = [
        Extension(
            name="torch_openreg._C",
            sources=["torch_openreg/csrc/stub.c"],
            language="c",
            extra_compile_args=extra_compile_args,
            libraries=["torch_bindings"],
            library_dirs=[os.path.join(BASE_DIR, "torch_openreg/lib")],
            extra_link_args=extra_link_args,
        )
    ]

    package_data = {
        "torch_openreg": [
            "lib/*.so*",
            "lib/*.dylib*",
            "lib/*.dll",
            "lib/*.lib",
        ]
    }

    # LITERALINCLUDE START: SETUP
    setup(
        packages=find_packages(),
        package_data=package_data,
        ext_modules=ext_modules,
        cmdclass={
            "clean": BuildClean,  # type: ignore[misc]
        },
        include_package_data=False,
        entry_points={
            "torch.backends": [
                "torch_openreg = torch_openreg:_autoload",
            ],
        },
    )
    # LITERALINCLUDE END: SETUP


if __name__ == "__main__":
    main()

```



## High-Level Overview


This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BuildClean`

**Functions defined**: `make_relative_rpath_args`, `get_pytorch_dir`, `build_deps`, `run`, `main`

**Key imports**: multiprocessing, os, platform, shutil, subprocess, sys, sysconfig, clean, Extension, find_packages, setup, error


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp_extensions/open_registration_extension/torch_openreg`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `multiprocessing`
- `os`
- `platform`
- `shutil`
- `subprocess`
- `sys`
- `sysconfig`
- `distutils.command.clean`: clean
- `setuptools`: Extension, find_packages, setup
- `error`
- `torch`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp_extensions/open_registration_extension/torch_openreg/setup.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp_extensions/open_registration_extension/torch_openreg`):

- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`pyproject.toml_docs.md`](./pyproject.toml_docs.md)


## Cross-References

- **File Documentation**: `setup.py_docs.md`
- **Keyword Index**: `setup.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp_extensions/open_registration_extension/torch_openreg`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp_extensions/open_registration_extension/torch_openreg`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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
python docs/test/cpp_extensions/open_registration_extension/torch_openreg/setup.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp_extensions/open_registration_extension/torch_openreg`):

- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`setup.py_kw.md_docs.md`](./setup.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`pyproject.toml_docs.md_docs.md`](./pyproject.toml_docs.md_docs.md)
- [`pyproject.toml_kw.md_docs.md`](./pyproject.toml_kw.md_docs.md)
- [`README.md_kw.md_docs.md`](./README.md_kw.md_docs.md)
- [`CMakeLists.txt_kw.md_docs.md`](./CMakeLists.txt_kw.md_docs.md)


## Cross-References

- **File Documentation**: `setup.py_docs.md_docs.md`
- **Keyword Index**: `setup.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
