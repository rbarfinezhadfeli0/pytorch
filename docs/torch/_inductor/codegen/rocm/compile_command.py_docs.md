# Documentation: `torch/_inductor/codegen/rocm/compile_command.py`

## File Metadata

- **Path**: `torch/_inductor/codegen/rocm/compile_command.py`
- **Size**: 4,662 bytes (4.55 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs
import logging
import os
from typing import Optional

from torch._inductor import config
from torch._inductor.utils import is_linux, try_import_ck_lib


log = logging.getLogger(__name__)


def _rocm_include_paths(dst_file_ext: str) -> list[str]:
    from torch.utils import cpp_extension

    rocm_include = (
        os.path.join(config.rocm.rocm_home, "include")
        if config.rocm.rocm_home
        else cpp_extension._join_rocm_home("include")
    )

    if config.is_fbcode():
        from libfb.py import parutil

        ck_path = parutil.get_dir_path("composable-kernel-headers")
    else:
        if not config.rocm.ck_dir:
            ck_dir, _, _, _ = try_import_ck_lib()
            if not ck_dir:
                log.warning("Unspecified Composable Kernel directory")
            config.rocm.ck_dir = ck_dir
        ck_path = config.rocm.ck_dir or cpp_extension._join_rocm_home(
            "composable_kernel"
        )

    log.debug("Using ck path %s", ck_path)

    ck_include = os.path.join(ck_path, "include")
    ck_library_include = os.path.join(ck_path, "library", "include")

    # CK has to take priority over ROCm include paths
    # Since CK is potentially more up-to-date
    paths = [
        os.path.realpath(p) for p in (ck_include, ck_library_include, rocm_include)
    ]
    if dst_file_ext == "exe":
        ck_utility_include = os.path.join(ck_path, "library", "src", "utility")
        paths.append(os.path.realpath(ck_utility_include))
    return paths


def _rocm_lib_options(dst_file_ext: str) -> list[str]:
    from torch.utils import cpp_extension

    rocm_lib_dir = (
        os.path.join(config.rocm.rocm_home, "lib")
        if config.rocm.rocm_home
        else cpp_extension._join_rocm_home("lib")
    )
    hip_lib_dir = (
        os.path.join(config.rocm.rocm_home, "hip", "lib")
        if config.rocm.rocm_home
        else cpp_extension._join_rocm_home("hip", "lib")
    )

    opts = [
        "-include __clang_hip_runtime_wrapper.h",
        f"-L{os.path.realpath(rocm_lib_dir)}",
        f"-L{os.path.realpath(hip_lib_dir)}",
        "-lamdhip64",
    ]
    if dst_file_ext == "exe":
        opts += ["-lpthread", "-lstdc++"]
    return opts


def _rocm_compiler_options() -> list[str]:
    arch_list = config.rocm.arch or ["native"]
    gpu_arch_flags = [f"--offload-arch={arch}" for arch in arch_list]
    opts = [
        config.rocm.compile_opt_level,
        "-x",
        "hip",
        "-std=c++17",
        *gpu_arch_flags,
        "-fno-gpu-rdc",
        "-fPIC",
        "-fvisibility=hidden",
        "-mllvm",
        "-amdgpu-early-inline-all=true",
        "-mllvm",
        "-amdgpu-function-calls=false",
        "-mllvm",
        "-enable-post-misched=0",
    ]
    if config.rocm.is_debug:
        opts += ["-DDEBUG_LOG=1", "-g"]
    if config.rocm.save_temps:
        opts += ["--save-temps=obj"]
    if config.rocm.print_kernel_resource_usage:
        opts += ["-Rpass-analysis=kernel-resource-usage"]
    if config.rocm.flush_denormals:
        opts += ["-fgpu-flush-denormals-to-zero"]
    if config.rocm.use_fast_math:
        opts += ["-ffast-math"]
    return opts


def rocm_compiler() -> Optional[str]:
    if is_linux():
        if config.rocm.rocm_home:
            return os.path.realpath(
                os.path.join(config.rocm.rocm_home, "llvm", "bin", "clang")
            )
        try:
            from torch.utils import cpp_extension

            return os.path.realpath(
                cpp_extension._join_rocm_home("llvm", "bin", "clang")
            )
        except OSError:
            # neither config.rocm.rocm_home nor env variable ROCM_HOME are set
            return "clang"
    return None


def rocm_compile_command(
    src_files: list[str],
    dst_file: str,
    dst_file_ext: str,
    extra_args: Optional[list[str]] = None,
) -> str:
    include_paths = _rocm_include_paths(dst_file_ext)
    lib_options = _rocm_lib_options(dst_file_ext)
    compiler_options = _rocm_compiler_options()
    compiler = rocm_compiler()
    options = (
        compiler_options
        + (extra_args or [])
        + [f"-I{path}" for path in include_paths]
        + lib_options
    )
    src_file = " ".join(src_files)
    # supported extensions: .o, .so, .exe
    if dst_file_ext == "o":
        options.append("-c")
    elif dst_file_ext == "so":
        options.append("-shared")
    elif dst_file_ext == "exe":
        options.append("-DGENERATE_CK_STANDALONE_RUNNER")
    else:
        raise NotImplementedError(f"Unsupported output file suffix {dst_file_ext}!")
    return f"{compiler} {' '.join(options)} -o {dst_file} {src_file}"

```



## High-Level Overview


This Python file contains 0 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `_rocm_include_paths`, `_rocm_lib_options`, `_rocm_compiler_options`, `rocm_compiler`, `rocm_compile_command`

**Key imports**: logging, os, Optional, config, is_linux, try_import_ck_lib, cpp_extension, parutil, cpp_extension, cpp_extension


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/codegen/rocm`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `logging`
- `os`
- `typing`: Optional
- `torch._inductor`: config
- `torch._inductor.utils`: is_linux, try_import_ck_lib
- `torch.utils`: cpp_extension
- `libfb.py`: parutil


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/_inductor/codegen/rocm`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`ck_tile_universal_gemm_template.py_docs.md`](./ck_tile_universal_gemm_template.py_docs.md)
- [`rocm_benchmark_request.py_docs.md`](./rocm_benchmark_request.py_docs.md)
- [`rocm_template_buffer.py_docs.md`](./rocm_template_buffer.py_docs.md)
- [`ck_conv_template.py_docs.md`](./ck_conv_template.py_docs.md)
- [`rocm_template.py_docs.md`](./rocm_template.py_docs.md)
- [`ck_tile_template.py_docs.md`](./ck_tile_template.py_docs.md)
- [`rocm_cpp_scheduling.py_docs.md`](./rocm_cpp_scheduling.py_docs.md)
- [`ck_template.py_docs.md`](./ck_template.py_docs.md)
- [`rocm_utils.py_docs.md`](./rocm_utils.py_docs.md)


## Cross-References

- **File Documentation**: `compile_command.py_docs.md`
- **Keyword Index**: `compile_command.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
