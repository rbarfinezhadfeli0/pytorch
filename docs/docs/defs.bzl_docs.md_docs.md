# Documentation: `docs/defs.bzl_docs.md`

## File Metadata

- **Path**: `docs/defs.bzl_docs.md`
- **Size**: 4,908 bytes (4.79 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `defs.bzl`

## File Metadata

- **Path**: `defs.bzl`
- **Size**: 2,653 bytes (2.59 KB)
- **Type**: Source File (.bzl)
- **Extension**: `.bzl`

## File Purpose

This is a source file (.bzl) that is part of the PyTorch project.

## Original Source

```
def get_blas_gomp_arch_deps():
    return [
        ("x86_64", [
            "fbsource//third-party/mkl:{}".format(native.read_config("fbcode", "mkl_lp64", "mkl_lp64_omp")),
        ]),
        ("aarch64", [
            "third-party//Arm-Performance-Libraries:armpl_lp64_mp",
            "third-party//openmp:omp",
        ]),
    ]

default_compiler_flags = [
    "-Wall",
    "-Wextra",
    "-Wno-unused-function",
    "-Wno-unused-parameter",
    "-Wno-error=strict-aliasing",
    "-Wno-shadow-compatible-local",
    "-Wno-maybe-uninitialized",  # aten is built with gcc as part of HHVM
    "-Wno-unknown-pragmas",
    "-Wno-strict-overflow",
    # See https://fb.facebook.com/groups/fbcode/permalink/1813348245368673/
    # These trigger on platform007
    "-Wno-stringop-overflow",
    "-Wno-class-memaccess",
    "-DHAVE_MMAP",
    "-DUSE_GCC_ATOMICS=1",
    "-D_FILE_OFFSET_BITS=64",
    "-DHAVE_SHM_OPEN=1",
    "-DHAVE_SHM_UNLINK=1",
    "-DHAVE_MALLOC_USABLE_SIZE=1",
    "-DCPU_CAPABILITY_DEFAULT",
    "-DTH_INDEX_BASE=0",
    "-DMAGMA_V2",
    "-DNO_CUDNN_DESTROY_HANDLE",
    "-DUSE_FBGEMM",
    "-DUSE_PYTORCH_QNNPACK",
    # The dynamically loaded NVRTC trick doesn't work in fbcode,
    # and it's not necessary anyway, because we have a stub
    # nvrtc library which we load canonically anyway
    "-DUSE_DIRECT_NVRTC",
    "-DUSE_RUY_QMATMUL",
] + select({
    # XNNPACK depends on an updated version of pthreadpool interface, whose implementation
    # includes <pthread.h> - a header not available on Windows.
    "DEFAULT": ["-DUSE_XNNPACK"],
    "ovr_config//os:windows": [],
})

compiler_specific_flags = {
    "clang": [
        "-Wno-absolute-value",
        "-Wno-pass-failed",
        "-Wno-braced-scalar-init",
    ],
    "gcc": [
        "-Wno-error=array-bounds",
    ],
}

def get_cpu_parallel_backend_flags():
    parallel_backend = native.read_config("pytorch", "parallel_backend", "openmp")
    defs = []
    if parallel_backend == "openmp":
        defs.append("-DAT_PARALLEL_OPENMP_FBCODE=1")
    elif parallel_backend == "native":
        defs.append("-DAT_PARALLEL_NATIVE_FBCODE=1")
    else:
        fail("Unsupported parallel backend: " + parallel_backend)
    if native.read_config("pytorch", "exp_single_thread_pool", "0") == "1":
        defs.append("-DAT_EXPERIMENTAL_SINGLE_THREAD_POOL=1")
    mkl_ver = native.read_config("fbcode", "mkl_lp64", "mkl_lp64_omp")
    if mkl_ver == "mkl_lp64_seq":
        defs.append("-DATEN_MKL_SEQUENTIAL_FBCODE=1")
    return defs

def is_cpu_static_dispatch_build():
    mode = native.read_config("fbcode", "caffe2_static_dispatch_mode", "none")
    return mode == "cpu"

```



## High-Level Overview

This file is part of the PyTorch framework located at ``.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `root`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`root`):

- [`AGENTS.md_docs.md`](./AGENTS.md_docs.md)
- [`pytest.ini_docs.md`](./pytest.ini_docs.md)
- [`codex_setup.sh_docs.md`](./codex_setup.sh_docs.md)
- [`pt_template_srcs.bzl_docs.md`](./pt_template_srcs.bzl_docs.md)
- [`aten.bzl_docs.md`](./aten.bzl_docs.md)
- [`build.bzl_docs.md`](./build.bzl_docs.md)
- [`buckbuild.bzl_docs.md`](./buckbuild.bzl_docs.md)
- [`Dockerfile_docs.md`](./Dockerfile_docs.md)
- [`.bc-linter.yml_docs.md`](./.bc-linter.yml_docs.md)
- [`setup.py_docs.md`](./setup.py_docs.md)


## Cross-References

- **File Documentation**: `defs.bzl_docs.md`
- **Keyword Index**: `defs.bzl_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs`):

- [`Makefile_docs.md`](./Makefile_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`requirements.txt_docs.md`](./requirements.txt_docs.md)
- [`libtorch.rst_docs.md`](./libtorch.rst_docs.md)
- [`BUILD.bazel_docs.md_docs.md`](./BUILD.bazel_docs.md_docs.md)
- [`generate_repo_docs.py_kw.md_docs.md`](./generate_repo_docs.py_kw.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`pt_template_srcs.bzl_kw.md_docs.md`](./pt_template_srcs.bzl_kw.md_docs.md)
- [`CLAUDE.md_docs.md_docs.md`](./CLAUDE.md_docs.md_docs.md)
- [`setup.py_kw.md_docs.md`](./setup.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `defs.bzl_docs.md_docs.md`
- **Keyword Index**: `defs.bzl_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
