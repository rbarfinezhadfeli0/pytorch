# Documentation: `c10/ovrsource_defs.bzl`

## File Metadata

- **Path**: `c10/ovrsource_defs.bzl`
- **Size**: 4,444 bytes (4.34 KB)
- **Type**: Source File (.bzl)
- **Extension**: `.bzl`

## File Purpose

This is a source file (.bzl) that is part of the PyTorch project.

## Original Source

```
load("//arvr/tools/build_defs:genrule_utils.bzl", "gen_cmake_header")
load("//arvr/tools/build_defs:oxx.bzl", "oxx_static_library")

cpu_supported_platforms = [
    "ovr_config//os:android",
    "ovr_config//os:iphoneos",
    "ovr_config//os:linux-x86_64",
    "ovr_config//os:macos",
    "ovr_config//os:windows-x86_64",
    "ovr_config//runtime:arm64-linux-ubuntu-neon",
    "ovr_config//os:linux-arm64",
]

cuda_supported_platforms = [
    "ovr_config//os:linux-cuda",
    "ovr_config//os:windows-cuda",
]

def define_c10_ovrsource(name, is_mobile):
    if is_mobile:
        pp_flags = ["-DC10_MOBILE=1"]
    else:
        pp_flags = []

    oxx_static_library(
        name = name,
        srcs = native.glob([
            "core/*.cpp",
            "core/impl/*.cpp",
            "mobile/*.cpp",
            "util/*.cpp",
        ]),
        compatible_with = cpu_supported_platforms,
        compiler_flags = select({
            "DEFAULT": [],
            "ovr_config//compiler:cl": [
                "/w",
            ],
            "ovr_config//toolchain/clang:win": [
                "-Wno-error",
                "-Wno-shadow",
                "-Wno-undef",
                "-Wno-unused-variable",
            ],
        }),
        include_directories = [".."],
        preprocessor_flags = [
            "-DNO_EXPORT",
            "-DC10_BUILD_MAIN_LIB=1",
            "-DSUPPORTS_BACKTRACE=0",
        ],
        fbobjc_compiler_flags = ["-Wno-error=global-constructors", "-Wno-error=missing-prototypes"],
        public_include_directories = [".."],
        public_preprocessor_flags = pp_flags,
        public_raw_headers = native.glob([
            "core/*.h",
            "macros/*.h",
            "mobile/*.h",
            "test/util/*.h",  # some external tests use this
            "util/*.h",
        ]),
        raw_headers = native.glob([
            "core/impl/*.h",
        ]),
        reexport_all_header_dependencies = False,
        visibility = [
            "//xplat/caffe2/c10:c10_ovrsource",
        ],
        deps = select({
            "DEFAULT": [],
            "ovr_config//os:linux": [
                "//third-party/numactl:numactl",
            ],
        }),
        exported_deps = [
            "//xplat/caffe2/torch/headeronly:torch_headeronly_ovrsource",
            "//arvr/third-party/gflags:gflags",
            "//third-party/cpuinfo:cpuinfo",
            "//third-party/fmt:fmt",
            "//third-party/glog:glog",
        ],
    )

def define_ovrsource_targets():
    c10_cuda_macros = gen_cmake_header(
        src = "cuda/impl/cuda_cmake_macros.h.in",
        defines = [
            ("#cmakedefine C10_CUDA_BUILD_SHARED_LIBS", ""),
        ],
        header = "c10/cuda/impl/cuda_cmake_macros.h",
        prefix = "ovrsource",
    )

    oxx_static_library(
        name = "c10_ovrsource",
        compatible_with = cpu_supported_platforms,
        exported_deps = select({
            "DEFAULT": [":c10_full_ovrsource"],
            "ovr_config//os:android": [":c10_mobile_ovrsource"],
            "ovr_config//os:iphoneos": [":c10_mobile_ovrsource"],
        }),
        visibility = ["PUBLIC"],
    )

    """
    Most users should use c10_ovrsource, not these targets directly.
    """
    define_c10_ovrsource("c10_mobile_ovrsource", True)
    define_c10_ovrsource("c10_full_ovrsource", False)

    oxx_static_library(
        name = "c10_cuda_ovrsource",
        srcs = native.glob([
            "cuda/*.cpp",
            "cuda/impl/*.cpp",
        ]),
        compatible_with = cuda_supported_platforms,
        compiler_flags = select({
            "DEFAULT": [],
            "ovr_config//compiler:cl": [
                "/w",
            ],
            "ovr_config//toolchain/clang:win": [
                "-Wno-error",
                "-Wno-shadow",
                "-Wno-undef",
                "-Wno-unused-variable",
            ],
        }),
        link_whole = True,
        preprocessor_flags = [
            "-DNO_EXPORT",
            "-DC10_CUDA_BUILD_MAIN_LIB=1",
        ],
        raw_headers = native.glob([
            "cuda/*.h",
            "cuda/impl/*.h",
        ]),
        reexport_all_header_dependencies = False,
        visibility = ["PUBLIC"],
        deps = [
            "//third-party/cuda:libcuda",
            "//third-party/cuda:libcudart",
        ],
        exported_deps = c10_cuda_macros + [
            ":c10_ovrsource",
        ],
    )

```



## High-Level Overview

This file is part of the PyTorch framework located at `c10`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



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

Files in the same folder (`c10`):

- [`build.bzl_docs.md`](./build.bzl_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`BUILD.bazel_docs.md`](./BUILD.bazel_docs.md)


## Cross-References

- **File Documentation**: `ovrsource_defs.bzl_docs.md`
- **Keyword Index**: `ovrsource_defs.bzl_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
