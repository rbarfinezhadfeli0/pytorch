# Documentation: `torch/headeronly/ovrsource_defs.bzl`

## File Metadata

- **Path**: `torch/headeronly/ovrsource_defs.bzl`
- **Size**: 4,006 bytes (3.91 KB)
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

def define_torch_headeronly_ovrsource(name, is_mobile):
    if is_mobile:
        pp_flags = ["-DC10_MOBILE=1"]
    else:
        pp_flags = []

    oxx_static_library(
        name = name,
        srcs = [],
        compatible_with = cpu_supported_platforms,
        compiler_flags = select({
            "DEFAULT": [],
        }),
        preprocessor_flags = ["-DC10_BUILD_MAIN_LIB=1",],
        fbobjc_compiler_flags = [],
        public_include_directories = ["../.."],
        public_preprocessor_flags = pp_flags,
        public_raw_headers = native.glob([
            "core/**/*.h",
            "cpu/**/*.h",
            "macros/*.h",
            "util/*.h",
        ]),
        reexport_all_header_dependencies = False,
        visibility = [
            "//xplat/caffe2/torch/headeronly:torch_headeronly_ovrsource",
        ],
        exported_deps = [
            ":ovrsource_torch_headeronly_cmake_macros.h",
            ":generate-version-header",
        ],
    )

def define_ovrsource_targets():
    common_c10_cmake_defines = [
        ("#cmakedefine C10_BUILD_SHARED_LIBS", ""),
        ("#cmakedefine C10_USE_NUMA", ""),
        ("#cmakedefine C10_USE_MSVC_STATIC_RUNTIME", ""),
        ("#cmakedefine C10_USE_ROCM_KERNEL_ASSERT", ""),
    ]

    mobile_c10_cmake_defines = [
        ("#cmakedefine C10_USE_GLOG", ""),
        ("#cmakedefine C10_USE_GFLAGS", ""),
    ]

    non_mobile_c10_cmake_defines = [
        ("#cmakedefine C10_USE_GLOG", "#define C10_USE_GLOG 1"),
        ("#cmakedefine C10_USE_GFLAGS", "#define C10_USE_GFLAGS 1"),
    ]

    gen_cmake_header(
        src = "macros/cmake_macros.h.in",
        defines = common_c10_cmake_defines + mobile_c10_cmake_defines,
        header = "torch/headeronly/macros/cmake_macros.h",
        prefix = "ovrsource_torch_headeronly_mobile_",
    )

    gen_cmake_header(
        src = "macros/cmake_macros.h.in",
        defines = common_c10_cmake_defines + non_mobile_c10_cmake_defines,
        header = "torch/headeronly/macros/cmake_macros.h",
        prefix = "ovrsource_torch_headeronly_non_mobile_",
    )

    oxx_static_library(
        name = "ovrsource_torch_headeronly_cmake_macros.h",
        compatible_with = [
            "ovr_config//os:android",
            "ovr_config//os:iphoneos",
            "ovr_config//os:linux",
            "ovr_config//os:macos",
            "ovr_config//os:windows",
        ],
        deps = select({
            "ovr_config//os:android": [":ovrsource_torch_headeronly_mobile_cmake_macros.h"],
            "ovr_config//os:iphoneos": [":ovrsource_torch_headeronly_mobile_cmake_macros.h"],
            "ovr_config//os:linux": [":ovrsource_torch_headeronly_non_mobile_cmake_macros.h"],
            "ovr_config//os:macos": [":ovrsource_torch_headeronly_non_mobile_cmake_macros.h"],
            "ovr_config//os:windows": [":ovrsource_torch_headeronly_non_mobile_cmake_macros.h"],
        }),
    )

    oxx_static_library(
        name = "torch_headeronly_ovrsource",
        compatible_with = cpu_supported_platforms,
        exported_deps = select({
            "DEFAULT": [":torch_headeronly_full_ovrsource"],
            "ovr_config//os:android": [":torch_headeronly_mobile_ovrsource"],
            "ovr_config//os:iphoneos": [":torch_headeronly_mobile_ovrsource"],
        }),
        visibility = ["PUBLIC"],
    )

    """
    Most users should use torch_headeronly_ovrsource, not these targets directly.
    """
    define_torch_headeronly_ovrsource("torch_headeronly_mobile_ovrsource", True)
    define_torch_headeronly_ovrsource("torch_headeronly_full_ovrsource", False)

```



## High-Level Overview

This file is part of the PyTorch framework located at `torch/headeronly`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/headeronly`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/headeronly`):

- [`build.bzl_docs.md`](./build.bzl_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`BUILD.bazel_docs.md`](./BUILD.bazel_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)


## Cross-References

- **File Documentation**: `ovrsource_defs.bzl_docs.md`
- **Keyword Index**: `ovrsource_defs.bzl_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
