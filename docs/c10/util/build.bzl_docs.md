# Documentation: `c10/util/build.bzl`

## File Metadata

- **Path**: `c10/util/build.bzl`
- **Size**: 2,961 bytes (2.89 KB)
- **Type**: Source File (.bzl)
- **Extension**: `.bzl`

## File Purpose

This is a source file (.bzl) that is part of the PyTorch project.

## Original Source

```
def define_targets(rules):
    rules.cc_library(
        name = "TypeCast",
        srcs = ["TypeCast.cpp"],
        hdrs = ["TypeCast.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":base",
            "//c10/core:ScalarType",
            "//c10/macros",
        ],
    )

    rules.cc_library(
        name = "base",
        srcs = rules.glob(
            ["*.cpp"],
            exclude = [
                "TypeCast.cpp",
                "typeid.cpp",
            ],
        ),
        hdrs = rules.glob(
            ["*.h"],
            exclude = [
                "TypeCast.h",
                "typeid.h",
            ],
        ),
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":bit_cast",
            "//torch/headeronly:torch_headeronly",
            "//c10/macros",
            "@fmt",
            "@moodycamel//:moodycamel",
        ] + rules.select({
            "//c10:using_gflags": ["@com_github_gflags_gflags//:gflags"],
            "//conditions:default": [],
        }) + rules.select({
            "//c10:using_glog": ["@com_github_glog//:glog"],
            "//conditions:default": [],
        }),
        linkopts = rules.select({
            "@bazel_tools//src/conditions:windows": [],
            "//conditions:default": ["-ldl"],
        }),
        # This library uses flags and registration. Do not let the
        # linker remove them.
        alwayslink = True,
    )

    rules.cc_library(
        name = "bit_cast",
        hdrs = ["bit_cast.h"],
        visibility = ["//:__subpackages__"],
        deps = [
            "//c10/macros",
        ],
    )

    rules.cc_library(
        name = "ssize",
        hdrs = ["ssize.h"],
        linkstatic = True,
        visibility = ["//:__subpackages__"],
        deps = [":base"],
    )

    rules.cc_library(
        name = "typeid",
        srcs = ["typeid.cpp"],
        hdrs = ["typeid.h"],
        linkstatic = True,
        local_defines = ["C10_BUILD_MAIN_LIB"],
        visibility = ["//visibility:public"],
        deps = [
            ":base",
            "//c10/core:ScalarType",
            "//c10/macros",
        ],
    )

    rules.cc_library(
        name = "base_headers",
        hdrs = rules.glob(
            ["*.h"],
            exclude = [
                "bit_cast.h",
                "ssize.h",
            ],
        ),
        deps = [
            "//torch/headeronly:torch_headeronly",
        ],
        visibility = ["//visibility:public"],
    )

    rules.filegroup(
        name = "headers",
        srcs = rules.glob(
            ["*.h"],
            exclude = [
                "bit_cast.h",
                "ssize.h",
            ],
        ),
        visibility = [
            "//:__pkg__",
            "//c10:__pkg__",
        ],
    )

```



## High-Level Overview

This file is part of the PyTorch framework located at `c10/util`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`c10/util`):

- [`CallOnce.h_docs.md`](./CallOnce.h_docs.md)
- [`Unicode.cpp_docs.md`](./Unicode.cpp_docs.md)
- [`logging_is_not_google_glog.h_docs.md`](./logging_is_not_google_glog.h_docs.md)
- [`Array.h_docs.md`](./Array.h_docs.md)
- [`complex_math.h_docs.md`](./complex_math.h_docs.md)
- [`order_preserving_flat_hash_map.h_docs.md`](./order_preserving_flat_hash_map.h_docs.md)
- [`flags_use_gflags.cpp_docs.md`](./flags_use_gflags.cpp_docs.md)
- [`flags_use_no_gflags.cpp_docs.md`](./flags_use_no_gflags.cpp_docs.md)
- [`Float8_e4m3fnuz.h_docs.md`](./Float8_e4m3fnuz.h_docs.md)
- [`typeid.cpp_docs.md`](./typeid.cpp_docs.md)


## Cross-References

- **File Documentation**: `build.bzl_docs.md`
- **Keyword Index**: `build.bzl_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
