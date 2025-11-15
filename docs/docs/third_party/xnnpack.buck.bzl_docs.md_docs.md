# Documentation: `docs/third_party/xnnpack.buck.bzl_docs.md`

## File Metadata

- **Path**: `docs/third_party/xnnpack.buck.bzl_docs.md`
- **Size**: 52,440 bytes (51.21 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `third_party/xnnpack.buck.bzl`

## File Metadata

- **Path**: `third_party/xnnpack.buck.bzl`
- **Size**: 76,650 bytes (74.85 KB)
- **Type**: Source File (.bzl)
- **Extension**: `.bzl`

## File Purpose

This is a source file (.bzl) that is part of the PyTorch project.

## Original Source

```
load("//tools/build_defs:fb_xplat_cxx_library.bzl", "fb_xplat_cxx_library")
load("//tools/build_defs:fbsource_utils.bzl", "is_arvr_mode")
load("//tools/build_defs:glob_defs.bzl", "subdir_glob")
load("//tools/build_defs:platform_defs.bzl", "ANDROID", "APPLE", "CXX", "IOS", "MACOSX", "WINDOWS")
load(
    "@fbsource//xplat/caffe2/third_party:xnnpack_buck_shim.bzl",
    "LOGGING_SRCS",
    "OPERATOR_SRCS",
    "SUBGRAPH_SRCS",
    "TABLE_SRCS",
    "XNNPACK_SRCS",
    "get_xnnpack_headers",
    "prod_srcs_for_arch_wrapper",
)

XNN_COMMON_PREPROCESSOR_FLAGS = [
    "-DXNN_PRIVATE=",
    "-DXNN_INTERNAL=",
    "-DXNN_LOG_LEVEL=0"
]

# This defines XNNPACK targets for both fbsource BUCK and OSS BUCK
# Note that the file path is relative to the BUCK file that called from, not to this bzl file.
# So for fbsource build it points to xplat/third-party/XNNPACK/XNNPACK,
# and for OSS it points to pytorch/third_party/XNNPACK
def define_xnnpack(third_party, labels = [], XNNPACK_WINDOWS_AVX512F_ENABLED = False):
    WINDOWS_FLAGS = [
        "/D__x86_64__",
        "/EHsc",
        "/wd4090",  # 'function': different 'const' qualifiers
        "/wd4146",  # unary minus operator applied to unsigned type, result still unsigned
    ] + ([
        "/D__AVX512F__",  # needed to avoid linkage errors
        "-mavx2",
        "/D__builtin_clz=__lzcnt",  # Intrinsics are spelled differently in MSVC
        "/Drestrict=",  # MSVC doesn't understand [restrict XNN_NUM_ELEMENTS(N)] syntax
    ] if XNNPACK_WINDOWS_AVX512F_ENABLED else [])

    WINDOWS_CLANG_COMPILER_FLAGS = [
        "-Wno-error",
        "-Wno-error=undef",
        "-Wno-error=incompatible-pointer-types",
        "-Wno-error=incompatible-pointer-types-discards-qualifiers",
    ]

    XNN_COMMON_MICROKERNEL_EXPORTED_DEPS = [
        ":interface",
        third_party("FP16"),
        third_party("FXdiv"),
    ]

    fb_xplat_cxx_library(
        name = "interface",
        header_namespace = "",
        exported_headers = {
            "xnnpack.h": "XNNPACK/include/xnnpack.h",
        },
        apple_sdks = (IOS, MACOSX),
        labels = labels,
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        exported_deps = [
            # Dependency only on pthreadpool interface
            third_party("pthreadpool_header"),
        ],
    )

    fb_xplat_cxx_library(
        name = "subgraph",
        srcs = SUBGRAPH_SRCS + ["XNNPACK/src/datatype.c"],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ],
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS + [
            "-DXNN_ENABLE_SPARSE=0",
            "-DXNN_ENABLE_MEMOPT",
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = [
            ":interface",
            third_party("FP16"),
            third_party("FXdiv"),
            third_party("clog"),
        ],
    )

    fb_xplat_cxx_library(
        name = "tables",
        srcs = TABLE_SRCS,
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ],
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = [
            ":interface",
            third_party("FP16"),
            third_party("FXdiv"),
            third_party("clog"),
        ],
    )

    fb_xplat_cxx_library(
        name = "ukernels_scalar",
        srcs = prod_srcs_for_arch_wrapper("scalar"),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
            "-fno-fast-math",
            "-fno-math-errno",
            "-ffp-contract=off",
        ],
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_sse",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("sse"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("sse"),
        }) if is_arvr_mode() else [],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-msse",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                prod_srcs_for_arch_wrapper("sse"),
            ),
        ] if not is_arvr_mode() else []),
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-msse"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-msse"],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_sse_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-msse",
                ],
            ),
        ],
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-msse"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-msse"],
        windows_srcs = prod_srcs_for_arch_wrapper("sse"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_sse2",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("sse2"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("sse2"),
        }) if is_arvr_mode() else [],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-msse2",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                prod_srcs_for_arch_wrapper("sse2"),
            ),
        ] if not is_arvr_mode() else []),
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-msse2"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-msse2"],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_sse2_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-msse2",
                ],
            ),
        ],
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-msse2"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-msse2"],
        windows_srcs = prod_srcs_for_arch_wrapper("sse2"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_ssse3",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("ssse3"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("ssse3"),
        }) if is_arvr_mode() else [],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-mssse3",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                prod_srcs_for_arch_wrapper("ssse3"),
            ),
        ] if not is_arvr_mode() else []),
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mssse3"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mssse3"],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_ssse3_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-mssse3",
                ],
            ),
        ],
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mssse3"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mssse3"],
        windows_srcs = prod_srcs_for_arch_wrapper("ssse3"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_sse41",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("sse41"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("sse41"),
        }) if is_arvr_mode() else [],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-msse4.1",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                prod_srcs_for_arch_wrapper("sse41"),
            ),
        ] if not is_arvr_mode() else []),
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-msse4.1"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-msse4.1"],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_sse41_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-msse4.1",
                ],
            ),
        ],
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-msse4.1"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-msse4.1"],
        windows_srcs = prod_srcs_for_arch_wrapper("sse41"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("avx"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("avx"),
        }) if is_arvr_mode() else [],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx",
            ],
        }),
        labels = labels,
        platform_compiler_flags = [
            (
                "x86|x86_64|platform009|platform010",
                [
                    "-mavx",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                prod_srcs_for_arch_wrapper("avx"),
            ),
        ] if not is_arvr_mode() else []),
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mavx"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mavx"],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
            "-mavx",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-mavx",
                ],
            ),
        ],
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mavx"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mavx"],
        windows_srcs = prod_srcs_for_arch_wrapper("avx"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512vnnigfni",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("avx512vnnigfni"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("avx512vnnigfni"),
        }) if is_arvr_mode() else [],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx",
                "-mgfni",
                "-mavx512vl",
                "-mavx512vnni",
                "-mavx512bw",
                "-mavx512dq",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx",
                "-mgfni",
                "-mavx512vl",
                "-mavx512vnni",
                "-mavx512bw",
                "-mavx512dq",
            ],
        }),
        labels = labels,
        platform_compiler_flags = [
            (
                "x86|x86_64|platform009|platform010",
                [
                    "-mavx512f",
                    "-mavx512cd",
                    "-mavx512bw",
                    "-mavx512dq",
                    "-mavx512vl",
                    "-mavx512vnni",
                    "-mgfni",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                prod_srcs_for_arch_wrapper("avx512vnnigfni"),
            ),
        ] if not is_arvr_mode() else []),
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mavx"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mavx"],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512vnnigfni_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86|x86_64|platform009|platform010",
                [
                    "-mavx512f",
                    "-mavx512cd",
                    "-mavx512bw",
                    "-mavx512dq",
                    "-mavx512vl",
                    "-mavx512vnni",
                    "-mgfni",
                ],
            ),
        ],
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mavx"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mavx"],
        windows_srcs = prod_srcs_for_arch_wrapper("avx512vnnigfni"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512vnni",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("avx512vnni"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("avx512vnni"),
        }) if is_arvr_mode() else [],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx512f",
                "-mavx512cd",
                "-mavx512bw",
                "-mavx512dq",
                "-mavx512vl",
                "-mavx512vnni",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx512f",
                "-mavx512cd",
                "-mavx512bw",
                "-mavx512dq",
                "-mavx512vl",
                "-mavx512vnni",
            ],
        }),
        labels = labels,
        platform_compiler_flags = [
            (
                "x86|x86_64|platform009|platform010",
                [
                    "-mavx512f",
                    "-mavx512cd",
                    "-mavx512bw",
                    "-mavx512dq",
                    "-mavx512vl",
                    "-mavx512vnni",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                prod_srcs_for_arch_wrapper("avx512vnni"),
            ),
        ] if not is_arvr_mode() else []),
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        exported_preprocessor_flags = [
            "-DXNN_ENABLE_AVX512VNNI"
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mavx"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mavx"],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512vnni_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86|x86_64|platform009|platform010",
                [
                    "-mavx512f",
                    "-mavx512cd",
                    "-mavx512bw",
                    "-mavx512dq",
                    "-mavx512vl",
                    "-mavx512vnni",
                ],
            ),
        ],
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        exported_preprocessor_flags = [
            "-DXNN_ENABLE_AVX512VNNI"
        ],
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mavx"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mavx"],
        windows_srcs = prod_srcs_for_arch_wrapper("avx512vnni"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avxvnni",
        srcs = prod_srcs_for_arch_wrapper("avxvnni") if is_arvr_mode() else [],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
            "-mavxvnni",
            "-mf16c",
            "-mfma",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86|x86_64|platform009|platform010",
                [
                    "-mavx2",
                    "-mavxvnni",
                    "-mf16c",
                    "-mfma",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                prod_srcs_for_arch_wrapper("avxvnni"),
            ),
        ] if not is_arvr_mode() else []),
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mavx"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mavx"],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avxvnni_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86|x86_64|platform009|platform010",
                [
                    "-mavx2",
                    "-mavxvnni",
                ],
            ),
        ],
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mavx"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mavx"],
        windows_srcs = prod_srcs_for_arch_wrapper("avxvnni"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_f16c",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("f16c"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("f16c"),
        }) if is_arvr_mode() else [],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mf16c",
            ],
            "ovr_config//cpu:x86_64": [
                "-mf16c",
            ],
        }),
        labels = labels,
        platform_compiler_flags = [
            (
                "x86|x86_64|platform009|platform010",
                [
                    "-mf16c",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                prod_srcs_for_arch_wrapper("f16c"),
            ),
        ] if not is_arvr_mode() else []),
        platforms = (APPLE, ANDROID, CXX, WINDOWS),
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mf16c"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mf16c"],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_f16c_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
            "-mf16c",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-mf16c",
                ],
            ),
        ],
        platforms = (APPLE, ANDROID, CXX, WINDOWS),
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mf16c"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mf16c"],
        windows_srcs = prod_srcs_for_arch_wrapper("f16c"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_fma3",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("fma3"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("fma3"),
        }) if is_arvr_mode() else [],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mfma",
                "-mf16c",
            ],
            "ovr_config//cpu:x86_64": [
                "-mfma",
                "-mf16c",
            ],
        }),
        labels = labels,
        platform_compiler_flags = [
            (
                "(i[3-6]86|x86|x86_64|AMD64)",
                [
                    "-mfma",
                    "-mf16c",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                prod_srcs_for_arch_wrapper("fma3"),
            ),
        ] if not is_arvr_mode() else []),
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + [
            "-mfma",
            "-mf16c",
        ],
        windows_compiler_flags_override = WINDOWS_FLAGS + [
            "-mfma",
            "-mf16c",
        ],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_fma3_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
            "-mfma",
            "-mf16c",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "^(i[3-6]86|x86|x86_64|AMD64)$",
                [
                    "-mfma",
                    "-mf16c",
                ],
            ),
        ],
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + [
            "-mfma",
            "-mf16c",
        ],
        windows_compiler_flags_override = WINDOWS_FLAGS + [
            "-mfma",
            "-mf16c",
        ],
        windows_srcs = prod_srcs_for_arch_wrapper("fma3"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx2",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("avx2"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("avx2"),
        }) if is_arvr_mode() else [],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx2",
                "-mfma",
                "-mf16c",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx2",
                "-mfma",
                "-mf16c",
            ],
        }),
        labels = labels,
        platform_compiler_flags = [
            (
                "x86|x86_64|platform009|platform010",
                [
                    "-mavx2",
                    "-mfma",
                    "-mf16c",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                prod_srcs_for_arch_wrapper("avx2"),
            ),
        ] if not is_arvr_mode() else []),
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + [
            "-mavx2",
            "-mfma",
            "-mf16c",
        ],
        windows_compiler_flags_override = WINDOWS_FLAGS + [
            "-mavx2",
            "-mfma",
            "-mf16c",
        ],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx2_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
            "-mavx2",
            "-mfma",
            "-mf16c",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-mavx2",
                    "-mfma",
                    "-mf16c",
                ],
            ),
        ],
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + [
            "-mavx2",
            "-mfma",
            "-mf16c",
        ],
        windows_compiler_flags_override = WINDOWS_FLAGS + [
            "/D__AVX2__",
            "-mavx2",
            "-mfma",
            "-mf16c",
        ],
        windows_srcs = prod_srcs_for_arch_wrapper("avx2"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("avx512f"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("avx512f"),
        }) if is_arvr_mode() else [],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx512f",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx512f",
            ],
        }),
        labels = labels,
        platform_compiler_flags = [
            (
                "x86|x86_64|platform009|platform010",
                [
                    "-mavx512f",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                prod_srcs_for_arch_wrapper("avx512f"),
            ),
        ] if not is_arvr_mode() else []),
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mavx512f"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mavx512f"],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512vbmi",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("avx512vbmi"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("avx512vbmi"),
        }) if is_arvr_mode() else [],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx512f",
                "-mavx512cd",
                "-mavx512bw",
                "-mavx512dq",
                "-mavx512vl",
                "-mavx512vbmi",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx512f",
                "-mavx512cd",
                "-mavx512bw",
                "-mavx512dq",
                "-mavx512vl",
                "-mavx512vbmi",
            ],
        }),
        labels = labels,
        platform_compiler_flags = [
            (
                "(i[3-6]86|x86|x86_64|AMD64)",
                [
                    "-mavx512f",
                    "-mavx512cd",
                    "-mavx512bw",
                    "-mavx512dq",
                    "-mavx512vl",
                    "-mavx512vbmi",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                prod_srcs_for_arch_wrapper("avx512vbmi"),
            ),
        ] if not is_arvr_mode() else []),
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + [
            "-mavx512f",
            "-mavx512cd",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
            "-mavx512vbmi",
        ],
        windows_compiler_flags_override = WINDOWS_FLAGS + [
            "-mavx512f",
            "-mavx512cd",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
            "-mavx512vbmi",
        ],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
            "-mavx512f",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "x86",
                [
                    "-mavx512f",
                ],
            ),
        ],
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + ["-mavx512f"],
        windows_compiler_flags_override = WINDOWS_FLAGS + ["-mavx512f"],
        windows_srcs = prod_srcs_for_arch_wrapper("avx512f"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512skx",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": prod_srcs_for_arch_wrapper("avx512skx"),
            "ovr_config//cpu:x86_64": prod_srcs_for_arch_wrapper("avx512skx"),
        }) if is_arvr_mode() else [],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:x86_32": [
                "-mavx512f",
                "-mavx512cd",
                "-mavx512bw",
                "-mavx512dq",
                "-mavx512vl",
            ],
            "ovr_config//cpu:x86_64": [
                "-mavx512f",
                "-mavx512cd",
                "-mavx512bw",
                "-mavx512dq",
                "-mavx512vl",
            ],
        }),
        labels = labels,
        platform_compiler_flags = [
            (
                "(i[3-6]86|x86|x86_64|AMD64)",
                [
                    "-mavx512f",
                    "-mavx512cd",
                    "-mavx512bw",
                    "-mavx512dq",
                    "-mavx512vl",
                ],
            ),
        ],
        platform_srcs = ([
            (
                "x86|x86_64|platform009|platform010",
                prod_srcs_for_arch_wrapper("avx512skx"),
            ),
        ] if not is_arvr_mode() else []),
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + [
            "-mavx512f",
            "-mavx512cd",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
        ],
        windows_compiler_flags_override = WINDOWS_FLAGS + [
            "-mavx512f",
            "-mavx512cd",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",

        ],
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_avx512skx_ovr_win32",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
            "-mavx512f",
            "-mavx512cd",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "^(i[3-6]86|x86|x86_64|AMD64)$",
                [
                    "-mavx512f",
                    "-mavx512cd",
                    "-mavx512bw",
                    "-mavx512dq",
                    "-mavx512vl",
                ],
            ),
        ],
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS + [
            "-mavx512f",
            "-mavx512cd",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
        ],
        windows_compiler_flags_override = WINDOWS_FLAGS + [
            "-mavx512f",
            "-mavx512cd",
            "-mavx512bw",
            "-mavx512dq",
            "-mavx512vl",
            "/D__AVX512BW__",
        ],
        windows_srcs = prod_srcs_for_arch_wrapper("avx512skx"),
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_armsimd32",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": prod_srcs_for_arch_wrapper("armsimd32"),
        }) if is_arvr_mode() else [],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
            "-fno-fast-math",
            "-fno-math-errno",
        ],
        labels = labels,
        platform_compiler_flags = [
            (
                "(arm32|aarch32|armv7)",
                [
                    "-marm",
                    "-march=armv6",
                    "-mfpu=vfp",
                    "-munaligned-access",
                ],
            ),
        ],
        platform_srcs = [
            (
                "(aarch32|arm32|armv7)",
                prod_srcs_for_arch_wrapper("armsimd32"),
            ),
        ] if not is_arvr_mode() else [],
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_neon",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": prod_srcs_for_arch_wrapper("neon"),
            "ovr_config//cpu:arm64": prod_srcs_for_arch_wrapper("neon"),
        }) if is_arvr_mode() else [],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": [
                "-marm",
                "-march=armv7-a",
                "-mfpu=neon",
            ],
        }),
        labels = labels,
        platform_compiler_flags = [
            (
                "(aarch32|arm32|armv7)",
                [
                    "-marm",
                    "-march=armv7-a",
                    "-mfpu=neon",
                ],
            ),
        ],
        platform_srcs = [
            (
                "(aarch32|arm32|armv7)",
                prod_srcs_for_arch_wrapper("neon"),
            ),
            (
                "(aarch64|arm64)",
                prod_srcs_for_arch_wrapper("neon"),
            ),
        ] if not is_arvr_mode() else [],
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_neon_aarch64",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm64": prod_srcs_for_arch_wrapper("neon_aarch64"), 
        }) if is_arvr_mode() else [],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ],
        platform_srcs = [
            (
                "(aarch64|arm64)",
                prod_srcs_for_arch_wrapper("neon_aarch64"),
            ),
        ] if not is_arvr_mode() else [],
        labels = labels,
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_neon_fma",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": prod_srcs_for_arch_wrapper("neonfma"),
        }) if is_arvr_mode() else [],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": [
                "-marm",
                "-march=armv7-a",
                "-mfpu=neon-vfpv4",
            ],
        }),
        labels = labels,
        platform_compiler_flags = [
            (
                "^iphoneos-armv7$",
                [
                    "-mcpu=cyclone",
                    "-mtune=generic",
                ],
            ),
            (
                "(aarch32|arm32|armv7)",
                [
                    "-marm",
                    "-march=armv7-a",
                    "-mfpu=neon-vfpv4",
                ],
            ),
        ],
        platform_srcs = [
            (
                "(aarch32|arm32|armv7)",
                prod_srcs_for_arch_wrapper("neonfma"),
            ),
        ] if not is_arvr_mode() else [],
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_neonfma_aarch64",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm64": prod_srcs_for_arch_wrapper("neonfma") + prod_srcs_for_arch_wrapper("neonfma_aarch64"),
        }) if is_arvr_mode() else [],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
        ],
        labels = labels,
        platform_srcs = [
            (
                "(arm64|aarch64)",
                prod_srcs_for_arch_wrapper("neonfma") + prod_srcs_for_arch_wrapper("neonfma_aarch64"),
            ),
        ] if not is_arvr_mode() else [],
        platforms = (APPLE, ANDROID, CXX, WINDOWS),
        fbandroid_link_whole = True,
        preferred_linkage = "static",
        preprocessor_flags = XNN_COMMON_PREPROCESSOR_FLAGS,
        visibility = ["PUBLIC"],
        windows_clang_compiler_flags_override = WINDOWS_FLAGS + WINDOWS_CLANG_COMPILER_FLAGS,
        windows_compiler_flags_override = WINDOWS_FLAGS,
        deps = XNN_COMMON_MICROKERNEL_EXPORTED_DEPS,
    )

    fb_xplat_cxx_library(
        name = "ukernels_fp16arith",
        srcs = select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": prod_srcs_for_arch_wrapper("fp16arith"),
            "ovr_config//cpu:arm64": prod_srcs_for_arch_wrapper("fp16arith"),
        }) if is_arvr_mode() else [],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        apple_sdks = (IOS, MACOSX),
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
            "-fno-fast-math",
            "-fno-math-errno",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": [
                "-marm",
                "-march=armv8.2-a+fp16",
                # GCC emits wrong directives for assembler with -mfpu=fp-armv8
                "-mfpu=neon-fp-armv8",
                # For vsqrth_f16 polyfill using sqrtf
                "-fno-math-errno",
                # For vminh_f16/vmaxh_f16 polyfills using compare + select
                "-ffinite-math-only",
            ],
            "ovr_config//cpu:arm64": [
                "-march=armv8.2-a+fp16",
            ],
        }),
        labels = labels,
        platform_compiler_flags = [
            (
                "(aarch32|arm32|armv7)",
                [
                    "-marm",
                    "-march=armv8.2-a+fp16",
                    # GCC emits wrong directives for assembler with -mfpu=fp-armv8
                    "-mfpu=neon-fp-armv8",
                    # For vsqrth_f16 polyfill using sqrtf
                    "-fno-math-errno",
                    # For vminh_f16/vmaxh_f16 polyfills using compare + select
                    "-ffinite-math-only",
                ],
            ),
            (
                "(aarch64|arm64)",
                [
                    "-march=armv8.2-a+fp16",
                ],
            )
        ],
        platform_srcs = [
            (
                "(aarch32|arm32|armv7)",
                prod_src
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/third_party`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/third_party`, which is part of the PyTorch project infrastructure.



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

Files in the same folder (`docs/third_party`):

- [`substitution.bzl_kw.md_docs.md`](./substitution.bzl_kw.md_docs.md)
- [`xnnpack_buck_shim.bzl_kw.md_docs.md`](./xnnpack_buck_shim.bzl_kw.md_docs.md)
- [`LICENSES_BUNDLED.txt_kw.md_docs.md`](./LICENSES_BUNDLED.txt_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`kineto.buck.bzl_docs.md_docs.md`](./kineto.buck.bzl_docs.md_docs.md)
- [`generate-cpuinfo-wrappers.py_kw.md_docs.md`](./generate-cpuinfo-wrappers.py_kw.md_docs.md)
- [`xnnpack_buck_shim.bzl_docs.md_docs.md`](./xnnpack_buck_shim.bzl_docs.md_docs.md)
- [`eigen_pin.txt_docs.md_docs.md`](./eigen_pin.txt_docs.md_docs.md)
- [`build_bundled.py_kw.md_docs.md`](./build_bundled.py_kw.md_docs.md)
- [`generate-cpuinfo-wrappers.py_docs.md_docs.md`](./generate-cpuinfo-wrappers.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `xnnpack.buck.bzl_docs.md_docs.md`
- **Keyword Index**: `xnnpack.buck.bzl_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
