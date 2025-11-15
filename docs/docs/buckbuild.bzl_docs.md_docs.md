# Documentation: `docs/buckbuild.bzl_docs.md`

## File Metadata

- **Path**: `docs/buckbuild.bzl_docs.md`
- **Size**: 52,228 bytes (51.00 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `buckbuild.bzl`

## File Metadata

- **Path**: `buckbuild.bzl`
- **Size**: 87,962 bytes (85.90 KB)
- **Type**: Source File (.bzl)
- **Extension**: `.bzl`

## File Purpose

This is a source file (.bzl) that is part of the PyTorch project.

## Original Source

```
# NOTE: This file is shared by internal and OSS BUCK build.
# These load paths point to different files in internal and OSS environment

load("@bazel_skylib//lib:paths.bzl", "paths")
load("//tools/build_defs:cell_defs.bzl", "get_fbsource_cell")
load("//tools/build_defs:fb_native_wrapper.bzl", "fb_native")
load("//tools/build_defs:fb_xplat_cxx_library.bzl", "fb_xplat_cxx_library")
load("//tools/build_defs:fb_xplat_genrule.bzl", "fb_xplat_genrule")
load("//tools/build_defs/windows:windows_flag_map.bzl", "windows_convert_gcc_clang_flags")
load("//tools/build_defs:fbsource_utils.bzl", "is_arvr_mode")
load("//tools/build_defs:glob_defs.bzl", "subdir_glob")
load("//tools/build_defs:platform_defs.bzl", "IOS", "MACOSX")
load("//tools/build_defs:type_defs.bzl", "is_list", "is_string")
load("//tools/build_defs/android:build_mode_defs.bzl", is_production_build_android = "is_production_build")
load("//tools/build_defs/apple:build_mode_defs.bzl", is_production_build_ios = "is_production_build", is_profile_build_ios = "is_profile_build")
load(
    ":build_variables.bzl",
    "aten_cpu_source_list",
    "aten_native_source_list",
    "core_sources_common",
    "core_sources_full_mobile_no_backend_interface_xplat",
    "core_trainer_sources",
    "jit_core_headers",
    "jit_core_sources",
    "libtorch_profiler_sources",
    "torch_cpp_srcs",
    "torch_mobile_tracer_sources",
)
load(
    ":pt_ops.bzl",
    "USED_PT_BACKENDS",
)
load(
    ":pt_template_srcs.bzl",
    "METAL_MASKRCNN_SOURCE_LIST",
    "METAL_SOURCE_LIST",
    "TEMPLATE_MASKRCNN_SOURCE_LIST",
    "TEMPLATE_SOURCE_LIST",
    "aten_ufunc_generated_all_cpu_sources",
    "get_gen_oplist_outs",
    "get_generate_code_bin_outs",
    "get_metal_registration_files_outs",
    "get_metal_registration_files_outs_windows",
    "get_metal_source_dict",
    "get_template_registration_file_rules",
    "get_template_registration_files_outs",
    "get_template_source_dict",
)
load(
    ":ufunc_defs.bzl",
    "aten_ufunc_generated_cpu_kernel_sources",
    "aten_ufunc_generated_cpu_sources",
    "aten_ufunc_generated_cuda_sources",
)

def read_bool(section, field, default, required = True):
    val = read_config(section, field)
    if val != None:
        if val in ["true", "True", "1"]:
            return True
        elif val in ["false", "False", "0"]:
            return False
        else:
            fail(
                "`{}:{}`: must be one of (0, 1, true, false, True, False), but was {}".format(section, field, val),
            )
    elif default != None:
        return default
    elif not required:
        return None
    else:
        fail("`{}:{}`: no value set".format(section, field))

def _is_build_mode_dev():
    if is_production_build_android():
        # Android Prod builds
        return False
    if is_production_build_ios() or is_profile_build_ios():
        # iOS Prod builds
        return False

    return True

def _get_enable_lightweight_dispatch():
    return read_bool("pt", "enable_lightweight_dispatch", False)

def _get_enable_record_kernel_dtype():
    return read_bool("pt", "enable_record_kernel_dtype", False)

def get_enable_mobile_dispatch_keys_trimming():
    return read_bool("pt", "enable_mobile_dispatch_keys_trimming", False)

def get_disable_per_op_profiling():
    return read_bool("pt", "disable_per_op_profiling", True)

def get_strip_error_messages():
    if IS_OSS:
        return True  # always strip in OSS CI to expose potential issues
    return read_bool("pt", "strip_error_messages", not _is_build_mode_dev())

def get_disable_warn():
    return read_bool("pt", "disable_warn", False)

def get_enable_eager_symbolication():
    return read_bool("pt", "enable_eager_symbolication", default = False, required = False)

def get_static_dispatch_backend():
    static_dispatch_backend = native.read_config("pt", "static_dispatch_backend", None)
    if static_dispatch_backend == None:
        return []
    return static_dispatch_backend.split(";")

def get_glsl_image_format():
    if read_config("pt", "vulkan_full_precision", "0") == "0":
        return "rgba16f"
    return "rgba32f"

def get_glsl_paths():
    paths = [
        "//xplat/caffe2:aten_vulkan_glsl_src_path",
        "aten/src/ATen/native/vulkan/glsl",
    ] + [
        p
        for p in read_config("gen_vulkan_spv", "additional_glsl_paths", "").split(" ")
        if p
    ]

    if len(paths) % 2 != 0:
        fail(
            "gen_vulkan_spv.additional_glsl_paths must contain an even number of elements",
        )

    return " ".join(
        [
            "$(location {})/{}".format(
                paths[i],
                paths[i + 1],
            )
            for i in range(
                0,
                len(paths),
                2,
            )
        ],
    )

def spv_shader_library():
    pass

IS_OSS = read_config("pt", "is_oss", "0") == "1"  # True for OSS BUCK build, and False for internal BUCK build

NOT_OSS = not IS_OSS

# for targets in caffe2 root path
ROOT = "//" if IS_OSS else "//xplat/caffe2"

# for targets in subfolders
ROOT_PATH = "//" if IS_OSS else "//xplat/caffe2/"

C10 = "//c10:c10" if IS_OSS else "//xplat/caffe2/c10:c10"

# a dictionary maps third party library name to fbsource and oss target
THIRD_PARTY_LIBS = {
    "FP16": ["//xplat/third-party/FP16:FP16", "//third_party:FP16"],
    "FXdiv": ["//xplat/third-party/FXdiv:FXdiv", "//third_party:FXdiv"],
    "XNNPACK": ["//xplat/third-party/XNNPACK:XNNPACK", "//third_party:XNNPACK"],
    "clog": ["//xplat/third-party/clog:clog", "//third_party:clog"],
    "cpuinfo": ["//third-party/cpuinfo:cpuinfo", "//third_party:cpuinfo"],
    "flatbuffers-api": ["//third-party/flatbuffers/fbsource_namespace:flatbuffers-api", "//third_party:flatbuffers-api"],
    "flatc": ["//third-party/flatbuffers/fbsource_namespace:flatc", "//third_party:flatc"],
    "fmt": ["//third-party/fmt:fmt", "//third_party:fmt"],
    "glog": ["//third-party/glog:glog", "//third_party:glog"],
    "gmock": ["//third-party/googletest:gmock_main", "//third_party:gmock"],
    "gtest": ["//third-party/googletest:gtest_main", "//third_party:gtest"],
    "kineto": ["//xplat/kineto/libkineto:libkineto", "//third_party:libkineto"],
    "libkineto_headers": ["//xplat/kineto/libkineto:libkineto_headers", "//third_party:libkineto_headers"],
    "omp": ["//xplat/third-party/linker_lib:omp", "//third_party:no-op"],
    "pocketfft": ["//third-party/pocket_fft:pocketfft", "//third_party:pocketfft_header"],
    "psimd": ["//xplat/third-party/psimd:psimd", "//third_party:psimd"],
    "pthreadpool": ["fbsource//xplat/third-party/pthreadpool:pthreadpool", "//third_party:pthreadpool"],
    "pthreadpool_header": ["fbsource//xplat/third-party/pthreadpool:pthreadpool_header", "//third_party:pthreadpool_header"],
    "moodycamel": ["//third-party/moodycamel:moodycamel", "//third_party:moodycamel"],
    "pyyaml": ["//third-party/pypi/pyyaml:pyyaml", "//third_party:pyyaml"],
    "rt": ["//xplat/third-party/linker_lib:rt", "//third_party:rt"],
    "ruy": ["//third-party/ruy:ruy_xplat_lib", "//third_party:ruy_lib"],
    "nlohmann-json": ["fbsource//third-party/nlohmann-json:nlohmann-json", "//third_party:nlohmann-json"],
    "sleef_arm": ["//third-party/sleef:sleef", "//third_party:sleef_arm"],
}

def third_party(name):
    if name not in THIRD_PARTY_LIBS:
        fail("Cannot find third party library " + name + ", please register it in THIRD_PARTY_LIBS first!")
    return THIRD_PARTY_LIBS[name][1] if IS_OSS else THIRD_PARTY_LIBS[name][0]

def get_pt_compiler_flags():
    return select({
        "DEFAULT": _PT_COMPILER_FLAGS,
        "ovr_config//compiler:cl": windows_convert_gcc_clang_flags(_PT_COMPILER_FLAGS),
    }) + select({
        "DEFAULT": [],
        "ovr_config//os:macos": ["-fvisibility=default"],
    })

_PT_COMPILER_FLAGS = [
    "-fexceptions",
    "-frtti",
    "-Os",
    "-Wno-unknown-pragmas",
    "-Wno-write-strings",
    "-Wno-unused-variable",
    "-Wno-unused-function",
    "-Wno-deprecated-declarations",
    "-Wno-shadow",
    "-Wno-global-constructors",
    "-Wno-missing-prototypes",
]

ATEN_COMPILER_FLAGS = [
    "-fexceptions",
    "-frtti",
    "-Os",
    "-Wno-absolute-value",
    "-Wno-deprecated-declarations",
    "-Wno-macro-redefined",
    "-Wno-tautological-constant-out-of-range-compare",
    "-Wno-unknown-pragmas",
    "-Wno-unknown-warning-option",
    "-Wno-unused-function",
    "-Wno-unused-variable",
    "-Wno-pass-failed",
    "-Wno-shadow",
] + select({
    # Not supported by clang on Windows
    "DEFAULT": ["-fPIC"],
    "ovr_config//compiler:clang-windows": [],
}) + select({
    "DEFAULT": [],
    "ovr_config//os:macos": ["-fvisibility=default"],
})

def get_aten_compiler_flags():
    return select({
        "DEFAULT": ATEN_COMPILER_FLAGS,
        "ovr_config//compiler:cl": windows_convert_gcc_clang_flags(ATEN_COMPILER_FLAGS),
    })

_COMMON_PREPROCESSOR_FLAGS = [
    "-DC10_MOBILE",
    "-DNO_EXPORT",
] + (
    ["-DC10_MOBILE_TRIM_DISPATCH_KEYS"] if get_enable_mobile_dispatch_keys_trimming() else []
) + (
    ["-DSTRIP_ERROR_MESSAGES"] if get_strip_error_messages() else []
) + (
    ["-DDISABLE_WARN"] if get_disable_warn() else []
)

def get_no_as_needed_linker_flag():
    return select({"DEFAULT": ["-Wl,--no-as-needed"], "ovr_config//os:macos": []})

def get_aten_preprocessor_flags():
    # read_config is not allowed outside of function in Starlark
    ATEN_PREPROCESSOR_FLAGS = _COMMON_PREPROCESSOR_FLAGS + [
        "-DCPU_CAPABILITY_DEFAULT",
        "-DCPU_CAPABILITY=DEFAULT",
        "-DCAFFE2_USE_LITE_PROTO",
        "-DATEN_CUDNN_ENABLED_FBXPLAT=0",
        "-DATEN_MKLDNN_ENABLED_FBXPLAT=0",
        "-DATEN_MKLDNN_ACL_ENABLED_FBXPLAT=0",
        "-DATEN_NNPACK_ENABLED_FBXPLAT=0",
        "-DATEN_MKL_ENABLED_FBXPLAT=0",
        "-DATEN_MKL_SEQUENTIAL_FBXPLAT=0",
        "-DUSE_PYTORCH_METAL",
        "-DUSE_PYTORCH_QNNPACK",
        "-DUSE_XNNPACK",
        "-DPYTORCH_QNNPACK_RUNTIME_QUANTIZATION",
        "-DAT_PARALLEL_OPENMP_FBXPLAT=0",
        "-DAT_PARALLEL_NATIVE_FBXPLAT=1",
        "-DUSE_LAPACK_FBXPLAT=0",
        "-DAT_BLAS_F2C_FBXPLAT=0",
        "-DAT_BLAS_USE_CBLAS_DOT_FBXPLAT=0",
        "-DUSE_RUY_QMATMUL",
    ]
    if get_disable_per_op_profiling():
        ATEN_PREPROCESSOR_FLAGS.append("-DPYTORCH_DISABLE_PER_OP_PROFILING")
    if _get_enable_record_kernel_dtype():
        ATEN_PREPROCESSOR_FLAGS.append("-DENABLE_RECORD_KERNEL_FUNCTION_DTYPE")
    return ATEN_PREPROCESSOR_FLAGS

def get_pt_preprocessor_flags():
    # read_config is not allowed outside of function in Starlark
    PT_PREPROCESSOR_FLAGS = _COMMON_PREPROCESSOR_FLAGS + [
        "-D_THP_CORE",
        "-DUSE_SCALARS",
        "-DNO_CUDNN_DESTROY_HANDLE",
    ]

    if _is_build_mode_dev():
        PT_PREPROCESSOR_FLAGS.append("-DENABLE_PYTORCH_NON_PRODUCTION_BUILDS")
    return PT_PREPROCESSOR_FLAGS

# This needs to be kept in sync with https://github.com/pytorch/pytorch/blob/release/1.9/torchgen/gen.py#L892  @lint-ignore
PT_BACKEND_HEADERS = [
    "CPU",
    "CUDA",
    "CompositeExplicitAutograd",
    "CompositeExplicitAutogradNonFunctional",
    "CompositeImplicitAutograd",
    "CompositeImplicitAutogradNestedTensor",
    "Meta",
]

def get_aten_static_dispatch_backend_headers(existing_headers):
    static_backends = get_static_dispatch_backend()
    for backend in static_backends:
        if backend != "CPU":
            existing_headers["{}Functions.h".format(backend)] = ":gen_aten[{}Functions.h]".format(backend)
            existing_headers["{}Functions_inl.h".format(backend)] = ":gen_aten[{}Functions_inl.h]".format(backend)
    return existing_headers

def get_aten_codegen_extra_params(backends):
    extra_params = {
        "force_schema_registration": True,
    }
    static_backends = get_static_dispatch_backend()
    if static_backends:
        extra_params["static_dispatch_backend"] = static_backends
        extra_params["enabled_backends"] = static_backends
    else:
        extra_params["enabled_backends"] = backends
    return extra_params

def get_jit_codegen_params():
    return []

def get_unboxing_generated_files():
    srcs = []
    if _get_enable_lightweight_dispatch():
        srcs = [
            "UnboxingFunctions.h",
            "UnboxingFunctions_0.cpp",
            "UnboxingFunctions_1.cpp",
            "UnboxingFunctions_2.cpp",
            "UnboxingFunctions_3.cpp",
            "UnboxingFunctions_4.cpp",
            "RegisterCodegenUnboxedKernels_0.cpp",
            "RegisterCodegenUnboxedKernels_1.cpp",
            "RegisterCodegenUnboxedKernels_2.cpp",
            "RegisterCodegenUnboxedKernels_3.cpp",
            "RegisterCodegenUnboxedKernels_4.cpp",
            "RegisterCodegenUnboxedKernels_5.cpp",
            "RegisterCodegenUnboxedKernels_6.cpp",
            "RegisterCodegenUnboxedKernels_7.cpp",
            "RegisterCodegenUnboxedKernels_8.cpp",
            "RegisterCodegenUnboxedKernels_9.cpp",
        ]
    res = {}
    for file_name in srcs:
        res[file_name] = [file_name]
    return res

def get_aten_generated_files(enabled_backends):
    # NB: RegisterMeta counts as an optionally enabled backend,
    # and is intentionally omitted from here
    src_files = [
        "RegisterBackendSelect.cpp",
        "RegisterCompositeImplicitAutograd_0.cpp",
        "RegisterCompositeImplicitAutogradNestedTensor_0.cpp",
        "RegisterCompositeExplicitAutograd_0.cpp",
        "RegisterCompositeExplicitAutogradNonFunctional_0.cpp",
        "CompositeViewCopyKernels.cpp",
        "RegisterSchema.cpp",
        "Declarations.yaml",
        "Functions.cpp",
        "Functions.h",
        "RedispatchFunctions.h",
        "NativeFunctions.h",
        "NativeMetaFunctions.h",
        "MethodOperators.h",
        "FunctionalInverses.h",
        "Operators.h",
        "Operators_0.cpp",
        "Operators_1.cpp",
        "Operators_2.cpp",
        "Operators_3.cpp",
        "Operators_4.cpp",
        "CompositeImplicitAutogradFunctions.h",
        "CompositeImplicitAutogradFunctions_inl.h",
        "CompositeImplicitAutogradNestedTensorFunctions.h",
        "CompositeImplicitAutogradNestedTensorFunctions_inl.h",
        "CompositeExplicitAutogradFunctions.h",
        "CompositeExplicitAutogradFunctions_inl.h",
        "CompositeExplicitAutogradNonFunctionalFunctions.h",
        "CompositeExplicitAutogradNonFunctionalFunctions_inl.h",
        "ViewMetaClasses.h",
        "ViewMetaClasses.cpp",
        "VmapGeneratedPlumbing.h",
        "core/ATenOpList.cpp",
        "core/TensorBody.h",
        "core/TensorMethods.cpp",
        "core/aten_interned_strings.h",
        "core/enum_tag.h",
        "torch/csrc/inductor/aoti_torch/generated/c_shim_cpu.cpp",
    ] + get_aten_derived_type_srcs(enabled_backends)

    # This is tiresome.  A better strategy would be to unconditionally
    # generate these files, and then only actually COMPILE them depended
    # on the generated set.  C'est la vie...  # codespell:ignore vie
    if "CPU" in enabled_backends:
        src_files.extend(aten_ufunc_generated_cpu_sources())
        src_files.extend(aten_ufunc_generated_cpu_kernel_sources())
    if "CUDA" in enabled_backends:
        # Cannot unconditionally include this, because in the Edge selective
        # build CUDA is not enabled and thus the ufunc codegen for CUDA gets
        # skipped
        src_files.extend(aten_ufunc_generated_cuda_sources())

    res = {}
    for file_name in src_files:
        res[file_name] = [file_name]
    return res

def get_aten_derived_type_src_rules(aten_rule_name, enabled_backends):
    return [
        ":{}[{}]".format(aten_rule_name, "Register" + backend + "_0.cpp")
        for backend in enabled_backends if backend != "CPU"
    ] + ([
        ":{}[RegisterCPU_{}.cpp]".format(aten_rule_name, x) for x in range(4)
    ] if "CPU" in enabled_backends else [])

def get_aten_selective_cpp_rules(aten_rule_name, enabled_backends):
    return [
        ":{}[{}]".format(aten_rule_name, f)
        for f in ["RegisterCompositeImplicitAutograd_0.cpp", "RegisterCompositeImplicitAutogradNestedTensor_0.cpp", "RegisterCompositeExplicitAutograd_0.cpp", "RegisterCompositeExplicitAutogradNonFunctional_0.cpp", "RegisterSchema.cpp", "RegisterBackendSelect.cpp", "CompositeViewCopyKernels.cpp"]
    ] + get_aten_derived_type_src_rules(aten_rule_name, enabled_backends)

def get_aten_derived_type_srcs(enabled_backends):
    return [
        "Register" + derived_type + "_0.cpp"
        for derived_type in enabled_backends if derived_type != "CPU"
    ] + [
        derived_type + "Functions.h"
        for derived_type in enabled_backends
        if derived_type in PT_BACKEND_HEADERS or derived_type in get_static_dispatch_backend()
    ] + [
        derived_type + "Functions_inl.h"
        for derived_type in enabled_backends
        if derived_type in PT_BACKEND_HEADERS or derived_type in get_static_dispatch_backend()
    ] + ([
        "RegisterCPU_{}.cpp".format(x) for x in range(4)
    ] if "CPU" in enabled_backends else [])

def gen_aten_files(
        name,
        extra_flags = {},
        visibility = [],
        compatible_with = [],
        apple_sdks = None):
    extra_params = []
    force_schema_registration = extra_flags.get("force_schema_registration", False)
    op_registration_allowlist = extra_flags.get("op_registration_allowlist", None)
    op_selection_yaml_path = extra_flags.get("op_selection_yaml_path", None)
    enabled_backends = extra_flags.get("enabled_backends", None)
    static_dispatch_backend = extra_flags.get("static_dispatch_backend", None)

    if force_schema_registration:
        extra_params.append("--force_schema_registration")
    if op_registration_allowlist != None and is_string(op_registration_allowlist):
        extra_params.append("--op_registration_whitelist")
        extra_params.append(op_registration_allowlist)
    if op_selection_yaml_path != None and is_string(op_selection_yaml_path):
        extra_params.append("--op_selection_yaml_path")
        extra_params.append(op_selection_yaml_path)
    if enabled_backends != None and is_list(enabled_backends):
        extra_params.append("--backend_whitelist")
        extra_params.extend(enabled_backends)
    if _get_enable_lightweight_dispatch():
        extra_params.append("--skip_dispatcher_op_registration")
    if static_dispatch_backend:
        extra_params.append("--static_dispatch_backend")
        extra_params.extend(static_dispatch_backend)
        backends = static_dispatch_backend
    else:
        backends = enabled_backends
    fb_xplat_genrule(
        name = name,
        default_outs = ["."],
        outs = get_aten_generated_files(backends),
        cmd = "$(exe {}torchgen:gen) ".format(ROOT_PATH) + " ".join([
            "--source-path $(location {}:aten_src_path)/aten/src/ATen".format(ROOT),
            "--install_dir $OUT",
            "--aoti_install_dir $OUT/torch/csrc/inductor/aoti_torch/generated"
        ] + extra_params),
        visibility = visibility,
        compatible_with = compatible_with,
        apple_sdks = apple_sdks,
    )

def gen_aten_unboxing_files(
        genrule_name,
        extra_flags = {}):
    extra_params = []
    op_selection_yaml_path = extra_flags.get("op_selection_yaml_path", None)
    op_registration_allowlist = extra_flags.get("op_registration_allowlist", None)
    if op_selection_yaml_path != None and is_string(op_selection_yaml_path):
        extra_params.append("--op_selection_yaml_path")
        extra_params.append(op_selection_yaml_path)
    if op_registration_allowlist != None and is_string(op_registration_allowlist):
        extra_params.append("--op_registration_allowlist")
        extra_params.append(op_registration_allowlist)

    fb_xplat_genrule(
        name = genrule_name,
        default_outs = ["."],
        outs = get_unboxing_generated_files(),
        cmd = "$(exe {}tools:gen_unboxing_bin) ".format(ROOT_PATH) + " ".join([
            "--source-path $(location {}:aten_src_path)/aten/src/ATen".format(ROOT),
            "--install_dir $OUT",
        ] + extra_params),
        visibility = ["PUBLIC"],
    )

def copy_template_registration_files(name, apple_sdks = None):
    cmd = []
    cmd_exe = []

    template_source_dict = get_template_source_dict()

    # Ideally, we would run one copy command for a single source directory along
    # with all its child directories, but it's somewhat hard to know if a directory
    # is a child of another just by looking at the metadata (directory relative
    # path) that we currently have since 1 directory could look like a parent of
    # another and yet come from a different filegroup() rule.
    #
    for (path_prefix, file_paths) in template_source_dict.items():
        cmd.append("mkdir -p $OUT/{}".format(path_prefix))
        cmd_exe.append("md $OUT/{}".format(path_prefix))

        # Adding *.cpp is a workaround to prevent cp from thrown an error when it
        # encounters a directory (since -r was not specified). If files with an
        # extension other than .cpp need to be copied, then the command below
        # will not work and will need to be updated.
        #
        cmd.append("cp -f $(location {0}:templated_selective_build_srcs)/{1}/*.cpp $OUT/{1}/".format(ROOT, path_prefix))
        cmd_exe.append("robocopy /E $(location {0}:templated_selective_build_srcs)/{1} $OUT/{1}".format(ROOT, path_prefix))

    if NOT_OSS:
        for file_path in TEMPLATE_MASKRCNN_SOURCE_LIST:
            maskrcnn_file = "$(location //xplat/caffe2/fb/custom_ops/maskrcnn:templated_selective_build_srcs)/" + file_path
            cmd.append("cp -f " + maskrcnn_file + " $OUT")
            cmd_exe.append("copy " + maskrcnn_file + " $OUT")

    cmd.append("mkdir -p $OUT/aten/src/ATen")
    cmd_exe.append("md $OUT/aten/src/ATen")

    # NB: CUDA is skipped here because this is selective build and CUDA is not
    # supported for selective build
    for ufunc_file in aten_ufunc_generated_all_cpu_sources("$(location " + ROOT + ":gen_aten[{}])"):
        cmd.append("cp -f " + ufunc_file + " $OUT/aten/src/ATen")
        cmd_exe.append("copy " + ufunc_file + " $OUT/aten/src/ATen")

    if NOT_OSS:
        pvd_batch_box_cox_file = "$(location //xplat/caffe2/fb/custom_ops/batch_box_cox:templated_selective_build_srcs)/register_batch_box_cox_ops.cpp"
        cmd.append("cp -f " + pvd_batch_box_cox_file + " $OUT")
        cmd_exe.append("copy " + pvd_batch_box_cox_file + " $OUT")

    fb_xplat_genrule(
        name = name,
        cmd = " && ".join(cmd),
        cmd_exe = "@powershell -Command " + ("; ".join(cmd_exe)),
        outs = get_template_registration_files_outs(IS_OSS),
        default_outs = ["."],
        apple_sdks = apple_sdks,
    )

def get_feature_tracer_source_list():
    """
    Return just the Feature specific handlers used in the model tracer.
    """
    sources = []
    for s in torch_mobile_tracer_sources:
        if s.endswith("Tracer.cpp"):
            sources.append(s)
    return sources

def pt_operator_query_codegen(
        name,
        deps = [],
        train = False,
        enforce_traced_op_list = False,
        pt_allow_forced_schema_registration = True,
        compatible_with = [],
        apple_sdks = None):
    if get_fbsource_cell() == "fbcode":
        return

    oplist_dir_name = name + "_pt_oplist"

    # @lint-ignore BUCKLINT
    fb_native.genrule(
        name = oplist_dir_name,
        cmd = ("$(exe {}tools:gen_oplist) ".format(ROOT_PATH) +
               "--model_file_list_path $(@query_outputs 'attrfilter(labels, pt_operator_library, deps(set({deps})))') " +
               ("" if enforce_traced_op_list else "--allow_include_all_overloads ") +
               "--output_dir $OUT ").format(deps = " ".join(["\"{}\"".format(d) for d in deps])),
        outs = get_gen_oplist_outs(),
        default_outs = ["."],
        compatible_with = compatible_with,
    )

    # Aten files
    aten_genrule = name + "_aten"
    extra_flags = {
        "enabled_backends": USED_PT_BACKENDS,
        "op_selection_yaml_path": "$(location :{}[selected_operators.yaml])".format(oplist_dir_name),
    }

    if train and pt_allow_forced_schema_registration:
        extra_flags["force_schema_registration"] = True

    unboxing_genrule = name + "_unboxing"
    if _get_enable_lightweight_dispatch():
        gen_aten_unboxing_files(
            unboxing_genrule,
            extra_flags = extra_flags,
        )

    static_dispatch_backend = get_static_dispatch_backend()
    if static_dispatch_backend:
        extra_flags["static_dispatch_backend"] = static_dispatch_backend

    gen_aten_files(
        aten_genrule,
        extra_flags = extra_flags,
        compatible_with = compatible_with,
        apple_sdks = apple_sdks,
    )

    # unboxing_wrappers files
    extra_params = [
        "--operators_yaml_path",
        "$(location :" + oplist_dir_name + "[selected_operators.yaml])",
    ]
    unboxing_and_autograd_genrule = name + "_unboxing_and_autograd"
    gen_aten_libtorch_files(
        unboxing_and_autograd_genrule,
        extra_params,
        compatible_with,
        apple_sdks = apple_sdks,
    )

    # Template runtime files (prim ops, etc)
    template_registration_genrule = name + "_template_registration"
    copy_template_registration_files(template_registration_genrule, apple_sdks = apple_sdks)

    # Files needed for metal
    if NOT_OSS:
        metal_genrule = name + "_metal"
        copy_metal(metal_genrule, apple_sdks = apple_sdks)

    srcs = get_aten_selective_cpp_rules(
        aten_genrule,
        static_dispatch_backend if static_dispatch_backend else USED_PT_BACKENDS,
    ) + get_template_registration_file_rules(template_registration_genrule, IS_OSS) + ([
        ":{}[autograd/generated/VariableType_0.cpp]".format(unboxing_and_autograd_genrule),
        ":{}[autograd/generated/VariableType_1.cpp]".format(unboxing_and_autograd_genrule),
        ":{}[autograd/generated/VariableType_2.cpp]".format(unboxing_and_autograd_genrule),
        ":{}[autograd/generated/VariableType_3.cpp]".format(unboxing_and_autograd_genrule),
        ":{}[autograd/generated/VariableType_4.cpp]".format(unboxing_and_autograd_genrule),
        ":{}[autograd/generated/ADInplaceOrViewType_0.cpp]".format(unboxing_and_autograd_genrule),
        ":{}[autograd/generated/ADInplaceOrViewType_1.cpp]".format(unboxing_and_autograd_genrule),
    ] if train else []) + ([
        ":{}[SupportedMobileModelsRegistration.cpp]".format(oplist_dir_name),
    ] if NOT_OSS else [])

    headers = {
        "selected_mobile_ops.h": ":{}[selected_mobile_ops.h]".format(oplist_dir_name),
    }

    if _get_enable_lightweight_dispatch():
        srcs.extend([
            ":{}[UnboxingFunctions_0.cpp]".format(unboxing_genrule),
            ":{}[UnboxingFunctions_1.cpp]".format(unboxing_genrule),
            ":{}[UnboxingFunctions_2.cpp]".format(unboxing_genrule),
            ":{}[UnboxingFunctions_3.cpp]".format(unboxing_genrule),
            ":{}[UnboxingFunctions_4.cpp]".format(unboxing_genrule),
            ":{}[RegisterCodegenUnboxedKernels_0.cpp]".format(unboxing_genrule),
            ":{}[RegisterCodegenUnboxedKernels_1.cpp]".format(unboxing_genrule),
            ":{}[RegisterCodegenUnboxedKernels_2.cpp]".format(unboxing_genrule),
            ":{}[RegisterCodegenUnboxedKernels_3.cpp]".format(unboxing_genrule),
            ":{}[RegisterCodegenUnboxedKernels_4.cpp]".format(unboxing_genrule),
            ":{}[RegisterCodegenUnboxedKernels_5.cpp]".format(unboxing_genrule),
            ":{}[RegisterCodegenUnboxedKernels_6.cpp]".format(unboxing_genrule),
            ":{}[RegisterCodegenUnboxedKernels_7.cpp]".format(unboxing_genrule),
            ":{}[RegisterCodegenUnboxedKernels_8.cpp]".format(unboxing_genrule),
            ":{}[RegisterCodegenUnboxedKernels_9.cpp]".format(unboxing_genrule),
        ])
        headers["UnboxingFunctions.h"] = ":{}[UnboxingFunctions.h]".format(unboxing_genrule)
    return {"headers": headers, "srcs": srcs}

def gen_aten_libtorch_files(name, extra_params = [], compatible_with = [], apple_sdks = None):
    fb_xplat_genrule(
        name = name,
        outs = get_generate_code_bin_outs(),
        default_outs = ["."],
        bash = "mkdir -p tools && " +
               "$(exe {}tools:generate_code_bin) ".format(ROOT_PATH) + " ".join(
            # Mobile build only needs libtorch - skip python bindings for now, except
            # for ovrsource, which needs Python bindings.
            (["--subset libtorch"] if not is_arvr_mode() else []) + [
                "--native-functions-path $(location {}:aten_src_path)/aten/src/ATen/native/native_functions.yaml".format(ROOT),
                "--tags-path $(location {}:aten_src_path)/aten/src/ATen/native/tags.yaml".format(ROOT),
                "--install_dir $OUT",
            ] + extra_params,
        ),
        cmd_exe = "@powershell -Command New-Item -Path tools -ItemType Directory -Force; " +
                  "$(exe {}tools:generate_code_bin) ".format(ROOT_PATH) + " ".join(
            # Mobile build only needs libtorch - skip python bindings for now, except
            # for ovrsource, which needs Python bindings.
            (["--subset libtorch"] if not is_arvr_mode() else []) + [
                "--native-functions-path $(location {}:aten_src_path)/aten/src/ATen/native/native_functions.yaml".format(ROOT),
                "--tags-path $(location {}:aten_src_path)/aten/src/ATen/native/tags.yaml".format(ROOT),
                "--install_dir $OUT",
            ] + extra_params,
        ),
        compatible_with = compatible_with,
        apple_sdks = apple_sdks,
    )

def vulkan_spv_shader_library(name, spv_filegroup):
    genrule_cmd = [
        "$(exe //xplat/caffe2/tools:gen_aten_vulkan_spv_bin)",
        "--glsl-paths $(location {})".format(spv_filegroup),
        "--output-path $OUT --env FLOAT_IMAGE_FORMAT={}".format(get_glsl_image_format()),
        "--glslc-path=$(exe //xplat/caffe2/fb/vulkan/dotslash:glslc)",
        "--tmp-dir-path=$TMP",
    ]

    genrule_name = "gen_{}_cpp".format(name)
    fb_xplat_genrule(
        name = "gen_{}_cpp".format(name),
        outs = {
            "{}.cpp".format(name): ["spv.cpp"],
        },
        cmd = " ".join(genrule_cmd),
        default_outs = ["."],
    )

    fb_xplat_cxx_library(
        name = name,
        srcs = [
            ":{}[{}.cpp]".format(genrule_name, name),
        ],
        # Static initialization is used to register shaders to the global shader registry,
        # therefore link_whole must be True to make sure unused symbols are not discarded.
        # @lint-ignore BUCKLINT: Avoid `link_whole=True`
        link_whole = True,
        # Define a soname that can be used for dynamic loading in Java, Python, etc.
        soname = "lib{}.$(ext)".format(name),
        visibility = ["PUBLIC"],
        exported_deps = [
            "//xplat/caffe2:torch_vulkan_api",
        ],
    )

def copy_metal(name, apple_sdks = None):
    cmd = []
    cmd_exe = []
    metal_source_dict = get_metal_source_dict()

    # Copy all source files over to bring them into the per app build
    for path_prefix in sorted(metal_source_dict.keys()):
        cmd.append("mkdir -p $OUT/{}".format(path_prefix))
        cmd_exe.append("mkdir -Force $OUT/{0}".format(path_prefix))

        # Not every directory has a mm or cpp file so '2>/dev/null || :' are tricks to suppress the error messages and codes.
        cmd.append("cp -f {0}/{1}/*.mm $OUT/{1}/ 2>/dev/null || :".format("$(location //xplat/caffe2:metal_build_srcs)", path_prefix))
        cmd.append("cp -f {0}/{1}/*.cpp $OUT/{1}/ 2>/dev/null || :".format("$(location //xplat/caffe2:metal_build_srcs)", path_prefix))

        # Robocopy has a default success code of 1 which buck treats as failure so the echo masks that problem
        cmd_exe.append("(robocopy /E /NFL /NDL /NJH /NJS {0}/{1} $OUT/{1}) || ECHO robocopy failed".format("$(location //xplat/caffe2:metal_build_srcs)", path_prefix))

    # Metal custom ops currently have to be brought into selective build because they directly reference metal ops instead of
    # going through the dispatcher. There is some weird issues with the genrule and these files locations on windows though, so
    # for now we simply skip building them for windows where they very likely aren't needed anyway.
    # Metal MaskRCNN custom op
    for full_path in METAL_MASKRCNN_SOURCE_LIST:
        path_prefix = paths.dirname(full_path)
        cmd.append("mkdir -p $OUT/{}".format(path_prefix))
        cmd.append("cp -f {0}/{1}/*.mm $OUT/{1}/ 2>/dev/null || :".format("$(location //xplat/caffe2/fb/metal:metal_maskrcnn_sources)", path_prefix))

    # Unet Metal Prepack Custom op
    unet_metal_prepack_file = "$(location //xplat/caffe2/fb/custom_ops/unet_metal_prepack:unet_metal_prepack_sources)"
    cmd.append("cp -f " + unet_metal_prepack_file + "/unet_metal_prepack.cpp" + " $OUT")
    cmd.append("cp -f " + unet_metal_prepack_file + "/unet_metal_prepack.mm" + " $OUT")

    fb_xplat_genrule(
        name = name,
        cmd = " && ".join(cmd),
        cmd_exe = "@powershell -Command " + ("; ".join(cmd_exe)),
        # due to an obscure bug certain custom ops weren't being copied correctly on windows. ARVR also sometimes builds android targets on windows,
        # so we just exclude those targets from being copied for those platforms (They end up uncompiled anyway).
        outs = select({
            "DEFAULT": get_metal_registration_files_outs(),
            "ovr_config//os:android": get_metal_registration_files_outs_windows(),
            "ovr_config//os:windows": get_metal_registration_files_outs_windows(),
        }),
        default_outs = ["."],
        apple_sdks = apple_sdks,
    )

def get_pt_operator_registry_dict(
        name,
        deps = [],
        train = False,
        labels = [],
        env = [],
        template_select = True,
        enforce_traced_op_list = False,
        pt_allow_forced_schema_registration = True,
        enable_flatbuffer = False,
        **kwargs):
    code_gen_files = pt_operator_query_codegen(
        name,
        deps = deps,
        train = train,
        enforce_traced_op_list = enforce_traced_op_list,
        pt_allow_forced_schema_registration = pt_allow_forced_schema_registration,
        compatible_with = kwargs.get("compatible_with", []),
        apple_sdks = kwargs.get("apple_sdks"),
    )

    # Extract existing linker_flags from kwargs and combine with default flags
    existing_linker_flags = kwargs.pop("linker_flags", [])
    combined_linker_flags = get_no_as_needed_linker_flag() + existing_linker_flags

    return dict(
        srcs = code_gen_files["srcs"],
        linker_flags = combined_linker_flags,
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        soname = "libtorch-code-gen.$(ext)",
        header_namespace = "ATen",
        compiler_flags = get_aten_compiler_flags(),
        exported_headers = code_gen_files["headers"],
        exported_preprocessor_flags = get_aten_preprocessor_flags() + (["-DTEMPLATE_SELECTIVE_BUILD"] if template_select else []),
        headers = kwargs.pop("headers", []),
        labels = kwargs.pop("labels", []) + [
            # This library has multiple sources with the same file name
            # and does not work with Buck filegroup used in bad practices.
            # Opt out of the bad practices check with the below label.
            "bad_practices_ignore_override",
            "pt_operator_registry",
        ],
        deps = [
            # need absolute path here
            ROOT + ":torch_mobile_core",
            ROOT + ":aten_cpu",
            ROOT + ":aten_metal_prepack_header",
            third_party("glog"),
            C10,
        ] + ([ROOT + ":torch_mobile_train"] if train else []),
        **kwargs
    )

# these targets are shared by internal and OSS BUCK
def define_buck_targets(
        aten_default_args = dict(),
        pt_xplat_cxx_library = fb_xplat_cxx_library,
        c2_fbandroid_xplat_compiler_flags = [],
        labels = []):
    if get_fbsource_cell() == "fbcode":
        return

    # @lint-ignore BUCKLINT
    fb_native.filegroup(
        name = "metal_build_srcs",
        srcs = glob(METAL_SOURCE_LIST),
        visibility = [
            "PUBLIC",
        ],
    )

    # @lint-ignore BUCKLINT
    fb_native.filegroup(
        name = "templated_selective_build_srcs",
        # NB: no glob here, there are generated targets in this list!
        srcs = glob(TEMPLATE_SOURCE_LIST) + aten_ufunc_generated_all_cpu_sources(":gen_aten[{}]"),
        visibility = [
            "PUBLIC",
        ],
    )

    fb_xplat_cxx_library(
        name = "aten_header",
        header_namespace = "",
        exported_headers = subdir_glob([
            # ATen Core
            ("aten/src", "ATen/core/**/*.h"),
            ("aten/src", "ATen/ops/*.h"),
            # ATen Base
            ("aten/src", "ATen/*.h"),
            ("aten/src", "ATen/cpu/**/*.h"),
            ("aten/src", "ATen/detail/*.h"),
            ("aten/src", "ATen/functorch/**/*.h"),
            ("aten/src", "ATen/quantized/*.h"),
            ("aten/src", "ATen/vulkan/*.h"),
            ("aten/src", "ATen/metal/*.h"),
            ("aten/src", "ATen/nnapi/*.h"),
            # ATen Native
            ("aten/src", "ATen/native/*.h"),
            ("aten/src", "ATen/native/ao_sparse/quantized/cpu/*.h"),
            ("aten/src", "ATen/native/cpu/**/*.h"),
            ("aten/src", "ATen/native/sparse/*.h"),
            ("aten/src", "ATen/native/nested/*.h"),
            ("aten/src", "ATen/native/quantized/*.h"),
            ("aten/src", "ATen/native/quantized/cpu/*.h"),
            ("aten/src", "ATen/native/transformers/*.h"),
            ("aten/src", "ATen/native/ufunc/*.h"),
            ("aten/src", "ATen/native/utils/*.h"),
            ("aten/src", "ATen/native/vulkan/ops/*.h"),
            ("aten/src", "ATen/native/xnnpack/*.h"),
            ("aten/src", "ATen/mps/*.h"),
            ("aten/src", "ATen/native/mps/*.h"),
            # Remove the following after modifying codegen for mobile.
            ("aten/src", "ATen/mkl/*.h"),
            ("aten/src", "ATen/native/mkl/*.h"),
            ("aten/src", "ATen/native/mkldnn/*.h"),
        ]),
        visibility = ["PUBLIC"],
        labels = labels,
    )

    fb_xplat_cxx_library(
        name = "aten_vulkan_header",
        header_namespace = "",
        exported_headers = subdir_glob([
            ("aten/src", "ATen/native/vulkan/*.h"),
            ("aten/src", "ATen/native/vulkan/ops/*.h"),
            ("aten/src", "ATen/vulkan/*.h"),
        ]),
        labels = labels,
        visibility = ["PUBLIC"],
    )

    fb_xplat_cxx_library(
        name = "jit_core_headers",
        header_namespace = "",
        exported_headers = subdir_glob([("", x) for x in jit_core_headers]),
        labels = labels,
    )

    fb_xplat_cxx_library(
        name = "torch_headers",
        header_namespace = "",
        exported_headers = subdir_glob(
            [
                ("torch/csrc/api/include", "torch/**/*.h"),
                ("", "torch/csrc/**/*.h"),
                ("", "torch/nativert/**/*.h"),
                ("", "torch/headeronly/**/*.h"),
                ("", "torch/script.h"),
                ("", "torch/library.h"),
                ("", "torch/custom_class.h"),
                ("", "torch/custom_class_detail.h"),
                # Add again due to namespace difference from aten_header.
                ("", "aten/src/ATen/*.h"),
                ("", "aten/src/ATen/functorch/**/*.h"),
                ("", "aten/src/ATen/quantized/*.h"),
            ],
            exclude = [
                # Don't need on mobile.
                "torch/csrc/Exceptions.h",
                "torch/csrc/python_headers.h",
                "torch/csrc/jit/serialization/mobile_bytecode_generated.h",
            ],
        ),
        labels = labels,
        visibility = ["PUBLIC"],
        deps = [
            ":generated-version-header",
        ],
    )

    fb_xplat_cxx_library(
        name = "aten_test_header",
        header_namespace = "",
        exported_headers = subdir_glob([
            ("aten/src", "ATen/test/*.h"),
        ]),
    )

    fb_xplat_cxx_library(
        name = "aten_metal_prepack_header",
        header_namespace = "",
        exported_headers = subdir_glob([
            ("aten/src", "ATen/native/metal/MetalPrepackOpContext.h"),
        ]),
        labels = labels,
        visibility = ["PUBLIC"],
    )

    fb_xplat_cxx_library(
        name = "torch_mobile_headers",
        header_namespace = "",
        compiler_flags = select({
            "DEFAULT": [],
            "ovr_config//os:macos": ["-fvisibility=default"],
        }),
        exported_headers = subdir_glob(
            [
                ("", "torch/csrc/jit/mobile/*.h"),
            ],
        ),
        labels = labels,
        visibility = ["PUBLIC"],
    )

    fb_xplat_cxx_library(
        name = "generated_aten_config_header",
        header_namespace = "ATen",
        exported_headers = {
            "Config.h": ":generate_aten_config[Config.h]",
        },
        labels = labels,
        visibility = ["PUBLIC"],
    )

    fb_xplat_cxx_library(
        name = "generated-autograd-headers",
        header_namespace = "torch/csrc/autograd/generated",
        exported_headers = {
            "Functions.h": ":gen_aten_libtorch[autograd/generated/Functions.h]",
            "VariableType.h": ":gen_aten_libtorch[autograd/generated/VariableType.h]",
            "variable_factories.h": ":gen_aten_libtorch[autograd/generated/variable_factories.h]",
            "ViewFuncs.h": ":gen_aten_libtorch[autograd/generated/ViewFuncs.h]",
            # Don't build python bindings on mobile.
            #"python_functions.h",
        },
        labels = labels,
        visibility = ["PUBLIC"],
    )

    fb_xplat_cxx_library(
        name = "generated-version-header",
        header_namespace = "torch",
        exported_headers = {
            "headeronly/version.h": ":generate-version-header[version.h]",
            "version.h": "torch/csrc/api/include/torch/version.h"
        },
        labels = labels,
    )

    # @lint-ignore BUCKLINT
    fb_native.genrule(
        name = "generate-version-header",
        srcs = [
            "torch/headeronly/version.h.in",
            "version.txt",
        ],
        cmd = "mkdir -p $OUT/torch/headeronly && $(exe {}tools:gen-version-header) ".format(ROOT_PATH) + " ".join([
            "--template-path",
            "torch/headeronly/version.h.in",
            "--version-path",
            "version.txt",
            "--output-path",
            "$OUT/torch/headeronly/version.h",
        ]),
        cmd_exe = "md $OUT\\torch\\headeronly 2>nul & $(exe {}tools:gen-version-header) ".format(ROOT_PATH) + " ".join([
            "--template-path",
            "torch/headeronly/version.h.in",
            "--version-path",
            "version.txt",
            "--output-path",
            "$OUT\\torch\\headeronly\\version.h",
        ]),
        outs = {
            "version.h": ["torch/headeronly/version.h"],
        },
        default_outs = ["."],
    )

    # @lint-ignore BUCKLINT
    fb_native.filegroup(
        name = "aten_src_path",
        srcs = [
            "aten/src/ATen/native/native_functions.yaml",
            "aten/src/ATen/native/tags.yaml",
        ] + glob(["aten/src/ATen/templates/*"]),
        visibility = [
            "PUBLIC",
        ],
    )

    fb_xplat_cxx_library(
        name = "common_core",
        srcs = [
            "caffe2/core/common.cc",
        ],
        apple_sdks = (IOS, MACOSX),
        compiler_flags = get_pt_compiler_flags(),
        labels = labels,
        # @lint-ignore BUCKLINT link_whole
        link_whole = True,
        visibility = ["PUBLIC"],
        windows_preferred_linkage = "static" if is_arvr_mode() else None,
        deps = [
            ":caffe2_headers",
            C10,
        ],
    )

    # TODO: Enable support for KleidiAI bazel build
    # @lint-ignore BUCKLINT
    fb_native.genrule(
        name = "generate_aten_config",
        srcs = [
            "aten/src/ATen/Config.h.in",
        ],
        cmd = "$(exe {}tools:substitute) ".format(ROOT_PATH) + " ".join([
            "--install_dir",
            "$OUT",
            "--input-file",
            "aten/src/ATen/Config.h.in",
            "--output-file",
            "Config.h",
            "--replace",
            "@AT_MKLDNN_ENABLED@",
            "ATEN_MKLDNN_ENABLED_FBXPLAT",
            "--replace",
            "@AT_MKLDNN_ACL_ENABLED@",
            "ATEN_MKLDNN_ACL_ENABLED_FBXPLAT",
            "--replace",
            "@AT_MKL_ENABLED@",
            "ATEN_MKL_ENABLED_FBXPLAT",
            "--replace",
            "@AT_MKL_SEQUENTIAL@",
            "ATEN_MKL_SEQUENTIAL_FBXPLAT",
            "--replace",
            "@AT_POCKETFFT_ENABLED@",
            "1",
            "--replace",
            "@AT_NNPACK_ENABLED@",
            "ATEN_NNPACK_ENABLED_FBXPLAT",
            "--replace",
            "@CAFFE2_STATIC_LINK_CUDA_INT@",
            "CAFFE2_STATIC_LINK_CUDA_FBXPLAT",
            "--replace",
            "@AT_BUILD_WITH_BLAS@",
            "USE_BLAS_FBXPLAT",
            "--replace",
            "@AT_PARALLEL_OPENMP@",
            "AT_PARALLEL_OPENMP_FBXPLAT",
            "--replace",
            "@AT_PARALLEL_NATIVE@",
            "AT_PARALLEL_NATIVE_FBXPLAT",
            "--replace",
            "@AT_BUILD_WITH_LAPACK@",
            "USE_LAPACK_FBXPLAT",
            "--replace",
            "@AT_BLAS_F2C@",
            "AT_BLAS_F2C_FBXPLAT",
            "--replace",
            "@AT_BLAS_USE_CBLAS_DOT@",
            "AT_BLAS_USE_CBLAS_DOT_FBXPLAT",
            "--replace",
            "@AT_KLEIDIAI_ENABLED@",
            "0",
            "--replace",
            "@AT_USE_EIGEN_SPARSE@",
            "0",
        ]),
        outs = {
            "Config.h": ["Config.h"],
        },
        default_outs = ["."],
    )

    gen_aten_files(
        name = "gen_aten",
        extra_flags = get_aten_codegen_extra_params(USED_PT_BACKENDS),
        visibility = ["PUBLIC"],
    )

    gen_aten_libtorch_files(name = "gen_aten_libtorch")

    gen_aten_libtorch_files(
        name = "gen_aten_libtorch_lite",
        extra_params = get_jit_codegen_params(),
    )

    fb_xplat_cxx_library(
        name = "generated_aten_headers_cpu",
        header_namespace = "ATen",
        exported_headers = get_aten_static_dispatch_backend_headers({
            "CPUFunctions.h": ":gen_aten[CPUFunctions.h]",
            "CPUFunctions_inl.h": ":gen_aten[CPUFunctions_inl.h]",
            "CompositeExplicitAutogradFunctions.h": ":gen_aten[CompositeExplicitAutogradFunctions.h]",
            "CompositeExplicitAutogradFunctions_inl.h": ":gen_aten[CompositeExplicitAutogradFunctions_inl.h]",
            "CompositeExplicitAutogradNonFunctionalFunctions.h": ":gen_aten[CompositeExplicitAutogradNonFunctionalFunctions.h]",
            "CompositeExplicitAutogradNonFunctionalFunctions_inl.h": ":gen_aten[CompositeExplicitAutogradNonFunctionalFunctions_inl.h]",
            "CompositeImplicitAutogradFunctions.h": ":gen_aten[CompositeImplicitAutogradFunctions.h]",
            "CompositeImplicitAutogradFunctions_inl.h": ":gen_aten[CompositeImplicitAutogradFunctions_inl.h]",
            "CompositeImplicitAutogradNestedTensorFunctions.h": ":gen_aten[CompositeImplicitAutogradNestedTensorFunctions.h]",
            "CompositeImplicitAutogradNestedTensorFunctions_inl.h": ":gen_aten[CompositeImplicitAutogradNestedTensorFunctions_inl.h]",
            "FunctionalInverses.h": ":gen_aten[FunctionalInverses.h]",
            "Functions.h": ":gen_aten[Functions.h]",
            "MethodOperators.h": ":gen_aten[MethodOperators.h]",
            "NativeFunctions.h": ":gen_aten[NativeFunctions.h]",
            "NativeMetaFunctions.h": ":gen_aten[NativeMetaFunctions.h]",
            "Operators.h": ":gen_aten[Operators.h]",
            "RedispatchFunctions.h": ":gen_aten[RedispatchFunctions.h]",
            "ViewMetaClasses.h": ":gen_aten[ViewMetaClasses.h]",
            "core/TensorBody.h": ":gen_aten[core/TensorBody.h]",
            "core/aten_interned_strings.h": ":gen_aten[core/aten_interned_strings.h]",
            "core/enum_tag.h": ":gen_aten[core/enum_tag.h]",
        }),
        labels = labels,
    )

    fb_xplat_cxx_library(
        name = "torch_mobile_observer",
        srcs = [
            "torch/csrc/jit/mobile/observer.cpp",
        ] + ([] if IS_OSS else ["torch/fb/observers/MobileObserverUtil.cpp"]),
        compiler_flags = ["-fexceptions"] + select({
            "DEFAULT": [],
            "ovr_config//os:macos": ["-fvisibility=default"],
        }),
        header_namespace = "",
        exported_headers = subdir_glob(
            [
                ("", "torch/csrc/jit/mobile/observer.h"),
            ] + ([] if IS_OSS else [
                ("", "torch/fb/observers/ObserverUtil.h"),
                ("", "torch/fb/observers/MobileObserverUtil.h"),
            ]),
        ),
        fbobjc_compiler_flags = [
            "-Wno-missing-prototypes",
        ],
        labels = labels,
        visibility = ["PUBLIC"],
        deps = [
            C10,
        ],
    )

    # Base library shared by lite-interpreter and full-jit.
    pt_xplat_cxx_library(
        name = "torch_common",
        srcs = core_sources_common,
        compiler_flags = get_pt_compiler_flags(),
        exported_preprocessor_flags = get_pt_preprocessor_flags(),
        # @lint-ignore BUCKLINT link_whole
        link
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

- **Serialization**: Uses pickle - be cautious with untrusted data

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

- **File Documentation**: `buckbuild.bzl_docs.md_docs.md`
- **Keyword Index**: `buckbuild.bzl_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
