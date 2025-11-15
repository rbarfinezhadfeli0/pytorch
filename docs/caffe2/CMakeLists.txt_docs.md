# Documentation: `caffe2/CMakeLists.txt`

## File Metadata

- **Path**: `caffe2/CMakeLists.txt`
- **Size**: 83,707 bytes (81.75 KB)
- **Type**: Source File (.txt)
- **Extension**: `.txt`

## File Purpose

This is a source file (.txt) that is part of the PyTorch project.

## Original Source

```
# ---[ Generate and install header and cpp files
include(../cmake/Codegen.cmake)

# ---[ Vulkan code gen
if(USE_VULKAN)
  include(../cmake/VulkanCodegen.cmake)
endif()

# Debug messages - if you want to get a list of source files and examine
# target information, enable the following by -DPRINT_CMAKE_DEBUG_INFO=ON.
set(PRINT_CMAKE_DEBUG_INFO FALSE CACHE BOOL "print cmake debug information")
if(PRINT_CMAKE_DEBUG_INFO)
  include(../cmake/DebugHelper.cmake)
endif()

# ATen parallelism settings
#  OMP - OpenMP for intra-op, native thread pool for inter-op parallelism
#  NATIVE - using native thread pool for intra- and inter-op parallelism
if(INTERN_BUILD_MOBILE)
  set(ATEN_THREADING "NATIVE" CACHE STRING "ATen parallel backend")
else()
  if(USE_OPENMP)
    set(ATEN_THREADING "OMP" CACHE STRING "ATen parallel backend")
  else()
    set(ATEN_THREADING "NATIVE" CACHE STRING "ATen parallel backend")
  endif()
endif()

set(AT_PARALLEL_OPENMP 0)
set(AT_PARALLEL_NATIVE 0)

message(STATUS "Using ATen parallel backend: ${ATEN_THREADING}")
if("${ATEN_THREADING}" STREQUAL "OMP")
  set(AT_PARALLEL_OPENMP 1)
elseif("${ATEN_THREADING}" STREQUAL "NATIVE")
  set(AT_PARALLEL_NATIVE 1)
else()
  message(FATAL_ERROR "Unknown ATen parallel backend: ${ATEN_THREADING}")
endif()

# ---[ Declare source file lists

# ---[ ATen build
if(INTERN_BUILD_ATEN_OPS)
  set(__torch_CMAKE_POSITION_INDEPENDENT_CODE ${CMAKE_POSITION_INDEPENDENT_CODE})
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  add_subdirectory(../aten aten)
  set(CMAKE_POSITION_INDEPENDENT_CODE ${__torch_CMAKE_POSITION_INDEPENDENT_CODE})

  # Generate the headers wrapped by our operator
  file(GLOB_RECURSE torchgen_python "${PROJECT_SOURCE_DIR}/torchgen/*.py")


  # Add source, includes, and libs to lists
  list(APPEND Caffe2_CPU_SRCS ${ATen_CPU_SRCS})
  list(APPEND Caffe2_GPU_SRCS ${ATen_CUDA_CPP_SRCS})
  list(APPEND Caffe2_XPU_SRCS ${ATen_XPU_SRCS})
  list(APPEND Caffe2_XPU_INCLUDE ${ATen_XPU_INCLUDE})
  list(APPEND Caffe2_XPU_DEPENDENCY_LIBS ${ATen_XPU_DEPENDENCY_LIBS})
  list(APPEND Caffe2_GPU_SRCS_W_SORT_BY_KEY ${ATen_CUDA_SRCS_W_SORT_BY_KEY})
  list(APPEND Caffe2_GPU_CU_SRCS ${ATen_CUDA_CU_SRCS})
  list(APPEND Caffe2_GPU_CU_SRCS_W_SORT_BY_KEY ${ATen_CUDA_CU_SRCS_W_SORT_BY_KEY})
  list(APPEND Caffe2_HIP_SRCS ${ATen_HIP_SRCS})
  list(APPEND Caffe2_MPS_SRCS ${ATen_MPS_SRCS})
  list(APPEND Caffe2_XPU_SRCS ${ATen_XPU_SRCS})
  list(APPEND Caffe2_HIP_SRCS ${ATen_HIP_SRCS_W_SORT_BY_KEY})
  list(APPEND Caffe2_CPU_TEST_SRCS ${ATen_CPU_TEST_SRCS})
  list(APPEND Caffe2_MPS_TEST_SRCS ${ATen_MPS_TEST_SRCS})
  list(APPEND Caffe2_GPU_TEST_SRCS ${ATen_CUDA_TEST_SRCS})
  list(APPEND Caffe2_HIP_TEST_SRCS ${ATen_HIP_TEST_SRCS})
  list(APPEND Caffe2_XPU_TEST_SRCS ${ATen_XPU_TEST_SRCS})
  list(APPEND Caffe2_CPU_TEST_SRCS ${ATen_CORE_TEST_SRCS})
  list(APPEND Caffe2_VULKAN_TEST_SRCS ${ATen_VULKAN_TEST_SRCS})
  list(APPEND Caffe2_CPU_INCLUDE ${ATen_CPU_INCLUDE})
  list(APPEND Caffe2_GPU_INCLUDE ${ATen_CUDA_INCLUDE})
  list(APPEND Caffe2_HIP_INCLUDE ${ATen_HIP_INCLUDE})
  list(APPEND Caffe2_XPU_INCLUDE ${ATen_XPU_INCLUDE})
  list(APPEND Caffe2_VULKAN_INCLUDE ${ATen_VULKAN_INCLUDE})
  list(APPEND Caffe2_DEPENDENCY_LIBS ${ATen_CPU_DEPENDENCY_LIBS})
  list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS ${ATen_CUDA_DEPENDENCY_LIBS})
  list(APPEND Caffe2_HIP_DEPENDENCY_LIBS ${ATen_HIP_DEPENDENCY_LIBS})
  list(APPEND Caffe2_DEPENDENCY_INCLUDE ${ATen_THIRD_PARTY_INCLUDE})
  set(Caffe2_CUDA_DEPENDENCY_LIBS ${Caffe2_CUDA_DEPENDENCY_LIBS} PARENT_SCOPE)
endif()

# ---[ Caffe2 build
# Note: the folders that are being commented out have not been properly
# addressed yet.

if(NOT MSVC AND USE_XNNPACK)
  if(NOT TARGET fxdiv)
    set(FXDIV_BUILD_TESTS OFF CACHE BOOL "")
    set(FXDIV_BUILD_BENCHMARKS OFF CACHE BOOL "")
    add_subdirectory(
      "${FXDIV_SOURCE_DIR}"
      "${CMAKE_BINARY_DIR}/FXdiv")
  endif()
endif()

add_subdirectory(core)
add_subdirectory(serialize)
add_subdirectory(utils)
if(NOT USE_FBGEMM)
  add_subdirectory(perfkernels)
endif()

# Advanced: if we have allow list specified, we will do intersections for all
# main lib srcs.
if(CAFFE2_ALLOWLISTED_FILES)
  caffe2_do_allowlist(Caffe2_CPU_SRCS CAFFE2_ALLOWLISTED_FILES)
  caffe2_do_allowlist(Caffe2_GPU_SRCS CAFFE2_ALLOWLISTED_FILES)
  caffe2_do_allowlist(Caffe2_XPU_SRCS CAFFE2_ALLOWLISTED_FILES)
  caffe2_do_allowlist(Caffe2_GPU_SRCS_W_SORT_BY_KEY CAFFE2_ALLOWLISTED_FILES)
  caffe2_do_allowlist(Caffe2_GPU_CU_SRCS CAFFE2_ALLOWLISTED_FILES)
  caffe2_do_allowlist(Caffe2_GPU_CU_SRCS_W_SORT_BY_KEY CAFFE2_ALLOWLISTED_FILES)
  caffe2_do_allowlist(Caffe2_HIP_SRCS CAFFE2_ALLOWLISTED_FILES)
endif()

if(PRINT_CMAKE_DEBUG_INFO)
  message(STATUS "CPU sources: ")
  foreach(tmp ${Caffe2_CPU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU sources: (for torch_cuda_cpp)")
  foreach(tmp ${Caffe2_GPU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU sources: (for torch_cuda_cu)")
  foreach(tmp ${Caffe2_GPU_CU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "torch_cuda_cu GPU sources (w/ sort by key): ")
  foreach(tmp ${Caffe2_GPU_CU_SRCS_W_SORT_BY_KEY})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "torch_cuda_cpp GPU sources (w/ sort by key): ")
  foreach(tmp ${Caffe2_GPU_SRCS_W_SORT_BY_KEY})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "CPU include: ")
  foreach(tmp ${Caffe2_CPU_INCLUDE})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU include: ")
  foreach(tmp ${Caffe2_GPU_INCLUDE})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "CPU test sources: ")
  foreach(tmp ${Caffe2_CPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU test sources: ")
  foreach(tmp ${Caffe2_GPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "HIP sources: ")
  foreach(tmp ${Caffe2_HIP_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "MPS sources: ")
  foreach(tmp ${Caffe2_MPS_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "XPU sources: ")
  foreach(tmp ${Caffe2_XPU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "HIP test sources: ")
  foreach(tmp ${Caffe2_HIP_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "ATen CPU test sources: ")
  foreach(tmp ${ATen_CPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "ATen MPS test sources: ")
  foreach(tmp ${ATen_MPS_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "ATen CUDA test sources: ")
  foreach(tmp ${ATen_CUDA_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "ATen HIP test sources: ")
  foreach(tmp ${ATen_HIP_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "ATen XPU test sources: ")
  foreach(tmp ${ATen_XPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "ATen Vulkan test sources: ")
  foreach(tmp ${ATen_VULKAN_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

endif()

# ==========================================================
# formerly-libtorch
# ==========================================================

set(TORCH_SRC_DIR "${PROJECT_SOURCE_DIR}/torch")
set(TORCH_ROOT "${PROJECT_SOURCE_DIR}")

if(NOT TORCH_INSTALL_BIN_DIR)
  set(TORCH_INSTALL_BIN_DIR bin)
endif()

if(NOT TORCH_INSTALL_INCLUDE_DIR)
  set(TORCH_INSTALL_INCLUDE_DIR include)
endif()

if(NOT TORCH_INSTALL_LIB_DIR)
  set(TORCH_INSTALL_LIB_DIR lib)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# Generate files
set(TOOLS_PATH "${TORCH_ROOT}/tools")

configure_file("${TORCH_SRC_DIR}/_utils_internal.py"
  "${TOOLS_PATH}/shared/_utils_internal.py"
  COPYONLY)

# Generate header with version info
configure_file("${TORCH_SRC_DIR}/headeronly/version.h.in"
  "${TORCH_SRC_DIR}/headeronly/version.h"
  @ONLY)

set(GENERATED_CXX_TORCH
  "${TORCH_SRC_DIR}/csrc/autograd/generated/Functions.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/ViewFuncs.cpp"
  )

if(NOT INTERN_DISABLE_AUTOGRAD AND NOT BUILD_LITE_INTERPRETER)
  list(APPEND GENERATED_CXX_TORCH
    "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_0.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_1.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_2.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_3.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType_4.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/TraceType_0.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/TraceType_1.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/TraceType_2.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/TraceType_3.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/TraceType_4.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/ADInplaceOrViewType_0.cpp"
    "${TORCH_SRC_DIR}/csrc/autograd/generated/ADInplaceOrViewType_1.cpp"
    "${TORCH_SRC_DIR}/csrc/inductor/aoti_torch/generated/c_shim_cpu.cpp"
    "${TORCH_SRC_DIR}/csrc/inductor/aoti_torch/generated/c_shim_aten.cpp"
  )
  if(BUILD_LAZY_TS_BACKEND)
    list(APPEND GENERATED_CXX_TORCH
      "${TORCH_SRC_DIR}/csrc/lazy/generated/LazyNativeFunctions.cpp"
      "${TORCH_SRC_DIR}/csrc/lazy/generated/RegisterAutogradLazy.cpp"
      "${TORCH_SRC_DIR}/csrc/lazy/generated/RegisterLazy.cpp"
    )
  endif()
  if(USE_MPS)
    list(APPEND GENERATED_CXX_TORCH
      "${TORCH_SRC_DIR}/csrc/inductor/aoti_torch/generated/c_shim_mps.cpp"
    )
  endif()
endif()

set(GENERATED_H_TORCH
  "${TORCH_SRC_DIR}/csrc/autograd/generated/Functions.h"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/variable_factories.h"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/ViewFuncs.h"
  )

if(NOT INTERN_DISABLE_AUTOGRAD)
  list(APPEND GENERATED_H_TORCH
    "${TORCH_SRC_DIR}/csrc/autograd/generated/VariableType.h"
    "${TORCH_SRC_DIR}/csrc/lazy/generated/LazyIr.h"
    "${TORCH_SRC_DIR}/csrc/lazy/generated/LazyNonNativeIr.h"
    "${TORCH_SRC_DIR}/csrc/lazy/generated/LazyNativeFunctions.h"
  )
endif()

set(GENERATED_CXX_PYTHON
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_functions_0.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_functions_1.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_functions_2.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_functions_3.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_functions_4.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_variable_methods.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_torch_functions_0.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_torch_functions_1.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_torch_functions_2.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_nn_functions.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_fft_functions.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_linalg_functions.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_nested_functions.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_sparse_functions.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_special_functions.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_return_types.cpp"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_enum_tag.cpp"
  "${TORCH_SRC_DIR}/csrc/functionalization/generated/ViewMetaClassesPythonBinding.cpp"
  )

set(GENERATED_H_PYTHON
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_functions.h"
  "${TORCH_SRC_DIR}/csrc/autograd/generated/python_return_types.h"
  )

set(GENERATED_TESTING_PYTHON
  "${TORCH_SRC_DIR}/testing/_internal/generated/annotated_fn_args.py"
  )

set(GENERATED_CXX_TORCH_CUDA
  "${TORCH_SRC_DIR}/csrc/inductor/aoti_torch/generated/c_shim_cuda.cpp"
  )

set(GENERATED_CXX_TORCH_XPU
  "${TORCH_SRC_DIR}/csrc/inductor/aoti_torch/generated/c_shim_xpu.cpp"
  )

set(TORCH_GENERATED_CODE
  ${GENERATED_CXX_TORCH}
  ${GENERATED_H_TORCH}
  ${GENERATED_CXX_PYTHON}
  ${GENERATED_H_PYTHON}
  ${GENERATED_TESTING_PYTHON}
  ${GENERATED_CXX_TORCH_CUDA}
  )

if(USE_XPU)
  list(APPEND TORCH_GENERATED_CODE ${GENERATED_CXX_TORCH_XPU})
endif()

set(GEN_PER_OPERATOR_FLAG)
if(USE_PER_OPERATOR_HEADERS)
  list(APPEND GEN_PER_OPERATOR_FLAG "--per_operator_headers")
endif()

file(GLOB_RECURSE autograd_python "${TOOLS_PATH}/autograd/*.py")
file(GLOB_RECURSE autograd_yaml "${TOOLS_PATH}/autograd/*.yaml")
file(GLOB_RECURSE autograd_templates "${TOOLS_PATH}/autograd/templates/*")
add_custom_command(
  OUTPUT
  ${TORCH_GENERATED_CODE}
  COMMAND
  Python::Interpreter tools/setup_helpers/generate_code.py
    --native-functions-path "aten/src/ATen/native/native_functions.yaml"
    --tags-path "aten/src/ATen/native/tags.yaml"
    $<$<BOOL:${INTERN_DISABLE_AUTOGRAD}>:--disable-autograd>
    $<$<BOOL:${SELECTED_OP_LIST}>:--selected-op-list-path="${SELECTED_OP_LIST}">
    --force_schema_registration
    --gen_lazy_ts_backend
    ${GEN_PER_OPERATOR_FLAG}
  DEPENDS
    "${TORCH_ROOT}/aten/src/ATen/native/native_functions.yaml"
    "${TORCH_ROOT}/aten/src/ATen/native/tags.yaml"
    "${TORCH_ROOT}/aten/src/ATen/native/ts_native_functions.yaml"
    "${TORCH_ROOT}/torch/csrc/lazy/core/shape_inference.h"
    "${TORCH_ROOT}/torch/csrc/lazy/ts_backend/ts_native_functions.cpp"
    "${TORCH_ROOT}/aten/src/ATen/templates/DispatchKeyNativeFunctions.h"
    "${TORCH_ROOT}/aten/src/ATen/templates/DispatchKeyNativeFunctions.cpp"
    "${TORCH_ROOT}/aten/src/ATen/templates/LazyIr.h"
    "${TORCH_ROOT}/aten/src/ATen/templates/LazyNonNativeIr.h"
    "${TORCH_ROOT}/aten/src/ATen/templates/RegisterDispatchKey.cpp"
    "${TORCH_ROOT}/aten/src/ATen/templates/ViewMetaClasses.h"
    "${TORCH_ROOT}/aten/src/ATen/templates/ViewMetaClasses.cpp"
    "${TORCH_ROOT}/aten/src/ATen/templates/ViewMetaClassesPythonBinding.cpp"
    ${autograd_python}
    ${autograd_yaml}
    ${autograd_templates}
    ${torchgen_python}
  WORKING_DIRECTORY "${TORCH_ROOT}")


# Required workaround for libtorch_python.so build
# see https://samthursfield.wordpress.com/2015/11/21/cmake-dependencies-between-targets-and-files-and-custom-commands/#custom-commands-in-different-directories
add_custom_target(
  generate-torch-sources
  DEPENDS ${TORCH_GENERATED_CODE}
  )

set(TORCH_SRCS ${GENERATED_CXX_TORCH})
list(APPEND TORCH_SRCS ${GENERATED_H_TORCH})
list(APPEND LIBTORCH_CMAKE_SRCS "")

list(APPEND LITE_EAGER_SYMOBLICATION_SRCS "")
if(USE_SOURCE_DEBUG_ON_MOBILE)
  append_filelist("libtorch_lite_eager_symbolication" LITE_EAGER_SYMOBLICATION_SRCS)
  # For source debug on lite interpreter, we have to add dependency on pickling
  # but references to read/writeArchiveAndTensor is not built for mobile
  # so this condition specifically says we are building for source debug
  # on mobile.
  if(BUILD_LITE_INTERPRETER)
    set_source_files_properties(${TORCH_SRC_DIR}/csrc/jit/serialization/pickle.cpp PROPERTIES COMPILE_FLAGS "-DC10_MOBILE -DFEATURE_TORCH_MOBILE")
  endif()
endif()

list(APPEND LITE_PROFILER_SRCS "")
if(USE_LITE_INTERPRETER_PROFILER)
  append_filelist("libtorch_edge_profiler_sources " LITE_PROFILER_SRCS)
endif()

# Switch between the full jit interpreter and lite interpreter
if(BUILD_LITE_INTERPRETER)
  append_filelist("libtorch_lite_cmake_sources" LIBTORCH_CMAKE_SRCS)
  list(APPEND LIBTORCH_CMAKE_SRCS ${LITE_EAGER_SYMOBLICATION_SRCS})
  list(APPEND LIBTORCH_CMAKE_SRCS ${LITE_PROFILER_SRCS})
  if(USE_LITE_AOTI)
    append_filelist("inductor_core_resources" LIBTORCH_CMAKE_SRCS)
  endif()
  set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
else()
  append_filelist("libtorch_cmake_sources" LIBTORCH_CMAKE_SRCS)
  list(APPEND LIBTORCH_CMAKE_SRCS ${LITE_EAGER_SYMOBLICATION_SRCS})
  if(BUILD_LAZY_TS_BACKEND)
    append_filelist("lazy_tensor_ts_sources" LIBTORCH_CMAKE_SRCS)
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # TODO: Delete this when https://github.com/pytorch/pytorch/issues/35026 is fixed
    set_source_files_properties(../torch/csrc/autograd/record_function_ops.cpp PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
  endif()
endif()
list(APPEND TORCH_SRCS ${LIBTORCH_CMAKE_SRCS})

if(PRINT_CMAKE_DEBUG_INFO)
  message(STATUS "Interpreter sources: ")
  foreach(tmp ${LIBTORCH_CMAKE_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()
endif()

# Mobile backend delegate srcs
if(INTERN_BUILD_MOBILE)
  set(DELEGATE_SRCS
    ${TORCH_SRC_DIR}/csrc/jit/backends/backend_debug_info.cpp
    ${TORCH_SRC_DIR}/csrc/jit/backends/backend_interface.cpp
  )
  list(APPEND TORCH_SRCS ${DELEGATE_SRCS})
  if(IOS AND USE_COREML_DELEGATE)
    set(COREML_DELEGATE_SRCS
      ${TORCH_SRC_DIR}/csrc/jit/backends/coreml/cpp/context.cpp
      ${TORCH_SRC_DIR}/csrc/jit/backends/coreml/objc/PTMCoreMLBackend.mm
      ${TORCH_SRC_DIR}/csrc/jit/backends/coreml/objc/PTMCoreMLExecutor.mm
      ${TORCH_SRC_DIR}/csrc/jit/backends/coreml/objc/PTMCoreMLCompiler.mm
      ${TORCH_SRC_DIR}/csrc/jit/backends/coreml/objc/PTMCoreMLFeatureProvider.mm
    )
    set_source_files_properties(${TORCH_SRC_DIR}/csrc/jit/backends/coreml/objc/PTMCoreMLBackend.mm PROPERTIES COMPILE_FLAGS "-fno-objc-arc")
    include_directories(${TORCH_ROOT}/third_party/nlohmann/single_include)
    list(APPEND TORCH_SRCS ${COREML_DELEGATE_SRCS})
  endif()
endif()

# Required workaround for LLVM 9 includes.
if(NOT MSVC)
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/jit/tensorexpr/llvm_jit.cpp PROPERTIES COMPILE_FLAGS -Wno-noexcept-type)
endif()
# Disable certain warnings for GCC-9.X
if(CMAKE_COMPILER_IS_GNUCXX)
  # See https://github.com/pytorch/pytorch/issues/38856
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/jit/tensorexpr/llvm_jit.cpp PROPERTIES COMPILE_FLAGS "-Wno-redundant-move -Wno-noexcept-type")
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/jit/tensorexpr/llvm_codegen.cpp PROPERTIES COMPILE_FLAGS "-Wno-init-list-lifetime")
endif()

# Enable conditional FP16 arithmetic intrinsics
if(CPU_AARCH64 AND LINUX)
set_source_files_properties(${TORCH_ROOT}/aten/src/ATen/native/BlasKernel.cpp PROPERTIES COMPILE_FLAGS "-march=armv8.2-a+fp16")
endif()


if(NOT INTERN_DISABLE_MOBILE_INTERP)
  set(MOBILE_SRCS
     ${TORCH_SRC_DIR}/csrc/jit/mobile/function.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/import.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/import_data.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/interpreter.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/compatibility/model_compatibility.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/module.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/flatbuffer_loader.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/observer.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/parse_bytecode.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/parse_operators.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/quantization.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/train/export_data.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/train/optim/sgd.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/train/random.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/train/sequential.cpp
     ${TORCH_SRC_DIR}/csrc/jit/mobile/upgrader_mobile.cpp
     ${TORCH_SRC_DIR}/csrc/jit/serialization/flatbuffer_serializer.cpp
     )
  list(APPEND TORCH_SRCS ${MOBILE_SRCS})
  list(APPEND TORCH_SRCS ${LITE_EAGER_SYMOBLICATION_SRCS})
endif()

# This one needs to be unconditionally added as Functions.cpp is also unconditionally added
list(APPEND TORCH_SRCS
  ${TORCH_SRC_DIR}/csrc/autograd/FunctionsManual.cpp
  ${TORCH_SRC_DIR}/csrc/utils/out_types.cpp
)

if(NOT INTERN_DISABLE_AUTOGRAD AND NOT BUILD_LITE_INTERPRETER)
  list(APPEND TORCH_SRCS
    ${TORCH_SRC_DIR}/csrc/autograd/TraceTypeManual.cpp
    ${TORCH_SRC_DIR}/csrc/autograd/VariableTypeManual.cpp
  )
endif()

if(${USE_ITT})
  list(APPEND TORCH_SRCS
    ${TORCH_SRC_DIR}/csrc/itt_wrapper.cpp
    ${TORCH_SRC_DIR}/csrc/profiler/stubs/itt.cpp
  )
endif()

if(NOT INTERN_BUILD_MOBILE AND NOT BUILD_LITE_INTERPRETER)
  list(APPEND TORCH_SRCS
    ${TORCH_SRC_DIR}/csrc/api/src/jit.cpp
    ${TORCH_SRC_DIR}/csrc/jit/mobile/compatibility/backport.cpp
    ${TORCH_SRC_DIR}/csrc/jit/mobile/compatibility/backport_manager.cpp
    ${TORCH_SRC_DIR}/csrc/jit/serialization/onnx.cpp
    ${TORCH_SRC_DIR}/csrc/jit/serialization/export.cpp
    ${TORCH_SRC_DIR}/csrc/jit/serialization/export_bytecode.cpp
    ${TORCH_SRC_DIR}/csrc/jit/serialization/export_module.cpp
    ${TORCH_SRC_DIR}/csrc/jit/serialization/flatbuffer_serializer.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/fuser/cpu/fused_kernel.cpp
    ${TORCH_SRC_DIR}/csrc/jit/api/module_save.cpp
    ${TORCH_SRC_DIR}/csrc/utils/byte_order.cpp
  )

  if(USE_DISTRIBUTED)
    append_filelist("libtorch_distributed_base_sources" TORCH_SRCS)
    if(NOT WIN32)
      append_filelist("libtorch_distributed_extra_sources" TORCH_SRCS)
    endif()
  endif()
endif()

if(USE_CUDA OR USE_ROCM)
  append_filelist("libtorch_cuda_core_sources" Caffe2_GPU_HIP_JIT_FUSERS_SRCS)
endif()

# NativeRT is disabled
# if(USE_CUDA)
#   append_filelist("libtorch_nativert_cuda_sources" Caffe2_GPU_SRCS)
# endif()
# if(USE_ROCM)
#   append_filelist("libtorch_nativert_cuda_sources" Caffe2_HIP_SRCS)
# endif()

if(USE_CUDA)
  list(APPEND Caffe2_GPU_CU_SRCS ${Caffe2_GPU_HIP_JIT_FUSERS_SRCS})
  add_library(caffe2_nvrtc SHARED ${ATen_NVRTC_STUB_SRCS})
  if(MSVC)
    # Delay load nvcuda.dll so we can import torch compiled with cuda on a CPU-only machine
    set(DELAY_LOAD_FLAGS "-DELAYLOAD:nvcuda.dll;delayimp.lib")
  else()
    set(DELAY_LOAD_FLAGS "")
  endif()

  target_link_libraries(caffe2_nvrtc PRIVATE caffe2::nvrtc ${DELAY_LOAD_FLAGS})
  install(TARGETS caffe2_nvrtc DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  if(USE_NCCL)
    list(APPEND Caffe2_GPU_SRCS
      ${TORCH_SRC_DIR}/csrc/cuda/nccl.cpp)
  endif()
  if(USE_DISTRIBUTED)
    append_filelist("libtorch_cuda_distributed_base_sources" Caffe2_GPU_SRCS)
    if(NOT WIN32)
      append_filelist("libtorch_cuda_distributed_extra_sources" Caffe2_GPU_SRCS)
      set_source_files_properties(
        ${TORCH_SRC_DIR}/csrc/distributed/c10d/ProcessGroupNCCL.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/c10d/cuda/utils.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/c10d/intra_node_comm.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/c10d/symm_mem/CudaDMAConnectivity.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu
        ${TORCH_SRC_DIR}/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryOps.cu
        ${TORCH_SRC_DIR}/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.cpp
        ${TORCH_SRC_DIR}/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.cu
        ${TORCH_SRC_DIR}/csrc/distributed/c10d/symm_mem/cuda_mem_pool.cpp
        PROPERTIES COMPILE_FLAGS "-DPYTORCH_C10_DRIVER_API_SUPPORTED=1"
      )
    endif()

    set(ASYNC_MM_FILE "${TORCH_SRC_DIR}/csrc/distributed/c10d/cuda/AsyncMM.cu")
    # Disable the warning to make cutlass warp-specialized cooperative kernel build for gcc-9
    if(CMAKE_COMPILER_IS_GNUCXX)
      set_source_files_properties(${ASYNC_MM_FILE} PROPERTIES COMPILE_FLAGS "-Wno-unused-but-set-variable")
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0 AND CUDA_NVCC_FLAGS MATCHES ".*compute_90.*")
      set_source_files_properties(${ASYNC_MM_FILE} PROPERTIES COMPILE_FLAGS "-gencode arch=compute_90a,code=sm_90a")
    endif()
  endif()
  if(NOT WIN32)
    set_source_files_properties(
      ${TORCH_ROOT}/aten/src/ATen/cuda/CUDAGreenContext.cpp
      PROPERTIES COMPILE_FLAGS "-DPYTORCH_C10_DRIVER_API_SUPPORTED=1"
    )
  endif()
  set_source_files_properties(
    ${TORCH_ROOT}/aten/src/ATen/cuda/detail/LazyNVRTC.cpp
    PROPERTIES COMPILE_DEFINITIONS "NVRTC_SHORTHASH=${CUDA_NVRTC_SHORTHASH}"
  )
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/jit/passes/frozen_conv_add_relu_fusion.cpp PROPERTIES COMPILE_FLAGS "-DUSE_CUDA=1")
  set_source_files_properties(${TORCH_SRC_DIR}/csrc/jit/codegen/cuda/interface.cpp PROPERTIES COMPILE_FLAGS "-DUSE_CUDA=1")
endif()

if(BUILD_ONEDNN_GRAPH)
  list(APPEND Caffe2_CPU_SRCS
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/LlgaTensorImpl.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/graph_fuser.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/graph_rewriter.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/graph_helper.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/register_interface.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/decompose_silu.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/interface.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/kernel.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/defer_size_check.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/layout_propagation.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/prepare_binary.cpp
    ${TORCH_SRC_DIR}/csrc/jit/codegen/onednn/guard_shape.cpp
  )
endif()

if(USE_ROCM)
  list(APPEND Caffe2_HIP_SRCS ${Caffe2_GPU_HIP_JIT_FUSERS_SRCS})
  if(USE_NCCL)
    list(APPEND Caffe2_HIP_SRCS
      ${TORCH_SRC_DIR}/csrc/cuda/nccl.cpp)
  endif()
  if(USE_DISTRIBUTED)
    append_filelist("libtorch_cuda_distributed_base_sources" Caffe2_HIP_SRCS)
    if(NOT WIN32)
      append_filelist("libtorch_cuda_distributed_extra_sources" Caffe2_HIP_SRCS)
    endif()
  endif()
  # caffe2_nvrtc's stubs to driver APIs are useful for HIP.
  # See NOTE [ ATen NVRTC Stub and HIP ]
  add_library(caffe2_nvrtc SHARED ${ATen_NVRTC_STUB_SRCS})
  target_link_libraries(caffe2_nvrtc hip::amdhip64 hiprtc::hiprtc)
  target_include_directories(caffe2_nvrtc PRIVATE ${CMAKE_BINARY_DIR})
  target_compile_definitions(caffe2_nvrtc PRIVATE USE_ROCM __HIP_PLATFORM_AMD__)
  install(TARGETS caffe2_nvrtc DESTINATION "${TORCH_INSTALL_LIB_DIR}")
endif()

if(NOT NO_API AND NOT BUILD_LITE_INTERPRETER)
  list(APPEND TORCH_SRCS
    ${TORCH_SRC_DIR}/csrc/api/src/cuda.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/data/datasets/mnist.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/distributed.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/random.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/sequential.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/data/samplers/stream.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/enum.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/imethod.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/serialize.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/jit.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/mps.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/init.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/module.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/_functions.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/activation.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/adaptive.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/batchnorm.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/normalization.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/instancenorm.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/conv.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/dropout.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/distance.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/embedding.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/fold.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/linear.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/loss.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/padding.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/pixelshuffle.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/pooling.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/rnn.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/upsampling.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/transformer.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/modules/container/functional.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/activation.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/adaptive.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/batchnorm.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/embedding.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/instancenorm.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/normalization.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/conv.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/dropout.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/linear.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/padding.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/pooling.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/rnn.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/vision.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/nn/options/transformer.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/adagrad.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/adam.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/adamw.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/lbfgs.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/optimizer.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/rmsprop.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/serialize.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/sgd.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/schedulers/lr_scheduler.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/schedulers/step_lr.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/optim/schedulers/reduce_on_plateau_scheduler.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/serialize/input-archive.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/serialize/output-archive.cpp
    ${TORCH_SRC_DIR}/csrc/api/src/xpu.cpp
  )
endif()

list(APPEND Caffe2_CPU_SRCS ${TORCH_SRCS})

if(USE_MPS)
  list(APPEND Caffe2_CPU_SRCS ${Caffe2_MPS_SRCS})
  list(APPEND Caffe2_CPU_SRCS ${TORCH_SRC_DIR}/csrc/inductor/aoti_torch/shim_mps.cpp)
  list(APPEND Caffe2_CPU_SRCS ${TORCH_SRC_DIR}/csrc/inductor/aoti_torch/shim_mps.mm)
  list(APPEND Caffe2_CPU_SRCS ${TORCH_SRC_DIR}/csrc/inductor/aoti_runner/model_container_runner_mps.cpp)
  if(CAN_COMPILE_METAL)
      file(TOUCH ${CMAKE_BINARY_DIR}/caffe2/aten/src/ATen/metallib_dummy.cpp)
      list(APPEND Caffe2_CPU_SRCS ${CMAKE_BINARY_DIR}/caffe2/aten/src/ATen/metallib_dummy.cpp)
  endif()
endif()

# NOTE [ Linking AVX and non-AVX files ]
#
# Regardless of the CPU capabilities, we build some files with AVX2, and AVX512
# instruction set. If the host CPU doesn't support those, we simply ignore their
# functions at runtime during dispatch.
#
# We must make sure that those files are at the end of the input list when
# linking the torch_cpu library. Otherwise, the following error scenario might
# occur:
# 1. A non-AVX2 and an AVX2 file both call a function defined with the `inline`
#    keyword
# 2. The compiler decides not to inline this function
# 3. Two different versions of the machine code are generated for this function:
#    one without AVX2 instructions and one with AVX2.
# 4. When linking, the AVX2 version is found earlier in the input object files,
#    so the linker makes the entire library use it, even in code not guarded by
#    the dispatcher.
# 5. A CPU without AVX2 support executes this function, encounters an AVX2
#    instruction and crashes.
#
# Thus we organize the input files in the following order:
# 1. All files with no AVX-n support
# 2. All files with AVX2 support ('*AVX2.cpp')
# 3. All files with AVX512 support ('*AVX512.cpp')
set(Caffe2_CPU_SRCS_NON_AVX)
set(Caffe2_CPU_SRCS_AVX2)
set(Caffe2_CPU_SRCS_AVX512)
foreach(input_filename ${Caffe2_CPU_SRCS})
  if(${input_filename} MATCHES "AVX2\\.cpp")
    list(APPEND Caffe2_CPU_SRCS_AVX2 ${input_filename})
  elseif(${input_filename} MATCHES "AVX512\\.cpp")
    list(APPEND Caffe2_CPU_SRCS_AVX512 ${input_filename})
  else()
    list(APPEND Caffe2_CPU_SRCS_NON_AVX ${input_filename})
  endif()
endforeach(input_filename)
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS_NON_AVX} ${Caffe2_CPU_SRCS_AVX2} ${Caffe2_CPU_SRCS_AVX512})

# ==========================================================
# END formerly-libtorch sources
# ==========================================================

if(BUILD_LIBTORCHLESS)
  find_library(TORCH_LIB torch PATHS $ENV{LIBTORCH_LIB_PATH} NO_DEFAULT_PATH)
  find_library(TORCH_CPU_LIB torch_cpu PATHS $ENV{LIBTORCH_LIB_PATH} NO_DEFAULT_PATH)

  if(USE_CUDA)
    find_library(TORCH_CUDA_LIB torch_cuda PATHS $ENV{LIBTORCH_LIB_PATH} NO_DEFAULT_PATH)
  endif()

  if(USE_ROCM)
    find_library(TORCH_HIP_LIB torch_hip PATHS $ENV{LIBTORCH_LIB_PATH} NO_DEFAULT_PATH)
  endif()

  if(USE_XPU)
    find_library(TORCH_XPU_LIB torch_xpu PATHS $ENV{LIBTORCH_LIB_PATH} NO_DEFAULT_PATH)
  endif()
  add_subdirectory(../torch torch)
  # ---[ Torch python bindings build
  set(TORCH_PYTHON_COMPILE_OPTIONS ${TORCH_PYTHON_COMPILE_OPTIONS} PARENT_SCOPE)
  set(TORCH_PYTHON_LINK_FLAGS ${TORCH_PYTHON_LINK_FLAGS} PARENT_SCOPE)
else()
  set(TORCH_LIB torch)
  set(TORCH_CPU_LIB torch_cpu)
  set(TORCH_CUDA_LIB torch_cuda)
  set(TORCH_HIP_LIB torch_hip)
  set(TORCH_XPU_LIB torch_xpu)
endif()


if(NOT BUILD_LIBTORCHLESS)
add_library(torch_cpu ${Caffe2_CPU_SRCS})
if(HAVE_SOVERSION)
  set_target_properties(torch_cpu PROPERTIES
      VERSION ${TORCH_VERSION} SOVERSION ${TORCH_SOVERSION})
endif()
torch_compile_options(torch_cpu)  # see cmake/public/utils.cmake

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND NOT USE_IOS AND NOT USE_COREML_DELEGATE)
  target_compile_options_if_supported(torch_cpu "-Wmissing-prototypes")
  target_compile_options_if_supported(torch_cpu "-Werror=missing-prototypes")
  if(TARGET torch_cuda)
    target_compile_options_if_supported(torch_cuda "-Wmissing-prototypes")
    target_compile_options_if_supported(torch_cuda "-Werror=missing-prototypes")
  endif()
  get_target_property(TORCH_CPU_SOURCES torch_cpu SOURCES)
  foreach(generated_file IN LISTS GENERATED_CXX_TORCH)
    set_source_files_properties(${generated_file} PROPERTIES COMPILE_OPTIONS "-Wno-missing-prototypes;-Wno-error=missing-prototypes")
  endforeach()
  foreach(source_file IN LISTS TORCH_CPU_SOURCES)
    get_filename_component(source_file "${source_file}" REALPATH)
    string(FIND "${source_file}" "${CMAKE_BINARY_DIR}" res)
    if(res GREATER -1)
      set_source_files_properties(${source_file} PROPERTIES COMPILE_OPTIONS "-Wno-missing-prototypes;-Wno-error=missing-prototypes")
      continue()
    endif()
    string(FIND "${source_file}" "embedding_lookup_idx_avx2.cc" res)
    if(res GREATER -1)
      set_source_files_properties(${source_file} PROPERTIES COMPILE_OPTIONS "-Wno-missing-prototypes;-Wno-error=missing-prototypes")
    endif()
  endforeach()
endif()
if(USE_MPS)
  if(CAN_COMPILE_METAL)
    add_dependencies(torch_cpu metallibs)
    target_link_options(torch_cpu PRIVATE -Wl,-sectcreate,__TEXT,metal_basic,${CMAKE_CURRENT_BINARY_DIR}/aten/src/ATen/kernels_basic.metallib)
  else()
    target_compile_definitions(torch_cpu PRIVATE PYTORCH_JIT_COMPILE_SHADERS)
  endif()
endif()

option(TORCH_USE_IWYU "Use include-what-you-use to clean up header inclusion" OFF)
if(TORCH_USE_IWYU)
  find_program(iwyu NAMES include-what-you-use)
  if(iwyu)
    set(iwyu_cmd
        "include-what-you-use"
        "-Xiwyu"
        "--transitive_includes_only"
        "-Xiwyu"
        "--no_fwd_decls"
        "-Xiwyu"
        "--prefix_header_includes=keep"
        "-Xiwyu"
        "--mapping_file=${CMAKE_CURRENT_LIST_DIR}/../tools/iwyu/all.imp"
        )
    set_property(TARGET torch_cpu PROPERTY CXX_INCLUDE_WHAT_YOU_USE ${iwyu_cmd})
  endif()
endif()

set_property(SOURCE ${ATen_CORE_SRCS} APPEND
    PROPERTY COMPILE_DEFINITIONS "TORCH_ASSERT_ONLY_METHOD_OPERATORS")
set_property(SOURCE ${ATen_ATTENTION_KERNEL_SRCS} APPEND
    PROPERTY COMPILE_DEFINITIONS "TORCH_ASSERT_NO_OPERATORS")

if(USE_MPS OR USE_PYTORCH_METAL)
  enable_language(OBJC OBJCXX)
endif()

if(USE_PRECOMPILED_HEADERS)
  target_precompile_headers(torch_cpu PRIVATE
      "$<$<COMPILE_LANGUAGE:CXX>:ATen/core/ATen_pch.h>")
  # Exclude some files from using PCH
  set_source_files_properties(
      # Not built with OpenMP, so PCH is invalid
      ${Torch_SOURCE_DIR}/aten/src/ATen/MapAllocator.cpp
      # Builds with incompatible compiler flags
      ${Caffe2_CPU_SRCS_AVX2}
      ${Caffe2_CPU_SRCS_AVX512}
      PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
endif()

# Pass path to PocketFFT
if(AT_POCKETFFT_ENABLED)
  set_source_files_properties(
      "${PROJECT_SOURCE_DIR}/aten/src/ATen/native/mkl/SpectralOps.cpp"
      PROPERTIES INCLUDE_DIRECTORIES "${POCKETFFT_INCLUDE_DIR}")
endif()

if(CMAKE_COMPILER_IS_GNUCXX AND BUILD_LIBTORCH_CPU_WITH_DEBUG)
  # To enable debug fission we need to build libtorch_cpu with debug info on,
  # but this increases link time and peak memory usage if we use the
  # REL_WITH_DEB_INFO env var since that enables it for everything, but it's
  # only really necessary for libtorch_cpu.
  target_compile_options(torch_cpu PRIVATE "-g")
endif()

if(USE_LLVM AND LLVM_FOUND)
  llvm_map_components_to_libnames(LLVM_LINK_LIBS
    support core analysis executionengine instcombine
    scalaropts transformutils ${LLVM_TARGETS_TO_BUILD} orcjit)
  target_link_libraries(torch_cpu PRIVATE ${LLVM_LINK_LIBS})
endif(USE_LLVM AND LLVM_FOUND)

# This is required for older versions of CMake, which don't allow
# specifying add_library() without a list of source files
set(DUMMY_EMPTY_FILE ${CMAKE_BINARY_DIR}/empty.cpp)

if(MSVC)
  set(DUMMY_FILE_CONTENT "__declspec(dllexport) int ignore_this_library_placeholder(){return 0\\;}")
else()
  set(DUMMY_FILE_CONTENT "")
endif()

file(WRITE ${DUMMY_EMPTY_FILE} ${DUMMY_FILE_CONTENT})

# Wrapper library for people who link against torch and expect both CPU and CUDA support
# Contains "torch_cpu" and "torch_cuda"
add_library(torch ${DUMMY_EMPTY_FILE})
if(HAVE_SOVERSION)
  set_target_properties(torch PROPERTIES
      VERSION ${TORCH_VERSION} SOVERSION ${TORCH_SOVERSION})
endif()

if(USE_ROCM)
  filter_list(__caffe2_hip_srcs_cpp Caffe2_HIP_SRCS "\\.(cu|hip)$")
  set_source_files_properties(${__caffe2_hip_srcs_cpp} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
endif()

# Compile exposed libraries.
if(USE_ROCM)
  set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
  list(APPEND Caffe2_HIP_SRCS ${GENERATED_CXX_TORCH_CUDA})
  hip_add_library(torch_hip ${Caffe2_HIP_SRCS})
  if(USE_FLASH_ATTENTION)
    target_link_libraries(torch_hip PRIVATE __caffe2_aotriton)
  endif()
  set(CUDA_LINK_LIBRARIES_KEYWORD)
  torch_compile_options(torch_hip)  # see cmake/public/utils.cmake
  # TODO: Not totally sure if this is live or not
  if(USE_NCCL)
    target_link_libraries(torch_hip PRIVATE __caffe2_nccl)
    target_compile_definitions(torch_hip PRIVATE USE_NCCL)
  endif()

  if(USE_PRECOMPILED_HEADERS)
    target_precompile_headers(torch_hip PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:ATen/core/ATen_pch.h>")
  endif()
elseif(USE_CUDA)
  set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
  list(APPEND Caffe2_GPU_SRCS ${GENERATED_CXX_TORCH_CUDA})
  if(CUDA_SEPARABLE_COMPILATION)
    # Separate compilation fails when kernels using `thrust::sort_by_key`
    # are linked with the rest of CUDA code. Workaround by linking them separately.
    add_library(torch_cuda ${Caffe2_GPU_SRCS} ${Caffe2_GPU_CU_SRCS})
    set_property(TARGET torch_cuda PROPERTY CUDA_SEPARABLE_COMPILATION ON)

    add_library(torch_cuda_w_sort_by_key OBJECT
        ${Caffe2_GPU_SRCS_W_SORT_BY_KEY}
        ${Caffe2_GPU_CU_SRCS_W_SORT_BY_KEY})
    set_property(TARGET torch_cuda_w_sort_by_key PROPERTY CUDA_SEPARABLE_COMPILATION OFF)
    target_link_libraries(torch_cuda PRIVATE torch_cuda_w_sort_by_key)
  else()
    add_library(torch_cuda
        ${Caffe2_GPU_SRCS} ${Caffe2_GPU_SRCS_W_SORT_BY_KEY}
        ${Caffe2_GPU_CU_SRCS} ${Caffe2_GPU_CU_SRCS_W_SORT_BY_KEY})
  endif()
  set(CUDA_LINK_LIBRARIES_KEYWORD)
  torch_compile_options(torch_cuda)  # see cmake/public/utils.cmake
  target_compile_definitions(torch_cuda PRIVATE USE_CUDA)

  if(USE_CUFILE)
    target_link_libraries(torch_cuda PRIVATE torch::cufile)
    target_compile_definitions(torch_cuda PRIVATE USE_CUFILE)
  endif()
  if(USE_CUSPARSELT)
      target_link_libraries(torch_cuda PRIVATE torch::cusparselt)
      target_compile_definitions(torch_cuda PRIVATE USE_CUSPARSELT)
  endif()
  if(USE_CUDSS)
    target_link_libraries(torch_cuda PRIVATE torch::cudss)
    target_compile_definitions(torch_cuda PRIVATE USE_CUDSS)
  endif()
  if(USE_NCCL)
    target_link_libraries(torch_cuda PRIVATE __caffe2_nccl)
    target_compile_definitions(torch_cuda PRIVATE USE_NCCL)
  endif()

  # Compile with NVSHMEM
  # Default value of `USE_NVSHMEM` is set in CMakeLists.txt under root, to ON.
  if(USE_NVSHMEM)
    message(STATUS "NVSHMEM_HOME set to:  '$ENV{NVSHMEM_HOME}'")
    message(STATUS "NVSHMEM wheel installed at:  '${NVSHMEM_PY_DIR}'")
    # Search order:
    # 1. If user has specified `NVSHMEM_HOME`, we use it;
    # 2. If NVSHMEM wheel has been installed, we use it, see
    # tools/setup_helpers/cmake.py, where we set `NVSHMEM_PY_DIR` to the wheel
    # location, e.g.
    # `/path/to/conda/lib/python3.10/site-packages/nvidia/nvshmem`,
    # 3. Let CMake find it in the default system paths, e.g. /usr/local.
    find_library(NVSHMEM_HOST_LIB
      # In pip install case, the lib suffix is `.so.3` instead of `.so`
      NAMES nvshmem_host libnvshmem_host.so.3 NAMES_PER_DIR
      HINTS $ENV{NVSHMEM_HOME} ${NVSHMEM_PY_DIR}
      PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64
      DOC "The location of NVSHMEM host library.")
    find_library(NVSHMEM_DEVICE_LIB
      # Device lib is a `.a` file
      NAMES nvshmem_device
      HINTS $ENV{NVSHMEM_HOME} ${NVSHMEM_PY_DIR}
      PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64
      DOC "The location of NVSHMEM device library.")
    find_path(NVSHMEM_INCLUDE_DIR
      NAMES nvshmem.h
      HINTS $ENV{NVSHMEM_HOME}/include ${NVSHMEM_PY_DIR}/include
      DOC "The location of NVSHMEM headers.")
    message(STATUS "NVSHMEM_HOST_LIB:  '${NVSHMEM_HOST_LIB}'")
    message(STATUS "NVSHMEM_DEVICE_LIB:  '${NVSHMEM_DEVICE_LIB}'")
    message(STATUS "NVSHMEM_INCLUDE_DIR:  '${NVSHMEM_INCLUDE_DIR}'")
  endif()

  # If NVSHMEM_LIBRARY is found, we build torch_cuda with NVSHMEM support.
  if(NVSHMEM_HOST_LIB AND NVSHMEM_DEVICE_LIB AND NVSHMEM_INCLUDE_DIR)
    message(STATUS "NVSHMEM found, building with NVSHMEM support")
    include_directories(${NVSHMEM_INCLUDE_DIR})

    # Linking with nvshmem requires the source binary to be built with -rdc
    # which is not viable for libtorch_cuda. So we isolate the linking of
    # nvshmem in torch_nvshmem.
    add_library(torch_nvshmem SHARED
        "${TORCH_SRC_DIR}/csrc/distributed/c10d/cuda/utils.cpp"
        "${TORCH_SRC_DIR}/csrc/distributed/c10d/symm_mem/nvshmem_extension.cu"
        "${TORCH_SRC_DIR}/csrc/distributed/c10d/symm_mem/NVSHMEMSymmetricMemory.cu"
        "${TORCH_SRC_DIR}/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.cpp"
    )
    set_target_properties(torch_nvshmem PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(torch_nvshmem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)
    target_compile_options(torch_nvshmem PRIVATE "-U__CUDA_NO_HALF_OPERATORS__")
    target_link_libraries(torch_nvshmem PRIVATE
        ${NVSHMEM_HOST_LIB}
        ${NVSHMEM_DEVICE_LIB}
    )
    target_compile_definitions(torch_cuda PUBLIC USE_NVSHMEM)
    target_compile_definitions(torch_nvshmem PUBLIC USE_NVSHMEM)
    target_link_libraries(torch_cuda PRIVATE torch_nvshmem)
    install(TARGETS torch_nvshmem EXPORT Caffe2Targets DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  else()
    message(STATUS "NVSHMEM not found, not building with NVSHMEM support.")
  endif()

  if(USE_UCC)
    target_link_libraries(torch_cuda PRIVATE __caffe2_ucc)
    target_compile_definitions(torch_cuda PRIVATE USE_UCC)
  endif()
  if(USE_FLASH_ATTENTION)
    target_compile_definitions(torch_cuda PRIVATE
        USE_FLASH_ATTENTION
        FLASHATTENTION_DISABLE_ALIBI    # Disable alibi attention as it's not currently used
        FLASHATTENTION_DISABLE_SOFTCAP
        FLASH_NAMESPACE=pytorch_flash
        UNFUSE_FMA                      # Addressing issue #121558
      )
    target_sources(torch_cuda PRIVATE $<TARGET_OBJECTS:flash_attention>)
    target_include_directories(torch_cuda SYSTEM PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/flash-attention/csrc>
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/flash-attention/include>
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/cutlass/include>
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/flash-attention/csrc/flash_attn/src>
      $<INSTALL_INTERFACE:include>
    )
  endif()
  if(USE_MEM_EFF_ATTENTION)
    target_compile_definitions(torch_cuda PRIVATE USE_MEM_EFF_ATTENTION)
  endif()
  if(BUILD_LAZY_CUDA_LINALG)
    add_library(torch_cuda_linalg ${ATen_CUDA_LINALG_SRCS})
    target_compile_definitions(torch_cuda_linalg PRIVATE USE_CUDA BUILD_LAZY_CUDA_LINALG)
    # Library order is important during static linking
    # `torch::magma` should be mentioned before other CUDA
    # to transitively include all symbols present in torch_cuda/torch_cpu
    if(USE_MAGMA)
      target_link_libraries(torch_cuda_linalg PRIVATE torch::magma)
      # CUDAHooks reports version of MAGMA PyTorch was compiled against, i.e. needs to be able to include magma headers
      get_target_property(HOOKS_INCLUDE_DIRECTORIES torch_cuda INCLUDE_DIRECTORIES)
      if(NOT "${MAGMA_INCLUDE_DIR}" IN_LIST HOOKS_INCLUDE_DIRECTORIES)
        set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/ATen/cuda/detail/CUDAHooks.cpp PROPERTIES INCLUDE_DIRECTORIES  "${MAGMA_INCLUDE_DIR}")
      endif()
    endif()
    target_link_libraries(torch_cuda_linalg PRIVATE
        torch_cpu
        torch_cuda
    )
    if($ENV{ATEN_STATIC_CUDA})
    target_link_libraries(torch_cuda_linalg PRIVATE
        CUDA::cusolver_static
        ${CUDAToolkit_LIBRARY_DIR}/libcusolver_lapack_static.a     # needed for libcusolver_static
    )
    else()
      target_link_libraries(torch_cuda_linalg PRIVATE
          CUDA::cusolver
      )
    endif()
    # NS: TODO, is this really necessary?
    if(USE_MAGMA AND CAFFE2_STATIC_LINK_CUDA)
      target_link_libraries(torch_cuda_linalg PRIVATE
          CUDA::culibos ${CMAKE_DL_LIBS})
    endif()
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/ATen/native/cuda/LinearAlgebraStubs.cpp PROPERTIES COMPILE_FLAGS "-DBUILD_LAZY_CUDA_LINALG")
    install(TARGETS torch_cuda_linalg DESTINATION "${TORCH_INSTALL_LIB_DIR}")
  endif()

  if(USE_PRECOMPILED_HEADERS)
    target_precompile_headers(torch_cuda PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:ATen/core/ATen_pch.h>")
  endif()

  # Apply suggestion from comment https://github.com/pytorch/pytorch/issues/113053#issuecomment-2115375714
  if(LINUX)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/ATen/cuda/CUDASparseDescriptors.cpp PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/ATen/cuda/CUDASparseBlas.cpp PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/ATen/native/sparse/cuda/SparseCUDABlas.cpp PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
  endif()
  # Set driver api defined for PeerToPeerAccess
  if(NOT WIN32)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../aten/src/ATen/cuda/PeerToPeerAccess.cpp PROPERTIES COMPILE_FLAGS "-DPYTORCH_C10_DRIVER_API_SUPPORTED=1")
  endif()

endif()

if(USE_XPU)
  list(APPEND Caffe2_XPU_SRCS ${GENERATED_CXX_TORCH_XPU})
  list(APPEND Caffe2_XPU_SRCS ${TORCH_SRC_DIR}/csrc/inductor/aoti_torch/shim_xpu.cpp)
  list(APPEND Caffe2_XPU_SRCS ${TORCH_SRC_DIR}/csrc/inductor/aoti_runner/model_container_runner_xpu.cpp)
  add_library(torch_xpu ${Caffe2_XPU_SRCS})
  torch_compile_options(torch_xpu)  # see cmake/public/utils.cmake
  target_compile_definitions(torch_xpu PRIVATE USE_XPU)

  # ATen XPU implementation
  set(TORCH_XPU_OPS_DIR ${TORCH_ROOT}/third_party/torch-xpu-ops)
  set(TORCH_XPU_OPS_REPO_URL https://github.com/intel/torch-xpu-ops.git)
  file(READ "${TORCH_ROOT}/third_party/xpu.txt" TORCH_XPU_OPS_COMMIT)
  string(REGEX REPLACE "\n$" "" TORCH_XPU_OPS_COMMIT "${TORCH_XPU_OPS_COMMIT}")
  if(NOT EXISTS "${TORCH_XPU_OPS_DIR}/.git")
    execute_process(
      COMMAND git clone --quiet ${TORCH_XPU_OPS_REPO_URL} ${TORCH_XPU_OPS_DIR}
      RESULT_VARIABLE _exitcode)
    if(NOT _exitcode EQUAL 0)
      message(FATAL_ERROR "Fail to clone ${TORCH_XPU_OPS_REPO_URL}")
    endif()
  endif()
  execute_process(
    COMMAND git fetch --quiet
    WORKING_DIRECTORY ${TORCH_XPU_OPS_DIR}
    RESULT_VARIABLE _exitcode)
  if(NOT _exitcode EQUAL 0)
    message(FATAL_ERROR "Fail to fetch ${TORCH_XPU_OPS_REPO_URL}")
  endif()
  execute_process(
    COMMAND git checkout --quiet ${TORCH_XPU_OPS_COMMIT}
    WORKING_DIRECTORY ${TORCH_XPU_OPS_DIR}
    RESULT_VARIABLE _exitcode)
  if(NOT _exitcode EQUAL 0)
    message(FATAL_ERROR "Fail to checkout ${TORCH_XPU_OPS_REPO_URL} to ${TORCH_XPU_OPS_COMMIT}")
  endif()

  set(TORCH_XPU_OPS_INCLUDE_DIRS
      ${TORCH_SRC_DIR}/csrc/api
      ${TORCH_SRC_DIR}/csrc/api/include
      ${Caffe2_CPU_INCLUDE}
      ${Caffe2_XPU_INCLUDE})
  # Pass the target as a dependency so that ATen headers generation
  # could be followed by torch-xpu-ops build.
  # 1. Sources in torch-xpu-ops depend on generated ATen headers.
  # 2. Using add_custom_command in torch-xpu-ops to define sycl device sources
  #    compilation. add_custom_command requires an explicit dependency.
  list(APPEND ${Caffe2_XPU_INCLUDE} ${TORCH_XPU_OPS_DIR}/src/ATen/)
  set(TORCH_XPU_OPS_PYTORCH_DEPS ATEN_CPU_FILES_GEN_TARGET)

  add_subdirectory(${TORCH_ROOT}/third_party/torch-xpu-ops
      ${CMAKE_BINARY_DIR}/caffe2/aten_xpu)
  if(NOT TARGET torch_xpu_ops)
    message(WARNING "Failed to include ATen XPU implementation target")
  else()
    # USE_C10D_XCCL to decide if XCCL backend is enabled in torch-xpu-ops build.
    if(USE_C10D_XCCL)
      target_compile_definitions(torch_xpu PUBLIC USE_C10D_XCCL)
    endif()
    target_link_libraries(torch_xpu PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,torch_xpu_ops>)

    # Set cached ${ATen_XPU_INCLUDE_DIRS} to torch
    include_directories(SYSTEM ${ATen_XPU_INCLUDE_DIRS})
    message(INFO "Install ${TORCH_XPU_OPS_DIR}/src/ATen/xpu to ${TORCH_IN
```



## High-Level Overview

This file is part of the PyTorch framework located at `caffe2`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `caffe2`, which is part of the **Caffe2** deep learning framework.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`caffe2`):



## Cross-References

- **File Documentation**: `CMakeLists.txt_docs.md`
- **Keyword Index**: `CMakeLists.txt_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
