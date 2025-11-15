# Documentation: `.ci/manywheel/build_xpu.sh`

## File Metadata

- **Path**: `.ci/manywheel/build_xpu.sh`
- **Size**: 3,573 bytes (3.49 KB)
- **Type**: Shell Script
- **Extension**: `.sh`

## File Purpose

This is a shell script that is part of the PyTorch project.

## Original Source

```bash
#!/usr/bin/env bash

set -ex

export TH_BINARY_BUILD=1
export USE_CUDA=0

# Keep an array of cmake variables to add to
if [[ -z "$CMAKE_ARGS" ]]; then
    # These are passed to tools/build_pytorch_libs.sh::build()
    CMAKE_ARGS=()
fi
if [[ -z "$EXTRA_CAFFE2_CMAKE_FLAGS" ]]; then
    # These are passed to tools/build_pytorch_libs.sh::build_caffe2()
    EXTRA_CAFFE2_CMAKE_FLAGS=()
fi


# Refer https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html
source /opt/intel/oneapi/compiler/latest/env/vars.sh
source /opt/intel/oneapi/pti/latest/env/vars.sh
source /opt/intel/oneapi/umf/latest/env/vars.sh
source /opt/intel/oneapi/ccl/latest/env/vars.sh
source /opt/intel/oneapi/mpi/latest/env/vars.sh
export USE_STATIC_MKL=1
export USE_ONEMKL=1
export USE_XCCL=1
export USE_MPI=0

WHEELHOUSE_DIR="wheelhousexpu"
LIBTORCH_HOUSE_DIR="libtorch_housexpu"
if [[ -z "$PYTORCH_FINAL_PACKAGE_DIR" ]]; then
    if [[ -z "$BUILD_PYTHONLESS" ]]; then
        PYTORCH_FINAL_PACKAGE_DIR="/remote/wheelhousexpu"
    else
        PYTORCH_FINAL_PACKAGE_DIR="/remote/libtorch_housexpu"
    fi
fi
mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR" || true

OS_NAME=$(awk -F= '/^NAME/{print $2}' /etc/os-release)
if [[ "$OS_NAME" == *"CentOS Linux"* ]]; then
    LIBGOMP_PATH="/usr/lib64/libgomp.so.1"
elif [[ "$OS_NAME" == *"Red Hat Enterprise Linux"* ]]; then
    LIBGOMP_PATH="/usr/lib64/libgomp.so.1"
elif [[ "$OS_NAME" == *"AlmaLinux"* ]]; then
    LIBGOMP_PATH="/usr/lib64/libgomp.so.1"
elif [[ "$OS_NAME" == *"Ubuntu"* ]]; then
    if [[ "$(uname -m)" == "s390x" ]]; then
        LIBGOMP_PATH="/usr/lib/s390x-linux-gnu/libgomp.so.1"
    else
        LIBGOMP_PATH="/usr/lib/x86_64-linux-gnu/libgomp.so.1"
    fi
fi

DEPS_LIST=(
    "$LIBGOMP_PATH"
    "/opt/intel/oneapi/compiler/latest/lib/libOpenCL.so.1"
)

DEPS_SONAME=(
    "libgomp.so.1"
    "libOpenCL.so.1"
)

if [[ -z "$PYTORCH_EXTRA_INSTALL_REQUIREMENTS" ]]; then
    echo "Bundling with xpu support package libs."
    DEPS_LIST+=(
        "/opt/intel/oneapi/compiler/latest/lib/libsycl.so.8"
        "/opt/intel/oneapi/compiler/latest/lib/libur_loader.so.0"
        "/opt/intel/oneapi/compiler/latest/lib/libur_adapter_level_zero.so.0"
        "/opt/intel/oneapi/compiler/latest/lib/libur_adapter_opencl.so.0"
        "/opt/intel/oneapi/compiler/latest/lib/libsvml.so"
        "/opt/intel/oneapi/compiler/latest/lib/libirng.so"
        "/opt/intel/oneapi/compiler/latest/lib/libimf.so"
        "/opt/intel/oneapi/compiler/latest/lib/libintlc.so.5"
        "/opt/intel/oneapi/pti/latest/lib/libpti_view.so.0.10"
        "/opt/intel/oneapi/umf/latest/lib/libumf.so.0"
        "/opt/intel/oneapi/tcm/latest/lib/libhwloc.so.15"
    )
    DEPS_SONAME+=(
        "libsycl.so.8"
        "libur_loader.so.0"
        "libur_adapter_level_zero.so.0"
        "libur_adapter_opencl.so.0"
        "libsvml.so"
        "libirng.so"
        "libimf.so"
        "libintlc.so.5"
        "libpti_view.so.0.10"
        "libumf.so.0"
        "libhwloc.so.15"
    )
else
    echo "Using xpu runtime libs from pypi."
    XPU_RPATHS=(
        '$ORIGIN/../../../..'
    )
    XPU_RPATHS=$(IFS=: ; echo "${XPU_RPATHS[*]}")
    export C_SO_RPATH=$XPU_RPATHS':$ORIGIN:$ORIGIN/lib'
    export LIB_SO_RPATH=$XPU_RPATHS':$ORIGIN'
    export FORCE_RPATH="--force-rpath"
fi

rm -rf /usr/local/cuda*

SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
if [[ -z "$BUILD_PYTHONLESS" ]]; then
    BUILD_SCRIPT=build_common.sh
else
    BUILD_SCRIPT=build_libtorch.sh
fi
source ${SOURCE_DIR}/${BUILD_SCRIPT}

```



## High-Level Overview

This file is part of the PyTorch framework located at `.ci/manywheel`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.ci/manywheel`, which is part of the PyTorch project infrastructure.



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

Files in the same folder (`.ci/manywheel`):

- [`build_common.sh_docs.md`](./build_common.sh_docs.md)
- [`test_wheel.sh_docs.md`](./test_wheel.sh_docs.md)
- [`build_cuda.sh_docs.md`](./build_cuda.sh_docs.md)
- [`build.sh_docs.md`](./build.sh_docs.md)
- [`build_cpu.sh_docs.md`](./build_cpu.sh_docs.md)
- [`LICENSE_docs.md`](./LICENSE_docs.md)
- [`build_rocm.sh_docs.md`](./build_rocm.sh_docs.md)
- [`build_libtorch.sh_docs.md`](./build_libtorch.sh_docs.md)
- [`set_desired_python.sh_docs.md`](./set_desired_python.sh_docs.md)


## Cross-References

- **File Documentation**: `build_xpu.sh_docs.md`
- **Keyword Index**: `build_xpu.sh_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
