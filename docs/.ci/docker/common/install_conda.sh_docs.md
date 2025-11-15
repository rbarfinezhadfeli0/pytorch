# Documentation: `.ci/docker/common/install_conda.sh`

## File Metadata

- **Path**: `.ci/docker/common/install_conda.sh`
- **Size**: 3,827 bytes (3.74 KB)
- **Type**: Shell Script
- **Extension**: `.sh`

## File Purpose

This file is part of the **documentation**.

## Original Source

```bash
#!/bin/bash

set -ex

# Optionally install conda
if [ -n "$ANACONDA_PYTHON_VERSION" ]; then
  BASE_URL="https://github.com/conda-forge/miniforge/releases/latest/download"  # @lint-ignore
  CONDA_FILE="Miniforge3-Linux-$(uname -m).sh"

  MAJOR_PYTHON_VERSION=$(echo "$ANACONDA_PYTHON_VERSION" | cut -d . -f 1)
  MINOR_PYTHON_VERSION=$(echo "$ANACONDA_PYTHON_VERSION" | cut -d . -f 2)

  case "$MAJOR_PYTHON_VERSION" in
    3);;
    *)
      echo "Unsupported ANACONDA_PYTHON_VERSION: $ANACONDA_PYTHON_VERSION"
      exit 1
      ;;
  esac
  mkdir -p /opt/conda
  chown jenkins:jenkins /opt/conda

  SCRIPT_FOLDER="$( cd "$(dirname "$0")" ; pwd -P )"
  source "${SCRIPT_FOLDER}/common_utils.sh"

  pushd /tmp
  wget -q "${BASE_URL}/${CONDA_FILE}"
  # NB: Manually invoke bash per https://github.com/conda/conda/issues/10431
  as_jenkins bash "${CONDA_FILE}" -b -f -p "/opt/conda"
  popd

  # NB: Don't do this, rely on the rpath to get it right
  #echo "/opt/conda/lib" > /etc/ld.so.conf.d/conda-python.conf
  #ldconfig
  sed -e 's|PATH="\(.*\)"|PATH="/opt/conda/bin:\1"|g' -i /etc/environment
  export PATH="/opt/conda/bin:$PATH"

  # Ensure we run conda in a directory that jenkins has write access to
  pushd /opt/conda

  # Prevent conda from updating to 4.14.0, which causes docker build failures
  # See https://hud.pytorch.org/pytorch/pytorch/commit/754d7f05b6841e555cea5a4b2c505dd9e0baec1d
  # Uncomment the below when resolved to track the latest conda update
  # as_jenkins conda update -y -n base conda

  if [[ $(uname -m) == "aarch64" ]]; then
    export SYSROOT_DEP="sysroot_linux-aarch64=2.17"
  else
    export SYSROOT_DEP="sysroot_linux-64=2.17"
  fi

# Install correct Python version
# Also ensure sysroot is using a modern GLIBC to match system compilers
if [ "$ANACONDA_PYTHON_VERSION" = "3.14" ]; then
  as_jenkins conda create -n py_$ANACONDA_PYTHON_VERSION -y\
             python="3.14.0" \
             ${SYSROOT_DEP} \
             -c conda-forge
else
  # Install correct Python version
  # Also ensure sysroot is using a modern GLIBC to match system compilers
  as_jenkins conda create -n py_$ANACONDA_PYTHON_VERSION -y\
             python="$ANACONDA_PYTHON_VERSION" \
             ${SYSROOT_DEP}
fi
  # libstdcxx from conda default channels are too old, we need GLIBCXX_3.4.30
  # which is provided in libstdcxx 12 and up.
  conda_install libstdcxx-ng=12.3.0 --update-deps -c conda-forge

  # Miniforge installer doesn't install sqlite by default
  if [[ "$BUILD_ENVIRONMENT" == *rocm* ]]; then
    conda_install sqlite
  fi

  # Install PyTorch conda deps, as per https://github.com/pytorch/pytorch README
  if [[ $(uname -m) != "aarch64" ]]; then
    pip_install mkl==2024.2.0
    pip_install mkl-static==2024.2.0
    pip_install mkl-include==2024.2.0
  fi

  # Install llvm-8 as it is required to compile llvmlite-0.30.0 from source
  # and libpython-static for torch deploy
  conda_install llvmdev=8.0.0 "libpython-static=${ANACONDA_PYTHON_VERSION}"

  # Magma package names are concatenation of CUDA major and minor ignoring revision
  # I.e. magma-cuda102 package corresponds to CUDA_VERSION=10.2 and CUDA_VERSION=10.2.89
  # Magma is installed from a tarball in the ossci-linux bucket into the conda env
  if [ -n "$CUDA_VERSION" ]; then
    conda_run ${SCRIPT_FOLDER}/install_magma_conda.sh $(cut -f1-2 -d'.' <<< ${CUDA_VERSION})
  fi

  if [[ "$UBUNTU_VERSION" == "24.04"* ]] ; then
    conda_install_through_forge libstdcxx-ng=14
  fi

  # Install some other packages, including those needed for Python test reporting
  pip_install -r /opt/conda/requirements-ci.txt

  if [ -n "$DOCS" ]; then
    apt-get update
    apt-get -y install expect-dev

    # We are currently building docs with python 3.8 (min support version)
    pip_install -r /opt/conda/requirements-docs.txt
  fi

  popd
fi

```



## High-Level Overview

This file is part of the PyTorch framework located at `.ci/docker/common`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.ci/docker/common`, which is part of the PyTorch project infrastructure.



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

Files in the same folder (`.ci/docker/common`):

- [`install_mnist.sh_docs.md`](./install_mnist.sh_docs.md)
- [`install_amdsmi.sh_docs.md`](./install_amdsmi.sh_docs.md)
- [`install_user.sh_docs.md`](./install_user.sh_docs.md)
- [`install_openblas.sh_docs.md`](./install_openblas.sh_docs.md)
- [`install_magma.sh_docs.md`](./install_magma.sh_docs.md)
- [`install_cuda.sh_docs.md`](./install_cuda.sh_docs.md)
- [`common_utils.sh_docs.md`](./common_utils.sh_docs.md)
- [`install_inductor_benchmark_deps.sh_docs.md`](./install_inductor_benchmark_deps.sh_docs.md)
- [`install_gcc.sh_docs.md`](./install_gcc.sh_docs.md)
- [`install_clang.sh_docs.md`](./install_clang.sh_docs.md)


## Cross-References

- **File Documentation**: `install_conda.sh_docs.md`
- **Keyword Index**: `install_conda.sh_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
