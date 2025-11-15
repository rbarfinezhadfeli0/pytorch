# Documentation: `.ci/docker/common/install_base.sh`

## File Metadata

- **Path**: `.ci/docker/common/install_base.sh`
- **Size**: 3,625 bytes (3.54 KB)
- **Type**: Shell Script
- **Extension**: `.sh`

## File Purpose

This file is part of the **documentation**.

## Original Source

```bash
#!/bin/bash

set -ex

install_ubuntu() {
  # NVIDIA dockers for RC releases use tag names like `11.0-cudnn9-devel-ubuntu18.04-rc`,
  # for this case we will set UBUNTU_VERSION to `18.04-rc` so that the Dockerfile could
  # find the correct image. As a result, here we have to check for
  #   "$UBUNTU_VERSION" == "18.04"*
  # instead of
  #   "$UBUNTU_VERSION" == "18.04"
  if [[ "$UBUNTU_VERSION" == "20.04"* ]]; then
    cmake3="cmake=3.16*"
    maybe_libiomp_dev=""
  elif [[ "$UBUNTU_VERSION" == "22.04"* ]]; then
    cmake3="cmake=3.22*"
    maybe_libiomp_dev=""
  elif [[ "$UBUNTU_VERSION" == "24.04"* ]]; then
    cmake3="cmake=3.28*"
    maybe_libiomp_dev=""
  else
    cmake3="cmake=3.5*"
    maybe_libiomp_dev="libiomp-dev"
  fi

  if [[ "$CLANG_VERSION" == 15 ]]; then
    maybe_libomp_dev="libomp-15-dev"
  elif [[ "$CLANG_VERSION" == 12 ]]; then
    maybe_libomp_dev="libomp-12-dev"
  elif [[ "$CLANG_VERSION" == 10 ]]; then
    maybe_libomp_dev="libomp-10-dev"
  else
    maybe_libomp_dev=""
  fi

  # Install common dependencies
  apt-get update
  # TODO: Some of these may not be necessary
  ccache_deps="asciidoc docbook-xml docbook-xsl xsltproc"
  deploy_deps="libffi-dev libbz2-dev libreadline-dev libncurses5-dev libncursesw5-dev libgdbm-dev libsqlite3-dev uuid-dev tk-dev"
  numpy_deps="gfortran"
  apt-get install -y --no-install-recommends \
    $ccache_deps \
    $numpy_deps \
    ${deploy_deps} \
    ${cmake3} \
    apt-transport-https \
    autoconf \
    automake \
    build-essential \
    ca-certificates \
    curl \
    git \
    libatlas-base-dev \
    libc6-dbg \
    ${maybe_libiomp_dev} \
    libyaml-dev \
    libz-dev \
    libjemalloc2 \
    libjpeg-dev \
    libasound2-dev \
    libsndfile-dev \
    ${maybe_libomp_dev} \
    software-properties-common \
    wget \
    sudo \
    vim \
    jq \
    libtool \
    vim \
    unzip \
    gpg-agent \
    gdb \
    bc

  # Should resolve issues related to various apt package repository cert issues
  # see: https://github.com/pytorch/pytorch/issues/65931
  apt-get install -y libgnutls30

  # Cleanup package manager
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

install_centos() {
  # Need EPEL for many packages we depend on.
  # See http://fedoraproject.org/wiki/EPEL
  yum --enablerepo=extras install -y epel-release

  ccache_deps="asciidoc docbook-dtds docbook-style-xsl libxslt"
  numpy_deps="gcc-gfortran"
  yum install -y \
    $ccache_deps \
    $numpy_deps \
    autoconf \
    automake \
    bzip2 \
    cmake \
    cmake3 \
    curl \
    gcc \
    gcc-c++ \
    gflags-devel \
    git \
    glibc-devel \
    glibc-headers \
    glog-devel \
    libstdc++-devel \
    libsndfile-devel \
    make \
    opencv-devel \
    sudo \
    wget \
    vim \
    unzip \
    gdb

  # Cleanup
  yum clean all
  rm -rf /var/cache/yum
  rm -rf /var/lib/yum/yumdb
  rm -rf /var/lib/yum/history
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    install_ubuntu
    ;;
  centos)
    install_centos
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac

# Install Valgrind separately since the apt-get version is too old.
mkdir valgrind_build && cd valgrind_build
VALGRIND_VERSION=3.20.0
wget https://ossci-linux.s3.amazonaws.com/valgrind-${VALGRIND_VERSION}.tar.bz2
tar -xjf valgrind-${VALGRIND_VERSION}.tar.bz2
cd valgrind-${VALGRIND_VERSION}
./configure --prefix=/usr/local
make -j$[$(nproc) - 2]
sudo make install
cd ../../
rm -rf valgrind_build
alias valgrind="/usr/local/bin/valgrind"

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

- Implements or uses **caching** mechanisms.

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

- **File Documentation**: `install_base.sh_docs.md`
- **Keyword Index**: `install_base.sh_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
