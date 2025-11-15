# Documentation: `.ci/docker/almalinux/Dockerfile`

## File Metadata

- **Path**: `.ci/docker/almalinux/Dockerfile`
- **Size**: 4,566 bytes (4.46 KB)
- **Type**: Source File ()
- **Extension**: ``

## File Purpose

This file is part of the **documentation**.

## Original Source

```
ARG CUDA_VERSION=12.6
ARG BASE_TARGET=cuda${CUDA_VERSION}
ARG ROCM_IMAGE=rocm/dev-almalinux-8:6.3-complete
FROM amd64/almalinux:8.10-20250519 as base

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

ARG DEVTOOLSET_VERSION=13

RUN yum -y update
RUN yum -y install epel-release
# install glibc-langpack-en make sure en_US.UTF-8 locale is available
RUN yum -y install glibc-langpack-en
RUN yum install -y sudo wget curl perl util-linux xz bzip2 git patch which perl zlib-devel openssl-devel yum-utils autoconf automake make gcc-toolset-${DEVTOOLSET_VERSION}-gcc gcc-toolset-${DEVTOOLSET_VERSION}-gcc-c++ gcc-toolset-${DEVTOOLSET_VERSION}-gcc-gfortran gcc-toolset-${DEVTOOLSET_VERSION}-gdb
# Just add everything as a safe.directory for git since these will be used in multiple places with git
RUN git config --global --add safe.directory '*'
ENV PATH=/opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/bin:$PATH

# cmake-3.18.4 from pip
RUN yum install -y python3-pip && \
    python3 -mpip install cmake==3.18.4 && \
    ln -s /usr/local/bin/cmake /usr/bin/cmake3
RUN rm -rf /usr/local/cuda-*

FROM base as openssl
ADD ./common/install_openssl.sh install_openssl.sh
RUN bash ./install_openssl.sh && rm install_openssl.sh

FROM base as patchelf
# Install patchelf
ADD ./common/install_patchelf.sh install_patchelf.sh
RUN bash ./install_patchelf.sh && rm install_patchelf.sh && cp $(which patchelf) /patchelf

FROM base as conda
# Install Anaconda
ADD ./common/install_conda_docker.sh install_conda.sh
RUN bash ./install_conda.sh && rm install_conda.sh

# Install CUDA
FROM base as cuda
ARG CUDA_VERSION=12.6
ARG DEVTOOLSET_VERSION=13
RUN rm -rf /usr/local/cuda-*
ADD ./common/install_cuda.sh install_cuda.sh
COPY ./common/install_nccl.sh install_nccl.sh
COPY ./ci_commit_pins/nccl-cu* /ci_commit_pins/
COPY ./common/install_cusparselt.sh install_cusparselt.sh
ENV CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
# Preserve CUDA_VERSION for the builds
ENV CUDA_VERSION=${CUDA_VERSION}
# Make things in our path by default
ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin:/opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/bin:$PATH


FROM cuda as cuda12.6
RUN bash ./install_cuda.sh 12.6
ENV DESIRED_CUDA=12.6

FROM cuda as cuda12.8
RUN bash ./install_cuda.sh 12.8
ENV DESIRED_CUDA=12.8

FROM cuda as cuda12.9
RUN bash ./install_cuda.sh 12.9
ENV DESIRED_CUDA=12.9

FROM cuda as cuda13.0
RUN bash ./install_cuda.sh 13.0
ENV DESIRED_CUDA=13.0

FROM ${ROCM_IMAGE} as rocm_base
ARG DEVTOOLSET_VERSION=13
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
# Install devtoolset on ROCm base image
RUN yum -y update && \
    yum -y install epel-release && \
    yum -y install glibc-langpack-en && \
    yum install -y sudo wget curl perl util-linux xz bzip2 git patch which perl zlib-devel openssl-devel yum-utils autoconf automake make gcc-toolset-${DEVTOOLSET_VERSION}-gcc gcc-toolset-${DEVTOOLSET_VERSION}-gcc-c++ gcc-toolset-${DEVTOOLSET_VERSION}-gcc-gfortran gcc-toolset-${DEVTOOLSET_VERSION}-gdb
RUN git config --global --add safe.directory '*'
ENV PATH=/opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/bin:$PATH

FROM rocm_base as rocm
ARG PYTORCH_ROCM_ARCH
ARG DEVTOOLSET_VERSION=13
ENV PYTORCH_ROCM_ARCH ${PYTORCH_ROCM_ARCH}
ADD ./common/install_mkl.sh install_mkl.sh
RUN bash ./install_mkl.sh && rm install_mkl.sh
ENV MKLROOT /opt/intel

# Install MNIST test data
FROM base as mnist
ADD ./common/install_mnist.sh install_mnist.sh
RUN bash ./install_mnist.sh

FROM base as all_cuda
COPY --from=cuda12.6  /usr/local/cuda-12.6 /usr/local/cuda-12.6
COPY --from=cuda12.8  /usr/local/cuda-12.8 /usr/local/cuda-12.8
COPY --from=cuda12.9  /usr/local/cuda-12.9 /usr/local/cuda-12.9
COPY --from=cuda13.0  /usr/local/cuda-13.0 /usr/local/cuda-13.0

# Final step
FROM ${BASE_TARGET} as final
ARG DEVTOOLSET_VERSION=13
COPY --from=openssl            /opt/openssl           /opt/openssl
COPY --from=patchelf           /patchelf              /usr/local/bin/patchelf
COPY --from=conda              /opt/conda             /opt/conda

# Add jni.h for java host build.
COPY ./common/install_jni.sh install_jni.sh
COPY ./java/jni.h jni.h
RUN bash ./install_jni.sh && rm install_jni.sh

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/lib64:/opt/rh/gcc-toolset-${DEVTOOLSET_VERSION}/root/usr/lib:$LD_LIBRARY_PATH
COPY --from=mnist  /usr/local/mnist /usr/local/mnist
RUN rm -rf /usr/local/cuda
RUN chmod o+rw /usr/local
RUN touch /.condarc && \
    chmod o+rw /.condarc && \
    chmod -R o+rw /opt/conda

```



## High-Level Overview

This file is part of the PyTorch framework located at `.ci/docker/almalinux`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.ci/docker/almalinux`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`.ci/docker/almalinux`):

- [`build.sh_docs.md`](./build.sh_docs.md)


## Cross-References

- **File Documentation**: `Dockerfile_docs.md`
- **Keyword Index**: `Dockerfile_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
