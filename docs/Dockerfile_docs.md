# Documentation: `Dockerfile`

## File Metadata

- **Path**: `Dockerfile`
- **Size**: 3,836 bytes (3.75 KB)
- **Type**: Source File ()
- **Extension**: ``

## File Purpose

This is a source file () that is part of the PyTorch project.

## Original Source

```
# syntax=docker/dockerfile:1

# NOTE: Building this image require's docker version >= 23.0.
#
# For reference:
# - https://docs.docker.com/build/dockerfile/frontend/#stable-channel

ARG BASE_IMAGE=ubuntu:22.04
ARG PYTHON_VERSION=3.11

FROM ${BASE_IMAGE} as dev-base
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ARG PYTHON_VERSION=3.11
# Automatically set by buildx
ARG TARGETPLATFORM
# translating Docker's TARGETPLATFORM into miniconda arches
RUN case ${TARGETPLATFORM} in \
         "linux/arm64")  MINICONDA_ARCH=aarch64  ;; \
         *)              MINICONDA_ARCH=x86_64   ;; \
    esac && \
    curl -fsSL -v -o ~/miniconda.sh -O  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${MINICONDA_ARCH}.sh"
COPY requirements.txt requirements-build.txt .
# Manually invoke bash on miniconda script per https://github.com/conda/conda/issues/10431
RUN chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml numpy ipython && \
    /opt/conda/bin/python -mpip install -r requirements.txt && \
    /opt/conda/bin/conda clean -ya

FROM dev-base as submodule-update
WORKDIR /opt/pytorch
COPY . .
RUN git submodule update --init --recursive

FROM conda as conda-installs
ARG PYTHON_VERSION=3.11
ARG CUDA_PATH=cu121
ARG INSTALL_CHANNEL=whl/nightly
# Automatically set by buildx
# pinning version of conda here see: https://github.com/pytorch/pytorch/issues/164574
RUN /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda=25.7.0

ARG TARGETPLATFORM

# INSTALL_CHANNEL whl - release, whl/nightly - nightly, whl/test - test channels
RUN case ${TARGETPLATFORM} in \
         "linux/arm64")  pip install --extra-index-url https://download.pytorch.org/whl/cpu/ torch torchvision torchaudio ;; \
         *)              pip install --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_PATH#.}/ torch torchvision torchaudio ;; \
    esac && \
    /opt/conda/bin/conda clean -ya
RUN /opt/conda/bin/pip install torchelastic
RUN IS_CUDA=$(python -c 'import torch ; print(torch.cuda._is_compiled())'); \
    echo "Is torch compiled with cuda: ${IS_CUDA}"; \
    if test "${IS_CUDA}" != "True" -a ! -z "${CUDA_VERSION}"; then \
        exit 1; \
    fi

FROM ${BASE_IMAGE} as official
ARG PYTORCH_VERSION
ARG TRITON_VERSION
ARG TARGETPLATFORM
ARG CUDA_VERSION
LABEL com.nvidia.volumes.needed="nvidia_driver"
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        && rm -rf /var/lib/apt/lists/*
COPY --from=conda-installs /opt/conda /opt/conda
RUN if test -n "${TRITON_VERSION}" -a "${TARGETPLATFORM}" != "linux/arm64"; then \
        DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends gcc; \
        rm -rf /var/lib/apt/lists/*; \
    fi
ENV PATH /opt/conda/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
ENV PYTORCH_VERSION ${PYTORCH_VERSION}
WORKDIR /workspace

FROM official as dev
# Should override the already installed version from the official-image stage
COPY --from=conda /opt/conda /opt/conda
COPY --from=submodule-update /opt/pytorch /opt/pytorch

```



## High-Level Overview

This file is part of the PyTorch framework located at ``.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `root`, which is part of the PyTorch project infrastructure.



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

Files in the same folder (`root`):

- [`AGENTS.md_docs.md`](./AGENTS.md_docs.md)
- [`pytest.ini_docs.md`](./pytest.ini_docs.md)
- [`codex_setup.sh_docs.md`](./codex_setup.sh_docs.md)
- [`pt_template_srcs.bzl_docs.md`](./pt_template_srcs.bzl_docs.md)
- [`aten.bzl_docs.md`](./aten.bzl_docs.md)
- [`build.bzl_docs.md`](./build.bzl_docs.md)
- [`buckbuild.bzl_docs.md`](./buckbuild.bzl_docs.md)
- [`.bc-linter.yml_docs.md`](./.bc-linter.yml_docs.md)
- [`setup.py_docs.md`](./setup.py_docs.md)


## Cross-References

- **File Documentation**: `Dockerfile_docs.md`
- **Keyword Index**: `Dockerfile_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
