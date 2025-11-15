# The PyTorch Repository: A Comprehensive Guide

## Preface

This book provides a complete, in-depth reference to the PyTorch repository. It covers every component, file, and concept in the codebase.

**Scope**: Entire PyTorch repository
**Target Audience**: Developers, researchers, contributors
**Depth**: Comprehensive technical documentation

---

## Table of Contents

- Part I: Project Overview
- Part II: Architecture & Design
- Part III: Core Components
- Part IV: Advanced Features
- Part V: Development & Testing
- Part VI: Performance & Optimization
- Part VII: Contributing & Extending

---

# Part I: Project Overview

## Chapter 1: Introduction to PyTorch

PyTorch is an open-source machine learning framework that provides:

1. **Tensor Computation**: GPU-accelerated tensor operations similar to NumPy
2. **Automatic Differentiation**: Tape-based autograd system for building neural networks
3. **Neural Network API**: High-level building blocks for deep learning models
4. **Production Ready**: Tools for deploying models to production

### Key Features

- **Dynamic Computation Graphs**: Build graphs on-the-fly for flexible model architectures
- **Python-First**: Native Python interface with intuitive APIs
- **Strong GPU Support**: Seamless CPU-GPU tensor transfers
- **Rich Ecosystem**: Extensive libraries for computer vision, NLP, and more

## Chapter 2: Repository Structure

The PyTorch repository is organized into several major components:


### android/

See [folder documentation](./android/doc.md) for details.

### aten/

See [folder documentation](./aten/doc.md) for details.

### benchmarks/

See [folder documentation](./benchmarks/doc.md) for details.

### binaries/

See [folder documentation](./binaries/doc.md) for details.

### c10/

See [folder documentation](./c10/doc.md) for details.

### caffe2/

See [folder documentation](./caffe2/doc.md) for details.

### cmake/

See [folder documentation](./cmake/doc.md) for details.

### docs/

See [folder documentation](./docs/doc.md) for details.

### functorch/

See [folder documentation](./functorch/doc.md) for details.

### mypy_plugins/

See [folder documentation](./mypy_plugins/doc.md) for details.

### scripts/

See [folder documentation](./scripts/doc.md) for details.

### test/

See [folder documentation](./test/doc.md) for details.

### third_party/

See [folder documentation](./third_party/doc.md) for details.

### tools/

See [folder documentation](./tools/doc.md) for details.

### torch/

See [folder documentation](./torch/doc.md) for details.

### torchgen/

See [folder documentation](./torchgen/doc.md) for details.


# Part II: Architecture & Design

## Chapter 3: Core Architecture

PyTorch follows a layered architecture:

1. **C10**: Core abstractions (tensors, devices, etc.)
2. **ATen**: Tensor library with operations
3. **Autograd**: Automatic differentiation engine
4. **Python API**: User-facing PyTorch interface

### Design Principles

- **Imperative Programming**: Code executes immediately (eager execution)
- **Extensibility**: Easy to add custom operations and modules
- **Performance**: Optimized C++ and CUDA kernels
- **Interoperability**: Works with NumPy, SciPy, and other Python libraries

## Chapter 4: Component Interactions

[Detailed component interaction diagrams and descriptions would go here]

# Part III: Core Components

## Chapter 5: Tensor Library (ATen)

[Detailed ATen documentation]

## Chapter 6: Autograd Engine

[Detailed autograd documentation]

## Chapter 7: Neural Network Modules

[Detailed nn module documentation]

# Part IV: Advanced Features

## Chapter 8: JIT Compilation

[TorchScript and JIT documentation]

## Chapter 9: Distributed Training

[Distributed training documentation]

## Chapter 10: GPU Acceleration

[CUDA integration documentation]

# Part V: Development & Testing

## Chapter 11: Testing Infrastructure


The repository contains 6306 test files ensuring code quality.


## Chapter 12: Build System

[Build system documentation]

# Part VI: Performance & Optimization

## Chapter 13: Performance Best Practices

[Performance guidelines]

## Chapter 14: Profiling & Benchmarking

[Profiling tools documentation]

# Part VII: Contributing & Extending

## Chapter 15: Contribution Guidelines

[Contribution process]

## Chapter 16: Extending PyTorch

[How to add new features]

---

# Appendices

## Appendix A: File Reference

Complete list of all files in the repository:

- [.bc-linter.yml](.//.bc-linter.yml_docs.md)
- [.ci/caffe2/README.md](./.ci/caffe2/README.md_docs.md)
- [.ci/caffe2/common.sh](./.ci/caffe2/common.sh_docs.md)
- [.ci/caffe2/test.sh](./.ci/caffe2/test.sh_docs.md)
- [.ci/docker/README.md](./.ci/docker/README.md_docs.md)
- [.ci/docker/almalinux/Dockerfile](./.ci/docker/almalinux/Dockerfile_docs.md)
- [.ci/docker/almalinux/build.sh](./.ci/docker/almalinux/build.sh_docs.md)
- [.ci/docker/build.sh](./.ci/docker/build.sh_docs.md)
- [.ci/docker/centos-rocm/Dockerfile](./.ci/docker/centos-rocm/Dockerfile_docs.md)
- [.ci/docker/ci_commit_pins/executorch.txt](./.ci/docker/ci_commit_pins/executorch.txt_docs.md)
- [.ci/docker/ci_commit_pins/halide.txt](./.ci/docker/ci_commit_pins/halide.txt_docs.md)
- [.ci/docker/ci_commit_pins/huggingface-requirements.txt](./.ci/docker/ci_commit_pins/huggingface-requirements.txt_docs.md)
- [.ci/docker/ci_commit_pins/jax.txt](./.ci/docker/ci_commit_pins/jax.txt_docs.md)
- [.ci/docker/ci_commit_pins/nccl-cu11.txt](./.ci/docker/ci_commit_pins/nccl-cu11.txt_docs.md)
- [.ci/docker/ci_commit_pins/nccl-cu12.txt](./.ci/docker/ci_commit_pins/nccl-cu12.txt_docs.md)
- [.ci/docker/ci_commit_pins/nccl-cu13.txt](./.ci/docker/ci_commit_pins/nccl-cu13.txt_docs.md)
- [.ci/docker/ci_commit_pins/rocm-composable-kernel.txt](./.ci/docker/ci_commit_pins/rocm-composable-kernel.txt_docs.md)
- [.ci/docker/ci_commit_pins/timm.txt](./.ci/docker/ci_commit_pins/timm.txt_docs.md)
- [.ci/docker/ci_commit_pins/torchbench.txt](./.ci/docker/ci_commit_pins/torchbench.txt_docs.md)
- [.ci/docker/ci_commit_pins/triton-cpu.txt](./.ci/docker/ci_commit_pins/triton-cpu.txt_docs.md)
- [.ci/docker/ci_commit_pins/triton-xpu.txt](./.ci/docker/ci_commit_pins/triton-xpu.txt_docs.md)
- [.ci/docker/ci_commit_pins/triton.txt](./.ci/docker/ci_commit_pins/triton.txt_docs.md)
- [.ci/docker/common/cache_vision_models.sh](./.ci/docker/common/cache_vision_models.sh_docs.md)
- [.ci/docker/common/common_utils.sh](./.ci/docker/common/common_utils.sh_docs.md)
- [.ci/docker/common/install_acl.sh](./.ci/docker/common/install_acl.sh_docs.md)
- [.ci/docker/common/install_amdsmi.sh](./.ci/docker/common/install_amdsmi.sh_docs.md)
- [.ci/docker/common/install_base.sh](./.ci/docker/common/install_base.sh_docs.md)
- [.ci/docker/common/install_cache.sh](./.ci/docker/common/install_cache.sh_docs.md)
- [.ci/docker/common/install_clang.sh](./.ci/docker/common/install_clang.sh_docs.md)
- [.ci/docker/common/install_conda.sh](./.ci/docker/common/install_conda.sh_docs.md)
- [.ci/docker/common/install_conda_docker.sh](./.ci/docker/common/install_conda_docker.sh_docs.md)
- [.ci/docker/common/install_cpython.sh](./.ci/docker/common/install_cpython.sh_docs.md)
- [.ci/docker/common/install_cuda.sh](./.ci/docker/common/install_cuda.sh_docs.md)
- [.ci/docker/common/install_cudss.sh](./.ci/docker/common/install_cudss.sh_docs.md)
- [.ci/docker/common/install_cusparselt.sh](./.ci/docker/common/install_cusparselt.sh_docs.md)
- [.ci/docker/common/install_devtoolset.sh](./.ci/docker/common/install_devtoolset.sh_docs.md)
- [.ci/docker/common/install_docs_reqs.sh](./.ci/docker/common/install_docs_reqs.sh_docs.md)
- [.ci/docker/common/install_executorch.sh](./.ci/docker/common/install_executorch.sh_docs.md)
- [.ci/docker/common/install_gcc.sh](./.ci/docker/common/install_gcc.sh_docs.md)
- [.ci/docker/common/install_glibc.sh](./.ci/docker/common/install_glibc.sh_docs.md)
- [.ci/docker/common/install_halide.sh](./.ci/docker/common/install_halide.sh_docs.md)
- [.ci/docker/common/install_inductor_benchmark_deps.sh](./.ci/docker/common/install_inductor_benchmark_deps.sh_docs.md)
- [.ci/docker/common/install_jax.sh](./.ci/docker/common/install_jax.sh_docs.md)
- [.ci/docker/common/install_jni.sh](./.ci/docker/common/install_jni.sh_docs.md)
- [.ci/docker/common/install_lcov.sh](./.ci/docker/common/install_lcov.sh_docs.md)
- [.ci/docker/common/install_libgomp.sh](./.ci/docker/common/install_libgomp.sh_docs.md)
- [.ci/docker/common/install_libpng.sh](./.ci/docker/common/install_libpng.sh_docs.md)
- [.ci/docker/common/install_linter.sh](./.ci/docker/common/install_linter.sh_docs.md)
- [.ci/docker/common/install_magma.sh](./.ci/docker/common/install_magma.sh_docs.md)
- [.ci/docker/common/install_magma_conda.sh](./.ci/docker/common/install_magma_conda.sh_docs.md)
- [.ci/docker/common/install_mingw.sh](./.ci/docker/common/install_mingw.sh_docs.md)
- [.ci/docker/common/install_miopen.sh](./.ci/docker/common/install_miopen.sh_docs.md)
- [.ci/docker/common/install_mkl.sh](./.ci/docker/common/install_mkl.sh_docs.md)
- [.ci/docker/common/install_mnist.sh](./.ci/docker/common/install_mnist.sh_docs.md)
- [.ci/docker/common/install_nccl.sh](./.ci/docker/common/install_nccl.sh_docs.md)
- [.ci/docker/common/install_ninja.sh](./.ci/docker/common/install_ninja.sh_docs.md)
- [.ci/docker/common/install_nvpl.sh](./.ci/docker/common/install_nvpl.sh_docs.md)
- [.ci/docker/common/install_onnx.sh](./.ci/docker/common/install_onnx.sh_docs.md)
- [.ci/docker/common/install_openblas.sh](./.ci/docker/common/install_openblas.sh_docs.md)
- [.ci/docker/common/install_openmpi.sh](./.ci/docker/common/install_openmpi.sh_docs.md)
- [.ci/docker/common/install_openssl.sh](./.ci/docker/common/install_openssl.sh_docs.md)
- [.ci/docker/common/install_patchelf.sh](./.ci/docker/common/install_patchelf.sh_docs.md)
- [.ci/docker/common/install_python.sh](./.ci/docker/common/install_python.sh_docs.md)
- [.ci/docker/common/install_rocm.sh](./.ci/docker/common/install_rocm.sh_docs.md)
- [.ci/docker/common/install_rocm_drm.sh](./.ci/docker/common/install_rocm_drm.sh_docs.md)
- [.ci/docker/common/install_rocm_magma.sh](./.ci/docker/common/install_rocm_magma.sh_docs.md)
- [.ci/docker/common/install_triton.sh](./.ci/docker/common/install_triton.sh_docs.md)
- [.ci/docker/common/install_ucc.sh](./.ci/docker/common/install_ucc.sh_docs.md)
- [.ci/docker/common/install_user.sh](./.ci/docker/common/install_user.sh_docs.md)
- [.ci/docker/common/install_vision.sh](./.ci/docker/common/install_vision.sh_docs.md)
- [.ci/docker/common/install_xpu.sh](./.ci/docker/common/install_xpu.sh_docs.md)
- [.ci/docker/common/patch_libstdc.sh](./.ci/docker/common/patch_libstdc.sh_docs.md)
- [.ci/docker/java/jni.h](./.ci/docker/java/jni.h_docs.md)
- [.ci/docker/libtorch/Dockerfile](./.ci/docker/libtorch/Dockerfile_docs.md)
- [.ci/docker/libtorch/build.sh](./.ci/docker/libtorch/build.sh_docs.md)
- [.ci/docker/linter-cuda/Dockerfile](./.ci/docker/linter-cuda/Dockerfile_docs.md)
- [.ci/docker/linter/Dockerfile](./.ci/docker/linter/Dockerfile_docs.md)
- [.ci/docker/manywheel/build.sh](./.ci/docker/manywheel/build.sh_docs.md)
- [.ci/docker/manywheel/build_scripts/build.sh](./.ci/docker/manywheel/build_scripts/build.sh_docs.md)
- [.ci/docker/manywheel/build_scripts/build_utils.sh](./.ci/docker/manywheel/build_scripts/build_utils.sh_docs.md)
- [.ci/docker/manywheel/build_scripts/manylinux1-check.py](./.ci/docker/manywheel/build_scripts/manylinux1-check.py_docs.md)
- [.ci/docker/manywheel/build_scripts/ssl-check.py](./.ci/docker/manywheel/build_scripts/ssl-check.py_docs.md)
- [.ci/docker/requirements-ci.txt](./.ci/docker/requirements-ci.txt_docs.md)
- [.ci/docker/requirements-docs.txt](./.ci/docker/requirements-docs.txt_docs.md)
- [.ci/docker/triton_version.txt](./.ci/docker/triton_version.txt_docs.md)
- [.ci/docker/triton_xpu_version.txt](./.ci/docker/triton_xpu_version.txt_docs.md)
- [.ci/docker/ubuntu-cross-riscv/Dockerfile](./.ci/docker/ubuntu-cross-riscv/Dockerfile_docs.md)
- [.ci/docker/ubuntu-rocm/Dockerfile](./.ci/docker/ubuntu-rocm/Dockerfile_docs.md)
- [.ci/docker/ubuntu-xpu/Dockerfile](./.ci/docker/ubuntu-xpu/Dockerfile_docs.md)
- [.ci/docker/ubuntu/Dockerfile](./.ci/docker/ubuntu/Dockerfile_docs.md)
- [.ci/libtorch/build.sh](./.ci/libtorch/build.sh_docs.md)
- [.ci/lumen_cli/README.md](./.ci/lumen_cli/README.md_docs.md)
- [.ci/lumen_cli/cli/build_cli/__init__.py](./.ci/lumen_cli/cli/build_cli/__init__.py_docs.md)
- [.ci/lumen_cli/cli/build_cli/register_build.py](./.ci/lumen_cli/cli/build_cli/register_build.py_docs.md)
- [.ci/lumen_cli/cli/lib/__init__.py](./.ci/lumen_cli/cli/lib/__init__.py_docs.md)
- [.ci/lumen_cli/cli/lib/common/cli_helper.py](./.ci/lumen_cli/cli/lib/common/cli_helper.py_docs.md)
- [.ci/lumen_cli/cli/lib/common/docker_helper.py](./.ci/lumen_cli/cli/lib/common/docker_helper.py_docs.md)
- [.ci/lumen_cli/cli/lib/common/envs_helper.py](./.ci/lumen_cli/cli/lib/common/envs_helper.py_docs.md)
- [.ci/lumen_cli/cli/lib/common/gh_summary.py](./.ci/lumen_cli/cli/lib/common/gh_summary.py_docs.md)
- [.ci/lumen_cli/cli/lib/common/git_helper.py](./.ci/lumen_cli/cli/lib/common/git_helper.py_docs.md)
- [.ci/lumen_cli/cli/lib/common/logger.py](./.ci/lumen_cli/cli/lib/common/logger.py_docs.md)
- [.ci/lumen_cli/cli/lib/common/path_helper.py](./.ci/lumen_cli/cli/lib/common/path_helper.py_docs.md)
- [.ci/lumen_cli/cli/lib/common/pip_helper.py](./.ci/lumen_cli/cli/lib/common/pip_helper.py_docs.md)
- [.ci/lumen_cli/cli/lib/common/utils.py](./.ci/lumen_cli/cli/lib/common/utils.py_docs.md)
- [.ci/lumen_cli/cli/lib/core/vllm/lib.py](./.ci/lumen_cli/cli/lib/core/vllm/lib.py_docs.md)
- [.ci/lumen_cli/cli/lib/core/vllm/vllm_build.py](./.ci/lumen_cli/cli/lib/core/vllm/vllm_build.py_docs.md)
- [.ci/lumen_cli/cli/lib/core/vllm/vllm_test.py](./.ci/lumen_cli/cli/lib/core/vllm/vllm_test.py_docs.md)
- [.ci/lumen_cli/cli/run.py](./.ci/lumen_cli/cli/run.py_docs.md)
- [.ci/lumen_cli/cli/test_cli/__init__.py](./.ci/lumen_cli/cli/test_cli/__init__.py_docs.md)
- [.ci/lumen_cli/cli/test_cli/register_test.py](./.ci/lumen_cli/cli/test_cli/register_test.py_docs.md)
- [.ci/lumen_cli/pyproject.toml](./.ci/lumen_cli/pyproject.toml_docs.md)
- [.ci/lumen_cli/tests/test_app.py](./.ci/lumen_cli/tests/test_app.py_docs.md)
- [.ci/lumen_cli/tests/test_cli_helper.py](./.ci/lumen_cli/tests/test_cli_helper.py_docs.md)
- [.ci/lumen_cli/tests/test_docker_helper.py](./.ci/lumen_cli/tests/test_docker_helper.py_docs.md)
- [.ci/lumen_cli/tests/test_envs_helper.py](./.ci/lumen_cli/tests/test_envs_helper.py_docs.md)
- [.ci/lumen_cli/tests/test_path_helper.py](./.ci/lumen_cli/tests/test_path_helper.py_docs.md)
- [.ci/lumen_cli/tests/test_run_plan.py](./.ci/lumen_cli/tests/test_run_plan.py_docs.md)
- [.ci/lumen_cli/tests/test_utils.py](./.ci/lumen_cli/tests/test_utils.py_docs.md)
- [.ci/lumen_cli/tests/test_vllm.py](./.ci/lumen_cli/tests/test_vllm.py_docs.md)
- [.ci/magma-rocm/Makefile](./.ci/magma-rocm/Makefile_docs.md)
- [.ci/magma-rocm/README.md](./.ci/magma-rocm/README.md_docs.md)
- [.ci/magma-rocm/build_magma.sh](./.ci/magma-rocm/build_magma.sh_docs.md)
- [.ci/magma-rocm/package_files/build.sh](./.ci/magma-rocm/package_files/build.sh_docs.md)
- [.ci/magma/Makefile](./.ci/magma/Makefile_docs.md)
- [.ci/magma/README.md](./.ci/magma/README.md_docs.md)
- [.ci/magma/build_magma.sh](./.ci/magma/build_magma.sh_docs.md)
- [.ci/magma/package_files/CMake.patch](./.ci/magma/package_files/CMake.patch_docs.md)
- [.ci/magma/package_files/build.sh](./.ci/magma/package_files/build.sh_docs.md)
- [.ci/magma/package_files/cmakelists.patch](./.ci/magma/package_files/cmakelists.patch_docs.md)
- [.ci/magma/package_files/cuda13.patch](./.ci/magma/package_files/cuda13.patch_docs.md)
- [.ci/magma/package_files/getrf_nbparam.patch](./.ci/magma/package_files/getrf_nbparam.patch_docs.md)
- [.ci/magma/package_files/getrf_shfl.patch](./.ci/magma/package_files/getrf_shfl.patch_docs.md)
- [.ci/magma/package_files/thread_queue.patch](./.ci/magma/package_files/thread_queue.patch_docs.md)
- [.ci/manywheel/LICENSE](./.ci/manywheel/LICENSE_docs.md)
- [.ci/manywheel/build.sh](./.ci/manywheel/build.sh_docs.md)
- [.ci/manywheel/build_common.sh](./.ci/manywheel/build_common.sh_docs.md)
- [.ci/manywheel/build_cpu.sh](./.ci/manywheel/build_cpu.sh_docs.md)
- [.ci/manywheel/build_cuda.sh](./.ci/manywheel/build_cuda.sh_docs.md)
- [.ci/manywheel/build_libtorch.sh](./.ci/manywheel/build_libtorch.sh_docs.md)
- [.ci/manywheel/build_rocm.sh](./.ci/manywheel/build_rocm.sh_docs.md)
- [.ci/manywheel/build_xpu.sh](./.ci/manywheel/build_xpu.sh_docs.md)
- [.ci/manywheel/set_desired_python.sh](./.ci/manywheel/set_desired_python.sh_docs.md)
- [.ci/manywheel/test_wheel.sh](./.ci/manywheel/test_wheel.sh_docs.md)
- [.ci/onnx/README.md](./.ci/onnx/README.md_docs.md)
- [.ci/onnx/common.sh](./.ci/onnx/common.sh_docs.md)
- [.ci/onnx/test.sh](./.ci/onnx/test.sh_docs.md)
- [.ci/pytorch/README.md](./.ci/pytorch/README.md_docs.md)
- [.ci/pytorch/build.sh](./.ci/pytorch/build.sh_docs.md)
- [.ci/pytorch/check_binary.sh](./.ci/pytorch/check_binary.sh_docs.md)
- [.ci/pytorch/codegen-test.sh](./.ci/pytorch/codegen-test.sh_docs.md)
- [.ci/pytorch/common-build.sh](./.ci/pytorch/common-build.sh_docs.md)
- [.ci/pytorch/common.sh](./.ci/pytorch/common.sh_docs.md)
- [.ci/pytorch/common_utils.sh](./.ci/pytorch/common_utils.sh_docs.md)
- [.ci/pytorch/cpp_doc_push_script.sh](./.ci/pytorch/cpp_doc_push_script.sh_docs.md)
- [.ci/pytorch/docker-build-test.sh](./.ci/pytorch/docker-build-test.sh_docs.md)
- [.ci/pytorch/docs-test.sh](./.ci/pytorch/docs-test.sh_docs.md)
- [.ci/pytorch/fake_numpy/numpy.py](./.ci/pytorch/fake_numpy/numpy.py_docs.md)
- [.ci/pytorch/install_cache_xla.sh](./.ci/pytorch/install_cache_xla.sh_docs.md)
- [.ci/pytorch/macos-build-test.sh](./.ci/pytorch/macos-build-test.sh_docs.md)
- [.ci/pytorch/macos-build.sh](./.ci/pytorch/macos-build.sh_docs.md)
- [.ci/pytorch/macos-common.sh](./.ci/pytorch/macos-common.sh_docs.md)
- [.ci/pytorch/macos-test.sh](./.ci/pytorch/macos-test.sh_docs.md)
- [.ci/pytorch/multigpu-test.sh](./.ci/pytorch/multigpu-test.sh_docs.md)
- [.ci/pytorch/numba-cuda-13.patch](./.ci/pytorch/numba-cuda-13.patch_docs.md)
- [.ci/pytorch/print_sccache_log.py](./.ci/pytorch/print_sccache_log.py_docs.md)
- [.ci/pytorch/python_doc_push_script.sh](./.ci/pytorch/python_doc_push_script.sh_docs.md)
- [.ci/pytorch/run_tests.sh](./.ci/pytorch/run_tests.sh_docs.md)
- [.ci/pytorch/smoke_test/check_binary_symbols.py](./.ci/pytorch/smoke_test/check_binary_symbols.py_docs.md)
- [.ci/pytorch/smoke_test/check_gomp.py](./.ci/pytorch/smoke_test/check_gomp.py_docs.md)
- [.ci/pytorch/smoke_test/max_autotune.py](./.ci/pytorch/smoke_test/max_autotune.py_docs.md)
- [.ci/pytorch/smoke_test/smoke_test.py](./.ci/pytorch/smoke_test/smoke_test.py_docs.md)
- [.ci/pytorch/test.sh](./.ci/pytorch/test.sh_docs.md)
- [.ci/pytorch/test_example_code/CMakeLists.txt](./.ci/pytorch/test_example_code/CMakeLists.txt_docs.md)
- [.ci/pytorch/test_example_code/check-torch-cuda.cpp](./.ci/pytorch/test_example_code/check-torch-cuda.cpp_docs.md)
- [.ci/pytorch/test_example_code/check-torch-mkl.cpp](./.ci/pytorch/test_example_code/check-torch-mkl.cpp_docs.md)
- [.ci/pytorch/test_example_code/check-torch-xnnpack.cpp](./.ci/pytorch/test_example_code/check-torch-xnnpack.cpp_docs.md)
- [.ci/pytorch/test_example_code/cnn_smoke.py](./.ci/pytorch/test_example_code/cnn_smoke.py_docs.md)
- [.ci/pytorch/test_example_code/cnn_smoke_win_arm64.py](./.ci/pytorch/test_example_code/cnn_smoke_win_arm64.py_docs.md)
- [.ci/pytorch/test_example_code/rnn_smoke.py](./.ci/pytorch/test_example_code/rnn_smoke.py_docs.md)
- [.ci/pytorch/test_example_code/rnn_smoke_win_arm64.py](./.ci/pytorch/test_example_code/rnn_smoke_win_arm64.py_docs.md)
- [.ci/pytorch/test_example_code/simple-torch-test.cpp](./.ci/pytorch/test_example_code/simple-torch-test.cpp_docs.md)
- [.ci/pytorch/test_fa3_abi_stable.sh](./.ci/pytorch/test_fa3_abi_stable.sh_docs.md)
- [.ci/pytorch/win-arm64-test.sh](./.ci/pytorch/win-arm64-test.sh_docs.md)
- [.ci/pytorch/win-build.sh](./.ci/pytorch/win-build.sh_docs.md)
- [.ci/pytorch/win-test-helpers/run_python_nn_smoketests.py](./.ci/pytorch/win-test-helpers/run_python_nn_smoketests.py_docs.md)
- [.ci/pytorch/win-test.sh](./.ci/pytorch/win-test.sh_docs.md)
- [.ci/wheel/build_wheel.sh](./.ci/wheel/build_wheel.sh_docs.md)
- [.circleci/README.md](./.circleci/README.md_docs.md)
- [.circleci/codegen_validation/compare_normalized_yaml.sh](./.circleci/codegen_validation/compare_normalized_yaml.sh_docs.md)
- [.circleci/codegen_validation/normalize_yaml_fragment.py](./.circleci/codegen_validation/normalize_yaml_fragment.py_docs.md)
- [.circleci/codegen_validation/overwrite_with_normalized.sh](./.circleci/codegen_validation/overwrite_with_normalized.sh_docs.md)
- [.circleci/scripts/README.md](./.circleci/scripts/README.md_docs.md)
- [.circleci/scripts/binary_linux_test.sh](./.circleci/scripts/binary_linux_test.sh_docs.md)
- [.circleci/scripts/binary_populate_env.sh](./.circleci/scripts/binary_populate_env.sh_docs.md)
- [.circleci/scripts/binary_upload.sh](./.circleci/scripts/binary_upload.sh_docs.md)
- [.circleci/scripts/binary_windows_build.sh](./.circleci/scripts/binary_windows_build.sh_docs.md)
- [.circleci/scripts/binary_windows_test.sh](./.circleci/scripts/binary_windows_test.sh_docs.md)
- [.circleci/scripts/publish_android_snapshot.sh](./.circleci/scripts/publish_android_snapshot.sh_docs.md)
- [.circleci/windows-jni/include/jni.h](./.circleci/windows-jni/include/jni.h_docs.md)
- [.claude/skills/add-uint-support/SKILL.md](./.claude/skills/add-uint-support/SKILL.md_docs.md)
- [.claude/skills/at-dispatch-v2/SKILL.md](./.claude/skills/at-dispatch-v2/SKILL.md_docs.md)
- [.claude/skills/docstring/SKILL.md](./.claude/skills/docstring/SKILL.md_docs.md)
- [.claude/skills/skill-writer/SKILL.md](./.claude/skills/skill-writer/SKILL.md_docs.md)
- [.devcontainer/Dockerfile](./.devcontainer/Dockerfile_docs.md)
- [.devcontainer/README.md](./.devcontainer/README.md_docs.md)
- [.devcontainer/cpu/devcontainer.json](./.devcontainer/cpu/devcontainer.json_docs.md)
- [.devcontainer/cuda/devcontainer.json](./.devcontainer/cuda/devcontainer.json_docs.md)
- [.devcontainer/cuda/requirements.txt](./.devcontainer/cuda/requirements.txt_docs.md)
- [.devcontainer/scripts/install-dev-tools.sh](./.devcontainer/scripts/install-dev-tools.sh_docs.md)
- [.devcontainer/scripts/update_alternatives_clang.sh](./.devcontainer/scripts/update_alternatives_clang.sh_docs.md)
- [.github/ISSUE_TEMPLATE/bug-report.yml](./.github/ISSUE_TEMPLATE/bug-report.yml_docs.md)
- [.github/ISSUE_TEMPLATE/ci-sev.md](./.github/ISSUE_TEMPLATE/ci-sev.md_docs.md)
- [.github/ISSUE_TEMPLATE/config.yml](./.github/ISSUE_TEMPLATE/config.yml_docs.md)
- [.github/ISSUE_TEMPLATE/disable-autorevert.md](./.github/ISSUE_TEMPLATE/disable-autorevert.md_docs.md)
- [.github/ISSUE_TEMPLATE/disable-ci-jobs.md](./.github/ISSUE_TEMPLATE/disable-ci-jobs.md_docs.md)
- [.github/ISSUE_TEMPLATE/documentation.yml](./.github/ISSUE_TEMPLATE/documentation.yml_docs.md)
- [.github/ISSUE_TEMPLATE/feature-request.yml](./.github/ISSUE_TEMPLATE/feature-request.yml_docs.md)
- [.github/ISSUE_TEMPLATE/pt2-bug-report.yml](./.github/ISSUE_TEMPLATE/pt2-bug-report.yml_docs.md)
- [.github/ISSUE_TEMPLATE/release-feature-request.yml](./.github/ISSUE_TEMPLATE/release-feature-request.yml_docs.md)
- [.github/PULL_REQUEST_TEMPLATE.md](./.github/PULL_REQUEST_TEMPLATE.md_docs.md)
- [.github/actionlint.yaml](./.github/actionlint.yaml_docs.md)
- [.github/actions/binary-docker-build/action.yml](./.github/actions/binary-docker-build/action.yml_docs.md)
- [.github/actions/build-external-packages/action.yml](./.github/actions/build-external-packages/action.yml_docs.md)
- [.github/actions/checkout-pytorch/action.yml](./.github/actions/checkout-pytorch/action.yml_docs.md)
- [.github/actions/chown-workspace/action.yml](./.github/actions/chown-workspace/action.yml_docs.md)
- [.github/actions/diskspace-cleanup/action.yml](./.github/actions/diskspace-cleanup/action.yml_docs.md)
- [.github/actions/download-build-artifacts/action.yml](./.github/actions/download-build-artifacts/action.yml_docs.md)
- [.github/actions/download-td-artifacts/action.yml](./.github/actions/download-td-artifacts/action.yml_docs.md)
- [.github/actions/filter-test-configs/action.yml](./.github/actions/filter-test-configs/action.yml_docs.md)
- [.github/actions/get-workflow-job-id/action.yml](./.github/actions/get-workflow-job-id/action.yml_docs.md)
- [.github/actions/linux-test/action.yml](./.github/actions/linux-test/action.yml_docs.md)
- [.github/actions/pytest-cache-download/action.yml](./.github/actions/pytest-cache-download/action.yml_docs.md)
- [.github/actions/pytest-cache-upload/action.yml](./.github/actions/pytest-cache-upload/action.yml_docs.md)
- [.github/actions/reuse-old-whl/action.yml](./.github/actions/reuse-old-whl/action.yml_docs.md)
- [.github/actions/reuse-old-whl/reuse_old_whl.py](./.github/actions/reuse-old-whl/reuse_old_whl.py_docs.md)
- [.github/actions/setup-linux/action.yml](./.github/actions/setup-linux/action.yml_docs.md)
- [.github/actions/setup-rocm/action.yml](./.github/actions/setup-rocm/action.yml_docs.md)
- [.github/actions/setup-win/action.yml](./.github/actions/setup-win/action.yml_docs.md)
- [.github/actions/setup-xpu/action.yml](./.github/actions/setup-xpu/action.yml_docs.md)
- [.github/actions/teardown-rocm/action.yml](./.github/actions/teardown-rocm/action.yml_docs.md)
- [.github/actions/teardown-win/action.yml](./.github/actions/teardown-win/action.yml_docs.md)
- [.github/actions/teardown-xpu/action.yml](./.github/actions/teardown-xpu/action.yml_docs.md)
- [.github/actions/test-pytorch-binary/action.yml](./.github/actions/test-pytorch-binary/action.yml_docs.md)
- [.github/actions/upload-sccache-stats/action.yml](./.github/actions/upload-sccache-stats/action.yml_docs.md)
- [.github/actions/upload-test-artifacts/action.yml](./.github/actions/upload-test-artifacts/action.yml_docs.md)
- [.github/actions/upload-utilization-stats/action.yml](./.github/actions/upload-utilization-stats/action.yml_docs.md)
- [.github/auto_request_review.yml](./.github/auto_request_review.yml_docs.md)
- [.github/ci_commit_pins/audio.txt](./.github/ci_commit_pins/audio.txt_docs.md)
- [.github/ci_commit_pins/data.txt](./.github/ci_commit_pins/data.txt_docs.md)
- [.github/ci_commit_pins/fbgemm.txt](./.github/ci_commit_pins/fbgemm.txt_docs.md)
- [.github/ci_commit_pins/fbgemm_rocm.txt](./.github/ci_commit_pins/fbgemm_rocm.txt_docs.md)
- [.github/ci_commit_pins/multipy.txt](./.github/ci_commit_pins/multipy.txt_docs.md)
- [.github/ci_commit_pins/text.txt](./.github/ci_commit_pins/text.txt_docs.md)
- [.github/ci_commit_pins/torchao.txt](./.github/ci_commit_pins/torchao.txt_docs.md)
- [.github/ci_commit_pins/torchrec.txt](./.github/ci_commit_pins/torchrec.txt_docs.md)
- [.github/ci_commit_pins/triton.txt](./.github/ci_commit_pins/triton.txt_docs.md)
- [.github/ci_commit_pins/vision.txt](./.github/ci_commit_pins/vision.txt_docs.md)
- [.github/ci_commit_pins/vllm.txt](./.github/ci_commit_pins/vllm.txt_docs.md)
- [.github/ci_commit_pins/xla.txt](./.github/ci_commit_pins/xla.txt_docs.md)
- [.github/ci_configs/vllm/Dockerfile](./.github/ci_configs/vllm/Dockerfile_docs.md)
- [.github/ci_configs/vllm/use_existing_torch.py](./.github/ci_configs/vllm/use_existing_torch.py_docs.md)
- [.github/copilot-instructions.md](./.github/copilot-instructions.md_docs.md)
- [.github/dependabot.yml](./.github/dependabot.yml_docs.md)
- [.github/label_to_label.yml](./.github/label_to_label.yml_docs.md)
- [.github/labeler.yml](./.github/labeler.yml_docs.md)
- [.github/merge_rules.yaml](./.github/merge_rules.yaml_docs.md)
- [.github/nitpicks.yml](./.github/nitpicks.yml_docs.md)
- [.github/pytorch-circleci-labels.yml](./.github/pytorch-circleci-labels.yml_docs.md)
- [.github/pytorch-probot.yml](./.github/pytorch-probot.yml_docs.md)
- [.github/regenerate.sh](./.github/regenerate.sh_docs.md)
- [.github/requirements-gha-cache.txt](./.github/requirements-gha-cache.txt_docs.md)
- [.github/requirements/README.md](./.github/requirements/README.md_docs.md)
- [.github/requirements/regenerate-requirements.txt](./.github/requirements/regenerate-requirements.txt_docs.md)
- [.github/scripts/README.md](./.github/scripts/README.md_docs.md)
- [.github/scripts/amd/package_triton_wheel.sh](./.github/scripts/amd/package_triton_wheel.sh_docs.md)
- [.github/scripts/amd/patch_triton_wheel.sh](./.github/scripts/amd/patch_triton_wheel.sh_docs.md)
- [.github/scripts/build_triton_wheel.py](./.github/scripts/build_triton_wheel.py_docs.md)
- [.github/scripts/check_labels.py](./.github/scripts/check_labels.py_docs.md)
- [.github/scripts/cherry_pick.py](./.github/scripts/cherry_pick.py_docs.md)
- [.github/scripts/close_nonexistent_disable_issues.py](./.github/scripts/close_nonexistent_disable_issues.py_docs.md)
- [.github/scripts/collect_ciflow_labels.py](./.github/scripts/collect_ciflow_labels.py_docs.md)
- [.github/scripts/comment_on_pr.py](./.github/scripts/comment_on_pr.py_docs.md)
- [.github/scripts/convert_lintrunner_annotations_to_github.py](./.github/scripts/convert_lintrunner_annotations_to_github.py_docs.md)
- [.github/scripts/delete_old_branches.py](./.github/scripts/delete_old_branches.py_docs.md)
- [.github/scripts/docathon-label-sync.py](./.github/scripts/docathon-label-sync.py_docs.md)
- [.github/scripts/ensure_actions_will_cancel.py](./.github/scripts/ensure_actions_will_cancel.py_docs.md)
- [.github/scripts/export_pytorch_labels.py](./.github/scripts/export_pytorch_labels.py_docs.md)
- [.github/scripts/file_io_utils.py](./.github/scripts/file_io_utils.py_docs.md)
- [.github/scripts/filter_test_configs.py](./.github/scripts/filter_test_configs.py_docs.md)
- [.github/scripts/generate_binary_build_matrix.py](./.github/scripts/generate_binary_build_matrix.py_docs.md)
- [.github/scripts/generate_ci_workflows.py](./.github/scripts/generate_ci_workflows.py_docs.md)
- [.github/scripts/generate_docker_release_matrix.py](./.github/scripts/generate_docker_release_matrix.py_docs.md)
- [.github/scripts/generate_pytorch_version.py](./.github/scripts/generate_pytorch_version.py_docs.md)
- [.github/scripts/get_aws_session_tokens.py](./.github/scripts/get_aws_session_tokens.py_docs.md)
- [.github/scripts/get_ci_variable.py](./.github/scripts/get_ci_variable.py_docs.md)
- [.github/scripts/get_workflow_job_id.py](./.github/scripts/get_workflow_job_id.py_docs.md)
- [.github/scripts/github_utils.py](./.github/scripts/github_utils.py_docs.md)
- [.github/scripts/gitutils.py](./.github/scripts/gitutils.py_docs.md)
- [.github/scripts/label_utils.py](./.github/scripts/label_utils.py_docs.md)
- [.github/scripts/lint_native_functions.py](./.github/scripts/lint_native_functions.py_docs.md)
- [.github/scripts/lintrunner.sh](./.github/scripts/lintrunner.sh_docs.md)
- [.github/scripts/parse_ref.py](./.github/scripts/parse_ref.py_docs.md)
- [.github/scripts/pr-sanity-check.sh](./.github/scripts/pr-sanity-check.sh_docs.md)
- [.github/scripts/prepare_vllm_wheels.sh](./.github/scripts/prepare_vllm_wheels.sh_docs.md)
- [.github/scripts/pytest_cache.py](./.github/scripts/pytest_cache.py_docs.md)
- [.github/scripts/pytest_caching_utils.py](./.github/scripts/pytest_caching_utils.py_docs.md)
- [.github/scripts/report_git_status.sh](./.github/scripts/report_git_status.sh_docs.md)
- [.github/scripts/runner_determinator.py](./.github/scripts/runner_determinator.py_docs.md)
- [.github/scripts/s390x-ci/README.md](./.github/scripts/s390x-ci/README.md_docs.md)
- [.github/scripts/s390x-ci/self-hosted-builder/helpers/app_token.sh](./.github/scripts/s390x-ci/self-hosted-builder/helpers/app_token.sh_docs.md)
- [.github/scripts/s390x-ci/self-hosted-builder/helpers/gh_cat_token.sh](./.github/scripts/s390x-ci/self-hosted-builder/helpers/gh_cat_token.sh_docs.md)
- [.github/scripts/s390x-ci/self-hosted-builder/helpers/gh_token_generator.sh](./.github/scripts/s390x-ci/self-hosted-builder/helpers/gh_token_generator.sh_docs.md)
- [.github/scripts/s390x-ci/self-hosted-builder/podman-patches/podman-25102-backport.patch](./.github/scripts/s390x-ci/self-hosted-builder/podman-patches/podman-25102-backport.patch_docs.md)
- [.github/scripts/s390x-ci/self-hosted-builder/podman-patches/podman-25245.patch](./.github/scripts/s390x-ci/self-hosted-builder/podman-patches/podman-25245.patch_docs.md)
- [.github/scripts/stop_runner_service.sh](./.github/scripts/stop_runner_service.sh_docs.md)
- [.github/scripts/td_llm_indexer.sh](./.github/scripts/td_llm_indexer.sh_docs.md)
- [.github/scripts/test_check_labels.py](./.github/scripts/test_check_labels.py_docs.md)
- [.github/scripts/test_delete_old_branches.py](./.github/scripts/test_delete_old_branches.py_docs.md)
- [.github/scripts/test_filter_test_configs.py](./.github/scripts/test_filter_test_configs.py_docs.md)
- [.github/scripts/test_gitutils.py](./.github/scripts/test_gitutils.py_docs.md)
- [.github/scripts/test_label_utils.py](./.github/scripts/test_label_utils.py_docs.md)
- [.github/scripts/test_pytest_caching_utils.py](./.github/scripts/test_pytest_caching_utils.py_docs.md)
- [.github/scripts/test_runner_determinator.py](./.github/scripts/test_runner_determinator.py_docs.md)
- [.github/scripts/test_trymerge.py](./.github/scripts/test_trymerge.py_docs.md)
- [.github/scripts/test_tryrebase.py](./.github/scripts/test_tryrebase.py_docs.md)
- [.github/scripts/trymerge.py](./.github/scripts/trymerge.py_docs.md)
- [.github/scripts/trymerge_explainer.py](./.github/scripts/trymerge_explainer.py_docs.md)
- [.github/scripts/tryrebase.py](./.github/scripts/tryrebase.py_docs.md)
- [.github/scripts/update_runner_determinator.py](./.github/scripts/update_runner_determinator.py_docs.md)
- [.github/scripts/upload_aws_ossci.sh](./.github/scripts/upload_aws_ossci.sh_docs.md)
- [.github/workflows/_bazel-build-test.yml](./.github/workflows/_bazel-build-test.yml_docs.md)
- [.github/workflows/_binary-build-linux.yml](./.github/workflows/_binary-build-linux.yml_docs.md)
- [.github/workflows/_binary-test-linux.yml](./.github/workflows/_binary-test-linux.yml_docs.md)
- [.github/workflows/_binary-upload.yml](./.github/workflows/_binary-upload.yml_docs.md)
- [.github/workflows/_docs.yml](./.github/workflows/_docs.yml_docs.md)
- [.github/workflows/_get-changed-files.yml](./.github/workflows/_get-changed-files.yml_docs.md)
- [.github/workflows/_link_check.yml](./.github/workflows/_link_check.yml_docs.md)
- [.github/workflows/_linux-build.yml](./.github/workflows/_linux-build.yml_docs.md)
- [.github/workflows/_linux-test-stable-fa3.yml](./.github/workflows/_linux-test-stable-fa3.yml_docs.md)
- [.github/workflows/_linux-test.yml](./.github/workflows/_linux-test.yml_docs.md)
- [.github/workflows/_mac-build.yml](./.github/workflows/_mac-build.yml_docs.md)
- [.github/workflows/_mac-test.yml](./.github/workflows/_mac-test.yml_docs.md)
- [.github/workflows/_rocm-test.yml](./.github/workflows/_rocm-test.yml_docs.md)
- [.github/workflows/_runner-determinator.yml](./.github/workflows/_runner-determinator.yml_docs.md)
- [.github/workflows/_win-build.yml](./.github/workflows/_win-build.yml_docs.md)
- [.github/workflows/_win-test.yml](./.github/workflows/_win-test.yml_docs.md)
- [.github/workflows/_xpu-test.yml](./.github/workflows/_xpu-test.yml_docs.md)
- [.github/workflows/assigntome-docathon.yml](./.github/workflows/assigntome-docathon.yml_docs.md)
- [.github/workflows/attention_op_microbenchmark.yml](./.github/workflows/attention_op_microbenchmark.yml_docs.md)
- [.github/workflows/auto_request_review.yml](./.github/workflows/auto_request_review.yml_docs.md)
- [.github/workflows/b200-distributed.yml](./.github/workflows/b200-distributed.yml_docs.md)
- [.github/workflows/b200-symm-mem.yml](./.github/workflows/b200-symm-mem.yml_docs.md)
- [.github/workflows/build-almalinux-images.yml](./.github/workflows/build-almalinux-images.yml_docs.md)
- [.github/workflows/build-libtorch-images.yml](./.github/workflows/build-libtorch-images.yml_docs.md)
- [.github/workflows/build-magma-linux.yml](./.github/workflows/build-magma-linux.yml_docs.md)
- [.github/workflows/build-magma-rocm-linux.yml](./.github/workflows/build-magma-rocm-linux.yml_docs.md)
- [.github/workflows/build-magma-windows.yml](./.github/workflows/build-magma-windows.yml_docs.md)
- [.github/workflows/build-manywheel-images-s390x.yml](./.github/workflows/build-manywheel-images-s390x.yml_docs.md)
- [.github/workflows/build-manywheel-images.yml](./.github/workflows/build-manywheel-images.yml_docs.md)
- [.github/workflows/build-triton-wheel.yml](./.github/workflows/build-triton-wheel.yml_docs.md)
- [.github/workflows/build-vllm-wheel.yml](./.github/workflows/build-vllm-wheel.yml_docs.md)
- [.github/workflows/check-labels.yml](./.github/workflows/check-labels.yml_docs.md)
- [.github/workflows/check_mergeability_ghstack.yml](./.github/workflows/check_mergeability_ghstack.yml_docs.md)
- [.github/workflows/cherry-pick.yml](./.github/workflows/cherry-pick.yml_docs.md)
- [.github/workflows/close-nonexistent-disable-issues.yml](./.github/workflows/close-nonexistent-disable-issues.yml_docs.md)
- [.github/workflows/create_release.yml](./.github/workflows/create_release.yml_docs.md)
- [.github/workflows/delete_old_branches.yml](./.github/workflows/delete_old_branches.yml_docs.md)
- [.github/workflows/docathon-sync-label.yml](./.github/workflows/docathon-sync-label.yml_docs.md)
- [.github/workflows/docker-builds.yml](./.github/workflows/docker-builds.yml_docs.md)
- [.github/workflows/docker-cache-rocm.yml](./.github/workflows/docker-cache-rocm.yml_docs.md)
- [.github/workflows/docker-release.yml](./.github/workflows/docker-release.yml_docs.md)
- [.github/workflows/dynamo-unittest.yml](./.github/workflows/dynamo-unittest.yml_docs.md)
- [.github/workflows/generated-linux-aarch64-binary-manywheel-nightly.yml](./.github/workflows/generated-linux-aarch64-binary-manywheel-nightly.yml_docs.md)
- [.github/workflows/generated-linux-binary-libtorch-nightly.yml](./.github/workflows/generated-linux-binary-libtorch-nightly.yml_docs.md)
- [.github/workflows/generated-linux-binary-manywheel-nightly.yml](./.github/workflows/generated-linux-binary-manywheel-nightly.yml_docs.md)
- [.github/workflows/generated-linux-s390x-binary-manywheel-nightly.yml](./.github/workflows/generated-linux-s390x-binary-manywheel-nightly.yml_docs.md)
- [.github/workflows/generated-macos-arm64-binary-libtorch-release-nightly.yml](./.github/workflows/generated-macos-arm64-binary-libtorch-release-nightly.yml_docs.md)
- [.github/workflows/generated-macos-arm64-binary-wheel-nightly.yml](./.github/workflows/generated-macos-arm64-binary-wheel-nightly.yml_docs.md)
- [.github/workflows/generated-windows-arm64-binary-libtorch-debug-nightly.yml](./.github/workflows/generated-windows-arm64-binary-libtorch-debug-nightly.yml_docs.md)
- [.github/workflows/generated-windows-arm64-binary-libtorch-release-nightly.yml](./.github/workflows/generated-windows-arm64-binary-libtorch-release-nightly.yml_docs.md)
- [.github/workflows/generated-windows-arm64-binary-wheel-nightly.yml](./.github/workflows/generated-windows-arm64-binary-wheel-nightly.yml_docs.md)
- [.github/workflows/generated-windows-binary-libtorch-debug-nightly.yml](./.github/workflows/generated-windows-binary-libtorch-debug-nightly.yml_docs.md)
- [.github/workflows/generated-windows-binary-libtorch-release-nightly.yml](./.github/workflows/generated-windows-binary-libtorch-release-nightly.yml_docs.md)
- [.github/workflows/generated-windows-binary-wheel-nightly.yml](./.github/workflows/generated-windows-binary-wheel-nightly.yml_docs.md)
- [.github/workflows/h100-cutlass-backend.yml](./.github/workflows/h100-cutlass-backend.yml_docs.md)
- [.github/workflows/h100-distributed.yml](./.github/workflows/h100-distributed.yml_docs.md)
- [.github/workflows/h100-symm-mem.yml](./.github/workflows/h100-symm-mem.yml_docs.md)
- [.github/workflows/inductor-micro-benchmark-x86.yml](./.github/workflows/inductor-micro-benchmark-x86.yml_docs.md)
- [.github/workflows/inductor-micro-benchmark.yml](./.github/workflows/inductor-micro-benchmark.yml_docs.md)
- [.github/workflows/inductor-nightly.yml](./.github/workflows/inductor-nightly.yml_docs.md)
- [.github/workflows/inductor-perf-compare.yml](./.github/workflows/inductor-perf-compare.yml_docs.md)
- [.github/workflows/inductor-perf-test-b200.yml](./.github/workflows/inductor-perf-test-b200.yml_docs.md)
- [.github/workflows/inductor-perf-test-nightly-aarch64.yml](./.github/workflows/inductor-perf-test-nightly-aarch64.yml_docs.md)
- [.github/workflows/inductor-perf-test-nightly-h100.yml](./.github/workflows/inductor-perf-test-nightly-h100.yml_docs.md)
- [.github/workflows/inductor-perf-test-nightly-macos.yml](./.github/workflows/inductor-perf-test-nightly-macos.yml_docs.md)
- [.github/workflows/inductor-perf-test-nightly-rocm-mi300.yml](./.github/workflows/inductor-perf-test-nightly-rocm-mi300.yml_docs.md)
- [.github/workflows/inductor-perf-test-nightly-rocm-mi355.yml](./.github/workflows/inductor-perf-test-nightly-rocm-mi355.yml_docs.md)
- [.github/workflows/inductor-perf-test-nightly-x86-zen.yml](./.github/workflows/inductor-perf-test-nightly-x86-zen.yml_docs.md)
- [.github/workflows/inductor-perf-test-nightly-x86.yml](./.github/workflows/inductor-perf-test-nightly-x86.yml_docs.md)
- [.github/workflows/inductor-perf-test-nightly-xpu.yml](./.github/workflows/inductor-perf-test-nightly-xpu.yml_docs.md)
- [.github/workflows/inductor-perf-test-nightly.yml](./.github/workflows/inductor-perf-test-nightly.yml_docs.md)
- [.github/workflows/inductor-periodic.yml](./.github/workflows/inductor-periodic.yml_docs.md)
- [.github/workflows/inductor-rocm-mi200.yml](./.github/workflows/inductor-rocm-mi200.yml_docs.md)
- [.github/workflows/inductor-rocm-mi300.yml](./.github/workflows/inductor-rocm-mi300.yml_docs.md)
- [.github/workflows/inductor-unittest.yml](./.github/workflows/inductor-unittest.yml_docs.md)
- [.github/workflows/inductor.yml](./.github/workflows/inductor.yml_docs.md)
- [.github/workflows/lint-autoformat.yml](./.github/workflows/lint-autoformat.yml_docs.md)
- [.github/workflows/lint-bc.yml](./.github/workflows/lint-bc.yml_docs.md)
- [.github/workflows/lint.yml](./.github/workflows/lint.yml_docs.md)
- [.github/workflows/linux-aarch64.yml](./.github/workflows/linux-aarch64.yml_docs.md)
- [.github/workflows/llm_td_retrieval.yml](./.github/workflows/llm_td_retrieval.yml_docs.md)
- [.github/workflows/mac-mps.yml](./.github/workflows/mac-mps.yml_docs.md)
- [.github/workflows/nightly-s3-uploads.yml](./.github/workflows/nightly-s3-uploads.yml_docs.md)
- [.github/workflows/nightly.yml](./.github/workflows/nightly.yml_docs.md)
- [.github/workflows/nitpicker.yml](./.github/workflows/nitpicker.yml_docs.md)
- [.github/workflows/operator_benchmark.yml](./.github/workflows/operator_benchmark.yml_docs.md)
- [.github/workflows/operator_microbenchmark.yml](./.github/workflows/operator_microbenchmark.yml_docs.md)
- [.github/workflows/periodic-rocm-mi200.yml](./.github/workflows/periodic-rocm-mi200.yml_docs.md)
- [.github/workflows/periodic-rocm-mi300.yml](./.github/workflows/periodic-rocm-mi300.yml_docs.md)
- [.github/workflows/periodic.yml](./.github/workflows/periodic.yml_docs.md)
- [.github/workflows/pull.yml](./.github/workflows/pull.yml_docs.md)
- [.github/workflows/quantization-periodic.yml](./.github/workflows/quantization-periodic.yml_docs.md)
- [.github/workflows/revert.yml](./.github/workflows/revert.yml_docs.md)
- [.github/workflows/riscv64.yml](./.github/workflows/riscv64.yml_docs.md)
- [.github/workflows/rocm-mi200.yml](./.github/workflows/rocm-mi200.yml_docs.md)
- [.github/workflows/rocm-mi300.yml](./.github/workflows/rocm-mi300.yml_docs.md)
- [.github/workflows/rocm-mi355.yml](./.github/workflows/rocm-mi355.yml_docs.md)
- [.github/workflows/rocm-navi31.yml](./.github/workflows/rocm-navi31.yml_docs.md)
- [.github/workflows/runner-determinator-validator.yml](./.github/workflows/runner-determinator-validator.yml_docs.md)
- [.github/workflows/runner_determinator_script_sync.yaml](./.github/workflows/runner_determinator_script_sync.yaml_docs.md)
- [.github/workflows/s390.yml](./.github/workflows/s390.yml_docs.md)
- [.github/workflows/s390x-periodic.yml](./.github/workflows/s390x-periodic.yml_docs.md)
- [.github/workflows/scorecards.yml](./.github/workflows/scorecards.yml_docs.md)
- [.github/workflows/slow-rocm-mi200.yml](./.github/workflows/slow-rocm-mi200.yml_docs.md)
- [.github/workflows/slow.yml](./.github/workflows/slow.yml_docs.md)
- [.github/workflows/stale.yml](./.github/workflows/stale.yml_docs.md)
- [.github/workflows/target-determination-indexer.yml](./.github/workflows/target-determination-indexer.yml_docs.md)
- [.github/workflows/target_determination.yml](./.github/workflows/target_determination.yml_docs.md)
- [.github/workflows/test-b200.yml](./.github/workflows/test-b200.yml_docs.md)
- [.github/workflows/test-check-binary.yml](./.github/workflows/test-check-binary.yml_docs.md)
- [.github/workflows/test-h100.yml](./.github/workflows/test-h100.yml_docs.md)
- [.github/workflows/tools-unit-tests.yml](./.github/workflows/tools-unit-tests.yml_docs.md)
- [.github/workflows/torchbench.yml](./.github/workflows/torchbench.yml_docs.md)
- [.github/workflows/trunk-rocm-mi300.yml](./.github/workflows/trunk-rocm-mi300.yml_docs.md)
- [.github/workflows/trunk-tagging.yml](./.github/workflows/trunk-tagging.yml_docs.md)
- [.github/workflows/trunk.yml](./.github/workflows/trunk.yml_docs.md)
- [.github/workflows/trymerge.yml](./.github/workflows/trymerge.yml_docs.md)
- [.github/workflows/tryrebase.yml](./.github/workflows/tryrebase.yml_docs.md)
- [.github/workflows/unstable-periodic.yml](./.github/workflows/unstable-periodic.yml_docs.md)
- [.github/workflows/unstable.yml](./.github/workflows/unstable.yml_docs.md)
- [.github/workflows/update-viablestrict.yml](./.github/workflows/update-viablestrict.yml_docs.md)
- [.github/workflows/update_pytorch_labels.yml](./.github/workflows/update_pytorch_labels.yml_docs.md)
- [.github/workflows/upload-test-stats-while-running.yml](./.github/workflows/upload-test-stats-while-running.yml_docs.md)
- [.github/workflows/upload-test-stats.yml](./.github/workflows/upload-test-stats.yml_docs.md)
- [.github/workflows/upload-torch-dynamo-perf-stats.yml](./.github/workflows/upload-torch-dynamo-perf-stats.yml_docs.md)
- [.github/workflows/upload_test_stats_intermediate.yml](./.github/workflows/upload_test_stats_intermediate.yml_docs.md)
- [.github/workflows/vllm.yml](./.github/workflows/vllm.yml_docs.md)
- [.github/workflows/weekly.yml](./.github/workflows/weekly.yml_docs.md)
- [.github/workflows/win-arm64-build-test.yml](./.github/workflows/win-arm64-build-test.yml_docs.md)
- [.github/workflows/xpu.yml](./.github/workflows/xpu.yml_docs.md)
- [.lintrunner.toml](.//.lintrunner.toml_docs.md)
- [.spin/cmds.py](./.spin/cmds.py_docs.md)
- [.vscode/extensions.json](./.vscode/extensions.json_docs.md)
- [.vscode/settings_recommended.json](./.vscode/settings_recommended.json_docs.md)
- [AGENTS.md](.//AGENTS.md_docs.md)
- [BUILD.bazel](.//BUILD.bazel_docs.md)
- [CLAUDE.md](.//CLAUDE.md_docs.md)
- [CMakeLists.txt](.//CMakeLists.txt_docs.md)
- [CODE_OF_CONDUCT.md](.//CODE_OF_CONDUCT.md_docs.md)
- [CONTRIBUTING.md](.//CONTRIBUTING.md_docs.md)
- [Dockerfile](.//Dockerfile_docs.md)
- [GLOSSARY.md](.//GLOSSARY.md_docs.md)
- [LICENSE](.//LICENSE_docs.md)
- [Makefile](.//Makefile_docs.md)
- [README.md](.//README.md_docs.md)
- [RELEASE.md](.//RELEASE.md_docs.md)
- [SECURITY.md](.//SECURITY.md_docs.md)
- [android/README.md](./android/README.md_docs.md)
- [android/common.sh](./android/common.sh_docs.md)
- [android/pytorch_android/CMakeLists.txt](./android/pytorch_android/CMakeLists.txt_docs.md)
- [android/pytorch_android/generate_test_asset.cpp](./android/pytorch_android/generate_test_asset.cpp_docs.md)
- [android/pytorch_android/generate_test_torchscripts.py](./android/pytorch_android/generate_test_torchscripts.py_docs.md)
- [android/pytorch_android/src/androidTest/cpp/pytorch_jni_common_test.cpp](./android/pytorch_android/src/androidTest/cpp/pytorch_jni_common_test.cpp_docs.md)
- [android/pytorch_android/src/androidTest/java/org/pytorch/PytorchHostTests.java](./android/pytorch_android/src/androidTest/java/org/pytorch/PytorchHostTests.java_docs.md)
- [android/pytorch_android/src/androidTest/java/org/pytorch/PytorchInstrumentedTests.java](./android/pytorch_android/src/androidTest/java/org/pytorch/PytorchInstrumentedTests.java_docs.md)
- [android/pytorch_android/src/androidTest/java/org/pytorch/PytorchLiteInstrumentedTests.java](./android/pytorch_android/src/androidTest/java/org/pytorch/PytorchLiteInstrumentedTests.java_docs.md)
- [android/pytorch_android/src/androidTest/java/org/pytorch/PytorchTestBase.java](./android/pytorch_android/src/androidTest/java/org/pytorch/PytorchTestBase.java_docs.md)
- [android/pytorch_android/src/androidTest/java/org/pytorch/suite/PytorchInstrumentedTestSuite.java](./android/pytorch_android/src/androidTest/java/org/pytorch/suite/PytorchInstrumentedTestSuite.java_docs.md)
- [android/pytorch_android/src/androidTest/java/org/pytorch/suite/PytorchLiteInstrumentedTestSuite.java](./android/pytorch_android/src/androidTest/java/org/pytorch/suite/PytorchLiteInstrumentedTestSuite.java_docs.md)
- [android/pytorch_android/src/main/cpp/cmake_macros.h](./android/pytorch_android/src/main/cpp/cmake_macros.h_docs.md)
- [android/pytorch_android/src/main/cpp/pytorch_jni_common.cpp](./android/pytorch_android/src/main/cpp/pytorch_jni_common.cpp_docs.md)
- [android/pytorch_android/src/main/cpp/pytorch_jni_common.h](./android/pytorch_android/src/main/cpp/pytorch_jni_common.h_docs.md)
- [android/pytorch_android/src/main/cpp/pytorch_jni_jit.cpp](./android/pytorch_android/src/main/cpp/pytorch_jni_jit.cpp_docs.md)
- [android/pytorch_android/src/main/cpp/pytorch_jni_lite.cpp](./android/pytorch_android/src/main/cpp/pytorch_jni_lite.cpp_docs.md)
- [android/pytorch_android/src/main/java/org/pytorch/DType.java](./android/pytorch_android/src/main/java/org/pytorch/DType.java_docs.md)
- [android/pytorch_android/src/main/java/org/pytorch/Device.java](./android/pytorch_android/src/main/java/org/pytorch/Device.java_docs.md)
- [android/pytorch_android/src/main/java/org/pytorch/INativePeer.java](./android/pytorch_android/src/main/java/org/pytorch/INativePeer.java_docs.md)
- [android/pytorch_android/src/main/java/org/pytorch/IValue.java](./android/pytorch_android/src/main/java/org/pytorch/IValue.java_docs.md)
- [android/pytorch_android/src/main/java/org/pytorch/LiteModuleLoader.java](./android/pytorch_android/src/main/java/org/pytorch/LiteModuleLoader.java_docs.md)
- [android/pytorch_android/src/main/java/org/pytorch/LiteNativePeer.java](./android/pytorch_android/src/main/java/org/pytorch/LiteNativePeer.java_docs.md)

*... and 28187 more files (see index.md for complete list)*


## Appendix B: Glossary

- **Tensor**: Multi-dimensional array, the fundamental data structure in PyTorch
- **Autograd**: Automatic differentiation system
- **ATen**: A Tensor Library, the C++ tensor library
- **C10**: Caffe2 Core, providing fundamental abstractions
- **JIT**: Just-In-Time compilation for optimizing models
- **CUDA**: NVIDIA's parallel computing platform for GPU acceleration

---

*Generated by PyTorch Repository Documentation System*

**Total Files Documented**: 28687
**Total Folders**: 2280
**Documentation Size**: Comprehensive (millions of words across all files)
