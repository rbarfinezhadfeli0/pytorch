# Documentation: `.github/labeler.yml`

## File Metadata

- **Path**: `.github/labeler.yml`
- **Size**: 4,764 bytes (4.65 KB)
- **Type**: YAML Configuration
- **Extension**: `.yml`

## File Purpose

This is a yaml configuration that is part of the PyTorch project.

## Original Source

```yaml
"module: dynamo":
- torch/_dynamo/**
- torch/csrc/dynamo/**
- benchmarks/dynamo/**
- test/dynamo/**

"module: inductor":
- torch/_inductor/**
- test/inductor/**

"ciflow/inductor":
- torch/_decomp/**
- torch/_dynamo/**
- torch/_export/**
- torch/_inductor/**
- benchmarks/dynamo/**
- torch/_subclasses/fake_tensor.py
- torch/_subclasses/fake_utils.py
- torch/_subclasses/meta_utils.py
- test/distributed/test_dynamo_distributed.py
- test/distributed/test_inductor_collectives.py
- torch/_functorch/_aot_autograd/**
- torch/_functorch/aot_autograd.py
- torch/_functorch/partitioners.py
- .ci/docker/ci_commit_pins/**
- .github/ci_commit_pins/**
- c10/core/Sym*
- torch/fx/experimental/symbolic_shapes.py
- torch/fx/experimental/recording.py
- torch/fx/experimental/sym_node.py
- torch/fx/experimental/validator.py
- torch/fx/experimental/proxy_tensor.py
- test/distributed/tensor/test_dtensor_compile.py
- test/distributed/tensor/parallel/test_fsdp_2d_parallel.py
- torch/distributed/tensor/**
- torch/distributed/fsdp/**
- torch/csrc/inductor/**
- torch/csrc/dynamo/**
- test/cpp/aoti_abi_check/**
- test/cpp/aoti_inference/**
- test/inductor/**
- test/dynamo/**

"module: cpu":
- aten/src/ATen/cpu/**
- aten/src/ATen/native/cpu/**
- aten/src/ATen/native/quantized/cpu/**
- aten/src/ATen/native/Convolution*.cpp
- aten/src/ATen/native/mkldnn/**
- torch/cpu/**
- torch/utils/mkldnn.py
- torch/utils/_sympy/**
- test/test_mkldnn.py

"module: mkldnn":
- third_party/ideep
- caffe2/ideep/**
- caffe2/python/ideep/**
- cmake/Modules/FindMKLDNN.cmake
- third_party/mkl-dnn.BUILD
- torch/csrc/jit/codegen/onednn/**
- test/test_jit_llga_fuser.py
- test/test_mkldnn.py

"ciflow/linux-aarch64":
- third_party/ideep
- caffe2/ideep/**
- caffe2/python/ideep/**
- cmake/Modules/FindMKLDNN.cmake
- third_party/mkl-dnn.BUILD
- torch/csrc/jit/codegen/onednn/**
- test/test_jit_llga_fuser.py
- test/test_mkldnn.py

"module: amp (automated mixed precision)":
- torch/amp/**
- aten/src/ATen/autocast_mode.*
- torch/csrc/jit/passes/autocast.cpp
- test/test_autocast.py

"NNC":
- torch/csrc/jit/tensorexpr/**

"release notes: quantization":
- torch/ao/quantization/**
- torch/quantization/**
- aten/src/ATen/quantized/**
- aten/src/ATen/native/quantized/cpu/**
- test/quantization/**

"ciflow/trunk":
- .ci/docker/ci_commit_pins/triton.txt

"oncall: distributed":
- torch/csrc/distributed/**
- torch/distributed/**
- torch/nn/parallel/**
- test/distributed/**
- torch/testing/_internal/distributed/**

"release notes: distributed (checkpoint)":
- torch/distributed/checkpoint/**
- test/distributed/checkpoint/**

"module: compiled autograd":
- torch/csrc/dynamo/python_compiled_autograd.cpp
- torch/csrc/dynamo/compiled_autograd.h
- torch/_dynamo/compiled_autograd.py
- torch/inductor/test_compiled_autograd.py

"ciflow/xpu":
- torch/csrc/inductor/aoti_include/xpu.h
- torch/csrc/inductor/cpp_wrapper/device_internal/xpu.h
- torch/csrc/inductor/cpp_wrapper/xpu.h

"release notes: inductor (aoti)":
- torch/_C/_aoti.pyi
- torch/_dynamo/repro/aoti.py
- torch/_higher_order_ops/aoti_call_delegate.py
- torch/_inductor/codegen/aoti_runtime/**
- torch/_inductor/codegen/aoti_hipify_utils.py
- torch/_inductor/codegen/cpp_wrapper_cpu.py
- torch/_inductor/codegen/cpp_wrapper_gpu.py
- torch/_inductor/aoti_eager.py
- torch/csrc/inductor/aoti_runtime/**
- torch/csrc/inductor/aoti_torch/**
- torch/csrc/inductor/aoti_runner/**
- torch/csrc/inductor/aoti_eager/**
- torch/csrc/inductor/aoti_package/**
- torch/csrc/inductor/aoti_include/**
- torchgen/aoti/**
- torchgen/gen_aoti_c_shim.py

"ciflow/vllm":
- .github/ci_commit_pins/vllm.txt

"ciflow/b200":
- test/test_matmul_cuda.py
- test/test_scaled_matmul_cuda.py
- test/inductor/test_fp8.py
- aten/src/ATen/native/cuda/*Blas.cpp
- aten/src/ATen/cuda/CUDA*Blas.*
- torch/**/*cublas*
- torch/_inductor/kernel/mm.py
- test/inductor/test_max_autotune.py
- third_party/fbgemm

"ciflow/h100":
- test/test_matmul_cuda.py
- test/test_scaled_matmul_cuda.py
- test/inductor/test_fp8.py
- aten/src/ATen/native/cuda/*Blas.cpp
- aten/src/ATen/cuda/CUDA*Blas.*
- torch/**/*cublas*
- torch/_inductor/kernel/mm.py
- test/inductor/test_max_autotune.py
- third_party/fbgemm

"ciflow/rocm":
- test/test_matmul_cuda.py
- test/test_scaled_matmul_cuda.py
- test/inductor/test_fp8.py
- aten/src/ATen/native/cuda/*Blas.cpp
- aten/src/ATen/cuda/CUDA*Blas.*
- torch/_inductor/kernel/mm.py
- test/inductor/test_max_autotune.py
- third_party/fbgemm

"ciflow/mps":
- aten/src/ATen/mps/**
- aten/src/ATen/native/mps/**
- torch/_inductor/codegen/mps.py
- test/test_mps.py
- test/inductor/test_mps_basic.py

"ciflow/h100-symm-mem":
- torch/csrc/distributed/c10d/symm_mem/**
- torch/distributed/_symmetric_memory/**
- test/distributed/**/*mem*
- test/distributed/**/*mem*/**

```



## High-Level Overview

This file is part of the PyTorch framework located at `.github`.

## Detailed Analysis

### Code Structure

This is a configuration file. See the original source for structure.


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `.github`, which is part of the PyTorch project infrastructure.



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

Files in the same folder (`.github`):

- [`auto_request_review.yml_docs.md`](./auto_request_review.yml_docs.md)
- [`label_to_label.yml_docs.md`](./label_to_label.yml_docs.md)
- [`merge_rules.yaml_docs.md`](./merge_rules.yaml_docs.md)
- [`pytorch-circleci-labels.yml_docs.md`](./pytorch-circleci-labels.yml_docs.md)
- [`regenerate.sh_docs.md`](./regenerate.sh_docs.md)
- [`requirements-gha-cache.txt_docs.md`](./requirements-gha-cache.txt_docs.md)
- [`PULL_REQUEST_TEMPLATE.md_docs.md`](./PULL_REQUEST_TEMPLATE.md_docs.md)
- [`pytorch-probot.yml_docs.md`](./pytorch-probot.yml_docs.md)
- [`actionlint.yaml_docs.md`](./actionlint.yaml_docs.md)


## Cross-References

- **File Documentation**: `labeler.yml_docs.md`
- **Keyword Index**: `labeler.yml_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
