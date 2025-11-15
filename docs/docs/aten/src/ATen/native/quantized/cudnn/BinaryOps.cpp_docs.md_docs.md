# Documentation: `docs/aten/src/ATen/native/quantized/cudnn/BinaryOps.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cudnn/BinaryOps.cpp_docs.md`
- **Size**: 14,779 bytes (14.43 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cudnn/BinaryOps.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cudnn/BinaryOps.cpp`
- **Size**: 11,873 bytes (11.59 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#if AT_CUDNN_ENABLED()

#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/native/quantized/cudnn/utils.h>
#include <ATen/native/utils/ParamsHash.h>
#include <ATen/TensorUtils.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/QScheme.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/ArrayRef.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/_empty_affine_quantized.h>
#endif

#include <unordered_map>


namespace at::native {
namespace {
constexpr uint8_t max_num_input_dim = 5;
struct AddParams {
  c10::DeviceIndex device_id;
  int64_t input_a_size[max_num_input_dim];
  int64_t input_b_size[max_num_input_dim];
  uint8_t input_dim; // we currently assume both inputs are given as the same size (i.e., no broadcasting)
  at::MemoryFormat memory_format;
  bool deterministic;
  bool allow_tf32;
};
struct CacheKey {
  AddParams params;
  uint8_t input_a_alignment;
  uint8_t input_b_alignment;
  uint8_t output_alignment;
  bool kReluFused;
};
void setAddParams(
    AddParams* params, const at::Tensor& input_a, const at::Tensor& input_b,
    bool deterministic, bool allow_tf32) {
  memset(params, 0, sizeof(AddParams));
  params->device_id = at::cuda::current_device();
  params->input_dim = input_a.dim();
  params->memory_format = input_a.suggest_memory_format();
  for (int i = 0; i < params->input_dim; ++i) {
    params->input_a_size[i] = input_a.sizes()[i];
    params->input_b_size[i] = input_b.sizes()[i];
  }
  params->deterministic = deterministic;
  params->allow_tf32 = allow_tf32;
}
// FIXME: make this thread-safe by reusing the benchmark cache in Conv_v7.cpp
// we currently set the maximum number of input dimensions to 5
// this can be increased, if necessary
std::unordered_map<CacheKey, cudnn_frontend::ManagedOpaqueDescriptor, at::native::ParamsHash<CacheKey>, at::native::ParamsEqual<CacheKey>> execution_plan_cache;

// TODO: this is also in BinaryOps.cpp and some other cpp files in quantized/cpu/. I think we should
// move everything into a utilities file in quantized/ directory later.
inline void check_inputs(const Tensor& qa, const Tensor& qb) {
  TORCH_CHECK(
      qa.qscheme() == kPerTensorAffine,
      "Only per tensor quantization is supported in Add.");
  TORCH_CHECK(
      qa.qscheme() == qb.qscheme(),
      "Both inputs to Add must have the same quantization scheme.");
  TORCH_CHECK(
      qa.scalar_type() == qb.scalar_type(),
      "Add operands should have same data type.");
}

// currently we only support int8 symmetric (zero_point = 0 for inputs and output) quantized add
// We implement relu ( (a_int8 + b_int8 * ( b_scale/a_scale) ) ) * ( a_scale / out_scale )
// which requires 4 cudnn ops (2 multiplication, 1 addition, and 1 relu ops)
// Multiplication ops: rhs_mult_op, requant_op
// Addition op: add_op
// Relu op: relu_op
template <bool kReluFused = false>
Tensor add(Tensor qa, Tensor qb, double output_scale, int64_t output_zero_point) {
  if (qa.numel() == 0) {
    return Tensor{};
  }
  // TODO: add shape checking when broadcasted add is supported. For now we assume the input tensors are the same shape
  TORCH_CHECK(qa.sizes() == qb.sizes(), "Quantized cudnn add currently expects both input tensors to be the same shape");

  check_inputs(qa, qb);

  // cudnn expects tensors to be at least 3D. So we will prepend dummy dimensions if the input tensors are not at least 3D
  auto orig_sizes = qa.sizes().vec();
  if (qa.dim() < 3) {
    std::vector<int64_t> new_sizes(3, 1);
    // cudnn expects leading dimensions to be the dummy dimensions
    new_sizes.back() = qa.sizes().back();
    if (qa.dim() == 2) {
      new_sizes[1] = qa.size(0);
    }
    qa = qa.view(new_sizes);
    qb = qb.view(new_sizes);
  } else if (qa.dim() == 4) {
    qa = qa.contiguous(c10::MemoryFormat::ChannelsLast);
    qb = qb.contiguous(c10::MemoryFormat::ChannelsLast);
  }

  auto memory_format = qa.dim() == 4 ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;
  at::Tensor add_output = at::empty(qa.sizes(), at::device(at::kCUDA).dtype(at::kFloat), memory_format);
  at::Tensor quantized_output = at::_empty_affine_quantized(qa.sizes(), at::device(at::kCUDA).dtype(at::ScalarType::QInt8),
                                                            output_scale, output_zero_point, memory_format);
  double requantize_multiplier = qa.q_scale() / output_scale;
  at::Tensor requantize_multiplier_tensor = cudnn_utils::getRequantMultiplierTensor(requantize_multiplier, quantized_output.dim());
  at::Tensor rhs_multiplier_tensor = at::empty(quantized_output.sizes(), at::device(at::kCUDA).dtype(at::kFloat), memory_format);
  rhs_multiplier_tensor.fill_(qb.q_scale() / qa.q_scale());

  cudnnHandle_t handle = at::native::getCudnnHandle();
  CacheKey key{};
  // memset is needed here because there is implicit packing added for CacheKey, and this can result in uninitialized padded values that are
  // used for hashing (see how at::native::ParamsHash is defined). Without memset, we can potentially come across a situation where two
  // CacheKey objects have the same user defined parameters, but
  // different padded values, resulting in different hash outputs.
  memset(&key, 0, sizeof(key));
  bool deterministic{true};
  bool allow_tf32{false};
  setAddParams(&key.params, qa, qb, deterministic, allow_tf32);
  key.kReluFused = kReluFused;
  key.input_a_alignment = cudnn_utils::getAlignment(qa);
  key.input_b_alignment = cudnn_utils::getAlignment(qb);
  key.output_alignment = cudnn_utils::getAlignment(add_output);

  auto run = [&](const cudnn_frontend::ManagedOpaqueDescriptor& plan_desc) {
    auto workspace_size = 0;
    auto workspace = at::empty({workspace_size}, qa.options().dtype(at::kByte));
    std::vector<void *> data_ptrs;
    std::vector<int64_t> uids;
    data_ptrs.reserve(8);
    uids.reserve(8);
    data_ptrs = {qb.data_ptr<int8_t>(), rhs_multiplier_tensor.data_ptr(), add_output.data_ptr(),
                 qa.data_ptr<int8_t>(), add_output.data_ptr(), requantize_multiplier_tensor.data_ptr(),
                 quantized_output.data_ptr<int8_t>()};
    uids = {'b', 'm', 'c', 'a', 'p', 'r', 'q'};
    if constexpr (kReluFused) {
        data_ptrs.emplace_back(add_output.data_ptr()),
        uids.emplace_back('f');
    }

    auto variantPack = cudnn_frontend::VariantPackBuilder()
      .setWorkspacePointer(workspace.data_ptr())
      .setDataPointers(static_cast<int64_t>(uids.size()), data_ptrs.data())
      .setUids(static_cast<int64_t>(uids.size()), uids.data())
      .build();
    auto variant_pack_desc = variantPack.get_raw_desc();
    AT_CUDNN_CHECK(cudnnBackendExecute(handle, plan_desc->get_backend_descriptor(), variant_pack_desc));
  };

  auto search = execution_plan_cache.find(key);
  if (search != execution_plan_cache.end()) {
    cudnn_frontend::ManagedOpaqueDescriptor plan_desc = search->second;
    run(plan_desc);
    return quantized_output.view(orig_sizes);
  }

  // computes qb_int8 * ( qb_scale/qa_scale )
  auto rhs_mult_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      .setxDesc(cudnn_utils::getTensorDescriptor(qb.sizes(), qb.strides(), CUDNN_DATA_INT8, 'b', key.input_b_alignment))
      .setbDesc(cudnn_utils::getTensorDescriptor(rhs_multiplier_tensor, 'm', cudnn_utils::getAlignment(rhs_multiplier_tensor)))
      .setyDesc(cudnn_utils::getTensorDescriptor(add_output, 'c', key.output_alignment))
      .setpwDesc(cudnn_utils::getPointWiseMulDescriptor(at::native::getCudnnDataType(add_output)))
      .build();

  // add_op computes (qa_int8 + qb_int8 * ( qb_scale/qa_scale ) )
  // add_output is a fp32 tensor for accumulation purposes
  auto add_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      .setxDesc(rhs_mult_op.getOutputTensor())
      .setbDesc(cudnn_utils::getTensorDescriptor(qa.sizes(), qa.strides(), CUDNN_DATA_INT8, 'a', key.input_a_alignment))
      .setyDesc(cudnn_utils::getTensorDescriptor(add_output, 'p', key.output_alignment))
      .setpwDesc(cudnn_utils::getPointWiseAddDescriptor(at::native::getCudnnDataType(add_output)))
      .build();

  // relu_op computes
  // relu( (qa_int8 + qb_int8 * ( qb_scale/qa_scale ) )  )
  // output is a fp32 tensor
  std::optional<cudnn_frontend::Operation> relu_op;
  if constexpr (kReluFused) {
    // we use inplace operation here where the output is assigned to the input
    relu_op.emplace(cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
      .setxDesc(add_op.getOutputTensor())
      .setyDesc(cudnn_utils::getTensorDescriptor(add_output, 'f', key.output_alignment))
      .setpwDesc(cudnn_utils::getPointWiseReluDescriptor(at::native::getCudnnDataType(add_output)))
      .build());
  }

  // requant_op computes
  // (a_int8 + b_int8 * ( b_scale/a_scale) ) * a_scale / out_scale
  auto requant_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
    .setxDesc(kReluFused ? relu_op.value().getOutputTensor() : add_op.getOutputTensor())
    .setbDesc(cudnn_utils::getTensorDescriptor(requantize_multiplier_tensor, 'r', cudnn_utils::getAlignment(requantize_multiplier_tensor)))
    .setyDesc(cudnn_utils::getTensorDescriptor(quantized_output.sizes(), quantized_output.strides(), CUDNN_DATA_INT8, 'q', cudnn_utils::getAlignment(quantized_output)))
    .setpwDesc(cudnn_utils::getPointWiseMulDescriptor(at::native::getCudnnDataType(requantize_multiplier_tensor)))
    .build();

  std::vector<cudnn_frontend::Operation const *> ops{&rhs_mult_op, &add_op};
  if constexpr (kReluFused) {
    ops.emplace_back(&(relu_op.value()));
  }
  ops.emplace_back(&requant_op);

  auto opGraph = cudnn_frontend::OperationGraphBuilder()
      .setHandle(handle)
      .setOperationGraph(static_cast<int64_t>(ops.size()), ops.data())
      .build();
  // std::cout << "opGraph: " << opGraph.describe() << std::endl;

  auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
      .setOperationGraph(opGraph)
      .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
      .build();
  auto fallback = cudnn_frontend::EngineFallbackListBuilder()
                    .setOperationGraph(opGraph)
                    .setOperation(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                    .build();

  auto& engine_configs = heuristics.getEngineConfig(heuristics.getEngineConfigCount());
  auto& fallback_list = fallback.getFallbackList();

  cudnn_frontend::EngineConfigList filtered_configs;
  cudnn_utils::filterEngineConfigs(engine_configs, filtered_configs, deterministic, allow_tf32, at::kChar);
  cudnn_utils::filterEngineConfigs(fallback_list, filtered_configs, deterministic, allow_tf32, at::kChar);
  for (auto &cfg : engine_configs) {
    try {
      auto plan = cudnn_frontend::ExecutionPlanBuilder()
        .setHandle(handle)
        .setEngineConfig(cfg)
        .build();
      auto plan_desc = plan.get_desc();
      run(plan_desc);
      execution_plan_cache[key] = plan_desc;
      return quantized_output.view(orig_sizes);
    } catch (cudnn_frontend::cudnnException &e) {std::cout << "cudnn error:" << e.what() << '\n';} catch(c10::CuDNNError &e) { std::cout << "other error" << e.what() << '\n';}
  }

  TORCH_CHECK(false, "Unable to find an engine to execute this computation in Quantized Add Cudnn");
}

TORCH_LIBRARY_IMPL(quantized, QuantizedCUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::add"), TORCH_FN(add</*ReLUFused=*/false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu"), TORCH_FN(add</*ReLUFused=*/true>));
}

} // namespace
} // namespace at::native

#endif  // AT_CUDNN_ENABLED
#endif  // USE_CUDA

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `AddParams`, `CacheKey`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cudnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/cuda/CUDAConfig.h`
- `ATen/core/TensorBase.h`
- `ATen/core/TensorBody.h`
- `ATen/cuda/Exceptions.h`
- `ATen/cudnn/Handle.h`
- `ATen/native/quantized/cudnn/utils.h`
- `ATen/native/utils/ParamsHash.h`
- `ATen/TensorUtils.h`
- `c10/core/MemoryFormat.h`
- `c10/core/QScheme.h`
- `c10/cuda/CUDAFunctions.h`
- `c10/util/ArrayRef.h`
- `torch/library.h`
- `ATen/Functions.h`
- `ATen/ops/empty.h`
- `ATen/ops/_empty_affine_quantized.h`
- `unordered_map`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

Files in the same folder (`aten/src/ATen/native/quantized/cudnn`):

- [`LinearUnpackImpl.cpp_docs.md`](./LinearUnpackImpl.cpp_docs.md)
- [`ConvUnpackImpl.cpp_docs.md`](./ConvUnpackImpl.cpp_docs.md)
- [`Pooling.cpp_docs.md`](./Pooling.cpp_docs.md)
- [`Linear.cpp_docs.md`](./Linear.cpp_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)
- [`LinearPrepack.cpp_docs.md`](./LinearPrepack.cpp_docs.md)
- [`ConvPrepack.cpp_docs.md`](./ConvPrepack.cpp_docs.md)
- [`Conv.cpp_docs.md`](./Conv.cpp_docs.md)


## Cross-References

- **File Documentation**: `BinaryOps.cpp_docs.md`
- **Keyword Index**: `BinaryOps.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/quantized/cudnn`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/quantized/cudnn`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/quantized/cudnn`):

- [`LinearUnpackImpl.cpp_kw.md_docs.md`](./LinearUnpackImpl.cpp_kw.md_docs.md)
- [`ConvPrepack.cpp_kw.md_docs.md`](./ConvPrepack.cpp_kw.md_docs.md)
- [`LinearUnpackImpl.cpp_docs.md_docs.md`](./LinearUnpackImpl.cpp_docs.md_docs.md)
- [`utils.h_docs.md_docs.md`](./utils.h_docs.md_docs.md)
- [`ConvUnpackImpl.cpp_docs.md_docs.md`](./ConvUnpackImpl.cpp_docs.md_docs.md)
- [`Conv.cpp_docs.md_docs.md`](./Conv.cpp_docs.md_docs.md)
- [`BinaryOps.cpp_kw.md_docs.md`](./BinaryOps.cpp_kw.md_docs.md)
- [`utils.h_kw.md_docs.md`](./utils.h_kw.md_docs.md)
- [`Pooling.cpp_docs.md_docs.md`](./Pooling.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `BinaryOps.cpp_docs.md_docs.md`
- **Keyword Index**: `BinaryOps.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
