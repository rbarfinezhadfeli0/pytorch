# Documentation: `docs/aten/src/ATen/native/quantized/cpu/ACLUtils.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/ACLUtils.h_docs.md`
- **Size**: 11,008 bytes (10.75 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/ACLUtils.h`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/ACLUtils.h`
- **Size**: 7,887 bytes (7.70 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/Config.h>
#if AT_MKLDNN_ACL_ENABLED()

#include <ATen/native/quantized/cpu/OnednnUtils.h>
#include <arm_compute/core/Error.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/function_info/ActivationLayerInfo.h>
#include <arm_compute/runtime/NEON/functions/NEActivationLayer.h>
#include <arm_compute/runtime/NEON/functions/NEArithmeticAddition.h>
#include <arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h>
#include <arm_compute/runtime/NEON/functions/NEQuantizationLayer.h>
#include <arm_compute/runtime/Tensor.h>
#include <array>

// Utilities for Arm Compute Library (ACL) quantized operations
// Provides interfaces to leverage ACL's accelerated kernels for statically and
// dynamically quantized matmuls (i.e. qlinear and qlinear_dynamic) These are
// utalized through PackedLinearWeightsACL which extends
// PackedLinearWeightsOnednn Note that PackedLinearWeightsACL extends rather
// than replaces PackedLinearWeightsOnednn for AArch64 because ACL currently
// only supports per_tensor weight quantization.
namespace at::native::acl_utils {

using QuantMatmulCacheKey = std::tuple<
    int64_t, // M
    bool, // FUSE_RELU
    int64_t, // NUM_THREADS
    double, // INPUT_SCALE
    int64_t, // INPUT_OFFSET
    double, // OUTPUT_SCALE
    int64_t, // OUTPUT_OFFSET
    bool // SIGNED_INPUT
    >;

enum class QuantMatmulCacheKeyIndex {
  M,
  FUSE_RELU,
  NUM_THREADS,
  INPUT_SCALE,
  INPUT_OFFSET,
  OUTPUT_SCALE,
  OUTPUT_OFFSET,
  SIGNED_INPUT
};

// Abstract interface to share common stuff between static/dynamic ACL matmuls.
struct QuantMatmul {
  arm_compute::NEGEMMLowpMatrixMultiplyCore gemm;
  // key for use in the cache
  QuantMatmulCacheKey key;

  QuantMatmul(
      int64_t weight_dim_0,
      int64_t weight_dim_1,
      double weight_scale,
      int64_t weight_offset,
      int8_t* weight_ptr,
      std::optional<float*> bias_ptr,
      const QuantMatmulCacheKey& cache_key);

  virtual ~QuantMatmul();
  virtual arm_compute::Status validate() = 0;
  virtual void configure() = 0;

 protected:
  arm_compute::Tensor wei_q_tensor_;
  std::optional<arm_compute::Tensor> bia_tensor_;
  arm_compute::GEMMInfo gemm_info_;
  std::optional<arm_compute::ActivationLayerInfo> relu_info_;
};

struct DynamicQuantMatmul : public QuantMatmul {
  arm_compute::Tensor src_q_tensor;
  arm_compute::Tensor src_tensor;
  arm_compute::Tensor dst_tensor;
  arm_compute::NEQuantizationLayer quant;
  // We need a ReLU layer here (unlike static quantization) because the ReLU
  // cannot be "truly" fused with the GEMM through gemm_info in ACL dynamically
  // quantized matmuls.
  std::optional<arm_compute::NEActivationLayer> relu;

  DynamicQuantMatmul(
      int64_t weight_dim_0,
      int64_t weight_dim_1,
      double weight_scale,
      int64_t weight_offset,
      int8_t* weight_ptr,
      std::optional<float*> bias_ptr,
      const QuantMatmulCacheKey& cache_key);

  ~DynamicQuantMatmul() override;

  arm_compute::Status validate() override;
  void configure() override;

 private:
  at::Tensor src_q_tensor_orig_;
};

struct StaticQuantMatmul : public QuantMatmul {
  arm_compute::Tensor src_q_tensor;
  arm_compute::Tensor dst_q_tensor;

  StaticQuantMatmul(
      int64_t weight_dim_0,
      int64_t weight_dim_1,
      double weight_scale,
      int64_t weight_offset,
      int8_t* weight_ptr,
      std::optional<float*> bias_ptr,
      const QuantMatmulCacheKey& cache_key);

  ~StaticQuantMatmul() override;

  arm_compute::Status validate() override;
  void configure() override;

 private:
  std::optional<arm_compute::Tensor> bia_q_tensor_;
  std::optional<at::Tensor> bia_q_tensor_orig_;
};

struct QuantAdd {
  arm_compute::Tensor qa_tensor;
  arm_compute::Tensor qb_tensor;
  arm_compute::Tensor qdst_tensor;
  arm_compute::NEArithmeticAddition q_add;

  QuantAdd(
      arm_compute::DataType dtype,
      const std::vector<int64_t>& input_dims,
      double qa_scale,
      int64_t qa_offset,
      double qb_scale,
      int64_t qb_offset,
      double dst_scale,
      int64_t dst_offset);

  arm_compute::Status validate();
  void configure();

 private:
  arm_compute::ConvertPolicy policy{arm_compute::ConvertPolicy::SATURATE};
};

} // namespace at::native::acl_utils
struct PackedLinearWeightsACL : public PackedLinearWeightsOnednn {
  using ACLQuantMatmul = at::native::acl_utils::QuantMatmul;
  using ACLDynamicQuantMatmul = at::native::acl_utils::DynamicQuantMatmul;
  using ACLStaticQuantMatmul = at::native::acl_utils::StaticQuantMatmul;
  using ACLQuantMatmulCacheKey = at::native::acl_utils::QuantMatmulCacheKey;
  using ACLQuantMatmulCacheKeyIndex =
      at::native::acl_utils::QuantMatmulCacheKeyIndex;

  PackedLinearWeightsACL(
      std::unique_ptr<ideep::tensor> weight,
      std::optional<ideep::tensor> bias,
      at::Tensor orig_weight,
      std::optional<at::Tensor> orig_bias);

  at::Tensor apply_dynamic(at::Tensor input, bool reduce_range = false)
      override;
  at::Tensor apply_dynamic_relu(at::Tensor input, bool reduce_range = false)
      override;

  at::Tensor apply(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;
  at::Tensor apply_relu(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point) override;

  template <typename ACLQuantMatmulT>
  std::shared_ptr<ACLQuantMatmulT> get_acl_quant_matmul(
      const ACLQuantMatmulCacheKey& key) {
    return std::dynamic_pointer_cast<ACLQuantMatmulT>(
        fetch_or_create_acl_quant_matmul<ACLQuantMatmulT>(key));
  }

 private:
  int64_t k_;
  int64_t n_;
  int64_t weight_zero_point_;
  double weight_scale_;

  // A 2 element (per layer) cache. Given it's not intended to store more than 2
  // elements, we do not need a fancy implementation. The idea behind it is to
  // allow for a (configuration free) fast path for autoregressive
  // transformer-like models which usually involve 2 input tensor shapes; one
  // for the prefill phase and another for the autoregressive phase
  std::array<std::shared_ptr<ACLQuantMatmul>, 2> cache_;

  template <typename ACLQuantMatmulT>
  std::shared_ptr<ACLQuantMatmul> fetch_or_create_acl_quant_matmul(
      const ACLQuantMatmulCacheKey& key) {
    // We're only maintaining a 2 element LRU cache
    // hit first
    if (cache_[0] != nullptr && cache_[0]->key == key) {
      return cache_[0];
    }
    // hit second
    if (cache_[1] != nullptr && cache_[1]->key == key) {
      // Update LRU
      std::swap(cache_[0], cache_[1]);
      return cache_[0];
    }
    // miss -> replace Least Recently Used - i.e. element at index 1
    cache_[1] = create_acl_quant_matmul<ACLQuantMatmulT>(key);
    std::swap(cache_[0], cache_[1]);
    return cache_[0];
  }

  template <typename ACLQuantMatmulT>
  std::shared_ptr<ACLQuantMatmulT> create_acl_quant_matmul(
      const ACLQuantMatmulCacheKey& key) {
    std::optional<float*> bias_ptr;
    if (bias_.has_value()) {
      bias_ptr = (float*)bias_.value().get_data_handle();
    }
    auto acl_gemm = std::make_shared<ACLQuantMatmulT>(
        k_,
        n_,
        weight_scale_,
        weight_zero_point_,
        (int8_t*)weight_.get()->get_data_handle(),
        bias_ptr,
        key);

    // validate
    auto status = acl_gemm->validate();
    if (status.error_code() != arm_compute::ErrorCode::OK) {
      TORCH_WARN(
          "Arm Compute Library's Quantized Matmul Validation Failed: " +
          status.error_description());
      return nullptr;
    }

    // configure
    acl_gemm->configure();
    return acl_gemm;
  }

  template <bool ReluFused>
  at::Tensor apply_dynamic_impl(at::Tensor input, bool reduce_range = false);

  template <bool ReluFused>
  at::Tensor apply_impl(
      at::Tensor input,
      double output_scale,
      int64_t output_zero_point);
};

#endif // AT_MKLDNN_ACL_ENABLED()

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 22 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `QuantMatmulCacheKeyIndex`, `QuantMatmul`, `DynamicQuantMatmul`, `StaticQuantMatmul`, `QuantAdd`, `PackedLinearWeightsACL`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Config.h`
- `ATen/native/quantized/cpu/OnednnUtils.h`
- `arm_compute/core/Error.h`
- `arm_compute/core/TensorInfo.h`
- `arm_compute/function_info/ActivationLayerInfo.h`
- `arm_compute/runtime/NEON/functions/NEActivationLayer.h`
- `arm_compute/runtime/NEON/functions/NEArithmeticAddition.h`
- `arm_compute/runtime/NEON/functions/NEGEMMLowpMatrixMultiplyCore.h`
- `arm_compute/runtime/NEON/functions/NEQuantizationLayer.h`
- `arm_compute/runtime/Tensor.h`
- `array`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`aten/src/ATen/native/quantized/cpu`):

- [`ACLUtils.cpp_docs.md`](./ACLUtils.cpp_docs.md)
- [`LinearUnpackImpl.cpp_docs.md`](./LinearUnpackImpl.cpp_docs.md)
- [`UpSampleNearest3d.cpp_docs.md`](./UpSampleNearest3d.cpp_docs.md)
- [`Pooling.cpp_docs.md`](./Pooling.cpp_docs.md)
- [`QnnpackUtils.h_docs.md`](./QnnpackUtils.h_docs.md)
- [`qembeddingbag_unpack.cpp_docs.md`](./qembeddingbag_unpack.cpp_docs.md)
- [`fbgemm_utils.h_docs.md`](./fbgemm_utils.h_docs.md)
- [`TensorOperators.cpp_docs.md`](./TensorOperators.cpp_docs.md)
- [`XnnpackUtils.h_docs.md`](./XnnpackUtils.h_docs.md)
- [`qconv_dynamic.cpp_docs.md`](./qconv_dynamic.cpp_docs.md)


## Cross-References

- **File Documentation**: `ACLUtils.h_docs.md`
- **Keyword Index**: `ACLUtils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/quantized/cpu`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/quantized/cpu`):

- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`init_qnnpack.cpp_docs.md_docs.md`](./init_qnnpack.cpp_docs.md_docs.md)
- [`qelu.cpp_kw.md_docs.md`](./qelu.cpp_kw.md_docs.md)
- [`UpSampleNearest2d.cpp_kw.md_docs.md`](./UpSampleNearest2d.cpp_kw.md_docs.md)
- [`qclamp.cpp_docs.md_docs.md`](./qclamp.cpp_docs.md_docs.md)
- [`qembeddingbag_prepack.h_docs.md_docs.md`](./qembeddingbag_prepack.h_docs.md_docs.md)
- [`qdropout.cpp_docs.md_docs.md`](./qdropout.cpp_docs.md_docs.md)
- [`qelu.cpp_docs.md_docs.md`](./qelu.cpp_docs.md_docs.md)
- [`qembeddingbag_unpack.cpp_docs.md_docs.md`](./qembeddingbag_unpack.cpp_docs.md_docs.md)
- [`LinearUnpackImpl.cpp_kw.md_docs.md`](./LinearUnpackImpl.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `ACLUtils.h_docs.md_docs.md`
- **Keyword Index**: `ACLUtils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
