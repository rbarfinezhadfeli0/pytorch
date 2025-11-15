# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qembeddingbag_unpack.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qembeddingbag_unpack.cpp_docs.md`
- **Size**: 14,394 bytes (14.06 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qembeddingbag_unpack.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qembeddingbag_unpack.cpp`
- **Size**: 11,464 bytes (11.20 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Parallel.h>
#include <ATen/native/quantized/cpu/EmbeddingPackedParams.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qembeddingbag.h>
#include <ATen/native/quantized/library.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_per_channel_affine_quantized.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/resize_native.h>
#endif

at::Tensor PackedEmbeddingBagWeight::unpack() {
  auto packed_weight = packed_w;
  at::Tensor weight_origin;

  if (bit_rate_ == 8 || bit_rate_ == 4) {
    const auto input_rows = packed_weight.size(0);
    const auto input_columns = packed_weight.size(1);
    int scale_bias_bytes = 0;
    const auto num_elem_per_byte = 8 / bit_rate_;
    if (bit_rate_ == 8) {
      // The last 2 values are used to store the FP32 scale and zero_point
      // values per row.
      scale_bias_bytes = 8;
    } else {
      scale_bias_bytes = 4;
    }

    const auto* input = packed_weight.const_data_ptr<uint8_t>();
    // Calculate the output shape, accounting for the last n bytes to be used
    // for scale/bias rest of the entries are packed depending on the bit_width.
    std::vector<int64_t> output_shape = {
        input_rows,
        static_cast<std::int64_t>(input_columns - scale_bias_bytes) *
            num_elem_per_byte};

    auto scales = at::from_blob(
        w_scale.data(), w_scale.size(), at::device(c10::kCPU).dtype(c10::kFloat));
    auto zero_points = at::from_blob(
        w_zp.data(), w_zp.size(), at::device(c10::kCPU).dtype(c10::kFloat));

    auto output_columns = output_shape[1];
    uint8_t* output_data = nullptr;

    // Allocate output weight tensor based on the bit_width
    if (bit_rate_ == 8) {
      weight_origin = at::_empty_per_channel_affine_quantized(
          output_shape,
          scales.toType(c10::kFloat),
          zero_points.toType(c10::kFloat),
          0, // The output channel axis is 0
          at::device(c10::kCPU).dtype(c10::kQUInt8));
      output_data = static_cast<uint8_t*>(weight_origin.data_ptr());
    } else {
      // We create empty qtensor with the full output shape, and dtype set to
      // quint4x2 This will internally allocate appropriate storage bytes to
      // account for the packed nature of this dtype.
      weight_origin = at::_empty_per_channel_affine_quantized(
          output_shape,
          scales.toType(c10::kFloat),
          zero_points.toType(c10::kFloat),
          0, // The output channel axis is 0
          at::device(c10::kCPU).dtype(c10::kQUInt4x2));
      output_data = static_cast<uint8_t*>(weight_origin.data_ptr());
    }

    // Copy over the data from the packed weight to the output.
    // For sub-byte tensors this will copy the packed bytes over since the
    // sub_byte qtensors are expected to store data in packed format.
    at::parallel_for(0, input_rows, 1, [&](int32_t start_idx, int32_t end_idx) {
      for (const auto row : c10::irange(start_idx, end_idx)) {
        const std::uint8_t* input_row = input + row * input_columns;
        uint8_t* output_row =
            output_data + row * output_columns / num_elem_per_byte;

        // output_columns
        for (const auto col : c10::irange(output_columns / num_elem_per_byte)) {
          output_row[col] = input_row[col];
        }
      }
    });

    return weight_origin;
  }
  TORCH_INTERNAL_ASSERT(
      false,
      "We currently only support 8-bit and 4-bit quantization of embedding_bag.");
  return weight_origin;
}

namespace at::native {

Tensor& qembeddingbag_byte_unpack_out(Tensor& output, const Tensor& packed_weight) {
  // The "last" dimension of an N-Dimensioned batch of embedding bags is
  // quantization channel. E.g. for a 2D embedding bag, this has
  // [ row, col ] dimensions, for batched of embedding bags, dimensions might be
  // [ batch, row, col ].
  //
  // Python Batched Embedding Example:
  // weights = torch.from_numpy((np.random.random_sample((
  //          2, 10, 3)).squeeze() + 1).astype(np.float32))
  // assert(weights.size() == torch.Size([2, 10, 3]))
  // # NOTE: 8 bytes (columns) are added due to fp32 zero_point and scales
  // packed_weights = torch.ops.quantized.embedding_bag_byte_prepack(weights)
  // assert(packed_weights.size() == torch.Size([2, 10, 11]))
  // unpacked_weights = torch.ops.quantized.embedding_bag_byte_unpack(packed_weights)
  // assert(unpacked_weights.size() == torch.Size([2, 10, 3]))
  const auto packed_weight_sizes = packed_weight.sizes();
  const auto col_dim = packed_weight_sizes.size() - 1;
  const int64_t input_rows = c10::size_to_dim_(col_dim, packed_weight_sizes);
  const int32_t input_columns = packed_weight_sizes[col_dim];
  // The last 2 values are used to store the FP32 scale and zero_point values
  // per row.
  const int32_t output_columns = input_columns - 2 * sizeof(float);
  const auto* input_data = packed_weight.const_data_ptr<uint8_t>();

  std::vector<int64_t> output_shape = packed_weight_sizes.vec();
  output_shape[col_dim] = output_columns;
  at::native::resize_(output, output_shape);
  auto output_contig = output.expect_contiguous();
  float* output_data = output_contig->data_ptr<float>();

#ifdef USE_FBGEMM
  at::parallel_for(0, input_rows, 1, [&](int64_t start_idx, int64_t end_idx) {
    fbgemm::Fused8BitRowwiseQuantizedSBFloatToFloatOrHalf<float>(
        input_data + start_idx * input_columns,
        end_idx - start_idx,
        input_columns,
        output_data + start_idx * output_columns);
  });
#else
  for (auto row : c10::irange(input_rows)) {
    const std::uint8_t* input_row = input_data + row * input_columns;
    const float* input_row_scale_zp =
        reinterpret_cast<const float*>(input_row + output_columns);
    float* output_row = output_data + row * output_columns;

    for (auto col : c10::irange(output_columns)) {
      output_row[col] =
          input_row[col] * input_row_scale_zp[0] + input_row_scale_zp[1];
    } // output_columns
  } // input_rows
#endif // USE_FBGEMM
  return output;
}

namespace {
Tensor qembeddingbag_byte_unpack(const Tensor& packed_weight) {
  at::Tensor output = at::empty(
      {},
      packed_weight.options().dtype(kFloat),
      packed_weight.suggest_memory_format());
  qembeddingbag_byte_unpack_out(output, packed_weight);
  return output;
}

Tensor qembeddingbag_byte_unpack_meta(const Tensor& packed_weight) {
  const auto packed_weight_sizes = packed_weight.sym_sizes();
  const auto col_dim = packed_weight_sizes.size() - 1;
  const auto input_columns = packed_weight_sizes[col_dim];
  // The last 2 values are used to store the FP32 scale and zero_point values
  // per row.
  const auto output_columns = input_columns - 2 * sizeof(float);

  auto output_shape = packed_weight_sizes.vec();
  output_shape[col_dim] = output_columns;

  at::SymDimVector output_shape_vec(output_shape);
  return at::empty_symint(output_shape_vec, packed_weight.options().dtype(kFloat), packed_weight.suggest_memory_format());
}

Tensor _qembeddingbag_nbit_unpack_helper(
    const Tensor& packed_weight,
    int BIT_RATE) {
  const auto input_rows = packed_weight.size(0);
  const auto input_columns = packed_weight.size(1);
  const auto* input_data = packed_weight.const_data_ptr<uint8_t>();
  int NUM_ELEM_PER_BYTE = 8 / BIT_RATE;

  // The last 4 bytes per row are two fp16 scale and zero_point.
  // The rest of input_columns is the number of values in the original row.
  std::vector<int64_t> output_dimensions = {
      input_rows,
      static_cast<std::int64_t>(input_columns - 2 * sizeof(at::Half)) *
          NUM_ELEM_PER_BYTE};

  auto output = at::empty(
      output_dimensions,
      packed_weight.options().dtype(kFloat),
      packed_weight.suggest_memory_format());
  float* output_data = output.data_ptr<float>();
#ifdef USE_FBGEMM
  at::parallel_for(0, input_rows, 1, [&](int64_t start_idx, int64_t end_idx) {
    fbgemm::FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf<float>(
        BIT_RATE,
        input_data + start_idx * input_columns,
        end_idx - start_idx,
        input_columns,
        output_data + start_idx * output_dimensions[1]);
  });
#else
  auto output_columns = output_dimensions[1];
  for (auto row : c10::irange(input_rows)) {
    float* output_row = output_data + row * output_columns;
    const std::uint8_t* input_row = input_data + row * input_columns;
    const at::Half* input_row_scale_zp = reinterpret_cast<const at::Half*>(
        input_row +
        (output_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE);
    float scale = input_row_scale_zp[0];
    float zero_point = input_row_scale_zp[1];

    for (const auto col : c10::irange(output_columns)) {
      std::uint8_t quantized = input_row[col / NUM_ELEM_PER_BYTE];
      quantized >>= (col % NUM_ELEM_PER_BYTE) * BIT_RATE;
      quantized &= (1 << BIT_RATE) - 1;
      output_row[col] = scale * quantized + zero_point;
    } // output_columns
  } // input_rows
#endif // USE_FBGEMM

  return output;
}

// De-quantizes the result of the qembeddingbag_4bit_prepack operator.
// The input is expected to first have quantized values,
// then 2-byte fp16 scale and 2-byte zero_offset.
// The output is a matrix containing only the values, but de-quantized.
// De-quantization is performed by multiplying each value by its
// row's scale and zero_point parameters. The de-quantized values
// will thus not be exactly equal to the original, un-quantized
// floating point values.
Tensor qembeddingbag_4bit_unpack(const Tensor& packed_weight) {
  return _qembeddingbag_nbit_unpack_helper(packed_weight, 4 /*BIT_RATE*/);
}

// De-quantizes the result of the qembeddingbag_2bit_prepack operator.
// The input is expected to first have quantized values,
// then 2-byte fp16 scale and 2-byte zero_offset.
// The output is a matrix containing only the values, but de-quantized.
// De-quantization is performed by multiplying each value by its
// row's scale and zero_point parameters. The de-quantized values
// will thus not be exactly equal to the original, un-quantized
// floating point values.
Tensor qembeddingbag_2bit_unpack(const Tensor& packed_weight) {
  return _qembeddingbag_nbit_unpack_helper(packed_weight, 2 /*BIT_RATE*/);
}

class QEmbeddingUnpackWeights final {
 public:
  static at::Tensor run(
      const c10::intrusive_ptr<EmbeddingPackedParamsBase>& packed_weight) {
    return packed_weight->unpack();
  }
};

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_byte_unpack"),
      qembeddingbag_byte_unpack);
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_4bit_unpack"),
      qembeddingbag_4bit_unpack);
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_2bit_unpack"),
      qembeddingbag_2bit_unpack);
}

TORCH_LIBRARY_IMPL(quantized, CatchAll, m) {
  // Unpack the packed embedding_bag weights using TorchBind custom class.
  // TODO extend to support 4-bit qtensor.
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_unpack"),
      TORCH_FN(QEmbeddingUnpackWeights::run));
}

TORCH_LIBRARY_IMPL(quantized, Meta, m) {
  m.impl(
      "quantized::embedding_bag_byte_unpack",
      qembeddingbag_byte_unpack_meta);
}

} // namespace
} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `QEmbeddingUnpackWeights`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Parallel.h`
- `ATen/native/quantized/cpu/EmbeddingPackedParams.h`
- `ATen/native/quantized/cpu/fbgemm_utils.h`
- `ATen/native/quantized/cpu/qembeddingbag.h`
- `ATen/native/quantized/library.h`
- `c10/util/irange.h`
- `torch/library.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/_empty_per_channel_affine_quantized.h`
- `ATen/ops/empty.h`
- `ATen/ops/from_blob.h`
- `ATen/ops/resize_native.h`


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

Files in the same folder (`aten/src/ATen/native/quantized/cpu`):

- [`ACLUtils.cpp_docs.md`](./ACLUtils.cpp_docs.md)
- [`LinearUnpackImpl.cpp_docs.md`](./LinearUnpackImpl.cpp_docs.md)
- [`UpSampleNearest3d.cpp_docs.md`](./UpSampleNearest3d.cpp_docs.md)
- [`Pooling.cpp_docs.md`](./Pooling.cpp_docs.md)
- [`QnnpackUtils.h_docs.md`](./QnnpackUtils.h_docs.md)
- [`fbgemm_utils.h_docs.md`](./fbgemm_utils.h_docs.md)
- [`TensorOperators.cpp_docs.md`](./TensorOperators.cpp_docs.md)
- [`XnnpackUtils.h_docs.md`](./XnnpackUtils.h_docs.md)
- [`qconv_dynamic.cpp_docs.md`](./qconv_dynamic.cpp_docs.md)


## Cross-References

- **File Documentation**: `qembeddingbag_unpack.cpp_docs.md`
- **Keyword Index**: `qembeddingbag_unpack.cpp_kw.md`
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
- [`LinearUnpackImpl.cpp_kw.md_docs.md`](./LinearUnpackImpl.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `qembeddingbag_unpack.cpp_docs.md_docs.md`
- **Keyword Index**: `qembeddingbag_unpack.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
