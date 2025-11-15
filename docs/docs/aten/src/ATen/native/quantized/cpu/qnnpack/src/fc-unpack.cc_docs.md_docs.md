# Documentation: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/fc-unpack.cc_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src/fc-unpack.cc_docs.md`
- **Size**: 4,866 bytes (4.75 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/quantized/cpu/qnnpack/src/fc-unpack.cc`

## File Metadata

- **Path**: `aten/src/ATen/native/quantized/cpu/qnnpack/src/fc-unpack.cc`
- **Size**: 2,362 bytes (2.31 KB)
- **Type**: C++ Source Code
- **Extension**: `.cc`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <pytorch_qnnpack.h>
#include <qnnpack/log.h>
#include <qnnpack/pack.h>
#include <qnnpack_func.h>
#include <cstdlib>
#include <cstring>
#include <cmath>

namespace qnnpack {
// For runtime quantization unpacking.
void PackBMatrix::unpackWeights(
  const uint8_t* kernel_zero_points,
  int8_t* kernel
) const {
  union {
    void* const as_void_ptr;
    uint8_t* as_uint8_ptr;
    int32_t* as_int32_ptr;
  } packed = {packed_weights_};

  // C = A * B
  // A = M*K
  // B = K*N
  const uint32_t nr = pytorch_qnnp_params.q8conv.nr;
  const uint32_t kr = pytorch_qnnp_params.q8conv.kr;

  // Convert prepacked weight to original weight / bias.
  for (size_t nr_block_start = 0; nr_block_start < output_channels_; nr_block_start += nr) {
    const size_t nr_block_size = min(output_channels_ - nr_block_start, nr);
    for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
         nr_block_offset++) {
      packed.as_int32_ptr++;
    }
    packed.as_int32_ptr += (nr - nr_block_size);
    for (size_t kr_block_start = 0; kr_block_start < input_channels_; kr_block_start += kr) {
      const size_t kr_block_size = min(input_channels_ - kr_block_start, kr);
      for (size_t nr_block_offset = 0; nr_block_offset < nr_block_size;
           nr_block_offset++) {
        for (size_t kr_block_offset = 0; kr_block_offset < kr_block_size;
             kr_block_offset++) {
          kernel[(nr_block_start + nr_block_offset) * input_channels_ +
          (kr_block_start + kr_block_offset)] = *(packed.as_uint8_ptr++);
        }
        if (kernel_zero_points != nullptr) {
          for (size_t kr_block_offset = 0; kr_block_offset < (kr - kr_block_size);
               kr_block_offset++) {
            packed.as_uint8_ptr++;
          }
        } else {
          packed.as_uint8_ptr += (kr - kr_block_size);
        }
      }
      if (kernel_zero_points != nullptr) {
        size_t remaining_nr_blocks = ((nr - nr_block_size) & (nr - 1));
        for (size_t nr_block_offset = 0; nr_block_offset < remaining_nr_blocks;
             nr_block_offset++) {
          for (size_t kr_block_offset = 0; kr_block_offset < kr;
               kr_block_offset++) {
            packed.as_uint8_ptr++;
          }
        }
      } else {
        packed.as_uint8_ptr += ((nr - nr_block_size) & (nr - 1)) * kr;
      }
    }
  }

}

} // namespace qnnpack

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `qnnpack`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/quantized/cpu/qnnpack/src`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `pytorch_qnnpack.h`
- `qnnpack/log.h`
- `qnnpack/pack.h`
- `qnnpack_func.h`
- `cstdlib`
- `cstring`
- `cmath`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`aten/src/ATen/native/quantized/cpu/qnnpack/src`):

- [`global-average-pooling.c_docs.md`](./global-average-pooling.c_docs.md)
- [`fully-connected-sparse.c_docs.md`](./fully-connected-sparse.c_docs.md)
- [`tanh.c_docs.md`](./tanh.c_docs.md)
- [`add.c_docs.md`](./add.c_docs.md)
- [`channel-shuffle.c_docs.md`](./channel-shuffle.c_docs.md)
- [`fc-dynamic-run.cc_docs.md`](./fc-dynamic-run.cc_docs.md)
- [`softargmax.c_docs.md`](./softargmax.c_docs.md)
- [`fully-connected.c_docs.md`](./fully-connected.c_docs.md)
- [`conv-run.cc_docs.md`](./conv-run.cc_docs.md)
- [`init.c_docs.md`](./init.c_docs.md)


## Cross-References

- **File Documentation**: `fc-unpack.cc_docs.md`
- **Keyword Index**: `fc-unpack.cc_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/quantized/cpu/qnnpack/src`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/aten/src/ATen/native/quantized/cpu/qnnpack/src`):

- [`hardsigmoid.c_kw.md_docs.md`](./hardsigmoid.c_kw.md_docs.md)
- [`indirection.c_docs.md_docs.md`](./indirection.c_docs.md_docs.md)
- [`conv-prepack.cc_kw.md_docs.md`](./conv-prepack.cc_kw.md_docs.md)
- [`deconvolution.c_docs.md_docs.md`](./deconvolution.c_docs.md_docs.md)
- [`fully-connected.c_docs.md_docs.md`](./fully-connected.c_docs.md_docs.md)
- [`fully-connected-sparse.c_docs.md_docs.md`](./fully-connected-sparse.c_docs.md_docs.md)
- [`softargmax.c_kw.md_docs.md`](./softargmax.c_kw.md_docs.md)
- [`operator-run.c_docs.md_docs.md`](./operator-run.c_docs.md_docs.md)
- [`indirection.c_kw.md_docs.md`](./indirection.c_kw.md_docs.md)
- [`tanh.c_docs.md_docs.md`](./tanh.c_docs.md_docs.md)


## Cross-References

- **File Documentation**: `fc-unpack.cc_docs.md_docs.md`
- **Keyword Index**: `fc-unpack.cc_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
