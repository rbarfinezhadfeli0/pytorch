# Documentation: `aten/src/ATen/core/QuantizerBase.h`

## File Metadata

- **Path**: `aten/src/ATen/core/QuantizerBase.h`
- **Size**: 2,687 bytes (2.62 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/core/ScalarType.h>
#include <c10/core/QScheme.h>
#include <c10/util/intrusive_ptr.h>

namespace at {

class Tensor;
struct QTensorImpl;
struct Quantizer;
using ConstQuantizerPtr = const c10::intrusive_ptr<Quantizer>&;
using QuantizerPtr = c10::intrusive_ptr<Quantizer>;

/**
 * Quantizer is the class for storing all the information
 * that's necessary to perform quantize and dequantize
 * operation.
 *
 * We might have different types of quantization schemes and this is
 * the base class for all quantizers.
 *
 * QTensorImpl will hold a pointer to Quantizer so that we can support
 * different quantization schemes on Tensor.
 *
 * For example, the most common quantization scheme, Affine Quantization,
 * requires scale and zero_point as parameters, we'll store scale and zero_point
 * inside the instance and we can use it to quantize a float Tensor or
 * dequantize a quantized Tensor.
 *
 * When you add new types of leaf Quantizer class, please also
 * make sure to add a corresponding QScheme enum since
 * they should have one to one mapping.
 *
 * Note about intrusive_ptr:
 * Quantized Tensor holds an intrusive_ptr to Quantizer, and multiple Tensor can
 * share the same Quantizer. Quantizer should be immutable.
 */
struct TORCH_API Quantizer : public c10::intrusive_ptr_target {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const ScalarType scalar_type_;
  explicit Quantizer(ScalarType scalar_type) : scalar_type_(scalar_type) {}
  ~Quantizer() override = default;

  // Copied from torch/csrc/jit/ir/scope.h
  QuantizerPtr intrusive_from_this() {
    c10::raw::intrusive_ptr::incref(this); // we are creating a new pointer
                                           // from a raw `this` pointer
                                           // so we need to bump the refcount
                                           // to account for this ownership
    return c10::intrusive_ptr<Quantizer>::reclaim(this);
  }

  /**
   * Each concrete Quantizer type should have a unique QScheme type.
   */
  virtual QScheme qscheme() const = 0;

  ScalarType scalar_type() const {
    return scalar_type_;
  }

  /**
   * quantize a float Tensor into a quantized Tensor.
   */
  virtual Tensor quantize(const Tensor& t) = 0;

  /**
   * dequantize a quantized Tensor into a float Tensor.
   */
  virtual Tensor dequantize(const Tensor& t) = 0;

  /**
   * dequantize a quantized Tensor into a float Tensor, out= variant
   */
  virtual Tensor& dequantize_out(Tensor& out, const Tensor& t) = 0;

  /**
   * Compare against `other` for equality.
   */
  virtual bool equalTo(QuantizerPtr other) const = 0;
};

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `Tensor`, `QTensorImpl`, `Quantizer`, `for`, `for`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/ScalarType.h`
- `c10/core/QScheme.h`
- `c10/util/intrusive_ptr.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`aten/src/ATen/core`):

- [`DistributionsHelper.h_docs.md`](./DistributionsHelper.h_docs.md)
- [`rref_interface.h_docs.md`](./rref_interface.h_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`enum_type.h_docs.md`](./enum_type.h_docs.md)
- [`Array.h_docs.md`](./Array.h_docs.md)
- [`MetaFallbackKernel.cpp_docs.md`](./MetaFallbackKernel.cpp_docs.md)
- [`ATenOpList.h_docs.md`](./ATenOpList.h_docs.md)
- [`ivalue_inl.h_docs.md`](./ivalue_inl.h_docs.md)
- [`TransformationHelper.h_docs.md`](./TransformationHelper.h_docs.md)


## Cross-References

- **File Documentation**: `QuantizerBase.h_docs.md`
- **Keyword Index**: `QuantizerBase.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
