# Documentation: epilogue_helpers.h

## File Metadata
- **Path**: `aten/src/ATen/native/cuda/cutlass_extensions/epilogue_helpers.h`
- **Size**: 4455 bytes
- **Lines**: 82
- **Extension**: .h
- **Type**: Regular file

## Original Source

```h
/**
 * @file epilogue_helpers.h
 *
 * This file includes types for the epilogues. The empty structs exist so we can signal to template
 * code the type of epilogue we want to run, and let the underlying code specify the details such as
 * element types, accumulator type and elements per vector access.
 *
 */

#pragma once

#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_generic.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/epilogue/thread/linear_combination_silu.h>
#include <ATen/native/cuda/cutlass_extensions/epilogue/thread/ft_fused_activations.h>

namespace fastertransformer {

struct EpilogueOpBiasSilu {};

struct EpilogueOpBiasReLU {};

struct EpilogueOpBiasFtGelu {};

struct EpilogueOpBias {};

struct EpilogueOpNoBias {};

template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator, typename Op>
struct Epilogue {
};

template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBiasSilu> {
    using Op = cutlass::epilogue::thread::LinearCombinationSilu<ElementType,
                                                                ElementsPerVectorAccess,
                                                                ElementAccumulator,
                                                                ElementAccumulator,
                                                                cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
};

template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBiasReLU> {
    using Op = cutlass::epilogue::thread::LinearCombinationRelu<ElementType,
                                                                ElementsPerVectorAccess,
                                                                ElementAccumulator,
                                                                ElementAccumulator,
                                                                cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
};

template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBiasFtGelu> {
    using Op = cutlass::epilogue::thread::LinearCombinationGeneric<cutlass::epilogue::thread::GELU_taylor,
                                                                   ElementType,
                                                                   ElementsPerVectorAccess,
                                                                   ElementAccumulator,
                                                                   ElementAccumulator,
                                                                   cutlass::epilogue::thread::ScaleType::NoBetaScaling,
                                                                   cutlass::FloatRoundStyle::round_to_nearest,
                                                                   true>;
};

template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBias> {
    using Op = cutlass::epilogue::thread::LinearCombination<ElementType,
                                                            ElementsPerVectorAccess,
                                                            ElementAccumulator,
                                                            ElementAccumulator,
                                                            cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
};

template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpNoBias> {
    using Op = cutlass::epilogue::thread::LinearCombination<ElementType,
                                                            ElementsPerVectorAccess,
                                                            ElementAccumulator,
                                                            ElementAccumulator,
                                                            cutlass::epilogue::thread::ScaleType::Default>;
};

}  // namespace fastertransformer

```

## High-Level Overview

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough

### Structures
This file defines 11 struct(s): EpilogueOpBiasSilu, EpilogueOpBiasReLU, EpilogueOpBiasFtGelu, EpilogueOpBias, EpilogueOpNoBias, Epilogue, Epilogue, Epilogue, Epilogue, Epilogue, Epilogue


## Key Components

The file contains 209 words across 82 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4455 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
