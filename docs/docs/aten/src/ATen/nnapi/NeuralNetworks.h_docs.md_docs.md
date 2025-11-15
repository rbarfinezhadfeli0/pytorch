# Documentation: `docs/aten/src/ATen/nnapi/NeuralNetworks.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/nnapi/NeuralNetworks.h_docs.md`
- **Size**: 5,230 bytes (5.11 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/nnapi/NeuralNetworks.h`

## File Metadata

- **Path**: `aten/src/ATen/nnapi/NeuralNetworks.h`
- **Size**: 2,757 bytes (2.69 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*

Most of NeuralNetworks.h has been stripped for simplicity.
We don't need any of the function declarations since
we call them all through dlopen/dlsym.
Operation codes are pulled directly from serialized models.

*/

#ifndef MINIMAL_NEURAL_NETWORKS_H
#define MINIMAL_NEURAL_NETWORKS_H

#include <stdint.h>

typedef enum {
    ANEURALNETWORKS_NO_ERROR = 0,
    ANEURALNETWORKS_OUT_OF_MEMORY = 1,
    ANEURALNETWORKS_INCOMPLETE = 2,
    ANEURALNETWORKS_UNEXPECTED_NULL = 3,
    ANEURALNETWORKS_BAD_DATA = 4,
    ANEURALNETWORKS_OP_FAILED = 5,
    ANEURALNETWORKS_BAD_STATE = 6,
    ANEURALNETWORKS_UNMAPPABLE = 7,
    ANEURALNETWORKS_OUTPUT_INSUFFICIENT_SIZE = 8,
    ANEURALNETWORKS_UNAVAILABLE_DEVICE = 9,
} ResultCode;

typedef enum {
    ANEURALNETWORKS_FLOAT32 = 0,
    ANEURALNETWORKS_INT32 = 1,
    ANEURALNETWORKS_UINT32 = 2,
    ANEURALNETWORKS_TENSOR_FLOAT32 = 3,
    ANEURALNETWORKS_TENSOR_INT32 = 4,
    ANEURALNETWORKS_TENSOR_QUANT8_ASYMM = 5,
    ANEURALNETWORKS_BOOL = 6,
    ANEURALNETWORKS_TENSOR_QUANT16_SYMM = 7,
    ANEURALNETWORKS_TENSOR_FLOAT16 = 8,
    ANEURALNETWORKS_TENSOR_BOOL8 = 9,
    ANEURALNETWORKS_FLOAT16 = 10,
    ANEURALNETWORKS_TENSOR_QUANT8_SYMM_PER_CHANNEL = 11,
    ANEURALNETWORKS_TENSOR_QUANT16_ASYMM = 12,
    ANEURALNETWORKS_TENSOR_QUANT8_SYMM = 13,
} OperandCode;

typedef enum {
    ANEURALNETWORKS_PREFER_LOW_POWER = 0,
    ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER = 1,
    ANEURALNETWORKS_PREFER_SUSTAINED_SPEED = 2,
} PreferenceCode;

typedef struct ANeuralNetworksMemory ANeuralNetworksMemory;
typedef struct ANeuralNetworksModel ANeuralNetworksModel;
typedef struct ANeuralNetworksDevice ANeuralNetworksDevice;
typedef struct ANeuralNetworksCompilation ANeuralNetworksCompilation;
typedef struct ANeuralNetworksExecution ANeuralNetworksExecution;
typedef struct ANeuralNetworksEvent ANeuralNetworksEvent;

typedef int32_t ANeuralNetworksOperationType;

typedef struct ANeuralNetworksOperandType {
    int32_t type;
    uint32_t dimensionCount;
    const uint32_t* dimensions;
    float scale;
    int32_t zeroPoint;
} ANeuralNetworksOperandType;

#endif  // MINIMAL_NEURAL_NETWORKS_H

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `ANeuralNetworksMemory`, `ANeuralNetworksModel`, `ANeuralNetworksDevice`, `ANeuralNetworksCompilation`, `ANeuralNetworksExecution`, `ANeuralNetworksEvent`, `ANeuralNetworksOperandType`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/nnapi`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `stdint.h`


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

Files in the same folder (`aten/src/ATen/nnapi`):

- [`nnapi_wrapper.cpp_docs.md`](./nnapi_wrapper.cpp_docs.md)
- [`nnapi_wrapper.h_docs.md`](./nnapi_wrapper.h_docs.md)
- [`codegen.py_docs.md`](./codegen.py_docs.md)
- [`nnapi_model_loader.h_docs.md`](./nnapi_model_loader.h_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`nnapi_bind.cpp_docs.md`](./nnapi_bind.cpp_docs.md)
- [`nnapi_model_loader.cpp_docs.md`](./nnapi_model_loader.cpp_docs.md)
- [`nnapi_register.cpp_docs.md`](./nnapi_register.cpp_docs.md)
- [`nnapi_bind.h_docs.md`](./nnapi_bind.h_docs.md)


## Cross-References

- **File Documentation**: `NeuralNetworks.h_docs.md`
- **Keyword Index**: `NeuralNetworks.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/nnapi`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/nnapi`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/nnapi`):

- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`nnapi_model_loader.h_docs.md_docs.md`](./nnapi_model_loader.h_docs.md_docs.md)
- [`nnapi_register.cpp_kw.md_docs.md`](./nnapi_register.cpp_kw.md_docs.md)
- [`nnapi_bind.h_kw.md_docs.md`](./nnapi_bind.h_kw.md_docs.md)
- [`nnapi_bind.cpp_docs.md_docs.md`](./nnapi_bind.cpp_docs.md_docs.md)
- [`nnapi_model_loader.h_kw.md_docs.md`](./nnapi_model_loader.h_kw.md_docs.md)
- [`nnapi_register.cpp_docs.md_docs.md`](./nnapi_register.cpp_docs.md_docs.md)
- [`codegen.py_kw.md_docs.md`](./codegen.py_kw.md_docs.md)
- [`nnapi_bind.cpp_kw.md_docs.md`](./nnapi_bind.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `NeuralNetworks.h_docs.md_docs.md`
- **Keyword Index**: `NeuralNetworks.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
