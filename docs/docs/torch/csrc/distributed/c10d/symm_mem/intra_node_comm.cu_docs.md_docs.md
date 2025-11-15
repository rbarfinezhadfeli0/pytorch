# Documentation: `docs/torch/csrc/distributed/c10d/symm_mem/intra_node_comm.cu_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/c10d/symm_mem/intra_node_comm.cu_docs.md`
- **Size**: 6,659 bytes (6.50 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/c10d/symm_mem/intra_node_comm.cu`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/symm_mem/intra_node_comm.cu`
- **Size**: 3,918 bytes (3.83 KB)
- **Type**: CUDA Source Code
- **Extension**: `.cu`

## File Purpose

This is a cuda source code that is part of the PyTorch project.

## Original Source

```cuda
#include <torch/csrc/distributed/c10d/symm_mem/intra_node_comm.hpp>

#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.h>

namespace c10d {
namespace intra_node_comm {

static constexpr size_t kOneShotThreshBytes = 256 * 1024;
static constexpr size_t kTwoShotThreshBytes = 10 * 1024 * 1024;

static void checkInput(const at::Tensor& input, int deviceIdx) {
  TORCH_CHECK(
      input.dtype() == at::kBFloat16 || input.dtype() == at::kFloat,
      "oneShotAllReduce only supports float and bf16 for now");
  TORCH_CHECK(input.is_non_overlapping_and_dense());
  TORCH_CHECK(input.device().is_cuda());
  TORCH_CHECK(
      input.get_device() == deviceIdx,
      "IntraNodeComm: expect input to be on device ",
      deviceIdx,
      ", got device ",
      input.get_device());
}

bool isIntraNodeCommSupported() {
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  return false;
#else
  return true;
#endif
}

at::Tensor IntraNodeComm::oneShotAllReduce(
    const at::Tensor& input,
    at::cuda::CUDAStream& stream) {
  checkInput(input, deviceIdx_);

  auto op = c10::Dispatcher::singleton()
                .findSchemaOrThrow("symm_mem::one_shot_all_reduce_out", "")
                .typed<at::Tensor(
                    const at::Tensor&, std::string, std::string, at::Tensor)>();

  auto symmMemTensor = at::from_blob(
      symmetricMemoryPtr_,
      input.sizes(),
      at::TensorOptions().dtype(input.dtype()).device(input.device()));

  symmMemTensor.copy_(input);
  op.call(symmMemTensor, "sum", "", input);
  return input;
}

at::Tensor IntraNodeComm::twoShotAllReduce(
    const at::Tensor& input,
    at::cuda::CUDAStream& stream) {
  checkInput(input, deviceIdx_);

  auto op = c10::Dispatcher::singleton()
                .findSchemaOrThrow("symm_mem::two_shot_all_reduce_", "")
                .typed<at::Tensor(at::Tensor, std::string, std::string)>();

  auto symmMemTensor = at::from_blob(
      symmetricMemoryPtr_,
      input.sizes(),
      at::TensorOptions().dtype(input.dtype()).device(input.device()));

  symmMemTensor.copy_(input);
  op.call(symmMemTensor, "sum", "");
  input.copy_(symmMemTensor);
  return input;
}

AllReduceAlgo IntraNodeComm::selectAllReduceAlgo(const at::Tensor& input) {
  // Only support float and bf16 for now
  if (input.dtype() != at::kBFloat16 && input.dtype() != at::kFloat) {
    return AllReduceAlgo::NONE;
  }
  const auto inputSize =
      static_cast<size_t>(input.numel() * input.element_size());
  const size_t ptrAlignment = get_alignment(
      static_cast<size_t>(input.storage_offset() * input.element_size()));
  const size_t sizeAlignment = get_alignment(inputSize);
  const size_t alignment = std::min(ptrAlignment, sizeAlignment);

  if (topology_ == Topology::FULLY_CONNECTED) {
    // Both symm_mem::one_shot_all_reduce and symm_mem::two_shot_all_reduce_
    // currently requires the input to be at least 4-bytes aligned.
    if (alignment >= 4 && inputSize <= kOneShotThreshBytes &&
        inputSize <= bufferSize_) {
      return AllReduceAlgo::ONE_SHOT;
    }
    if (alignment >= 4 && inputSize <= kTwoShotThreshBytes &&
        inputSize <= bufferSize_) {
      return AllReduceAlgo::TWO_SHOT;
    }
  }
  return AllReduceAlgo::NONE;
}

static int64_t usageCounter = 0;

at::Tensor IntraNodeComm::allReduce(
    const at::Tensor& input,
    AllReduceAlgo algo) {
  // Report usage for testing purposes.
  // We don't care about overflowing.
  ++usageCounter;
  auto stream = at::cuda::getCurrentCUDAStream();
  switch (algo) {
    case AllReduceAlgo::ONE_SHOT:
      return oneShotAllReduce(input, stream);
    case AllReduceAlgo::TWO_SHOT:
      return twoShotAllReduce(input, stream);
    default:
      C10_THROW_ERROR(ValueError, "IntraNodeComm: invalid algo");
  }
}

int64_t getIntraNodeCommUsageCounter() {
  return usageCounter;
}

} // namespace intra_node_comm
} // namespace c10d

```



## High-Level Overview

This file is part of the PyTorch framework located at `torch/csrc/distributed/c10d/symm_mem`.

## Detailed Analysis

### Code Structure

**Namespaces**: `intra_node_comm`, `c10d`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d/symm_mem`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/c10d/symm_mem/intra_node_comm.hpp`
- `torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.h`


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

Files in the same folder (`torch/csrc/distributed/c10d/symm_mem`):

- [`SymmetricMemory.cpp_docs.md`](./SymmetricMemory.cpp_docs.md)
- [`CUDASymmetricMemoryOps.cu_docs.md`](./CUDASymmetricMemoryOps.cu_docs.md)
- [`NVSHMEMSymmetricMemory.cu_docs.md`](./NVSHMEMSymmetricMemory.cu_docs.md)
- [`SymmetricMemory.hpp_docs.md`](./SymmetricMemory.hpp_docs.md)
- [`DMAConnectivity.hpp_docs.md`](./DMAConnectivity.hpp_docs.md)
- [`DMAConnectivity.cpp_docs.md`](./DMAConnectivity.cpp_docs.md)
- [`nvshmem_team_manager.hpp_docs.md`](./nvshmem_team_manager.hpp_docs.md)
- [`nvshmem_extension.cu_docs.md`](./nvshmem_extension.cu_docs.md)
- [`nvshmem_extension.cuh_docs.md`](./nvshmem_extension.cuh_docs.md)
- [`CUDASymmetricMemoryUtils.cpp_docs.md`](./CUDASymmetricMemoryUtils.cpp_docs.md)


## Cross-References

- **File Documentation**: `intra_node_comm.cu_docs.md`
- **Keyword Index**: `intra_node_comm.cu_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/distributed/c10d/symm_mem`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/distributed/c10d/symm_mem`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/distributed/c10d/symm_mem`):

- [`SymmetricMemory.hpp_docs.md_docs.md`](./SymmetricMemory.hpp_docs.md_docs.md)
- [`CUDASymmetricMemory.hpp_docs.md_docs.md`](./CUDASymmetricMemory.hpp_docs.md_docs.md)
- [`nvshmem_extension.cuh_docs.md_docs.md`](./nvshmem_extension.cuh_docs.md_docs.md)
- [`DMAConnectivity.cpp_docs.md_docs.md`](./DMAConnectivity.cpp_docs.md_docs.md)
- [`CudaDMAConnectivity.cpp_docs.md_docs.md`](./CudaDMAConnectivity.cpp_docs.md_docs.md)
- [`NCCLSymmetricMemory.cu_kw.md_docs.md`](./NCCLSymmetricMemory.cu_kw.md_docs.md)
- [`CUDASymmetricMemory.cu_kw.md_docs.md`](./CUDASymmetricMemory.cu_kw.md_docs.md)
- [`nvshmem_extension.cu_docs.md_docs.md`](./nvshmem_extension.cu_docs.md_docs.md)
- [`DMAConnectivity.hpp_docs.md_docs.md`](./DMAConnectivity.hpp_docs.md_docs.md)
- [`CUDASymmetricMemory-inl.h_kw.md_docs.md`](./CUDASymmetricMemory-inl.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `intra_node_comm.cu_docs.md_docs.md`
- **Keyword Index**: `intra_node_comm.cu_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
