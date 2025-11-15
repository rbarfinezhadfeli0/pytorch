# Documentation: `torch/csrc/jit/codegen/onednn/kernel.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/codegen/onednn/kernel.cpp`
- **Size**: 10,476 bytes (10.23 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/jit/codegen/onednn/graph_helper.h>
#include <torch/csrc/jit/codegen/onednn/kernel.h>

#include <ATen/core/functional.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch::jit::fuser::onednn {

using namespace dnnl::graph;
using data_type = dnnl::graph::logical_tensor::data_type;

LlgaKernel::LlgaKernel(const Node* fusionNode)
    : fusionNode_(fusionNode),
      graph_(fusionNode->g(attr::Subgraph)),
      nGraphInputs_(graph_->inputs().size()),
      nOutputs_(graph_->outputs().size()),
      debugName_(genDebugName()) {
  // TODO: This is a workaround to recreate the partitions here.
  // The ideal way is to use the partition serialization API (not available from
  // LLGA now) to carry a serialized string representation from graph rewrite
  // and deserialize it here.
  auto llgaGraphHelper = LlgaGraphHelper(graph_);
  auto partitions = llgaGraphHelper.getPartitions();
  tensorIdToValue_ = llgaGraphHelper.getTensorIdToValue();
  TORCH_CHECK(
      partitions.size() == 1,
      "LLGA subgraph should contain only one partition");
  partition_ = partitions[0];
  nPartitionInputs_ = partition_.get_input_ports().size();
#ifdef GRAPH_DEBUG_ENABLED
  GRAPH_DEBUG("Initialized ", debugName(), "\n", graph_->toString());
#endif
}

bool LlgaKernel::useOpaqueLayout(size_t offset) const {
  return LlgaNodeWrapper(fusionNode_).useOpaqueLayout(offset);
}

void LlgaKernel::initializeConstantInputs() {
  for (auto& lt : partition_.get_input_ports()) {
    auto inputId = lt.get_id();
    if (initializedInputIds_.find(inputId) == initializedInputIds_.end()) {
      TORCH_CHECK(
          tensorIdToValue_.count(inputId) > 0,
          "inputs with inputId ",
          inputId,
          " is missing");
      auto* value = tensorIdToValue_[inputId];

      TORCH_CHECK(
          value->node()->kind() == prim::Constant &&
              value->type()->cast<TensorType>(),
          "inputs with inputId ",
          inputId,
          " should be a Constant tensor");
      constantValues_.emplace_back(value);

      auto const_tensor = toIValue(value)->toTensor();
      constantInputs_.emplace_back(const_tensor);
    }
  }
}

std::map<size_t, int64_t> LlgaKernel::initializeTensorIdToOccurence() const {
  std::map<size_t, int64_t> tensorIdToOccurence;
  for (auto& lt : partition_.get_input_ports()) {
    auto inputId = lt.get_id();
    std::map<size_t, int64_t>::iterator it(tensorIdToOccurence.find(inputId));
    if (it != tensorIdToOccurence.end()) {
      it->second++;
    } else {
      tensorIdToOccurence[inputId] = 1;
    }
  }
  return tensorIdToOccurence;
}

ArgSpecs LlgaKernel::initializeInputSpecs(const TensorArgs& inputs) {
  ArgSpecs inputSpecs;
  inputSpecs.reserve(nPartitionInputs_);
  GRAPH_DEBUG("Initializing graph input logical tensors");
  std::map<size_t, int64_t> tensorIdToOccurence =
      initializeTensorIdToOccurence();
  for (const auto i : c10::irange(nGraphInputs_)) {
    auto spec = ArgSpec(graph_->inputs()[i]).supplementTensorInfo(inputs[i]);
    initializedInputIds_.insert(spec.tid());
    int64_t occurrence = tensorIdToOccurence[spec.tid()];
    inputSpecs.insert(inputSpecs.end(), occurrence, spec);
    runArgsIdx_.insert(runArgsIdx_.end(), occurrence, i);
  }
  GRAPH_DEBUG("Initializing constant input tensors");
  initializeConstantInputs();

  TORCH_CHECK(
      inputSpecs.size() + constantValues_.size() ==
          static_cast<size_t>(nPartitionInputs_),
      "Partition inputs are missing");
  GRAPH_DEBUG(
      "Concatenating constant input logical tensors to graph input "
      "logical tensors");
  for (Value* constant_value : constantValues_) {
    ArgSpec constantInputSpec(constant_value);
    inputSpecs.emplace_back(constantInputSpec);
    constantLogicalTensors_.emplace_back(constantInputSpec.logical_tensor());
  }
  return inputSpecs;
}

ArgSpecs LlgaKernel::initializeOutputSpecs() const {
  ArgSpecs outputSpecs;
  outputSpecs.reserve(nOutputs_);
  for (const auto i : c10::irange(nOutputs_)) {
    auto spec = ArgSpec(graph_->outputs()[i]);
    if (useOpaqueLayout(i)) {
      spec = spec.any();
    }
    outputSpecs.emplace_back(spec);
  }
  return outputSpecs;
}

std::tuple<RunArgs, RunArgs> LlgaKernel::prepareRunArgs(
    const TensorArgs& inputs,
    TensorArgs& outputs) const {
  RunArgs runInputs, runOutputs;
  auto numInputs = runArgsIdx_.size();
  for (const auto i : c10::irange(numInputs)) {
    auto spec = inputSpecs_[i];
    const auto& input = inputs[runArgsIdx_[i]];
    runInputs.push_back(
        {spec.logical_tensor(), Engine::getEngine(), input.data_ptr()});
  }
  auto numConstantInputs = constantInputs_.size();
  for (size_t i = 0; i < numConstantInputs; i++) {
    // constantInputSpecs are placed after graphInputSpecs
    auto constantInputSpecIdx = nGraphInputs_ + i;
    auto constantInputSpec = inputSpecs_[constantInputSpecIdx];
    runInputs.push_back(
        {constantLogicalTensors_[i],
         Engine::getEngine(),
         constantInputs_[i].data_ptr()});
  }

  for (const auto i : c10::irange(nOutputs_)) {
    auto spec = outputSpecs_[i];
    auto opt = c10::TensorOptions(spec.aten_scalar_type()).device(device_);

    if (spec.reuses_input_tensor()) {
#ifdef GRAPH_DEBUG_ENABLED
      GRAPH_DEBUG("inplace computation - input tensor would be reused");
#endif
      auto inputTensor = inputs[spec.get_input_tensor_index()];
      if (inputTensor.is_mkldnn()) {
        auto dataType = spec.dtype();
        if (C10_UNLIKELY(!useOpaqueLayout(i))) {
          // If the input tensor was between two partitions, it would've been
          // wrapped with LlgaTensorImpl. But if it's being reused as the output
          // tensor, which is not between two partitions, then we'd have to
          // re-wrap it with a sub-class of TensorImpl, as it'd be fed into a
          // PyTorch op.
#ifdef GRAPH_DEBUG_ENABLED
          GRAPH_DEBUG("rewrap tensors");
#endif
          auto llgaImpl =
              static_cast<LlgaTensorImpl*>(inputTensor.unsafeGetTensorImpl());
          switch (dataType) {
            case data_type::f32:
            case data_type::bf16:
              inputTensor = LlgaTensorImpl::llga_to_aten_tensor(llgaImpl);
              break;
            case data_type::s32:
            default:
              TORCH_CHECK(
                  false, "Invalid data type ", static_cast<size_t>(dataType));
          }
        }
        outputs.push_back(inputTensor);
        runOutputs.push_back(
            {spec.logical_tensor(),
             Engine::getEngine(),
             inputTensor.data_ptr()});
        return std::make_tuple(runInputs, runOutputs);
      }
    }
    if (useOpaqueLayout(i)) {
      // Wrap tensors between partitions with LlgaTensorImpl wrapper, so that we
      // can bypass guard-check, as strides would be different than those
      // expected.
#ifdef GRAPH_DEBUG_ENABLED
      GRAPH_DEBUG("Between two oneDNN Graph partitions");
#endif
      auto tensor = empty_llga(spec, opt);
      outputs.push_back(tensor);
      runOutputs.push_back(llga_from_aten_tensor(tensor));
    } else {
#ifdef GRAPH_DEBUG_ENABLED
      GRAPH_DEBUG("Neither opaque to PyTorch nor inplace-computation");
#endif
      auto tensor = at::empty_strided(spec.sizes(), spec.strides(), opt);
      outputs.push_back(tensor);
      runOutputs.push_back(
          {spec.logical_tensor(), Engine::getEngine(), tensor.data_ptr()});
    }
  }

  return std::make_tuple(runInputs, runOutputs);
}

compiled_partition LlgaKernel::compile(const partition& partition) {
  auto inputs = fmap(inputSpecs_, toLogicalTensor);
  auto outputs = fmap(outputSpecs_, toLogicalTensor);
  auto compilation = partition.compile(inputs, outputs, Engine::getEngine());

  // Since layouts of opaque outputs would be known after compilation,
  // we need to query them out from compilation and update outputSpecs
  for (const auto i : c10::irange(nOutputs_)) {
    auto tid = outputSpecs_[i].tid();
    outputSpecs_[i] = compilation.query_logical_tensor(tid);
  }

  // Build static mapping from output id to input offset
  // in accordance with available inplace options
  for (auto&& option : compilation.get_inplace_ports()) {
    size_t inputId = option.first;
    size_t outputId = option.second;
    auto inputSpecIter =
        std::find_if(inputSpecs_.begin(), inputSpecs_.end(), [&](auto& spec) {
          return spec.tid() == inputId;
        });
    TORCH_CHECK(inputSpecIter != inputSpecs_.end(), "In-place input not found");
    auto inputOffset = inputSpecIter - inputSpecs_.begin();
    auto outputSpecIter =
        std::find_if(outputSpecs_.begin(), outputSpecs_.end(), [&](auto& spec) {
          return spec.tid() == outputId;
        });
    auto outputOffset = outputSpecIter - outputSpecs_.begin();
    outputSpecs_[outputOffset].set_compute_inplace();
    outputSpecs_[outputOffset].set_input_tensor_index(inputOffset);
  }

  return compilation;
}

void LlgaKernel::run(Stack& stack) {
#ifdef GRAPH_DEBUG_ENABLED
  GRAPH_DEBUG("In ", debugName(), "\n");
#endif

  // Grab input values from stack
  auto stackInputs = last(stack, nGraphInputs_);
  auto inputs = fmap(stackInputs, [&](const IValue& v) {
    TORCH_CHECK(
        v.isTensor(), "Stack values for LLGA partition must be Tensor type");
    return v.toTensor();
  });

  // Even in case of concurrent threads, the kernel would be initialized once.
  // TODO: Try not using an atomic lock
  c10::call_once(
      initialized_flag,
      [&](const TensorArgs& inputs) {
        GRAPH_DEBUG("Initializing input logical tensors");
        inputSpecs_ = initializeInputSpecs(inputs);
        GRAPH_DEBUG("Initializing output logical tensors");
        outputSpecs_ = initializeOutputSpecs();
        GRAPH_DEBUG("Compiling partition");
        compilation_ = compile(partition_);
        is_initialized_ = true;
      },
      inputs);
#ifdef GRAPH_DEBUG_ENABLED
  GRAPH_DEBUG("Preparing runtime tensors");
#endif
  TensorArgs outputs;
  auto [runInputs, runOutputs] = prepareRunArgs(inputs, outputs);
#ifdef GRAPH_DEBUG_ENABLED
  GRAPH_DEBUG("Executing partition");
#endif
  compilation_.execute(Stream::getStream(), runInputs, runOutputs);
#ifdef GRAPH_DEBUG_ENABLED
  GRAPH_DEBUG("Partition executed");
#endif

  // Update the stack.
  drop(stack, nGraphInputs_);
  for (auto& o : outputs)
    push_one(stack, std::move(o));
#ifdef GRAPH_DEBUG_ENABLED
  GRAPH_DEBUG("Stack updated");
#endif
}

} // namespace torch::jit::fuser::onednn

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 16 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `dnnl`, `torch`

**Classes/Structs**: `of`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/codegen/onednn`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/codegen/onednn/graph_helper.h`
- `torch/csrc/jit/codegen/onednn/kernel.h`
- `ATen/core/functional.h`
- `torch/csrc/jit/jit_log.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`torch/csrc/jit/codegen/onednn`):

- [`graph_rewriter.cpp_docs.md`](./graph_rewriter.cpp_docs.md)
- [`guard_shape.cpp_docs.md`](./guard_shape.cpp_docs.md)
- [`prepare_binary.h_docs.md`](./prepare_binary.h_docs.md)
- [`graph_fuser.h_docs.md`](./graph_fuser.h_docs.md)
- [`kernel.h_docs.md`](./kernel.h_docs.md)
- [`decompose_silu.cpp_docs.md`](./decompose_silu.cpp_docs.md)
- [`prepare_binary.cpp_docs.md`](./prepare_binary.cpp_docs.md)
- [`graph_helper.cpp_docs.md`](./graph_helper.cpp_docs.md)
- [`register_interface.cpp_docs.md`](./register_interface.cpp_docs.md)


## Cross-References

- **File Documentation**: `kernel.cpp_docs.md`
- **Keyword Index**: `kernel.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
