# Documentation: `docs/torch/nativert/executor/Weights.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/nativert/executor/Weights.cpp_docs.md`
- **Size**: 19,044 bytes (18.60 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nativert/executor/Weights.cpp`

## File Metadata

- **Path**: `torch/nativert/executor/Weights.cpp`
- **Size**: 16,283 bytes (15.90 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp

#include <c10/util/Logging.h>
#include <utility>

#include <torch/csrc/export/pt2_archive_constants.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/serialization/import_read.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/nativert/executor/Weights.h>
#include <unordered_map>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/ops/scalar_tensor.h>
#endif

#include <caffe2/serialize/inline_container.h>

namespace torch::nativert {

WeightVersion Weights::globalVersion_ = 0;

Weights::Weights(
    const Graph* graph,
    const std::optional<std::unordered_map<std::string, c10::IValue>>&
        stateDict,
    const std::optional<std::unordered_map<std::string, c10::IValue>>&
        constants)
    : graph_(graph),
      weightsMeta_(graph->weightsMeta()),
      version_(globalVersion_++) {
  if (stateDict.has_value()) {
    loadStateDict(stateDict.value());
  }
  if (constants.has_value()) {
    for (const auto& [name, value] : constants.value()) {
      if (value.isTensor()) {
        allValues_[name] = value.toTensor();
      } else if (value.isCustomClass()) {
        customObjs_[name] = value;
      } else {
        TORCH_CHECK(false, "Unknown constant type: ", value.tagKind());
      }
    }
  }
}

Weights::Weights(
    const Graph* graph,
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> pytorchStreamReader,
    const std::unordered_map<std::string, std::string>& stateDictPaths,
    std::string_view stateDictPathPrefix,
    const std::unordered_map<std::string, std::string>& constantPaths,
    std::string_view constantPathPrefix,
    std::function<bool(const std::string&)> skipSizeCheck,
    std::function<bool(const std::string&)> skipDtypeCheck,
    std::shared_ptr<std::unordered_map<
        std::string,
        std::shared_ptr<torch::nativert::TensorMeta>>> maybeNewWeightsMeta)
    : graph_(graph),
      weightsMeta_(graph->weightsMeta()),
      version_(globalVersion_++),
      skipSizeCheck_(std::move(skipSizeCheck)),
      skipDtypeCheck_(std::move(skipDtypeCheck)) {
  auto loadAndInsert = [&](const std::string& tensorName,
                           std::string_view pathPrefix,
                           const std::unordered_map<std::string, std::string>&
                               tensorPaths,
                           bool isUsed,
                           std::shared_ptr<std::unordered_map<
                               std::string,
                               std::shared_ptr<torch::nativert::TensorMeta>>>
                               maybeNewWeightsMeta) {
    auto pathIt = tensorPaths.find(tensorName);
    TORCH_CHECK(
        pathIt != tensorPaths.end(),
        "Couldn't find ",
        tensorName,
        " in tensorPaths");

    const std::string tensorPath = std::string{pathPrefix} + pathIt->second;
    VLOG(1) << "Loading weight from: " << tensorPath;
    TORCH_CHECK(
        pytorchStreamReader->hasRecord(tensorPath), tensorPath, " not found");

    auto [tensorData, tensorDataSize] =
        pytorchStreamReader->getRecord(tensorPath);

    // TODO: We now have two copies of metadata for weights, one in
    // model definition /models/<model_name>.json, another in
    // /extra/xl_weights/<model_name>_model_param_config.json
    // Currently, we only use the metadata from model definition.
    std::optional<TensorMeta> tensorMeta;
    if (weightsMeta_.find(tensorName) != weightsMeta_.end()) {
      tensorMeta = weightsMeta_.at(tensorName);
    } else {
      TORCH_CHECK(
          false,
          "Tensor meta not found for: ",
          tensorName,
          " in base weights.");
    }
    std::optional<TensorMeta> newTensorMeta;
    if (maybeNewWeightsMeta) {
      if (stateDictPaths.find(tensorName) == stateDictPaths.end()) {
        TORCH_CHECK(false, "Tensor name not found in state dict paths");
      }

      std::string paramName = stateDictPaths.at(tensorName);
      if (maybeNewWeightsMeta->find(paramName) != maybeNewWeightsMeta->end()) {
        newTensorMeta = *maybeNewWeightsMeta->at(paramName);
      } else {
        TORCH_CHECK(
            false,
            "Tensor meta not found for: ",
            tensorName,
            " in new weights from: ",
            paramName);
      }
    }
    std::optional<TensorMeta> curTensorMeta =
        newTensorMeta ? newTensorMeta : tensorMeta;

    if (tensorDataSize == 0 && tensorMeta->numel() > 0) {
      VLOG(1) << "Tensor " << tensorName
              << " does not have data and create on Meta device";
      allValues_[tensorName] = at::empty_strided(
          curTensorMeta->sizes(),
          curTensorMeta->strides(),
          curTensorMeta->asTensorOptions().device(at::kMeta));
      return;
    }

    if (!isUsed) {
      VLOG(1) << "Tensor " << tensorName << " is not used during inference";
      auto targetDevice = curTensorMeta->device();
      allValues_[tensorName] =
          at::scalar_tensor(0, at::TensorOptions().device(targetDevice));
      return;
    }

    size_t bytesPerEntry =
        c10::scalarTypeToTypeMeta(curTensorMeta->dtype()).itemsize();
    auto device = tensorData.device();
    auto storage = c10::Storage(
        c10::Storage::use_byte_size_t(),
        at::detail::computeStorageNbytes(
            curTensorMeta->sizes(), curTensorMeta->strides(), bytesPerEntry),
        std::move(tensorData), // ownership is transferred
        nullptr,
        false);
    const auto tensorOptions = at::TensorOptions(device)
                                   .dtype(curTensorMeta->dtype())
                                   .requires_grad(false);
    auto tensor =
        at::empty({0}, tensorOptions)
            .set_(storage, 0, curTensorMeta->sizes(), curTensorMeta->strides());

    auto targetDevice = tensorMeta->device();
    VLOG(1) << "Loading weight " << tensorName << " on " << targetDevice;
    if (!isSameDevice(targetDevice, tensor.device())) {
      tensor = tensor.to(targetDevice);
    }
    if (tensor.dtype() != tensorMeta->dtype()) {
      tensor = tensor.to(tensorMeta->dtype());
    }

    allValues_[tensorName] = tensor;
  };

  auto loadAndInsertParamsBuffers = [&](const auto& tensorName, bool isUsed) {
    return loadAndInsert(
        std::string(tensorName),
        stateDictPathPrefix,
        stateDictPaths,
        isUsed,
        maybeNewWeightsMeta);
  };

  size_t weightIndex = 0;
  bool isUsed = true;
  const auto& weightValues = graph->weightValues();

  for (const auto& tensorName : graph->signature().parameters()) {
    isUsed = !weightValues[weightIndex]->users().empty();
    if (!isUsed) {
      unusedWeights_.insert(std::string(tensorName));
    }
    loadAndInsertParamsBuffers(tensorName, isUsed);
    weightIndex++;
  }
  for (const auto& tensorName : graph->signature().buffers()) {
    isUsed = !weightValues[weightIndex]->users().empty();
    if (!isUsed) {
      unusedWeights_.insert(std::string(tensorName));
    }
    loadAndInsertParamsBuffers(tensorName, isUsed);
    weightIndex++;
  }

  // Load tensor constants and custom object constants, they are both stored
  // in the same directory in the archive, i.e. "extra/constants/" tensor
  // constants are prefixed with "tensor_" custom objects are prefixed with
  // "custom_obj_"
  auto loadConstants = [&](const auto& constants) {
    for (const auto& constantName : constants) {
      auto pathIt = constantPaths.find(std::string(constantName));
      TORCH_CHECK(
          pathIt != constantPaths.end(),
          "Couldn't find ",
          constantName,
          " in constantPaths");
      auto& fileName = pathIt->second;

      if (c10::starts_with(
              fileName,
              torch::_export::archive_spec::TENSOR_CONSTANT_FILENAME_PREFIX)) {
        // tensor constants
        isUsed = !weightValues[weightIndex]->users().empty();
        if (!isUsed) {
          unusedWeights_.insert(std::string(constantName));
        }
        loadAndInsert(
            std::string(constantName),
            constantPathPrefix,
            constantPaths,
            isUsed,
            nullptr);
        weightIndex++;
      } else {
        TORCH_CHECK(false, "Unknown constant path: ", fileName);
      }
    }
  };
  loadConstants(graph->signature().nonPersistentBuffers());
  loadConstants(graph->signature().tensorConstants());

  // custom object constants
  for (const auto& customObjName : graph->signature().customObjs()) {
    auto pathIt = constantPaths.find(std::string(customObjName));
    TORCH_CHECK(
        pathIt != constantPaths.end(),
        "Couldn't find ",
        customObjName,
        " in constantPaths");
    auto& fileName = pathIt->second;

    if (!c10::starts_with(
            fileName,
            torch::_export::archive_spec::CUSTOM_OBJ_FILENAME_PREFIX)) {
      TORCH_CHECK(false, "Unknown constant path: ", fileName);
    }
    std::string customObjPath = std::string{constantPathPrefix} + fileName;
    LOG(INFO) << "Loading custom object from: " << customObjPath;

    TORCH_CHECK(
        pytorchStreamReader->hasRecord(customObjPath),
        customObjPath,
        " not found");

    const auto& [customObjData, customObjDataSize] =
        pytorchStreamReader->getRecord(customObjPath);

    const char* customObjDataPtr =
        reinterpret_cast<const char*>(customObjData.get());
    std::string customObjBytes(
        customObjDataPtr, customObjDataPtr + customObjDataSize);

    c10::IValue customObj = torch::jit::pickle_load_obj(customObjBytes);
    TORCH_CHECK(
        customObj.isCustomClass(), "Custom object is not a custom class");
    TORCH_CHECK(!customObj.isNone(), "Custom object is None");
    customObjs_[std::string(customObjName)] = std::move(customObj);
    customObjsPaths_[customObjPath] = std::string(customObjName);
  }
}

std::unordered_map<std::string, at::Tensor> Weights::parameters() const {
  std::unordered_map<std::string, at::Tensor> result;
  for (const auto& name : graph_->signature().parameters()) {
    result.emplace(name, allValues_.at(std::string(name)));
  }
  return result;
}

std::unordered_map<std::string, at::Tensor> Weights::buffers() const {
  std::unordered_map<std::string, at::Tensor> result;
  for (const auto& name : graph_->signature().buffers()) {
    result.emplace(name, allValues_.at(std::string(name)));
  }
  return result;
}

std::unordered_map<std::string, at::Tensor> Weights::attributes() const {
  return allValues_;
}

at::Tensor Weights::at(const std::string& name) const {
  auto it = allValues_.find(name);
  if (it != allValues_.end()) {
    return it->second;
  }

  TORCH_CHECK(false, name, " not found in Weights ", toString());
}

at::Tensor& Weights::at(const std::string& name) {
  auto it = allValues_.find(name);
  if (it != allValues_.end()) {
    return it->second;
  }

  TORCH_CHECK(false, name, " not found in Weights ", toString());
}

bool Weights::contains(const std::string& name) const {
  return allValues_.find(name) != allValues_.end();
}

c10::IValue Weights::getCustomObj(const std::string& name) const {
  auto it = customObjs_.find(name);
  if (it != customObjs_.end()) {
    return it->second;
  }

  TORCH_CHECK(false, "Custom objects ", name, " not found in Weights");
}

c10::IValue Weights::getCustomObjByFileName(const std::string& name) const {
  auto it = customObjsPaths_.find(name);
  TORCH_CHECK(
      it != customObjsPaths_.end(),
      "Custom objects with file name ",
      name,
      " not found in Weights");
  const std::string obj_name = it->second;
  return getCustomObj(obj_name);
}

void Weights::loadStateDict(
    const std::unordered_map<std::string, c10::IValue>& stateDict) {
  auto validateAndInsert = [&](const std::string& name) {
    auto stateDictIt = stateDict.find(name);
    TORCH_CHECK(
        stateDictIt != stateDict.end(),
        "Couldn't find ",
        name,
        " in stateDict");

    // Verify that the tensor matches the tensorMeta
    auto it = weightsMeta_.find(name);
    TORCH_CHECK(
        it != weightsMeta_.end(), "Couldn't find ", name, " in weightsMeta");

    auto targetDevice = it->second.device();
    auto tensor = stateDictIt->second.toTensor().to(targetDevice);

    TORCH_CHECK(tensor.sizes() == it->second.sizes());
    TORCH_CHECK(tensor.dtype() == it->second.dtype());

    allValues_.emplace(name, tensor);
  };

  for (const auto& name : graph_->signature().parameters()) {
    validateAndInsert(std::string(name));
  }
  for (const auto& name : graph_->signature().buffers()) {
    validateAndInsert(std::string(name));
  }
  // TensorConstants_ not filled !!
}

void Weights::validateValue(const std::string& name, const at::Tensor& newValue)
    const {
  validateValue(name, newValue, /*skipDeviceCheck=*/false);
}

void Weights::validateValue(
    const std::string& name,
    const at::Tensor& newValue,
    bool skipDeviceCheck) const {
  auto& weightMeta = weightsMeta_.at(name);

  TORCH_CHECK(
      weightMeta.sizes() == newValue.sizes() ||
          (skipSizeCheck_ && skipSizeCheck_(name)) ||
          unusedWeights_.find(name) != unusedWeights_.end(),
      "Mismatched sizes for ",
      name,
      ": ",
      weightMeta.sizes(),
      " vs ",
      newValue.sizes());
  TORCH_CHECK(
      weightMeta.dtype() == newValue.dtype() ||
          (skipDtypeCheck_ && skipDtypeCheck_(name)) ||
          unusedWeights_.find(name) != unusedWeights_.end(),
      "Mismatched dtype for ",
      name,
      ": ",
      weightMeta.dtype(),
      " vs ",
      newValue.dtype());

  if (!skipDeviceCheck) {
    auto targetDevice = weightMeta.device();
    if (targetDevice.is_cpu() && targetDevice.has_index()) {
      LOG(WARNING) << "Target device is cpu but has index: " << targetDevice;
    }
    TORCH_CHECK(
        isSameDevice(targetDevice, newValue.device()),
        "Mismatched device for ",
        name,
        ": ",
        targetDevice,
        " vs ",
        newValue.device());
  }
}

void Weights::setValue(const std::string& name, const at::Tensor& newValue) {
  setValue(name, newValue, /*skipDeviceCheck=*/false);
}

void Weights::setValue(
    const std::string& name,
    const at::Tensor& newValue,
    bool skipDeviceCheck) {
  if (allValues_.find(name) != allValues_.end()) {
    validateValue(name, newValue, skipDeviceCheck);
  } else {
    LOG(WARNING) << name << " is not found in the registered weights";
  }

  allValues_[name] = newValue;
}

void Weights::updateValue(const std::string& name, const at::Tensor& newValue) {
  auto it = allValues_.find(name);
  TORCH_CHECK(
      it != allValues_.end(), name, " not found in Weights ", toString());
  validateValue(name, newValue);

  it->second.copy_(newValue);
}

void Weights::updateValues(
    const std::unordered_map<std::string, at::Tensor>& newValues) {
  for (auto& [name, newValue] : newValues) {
    updateValue(name, newValue);
  }
}

std::string Weights::toString() const {
  std::stringstream ss;
  ss << '[';
  for (const auto& [name, _] : allValues_) {
    ss << name << ", ";
  }
  ss << ']';
  ss << '[';
  for (const auto& [name, _] : customObjs_) {
    ss << name << ", ";
  }
  ss << ']';
  return ss.str();
}

void Weights::validateAllWeightsLoaded() {
  auto checkNames = [&](const auto& names) {
    for (const auto& name : names) {
      if (unusedWeights_.find(std::string(name)) != unusedWeights_.end()) {
        continue;
      }
      auto it = allValues_.find(std::string(name));
      TORCH_CHECK(it != allValues_.end(), "Missing weight: ", name);
      TORCH_CHECK(it->second.defined(), "Weight not defined: ", name);
      if (it->second.device().is_meta()) {
        LOG(WARNING) << "Weight is on meta device: " << name;
      }
    }
  };
  checkNames(graph_->signature().parameters());
  checkNames(graph_->signature().buffers());
  checkNames(graph_->signature().nonPersistentBuffers());
  checkNames(graph_->signature().tensorConstants());
}

void Weights::updateFoldedConst(std::string_view name, c10::IValue tensor) {
  foldedConstsMap_[std::string{name}] = std::move(tensor);
}

const std::unordered_map<std::string, c10::IValue>& Weights::getFoldedConsts()
    const {
  return foldedConstsMap_;
}

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nativert/executor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/Logging.h`
- `utility`
- `torch/csrc/export/pt2_archive_constants.h`
- `torch/csrc/jit/serialization/import.h`
- `torch/csrc/jit/serialization/import_read.h`
- `torch/csrc/jit/serialization/pickle.h`
- `torch/nativert/executor/Weights.h`
- `unordered_map`
- `ATen/Functions.h`
- `ATen/ops/empty.h`
- `ATen/ops/empty_strided.h`
- `ATen/ops/scalar_tensor.h`
- `caffe2/serialize/inline_container.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/nativert/executor`):

- [`OpKernel.cpp_docs.md`](./OpKernel.cpp_docs.md)
- [`AOTInductorDelegateExecutor.cpp_docs.md`](./AOTInductorDelegateExecutor.cpp_docs.md)
- [`ExecutionFrame.cpp_docs.md`](./ExecutionFrame.cpp_docs.md)
- [`ParallelGraphExecutor.h_docs.md`](./ParallelGraphExecutor.h_docs.md)
- [`ExecutionFrame.h_docs.md`](./ExecutionFrame.h_docs.md)
- [`ExecutorConfig.h_docs.md`](./ExecutorConfig.h_docs.md)
- [`SerialGraphExecutor.h_docs.md`](./SerialGraphExecutor.h_docs.md)
- [`OpKernelKind.h_docs.md`](./OpKernelKind.h_docs.md)
- [`Executor.h_docs.md`](./Executor.h_docs.md)


## Cross-References

- **File Documentation**: `Weights.cpp_docs.md`
- **Keyword Index**: `Weights.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nativert/executor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nativert/executor`, which is part of the **core PyTorch library**.



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

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/nativert/executor`):

- [`GraphExecutorBase.h_docs.md_docs.md`](./GraphExecutorBase.h_docs.md_docs.md)
- [`Placement.h_kw.md_docs.md`](./Placement.h_kw.md_docs.md)
- [`ParallelGraphExecutor.cpp_kw.md_docs.md`](./ParallelGraphExecutor.cpp_kw.md_docs.md)
- [`ExecutionFrame.h_docs.md_docs.md`](./ExecutionFrame.h_docs.md_docs.md)
- [`Executor.cpp_kw.md_docs.md`](./Executor.cpp_kw.md_docs.md)
- [`SerialGraphExecutor.h_docs.md_docs.md`](./SerialGraphExecutor.h_docs.md_docs.md)
- [`AOTInductorModelContainerCudaShim.cpp_docs.md_docs.md`](./AOTInductorModelContainerCudaShim.cpp_docs.md_docs.md)
- [`ParallelGraphExecutor.h_docs.md_docs.md`](./ParallelGraphExecutor.h_docs.md_docs.md)
- [`Placement.cpp_kw.md_docs.md`](./Placement.cpp_kw.md_docs.md)
- [`DelegateExecutor.h_docs.md_docs.md`](./DelegateExecutor.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `Weights.cpp_docs.md_docs.md`
- **Keyword Index**: `Weights.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
