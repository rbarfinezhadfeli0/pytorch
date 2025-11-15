# Documentation: `docs/torch/csrc/lazy/ts_backend/ts_backend_impl.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/lazy/ts_backend/ts_backend_impl.cpp_docs.md`
- **Size**: 13,417 bytes (13.10 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/lazy/ts_backend/ts_backend_impl.cpp`

## File Metadata

- **Path**: `torch/csrc/lazy/ts_backend/ts_backend_impl.cpp`
- **Size**: 10,436 bytes (10.19 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/lazy/ts_backend/ts_backend_impl.h>

#include <ATen/Functions.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/lazy_graph_executor.h>
#include <torch/csrc/lazy/generated/LazyNativeFunctions.h>
#include <torch/csrc/lazy/ts_backend/config.h>
#include <torch/csrc/lazy/ts_backend/ir_builder.h>
#include <torch/csrc/lazy/ts_backend/ts_eager_fallback.h>
#include <torch/csrc/lazy/ts_backend/ts_lowering_context.h>
#include <memory>

namespace at {
// This function is defined in the codegenerated RegisterDispatchKey.cpp file.
// For the TorchScript backend, we have a special case where the registration
// does not happen immediately (at static initialization time), so that if an
// external backend is loaded, it has a chance to register itself, and
// TorchScript only registers itself if explicitly initialized
extern TORCH_API void RegisterTorchScriptLazyNativeFunctions();
extern TORCH_API void RegisterTorchScriptAutogradLazyNativeFunctions();
} // namespace at

namespace torch::lazy {

struct TSBackendDeviceType : public BackendDeviceType {
  TSBackendDeviceType() = delete;
  TSBackendDeviceType(c10::DeviceType deviceType)
      : BackendDeviceType((int8_t)deviceType) {
    TORCH_CHECK(deviceType == at::kCPU || deviceType == at::kCUDA);
  }

  std::string toString() const override {
    return c10::DeviceTypeName((c10::DeviceType)type);
  }

  c10::DeviceType c10Type() const {
    return (c10::DeviceType)type;
  }
};

class TSBackendImpl : public torch::lazy::BackendImplInterface {
 public:
  TSBackendImpl() {
    // TODO(whc) unify how all our flags are set and parsed as envs
    static bool env_use_cuda = c10::utils::has_env("LTC_TS_CUDA");
    auto type =
        (env_use_cuda || FLAGS_torch_lazy_ts_cuda) ? at::kCUDA : at::kCPU;
    default_device_type_ = std::make_shared<TSBackendDeviceType>(type);
  }

  const IrBuilder* GetIrBuilder() const override {
    static const IrBuilder* builder = new TorchScriptIrBuilder();
    return builder;
  }

  std::string CreateMetricReport() const override {
    return "TSBackendImpl: N/A";
  }

  std::unique_ptr<torch::lazy::LoweringContext> CreateLoweringContext(
      const std::string& name,
      torch::lazy::BackendDevice device,
      c10::ArrayRef<const torch::lazy::Node*> post_order,
      torch::lazy::Util::EmissionMap emit_status) const override {
    return std::make_unique<torch::lazy::TSLoweringContext>(
        name, device, post_order, emit_status);
  }

  std::unique_ptr<torch::lazy::LoweringContext> CreateLoweringContext(
      const std::string& name,
      torch::lazy::BackendDevice device) const override {
    return std::make_unique<torch::lazy::TSLoweringContext>(name, device);
  }

  std::vector<std::string> GetCompilationDevices(
      const std::string& device,
      c10::ArrayRef<std::string> devices) const override {
    return std::vector<std::string>(devices.begin(), devices.end());
  }

  at::Tensor MakeTensorFromComputationData(
      const torch::lazy::BackendDataPtr data,
      std::optional<at::ScalarType> logical_scalar_type) const override {
    const auto ts_data = std::static_pointer_cast<TSData>(data);
    return ts_data->data();
  }

  torch::lazy::BackendDataPtr MakeComputationDataFromTensor(
      const at::Tensor& tensor,
      const torch::lazy::Shape& shape,
      const torch::lazy::BackendDevice& device) const override {
    at::TensorOptions options = tensor.options().device(
        default_device_type_->c10Type(), device.ordinal());
    if (tensor.device().type() == default_device_type_->c10Type() &&
        default_device_type_->c10Type() == at::kCUDA) {
      return std::make_shared<TSData>(
          tensor.to(options, /*non_blocking=*/true), shape, device);
    } else if (tensor.device().type() == at::kCPU && tensor.numel() == 1) {
      // calling .item() on singleton cpu tensor is fast, and using fill is a
      // safe, async way to copy cpu to cuda for a single value
      auto device_tensor = at::full(tensor.sizes(), tensor.item(), options);
      return std::make_shared<TSData>(device_tensor, shape, device);
    } else {
      return std::make_shared<TSData>(
          tensor.to(options, /*non_blocking=*/false), shape, device);
    }
  }

  torch::lazy::BackendDataPtr MakeComputationDataFromScalar(
      const at::Scalar& scalar,
      const torch::lazy::BackendDevice& device) const override {
    return std::make_shared<TSData>(scalar, device);
  }

  torch::lazy::BackendDataPtr GetComputationDataFromNode(
      const Node* node) const override {
    auto* device_data_node = DeviceData::Cast(node);
    if (!device_data_node) {
      return nullptr;
    }
    return device_data_node->data();
  }

  std::string GetComputationBackendText(
      const torch::lazy::ComputationPtr computation) const override {
    auto ts_computation =
        static_cast<torch::lazy::TSComputation*>(computation.get());
    return ts_computation->graph()->toString();
  }

  //////////////computation client interfaces///////////////////////

 public:
  torch::lazy::BackendDataPtr CreateDataPlaceholder(
      const torch::lazy::BackendDevice& device,
      const torch::lazy::Shape& shape) const override;

  std::vector<torch::lazy::ComputationPtr> Compile(
      std::vector<torch::lazy::ComputationPtr> instances) const override;

  std::vector<torch::lazy::BackendDataPtr> ExecuteComputation(
      torch::lazy::ComputationPtr computation,
      c10::ArrayRef<torch::lazy::BackendDataPtr> arguments,
      const torch::lazy::BackendDevice& device) const override;

  std::shared_ptr<torch::lazy::BackendDeviceType> GetDefaultDeviceType()
      const override {
    return default_device_type_;
  }

  at::DeviceType EagerFallbackDeviceType() const override;

  void SetDefaultDeviceType(int8_t type) override {
    default_device_type_ = std::make_shared<TSBackendDeviceType>(
        static_cast<c10::DeviceType>(type));
  }

  int64_t GetDefaultDeviceOrdinal() const override {
    return default_device_ordinal_;
  }

  void SetDefaultDeviceOrdinal(int64_t ordinal) override {
    default_device_ordinal_ = ordinal;
  }

  std::vector<torch::lazy::BackendDevice> GetBackendDevices() const override;

  torch::lazy::BackendDevice GetBackendDevice(
      c10::Device device) const override;

  void SetRngSeed(size_t seed) const override {
    LOG(FATAL) << "Not implemented yet.";
  }

  // std::map<std::string, Metric> GetMetrics() const override { return {}; }

  // MemoryInfo GetMemoryInfo(const std::string& device) override {
  //   LOG(FATAL) << "Not implemented yet.";
  // }

  void PrepareToExit() const override;

 private:
  std::shared_ptr<TSBackendDeviceType> default_device_type_;
  int64_t default_device_ordinal_{0};
};

torch::lazy::BackendDataPtr TSBackendImpl::CreateDataPlaceholder(
    const torch::lazy::BackendDevice& device,
    const torch::lazy::Shape& shape) const {
  return std::make_shared<TSData>(shape, device);
}

std::vector<torch::lazy::ComputationPtr> TSBackendImpl::Compile(
    std::vector<torch::lazy::ComputationPtr> instances) const {
  for (const auto& instance : instances) {
    auto ts_computation =
        static_cast<torch::lazy::TSComputation*>(instance.get());
    if (!ts_computation->in_mark_step) {
      LOG(WARNING) << "Compile outside of mark step";
    }
  }
  return instances;
}

std::vector<torch::lazy::BackendDataPtr> TSBackendImpl::ExecuteComputation(
    torch::lazy::ComputationPtr computation,
    c10::ArrayRef<torch::lazy::BackendDataPtr> arguments,
    const torch::lazy::BackendDevice& device) const {
  auto ts_computation =
      std::dynamic_pointer_cast<torch::lazy::TSComputation>(computation);
  TORCH_CHECK(ts_computation, "Computation isn't TSComputation");
  torch::jit::GraphExecutor& graph_executor = ts_computation->graph_executor();
  std::vector<torch::jit::IValue> stack;
  for (const auto& argument : arguments) {
    const auto ts_data = std::static_pointer_cast<TSData>(argument);
    const auto& scalar = ts_data->scalar;
    if (scalar.has_value()) {
      stack.emplace_back(scalar.value());
    } else {
      // TODO(whc) should this check be made more general? it's written somewhat
      // oddly
      TORCH_CHECK(
          static_cast<c10::DeviceType>(default_device_type_->type) !=
              at::kCUDA ||
          ts_data->data().device().type() == at::kCUDA);
      stack.emplace_back(ts_data->data());
    }
  }
  graph_executor.run(stack);
  std::vector<torch::lazy::BackendDataPtr> results;
  for (torch::jit::IValue component : stack) {
    at::Tensor result = component.toTensor();
    at::IntArrayRef result_sizes = result.sizes();
    torch::lazy::Shape shape(
        result.scalar_type(),
        std::vector<int64_t>(result_sizes.begin(), result_sizes.end()));
    results.push_back(std::make_shared<TSData>(result, shape, device));
  }
  return results;
}

std::vector<torch::lazy::BackendDevice> TSBackendImpl::GetBackendDevices()
    const {
  std::vector<torch::lazy::BackendDevice> devices;
  // TODO(whc) figure out how to query available devices from pytorch
  devices.emplace_back(GetBackendDevice(c10::Device(c10::kCPU, 0)));
  devices.emplace_back(GetBackendDevice(c10::Device(c10::kCUDA, 0)));
  return devices;
}

torch::lazy::BackendDevice TSBackendImpl::GetBackendDevice(
    c10::Device device) const {
  // Note, we ignore the device type specified by the c10::Device since it is
  // expected to be a virtual device (lazy::), but we need to change this when
  // we support lazy as a mode
  return torch::lazy::BackendDevice(GetDefaultDeviceType(), device.index());
}

void TSBackendImpl::PrepareToExit() const {}

c10::DeviceType TSBackendImpl::EagerFallbackDeviceType() const {
  // For TS backend, hardware device _is_ eager device
  return (c10::DeviceType)GetDefaultDeviceType()->type;
}

torch::lazy::BackendImplInterface* GetTSBackendImpl() {
  static TSBackendImpl* ts_backend_impl = new TSBackendImpl();
  return ts_backend_impl;
}

void InitTorchScriptBackend() {
  at::RegisterTorchScriptLazyNativeFunctions();
  at::RegisterTorchScriptAutogradLazyNativeFunctions();
  register_ts_ltc_eager_fallback();
  static std::unique_ptr<BackendRegistrar> s_registrar;
  s_registrar = std::make_unique<BackendRegistrar>(GetTSBackendImpl());

  static LazyGraphExecutor* executor = new LazyGraphExecutor();
  LazyGraphExecutor::Register(executor);
}

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 29 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `at`

**Classes/Structs**: `TSBackendDeviceType`, `TSBackendImpl`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/lazy/ts_backend`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/lazy/ts_backend/ts_backend_impl.h`
- `ATen/Functions.h`
- `torch/csrc/lazy/backend/backend_device.h`
- `torch/csrc/lazy/core/lazy_graph_executor.h`
- `torch/csrc/lazy/generated/LazyNativeFunctions.h`
- `torch/csrc/lazy/ts_backend/config.h`
- `torch/csrc/lazy/ts_backend/ir_builder.h`
- `torch/csrc/lazy/ts_backend/ts_eager_fallback.h`
- `torch/csrc/lazy/ts_backend/ts_lowering_context.h`
- `memory`


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

Files in the same folder (`torch/csrc/lazy/ts_backend`):

- [`ts_node.h_docs.md`](./ts_node.h_docs.md)
- [`dynamic_ir.cpp_docs.md`](./dynamic_ir.cpp_docs.md)
- [`ts_backend_impl.h_docs.md`](./ts_backend_impl.h_docs.md)
- [`config.cpp_docs.md`](./config.cpp_docs.md)
- [`ts_autograd_functions.cpp_docs.md`](./ts_autograd_functions.cpp_docs.md)
- [`ts_eager_fallback.h_docs.md`](./ts_eager_fallback.h_docs.md)
- [`dynamic_ir.h_docs.md`](./dynamic_ir.h_docs.md)
- [`tensor_aten_ops.cpp_docs.md`](./tensor_aten_ops.cpp_docs.md)
- [`tensor_aten_ops.h_docs.md`](./tensor_aten_ops.h_docs.md)
- [`ts_lowering_context.cpp_docs.md`](./ts_lowering_context.cpp_docs.md)


## Cross-References

- **File Documentation**: `ts_backend_impl.cpp_docs.md`
- **Keyword Index**: `ts_backend_impl.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/lazy/ts_backend`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/lazy/ts_backend`, which is part of the **core PyTorch library**.



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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/lazy/ts_backend`):

- [`ts_native_functions.cpp_kw.md_docs.md`](./ts_native_functions.cpp_kw.md_docs.md)
- [`ts_native_functions.cpp_docs.md_docs.md`](./ts_native_functions.cpp_docs.md_docs.md)
- [`ts_autograd_functions.cpp_docs.md_docs.md`](./ts_autograd_functions.cpp_docs.md_docs.md)
- [`ir_builder.h_docs.md_docs.md`](./ir_builder.h_docs.md_docs.md)
- [`ts_node.cpp_docs.md_docs.md`](./ts_node.cpp_docs.md_docs.md)
- [`ts_lowering_context.h_kw.md_docs.md`](./ts_lowering_context.h_kw.md_docs.md)
- [`ts_lowering_context.cpp_kw.md_docs.md`](./ts_lowering_context.cpp_kw.md_docs.md)
- [`tensor_aten_ops.cpp_kw.md_docs.md`](./tensor_aten_ops.cpp_kw.md_docs.md)
- [`tensor_aten_ops.cpp_docs.md_docs.md`](./tensor_aten_ops.cpp_docs.md_docs.md)
- [`config.h_docs.md_docs.md`](./config.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `ts_backend_impl.cpp_docs.md_docs.md`
- **Keyword Index**: `ts_backend_impl.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
