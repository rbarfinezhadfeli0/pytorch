# Documentation: `docs/torch/nativert/executor/OpKernel.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/nativert/executor/OpKernel.cpp_docs.md`
- **Size**: 7,614 bytes (7.44 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nativert/executor/OpKernel.cpp`

## File Metadata

- **Path**: `torch/nativert/executor/OpKernel.cpp`
- **Size**: 5,026 bytes (4.91 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/nativert/executor/OpKernel.h>

#include <fmt/ostream.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/Logging.h>

#include <c10/util/Enumerate.h>
#include <c10/util/StringUtil.h>
#include <c10/util/env.h>
#include <torch/nativert/executor/ExecutionFrame.h>

namespace torch::nativert {

c10::OperatorHandle getOperatorForTarget(
    std::string_view target,
    const Node* node) {
  // target could come as either "torch.ops.aten.add.default" or
  // "aten.add.default"
  std::vector<std::string_view> atoms = c10::split(target, '.');

  size_t numAtoms = atoms.size();
  if (numAtoms < 3) {
    TORCH_CHECK(false, "Invalid target: ", target);
  }

  const std::string_view ns = atoms[numAtoms - 3];
  const std::string_view opName = atoms[numAtoms - 2];
  const std::string_view overloadName = atoms[numAtoms - 1];

  const auto operatorName = fmt::format("{}::{}", ns, opName);
  std::string normalizedOverloadName;
  if (overloadName == "default") {
    normalizedOverloadName = "";
  } else {
    normalizedOverloadName = overloadName;
  }

  auto handle = c10::Dispatcher::singleton().findSchemaOrThrow(
      operatorName.c_str(), normalizedOverloadName.c_str());

  return handle;
}

std::string readableArgs(
    const c10::FunctionSchema& schema,
    const std::vector<c10::IValue>& stack) {
  const auto& schemaArgs = schema.arguments();
  std::stringstream ss;
  for (const auto& [i, arg] : c10::enumerate(stack)) {
    ss << "arg" << i << ' ' << schemaArgs[i].name() << ": " << arg.tagKind()
       << ' ';
    if (arg.isTensor()) {
      auto t = arg.toTensor();
      ss << t.dtype() << t.sizes() << t.device();
    } else if (arg.isTensorList()) {
      auto tl = arg.toTensorVector();
      ss << '[';
      for (const auto& t : tl) {
        ss << t.dtype() << t.sizes() << t.device() << ", ";
      }
      ss << ']';
    } else if (arg.isNone()) {
      // pass
    } else {
      ss << arg;
    }
    ss << "\n";
  }
  return ss.str();
}

const bool OpKernel::blockingEnabled_ =
    c10::utils::get_env("CUDA_LAUNCH_BLOCKING").value_or("0") == "1";

void OpKernel::compute(ExecutionFrame& executionFrame) const {
  VLOG(2) << "Executing: " << *node_;

  computeInternal(executionFrame);

  VLOG(2) << "Completed: " << *node_;
}

Arguments prefillStackWithStaticArgs(
    const Node* node,
    const c10::FunctionSchema& schema) {
  std::vector<c10::IValue> stackWithStaticArgs;
  std::vector<Value*> dynamicArgs;
  const auto& schemaArgs = schema.arguments();
  stackWithStaticArgs.resize(schemaArgs.size());
  dynamicArgs.resize(schemaArgs.size());

  // initialized stackWithStaticArgs_ with static inputs
  for (const auto& [idx, schemaArg] : c10::enumerate(schemaArgs)) {
    const auto& argName = schemaArg.name();

    // Check if this is a dynamic input to the op.
    const auto input = node->tryGetInput(argName);
    if (input != nullptr) {
      stackWithStaticArgs.at(idx) = c10::IValue();
      dynamicArgs.at(idx) = input->value;
      continue;
    }

    // Check if this is a statically known input to the op.
    const auto attribute = node->tryGetAttribute(argName);
    if (attribute != nullptr) {
      stackWithStaticArgs.at(idx) = constantToIValue(attribute->value);
      continue;
    }

    // Otherwise, it must have a default value
    auto defaultValueOpt = schemaArg.default_value();
    if (defaultValueOpt.has_value()) {
      stackWithStaticArgs.at(idx) = defaultValueOpt.value();
      continue;
    }

    TORCH_CHECK(
        false,
        "Cannot initialize argument ",
        argName,
        " for node ",
        *node,
        " with schema ",
        schema);
  }
  return Arguments{std::move(stackWithStaticArgs), std::move(dynamicArgs)};
}

void fillDynamicInputs(
    const ExecutionFrame& executionFrame,
    const Arguments& arguments,
    std::vector<c10::IValue>& stack) {
  // fill the stack with dynamic values from execution frame,
  // including tensor, tensors, symint, symints

  for (auto [idx, value] : arguments.getDynamicArgs()) {
    TORCH_CHECK(
        idx < stack.size(),
        "Invalid index",
        idx,
        " for stack size ",
        stack.size());
    TORCH_CHECK(stack.at(idx).isNone(), "Encountered None at index ", idx);
    if (value->type() == Type::Kind::TensorList) {
      // TODO: This is for passing List<Tensor> as an input to op that takes a
      // List<Optional<Tensor>>.
      // Need to cast it to a vector and back to a list, otherwise will get
      // list covariance problems where List<Tensor> is not a subtype
      // of List<Optional<Tensor>> when trying to execute aten.index.Tensor.
      // Our lists should be covariant because they are static,
      // but IValues don't know that :(
      stack[idx] = executionFrame.getIValue(value->id()).toTensorList().vec();
    } else if (value->type() == Type::Kind::None) {
      stack[idx] = c10::IValue();
    } else {
      stack[idx] = executionFrame.getIValue(value->id());
    }
  }
}

} // namespace torch::nativert

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 9 function(s).

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

- `torch/nativert/executor/OpKernel.h`
- `fmt/ostream.h`
- `ATen/core/dispatch/Dispatcher.h`
- `c10/util/Logging.h`
- `c10/util/Enumerate.h`
- `c10/util/StringUtil.h`
- `c10/util/env.h`
- `torch/nativert/executor/ExecutionFrame.h`


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

Files in the same folder (`torch/nativert/executor`):

- [`AOTInductorDelegateExecutor.cpp_docs.md`](./AOTInductorDelegateExecutor.cpp_docs.md)
- [`ExecutionFrame.cpp_docs.md`](./ExecutionFrame.cpp_docs.md)
- [`ParallelGraphExecutor.h_docs.md`](./ParallelGraphExecutor.h_docs.md)
- [`ExecutionFrame.h_docs.md`](./ExecutionFrame.h_docs.md)
- [`ExecutorConfig.h_docs.md`](./ExecutorConfig.h_docs.md)
- [`SerialGraphExecutor.h_docs.md`](./SerialGraphExecutor.h_docs.md)
- [`Weights.cpp_docs.md`](./Weights.cpp_docs.md)
- [`OpKernelKind.h_docs.md`](./OpKernelKind.h_docs.md)
- [`Executor.h_docs.md`](./Executor.h_docs.md)


## Cross-References

- **File Documentation**: `OpKernel.cpp_docs.md`
- **Keyword Index**: `OpKernel.cpp_kw.md`
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

- **File Documentation**: `OpKernel.cpp_docs.md_docs.md`
- **Keyword Index**: `OpKernel.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
