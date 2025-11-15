# Documentation: `docs/test/cpp/rpc/e2e_test_base.h_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/rpc/e2e_test_base.h_docs.md`
- **Size**: 8,358 bytes (8.16 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/rpc/e2e_test_base.h`

## File Metadata

- **Path**: `test/cpp/rpc/e2e_test_base.h`
- **Size**: 5,578 bytes (5.45 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```c
#include <gtest/gtest.h>

#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/context/context.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/script_call.h>
#include <torch/csrc/distributed/rpc/script_remote_call.h>
#include <torch/csrc/distributed/rpc/script_resp.h>
#include <torch/csrc/distributed/rpc/utils.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace torch {
namespace distributed {
namespace rpc {

using torch::distributed::autograd::DistAutogradContainer;
using torch::distributed::autograd::DistAutogradContext;

DistAutogradContainer* getDistAutogradContainer();

class TestE2EBase : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup distributed autograd.
    autogradContainer = getDistAutogradContainer();

    // Setup server store.
    c10d::TCPStoreOptions opts{
        /* port */ 0,
        /* isServer */ true,
        numWorkers,
        /* waitWorkers */ true,
        /* timeout */ std::chrono::seconds(10)};

    store = c10::make_intrusive<c10d::TCPStore>(serverAddress, opts);

    buildRpcAgent();

    rpcAgentPostProcessing();
  }

  void rpcAgentPostProcessing() {
    RpcAgent::setCurrentRpcAgent(rpcAgent);
    std::shared_ptr<TypeResolver> typeResolver =
        std::make_shared<TypeResolver>([&](const c10::QualifiedName& qn) {
          // For Dict that is used for device map.
          auto pos = qn.name().find("Dict");
          if (pos != std::string::npos) {
            return c10::StrongTypePtr(
                nullptr,
                c10::DictType::create(
                    c10::StringType::get(), c10::StringType::get()));
          }
          return c10::StrongTypePtr(
              nullptr, c10::TensorType::create(at::Tensor()));
        });
    rpcAgent->setTypeResolver(typeResolver);
    rpcAgent->start();
  }

  void TearDown() override {
    rpcAgent->join();
    rpcAgent->shutdown();
    RpcAgent::setCurrentRpcAgent(nullptr);
  }

  c10::intrusive_ptr<OwnerRRef> createRemoteRRef(
      at::Tensor t1,
      at::Tensor t2,
      std::shared_ptr<torch::jit::Operator> op) {
    auto& ctx = RRefContext::getInstance();
    auto ownerRRef = ctx.createOwnerRRef(c10::TensorType::create(t1));
    // prevent this owner RRef being deleted due to other forks
    ctx.addSelfAsFork(ownerRRef);

    ScriptRemoteCall scriptRemoteCall(
        op, {t1, t2, 1}, ownerRRef->rrefId(), ownerRRef->rrefId());
    auto jitFuture = autograd::sendMessageWithAutograd(
        *rpcAgent,
        rpcAgent->getWorkerInfo("worker"),
        std::move(scriptRemoteCall).toMessage(),
        false);

    ownerRRef->registerOwnerCreationFuture(jitFuture);

    // Builtin operators does not return py::object, and hence does not require
    // GIL for destructing the potentially deleted OwerRRef.
    jitFuture->addCallback(
        [ownerRRefId = ownerRRef->rrefId()](JitFuture& jitFuture) {
          callback::finishCreatingOwnerRRef(jitFuture, ownerRRefId);
        });
    return ownerRRef;
  }

  at::Tensor remoteAdd(
      at::Tensor t1,
      at::Tensor t2,
      std::shared_ptr<torch::jit::Operator> op) {
    ScriptCall scriptCall(op, {t1, t2, /* alpha */ 1});

    // Send the RPC and return result.
    auto response = autograd::sendMessageWithAutograd(
        *rpcAgent,
        rpcAgent->getWorkerInfo("worker"),
        std::move(scriptCall).toMessage());
    response->waitAndThrow();

    MessageType messageType = MessageType::FORWARD_AUTOGRAD_RESP;
    auto wrappedResponse = deserializeResponse(
        std::move(*response->value().toCustomClass<Message>()), messageType);
    return static_cast<ScriptResp&>(*wrappedResponse).value().toTensor();
  }

  virtual void buildRpcAgent() = 0;

  class AutogradContextGuard {
   public:
    explicit AutogradContextGuard()
        : context(DistAutogradContainer::getInstance().newContext()) {}

    ~AutogradContextGuard() {
      DistAutogradContainer::getInstance().releaseContext(context->contextId());
    }

   private:
    std::shared_ptr<DistAutogradContext> context;
  };

  void runTrainingLoop() {
    auto options = at::TensorOptions().requires_grad(true);
    auto t1 = torch::ones({3, 3}, options);
    auto t2 = torch::ones({3, 3}, options);

    c10::OperatorName full_name("aten::add", "Tensor");
    auto matchedOp = torch::jit::findOperatorFor(full_name);
    ASSERT_TRUE(matchedOp);

    for (size_t i = 0; i < numIters; i++) {
      // Create the autograd context guard.
      AutogradContextGuard guard;

      // Multiple RPCs within one autograd context for the forward pass.
      auto result = remoteAdd(t1, t2, matchedOp);
      for (size_t j = 0; j < 5; j++) {
        result = remoteAdd(t1, result, matchedOp);
      }

      auto rref = createRemoteRRef(t1, result, matchedOp);
      result = rref->getValue().toTensor();

      // Run backward pass now.
      autograd::DistEngine::getInstance().execute(
          DistAutogradContainer::currentContextId(),
          {torch::sum(result)},
          /* retainGraph */ false);
    }
  }

  DistAutogradContainer* autogradContainer;
  std::shared_ptr<RpcAgent> rpcAgent;
  static const size_t numIters;
  static const size_t numWorkers;
  c10::intrusive_ptr<c10d::Store> store;
  static const char* serverAddress;
};

} // namespace rpc
} // namespace distributed
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `distributed`, `torch`, `rpc`

**Classes/Structs**: `TestE2EBase`, `AutogradContextGuard`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/rpc`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `torch/csrc/distributed/autograd/context/container.h`
- `torch/csrc/distributed/autograd/context/context.h`
- `torch/csrc/distributed/autograd/engine/dist_engine.h`
- `torch/csrc/distributed/autograd/utils.h`
- `torch/csrc/distributed/c10d/TCPStore.hpp`
- `torch/csrc/distributed/rpc/rref_context.h`
- `torch/csrc/distributed/rpc/script_call.h`
- `torch/csrc/distributed/rpc/script_remote_call.h`
- `torch/csrc/distributed/rpc/script_resp.h`
- `torch/csrc/distributed/rpc/utils.h`
- `torch/csrc/jit/runtime/operator.h`


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

This is a test file. Run it with:

```bash
python test/cpp/rpc/e2e_test_base.h
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/rpc`):

- [`test_wire_serialization.cpp_docs.md`](./test_wire_serialization.cpp_docs.md)
- [`test_e2e_tensorpipe.cpp_docs.md`](./test_e2e_tensorpipe.cpp_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`test_tensorpipe_serialization.cpp_docs.md`](./test_tensorpipe_serialization.cpp_docs.md)
- [`e2e_test_base.cpp_docs.md`](./e2e_test_base.cpp_docs.md)


## Cross-References

- **File Documentation**: `e2e_test_base.h_docs.md`
- **Keyword Index**: `e2e_test_base.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp/rpc`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp/rpc`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/cpp/rpc/e2e_test_base.h_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/rpc`):

- [`test_e2e_tensorpipe.cpp_kw.md_docs.md`](./test_e2e_tensorpipe.cpp_kw.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`e2e_test_base.cpp_kw.md_docs.md`](./e2e_test_base.cpp_kw.md_docs.md)
- [`test_e2e_tensorpipe.cpp_docs.md_docs.md`](./test_e2e_tensorpipe.cpp_docs.md_docs.md)
- [`test_wire_serialization.cpp_docs.md_docs.md`](./test_wire_serialization.cpp_docs.md_docs.md)
- [`test_tensorpipe_serialization.cpp_kw.md_docs.md`](./test_tensorpipe_serialization.cpp_kw.md_docs.md)
- [`test_tensorpipe_serialization.cpp_docs.md_docs.md`](./test_tensorpipe_serialization.cpp_docs.md_docs.md)
- [`e2e_test_base.cpp_docs.md_docs.md`](./e2e_test_base.cpp_docs.md_docs.md)
- [`test_wire_serialization.cpp_kw.md_docs.md`](./test_wire_serialization.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `e2e_test_base.h_docs.md_docs.md`
- **Keyword Index**: `e2e_test_base.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
