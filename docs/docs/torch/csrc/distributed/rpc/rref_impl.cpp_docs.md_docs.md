# Documentation: `docs/torch/csrc/distributed/rpc/rref_impl.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/rpc/rref_impl.cpp_docs.md`
- **Size**: 13,235 bytes (12.92 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/rpc/rref_impl.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/rpc/rref_impl.cpp`
- **Size**: 10,254 bytes (10.01 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/distributed/rpc/rref_impl.h>

#include <ATen/record_function.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <fmt/format.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h>
#include <torch/csrc/distributed/autograd/utils.h>
#include <torch/csrc/distributed/rpc/profiler/remote_profiler_manager.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/distributed/rpc/rref_proto.h>
#include <torch/csrc/distributed/rpc/utils.h>

#include <utility>

namespace {
// If the type is subtype of named type, return its qualifiedname, otherwise
// return its type str.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
std::string getTypeStr(const c10::TypePtr& type) {
  switch (type->kind()) {
    case c10::TypeKind::FunctionType:
      return type->castRaw<c10::FunctionType>()->name()->qualifiedName();
    case c10::TypeKind::TupleType:
      return type->castRaw<c10::TupleType>()->name()->qualifiedName();
    case c10::TypeKind::ClassType:
      return type->castRaw<c10::ClassType>()->name()->qualifiedName();
    case c10::TypeKind::InterfaceType:
      return type->castRaw<c10::InterfaceType>()->name()->qualifiedName();
    default:
      return type->annotation_str();
  }
}
// NOLINTEND(bugprone-unchecked-optional-access)

} // namespace

namespace torch::distributed::rpc {

std::atomic<local_id_t> RRefContext::nextLocalId_{0};

//////////////////////////  RRefForkData  /////////////////////////////////

RRefForkData::RRefForkData(
    worker_id_t ownerId,
    const RRefId& rrefId,
    const ForkId& forkId,
    worker_id_t parent,
    std::string typeStr)
    : ownerId_(ownerId),
      rrefId_(rrefId),
      forkId_(forkId),
      parent_(parent),
      typeStr_(std::move(typeStr)) {}

//////////////////////////////  RRef  /////////////////////////////////////

RRef::RRef(worker_id_t ownerId, const RRefId& rrefId, TypePtr type)
    : ownerId_(ownerId), rrefId_(rrefId), type_(std::move(type)) {}

RRefForkData RRef::fork() const {
  auto& ctx = RRefContext::getInstance();
  return RRefForkData(
      ownerId_,
      rrefId_,
      ctx.genGloballyUniqueId(),
      ctx.getWorkerId(),
      getTypeStr(type_));
}

void RRef::handleError(RPCErrorType errorType, const JitFuture& jitFuture) {
  static std::unordered_map<
      RPCErrorType,
      std::function<void(const JitFuture& jitFuture)>,
      std::hash<int>>
      errorHandlers = {
          {RPCErrorType::TIMEOUT,
           [this](const JitFuture& /* unused */) { setTimedOut(); }},
          {RPCErrorType::INTENTIONAL_FAILURE,
           [this](const JitFuture& /* unused */) { setTimedOut(); }},
          {RPCErrorType::UNKNOWN_ERROR, [](const JitFuture& jitFuture) {
             // Default error handler
             RRefContext::handleException(jitFuture);
           }}};
  errorHandlers.find(errorType)->second(jitFuture);
}

//////////////////////////  UserRRef  /////////////////////////////////////

UserRRef::UserRRef(
    worker_id_t ownerId,
    const RRefId& rrefId,
    const ForkId& forkId,
    TypePtr type)
    : RRef(ownerId, rrefId, std::move(type)),
      forkId_(forkId),
      confirmedByOwner_(false) {
  // Do nothing,
  // (1) If this UserRRef is a fork of an existing RRef, RRefContext will send
  //     a RREF_FORK_REQUEST message to the owner.
  // (2) If this the creator UserRRef, ScriptRemoteCall or PythonRemoteCall will
  //     properly notify the owner.
}

void UserRRef::tryDel() {
  std::lock_guard<std::mutex> lockGuard(deletedOnOwnerMutex_);
  if (!deletedOnOwner_) {
    try {
      RRefContext::getInstance().delUser(ownerId_, rrefId_, forkId_);
      deletedOnOwner_ = true;
    } catch (const std::exception& ex) {
      LOG(ERROR) << "Error occurred when deleting" << *this << " : "
                 << ex.what();
    } catch (...) {
      LOG(ERROR) << "Error occurred when deleting" << *this << " : "
                 << "unknown error";
    }
  }
}

UserRRef::~UserRRef() {
  tryDel();
}

void UserRRef::release_resources() {
  tryDel();
}

const ForkId& UserRRef::forkId() const {
  return forkId_;
}

IValue UserRRef::toHere(const float timeoutSeconds) const {
  TORCH_CHECK(
      !getTimedOut(),
      "RRef creation via rpc.remote() timed out, and it "
      "is possible that the RRef on the owner node does not exist.");
  // see Note [Best-Effort Check on Deleted UserRRefs]
  TORCH_CHECK(
      !deletedOnOwner_,
      *this,
      " has been deleted. Cannot call to_here() on it after deletion.");
  auto toHereKey = std::string("");
  if (torch::autograd::profiler::profilerEnabled()) {
    toHereKey = fmt::format(
        "to_here#({})->({})",
        RpcAgent::getCurrentRpcAgent()->getWorkerInfo().name_,
        RpcAgent::getCurrentRpcAgent()->getWorkerInfo(ownerId_).name_);
  }
  RECORD_USER_SCOPE(toHereKey);
  TORCH_CHECK(
      !type_->is_module(),
      *this,
      " is an RRef to a ScriptModule. "
      "It can't be sent through RPC "
      "from owner, ",
      ownerWorkerInfo(),
      ", to user, ",
      RpcAgent::getCurrentRpcAgent()->getWorkerInfo(),
      ".");

  auto agent = RpcAgent::getCurrentRpcAgent();

  // ScriptRRefFetchCall message always carries autograd context id even if
  // the message itself does not contain any tensor, because the response would
  // potentially contain tensors.
  c10::intrusive_ptr<Message> msgToSend;

  if (isPyObj()) {
    msgToSend = PythonRRefFetchCall(ownerId_, rrefId()).toMessage();
  } else {
    msgToSend = ScriptRRefFetchCall(ownerId_, rrefId()).toMessage();
  }

  // toHere is profiled as a blocking call, and does not execute operations on
  // the remote node. Hence, don't wrap it with a profiling message since we
  // don't need the profiler to be enabled remotely.
  auto jitFuture = autograd::sendMessageWithAutograd(
      *agent,
      agent->getWorkerInfo(ownerId_),
      std::move(msgToSend),
      true /* forceGradRecording */,
      timeoutSeconds,
      true /* forceDisableProfiling */);

  // TODO: we should ideally be able to interrupt this blocking wait if we check
  // getTimedOut() and it is true
  // (https://github.com/pytorch/pytorch/issues/39411).
  jitFuture->waitAndThrow();
  auto messagePtr = jitFuture->constValue().toCustomClass<Message>();
  MessageType msgType = messagePtr->type();
  auto response = deserializeResponse(*messagePtr, msgType);
  TORCH_INTERNAL_ASSERT(
      msgType == MessageType::SCRIPT_RREF_FETCH_RET ||
          msgType == MessageType::PYTHON_RREF_FETCH_RET,
      "Message type should either be SCRIPT_RREF_FETCH_RET "
      "or PYTHON_RREF_FETCH_RET");
  RpcCommandBase& rpc = *response;
  auto& rrefFetchRet = static_cast<RRefFetchRet&>(rpc);
  if (isPyObj()) {
    // wrap python serialized vector of ivalues into tuple, this
    // made the C++ toHere interface to return single IValue
    return ivalue::Tuple::create(rrefFetchRet.values());
  } else {
    return rrefFetchRet.values().front();
  }
}

RRefForkData UserRRef::fork() const {
  // Note [Best-Effort Check on Deleted UserRRefs]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // This check does not guarantee correctness, as there could be another thread
  // trying to delete this UserRRef concurrently. Passing this check does not
  // mean this RRef will be alive throughout this function. This is just our
  // best-effort attempt to raise proper error messages. The behavior of using
  // deleted UserRRefs is undefined.
  //
  // The reason for not implementing strict checks are:
  // 1. This would need to acquire lock on deletedOnOwnerMutex_, which would
  //    introduce unnecessary overhead for most normal use cases.
  // 2. This would introduce a lot of complexities to get the behavior correct.
  //    Assume we acquired the lock here, and there is another thread X block
  //    waiting in tryDel() on the lock. Exiting this fork function would
  //    unblock thread X. However, while X proceeds with deleting this UserRRef,
  //    the call site of fork() might have added the UserRRef to
  //    pendingChildren_ map, but up to this point, nothing prevents X from
  //    deleting this RRef even if it shouldn't do so due to the state change
  //    in pendingChildren_. We might be able to get it right for now by locking
  //    and checking pendingChildren_ in X, but the gain does not seem to
  //    worth the complexity.
  TORCH_CHECK(
      !deletedOnOwner_,
      *this,
      " has been deleted. Cannot call fork an UserRRef after deletion.");
  return RRef::fork();
}

//////////////////////////  OwnerRRef  /////////////////////////////////////

OwnerRRef::OwnerRRef(
    worker_id_t ownerId,
    const RRefId& rrefId,
    TypePtr type,
    std::vector<c10::Device> devices)
    : OwnerRRef(
          ownerId,
          rrefId,
          std::move(type),
          /* value */ {},
          std::move(devices)) {}

OwnerRRef::OwnerRRef(
    worker_id_t ownerId,
    const RRefId& rrefId,
    TypePtr type,
    std::optional<IValue> value,
    std::vector<c10::Device> devices)
    : RRef(ownerId, rrefId, std::move(type)) {
  future_ = c10::make_intrusive<JitFuture>(type_, std::move(devices));

  if (value.has_value()) {
    future_->markCompleted(value.value());
  }
}

const IValue& OwnerRRef::getValue() const {
  TORCH_CHECK(
      !getTimedOut(),
      "RRef creation via rpc.remote() timed out, and it "
      "is possible that the RRef on the owner node does not exist.");
  future_->waitAndThrow();
  return future_->constValue();
}

bool OwnerRRef::hasValue() const {
  return future_->completed();
}

c10::intrusive_ptr<JitFuture> OwnerRRef::getFuture() {
  return future_;
}

void OwnerRRef::setValue(IValue&& value) {
  future_->markCompleted(std::move(value));
}

void OwnerRRef::setError(std::exception_ptr eptr) {
  future_->setErrorIfNeeded(std::move(eptr));
}

std::ostream& operator<<(std::ostream& os, const RRef& rref) {
  if (rref.isOwner()) {
    return os << "OwnerRRef("
              << "rref_id=" << rref.rrefId() << ")";
  } else {
    return os << "UserRRef("
              << "rref_id=" << rref.rrefId()
              << ", fork_id=" << static_cast<const UserRRef*>(&rref)->forkId()
              << ")";
  }
}

} // namespace torch::distributed::rpc

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `namespace`, `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/rpc`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/rpc/rref_impl.h`
- `ATen/record_function.h`
- `c10/core/impl/DeviceGuardImplInterface.h`
- `fmt/format.h`
- `torch/csrc/distributed/autograd/rpc_messages/rpc_with_autograd.h`
- `torch/csrc/distributed/autograd/utils.h`
- `torch/csrc/distributed/rpc/profiler/remote_profiler_manager.h`
- `torch/csrc/distributed/rpc/rref_context.h`
- `torch/csrc/distributed/rpc/rref_proto.h`
- `torch/csrc/distributed/rpc/utils.h`
- `utility`


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

Files in the same folder (`torch/csrc/distributed/rpc`):

- [`request_callback.cpp_docs.md`](./request_callback.cpp_docs.md)
- [`python_rpc_handler.cpp_docs.md`](./python_rpc_handler.cpp_docs.md)
- [`tensorpipe_agent.h_docs.md`](./tensorpipe_agent.h_docs.md)
- [`torchscript_functions.cpp_docs.md`](./torchscript_functions.cpp_docs.md)
- [`utils.cpp_docs.md`](./utils.cpp_docs.md)
- [`unpickled_python_call.cpp_docs.md`](./unpickled_python_call.cpp_docs.md)
- [`request_callback.h_docs.md`](./request_callback.h_docs.md)
- [`rref_context.cpp_docs.md`](./rref_context.cpp_docs.md)
- [`request_callback_impl.h_docs.md`](./request_callback_impl.h_docs.md)
- [`py_rref.h_docs.md`](./py_rref.h_docs.md)


## Cross-References

- **File Documentation**: `rref_impl.cpp_docs.md`
- **Keyword Index**: `rref_impl.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/distributed/rpc`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/distributed/rpc`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/distributed/rpc`):

- [`script_resp.cpp_docs.md_docs.md`](./script_resp.cpp_docs.md_docs.md)
- [`python_rpc_handler.cpp_docs.md_docs.md`](./python_rpc_handler.cpp_docs.md_docs.md)
- [`tensorpipe_utils.h_kw.md_docs.md`](./tensorpipe_utils.h_kw.md_docs.md)
- [`request_callback_impl.h_docs.md_docs.md`](./request_callback_impl.h_docs.md_docs.md)
- [`types.cpp_docs.md_docs.md`](./types.cpp_docs.md_docs.md)
- [`utils.cpp_docs.md_docs.md`](./utils.cpp_docs.md_docs.md)
- [`rref_impl.h_kw.md_docs.md`](./rref_impl.h_kw.md_docs.md)
- [`rpc_agent.cpp_kw.md_docs.md`](./rpc_agent.cpp_kw.md_docs.md)
- [`request_callback_impl.cpp_kw.md_docs.md`](./request_callback_impl.cpp_kw.md_docs.md)
- [`script_call.cpp_docs.md_docs.md`](./script_call.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `rref_impl.cpp_docs.md_docs.md`
- **Keyword Index**: `rref_impl.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
