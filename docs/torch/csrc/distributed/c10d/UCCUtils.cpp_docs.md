# Documentation: `torch/csrc/distributed/c10d/UCCUtils.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/UCCUtils.cpp`
- **Size**: 7,382 bytes (7.21 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#ifdef USE_C10D_UCC

#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/UCCTracing.hpp>
#include <torch/csrc/distributed/c10d/UCCUtils.hpp>
#include <cctype>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace c10d {

namespace {
// Constants for store keys.
constexpr char kTeamRank[] = "teamr";
constexpr char kAllGatherDone[] = "ag_done";
constexpr char kAllGatherFree[] = "ag_free";
} // namespace

ucc_status_t oob_allgather(
    void* sbuf,
    void* rbuf,
    size_t msglen,
    void* coll_info,
    void** req) {
  auto* info = reinterpret_cast<torch_ucc_oob_coll_info_t*>(coll_info);
  TORCH_CHECK(info != nullptr);
  std::vector<uint8_t> val = std::vector<uint8_t>(
      reinterpret_cast<uint8_t*>(sbuf),
      reinterpret_cast<uint8_t*>(sbuf) + msglen);
  try {
    info->store->set(info->getKey(kTeamRank + std::to_string(info->rank)), val);
    info->rbuf = rbuf;
    info->msglen = msglen;
    *req = coll_info;
  } catch (std::exception& ex) {
    LOG(ERROR) << "(oob_allgather) Caught exception in Store Operation .. "
               << "[" << ex.what() << "]";
    return UCC_ERR_NO_MESSAGE;
  }
  return UCC_OK;
}

ucc_status_t oob_allgather_test(void* req) {
  auto* info = reinterpret_cast<torch_ucc_oob_coll_info_t*>(req);
  TORCH_CHECK(info != nullptr);

  try {
    for (int r = 0; r < info->size; r++) {
      if (!info->store->check({info->getKey(kTeamRank + std::to_string(r))})) {
        return UCC_INPROGRESS;
      }
    }
    for (int r = 0; r < info->size; r++) {
      std::vector<uint8_t> data =
          info->store->get(info->getKey(kTeamRank + std::to_string(r)));
      memcpy(
          (void*)((ptrdiff_t)info->rbuf + info->msglen * r),
          data.data(),
          info->msglen);
    }
  } catch (std::exception& ex) {
    LOG(ERROR) << "(oob_allgather) Caught exception in Store Operation .. "
               << "[" << ex.what() << "]";
    return UCC_ERR_NO_MESSAGE;
  }
  return UCC_OK;
}

ucc_status_t oob_allgather_free(void* req) {
  auto* info = reinterpret_cast<torch_ucc_oob_coll_info_t*>(req);
  TORCH_CHECK(info != nullptr);
  try {
    int num_done = info->store->add({info->getKey(kAllGatherDone)}, 1);
    if (num_done == info->size) {
      info->store->deleteKey(info->getKey(kAllGatherDone));
      // Note: to avoid race condition, it's important to remove all keys in
      // oob_allgather_free first and only after that signal completion to
      // other ranks
      for (const auto r : c10::irange(info->size)) {
        info->store->deleteKey(info->getKey(kTeamRank + std::to_string(r)));
      }
      for (const auto r : c10::irange(info->size)) {
        info->store->add({info->getKey(kAllGatherFree + std::to_string(r))}, 1);
      }
    } else {
      info->store->wait(
          {info->getKey(kAllGatherFree + std::to_string(info->rank))});
    }
    info->store->deleteKey(
        info->getKey(kAllGatherFree + std::to_string(info->rank)));
  } catch (std::exception& ex) {
    LOG(ERROR) << "(oob_allgather) Caught exception in Store Operation .. "
               << "[" << ex.what() << "]";
    return UCC_ERR_NO_MESSAGE;
  }
  return UCC_OK;
}

CommUCC::CommUCC(
    std::shared_ptr<torch_ucc_oob_coll_info_t> oob,
    const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger)
    : CommBase(logger) {
  ucc_lib_config_h lib_config;
  ucc_context_config_h context_config;
  ucc_lib_params_t lib_params;
  ucc_context_params_t context_params;
  ucc_status_t st;

  TORCH_UCC_CHECK(
      ucc_lib_config_read("TORCH", nullptr, &lib_config),
      "failed to read UCC lib config");
  memset(&lib_params, 0, sizeof(ucc_lib_params_t));
  lib_params.mask = UCC_LIB_PARAM_FIELD_THREAD_MODE;
  lib_params.thread_mode = UCC_THREAD_MULTIPLE;
  TORCH_UCC_CHECK(
      ucc_init(&lib_params, lib_config, &lib), "failed to init UCC lib");
  ucc_lib_config_release(lib_config);
  ucc_lib_attr_t lib_attr;
  lib_attr.mask = UCC_LIB_ATTR_FIELD_THREAD_MODE;
  TORCH_UCC_CHECK(
      ucc_lib_get_attr(lib, &lib_attr), "failed to query for lib attr");
  TORCH_CHECK(
      lib_attr.thread_mode == UCC_THREAD_MULTIPLE,
      "ucc library wasn't initialized with multithreading support, "
      "please check ucc build options");
  st = ucc_context_config_read(lib, NULL, &context_config);
  if (st != UCC_OK) {
    // FIXME: would this cause deadlock if only one rank fails?
    TORCH_UCC_CHECK(
        ucc_finalize(lib),
        "failed to finalize UCC library when failing to read UCC context config");
    TORCH_UCC_LOG_ERROR(
        TORCH_UCC_INIT,
        c10::str("failed to read UCC context config: ", ucc_status_string(st)));
    TORCH_CHECK(false, ucc_status_string(st));
  }
  st = ucc_context_config_modify(
      context_config,
      NULL,
      "ESTIMATED_NUM_EPS",
      std::to_string(oob->size).c_str());
  if (st != UCC_OK) {
    ucc_context_config_release(context_config);
    ucc_finalize(lib);
    TORCH_UCC_LOG_ERROR(
        TORCH_UCC_INIT,
        c10::str(
            "UCC failed to modify UCC context config: ",
            ucc_status_string(st)));
    TORCH_CHECK(false, ucc_status_string(st));
  }
  memset(&context_params, 0, sizeof(ucc_context_params_t));
  context_params.mask =
      UCC_CONTEXT_PARAM_FIELD_TYPE | UCC_CONTEXT_PARAM_FIELD_OOB;
  context_params.type = UCC_CONTEXT_SHARED;
  context_params.oob.n_oob_eps = oob->size;
  context_params.oob.oob_ep = oob->rank;
  context_params.oob.allgather = oob_allgather;
  context_params.oob.req_test = oob_allgather_test;
  context_params.oob.req_free = oob_allgather_free;
  context_params.oob.coll_info = oob.get();
  st = ucc_context_create(lib, &context_params, context_config, &context);
  ucc_context_config_release(context_config);
  if (st != UCC_OK) {
    TORCH_UCC_CHECK(
        ucc_finalize(lib),
        "failed to finalize UCC library when failing to creat UCC context");
    TORCH_UCC_LOG_ERROR(
        TORCH_UCC_INIT,
        c10::str("UCC failed to create UCC context: ", ucc_status_string(st)));
    TORCH_CHECK(false, ucc_status_string(st));
  }
}

void CommUCC::progress() {
  TORCH_UCC_CHECK(
      ucc_context_progress(context), "failed to progress UCC collective");
}

void CommUCC::free_request(ucc_coll_req_h request) {
  TORCH_UCC_CHECK(
      ucc_collective_finalize(request), "failed to release UCC request");
}

CommUCC::~CommUCC() {
  if (context != nullptr) {
    TORCH_UCC_CHECK(
        ucc_context_destroy(context), "failed to destroy UCC context");
  }
  if (lib != nullptr) {
    TORCH_UCC_CHECK(ucc_finalize(lib), "failed to finalize UCC library");
  }
  context = nullptr;
  lib = nullptr;
}

std::string ProcessGroupUCCLogger::getLogPrefix(torch_ucc_phase_t phase) {
  // caller can override the phase stored locally
  torch_ucc_phase_t phase_ =
      (local_phase != phase && phase != TORCH_UCC_UNKNOWN) ? phase
                                                           : local_phase;
  return c10::str(log_prefix, "[", ucc_phase_map.at(phase_), "]");
}
void ProcessGroupUCCLogger::setLogPrefix(std::string log_prefix_) {
  log_prefix = log_prefix_;
}

ProcessGroupUCCLogger::ProcessGroupUCCLogger() {
  setLogPrefix("[ProcessGroupUCC]");
}
ProcessGroupUCCLogger::ProcessGroupUCCLogger(
    std::string log_prefix,
    torch_ucc_phase_t phase)
    : local_phase(phase) {
  setLogPrefix(log_prefix);
}

} // namespace c10d

#endif // USE_C10D_UCC

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `ucc_status_t`, `c10d`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/Exception.h`
- `torch/csrc/distributed/c10d/UCCTracing.hpp`
- `torch/csrc/distributed/c10d/UCCUtils.hpp`
- `cctype`
- `string`
- `unordered_map`
- `unordered_set`


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

Files in the same folder (`torch/csrc/distributed/c10d`):

- [`Utils.hpp_docs.md`](./Utils.hpp_docs.md)
- [`Ops.cpp_docs.md`](./Ops.cpp_docs.md)
- [`Store.hpp_docs.md`](./Store.hpp_docs.md)
- [`WinSockUtils.hpp_docs.md`](./WinSockUtils.hpp_docs.md)
- [`FakeProcessGroup.hpp_docs.md`](./FakeProcessGroup.hpp_docs.md)
- [`Work.cpp_docs.md`](./Work.cpp_docs.md)
- [`PrefixStore.hpp_docs.md`](./PrefixStore.hpp_docs.md)
- [`PyProcessGroup.hpp_docs.md`](./PyProcessGroup.hpp_docs.md)
- [`debug.h_docs.md`](./debug.h_docs.md)
- [`exception.h_docs.md`](./exception.h_docs.md)


## Cross-References

- **File Documentation**: `UCCUtils.cpp_docs.md`
- **Keyword Index**: `UCCUtils.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
