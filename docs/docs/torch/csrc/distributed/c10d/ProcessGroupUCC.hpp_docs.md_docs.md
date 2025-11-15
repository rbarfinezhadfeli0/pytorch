# Documentation: `docs/torch/csrc/distributed/c10d/ProcessGroupUCC.hpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/c10d/ProcessGroupUCC.hpp_docs.md`
- **Size**: 14,171 bytes (13.84 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/c10d/ProcessGroupUCC.hpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/ProcessGroupUCC.hpp`
- **Size**: 11,264 bytes (11.00 KB)
- **Type**: C++ Header File
- **Extension**: `.hpp`

## File Purpose

This is a c++ header file that is part of the PyTorch project.

## Original Source

```cpp
#pragma once

#ifdef USE_C10D_UCC

#include <torch/csrc/distributed/c10d/UCCUtils.hpp>

#include <exception>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace c10d {

#define TORCH_UCC_DEVICE_NOT_SET -2

#ifdef USE_CUDA
#define SAVE_TENSORS(_TENSORS, _DATA)                       \
  do {                                                      \
    if ((_TENSORS)[0].device().is_cuda()) {                 \
      for (const auto i : c10::irange((_TENSORS).size())) { \
        c10::cuda::CUDACachingAllocator::recordStream(      \
            (_TENSORS)[i].storage().data_ptr(), (*stream)); \
      }                                                     \
    } else {                                                \
      (_DATA) = (_TENSORS);                                 \
    }                                                       \
  } while (0)

#else
#define SAVE_TENSORS(_TENSORS, _DATA) (_DATA) = (_TENSORS);
#endif

constexpr const char* UCC_BACKEND_NAME = "ucc";

struct event_pool_t {
#ifdef USE_CUDA
  std::queue<std::unique_ptr<at::cuda::CUDAEvent>> event_pool;
#endif
  std::mutex event_pool_mutex;
};

class Comm;

// UCC does not support multiple CUDA devices per process.
class TORCH_API ProcessGroupUCC : public Backend {
 private:
  void set_timeout(ucc_coll_args_t& args);

 public:
  class WorkData {
   public:
    std::vector<at::Tensor> src;
    std::vector<at::Tensor> dst;
    std::vector<at::Tensor> flat;
    WorkData() {}
    virtual ~WorkData() = default;
  };
  class AlltoallWorkData : public WorkData {
   public:
    AlltoallWorkData(int size)
        : send_lengths(size),
          send_offsets(size),
          recv_lengths(size),
          recv_offsets(size) {}
    std::vector<uint64_t> send_lengths;
    std::vector<uint64_t> send_offsets;
    std::vector<uint64_t> recv_lengths;
    std::vector<uint64_t> recv_offsets;
  };

  class AllgathervWorkData : public WorkData {
   public:
    AllgathervWorkData(int size) : recv_lengths(size), recv_offsets(size) {}
    std::vector<uint64_t> recv_lengths;
    std::vector<uint64_t> recv_offsets;
  };

  class ScattervWorkData : public WorkData {
   public:
    ScattervWorkData(int size) : send_lengths(size), send_offsets(size) {}
    std::vector<uint64_t> send_lengths;
    std::vector<uint64_t> send_offsets;
  };

  class ProgressEntry {
    friend class ProcessGroupUCC;
    friend class Comm;

   public:
    ProgressEntry(CommBase* comm, ucc_coll_req_h request)
        : status_(UCC_INPROGRESS), comm_(comm), request_(request) {}
    // Finalizes UCC status or exception of collective request.
    void finalize(std::exception_ptr eptr = nullptr);
    ucc_status_t status_;
    CommBase* comm_;
    ucc_coll_req_h request_;
    std::unique_ptr<WorkData> data;
    c10::intrusive_ptr<c10::ivalue::Future> future_;
    std::exception_ptr eptr_;
  };

  class WorkUCC : public Work {
    friend class ProcessGroupUCC;
    friend class Comm;

   public:
    WorkUCC(
        OpType opType,
        uint64_t seq,
        const char* prof_title,
        const std::optional<std::vector<at::Tensor>>& inputs,
        const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger)
        : Work(-1, opType, prof_title, inputs), logger_(logger), seq_(seq) {}
    ~WorkUCC();
    void setException();
    void setAndThrowException();
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;
    std::vector<at::Tensor> result() override;
    int sourceRank() const override;
#ifdef USE_CUDA
    std::unique_ptr<at::cuda::CUDAEvent> fence = nullptr;
    event_pool_t* ep = nullptr;
#endif
    int sourceRank_;

   protected:
    std::shared_ptr<ProgressEntry> entry_;
    c10::intrusive_ptr<ProcessGroupUCCLogger> logger_;
    uint64_t seq_;

   private:
    // The future returned by getFuture.
    c10::intrusive_ptr<at::ivalue::Future> future_;
    // Store a reference to collective's outputs, used by result
    std::shared_ptr<std::vector<at::Tensor>> outputs_;
  };

  explicit ProcessGroupUCC(
      const c10::intrusive_ptr<Store>& store,
      int rank = -1,
      int size = -1,
      std::chrono::duration<float> timeout = kBackendDefaultTimeout);

  void initComm(c10::Device dev);

  ~ProcessGroupUCC() override;

  const std::string getBackendName() const override {
    return std::string(UCC_BACKEND_NAME);
  }

#ifdef USE_CUDA
  std::unique_ptr<at::cuda::CUDAEvent> getPooledEvent();
#endif

  // Performs a health check by initializing dummy UCC & UCX communicators and
  // then destroying them. This will help indicate and signal any
  // UCC/UCX-related issues prior to the first collective. The actual
  // initialization and subsequent destruction is ran on a separate thread and
  // the main thread is signalled about timeouts/errors to report to the
  // application.
  void runHealthCheck();

  template <typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> collective_post(
      OpType opType,
      PreProcess preproc,
      PostProcess postproc,
      ucc_coll_args_t& coll,
      std::unique_ptr<ProcessGroupUCC::WorkData> data,
      c10::Device dev,
      std::vector<at::Tensor>& inputTensors,
      std::vector<at::Tensor>& outputTensors,
      const char* prof_title);

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  // Counting for the sequential number of UCC collective_post call.
  uint64_t seq_{0};

  // Agrees on an initial sequence number for the whole group by having rank 0
  // create it and broadcast it to other ranks using the store.
  void setSequenceNumberForGroup() override;

  // Retrieves the current sequence number for the whole group, which should be
  // in sync. If the returned number is not consistent across the group, it
  // may indicate that there is some sort of collective desynchronization.
  uint64_t getSequenceNumberForGroup() override;

  static c10::intrusive_ptr<Backend> createProcessGroupUCC(
      const c10::intrusive_ptr<::c10d::Store>& store,
      int rank,
      int size,
      const std::chrono::duration<float>& timeout);

 protected:
  const std::chrono::duration<float> timeout_;
  std::shared_ptr<torch_ucc_oob_coll_info_t> oob;
  std::shared_ptr<Comm> comm = {nullptr};
  uint32_t comm_id;
  ucc_team_h team{nullptr};
  ucc_ee_h cuda_ee{nullptr};
  ucc_ee_h cuda_ee_p2p[2]{nullptr, nullptr};

#ifdef USE_CUDA
  std::unique_ptr<at::cuda::CUDAStream> stream = nullptr;
  std::unique_ptr<at::cuda::CUDAStream> stream_p2p[2] = {nullptr, nullptr};
  event_pool_t ep;
#endif
  c10::intrusive_ptr<ProcessGroupUCCLogger> logger;
};

class Comm {
  c10::intrusive_ptr<ProcessGroupUCCLogger> logger;
  std::shared_ptr<torch_ucc_oob_coll_info_t> oob;
  CommUCC ucc_comm;
  std::mutex mutex;
  std::thread progress_thread;
  std::condition_variable queue_produce_cv;
  std::condition_variable queue_consume_cv;
  std::deque<std::shared_ptr<ProcessGroupUCC::ProgressEntry>> progress_queue;
  bool stop_progress_loop;
  bool collective_inprogress;
  torch_ucc_phase_t finalize_phase;

 public:
  c10::DeviceIndex cuda_device_index;
  Comm(
      const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger,
      std::shared_ptr<torch_ucc_oob_coll_info_t> oob,
      c10::Device dev,
      bool is_health_check);

  ~Comm();

  void ucc_create_team(
      ucc_team_h& team,
      std::shared_ptr<torch_ucc_oob_coll_info_t> oob);

  void ucc_destroy_team(ucc_team_h& team);

  c10::intrusive_ptr<Work> enqueue_p2p(
      OpType opType,
      ucc_coll_req_h request,
      const char* prof_title);

#ifdef USE_CUDA
  void enqueue_cuda_collective(
      std::unique_ptr<ProcessGroupUCC::WorkData> data,
      c10::intrusive_ptr<ProcessGroupUCC::WorkUCC> work,
      ucc_coll_args_t& coll,
      ucc_team_h team,
      ucc_ee_h ee);
#endif

  void enqueue_collective(
      std::unique_ptr<ProcessGroupUCC::WorkData> data,
      c10::intrusive_ptr<ProcessGroupUCC::WorkUCC> work,
      ucc_coll_args_t& coll,
      ucc_team_h team);

  static std::shared_ptr<Comm> get_comm(
      uint32_t& id,
      c10::Device dev,
      std::shared_ptr<torch_ucc_oob_coll_info_t> oob,
      const c10::intrusive_ptr<ProcessGroupUCCLogger>& logger,
      bool is_health_check = false);

  void progress_loop();
};

} // namespace c10d

#endif // USE_C10D_UCC

```



## High-Level Overview


This C++ file contains approximately 13 class(es)/struct(s) and 21 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10d`

**Classes/Structs**: `event_pool_t`, `Comm`, `TORCH_API`, `WorkData`, `AlltoallWorkData`, `AllgathervWorkData`, `ScattervWorkData`, `ProgressEntry`, `ProcessGroupUCC`, `Comm`, `WorkUCC`, `ProcessGroupUCC`, `Comm`, `Comm`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/c10d/UCCUtils.hpp`
- `exception`
- `memory`
- `mutex`
- `queue`
- `thread`
- `vector`
- `torch/csrc/distributed/c10d/Backend.hpp`
- `torch/csrc/distributed/c10d/Store.hpp`
- `torch/csrc/distributed/c10d/Types.hpp`
- `torch/csrc/distributed/c10d/Utils.hpp`
- `ATen/cuda/CUDAEvent.h`
- `c10/cuda/CUDAStream.h`


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

- **File Documentation**: `ProcessGroupUCC.hpp_docs.md`
- **Keyword Index**: `ProcessGroupUCC.hpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/distributed/c10d`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/distributed/c10d`):

- [`ProcessGroupWrapper.cpp_docs.md_docs.md`](./ProcessGroupWrapper.cpp_docs.md_docs.md)
- [`c10d.h_kw.md_docs.md`](./c10d.h_kw.md_docs.md)
- [`TCPStoreLibUvBackend.cpp_kw.md_docs.md`](./TCPStoreLibUvBackend.cpp_kw.md_docs.md)
- [`ProcessGroupGlooCuda.cpp_docs.md_docs.md`](./ProcessGroupGlooCuda.cpp_docs.md_docs.md)
- [`NanCheck.cu_docs.md_docs.md`](./NanCheck.cu_docs.md_docs.md)
- [`python_callback_work.hpp_kw.md_docs.md`](./python_callback_work.hpp_kw.md_docs.md)
- [`sequence_num.hpp_kw.md_docs.md`](./sequence_num.hpp_kw.md_docs.md)
- [`Functional.hpp_kw.md_docs.md`](./Functional.hpp_kw.md_docs.md)
- [`TCPStoreBackend.cpp_kw.md_docs.md`](./TCPStoreBackend.cpp_kw.md_docs.md)
- [`ProcessGroupUCC.cpp_kw.md_docs.md`](./ProcessGroupUCC.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `ProcessGroupUCC.hpp_docs.md_docs.md`
- **Keyword Index**: `ProcessGroupUCC.hpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
