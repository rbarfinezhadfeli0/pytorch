# Documentation: `docs/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp_docs.md`
- **Size**: 53,418 bytes (52.17 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp`
- **Size**: 223,462 bytes (218.22 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#ifdef USE_C10D_NCCL

#include <nlohmann/json.hpp>
#include <exception>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <c10/util/WaitCounter.h>
#include <c10/util/hash.h>
#include <c10/util/irange.h>
#include <c10/util/thread_name.h>
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>
#include <torch/csrc/cuda/nccl.h>
#include <torch/csrc/distributed/c10d/FlightRecorder.hpp>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/NanCheck.hpp>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/cuda/utils.hpp>
#include <torch/torch.h>
#include <optional>

namespace c10d {

constexpr const char* const kNCCLAbortedCommStoreKey = "NCCLABORTEDCOMM";
using FlightRecorderCUDA = FlightRecorder<at::cuda::CUDAEvent>;

namespace {

#if defined(NCCL_MAJOR) && \
    ((NCCL_MAJOR > 2) || (NCCL_MAJOR == 2) && (NCCL_MINOR >= 10))
#define NCCL_HAS_AVG 1
#endif // NCCL version >= 2.10

// NCCL op mapping
const std::map<ReduceOp::RedOpType, ncclRedOp_t> ncclOp = {
    {ReduceOp::MIN, ncclMin},
    {ReduceOp::MAX, ncclMax},
    {ReduceOp::SUM, ncclSum},
    {ReduceOp::PRODUCT, ncclProd},
#ifdef NCCL_HAS_AVG
    {ReduceOp::AVG, ncclAvg},
#endif // NCCL_HAS_AVG
};

inline bool isUnsupportedFloat8(at::ScalarType t) {
  return (
      t == at::ScalarType::Float8_e5m2fnuz ||
      t == at::ScalarType::Float8_e4m3fnuz ||
      t == at::ScalarType::Float8_e8m0fnu
#ifndef NCCL_SUPPORTS_FP8
      || t == at::ScalarType::Float8_e5m2 || t == at::ScalarType::Float8_e4m3fn
#endif
  );
}

#ifdef ENABLE_NCCL_PREMUL_SUM_SUPPORT
template <typename T, ncclDataType_t dataType>
ncclRedOpRAII unpackPreMulSum(
    const ReduceOp& reduceOp,
    const ncclComm_t& comm) {
  const auto* preMulSupplement =
      reinterpret_cast<NCCLPreMulSumSupplement*>(reduceOp.supplement_.get());
  ncclRedOp_t preMulSum{};
  bool has_tensor = preMulSupplement->tensor_factor.defined();
  auto residence = has_tensor ? ncclScalarDevice : ncclScalarHostImmediate;
  const T* ptr_factor = has_tensor
      ? preMulSupplement->tensor_factor.const_data_ptr<T>()
      : nullptr;
  T scalar_factor = T(preMulSupplement->double_factor);
  ncclRedOpCreatePreMulSum(
      &preMulSum,
      // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/ops.html#ncclredopcreatepremulsum
      // tells us that the scalar input is strictly a multiplier.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
      /*scalar=*/has_tensor ? const_cast<T*>(ptr_factor) : &scalar_factor,
      dataType,
      residence,
      comm);
  return ncclRedOpRAII(preMulSum, comm);
}
#endif // ENABLE_NCCL_PREMUL_SUM_SUPPORT

ncclRedOpRAII getNcclReduceOp(
    const ReduceOp& reduceOp,
    at::Tensor& input,
    const ncclDataType_t& dataType,
    const ncclComm_t& comm) {
  try {
    if (input.scalar_type() == at::kBool) {
      if (reduceOp == ReduceOp::SUM) {
        // For bool tensors, map sum to max, which both represent a bitwise or.
        // This is to prevent overflow issues with sum, since we use uint8 to
        // represent a bool (see ncclDataType mapping).
        return ncclMax;
      }
#ifdef NCCL_HAS_AVG
      if (reduceOp == ReduceOp::AVG) {
        C10_THROW_ERROR(
            TypeError, "Cannot use ReduceOp.AVG with boolean inputs");
      }
#endif // NCCL_HAS_AVG
    }
    if (reduceOp == ReduceOp::PREMUL_SUM) {
#ifdef ENABLE_NCCL_PREMUL_SUM_SUPPORT
      switch (dataType) {
        case ncclHalf:
          return unpackPreMulSum<at::Half, ncclHalf>(reduceOp, comm);
        case ncclFloat:
          return unpackPreMulSum<float, ncclFloat>(reduceOp, comm);
        case ncclBfloat16:
          return unpackPreMulSum<float, ncclBfloat16>(reduceOp, comm);
        case ncclDouble:
          return unpackPreMulSum<double, ncclDouble>(reduceOp, comm);
        default:
          C10_THROW_ERROR(
              TypeError,
              "PreMulSum Data type must be half, float, bfloat16 or double");
          return ncclRedOp_t{};
      }
#else
      C10_THROW_ERROR(ValueError, "PreMulSum requires NCCL>=2.11.1");
#endif // ENABLE_NCCL_PREMUL_SUM_SUPPORT
    }
    return ncclOp.at(reduceOp);
  } catch (const std::out_of_range&) {
    switch (reduceOp) {
      case ReduceOp::AVG:
        C10_THROW_ERROR(
            ValueError,
            c10::str(
                "AVG requires NCCL 2.10+. The current version is ",
                NCCL_MAJOR,
                ".",
                NCCL_MINOR));
        break;
      case ReduceOp::BAND:
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BAND with NCCL");
        break;
      case ReduceOp::BOR:
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BOR with NCCL");
        break;
      case ReduceOp::BXOR:
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BXOR with NCCL");
        break;
      default:
        C10_THROW_ERROR(ValueError, "Unhandled ReduceOp");
        break;
    }
  }
}

// Get a key string from device
inline std::string getKeyFromDevice(const at::Device& device) {
  return std::to_string(device.index());
}

std::string getKeySendRecv(int myRank, int peer) {
  int lowRank = myRank < peer ? myRank : peer;
  int highRank = myRank < peer ? peer : myRank;
  std::string sendRecvPair =
      std::to_string(lowRank) + ":" + std::to_string(highRank);
  return sendRecvPair;
}

// Get device from tensor
inline at::Device getDevice(at::Tensor& tensor) {
  return tensor.device();
}

// [Sync Streams] Helper that lets the input ncclStreams to wait for the current
// stream. NCCL communications run on ncclStreams, but input tensors are
// allocated on different streams (i.e., current streams). Communications on
// ncclStreams cannot start before pending input tensor ops on current streams
// finish. Otherwise, ops on two streams might read/write same tensors
// concurrently.
//
// The synchronization above alone is not enough. We also need to make sure
// input tensors are not freed before their usages on ncclStreams finish. This
// can be achieved by calling c10::cuda::CUDACachingAllocator::recordStream,
// which remembers the usage stream (ncclStream), creates an event on the usage
// stream when GC attempts to free the input tensor, and delays GC until that
// event is done.
void syncStream(
    at::Device& device,
    at::cuda::CUDAEvent& ncclEvent,
    at::cuda::CUDAStream& ncclStream) {
  ncclEvent.record(at::cuda::getCurrentCUDAStream(device.index()));
  ncclEvent.block(ncclStream);
}

std::string getNcclAbortedCommStoreKey(const std::string& ncclIdStr) {
  return std::string(kNCCLAbortedCommStoreKey) + ":" + ncclIdStr;
}

// Returns exception's what() given an exception_ptr instance.
std::string getExceptionMsgFromExceptionPtr(
    const std::exception_ptr& exceptionPtr) {
  TORCH_CHECK(exceptionPtr != nullptr);
  try {
    std::rethrow_exception(exceptionPtr);
  } catch (const std::exception& e) {
    return e.what();
  } catch (...) {
    return "Unknown exception type";
  }
}

inline void errorIfCapturingNonCapturableNCCL(c10::cuda::CaptureStatus status) {
  // parentheses avoid some compiler warnings
  static const uint64_t min_version =
      (((uint64_t)2) << 32) + (((uint64_t)9) << 16) + ((uint64_t)6);
  static const uint64_t cur_version = torch::cuda::nccl::version();
  if (cur_version < min_version) {
    TORCH_CHECK_WITH(
        NotImplementedError,
        status == c10::cuda::CaptureStatus::None,
        "Capturing NCCL collectives is only allowed with NCCL >= 2.9.6");
  }
}

// When TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK is set, all tensors (no
// matter how they have been allocated) are registered with all NCCL comms.
bool shouldAllCommunicatorsRegisterAllTensors() {
#ifdef NCCL_HAS_COMM_REGISTER
  static const bool flag = [] {
    const bool flag =
        getCvarBool(TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK, false);
    if (flag &&
        c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::
            expandable_segments()) {
      LOG(INFO)
          << "disables TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK because it is not compatible with CUDA allocator expandable segments mode.";
      return false;
    }
    return flag;
  }();
  return flag;
#else
  return false;
#endif // NCCL_HAS_COMM_REGISTER
}

} // namespace

// Map each communicator to the memory pools registered with it.
// This map is used when the caching allocator allocates or frees segments, in
// order to register or deregister them with the relevant NCCL communicators.
// There are two modes to do so:
// - If TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=1 then *ALL* segments
//   will be registered with *ALL* NCCL communicators (for the same device),
//   even if they were allocated with cudaMalloc (which NCCL doesn't like).
// - If a MemPool is explicitly registered with a ProcessGroup, then all its
//   segments (current and future) will be registered with the NCCL communicator
//   corresponding to the pool's device. This works best if the MemPool is set
//   up to use ncclMemAlloc (which is exposed by the ProcessGroup).
// Implementation notes:
// - We cannot reuse devNCCLCommMap_ in each ProcessGroup because the key may be
//   ranks rather than device in point-to-point case.
// - This map has also to be maintained as global variable since the register
//   hooks are called outside the scope of any PG, thus we need traverse
//   communicators in all PGs.

// MemPoolSet has ids of mempools used with this communicator, and whether they
// were registered with window APIs or not
using MemPoolSet = std::unordered_set<
    std::tuple<c10::cuda::MempoolId_t, bool>,
    c10::hash<std::tuple<c10::cuda::MempoolId_t, bool>>>;
static std::unordered_map<std::shared_ptr<NCCLComm>, MemPoolSet>
    ncclCommMemPoolMap;
static std::mutex ncclCommMemPoolMapMutex;

std::atomic<bool> ProcessGroupNCCL::shouldDump_(false);

static void cacheAllocatorRegisterHook(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  // Register after SEGMENT_ALLOC
  if (te.action_ !=
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_ALLOC) {
    return;
  }

  std::lock_guard<std::mutex> lock(ncclCommMemPoolMapMutex);
  for (auto& [ncclComm, memPools] : ncclCommMemPoolMap) {
    if (te.device_ == ncclComm->getDeviceIndex()) {
      bool symm = false;
      bool should_register = shouldAllCommunicatorsRegisterAllTensors();
      auto it =
          std::find_if(memPools.begin(), memPools.end(), [&](const auto& tup) {
            return std::get<0>(tup) == te.mempool_;
          });
      if (it != memPools.end()) {
        should_register = true;
        symm = std::get<1>(*it);
      }
      if (should_register) {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        ncclComm->registerSegment(
            reinterpret_cast<void*>(te.addr_),
            te.size_,
            /*errorOnRereg*/ false,
            /*window*/ symm);
      }
    }
  }
}

static void cacheAllocatorDeregisterHook(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  // deregister before SEGMENT_FREE
  if (te.action_ !=
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_FREE) {
    return;
  }

  std::lock_guard<std::mutex> lock(ncclCommMemPoolMapMutex);
  for (auto& [ncclComm, memPools] : ncclCommMemPoolMap) {
    if (te.device_ == ncclComm->getDeviceIndex()) {
      bool symm = false;
      bool should_register = shouldAllCommunicatorsRegisterAllTensors();
      auto it =
          std::find_if(memPools.begin(), memPools.end(), [&](const auto& tup) {
            return std::get<0>(tup) == te.mempool_;
          });
      if (it != memPools.end()) {
        should_register = true;
        symm = std::get<1>(*it);
      }
      if (should_register) {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        ncclComm->deregisterSegment(reinterpret_cast<void*>(te.addr_), symm);
      }
    }
  }
}

static void attachAllocatorHooks() {
  static auto flag [[maybe_unused]] = [] {
    // Attaching hooks fails if CUDACachingAllocator is not initialized, so
    // Init for CUDA is called (and is a no-op if CUDA is already
    // initialized).
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(
        &cacheAllocatorRegisterHook);
    c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(
        &cacheAllocatorDeregisterHook);
    return true;
  }();
}

static std::
    unordered_map<std::string, std::unordered_map<std::string, std::string>>
    getNCCLCommDumpMap() {
#if (defined(IS_NCCLX) || defined(USE_ROCM)) && defined(NCCL_COMM_DUMP)
  std::unordered_map<
      std::string /* ncclUniqueID */,
      std::unordered_map<std::string, std::string> /* dump from this comm */>
      ncclDumpMap;
  // dump_nccl_trace is only called from the default PG (local_id_=0), but we
  // want to dump from all comms so we need to iterate over ncclCommMemPoolMap,
  // which is static
  std::vector<std::shared_ptr<NCCLComm>> allNCCLComms;
  // within the critical section, we don't want to dump while holding the lock
  // as dump might hang
  {
    std::lock_guard<std::mutex> lock(ncclCommMemPoolMapMutex);
    for (auto& [ncclComm, _] : ncclCommMemPoolMap) {
      allNCCLComms.push_back(ncclComm);
    }
  }
  for (auto& ncclComm : allNCCLComms) {
    ncclDumpMap[ncclComm->getUniqueHash()] = ncclComm->ncclCommDump();
  }
  return ncclDumpMap;
#else
  return std::unordered_map<
      std::string,
      std::unordered_map<std::string, std::string>>();
#endif // (defined(IS_NCCLX) || defined(USE_ROCM)) && defined(NCCL_COMM_DUMP)
}

void reset_nccl_trace() {
  FlightRecorderCUDA::get()->reset_all();
}

std::string dump_nccl_trace(
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive) {
  auto ncclDumpMap = getNCCLCommDumpMap();
#if defined(USE_ROCM) && defined(NCCL_COMM_DUMP)
  for (const auto& [ncclUniqueIDStr, dump] : ncclDumpMap) {
    printNcclCommProxyTrace("Received dump signal " + ncclUniqueIDStr, dump);
  }
#endif // defined(USE_ROCM) && defined(NCCL_COMM_DUMP)
  return FlightRecorderCUDA::get()->dump(
      ncclDumpMap, includeCollectives, includeStackTraces, onlyActive);
}

std::string dump_nccl_trace_json(bool includeCollectives, bool onlyActive) {
  auto ncclDumpMap = getNCCLCommDumpMap();
  return FlightRecorderCUDA::get()->dump_json(
      ncclDumpMap, includeCollectives, onlyActive);
}

std::optional<std::function<void(std::function<void(const std::string&)>)>>&
get_cpp_trace_dumper() {
  static std::optional<
      std::function<void(std::function<void(const std::string&)>)>>
      dumper(std::nullopt);
  return dumper;
}

gil_checker_t& get_gil_checker() {
  static gil_checker_t gil_checker = nullptr;
  return gil_checker;
}

static std::future<bool> launchAsyncGilCheck() {
  std::promise<bool> resultPromise;
  std::future<bool> resultFuture = resultPromise.get_future();
  TORCH_CHECK(get_gil_checker(), "Can't check GIL with null GIL checker");
  std::thread workerThread([promise = std::move(resultPromise)]() mutable {
    c10::setThreadName("pt_nccl_gil_chk");

    try {
      auto& gil_checker = get_gil_checker();
      promise.set_value((*gil_checker)());
    } catch (...) {
      promise.set_exception(std::current_exception());
    }
  });

  // Detach the thread to allow it to run independently
  workerThread.detach();

  return resultFuture;
}

const int64_t ProcessGroupNCCL::kWatchdogThreadSleepMillis = 100;
constexpr int64_t kSynchronizeBusyWaitMillis = 1;
thread_local uint64_t ProcessGroupNCCL::ncclActiveGroupCounter_ = 0;

std::ostream& operator<<(
    std::ostream& output,
    const ProcessGroupNCCL::WorkNCCL& workNCCL) {
  std::string workInfo;
  workInfo = c10::str(
      "WorkNCCL(",
      "SeqNum=",
      workNCCL.seq_,
      ", OpType=",
      opTypeToString(workNCCL.opType_),
      ", NumelIn=",
      workNCCL.numelIn_,
      ", NumelOut=",
      workNCCL.numelOut_,
      ", Timeout(ms)=",
      workNCCL.opTimeout_.count(),
      ")");
  return output << workInfo;
}

/* Implementation of TensorShelf class */

void TensorShelf::stash(std::vector<at::Tensor>& tensors) {
  std::lock_guard<std::mutex> lock(mutex_);
  tVector_.insert(tVector_.end(), tensors.begin(), tensors.end());
}

void TensorShelf::stash(TensorShelf& other) {
  std::vector<at::Tensor>& otherVec = other.get();
  this->stash(otherVec);
}

void TensorShelf::unstash() {
  this->clear();
}

bool TensorShelf::empty() {
  std::lock_guard<std::mutex> lock(mutex_);
  return tVector_.empty();
}

void TensorShelf::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  tVector_.clear();
}

std::vector<at::Tensor>& TensorShelf::get() {
  return tVector_;
}

ProcessGroupNCCL::WorkNCCL::WorkNCCL(
    std::string pgUID,
    std::string pgDesc,
    at::Device& device,
    int rank,
    OpType opType,
    uint64_t seq,
    bool isP2P,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputs,
    bool enableTiming,
    bool cudaEventCacheEnabled,
    DebugLevel distDebugLevel)
    : Work(rank, opType, profilingTitle, inputs),
      pgUID_(std::move(pgUID)),
      pgDesc_(std::move(pgDesc)),
      device_(device),
      workStartTime_(std::chrono::steady_clock::now()),
      seq_(seq),
      isP2P_(isP2P),
      timingEnabled_(enableTiming),
      distDebugLevel_(distDebugLevel) {
  // Creates the CUDA event wrappers
  // Note: The actual events are lazily created when first recorded to with
  // DEFAULT_FLAGS = cudaEventDisableTiming.
  if (cudaEventCacheEnabled) {
    ncclStartEvent_ = enableTiming
        ? CUDAEventCache::get(device.index())->create(enableTiming)
        : nullptr;
    ncclEndEvent_ = CUDAEventCache::get(device.index())->create(enableTiming);
  } else {
    ncclStartEvent_ = enableTiming
        ? std::make_shared<at::cuda::CUDAEvent>(cudaEventDefault)
        : nullptr;
    ncclEndEvent_ = std::make_shared<at::cuda::CUDAEvent>(
        enableTiming ? cudaEventDefault : cudaEventDisableTiming);
  }
  futureWorkResult_ =
      c10::make_intrusive<at::ivalue::Future>(c10::AnyEnumType::get());
  // other functions expect an initialized ptr
  stashed_for_allocator_safety_ = std::make_shared<TensorShelf>();
}

ProcessGroupNCCL::WorkNCCL::WorkNCCL(const WorkNCCL& w)
    : Work(w.rank_, w.opType_),
      std::enable_shared_from_this<WorkNCCL>(w),
      pgUID_(w.pgUID_),
      pgDesc_(w.pgDesc_),
      device_(w.device_),
      ncclStartEvent_(w.ncclStartEvent_),
      ncclEndEvent_(w.ncclEndEvent_),
      ncclComm_(w.ncclComm_),
      blockingWait_(w.blockingWait_),
      opTimeout_(w.opTimeout_),
      ownedEphermeralTimeout_(w.ownedEphermeralTimeout_),
      workStartTime_(w.workStartTime_),
      seq_(w.seq_),
      isP2P_(w.isP2P_),
      startTraceUpdated_(w.startTraceUpdated_),
      numelIn_(w.numelIn_),
      numelOut_(w.numelOut_),
      store_(w.store_),
      // Note: the `work` returned to user and the `work` enqueued to watchdog
      // share the pointer to the tensor stash.  At least one of them should
      // clean the tensor stash, the earlier the better, i.e. user calling
      // `work.wait` than watchdog detecting work completion.
      stashed_for_allocator_safety_(w.stashed_for_allocator_safety_),
      futureWorkResult_(w.futureWorkResult_),
      timingEnabled_(w.timingEnabled_),
      trace_id_(w.trace_id_),
      trace_reset_epoch_(w.trace_reset_epoch_),
      distDebugLevel_(w.distDebugLevel_) {
  exception_ = w.exception_;
}

bool ProcessGroupNCCL::WorkNCCL::isCompleted() {
  if (!ncclComm_->isAborted()) {
    checkAndSetException();
  }
  return exception() || finishedGPUExecutionInternal();
}

bool ProcessGroupNCCL::WorkNCCL::isStarted() {
  if (!ncclComm_->isAborted()) {
    checkAndSetException();
  }
  return exception() || startedGPUExecutionInternal();
}

bool ProcessGroupNCCL::WorkNCCL::isSuccess() const {
  C10_THROW_ERROR(NotImplementedError, "WorkNCCL::isSuccess() is deprecated");
}

void ProcessGroupNCCL::WorkNCCL::checkAndSetException() {
  if (exception()) {
    // We already have an exception.
    return;
  }

  auto exception_ptr = checkForNCCLErrors();
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = exception_ptr;
  if (exception_) {
    LOG(ERROR) << logPrefix() << "Collective " << *this
               << " raised the following async exception: "
               << getExceptionMsgFromExceptionPtr(exception_);

    // Mark future result as ERROR
    if (futureWorkResult_ && !futureWorkResult_->completed()) {
      futureWorkResult_->markCompleted(
          at::IValue(static_cast<uint8_t>(WorkResult::COMM_ERROR)));
    }
  }
}

const std::string& ProcessGroupNCCL::WorkNCCL::logPrefix() const {
  static std::string prefix = c10::str("[Rank ", rank_, "] ");
  return prefix;
}

void ProcessGroupNCCL::WorkNCCL::setException(
    std::exception_ptr exception_ptr) {
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = std::move(exception_ptr);
}

// Helper that checks if the NCCL kernels are completed on the GPUs
bool ProcessGroupNCCL::WorkNCCL::finishedGPUExecution() {
  checkAndSetException();
  return finishedGPUExecutionInternal();
}

bool ProcessGroupNCCL::WorkNCCL::startedGPUExecutionInternal() const {
  // if timing is disabled we won't have allocated start events
  if (!timingEnabled_) {
    return false;
  }
  // Checking the work's corresponding CUDA event's status
  if (!ncclStartEvent_->query()) {
    return false;
  }
  return true;
}

bool ProcessGroupNCCL::WorkNCCL::finishedGPUExecutionInternal() const {
  // Checking the work's corresponding CUDA event's status
  // It calls `cudaEventQuery` eventually. Although this seems to be a
  // non-blocking call, but we did notice hangs in the past. It can
  // hang if another thread is holding the CUDA global context lock. For
  // example, when doing a `cudaDeviceSynchronize` or even
  // `cudaStreamSynchronize`.
  if (!ncclEndEvent_->query()) {
    return false;
  }
  return true;
}

bool ProcessGroupNCCL::WorkNCCL::checkTimeout(
    std::optional<std::chrono::milliseconds> timeout) {
  STATIC_SCOPED_WAIT_COUNTER(
      pytorch.wait_counter.ProcessGroupNCCL__checkTimeout);
  auto currentTimepoint = std::chrono::steady_clock::now();
  auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      currentTimepoint - workStartTime_);
  auto workTimeout = timeout ? *timeout : opTimeout_;

  if (timeElapsed < workTimeout) {
    return false;
  }

  // Timed out

  std::string exceptionMsg = c10::str(
      logPrefix(),
      "Watchdog caught collective operation timeout: ",
      *this,
      " ran for ",
      timeElapsed.count(),
      " milliseconds before timing out.");

  LOG(ERROR) << exceptionMsg;

  std::exception_ptr exception_ptr =
      std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exceptionMsg));
  if (!exception()) {
    // if there is already an error, we don't override it
    setException(exception_ptr);
  }

  // Mark future result as TIMEOUT
  if (futureWorkResult_ && !futureWorkResult_->completed()) {
    futureWorkResult_->markCompleted(
        at::IValue(static_cast<uint8_t>(WorkResult::TIMEOUT)));
  }
  return true;
}

// Print the traceback of the collective at call time
std::string ProcessGroupNCCL::WorkNCCL::getTraceback() const {
  // First step we get the corresponding record entry from FR, based on work's
  // trace_id_ and trace_reset_epoch_
  std::optional<FlightRecorderCUDA::Entry> entry =
      FlightRecorderCUDA::get()->getEntry(trace_id_, trace_reset_epoch_);
  if (entry.has_value()) {
    auto entryVal = entry.value();
    // Get stack trace from FR entry, in string format
    // Note: `getTraceback` call below invokes `torch::symbolize`, which may
    // need to acquire the GIL. In order for watchdog to be block-free, we make
    // the call with std::async.
    auto future = std::async(
        std::launch::async, [&entryVal]() { return entryVal.getTraceback(); });
    // Wait for the future to complete or timeout
    auto status = future.wait_for(std::chrono::seconds(8));
    if (status == std::future_status::ready) {
      return future.get();
    }
  }
  return "";
}

// Print the traceback of the collective at call time
void ProcessGroupNCCL::WorkNCCL::printTraceback() const {
  std::string tracebackStr = getTraceback();
  if (!tracebackStr.empty()) {
    LOG(ERROR) << "Stack trace of the failed collective: \n" << tracebackStr;
  } // else, symbolizer probably timed out, we skip logging the stack trace.
  else {
    LOG(ERROR)
        << "Stack trace of the failed collective not found, "
        << "potentially because FlightRecorder is disabled. "
        << "You can enable it by setting TORCH_NCCL_TRACE_BUFFER_SIZE to a non-zero value.";
  }
}

void ProcessGroupNCCL::WorkNCCL::handleException(
    ErrorHandlingMode errorHandling) {
  if (exception_) {
    auto exceptionMsg = c10::str(
        "Some NCCL operations have failed or timed out. Due to the ",
        "asynchronous nature of CUDA kernels, subsequent GPU operations ",
        "might run on corrupted/incomplete data.");
    LOG(ERROR) << logPrefix() << exceptionMsg;
    C10_LOG_API_USAGE_ONCE("ProcessGroupNCCL.WorkNCCL.handleException");

    auto logger = c10d::C10dLogger::getLogger();
    if (logger) {
      ::c10d::C10dLoggingData data;
      data.strings["work_nccl_exception"] =
          getExceptionMsgFromExceptionPtr(exception_);
      logger->log(data);
    }

    if (SHOULD_TEAR_DOWN(errorHandling)) {
      auto tearDownMsg = c10::str(
          "To avoid data inconsistency, we are taking the entire process down.");
      LOG(ERROR) << logPrefix() << tearDownMsg;
      std::rethrow_exception(exception_);
    }
  }
}

void ProcessGroupNCCL::WorkNCCL::synchronize() {
  synchronizeStream();
  if (c10d::allow_inflight_collective_as_graph_input()) {
    c10d::unregister_work(
        c10::intrusive_ptr<
            ProcessGroupNCCL::WorkNCCL>::unsafe_reclaim_from_nonowning(this));
  }
}

void ProcessGroupNCCL::WorkNCCL::synchronizeStream() {
  auto currentStream = at::cuda::getCurrentCUDAStream(device_.index());
  // Block the current stream on the NCCL stream
  ncclEndEvent_->block(currentStream);
  // Unstage the stashed tensors so that CachingAllocator can recycle them
  // THIS MUST HAPPEN AFTER THE BLOCKING CALL ABOVE
  stashed_for_allocator_safety_->unstash();
}

// Same as calling synchronize() when blockingWait_ is false
bool ProcessGroupNCCL::WorkNCCL::wait(std::chrono::milliseconds timeout) {
  RECORD_PARAM_COMMS(
      std::make_tuple(static_cast<int64_t>(this->seq_), this->isP2P_), // seq
      std::make_tuple(pgUID_, pgDesc_), // PG name tuple
      rank_, // rank
      "wait", // collective name
      0, // inNelems
      0, // outNelems
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      -1,
      -1,
      static_cast<int>(1)); // number of device?

  // synchronize() will block the current stream on the NCCL stream
  synchronize();

  // In case of blockingWait or a timeout value is specified by the user, we
  // block the CPU thread until the work is completed or timed out.
  if (blockingWait_ || timeout != kNoTimeout) {
    while (!isCompleted()) {
      bool timedOut = checkTimeout(
          timeout == kNoTimeout ? std::nullopt : std::make_optional(timeout));
      // Explicitly abort ncclComms here before throwing this timed out
      // exception to users.
      // If throwing timed out excepiton without aborting nccl communicators
      // here, it was observed that CUDA GPU will have 100% utilization and
      // can not run new events successfully.
      if (timedOut) {
        std::string exceptionMsg = c10::str(
            logPrefix(), "Work ", (*this), " timed out in blocking wait.");
        LOG(ERROR) << exceptionMsg;
        break;
      }
      // Yield
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }
  } else if (isBarrierOp_ && !isCompleted()) {
    // For barrier wait when timeout is unspecified, we block the CPU thread on
    // current stream. This is to minimize the CPU barrier wait time in healthy
    // path
    auto currentStream = at::cuda::getCurrentCUDAStream(device_.index());
    // CUDAStream wrapper will correctly use a DeviceGuard here
    currentStream.synchronize();
  }

  // If exception is detected, throw it from the main CPU thread
  if (exception()) {
    // Abort NCCL communicators
    abort();
    // Throw exception (from main thread here)
    handleException(TearDown);
  }

  // TODO(kwen2501): this should be moved to c10d tests, to qualify a NCCL
  // upgrade. Once a NCCL version is qualified, this code should not be needed
  // at runtime.
#ifdef PGNCCL_ENABLE_HASH
  if (enableCollectiveHashDebug_.load()) {
    auto numel = getTensorsNumel(*outputs_);
    auto hashValue = hashTensors(*outputs_);
    PRINT_COLLECTIVE_HASH_SIGNATURE(
        "output", opTypeToString(opType_), numel, hashValue);
  }
#endif // PGNCCL_ENABLE_HASH
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroupNCCL::WorkNCCL::abort() {
  // dump before aborting for rcclexp
#if defined(USE_ROCM) && defined(NCCL_COMM_DUMP)
  auto dumpMap = ncclComm_->ncclCommDump();
  printNcclCommProxyTrace("WorkNCCL::abort", dumpMap);
#endif // USE_ROCM && NCCL_COMM_DUMP

  // Abort all communicators of this work
  ncclComm_->abort();

  {
    std::lock_guard<std::mutex> lock(ncclCommMemPoolMapMutex);
    ncclCommMemPoolMap.erase(ncclComm_);
  }
}

static std::atomic<size_t> process_group_id = 0;

constexpr const char* MULTI_DEVICE_ERROR_MSG =
    "Expecting one tensor only but got multiple. You are probably using multiple "
    "devices under one thread. The support for such usage has been deprecated. "
    "For details, please refer to "
    "https://pytorch.org/docs/stable/distributed.html#multi-gpu-collective-functions. "
    "ProcessGroupNCCL continues supporting multi-process and multi-thread modes.";

ProcessGroupNCCL::ProcessGroupNCCL(
    c10::intrusive_ptr<Store> store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      store_(std::move(store)),
      options_(std::move(options)),
      terminateProcessGroup_(false),
      local_id_(process_group_id++),
      intraNodeComm_(initIntraNodeComm()) {
  TORCH_CHECK_WITH(
      ValueError,
      at::cuda::getNumGPUs() != 0,
      "ProcessGroupNCCL is only supported with GPUs, no GPUs found!");

  // getNcclVersion needs to get called before launching threads which can
  // potentially call getenv. getNcclVersion internally calls setenv to set some
  // environment variables from config file, which can race with getenv from
  // other threads and cause segfaults.
  const auto ncclVersion = getNcclVersion();
  this->setGroupUid(options_->group_name);
  this->localDeviceCount_ = static_cast<int>(at::cuda::getNumGPUs());
  logPrefix_ = createLogPrefix();
  blockingWait_ = getCvarBool(TORCH_NCCL_BLOCKING_WAIT, false);
  asyncErrorHandling_ = static_cast<ErrorHandlingMode>(
      getCvarInt(TORCH_NCCL_ASYNC_ERROR_HANDLING, 3 /*SkipCleanUp*/));
  enableNanCheck_ = getCvarBool(TORCH_NCCL_NAN_CHECK, false);
  cudaEventCacheEnabled_.store(getCvarBool(TORCH_NCCL_CUDA_EVENT_CACHE, true));
  traceBufferSize_ = getCvarInt(TORCH_NCCL_TRACE_BUFFER_SIZE, 2000);
  enableCollectiveHashDebug_ = (dist_debug_level_ >= DebugLevel::Detail);
  // store_ usually is wrapped with PrefixStore and the prefix is different
  // across different ProcessGroupNCCL(PG) instances. We need to get the
  // underlying non-PrefixStore for sharing global information shared across
  // different PGs.
  PrefixStore* prefixStore = dynamic_cast<PrefixStore*>(store_.get());
  globalStore_ =
      prefixStore ? prefixStore->getUnderlyingNonPrefixStore() : store_;
  auto desyncDebug = getCvarBool(TORCH_NCCL_DESYNC_DEBUG, false) ||
      (dist_debug_level_ >= DebugLevel::Detail);
#ifdef ENABLE_NCCL_ERROR_CHECKING
  enableTiming_.store(
      getCvarBool(TORCH_NCCL_ENABLE_TIMING, false) || desyncDebug);
#endif // ENABLE_NCCL_ERROR_CHECKING
  if (getCvarBool(TORCH_NCCL_AVOID_RECORD_STREAMS, false)) {
    TORCH_WARN_ONCE(
        "TORCH_NCCL_AVOID_RECORD_STREAMS is the default now, this environment variable is thus deprecated.");
  }
  showSerializationWarning_ =
      getCvarBool(TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING, true);

  if (blockingWait_) {
    LOG(INFO)
        << logPrefix()
        << "TORCH_NCCL_BLOCKING_WAIT is enabled, NO watchdog thread is created.";
  } else {
    if (desyncDebug && asyncErrorHandling_ == NoHandling) {
      LOG(INFO)
          << logPrefix()
          << "TORCH_NCCL_DESYNC_DEBUG and TORCH_NCCL_ASYNC_ERROR_HANDLING "
          << "must both be enabled. "
          << "Enabling TORCH_NCCL_ASYNC_ERROR_HANDLING.";
      asyncErrorHandling_ = SkipCleanUp;
    }
  }

  // If deterministic mode is enabled, we need to disable the NVLS algorithm in
  // NCCL.
  // TODO: remove this once NVLS supports deterministic mode.
  if (at::globalContext().deterministicAlgorithms()) {
    // Check if user have already set NCCL_ALGO. If already set, leave it.
    auto nccl_algo = c10::utils::get_env("NCCL_ALGO");
    if (!nccl_algo.has_value()) {
      LOG(INFO)
          << "torch deterministic mode is enabled, "
          << "disabling NVLS algorithm in NCCL which can lead to non-deterministic reduction.";
      // Sorry we have to disable NVLS for all collectives, be it all-reduce
      // or all-gather, because NCCL does not support per-collective
      // algorithm selection today.
      c10::utils::set_env("NCCL_ALGO", "^NVLS");
    }
  }

  // Initialize the heartbeat monitor/watchdog instance. This has to be done
  // before the corresponding thread is launched to avoid the error.
  heartbeatMonitor_ = std::make_unique<HeartbeatMonitor>(this);
  watchdog_ = std::make_unique<Watchdog>(this);

#ifdef ENABLE_NCCL_ERROR_CHECKING
  // in blockingWait mode, we don't need to enable the watchdog thread to check
  // the timeout or nccl error because the main thread would throw an exception
  // and it is the user's responsibility to handle the exception.
  if (!blockingWait_) {
    watchdog_->start();
  }
#endif // ENABLE_NCCL_ERROR_CHECKING

  init();
  const std::string OFF = "OFF";
  std::string torch_distributed_debug =
      getCvarString({"TORCH_DISTRIBUTED_DEBUG"}, OFF.c_str());
  LOG(INFO) << logPrefix()
            << "ProcessGroupNCCL initialization options: " << "size: " << size
            << ", global rank: " << globalRank()
            << ", TIMEOUT(ms): " << options_->timeout.count()
            << ", USE_HIGH_PRIORITY_STREAM: "
            << options_->is_high_priority_stream
            << ", SPLIT_FROM: " << options_->split_from
            << ", SPLIT_COLOR: " << options_->split_color
            << ", PG Name: " << options_->group_name;

  LOG(INFO) << logPrefix() << "ProcessGroupNCCL environments: "
            << "NCCL version: " << ncclVersion
            << ", TORCH_NCCL_ASYNC_ERROR_HANDLING: " << asyncErrorHandling_
            << ", TORCH_NCCL_ENABLE_TIMING: " << enableTiming_.load()
            << ", TORCH_NCCL_BLOCKING_WAIT: " << blockingWait_
            << ", TORCH_DISTRIBUTED_DEBUG: " << torch_distributed_debug
#ifdef NCCL_HAS_COMM_REGISTER
            << ", TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK: "
            << shouldAllCommunicatorsRegisterAllTensors()
#endif // NCCL_HAS_COMM_REGISTER
            << ", TORCH_NCCL_TRACE_BUFFER_SIZE: " << traceBufferSize_
            << ", TORCH_NCCL_NAN_CHECK: " << enableNanCheck_
            << ", TORCH_NCCL_CUDA_EVENT_CACHE: " << cudaEventCacheEnabled_;

  getGlobalRankStartAndStride(
      options_->global_ranks_in_group,
      this->globalRankStart_,
      this->globalRankStride_);

  // Attach hooks to cache allocator to trigger the hooks whenever a traced
  // action is called. In the following hooks, we register a newly allocated
  // segment when SEGMENT_ALLOC action occurs, and deregister a segment when
  // SEGMENT_FREE action occurs.
  if (shouldAllCommunicatorsRegisterAllTensors()) {
    // This call is idempotent.
    attachAllocatorHooks();
  }
}

void ProcessGroupNCCL::eagerConnectSingleDevice(at::Device device) {
  const auto key = getKeyFromDevice(device);
  LOG(INFO) << logPrefix() << "Eagerly connecting nccl backend with device "
            << device;
  initNCCLComm(key, device, OpType::ALLREDUCE);
  eagerInit_ = true;
}

bool ProcessGroupNCCL::useNonblocking() {
#ifndef NCCL_HAS_COMM_NONBLOCKING
  return false;
#endif // NCCL_HAS_COMM_NONBLOCKING
  // Already parsed, return the cached value
  if (useNonblocking_.has_value()) {
    return useNonblocking_.value();
  }
  // Get environment variable.
  auto nbEnv = c10::utils::check_env("TORCH_NCCL_USE_COMM_NONBLOCKING");

  // 1st priority: Respect the user's setting
  if (options_->config.blocking != NCCL_CONFIG_UNDEF_INT) {
    useNonblocking_ = options_->config.blocking == 0;
  }
  // 2nd priority: Respect the environment variable
  else if (nbEnv.has_value()) {
    useNonblocking_ = nbEnv;
  }
  // 3rd priority: automatically use nonblocking if we are in eager init mode
  // Note: this automatic selection is disabled in torch 2.7.1 to work around a
  // hang in NCCL 2.26 in non-blocking mode. We can revisit if NCCL fixes the
  // bug. See https://github.com/pytorch/pytorch/issues/153960
  // else if (getBoundDeviceId()) {
  //   useNonblocking_ = true;
  // }
  // 4th priority: otherwise, nonblocking = false to preserve old behavior
  else {
    useNonblocking_ = false;
  }

  LOG(INFO) << logPrefix()
            << "Using non-blocking mode: " << useNonblocking_.value();
  return useNonblocking_.value();
}

void ProcessGroupNCCL::performNocolorSplit(at::Device device) {
  // If our backend doesn't support splitting, this is a no-op for
  // ranks not in the new subgroup (and ranks that would be in it will
  // just use a new communicator rather than split).
#ifdef NCCL_HAS_COMM_SPLIT
  const auto key = getKeyFromDevice(device);
  LOG(INFO) << logPrefix() << "Performing nocolor split on backend device "
            << device << ", key " << key << ", i am " << this;
  bool useNb = useNonblocking();
  options_->config.blocking = useNb ? 0 : 1;
  auto comm = getNCCLComm(key);
  if (comm == nullptr) {
    LOG(ERROR) << logPrefix()
               << "No parent communicator exists for nocolor split";
  }
  NCCLComm::split(comm.get(), NCCL_SPLIT_NOCOLOR, rank_, options_->config);
#endif // NCCL_HAS_COMM_SPLIT
}

bool ProcessGroupNCCL::isInitialized() {
  if (devNCCLCommMap_.empty()) {
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  bool initialized = true;
  for (const auto& [_, comm] : devNCCLCommMap_) {
    if (!comm->isInitialized()) {
      initialized = false;
      break;
    }
  }
  return initialized;
}

ErrorType ProcessGroupNCCL::getError() {
  std::lock_guard<std::mutex> lock(errorMutex_);
  return error_;
}

void ProcessGroupNCCL::registerMemPool(at::cuda::MemPool* pool, bool symm) {
  const auto key = std::to_string(pool->device());
  LOG(INFO) << logPrefix()
            << "Performing NCCL user buffer registration for all buffers in "
            << "MemPool: " << pool->id() << ", device index: " << key
            << ", i am " << this;
  auto ncclComm = getNCCLComm(key);
  if (ncclComm == nullptr) {
    C10_THROW_ERROR(
        DistBackendError,
        "NCCL communicator has not been initialized before mem pool creation. You can pass `device_id` to init_process_group -- one way of eager initialization -- to work around this issue");
  }
  {
    std::lock_guard<std::mutex> lock(ncclCommMemPoolMapMutex);
    auto iter = ncclCommMemPoolMap.find(ncclComm);
    iter->second.insert(std::make_tuple(pool->id(), symm));
  }
  // We must ensure we're listening for allocator trace events in order to
  // register future segments allocated in this pool (this call is idempotent).
  attachAllocatorHooks();
  auto snapshot = c10::cuda::CUDACachingAllocator::snapshot(pool->id());
  for (const auto& segmentInfo : snapshot.segments) {
    TORCH_INTERNAL_ASSERT(
        segmentInfo.device == pool->device(),
        "Mismatch between CUDA memory segment device and pool's device");
    ncclComm->registerSegment(
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        reinterpret_cast<void*>(segmentInfo.address),
        segmentInfo.total_size,
        /*errorOnRereg=*/false, // ignores reregistration error
        /*window*/ symm); // whether to use NCCL symmetric memory
  }
}

void ProcessGroupNCCL::deregisterMemPool(at::cuda::MemPool* pool) {
  const auto key = std::to_string(pool->device());
  LOG(INFO) << logPrefix()
            << "Performing NCCL user buffer deregistration for all buffers in "
            << "MemPool: " << pool->id() << ", device index: " << key
            << ", i am " << this;
  auto ncclComm = getNCCLComm(key);
  if (ncclComm == nullptr) {
    C10_THROW_ERROR(
        DistBackendError,
        "NCCL communicator has not been initialized before mem pool creation. You can pass `device_id` to init_process_group -- one way of eager initialization -- to work around this issue");
  }
  bool symm;
  {
    std::lock_guard<std::mutex> lock(ncclCommMemPoolMapMutex);
    auto iter = ncclCommMemPoolMap.find(ncclComm);
    auto mempool_it = std::find_if(
        iter->second.begin(), iter->second.end(), [&](const auto& tup) {
          return std::get<0>(tup) == pool->id();
        });
    TORCH_CHECK(
        mempool_it != iter->second.end(),
        "Trying to unregister not previously registered pool");
    symm = std::get<1>(*mempool_it);
    iter->second.erase(mempool_it);
  }
  auto snapshot = c10::cuda::CUDACachingAllocator::snapshot(pool->id());
  for (const auto& segmentInfo : snapshot.segments) {
    TORCH_INTERNAL_ASSERT(
        segmentInfo.device == pool->device(),
        "Mismatch between CUDA memory segment device and pool's device");
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    ncclComm->deregisterSegment(
        reinterpret_cast<void*>(segmentInfo.address), symm);
  }
}

c10::intrusive_ptr<intra_node_comm::IntraNodeComm> ProcessGroupNCCL::
    initIntraNodeComm() {
  using IntraNodeComm = intra_node_comm::IntraNodeComm;
  if (!IntraNodeComm::isEnabled()) {
    return nullptr;
  }
  auto prefixStore = c10::make_intrusive<PrefixStore>("IntraNodeComm", store_);
  auto comm = c10::make_intrusive<IntraNodeComm>(prefixStore, rank_, size_);
  if (comm->rendezvous()) {
    return comm;
  } else {
    return nullptr;
  }
}

void ProcessGroupNCCL::setSequenceNumberForGroup() {
} // NCCL just starts sequence numbers at 0.

uint64_t ProcessGroupNCCL::getSequenceNumberForGroup() {
  return seqCollective_;
}

void ProcessGroupNCCL::registerOnCompletionHook(
    std::function<void(std::shared_ptr<WorkInfo>)>&& hook) {
  TORCH_WARN_ONCE(
      "ProcessGroupNCCL OnCompletion hook will be deprecated in favor of Flight Recorder. "
      "Please check out FlightRecorder.hpp for information that is recorded at work completion. "
      "You can file an issue if you want additional information to be recorded. "
      "You can also file an RFC if you want Flight Recorder to accept plugins that customize the recording.")

  TORCH_CHECK_WITH(
      DistBackendError,
      onCompletionHook_ == nullptr,
      "ProcessGroupNCCL OnCompletion hook already registered");

  TORCH_CHECK_WITH(
      ValueError,
      enableTiming_.load(),
      "ProcessGroupNCCL OnCompletion hook requires recording start and end "
      "events which require setting TORCH_NCCL_ENABLE_TIMING environment variable. "
      "This is only available for NCCL version >= 2.4.");
  onCompletionHook_ = std::move(hook);
  onCompletionHookThread_ = std::thread(&ProcessGroupNCCL::runHookLoop, this);
}

// must release GIL when calling this method
void ProcessGroupNCCL::waitForPendingWorks() {
  // Reasoning about hook completion:
  // 1. waitForPendingWorks should be called after user code has finished
  // calling
  //    all collectives. This means, when we got here, all of the collectives
  //    are either in workMetaList_ or has been erased from workMetaList_.
  // 2. The watchdog thread grabs both locks to move Work object from the
  //    workMetaList_ to the completedWorkList_, and the hook thread only erases
  //    a Work object after the hook is returned. Therefore, after user code
  //    calls a collective, its Work object is either in workMetaList_ or in
  //    completedWorkList_ before it finishes.
  // 3. We have three threads and two locks.
  //      a. main thread (this function) grabs two locks atomically
  //      b. watchdog thread (runLoop function) always grabs
  //      workMetaListMutex_
  //         first and then grabs completedWorkListMutex_.
  //      c. hook thread (runHookLoop function) only grabs
  //      completedWorkListMutex_. Therefore, locks are always acquired in the
  //      same order and hence no deadlocks.
  while (true) {
    {
      std::lock(workMetaListMutex_, completedWorkListMutex_);
      std::lock_guard<std::mutex> lockWork(workMetaListMutex_, std::adopt_lock);
      std::lock_guard<std::mutex> lockHook(
          completedWorkListMutex_, std::adopt_lock);

      if (workMetaList_.empty() && completedWorkList_.empty()) {
        return;
      }
    }

    std::this_thread::sleep_for(
        std::chrono::milliseconds(kWatchdogThreadSleepMillis));
  }
}

void ProcessGroupNCCL::enableCollectivesTiming() {
  enableTiming_.store(true);
}

c10::intrusive_ptr<Backend> ProcessGroupNCCL::split(
    const c10::intrusive_ptr<Store>& store,
    const std::vector<int>& ranks,
    const c10::intrusive_ptr<Backend::Options>& opts) {
  auto deviceIdx = guessDeviceId();
  TORCH_CHECK(
      deviceIdx >= 0,
      "ProcessGroupNCCL::split: rank ",
      rank_,
      " has no device is bound to this rank.");
  auto device = at::Device(at::DeviceType::CUDA, deviceIdx);
  auto it = std::find(ranks.begin(), ranks.end(), rank_);
  int groupRank;
  if (it == ranks.end()) {
    // This rank is not in the new group, so no_color split should be called
    performNocolorSplit(device);
    return nullptr;
  } else {
    groupRank = std::distance(ranks.begin(), it);
  }

  auto ncclOpts = c10::dynamic_intrusive_pointer_cast<Options>(opts);
  TORCH_CHECK(ncclOpts != nullptr, "opts not a ProcessGroupNCCL::Options.");

  // TODO: we need to get rid of globalRanksInGroup eventually.
  std::vector<uint64_t> globalRanksInGroup;
  for (auto rank : ranks) {
    globalRanksInGroup.emplace_back(groupRanks()[rank]);
  }
  ncclOpts->split_from =
      c10::intrusive_ptr<ProcessGroupNCCL>::unsafe_reclaim_from_nonowning(this);
  ncclOpts->global_ranks_in_group = std::move(globalRanksInGroup);
  auto color = genNcclSplitColor(ranks);
  ncclOpts->split_color = color;
  auto pg = c10::make_intrusive<ProcessGroupNCCL>(
      store->clone(), groupRank, ranks.size(), ncclOpts);
  pg->eagerConnectSingleDevice(device);
  return c10::static_intrusive_pointer_cast<Backend>(pg);
}

c10::intrusive_ptr<Backend> ProcessGroupNCCL::merge(
    const c10::intrusive_ptr<Store>& store,
    const c10::intrusive_ptr<Backend::Options>& opts,
    const int& rank,
    const int& size) {
  auto ncclOpts = c10::dynamic_intrusive_pointer_cast<Options>(opts);
  TORCH_CHECK(ncclOpts != nullptr, "opts not a ProcessGroupNCCL::Options.");
  auto pg = c10::make_intrusive<ProcessGroupNCCL>(
      store->clone(), rank, size, ncclOpts);
  return c10::static_intrusive_pointer_cast<Backend>(pg);
}

bool ProcessGroupNCCL::waitForFutureOrTimeout(
    std::future<bool>& fut,
    const std::chrono::milliseconds& timeOutMilSec,
    const std::string& futDescription,
    ::c10d::C10dLoggingData& debugLog,
    bool throwException) {
  std::string errorMsg;
  bool complete = false;

  TORCH_CHECK(fut.valid(), "Expected a valid future");
  std::future_status status = fut.wait_for(timeOutMilSec);
  if (status == std::future_status::ready) {
    // Calling .get() will re-raise any exception from the future, and we don't
    // care about the retval
    try {
      bool result = fut.get();
      if (result) {
        VLOG(2) << logPrefix()
                << "future successfully executed for: " << futDescription;
        debugLog.strings["status"] = "SUCCESS";
        complete = true;
      }
    } catch (const std::exception& e) {
      errorMsg = c10::str(
          logPrefix(),
          "Exception thrown when waiting for future ",
          futDescription,
          ": ",
          e.what());

      debugLog.strings["status"] = "EXCEPTION";
      debugLog.strings["exception_msg"] = e.what();
      LOG(ERROR) << errorMsg;
    } catch (...) {
      errorMsg = c10::str(
          logPrefix(),
          "Unknown exception thrown when waiting for future ",
          futDescription);
      debugLog.strings["status"] = "EXCEPTION";
      debugLog.strings["exception_msg"] = "Unknown
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

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

- **File Documentation**: `ProcessGroupNCCL.cpp_docs.md_docs.md`
- **Keyword Index**: `ProcessGroupNCCL.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
