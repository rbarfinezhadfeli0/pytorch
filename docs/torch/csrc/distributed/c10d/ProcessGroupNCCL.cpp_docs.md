# Documentation: ProcessGroupNCCL.cpp

## File Metadata
- **Path**: `torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp`
- **Size**: 223458 bytes
- **Lines**: 5987
- **Extension**: .cpp
- **Type**: Regular file

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
      debugLog.strings["exception_msg"] = "Unknown exception";
      LOG(ERROR) << errorMsg;
    }
  } else {
    errorMsg = c10::str(
        logPrefix(),
        "Future for ",
        futDescription,
        " timed out after ",
        timeOutMilSec.count(),
        " ms");
    debugLog.strings["status"] = "TIMEOUT";
    LOG(ERROR) << errorMsg;
  }
  if (throwException && !errorMsg.empty()) {
    C10_THROW_ERROR(DistBackendError, errorMsg);
  }
  return complete;
}

void ProcessGroupNCCL::abortCommsFromMap(
    std::unordered_map<std::string, std::shared_ptr<NCCLComm>>& ncclCommsMap,
    const std::optional<std::string>& abortReason) {
  // The process may control multiple devices, loop through the communicators on
  // each device
  // NCCL expects Group abort when there are multiple communicators created in a
  // device. Group abort requires 2.22.0 release and up.
  if (getNcclVersionNumber() >= NCCL_VERSION(2, 22, 0)) {
    groupStart();
  }
  for (auto& it : ncclCommsMap) {
    auto& devName = it.first;
    auto& ncclComm = it.second;
    VLOG(2) << logPrefix() << "ProcessGroupNCCL destroying ncclComm_ "
            << ncclComm->repr() << " on CUDA device: " << devName;
    // abort() call now has GPU guard inside
    ncclComm->abort(abortReason);
    // Note that we don't remove the aborted communicators from the
    // cache. The reason is that if we do remove the communicator
    // from the cache, it is possible that a new collective operation
    // calls `ncclCommInitRank` to create a new communicator whereas
    // other ranks might have failed/timed out and didn't enter
    // `ncclCommInitRank`. As a result, when there is a failure on
    // a communicator the application receives an exception and its
    // their responsibility to destroy the process group and recreate
    // it to recover from errors.

    VLOG(2) << logPrefix() << "ProcessGroupNCCL destroyed "
            << " communicator on CUDA device: " << devName;
  }
  if (getNcclVersionNumber() >= NCCL_VERSION(2, 22, 0)) {
    groupEnd();
  }
}

// Abort all communicators on this rank
// Note: original name of this method is `abort`. It was renamed to
// `abortComms` to distinguish from the `abort` method below. The `abort`
// method calls `abortComms` but does more destruction than the latter.
bool ProcessGroupNCCL::abortComms(
    const std::optional<std::string>& abortReason) {
  // Remove record from global ncclCommMemPoolMapMutex before aboarting,
  // so that a new cache segment would not register to already aborted
  // communicators. Note that ncclCommMemPoolMap is a global container which may
  // contain other PG's communicators, thus we need to only erase communicators
  // for the current PG.
  {
    std::lock_guard<std::mutex> lock(ncclCommMemPoolMapMutex);
    for (auto& [_, ncclComm] : devNCCLCommMap_) {
      ncclCommMemPoolMap.erase(ncclComm);
    }
  }

  std::lock_guard<std::mutex> lock(mutex_);
  abortCommsFromMap(devNCCLCommMap_, abortReason);
  abortCommsFromMap(inInitializationCommMap_, abortReason);
  return true;
}

void ProcessGroupNCCL::dumpExtraDebuggingInfo() {
  // This extra dump is intended to capture the current snapshot of collectives
  // When this process group is terminated for some exception out of NCCL
  bool dumpExtraOnExec_ = getCvarBool(TORCH_NCCL_EXTRA_DUMP_ON_EXEC, false);
  if (dumpExtraOnExec_) {
    bool should_dump_local = false;
    bool succeeded = shouldDump_.compare_exchange_strong(
        should_dump_local,
        true,
        std::memory_order_release,
        std::memory_order_acquire);
    if (succeeded) {
      LOG(INFO) << logPrefix() << "Sending extra dumping signal";
      broadcastDumpSignal();
      // When this routine is called, exception is captured so
      // dumping by default_pg is not guaranteed due to early termination of
      // process So we call dumping manually here
      bool onlyActive = getCvarBool(TORCH_INCLUDE_ONLY_ACTIVE, false);
      // Stacktrace is not included at the moment to prevent deadlock due to GIL
      dumpDebuggingInfo(false, onlyActive);
    }
  }
}

// Abort this backend.
void ProcessGroupNCCL::abort() {
  // This will log counter for how long the abort actually takes.
  STATIC_SCOPED_WAIT_COUNTER(pytorch.ProcessGroupNCCL__abort);

  dumpExtraDebuggingInfo();
  // Don't join threads here since the purpose of this method is to abort all
  // communicators and signal the threads to exit. Joining on the threads could
  // potentially block and hence avoid it in this method.
  terminateProcessGroup_.store(true);
  watchdog_->notify();
  // launch abort asynchronously and wait for it to complete or timeout
  LOG(INFO) << logPrefix()
            << "Launching ProcessGroupNCCL abort asynchronously.";
  std::future<bool> fut =
      std::async(std::launch::async, [this]() { return this->abortComms(); });

  ::c10d::C10dLoggingData debugLog;
  waitForFutureOrTimeout(
      fut, options_->timeout, "ProcessGroup abort", debugLog, true);
  LOG(INFO) << logPrefix() << "ProcessGroupNCCL aborts successfully.";

  // We need to wait for abort to finish before we can safely shut down
  // heartbeat monitoring thread.
  heartbeatMonitor_->stop();
}

// Difference between `abort()` and `shutdown()`:
// 1. `abort()` will signal communicators to terminate all NCCL kernels
// immediately.
// 2. `shutdown()` will wait for all NCCL kernels to finish before destroying
// communicators.

// Destroy (shutdown) this backend -- normal exit.
void ProcessGroupNCCL::shutdown() {
  LOG(INFO) << logPrefix()
            << "Starting to destroy process group, flushing operations.";
  // Flush all collectives
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& it : devNCCLCommMap_) {
      auto& ncclComm = it.second;
      ncclComm->finalize();
    }
  }
  // Wait for all operations to complete.  If NCCL comm is non-blocking and
  // timeout is reach, this will throw an exception.
  for (auto& it : devNCCLCommMap_) {
    auto& ncclComm = it.second;
    // Use long interval to avoid acquiring CPU too frequently
    ncclComm->waitReady(true);
  }
  // Deregister memory pool after finalizing all collectives
  if (memPool_) {
    try {
      deregisterMemPool(memPool_.get());
    } catch (...) {
      LOG(ERROR) << logPrefix() << "Failed to deregister memory pool, ignoring";
    }
  }
  // Tell watchdog to (1) flush its queue and (2) do not use comm objects
  // anymore because I am going to destroy them now
  LOG(INFO) << logPrefix() << "Operations flushed, joining watchdog thread.";
  terminateProcessGroup_.store(true);
  watchdog_->notify();
  watchdog_->join();
  if (onCompletionHookThread_.joinable()) {
    onCompletionHookThread_.join();
  }
  // Watchdog thread exiting, retire heartbeat monitoring thread now to avoid
  // false alarm
  heartbeatMonitor_->stop();
  // Destroy the communicator, reclaim resources
  LOG(INFO) << logPrefix() << "Watchdog joined, destroying NCCL communicators.";
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& it : devNCCLCommMap_) {
      auto& ncclComm = it.second;
      ncclComm->destroy();
    }
  }
  LOG(INFO) << logPrefix() << "Destroy complete.";
}

// NOLINTNEXTLINE(bugprone-exception-escape)
ProcessGroupNCCL::~ProcessGroupNCCL() {
  LOG(INFO) << logPrefix() << "ProcessGroupNCCL destructor entered.";

  // `shutdown()` or `abort` already called. Skip the favor of disposing
  // communicators.
  if (!terminateProcessGroup_.load()) {
    // If user haven't explicitly destroy/shutdown process group, destructor
    // needs to do so
    // First print warning on first rank of each node
    if (rank_ % localDeviceCount_ == 0) {
      TORCH_WARN_ONCE(
          "WARNING: destroy_process_group() was not called before program exit, "
          "which can leak resources. For more info, please see "
          "https://pytorch.org/docs/stable/distributed.html#shutdown");
    }

    // Note 1: in distributed_c10d.py, a reference to PG is held by the global
    // context. Therefore, we are here only when the global context is tearing
    // down, which means the entire program is exiting.  At this point, user
    // will no longer care about the result of any collective, thus we can use
    // abort instead of destroy to make the destruction non-blocking.

    // TODO: Note 1 is not true in case of a C++ program using libtorch, which
    // does not have the global context mentioned. In that case, calling
    // `abort()` here could lead to corrupted result. We should consider not
    // doing anything and just let things leak. Adversarial example:
    /*
      Work routine(Tensor& t) {
        pg = ProcessGroupNCCL(…);
        w = pg.allReduce(t);
        return w;
      }
    */
    abort();
  }

  // Make sure we've told threads to stop; doesn't hurt if we'd done so before.
  // Tell watchdog and onCompletionHook:
  terminateProcessGroup_.store(true);
  watchdog_->notify();
  // Tell heartbeat thread:
  heartbeatMonitor_->stop();

  // Wait for all threads to finish before returning
  watchdog_->join();
  heartbeatMonitor_->join();
  if (onCompletionHookThread_.joinable()) {
    onCompletionHookThread_.join();
    LOG(INFO) << logPrefix()
              << "ProcessGroupNCCL onCompletionHookThread thread joined.";
  }
}

bool ProcessGroupNCCL::dumpDebuggingInfo(
    bool includeStackTrace /*=true*/,
    bool onlyActive /*=false*/) {
  // This will log counter for how long dumpDebuggingInfo actually takes.
  STATIC_SCOPED_WAIT_COUNTER(pytorch.ProcessGroupNCCL__dumpDebuggingInfo);

  // Serialize all calls to this function to avoid corrupting data, but allow
  // multiple calls in one runtime. User is responsible for preserving the
  // output file from an earlier call before a later call overwrites it.
  static std::mutex writeDebugInfoMutex;
  LOG(ERROR)
      << logPrefix()
      << "ProcessGroupNCCL preparing to dump debug info. Include stack trace: "
      << includeStackTrace << ", only active collectives: " << onlyActive;
  if (traceBufferSize_ > 0) {
    // We dump nccl trace into local disk by default and users can register
    // their customized writer by inheriting `DebugInfoWriter` via
    // `registerDebugInfoWriter`.
    auto ncclTrace = dump_nccl_trace(true, includeStackTrace, onlyActive);
    // dump_nccl_trace will hang so we don't grab the global lock until we get
    // the trace.
    std::lock_guard<std::mutex> lock(writeDebugInfoMutex);
    DebugInfoWriter& writer = DebugInfoWriter::getWriter(globalRank());
    LOG(INFO) << logPrefix() << "ProcessGroupNCCL dumping nccl trace to "
              << writer.getWriterTarget();
    writer.write(ncclTrace);
    LOG(INFO) << logPrefix() << "Flight Recorder trace successfully dumped.";
    return true;
  }
  return false;
}

void ProcessGroupNCCL::terminateProcess(const std::string& errMsg) {
  // Logging with `FATAL`, after errMsg printed, it calls `std::abort()`
  // to terminate the program execution.
  LOG(FATAL) << logPrefix() << errMsg;
}

static long computeDeltaMS(
    std::chrono::time_point<std::chrono::steady_clock> start,
    std::chrono::time_point<std::chrono::steady_clock> end) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

void ProcessGroupNCCL::setEnableNanCheck(bool enableNanCheck) {
  enableNanCheck_ = enableNanCheck;
}

std::string ProcessGroupNCCL::HeartbeatMonitor::getNCCLWatchdogTimeoutErrorMsg(
    const std::string& extraMsg) {
  return c10::str(
      pg_->logPrefix(),
      "Received a dump signal due to a collective timeout from ",
      extraMsg,
      " and we will try our best to dump the debug info. ",
      "Last enqueued NCCL work: ",
      pg_->pgStatus_->lastEnqueuedSeq,
      ", last completed NCCL work: ",
      pg_->pgStatus_->lastCompletedSeq,
      ".",
      "This is most likely caused by incorrect usages of collectives, e.g., wrong ",
      "sizes used across ranks, the order of collectives is not same for all ranks ",
      "or the scheduled collective, for some reason, didn't run. Additionally, ",
      "this can be caused by GIL deadlock or other reasons such as network errors or ",
      "bugs in the communications library (e.g. NCCL), etc. ");
}

std::string ProcessGroupNCCL::HeartbeatMonitor::getNCCLWatchdogTimeoutExitMsg(
    const std::string& exitReason) {
  return c10::str(
      pg_->logPrefix(),
      "Terminating the process after attempting to dump debug info, due to ",
      exitReason,
      ".");
}

void ProcessGroupNCCL::HeartbeatMonitor::setLastWorkListUpdateTime(
    std::chrono::time_point<std::chrono::steady_clock> time) {
  // We intentionally let the race condition to happen but this is ok
  // as long as we update the time, we know we are making progress.
  lastWorkListUpdateTime_ = time;
}

int ProcessGroupNCCL::HeartbeatMonitor::getDumpTimeout() const {
  return waitTimeoutDumpInMilSec_;
}

ProcessGroupNCCL::HeartbeatMonitor::HeartbeatMonitor(ProcessGroupNCCL* pg) {
  pg_ = pg;
  heartbeatTimeoutInSec_ =
      getCvarInt(TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC, 60 * 8 /*8 Mins*/);
  waitTimeoutDumpInMilSec_ =
      getCvarInt(TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC, 15 * 1000 /*15 Sec*/);
  coordCheckIntervalMilSec_ = getCvarInt(TORCH_NCCL_COORD_CHECK_MILSEC, 1000);
  // TODO, we should either deprecate TORCH_NCCL_DUMP_ON_TIMEOUT
  // or change its name to reflect that dump happens on exception including
  // both timeout and other errors.
  dumpOnTimeoutOrEx_ = getCvarBool(TORCH_NCCL_DUMP_ON_TIMEOUT, true);
  // logging C++ stack isn't safe. Gate it with an ENV.
  logCppStackOnUncleanShutdown_ =
      getCvarBool(TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN, true);
  watchdogHeartbeatMonitorEnabled_ =
      getCvarBool(TORCH_NCCL_ENABLE_MONITORING, true);

  // print out ENV settings for the heartbeat monitor thread.
  LOG(INFO)
      << pg_->logPrefix() << "HeartbeatMonitor environments: "
      << "TORCH_NCCL_ENABLE_MONITORING (Whether to kill program when no watchdog heartbeat detected): "
      << watchdogHeartbeatMonitorEnabled_
      << ", TORCH_NCCL_DUMP_ON_TIMEOUT: " << dumpOnTimeoutOrEx_
      << ", TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC: " << waitTimeoutDumpInMilSec_
      << ", TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC: " << heartbeatTimeoutInSec_
      << ", TORCH_NCCL_COORD_CHECK_MILSEC: " << coordCheckIntervalMilSec_
      << ", TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: "
      << logCppStackOnUncleanShutdown_;
}

void ProcessGroupNCCL::HeartbeatMonitor::stop() {
  terminateHeartbeatMonitorThread_.store(true);
  monitorWakeUpCV_.notify_one();
}

void ProcessGroupNCCL::HeartbeatMonitor::start() {
  TORCH_CHECK(
      !ncclHeartbeatMonitorThread_.joinable(),
      "HeartbeatMonitor thread already started");
  ncclHeartbeatMonitorThread_ =
      std::thread(&ProcessGroupNCCL::HeartbeatMonitor::runLoop, this);
}

void ProcessGroupNCCL::HeartbeatMonitor::join() {
  if (ncclHeartbeatMonitorThread_.joinable()) {
    ncclHeartbeatMonitorThread_.join();
    LOG(INFO) << pg_->logPrefix()
              << "ProcessGroupNCCL heart beat monitor thread joined.";
  }
}

void ProcessGroupNCCL::HeartbeatMonitor::runLoop() {
  c10::setThreadName("pt_nccl_heartbt");
  STATIC_SCOPED_WAIT_COUNTER(
      pytorch.ProcessGroupNCCL__HeartbeatMonitor__runLoop);

  uint64_t heartBeatCounter = 0ULL;
  std::string errorMsg;
  std::string exitReason;
  bool checkDumpSignal = (dumpOnTimeoutOrEx_ && pg_->getUid() == 0);
  int monitorPollInterval = checkDumpSignal ? coordCheckIntervalMilSec_
                                            : heartbeatTimeoutInSec_ * 1000;
  auto lastTimePollStore = std::chrono::steady_clock::now();
  auto lastTimeHeartBeatCheck = std::chrono::steady_clock::now();
  std::optional<DumpPipe> dumpPipe = std::nullopt;
  // Use a pool to temporarily store the futures to avoid blocking when the code
  // exits the scope of when future is generated by std::async.
  std::vector<std::future<bool>> futures;

  if (pg_->getUid() == 0) {
    // DumpPipe is one per-trainer process, and its convenient to name them
    // after 'global' ranks in the system, So we assume processgroup (uid)==0 is
    // the global PG and has globally unique rank ids across trainers.
    dumpPipe.emplace(pg_->globalRank());
  }
  while (true) {
    // This won't have any lock since this lock is only used here.
    // Please be aware that mutex `monitorMutex_` should not be used
    // somewhere else to avoid the deadlock.
    std::unique_lock<std::mutex> lock(monitorMutex_);
    if (monitorWakeUpCV_.wait_for(
            lock, std::chrono::milliseconds(monitorPollInterval), [&] {
              return terminateHeartbeatMonitorThread_.load();
            })) {
      // For the normal complete or user interception, monitorWakeUpCV_
      // will get notified, we early return and exit heartbeatMonitor.
      return;
    }
    auto currentTime = std::chrono::steady_clock::now();

    // We put extra functionality in the thread for the default PG (aka,
    // local_id_=0) because the signal is same across different PGs. We only
    // need to run once per process to avoid duplicate things performed in too
    // many separate threads. For example, we check a global flag on the
    // TCPStore periodically to see if any PG on any rank observed a timeout and
    // signaled peers to dump debugging info, and we avoid hammering the
    // TCPStore from all PGs on the same rank.
    if (checkDumpSignal) {
      // There are two scenarios where monitor thread will dump on timeout:
      // 1. The current rank is the first to observe a timeout in watchdog.
      // (shouldDump_ was set to true by the watchdog thread).
      // 2. Other ranks detected the timeout and signal the current rank to
      // dump. In addition, monitor threads will dump if watchdog threads has no
      // heartbeat or dumpPipe is not empty.
      if (shouldDump_.load()) {
        errorMsg = getNCCLWatchdogTimeoutErrorMsg("this local rank");
        exitReason = "collective timeout or exception";
        break;
      }
      // We poll store to see if some ranks have flagged a timeout when
      // we haven't polled for `heartbeat_timeout` seconds and there haven't
      // any work added or removed for `watchdog_timeout` seconds.
      if (computeDeltaMS(lastWorkListUpdateTime_, currentTime) >=
              kWatchdogThreadSleepMillis &&
          computeDeltaMS(lastTimePollStore, currentTime) >=
              coordCheckIntervalMilSec_) {
        lastTimePollStore = currentTime;
        auto handleError = [&](const std::string& errorMessage) {
          LOG(WARNING)
              << pg_->logPrefix()
              << "Failed to check the \"should dump\" flag on TCPStore, "
              << "(maybe TCPStore server has shut down too early), with error: "
              << errorMessage;
          // We give up for now assuming TCPStore has been torn down.
          return;
        };
        // Wrap globalStore_->check() in a try-catch block to avoid crashing if
        // the store is not available.
        bool checkExceptionDump = false;
        try {
          checkExceptionDump =
              pg_->globalStore()->check({std::string(kStoreDumpKey)});
        } catch (const c10::DistNetworkError& e) {
          handleError(e.msg());
        } catch (const std::exception& e) {
          handleError(e.what());
        }

        if (checkExceptionDump) {
          int timeOutRank = -1;
          if (!shouldDump_.load()) {
            LOG(ERROR)
                << pg_->logPrefix()
                << "Observed flight recorder dump signal from another rank via TCPStore.";
          }
          shouldDump_.store(true);
          try {
            auto vec = pg_->globalStore()->get(std::string(kStoreDumpKey));
            TORCH_CHECK_WITH(
                DistBackendError,
                vec.size() == sizeof(int),
                "Invalid size for the timeout rank ID");
            std::memcpy(&timeOutRank, vec.data(), vec.size());
          } catch (const std::exception& e) {
            LOG(ERROR) << pg_->logPrefix()
                       << "Failed to get timeout rank ID from TCPStore."
                       << e.what();
          }
          errorMsg =
              getNCCLWatchdogTimeoutErrorMsg(c10::str(" rank ", timeOutRank));
          exitReason = "collective timeout or exception";
          break;
        }
      }
    }

    if (computeDeltaMS(lastTimeHeartBeatCheck, currentTime) >=
        heartbeatTimeoutInSec_ * 1000l) {
      // Check the heart beat of watchdog thread.
      lastTimeHeartBeatCheck = currentTime;
      auto heartbeat = pg_->getWatchdogHeartbt();
      if (heartbeat != heartBeatCounter) {
        heartBeatCounter = heartbeat;
      } else {
        shouldDump_.store(true);
        // Watchdog heartbeat timeout.
        errorMsg = c10::str(
            pg_->logPrefix(),
            "ProcessGroupNCCL's watchdog got stuck for ",
            heartbeatTimeoutInSec_,
            " seconds without making progress in monitoring enqueued collectives. ",
            "This typically indicates a NCCL/CUDA API (e.g., CudaEventDestroy) hang blocking the watchdog, ",
            "and could be triggered by another thread holding the GIL inside a ",
            "CUDA api (for example, CudaEventDestroy), or other deadlock-prone behaviors.",
            "If you suspect the watchdog is not actually stuck and a longer timeout would help, ",
            "you can either increase the timeout (TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC) to a larger value "
            "or disable the heartbeat monitor (TORCH_NCCL_ENABLE_MONITORING=0)."
            "If either of aforementioned helps, feel free to file an issue to PyTorch about the short timeout "
            "or false positive abort; otherwise, please attempt to debug the hang. ");
        exitReason = "ProcessGroupNCCL watchdog hang";
        break;
      }
    }
    // process a request to dump the trace. only PG uid 0 will respond to dump
    // requests, but this is fine since all PG's feed into the same flight
    // recorder and dump. After dump, the training should continue.
    if (dumpPipe.has_value() && dumpPipe->shouldDump()) {
      // best effort dump, not waiting for the dump here
      bool onlyActive = getCvarBool(TORCH_INCLUDE_ONLY_ACTIVE, false);
      LOG(INFO) << pg_->logPrefix()
                << "Dump signal received through pipe, triggering FR dump.";
      futures.emplace_back(std::async(std::launch::async, [this, onlyActive]() {
        return this->pg_->dumpDebuggingInfo(true, onlyActive);
      }));
    }
  }
  LOG(ERROR) << errorMsg;

  // We perform some checks to help users debug the timeout/hang issue:
  // 1. Dump the nccl trace (flight recorder) to help debug the issue
  //    (timeout after waitTimeoutDumpInMilSec_, which is one minute).
  // 2. Check if there is a GIL deadlock (timeout after 300ms).
  // 3. Try to dump the c++ stacktraces (blocking and would hang,
  //    users can turn this off by set
  //    TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN=0).

  // Dump the nccl trace (flight recorder).
  if (checkDumpSignal && shouldDump_.load()) {
    // Store debug info to storage if no other thread does it. (By default to
    // local disk)
    bool dumpStackTrace = getCvarBool(TORCH_INCLUDE_STACK_TRACE, true);
    bool onlyActive = getCvarBool(TORCH_INCLUDE_ONLY_ACTIVE, false);
    ::c10d::C10dLoggingData debugLog;
    debugLog.integers["pg_id"] = static_cast<int64_t>(pg_->getUid());
    debugLog.integers["rank"] = pg_->getRank();
    debugLog.integers["global_rank"] = pg_->globalRank();
    debugLog.integers["world_size"] = pg_->getSize();
    debugLog.strings["flight_recorder_version"] = c10d::version_val_str;
    for (int i = 0; i < 2; i++) {
      std::future<bool> asyncDebugDump =
          std::async(std::launch::async, [this, dumpStackTrace, onlyActive]() {
            return this->pg_->dumpDebuggingInfo(dumpStackTrace, onlyActive);
          });

      // wait for the dump until timeout - log data
      auto complete = pg_->waitForFutureOrTimeout(
          asyncDebugDump,
          std::chrono::milliseconds(waitTimeoutDumpInMilSec_),
          "Flight recorder dump in heartbeatMonitor",
          debugLog,
          false);

      if (complete) {
        LOG(INFO)
            << pg_->logPrefix()
            << "Finished flight recorder successfully. Output can be analyzed using the fr_trace script.";
        if (i > 0) {
          debugLog.strings["exception_msg"] = "Dump with stack trace failed.";
        }
        break;
      }
      // If we failed to dump, try dumping without stack trace in the 2nd
      // iteration.
      dumpStackTrace = false;
      futures.emplace_back(std::move(asyncDebugDump));
    }
    debugLog.integers["trace_enabled"] = int64_t(dumpStackTrace);
    auto logger = c10d::C10dLogger::getLogger();
    if (logger) {
      logger->log(debugLog);
    }
  }

  // GIL deadlock check.
  if (get_gil_checker() != nullptr) {
    auto fut = launchAsyncGilCheck();
    auto kGilCheckTimeout = std::chrono::milliseconds(300);
    auto futStatus = fut.wait_for(kGilCheckTimeout);
    if (futStatus != std::future_status::ready) {
      TORCH_CHECK(
          futStatus != std::future_status::deferred,
          "Expected the future to have been launched eagerly.");
      LOG(ERROR)
          << pg_->logPrefix()
          << "Could not acquire GIL within 300 ms on exit, possible GIL induced hang";
    }
  } else {
    VLOG(2)
        << pg_->logPrefix()
        << "GIL checker was not registered, perhaps this is a no-python build?";
  }

  // Dump the c++ stacktraces.
  auto& cpp_dumper = get_cpp_trace_dumper();
  if (logCppStackOnUncleanShutdown_ && cpp_dumper.has_value()) {
    LOG(INFO) << pg_->logPrefix() << "Dumping c++ stacktraces:";
    cpp_dumper.value()([&](const std::string& line) {
      LOG(INFO) << pg_->logPrefix() << line;
    });
    LOG(INFO) << pg_->logPrefix() << "Finished c++ stacktraces dump.";
  }

  // There are two possible cases for the watchdog thread exit:
  // Case one: desync report runs quickly, and it follows the step:
  // collective timeout -> desync -> exception handling -> throwing exception.
  // The program will exit because of exception thrown and the code below will
  // not be run.
  //
  // Case two: desync might be slow or get stuck and we need to wait
  // extra time to avoid we kill the program too early.
  //
  // Or we get stuck in destructors, we will sleep for some time before calling
  // std::abort() to kill the whole process.
  if ((pg_->terminateProcessGroup_.load() || shouldDump_.load()) &&
      !terminateHeartbeatMonitorThread_.load()) {
    std::this_thread::sleep_for(std::chrono::seconds(heartbeatTimeoutInSec_));
    LOG(INFO)
        << pg_->logPrefix() << "slept for " << heartbeatTimeoutInSec_
        << " because we want to wait longer to verify there is indeed a watchdog hang.";
  }

  // At this point, we either already sleep for another `heartbeatTimeoutInSec_`
  // or the thread has finished. Because we don't want to block the monitor
  // thread, so We mark the thread detach and the dump of debug info becomes
  // "best effort". If the process exit normally, marking it detach also makes
  // sense because we don't really care about dumping the debug info.

  // We already log completion inside the thread, so it may not be necessary to
  // check the return value here.  We mainly use a future so we can exit early
  // if done.
  if (!terminateHeartbeatMonitorThread_.load()) {
    // Create a error message reported from MonitorThread, so
    // we throw exception and make the whole process to be killed.
    // TODO(fduwjj): After having a hang debug wiki, we need to update the wiki
    // url here.
    if (watchdogHeartbeatMonitorEnabled_) {
      pg_->terminateProcess(getNCCLWatchdogTimeoutExitMsg(exitReason));
    } else {
      // Ideally we want to merge this one with the above one, but we are going
      // to remove the kill switch for monitor thread soon, so we keep this one
      // for now.
      LOG(ERROR)
          << pg_->logPrefix()
          << "ProcessGroupNCCL monitor thread is disabled, but would have terminated the process"
          << "after attempting to dump debug info, due to " << exitReason
          << ".";
    }
  }
}

ProcessGroupNCCL::Watchdog::Watchdog(ProcessGroupNCCL* pg) {
  pg_ = pg;
  heartbeat_ = 1ULL;
  rethrowCUDAErrors_ = getCvarBool(TORCH_NCCL_RETHROW_CUDA_ERRORS, true);
  propagatePgError_ = getCvarBool(TORCH_NCCL_PROPAGATE_ERROR, false);
  desyncDebug_ = getCvarBool(TORCH_NCCL_DESYNC_DEBUG, false) ||
      (pg_->dist_debug_level_ >= DebugLevel::Detail);

  // print out ENV settings for the watchdog thread.
  LOG(INFO) << pg_->logPrefix() << "PGNCCL Watchdog environments: "
            << "TORCH_NCCL_RETHROW_CUDA_ERRORS: " << rethrowCUDAErrors_
            << ", TORCH_NCCL_PROPAGATE_ERROR: " << propagatePgError_
            << ", TORCH_NCCL_DESYNC_DEBUG: " << desyncDebug_;

  // Enable Desync Debugger per user setting
  if (desyncDebug_) {
    desyncDebugger_.init(
        pg_->getRank(),
        pg_->getSize(),
        pg_->globalRank(),
        pg_->getUid(),
        pg_->store_);
  }
}

void ProcessGroupNCCL::Watchdog::notify() {
  workMetaListCV_.notify_one();
}

void ProcessGroupNCCL::Watchdog::start() {
  TORCH_CHECK(
      !ncclCommWatchdogThread_.joinable(), "Watchdog thread already started");
  ncclCommWatchdogThread_ = std::thread(&ProcessGroupNCCL::Watchdog::run, this);
}

void ProcessGroupNCCL::Watchdog::join() {
  if (ncclCommWatchdogThread_.joinable()) {
    ncclCommWatchdogThread_.join();
    LOG(INFO) << pg_->logPrefix() << "ProcessGroupNCCL watchdog thread joined.";
  }
}

void ProcessGroupNCCL::Watchdog::run() {
  c10::setThreadName("pt_nccl_watchdg");
  STATIC_SCOPED_WAIT_COUNTER(pytorch.ProcessGroupNCCL__Watchdog__run);

  try {
    VLOG(2) << pg_->logPrefix() << "Process group watchdog thread started!";
    pg_->heartbeatMonitor_->start();
    runLoop();
    VLOG(2) << pg_->logPrefix()
            << "Process group watchdog thread terminated normally";
  } catch (std::exception& e) {
    // This condition is triggered when any routine in watchdog gets an
    // exception
    pg_->dumpExtraDebuggingInfo();
    if (std::string(e.what()).find("driver shutting down") !=
        std::string::npos) {
      VLOG(2)
          << pg_->logPrefix()
          << "main process destroyed cuda before watchdog loop exited, terminating watchdog."
          << " (Watchdog caught exception: " << e.what();

    } else {
      // Append error message reported from runLoop
      const auto exitMsg = c10::str(
          pg_->logPrefix(),
          "Process group watchdog thread terminated with exception: ",
          e.what());
      LOG(ERROR) << exitMsg;
      if (C10_LIKELY(rethrowCUDAErrors_) ||
          !(std::string(e.what()).find("CUDA Error"))) {
        // TODO(whc) clean up the rethrow - why is it stored in a class var and
        // rethrown?
        watchDogException_ =
            std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exitMsg));
        std::rethrow_exception(watchDogException_);
      }
    }
  } catch (...) {
    const auto exitMsg = c10::str(
        pg_->logPrefix(),
        "Process group watchdog thread terminated with exception: unknown");
    LOG(ERROR) << exitMsg;
    watchDogException_ =
        std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exitMsg));
    std::rethrow_exception(watchDogException_);
  }
}

int ProcessGroupNCCL::Watchdog::getSignalSrcRank(
    c10::intrusive_ptr<Store>& store,
    const std::string& signal) {
  // This function is 'non blocking'. We first 'check' if the key exists in the
  // store, then read/get the value only if the key exists.
  int srcRank = -1;
  bool signalExists = false;
  try {
    signalExists = store->check({signal});
  } catch (const std::exception& e) {
    LOG(WARNING) << pg_->logPrefix() << "Failed to check the signal " << signal
                 << " on TCPStore, " << e.what();
  }
  if (!signalExists) {
    return srcRank;
  }

  // key exists, now read and parse the value (source rank)
  std::vector<uint8_t> vec;
  try {
    vec = store->get(std::string(signal));
  } catch (const std::exception& e) {
    LOG(ERROR) << pg_->logPrefix() << "Failed to get source rank of the signal "
               << signal << " from TCPStore." << e.what();
  }
  TORCH_CHECK_WITH(
      DistBackendError,
      vec.size() == sizeof(int),
      "Invalid size for the timeout rank ID");
  std::memcpy(&srcRank, vec.data(), vec.size());
  return srcRank;
}

void ProcessGroupNCCL::Watchdog::checkAndSetRemoteError() {
  // if the error is already set, no need to check again
  if (pg_->getError() != ErrorType::SUCCESS) {
    return;
  }
  // key/signal to read from the tcpstore is a string and pg specific:
  // format is: remote_error:pg_uid
  int remoteErrorRank = getSignalSrcRank(
      pg_->store_, std::string(kStoreErrorSignalKey) + ':' + pg_->pg_uid_);
  if (remoteErrorRank != -1) {
    std::lock_guard<std::mutex> lock(pg_->errorMutex_);
    pg_->error_ = ErrorType::REMOTE_ERROR;
    LOG(ERROR) << c10::str(
        pg_->logPrefix(),
        " remote error detected from rank: ",
        remoteErrorRank);
  }
}

void ProcessGroupNCCL::Watchdog::runLoop() {
  bool done = false;
  pg_->heartbeatMonitor_->setLastWorkListUpdateTime(
      std::chrono::steady_clock::now());
  auto lastStatusUpdateTime = std::chrono::steady_clock::now();
  std::list<ProcessGroupNCCL::WorkNCCL> completedWorkList;

  while (!done || !pg_->terminateProcessGroup_.load()) {
    std::unique_lock<std::mutex> lock(pg_->workMetaListMutex_);
    // We busy-poll the work vector every kWatchdogThreadSleepMillis
    // milliseconds as long as the atomic is True.
    workMetaListCV_.wait_for(
        lock,
        std::chrono::milliseconds(kWatchdogThreadSleepMillis),
        [&]() -> bool { return pg_->terminateProcessGroup_.load(); });
    // Bump up heart beat by one.
    heartbeat_++;

// Some versions of GLOG support less-spammy version of LOG_EVERY_MS
// in which case we don't want to spam the logs.
#ifdef LOG_EVERY_MS
    // Log the progress of this PG periodically
    C10_LOG_EVERY_MS(INFO, kWorkStatusUpdatePeriodMs) << c10::str(
        logPrefix(),
        "NCCL Work update periodically: ",
        "last enqueued NCCL work: ",
        pg_->pgStatus_->lastEnqueuedSeq,
        ", last completed NCCL work: ",
        pg_->pgStatus_->lastCompletedSeq,
        ".");
#endif // LOG_EVERY_MS
    auto logger = ::c10d::C10dLogger::getLogger();
    if (logger &&
        computeDeltaMS(
            lastStatusUpdateTime, std::chrono::steady_clock::now()) >=
            kWorkStatusUpdatePeriodMs) {
      ::c10d::C10dLoggingData data;
      // logging integers
      data.integers["pg_id"] = static_cast<int64_t>(pg_->local_id_);
      data.integers["rank"] = pg_->rank_;
      data.integers["global_rank"] = pg_->globalRank();
      data.integers["last_enqueued_work"] = pg_->pgStatus_->lastEnqueuedSeq;
      data.integers["last_started_work"] = pg_->pgStatus_->lastStartedSeq;
      data.integers["last_completed_work"] = pg_->pgStatus_->lastCompletedSeq;
      data.integers["last_enqueued_numel_in"] =
          static_cast<int64_t>(pg_->pgStatus_->lastEnqueuedNumelIn);
      data.integers["last_enqueued_numel_out"] =
          static_cast<int64_t>(pg_->pgStatus_->lastEnqueuedNumelOut);
      data.integers["last_completed_numel_in"] =
          static_cast<int64_t>(pg_->pgStatus_->lastCompletedNumelIn);
      data.integers["last_completed_numel_out"] =
          static_cast<int64_t>(pg_->pgStatus_->lastCompletedNumelOut);
      data.integers["last_started_numel_in"] =
          static_cast<int64_t>(pg_->pgStatus_->lastStartedNumelIn);
      data.integers["last_started_numel_out"] =
          static_cast<int64_t>(pg_->pgStatus_->lastStartedNumelOut);
      // logging strings
      data.strings["last_enqueued_work_name"] =
          pg_->pgStatus_->lastEnqueuedWorkName;
      data.strings["last_started_work_name"] =
          pg_->pgStatus_->lastStartedWorkName;
      data.strings["last_completed_work_name"] =
          pg_->pgStatus_->lastCompletedWorkName;
      data.strings["pg_name"] = pg_->pg_uid_;
      data.strings["pg_desc"] = pg_->pg_desc_;
      logger->log(data);
      lastStatusUpdateTime = std::chrono::steady_clock::now();
    }

    if (propagatePgError_) {
      // Check and set remote error if it has not been set before
      checkAndSetRemoteError();
    }

    for (auto it = pg_->workMetaList_.begin(); it != pg_->workMetaList_.end();
         /* no increment */) {
      auto& work = *it;
      // When terminateProcessGroup_ is true, communicators have already been
      // aborted, So cannot check exception based on them. But watchdog needs to
      // finish the check for the works that have already been enqueued to
      // workMetaList_

      // check NCCL errors first
      if (!pg_->terminateProcessGroup_.load()) {
        work.checkAndSetException();
      }

      if (work.exception()) {
        // set the error to the first error found
        std::lock_guard<std::mutex> lock(pg_->errorMutex_);
        if (pg_->error_ == ErrorType::SUCCESS) {
          pg_->error_ = ErrorType::COMM_ERROR;
        }
      }

      // Then check if work has timed out
      // Skip if work has encountered an error
      bool timedout = !work.exception() && work.checkTimeout();

      // Report desync state in case of timeout (if TORCH_NCCL_DESYNC_DEBUG is
      // turned on; otherwise, run() is no-op)
      if (timedout) {
        std::lock_guard<std::mutex> lock(pg_->errorMutex_);
        if (pg_->error_ == ErrorType::SUCCESS) {
          pg_->error_ = ErrorType::TIMEOUT;
        }
        desyncDebugger_.run();
      }

      // If work hits an exception (either an error or timeout)
      if (work.exception()) {
        LOG(ERROR) << c10::str(
            pg_->logPrefix(),
            " failure detected by watchdog at work sequence id: ",
            work.seq_,
            " PG status: last enqueued work: ",
            pg_->pgStatus_->lastEnqueuedSeq,
            ", last completed work: ",
            pg_->pgStatus_->lastCompletedSeq);

        // Print the traceback of the collective at call time
        work.printTraceback();

        // broadcast remote error signal to all other ranks in this specific PG.
        // key/signal to write in the tcpstore is a string and pg specific:
        // format is: remote_error:pg_uid
        if (propagatePgError_) {
          pg_->broadcastSignal(
              pg_->store_,
              std::string(kStoreErrorSignalKey) + ':' + pg_->pg_uid_,
              pg_->rank_);
        }

        // try to notify other ranks via global TCPStore to dump the flight
        // recorder when a collective timeout or exception happens. Flight
        // recorder behavior is independent of desync Debug.
        pg_->broadcastDumpSignal();
        // Give time for dumping before throwing exception for all ranks.
        // It is hard to presume or control what the pattern of watchdog might
        // look like, so it is better to let all ranks universally sleep for a
        // short period of time, in this case, 60 seconds, which is also the
        // maximum time we leave for FR dump.
        std::this_thread::sleep_for(std::chrono::milliseconds(
            pg_->heartbeatMonitor_->getDumpTimeout() * 4));

        if (SHOULD_CLEAN_UP(pg_->asyncErrorHandling_)) {
          // Abort work and corresponding communicators
          work.abort();
          // PG level abort, which would abort all other communicators on this
          // rank
          pg_->abortComms();
        }
        // Throw exception
        work.handleException(pg_->asyncErrorHandling_);
      }

      // Work status logging for desync debug
      desyncDebugger_.logWorkStart(work);

      // allow watchdog to do an event query on a side thread
      at::cuda::CUDAGuard device_guard(work.ncclEndEvent_->device_index());
      at::cuda::CUDAStreamCaptureModeGuard g{cudaStreamCaptureModeThreadLocal};

      // a work could be started but not completed, so we should not update
      // lastStartedSeq and lastStartedOpName if the work state is checked
      // multiple times after the start
      if (pg_->pgStatus_->lastStartedSeq < static_cast<int64_t>(work.seq_) &&
          work.isStarted()) {
        pg_->pgStatus_->lastStartedSeq = static_cast<int64_t>(work.seq_);
        pg_->pgStatus_->lastStartedWorkName = opTypeToString(work.opType_);
        pg_->pgStatus_->lastStartedNumelIn = work.numelIn_;
        pg_->pgStatus_->lastStartedNumelOut = work.numelOut_;
      }

      // Clean up completed work
      if (work.isCompleted()) {
        // In case user didn't call `work.wait()` with async collectives,
        // watchdog would unstage the stashed tensors when detecting completion
        // of the collective, to prevent ProcessGroupNCCL from holding reference
        // to those tensors forever.
        // work.stashed_for_allocator_safety_->unstash();
        // Update: it seems directly unstashing from watchdog thread would cause
        // some rare problems. We thus move the unstashing to main thread,
        // triggered by a next user call, see `workEnqueue`. But `work` is going
        // to be destructed, so we transfer the work's shelf to a shelves
        // structure owned by the PG.
        if (!work.stashed_for_allocator_safety_->empty()) {
          std::lock_guard<std::mutex> lock(pg_->shelvesMutex_);
          // We are just pushing back a shared_ptr here, so the cost should be
          // minimal
          pg_->shelvesToUnstash_.push_back(work.stashed_for_allocator_safety_);
        }

        if (pg_->enableTiming_ && logger) {
          ::c10d::C10dLoggingData data;
          // logging integers
          data.strings["collective_duration"] =
              std::to_string(work.getDuration());
          data.integers["global_rank"] = pg_->globalRank();
          data.integers["pg_id"] = static_cast<int64_t>(pg_->local_id_);
          data.strings["pg_name"] = pg_->pg_uid_;
          data.strings["pg_desc"] = pg_->pg_desc_;
          data.integers["pg_rank"] = pg_->rank_;
          data.integers["world_size"] = pg_->size_;
          data.strings["comm_backend"] = "nccl";
          data.strings["comm_backend_version"] = getNcclVersion();
          // TODO: We see errors for this line, revert it for now.
          data.strings["collective_stack"] = "";
          data.strings["collective_name"] = opTypeToString(work.opType_);
          logger->log(data);
        }

        // Work status logging for desync debug
        desyncDebugger_.logWorkEnd(work);

        if (work.futureWorkResult_ && work.finishedGPUExecutionInternal() &&
            !work.futureWorkResult_->completed()) {
          work.futureWorkResult_->markCompleted(
              at::IValue(static_cast<uint8_t>(WorkResult::SUCCESS)));
        }
        {
          // Reset the timeout and first work if the work is completed.
          std::lock_guard<std::mutex> timeoutLock(pg_->mtxTimeoutExtension_);
          if (work.ownedEphermeralTimeout_.count() > 0) {
            pg_->ephemeralTimeoutActive_ -= work.ownedEphermeralTimeout_;
            pg_->ephemeralTimeoutInflight_ -= work.ownedEphermeralTimeout_;
          }
        }
        pg_->pgStatus_->lastCompletedSeq = static_cast<int64_t>(work.seq_);
        pg_->pgStatus_->lastCompletedWorkName = opTypeToString(work.opType_);
        pg_->pgStatus_->lastCompletedNumelIn = work.numelIn_;
        pg_->pgStatus_->lastCompletedNumelOut = work.numelOut_;
        FlightRecorderCUDA::get()->retire_id(
            work.trace_id_, work.trace_reset_epoch_, true);
        if (pg_->onCompletionHook_) {
          // Move Work object to completedWorkList_ to be consumed by the hook
          // thread
          {
            const std::lock_guard<std::mutex> lock(
                pg_->completedWorkListMutex_);
            pg_->completedWorkList_.splice(
                pg_->completedWorkList_.end(), pg_->workMetaList_, it++);
          }
          pg_->completedWorkListCV_.notify_one();
        } else {
          it = pg_->workMetaList_.erase(it);
          pg_->heartbeatMonitor_->setLastWorkListUpdateTime(
              std::chrono::steady_clock::now());
        }
      } else {
        // Increment the iterator if the current WorkNCCL object is not
        // completed.
        ++it;
      }
      // Increment heartbeat after each work processed,
      // in case processing is slowed down (but not hung) by cuda api contention
      heartbeat_++;
    }
    done = pg_->workMetaList_.empty();
  }
}

uint64_t ProcessGroupNCCL::Watchdog::getHeartbt() const {
  return heartbeat_.load();
}

void ProcessGroupNCCL::Watchdog::setDesyncDebug(bool desyncDebug) {
  desyncDebug_ = desyncDebug;
}

// Initialize and enable DesyncDebugger
void ProcessGroupNCCL::DesyncDebugger::init(
    int rank,
    int size,
    int globalRank,
    int pgId,
    c10::intrusive_ptr<Store> store) {
  rank_ = rank;
  size_ = size;
  globalRank_ = globalRank;
  pgId_ = pgId;
  store_ = std::move(store);
  enabled_ = true;
  traceKeyStart_ = getTraceStartKey("NCCL", rank);
  traceKeyEnd_ = getTraceEndKey("NCCL", rank);
}

// Run desync debug. This function is called by watchdog at time of timeout.
void ProcessGroupNCCL::DesyncDebugger::run() {
  if (!enabled_)
    return;
  auto logPrefix = c10::str("Rank ", rank_);
  ::c10d::C10dLoggingData log;
  log.integers["pg_id"] = pgId_;
  log.integers["rank"] = rank_;
  log.integers["global_rank"] = globalRank_;
  log.integers["world_size"] = size_;
  // Use this to differentiate between flight recorder and desync debug report.
  log.strings["flight_recorder_version"] = "-1";

  try {
    std::string desyncMsg = retrieveDesyncReport(store_, "NCCL", rank_, size_);
    log.strings["status"] = "SUCCESS";
    LOG(ERROR) << logPrefix << desyncMsg;
  } catch (const std::exception& e) {
    log.strings["status"] = "EXCEPTION";
    log.strings["exception_msg"] = e.what();
    enabled_ = false;
    LOG(ERROR) << logPrefix
               << " Failed to retrieve TORCH_NCCL_DESYNC_DEBUG report. "
               << " Please file an issue. Error: " << e.what();
  } catch (...) {
    enabled_ = false;
    log.strings["status"] = "EXCEPTION";
    log.strings["exception_msg"] = "Unknown exception";
    LOG(ERROR)
        << logPrefix
        << " Failed to rerieve TORCH_NCCL_DESYNC_DEBUG report with unknown error."
        << " Please file an issue.";
  }
  auto logger = c10d::C10dLogger::getLogger();
  if (logger) {
    logger->log(log);
  }
}

// Log work start to store.
void ProcessGroupNCCL::DesyncDebugger::logWorkStart(WorkNCCL& work) {
  if (!enabled_)
    return;
  if (work.startTraceUpdated_)
    return;

  work.startTraceUpdated_ = true;
  // If not successful, disable the debugger
  enabled_ = c10d::traceUpdate(
      store_, traceKeyStart_, work.seq_, opTypeToString(work.opType_));
}

// Log work end to store.
void ProcessGroupNCCL::DesyncDebugger::logWorkEnd(WorkNCCL& work) {
  if (!enabled_)
    return;

  // In case the start of the work hasn't been logged
  if (!work.startTraceUpdated_) {
    logWorkStart(work);
  }

  // If not successful, disable the debugger
  enabled_ = c10d::traceUpdate(
      store_, traceKeyEnd_, work.seq_, opTypeToString(work.opType_));
}

// We want to have both PG ID and global unique ID (guid) for the logging
// prefix. PG ID records how many ProcessGroupNCCL objects were created on a
// specific rank and is a stable index across ranks, which lets users reason
// about, for example, the second PG we initialized on this rank is for FSDP,
// and corresponds with PG ID = 1 on other ranks as well. Unlike PG ID, guid (or
// group name) is a global unique ID across ranks. The guid is either a hash of
// all the ranks in the group or a counter of how many times
// `_process_group_name` is called, essentially it means how many times we
// have PGs users have created. Before using split_group, even if
// we are creating a new sub-PG, all ranks have to call the API at the same
// time, and this makes `group_name` a unique identifier for a group (PG).
std::string ProcessGroupNCCL::createLogPrefix() const {
  if (!pg_desc_.empty() && pg_desc_ != "undefined") {
    return c10::str(
        "[PG ID ",
        local_id_,
        " PG GUID ",
        pg_uid_,
        "(",
        pg_desc_,
        ") Rank ",
        rank_,
        "] ");
  }
  return c10::str(
      "[PG ID ", local_id_, " PG GUID ", pg_uid_, " Rank ", rank_, "] ");
}

const std::string& ProcessGroupNCCL::logPrefix() const {
  return logPrefix_;
}

const int& ProcessGroupNCCL::globalRank() const {
  static int globalRank = rank_;
  return globalRank;
}

const c10::intrusive_ptr<Store>& ProcessGroupNCCL::globalStore() const {
  return globalStore_;
}

const std::vector<uint64_t>& ProcessGroupNCCL::groupRanks() const {
  if (options_->global_ranks_in_group.empty() && local_id_ == 0) {
    static std::vector<uint64_t> globalRanks(size_);
    std::iota(globalRanks.begin(), globalRanks.end(), 0);
    return globalRanks;
  }
  return options_->global_ranks_in_group;
}

void ProcessGroupNCCL::addEphemeralTimeout(
    const std::chrono::milliseconds& timeout) {
  std::lock_guard<std::mutex> timeoutLock(mtxTimeoutExtension_);
  ephemeralTimeoutActive_ += timeout;
}

bool ProcessGroupNCCL::verifyWorkTimeoutForTest(
    const c10::intrusive_ptr<Work>& work,
    const std::chrono::milliseconds& timeout) {
  // Since collective returns a c10d::Work, we need to cast it to WorkNCCL.
  if (auto workNCCL = c10::dynamic_intrusive_pointer_cast<WorkNCCL>(work)) {
    // workNCCL is now a c10::intrusive_ptr<WorkNCCL>
    return workNCCL->opTimeout_ == timeout;
  }
  C10_THROW_ERROR(
      DistBackendError, "Non c10d::WorkNCCL object returned from collective");
}

void ProcessGroupNCCL::broadcastSignal(
    c10::intrusive_ptr<Store>& store,
    const std::string& signal,
    int srcRank) {
  try {
    auto vec = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(&srcRank),
        reinterpret_cast<uint8_t*>(&srcRank) + sizeof(srcRank));
    store->set(signal, vec);
    LOG(INFO) << logPrefix() << "Broadcasting 

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough

### Classes
This file defines 1 class(es): var


## Key Components

The file contains 22413 words across 5987 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 223458 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
