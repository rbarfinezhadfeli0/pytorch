# Documentation: `torch/csrc/distributed/c10d/ProcessGroupGloo.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/ProcessGroupGloo.cpp`
- **Size**: 87,701 bytes (85.65 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/Exception.h>
#include <c10/util/error.h>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>

#ifdef USE_C10D_GLOO

#include <torch/csrc/distributed/c10d/FlightRecorder.hpp>
#include <torch/csrc/distributed/c10d/GlooDeviceFactory.hpp>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGlooDetail.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <chrono>
#include <exception>

#ifdef _WIN32
#include <gloo/common/win.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#endif
#include <sys/types.h>

#include <type_traits>
#include <utility>

#include <ATen/ThreadLocalState.h>
#include <ATen/native/SparseTensorUtils.h>

#include <c10/util/StringUtil.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <gloo/config.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/prefix_store.h>

namespace c10d {

namespace {

using steady_clock_time_point =
    std::chrono::time_point<std::chrono::steady_clock>;

std::chrono::milliseconds getRemainingTime(
    steady_clock_time_point startTime,
    const std::chrono::milliseconds& timeout,
    bool waitAllRanks) {
  if (waitAllRanks) {
    // See Note in monitoredBarrier
    return timeout;
  }
  auto elapsedTime = std::chrono::steady_clock::now() - startTime;
  auto remainingMillis = timeout -
      std::chrono::duration_cast<std::chrono::milliseconds>(elapsedTime);

  // If no more remaining time, return -1 to indicate to caller.
  if (remainingMillis.count() <= 0) {
    return std::chrono::milliseconds(-1);
  }

  return remainingMillis;
}

// Emit a LOG(ERROR) and throws using TORCH_CHECK with the given messages.
void logAndThrow(
    const std::string& logMessage,
    const std::string& errorMessage) {
  LOG(ERROR) << logMessage;
  TORCH_CHECK(false, errorMessage);
}

// For monitoredBarrier, checks remaining time left to finish processing ranks
// and throws error if timeout.
void checkRemainingTime(
    const std::chrono::milliseconds& monitoredBarrierTimeout,
    const std::chrono::milliseconds& remainingTime,
    const std::vector<int>& processedRanks,
    int currentRank) {
  const std::string kNoRemainingTimeError = c10::str(
      "Rank ",
      currentRank,
      " timed out in monitoredBarrier after ",
      monitoredBarrierTimeout.count(),
      " ms.");
  if (remainingTime.count() < 0) {
    std::string rankInfo;
    if (!processedRanks.empty()) {
      rankInfo = c10::str(
          "Successfully processed ranks: ", c10::Join(", ", processedRanks));
    } else {
      rankInfo = "No ranks successfully processed in monitoredBarrier.";
    }
    auto error = c10::str(kNoRemainingTimeError, "\n", rankInfo);
    logAndThrow(error, error);
  }
}

const auto kLoopbackAddress = "127.0.0.1";

} // namespace

// This function initializes a vector of CUDA streams, one for every
// tensor in the input tensor vector, and ensures that these streams are
// synchronized with the current default streams. This is needed so
// that new work on the new streams is serialized w.r.t. all operations
// on the tensors.
void initializeStreamsEvents(
    const std::vector<at::Tensor>& tensors,
    std::vector<c10::Stream>& streams,
    std::vector<c10::Event>& events) {
  streams.reserve(tensors.size());
  events.reserve(tensors.size());
  for (const auto i : c10::irange(tensors.size())) {
    c10::Device device = tensors[i].device();
    c10::impl::VirtualGuardImpl impl(device.type());
    // Record event on current stream
    events.emplace_back(device.type());
    events[i].record(impl.getStream(device));
    // Get a non-default stream to execute asynchronous CUDA operations
    // on this device. This ensures that the default stream used
    // by the caller is not occupied by c10d related operations.
    streams.push_back(
        impl.getStreamFromGlobalPool(device, /*isHighPriority=*/true));
    // Ensure the new stream is synchronized with the current stream.
    events[i].block(streams[i]);

    // `tensors` are created on a different stream. Hence, they must record
    // new streams in this Work to prevent being freed before the Work finishes.
    if (tensors[i].is_sparse()) {
      if (tensors[i].is_coalesced()) {
        impl.recordDataPtrOnStream(
            tensors[i].indices().storage().data_ptr(), streams[i]);
        impl.recordDataPtrOnStream(
            tensors[i].values().storage().data_ptr(), streams[i]);
      } else {
        // We will need to coalesce first, which means new tensors will
        // be allocated on the streams we just allocated, and there
        // is no need to record them separately.
      }
    } else {
      impl.recordDataPtrOnStream(tensors[i].storage().data_ptr(), streams[i]);
    }
  }
}

// This function initializes a vector of CUDA streams, one per device,
// and ensures that these streams are synchronized with the current default
// streams. It is assumed that the tensors in the nested tensor vectors are
// on the same device.
void initializeStreamsEvents(
    std::vector<std::vector<at::Tensor>>& tensors,
    std::vector<c10::Stream>& streams,
    std::vector<c10::Event>& events) {
  // Ensure that the tensors in the nested tensor vectors are on the same
  // device.
  for (const auto& tensorgroup : tensors) {
    const auto device_id = tensorgroup[0].device().index();
    for (const auto& tensor : tensorgroup) {
      if (tensor.device().index() != device_id) {
        TORCH_CHECK(
            false,
            "tensors in the nested tensor vectors need to "
            "be on the same device");
      }
    }
  }

  streams.reserve(tensors.size());
  events.reserve(tensors.size());
  for (const auto i : c10::irange(tensors.size())) {
    c10::Device device = tensors[i][0].device();
    c10::impl::VirtualGuardImpl impl(device.type());
    // Record event on current stream
    events.emplace_back(device.type());
    events[i].record(impl.getStream(device));
    // Get a non-default stream to execute asynchronous CUDA operations
    // on for this output. This ensures that the default stream used
    // by the caller is not occupied by c10d related operations.
    streams.push_back(
        impl.getStreamFromGlobalPool(device, /*isHighPriority=*/true));
    // Ensure the new stream is synchronized with the current stream.
    events[i].block(streams[i]);

    for (at::Tensor& tensor : tensors[i]) {
      // `tensors` are created on a different stream. Hence, they must record
      // new streams in this Work to prevent being freed before the Work
      // finishes.
      impl.recordDataPtrOnStream(tensor.storage().data_ptr(), streams[i]);
    }
  }
}

bool getDefaultGlooLazyInit() {
  return ::c10d::getCvarBool(TORCH_GLOO_LAZY_INIT, false);
}

// static
void ProcessGroupGloo::AsyncWork::execute(
    const c10::intrusive_ptr<AsyncWork>& work) {
  if (work->recordFunctionBeforeCallback_) {
    work->recordFunctionBeforeCallback_();
  }
  try {
    at::ThreadLocalStateGuard g(work->getTLS());
    work->run();
  } catch (...) {
    work->finishWorkGlooError(std::current_exception());
    return;
  }

  // FIXME: We need to call it here since Future completion requires all
  // the work to be synchronized to CUDA.
  work->synchronize();
  work->finishWorkGloo();
}

std::vector<at::Tensor> ProcessGroupGloo::AsyncWork::result() {
  TORCH_CHECK(
      isCompleted(),
      "Work needs to be completed before calling result(). "
      "Should call wait() before result().");
  TORCH_CHECK(
      outputTensors_.size() <= 1,
      "work result does not support list of lists, use .getFuture() and value()");
  return outputTensors_.empty() ? std::vector<at::Tensor>()
                                : outputTensors_.at(0);
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupGloo::AsyncWork::
    getFuture() {
  return future_;
}

std::chrono::milliseconds ProcessGroupGloo::AsyncWork::getTimeout() const {
  return context_->getTimeout();
}

namespace {
c10::intrusive_ptr<c10::ivalue::Future> createFutureAsOutput(
    const std::vector<std::vector<at::Tensor>>& outputTensors) {
  if (outputTensors.size() > 1) {
    return c10::make_intrusive<c10::ivalue::Future>(
        c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  }
  return c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
}

void returnFutureWithOutput(
    c10::intrusive_ptr<c10::ivalue::Future>& future,
    const std::vector<std::vector<at::Tensor>>& outputTensors) {
  if (outputTensors.empty()) {
    future->markCompleted(c10::IValue(std::vector<at::Tensor>()));
    return;
  }
  if (outputTensors.size() > 1) {
    future->markCompleted(c10::IValue(outputTensors));
    return;
  }
  future->markCompleted(c10::IValue(outputTensors[0]));
}
} // namespace

inline void ProcessGroupGloo::AsyncWork::recordAsyncWorkProfilingInfo(
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors) {
  auto recordingFunction =
      std::make_shared<at::RecordFunction>(at::RecordScope::USER_SCOPE);
  if (recordingFunction->isActive()) {
    std::function<void()> before_handler =
        [inputTensors, profilingTitle, recordingFunction]() {
          // The work will be started and completed by different threads.
          recordingFunction->_setAsync();
          std::vector<c10::IValue> inputs;
          if (inputTensors) {
            inputs.reserve(inputTensors->size());
            for (const auto& tensor : *inputTensors) {
              inputs.emplace_back(tensor);
            }
          }
          recordingFunction->before(
              profilingTitle,
              c10::ArrayRef<const c10::IValue>(inputs.data(), inputs.size()));
        };
    recordFunctionBeforeCallback_ =
        at::wrapPropagateTLSState(std::move(before_handler));
    std::function<void()> end_handler = [recordingFunction]() {
      recordingFunction->end();
    };
    recordFunctionEndCallback_ = at::wrapPropagateTLSState(end_handler);
  }
}

ProcessGroupGloo::AsyncWork::AsyncWork(
    std::shared_ptr<gloo::Context> context,
    std::vector<std::vector<at::Tensor>> outputTensors,
    OpType opType,
    uint64_t seq,
    std::chrono::milliseconds timeout,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors)
    // Profiler: Pass nullptr as profilingTitle to parent constructor to
    // replace default profiler implementation with async version that reports
    // correct timestamps for work that is asynchronously executed.
    : Work(-1, opType, nullptr, inputTensors),
      context_(std::move(context)),
      timeout_(timeout == kUnsetTimeout ? context_->getTimeout() : timeout),
      outputTensors_(std::move(outputTensors)),
      future_(createFutureAsOutput(outputTensors_)),
      seq_(seq) {
  if (profilingTitle != nullptr) {
    recordAsyncWorkProfilingInfo(profilingTitle, inputTensors);
    profilingTitle_ = profilingTitle;
  }
}

uint64_t ProcessGroupGloo::AsyncWork::getSequencenumber() const {
  return seq_;
}

void ProcessGroupGloo::AsyncWork::finishWorkGlooError(
    const std::exception_ptr& eptr) {
  future_->setError(eptr);
  finish(eptr);
}

void ProcessGroupGloo::AsyncWork::finishWorkGloo() {
  returnFutureWithOutput(future_, outputTensors_);
  finish();
}

ProcessGroupGloo::SendWork::SendWork(
    at::Tensor& tensor,
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer,
    uint64_t seq)
    : Work(
          -1,
          OpType::SEND,
          "gloo:send",
          std::optional<std::vector<at::Tensor>>({tensor})),
      tensor_(tensor),
      buffer_(std::move(buffer)),
      seq_(seq) {}

uint64_t ProcessGroupGloo::SendWork::getSequencenumber() const {
  return seq_;
}

bool ProcessGroupGloo::SendWork::wait(std::chrono::milliseconds timeout) {
  bool sendCompleted = false;
  std::exception_ptr exception{nullptr};
  try {
    if (timeout == kNoTimeout) {
      sendCompleted = buffer_->waitSend();
    } else {
      sendCompleted = buffer_->waitSend(timeout);
    }
  } catch (...) {
    exception = std::current_exception();
  }

  // Completes the Work object and throws the exception.
  finishAndThrow(exception);
  if (c10d::allow_inflight_collective_as_graph_input()) {
    c10d::unregister_work(
        c10::intrusive_ptr<
            ProcessGroupGloo::SendWork>::unsafe_reclaim_from_nonowning(this));
  }
  return sendCompleted;
}

void ProcessGroupGloo::SendWork::abort() {
  buffer_->abortWaitSend();
}

ProcessGroupGloo::RecvWork::RecvWork(
    at::Tensor& tensor,
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer,
    OpType opType,
    uint64_t seq,
    const char* profilingTitle)
    : Work(
          -1,
          opType,
          profilingTitle,
          std::optional<std::vector<at::Tensor>>({tensor})),
      tensor_(tensor),
      buffer_(std::move(buffer)),
      srcRank_(-1),
      seq_(seq) {}

uint64_t ProcessGroupGloo::RecvWork::getSequencenumber() const {
  return seq_;
}

int ProcessGroupGloo::RecvWork::sourceRank() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return srcRank_;
}

bool ProcessGroupGloo::RecvWork::wait(std::chrono::milliseconds timeout) {
  bool recvCompleted = false;
  std::exception_ptr exception{nullptr};
  try {
    if (timeout == kNoTimeout) {
      recvCompleted = buffer_->waitRecv(&srcRank_);
    } else {
      recvCompleted = buffer_->waitRecv(&srcRank_, timeout);
    }
  } catch (...) {
    exception = std::current_exception();
  }

  // Completes the Work object and throws the exception.
  finishAndThrow(exception);
  if (c10d::allow_inflight_collective_as_graph_input()) {
    c10d::unregister_work(
        c10::intrusive_ptr<
            ProcessGroupGloo::RecvWork>::unsafe_reclaim_from_nonowning(this));
  }
  return recvCompleted;
}

void ProcessGroupGloo::RecvWork::abort() {
  buffer_->abortWaitRecv();
}

ProcessGroupGloo::Options::Options(std::chrono::milliseconds timeout)
    : Backend::Options(GLOO_BACKEND_NAME, timeout), threads(2) {}

namespace {

void socketInitialize() {
#ifdef _WIN32
  ::gloo::init_winsock();
#endif
}

// Gloo assumes that this machine's hostname can always be resolved
// to an address. If it doesn't it throws a runtime error saying
// that it can't be resolved. Instead of catching it, we choose
// to proactively check if an address can be resolved, so we can
// gracefully fall back to an alternative if it doesn't.
bool doesHostnameResolveToUsableAddress(const std::string& hostname) {
  socketInitialize();
  struct addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  struct addrinfo* result = nullptr;
  auto rv = getaddrinfo(hostname.c_str(), nullptr, &hints, &result);
  if (rv < 0) {
    return false;
  }
  struct addrinfo* rp = nullptr;
  for (rp = result; rp != nullptr; rp = rp->ai_next) {
    auto fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    if (fd == -1) {
      continue;
    }
    rv = bind(fd, rp->ai_addr, rp->ai_addrlen);
#ifdef _WIN32
    closesocket(fd);
#else
    close(fd);
#endif
    if (rv == -1) {
      continue;
    }
    break;
  }
  freeaddrinfo(result);
  return rp != nullptr;
}

} // namespace

std::shared_ptr<::gloo::transport::Device> ProcessGroupGloo::
    createDeviceForInterface(const std::string& interface_name, bool lazyInit) {
  return ::c10d::GlooDeviceFactory::makeDeviceForInterface(
      interface_name, lazyInit);
}

std::shared_ptr<::gloo::transport::Device> ProcessGroupGloo::
    createDeviceForHostname(const std::string& hostname, bool lazyInit) {
  TORCH_CHECK(
      doesHostnameResolveToUsableAddress(hostname),
      "Cannot resolve ",
      hostname,
      " to a (local) address");
  return ::c10d::GlooDeviceFactory::makeDeviceForHostname(hostname, lazyInit);
}

#if defined(__linux__) || defined(_WIN32)
std::shared_ptr<::gloo::transport::Device> ProcessGroupGloo::
    createDefaultDevice(bool lazyInit) {
  // Use the hostname to resolve the network address to
  // use. Note: if the hostname does not resolve to an address (e.g.
  // because of misconfigured /etc/hosts file), this will not work.
  socketInitialize();
  std::array<char, HOST_NAME_MAX> hostname{};
  auto rv = gethostname(hostname.data(), HOST_NAME_MAX);
  if (rv != 0) {
    C10_THROW_ERROR(DistBackendError, c10::utils::str_error(errno));
  }

  // Use this machine's hostname if it resolves to an address.
  if (doesHostnameResolveToUsableAddress(hostname.data())) {
    return ::c10d::GlooDeviceFactory::makeDeviceForHostname(
        hostname.data(), lazyInit);
  }

  // Otherwise, use the loopback address.
  TORCH_WARN_ONCE(
      "Unable to resolve hostname to a (local) address. ",
      "Using the loopback address as fallback. ",
      "Manually set the network interface to bind to with GLOO_SOCKET_IFNAME.");
  return createDeviceForHostname(kLoopbackAddress, lazyInit);
}
#endif

#ifdef __APPLE__
std::shared_ptr<::gloo::transport::Device> ProcessGroupGloo::
    createDefaultDevice(bool lazyInit) {
  // Use the hostname to resolve the network address to
  // use. Note: if the hostname does not resolve to an address (e.g.
  // because of misconfigured /etc/hosts file), this will not work.
  const auto hostNameMax = sysconf(_SC_HOST_NAME_MAX);
  std::string hostname(hostNameMax, '\0');
  auto rv = gethostname(hostname.data(), hostNameMax);
  if (rv != 0) {
    C10_THROW_ERROR(DistBackendError, c10::utils::str_error(errno));
  }

  // Use this machine's hostname if it resolves to an address.
  if (doesHostnameResolveToUsableAddress(hostname.data())) {
    return ::c10d::GlooDeviceFactory::makeDeviceForHostname(
        hostname.data(), lazyInit);
  }

  // Otherwise, use the loopback address.
  TORCH_WARN_ONCE(
      "Unable to resolve hostname to a (local) address. ",
      "Using the loopback address as fallback. ",
      "Manually set the network interface to bind to with GLOO_SOCKET_IFNAME.");
  return createDeviceForHostname(kLoopbackAddress, lazyInit);
}
#endif

static std::atomic<size_t> process_group_id = 0;

c10::intrusive_ptr<ProcessGroupGloo::Options> ProcessGroupGloo::Options::
    create_default(std::chrono::milliseconds timeout) {
  auto options = ::c10d::ProcessGroupGloo::Options::create();
  bool lazyInit = ::c10d::getDefaultGlooLazyInit();

  // Use interfaces listed in "GLOO_SOCKET_IFNAME", if set.
  auto ifnameEnv = c10::utils::get_env("GLOO_SOCKET_IFNAME");
  if (ifnameEnv && ifnameEnv->size() > 1) {
    for (const auto& iface : ::c10d::split(',', *ifnameEnv)) {
      options->devices.push_back(
          ::c10d::ProcessGroupGloo::createDeviceForInterface(iface, lazyInit));
    }
  } else {
    // If no hostname is specified, this function looks up
    // the machine's hostname and returns a device instance
    // associated with the address that the hostname resolves to.
    options->devices.push_back(
        ::c10d::ProcessGroupGloo::createDefaultDevice(lazyInit));
  }

  options->timeout = timeout;
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  options->threads = options->devices.size() * 2;
  return options;
}

ProcessGroupGloo::ProcessGroupGloo(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      store_(new GlooStore(store)),
      options_(std::move(options)),
      stop_(false),
      collectiveCounter_(0),
      local_id_(process_group_id++) {
  auto& devices = options_->devices;
  if (devices.empty()) {
    TORCH_CHECK(false, "No device(s) specified");
  }

  // Create and connect a context for every device.
  //
  // Note that the same device can be specified multiple times, either
  // the same object, or the same logical device as different objects.
  // Either mode is fine and only has performance implications.
  //
  // Using the same object multiple times means all contexts share a
  // single I/O thread. If you use different objects for the same
  // logical device they will have independent I/O threads. The latter
  // option is needed if you have a fast NIC that cannot be saturated
  // by a single I/O thread.
  //
  contexts_.reserve(options_->devices.size());
  for (const auto i : c10::irange(options_->devices.size())) {
    auto context = std::make_shared<::gloo::rendezvous::Context>(rank_, size_);

#ifdef GLOO_SHARED_STORE
    auto underlyingStore = store_;
#else
    auto& underlyingStore = *store_;
#endif

    auto store = std::make_shared<::gloo::rendezvous::PrefixStore>(
        std::to_string(i), underlyingStore);

#ifdef GLOO_SHARED_STORE
    auto connectStore = store;
#else
    auto& connectStore = *store;
#endif

    context->setTimeout(options_->timeout);
    try {
      context->connectFullMesh(connectStore, options_->devices[i]);
    } catch (const std::runtime_error& e) {
      auto err = e.what();
      // TORCH_CHECK to print the cpp stacktrace.
      auto msg = c10::str("Gloo connectFullMesh failed with ", err);
      logAndThrow(msg, msg);
    }
    contexts_.push_back(std::move(context));
  }

  // Every worker thread stores the AsyncWork object it's currently
  // working on in the workInProgress_ vector. It must have size equal
  // to the number of workers such that they can simply index into it
  // using the worker index they are started with.
  workInProgress_.resize(options_->threads);

  threads_.resize(options_->threads);
  for (const auto i : c10::irange(threads_.size())) {
    threads_[i] = std::thread(&ProcessGroupGloo::runLoop, this, i);
  }
  this->setGroupUid(options_->group_name);

  // TODO: If gloo has version, we also need to log gloo version into FR.
  FlightRecorder<c10::Event>::get()->record_pg_ranks(
      std::make_tuple(pg_uid_, pg_desc_), groupRanks());
  init();

  // TODO: Add configs print like ProcessGroupNCCL.
}

ProcessGroupGloo::~ProcessGroupGloo() {
  std::unique_lock<std::mutex> lock(workMutex_);
  workConsumeCV_.wait(lock, [&] { return workQueue_.empty(); });

  // Queue is empty, signal stop
  stop_ = true;

  // Release lock to allow threads to terminate
  lock.unlock();

  workProduceCV_.notify_all();

  // Wait for worker threads to terminate
  for (auto& thread : threads_) {
    thread.join();
  }
}

uint32_t ProcessGroupGloo::nextTag() {
  return collectiveCounter_++;
}

std::shared_ptr<::gloo::Context> ProcessGroupGloo::getContext(uint32_t tag) {
  return contexts_[tag % contexts_.size()];
}

void ProcessGroupGloo::runLoop(int workerIndex) {
  std::unique_lock<std::mutex> lock(workMutex_);

  while (!stop_) {
    if (workQueue_.empty()) {
      workProduceCV_.wait(lock);
      continue;
    }

    auto work = std::move(workQueue_.front());
    workQueue_.pop_front();
    workInProgress_[workerIndex] = work;
    lock.unlock();

    // Notify after releasing the lock so that the waiter
    // does not immediately block.
    workConsumeCV_.notify_one();

    AsyncWork::execute(work);
    // TODO: Need to find a way to calculate the difference of duration of two
    // c10d::Event
    pgStatus_->lastCompletedSeq = static_cast<int64_t>(work->seq_);
    pgStatus_->lastCompletedWorkName = opTypeToString(work->opType_);
    // TODO: We need to have numel of tensors for gloo as well.
    pgStatus_->lastCompletedNumelIn = 0;
    pgStatus_->lastCompletedNumelOut = 0;
    FlightRecorder<c10::Event>::get()->retire_id(
        work->trace_id_, work->trace_reset_epoch_, false);
    lock.lock();
    workInProgress_[workerIndex].reset();
  }
}

const std::vector<uint64_t>& ProcessGroupGloo::groupRanks() const {
  if (options_->global_ranks_in_group.empty() && local_id_ == 0) {
    static std::vector<uint64_t> globalRanks(size_);
    std::iota(globalRanks.begin(), globalRanks.end(), 0);
    return globalRanks;
  }
  return options_->global_ranks_in_group;
}

c10::intrusive_ptr<Backend> ProcessGroupGloo::split(
    const c10::intrusive_ptr<Store>& store,
    const std::vector<int>& ranks,
    const c10::intrusive_ptr<Backend::Options>& opts) {
  auto it = std::find(ranks.begin(), ranks.end(), rank_);
  int groupRank;
  if (it == ranks.end()) {
    return nullptr;
  } else {
    groupRank = std::distance(ranks.begin(), it);
  }

  auto glooOpts = c10::dynamic_intrusive_pointer_cast<Options>(opts);
  if (glooOpts == nullptr) {
    TORCH_WARN_ONCE(
        "Tried to pass options to ProcessGroupGloo::split that are not ProcessGroupGloo::Options."
        "Falling back to default options.");
    glooOpts = ProcessGroupGloo::Options::create_default();
  }

  // TODO: we need to get rid of globalRanksInGroup eventually.
  std::vector<uint64_t> globalRanksInGroup;
  for (auto rank : ranks) {
    globalRanksInGroup.emplace_back(groupRanks()[rank]);
  }
  glooOpts->global_ranks_in_group = std::move(globalRanksInGroup);
  auto pg = c10::make_intrusive<ProcessGroupGloo>(
      store->clone(), groupRank, ranks.size(), glooOpts);
  return c10::static_intrusive_pointer_cast<Backend>(pg);
}

c10::intrusive_ptr<Backend> ProcessGroupGloo::merge(
    const c10::intrusive_ptr<Store>& store,
    const c10::intrusive_ptr<Backend::Options>& opts,
    const int& rank,
    const int& size) {
  auto glooOpts = c10::dynamic_intrusive_pointer_cast<Options>(opts);
  if (glooOpts == nullptr) {
    TORCH_WARN_ONCE(
        "Tried to pass options to ProcessGroupGloo::merge that are not ProcessGroupGloo::Options."
        "Falling back to default options.");
    glooOpts = ProcessGroupGloo::Options::create_default();
  }
  auto pg = c10::make_intrusive<ProcessGroupGloo>(
      store->clone(), rank, size, glooOpts);
  return c10::static_intrusive_pointer_cast<Backend>(pg);
}

void ProcessGroupGloo::enqueue(c10::intrusive_ptr<AsyncWork> work) {
  std::unique_lock<std::mutex> lock(workMutex_);
  pgStatus_->lastEnqueuedSeq = static_cast<int64_t>(work->seq_);
  pgStatus_->lastEnqueuedWorkName = opTypeToString(work->opType_);
  // TODO: We need to have numel of tensors for gloo as well.
  pgStatus_->lastEnqueuedNumelIn = 0;
  pgStatus_->lastEnqueuedNumelOut = 0;
  // using c10d::FlightRecorder;
  // TODO: We need to have a way to use c10::Event inside gloo as well.
  auto traceId = FlightRecorder<c10::Event>::get()->recordWithResetEnabled(
      local_id_,
      std::make_tuple(pg_uid_, pg_desc_),
      collectiveCounter_,
      0, // p2p_seq_id, set 0 for now since p2p does not call enqueue
      work->getSequencenumber(), // We need to differentiate between p2p and
                                 // non-p2p op.
      work->getProfilerTitle(),
      work->getInputTensors(),
      work->getOutputTensors(),
      nullptr,
      nullptr,
      work->getTimeout(),
      pgStatus_,
      false);
  work->trace_id_ = traceId.id;
  work->trace_reset_epoch_ = traceId.reset_epoch;
  workQueue_.push_back(std::move(work));
  lock.unlock();

  // Notify after releasing the lock so that the waiter
  // does not immediately block.
  workProduceCV_.notify_one();
}

namespace {

class AsyncBroadcastWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncBroadcastWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<at::Tensor>& inputs,
      int rootRank,
      int rootTensor,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : ProcessGroupGloo::AsyncWork(
            std::move(context),
            {inputs},
            OpType::BROADCAST,
            seq,
            timeout,
            "gloo:broadcast",
            inputs),
        inputs(inputs),
        rootRank(rootRank),
        rootTensor(rootTensor),
        tag(tag) {}

  std::vector<at::Tensor> inputs;
  const int rootRank;
  const int rootTensor;
  const uint32_t tag;

  void broadcast(at::Tensor tensor) {
    if (tensor.is_complex()) {
      tensor = at::view_as_real(tensor);
    }
    const auto& scalarType = tensor.scalar_type();
    gloo::BroadcastOptions opts(context_);
    opts.setRoot(rootRank);
    opts.setTag(tag);
    opts.setTimeout(timeout_);
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, tensor);
    gloo::broadcast(opts);
  }

  const std::vector<at::Tensor> getInputTensors() override {
    return inputs;
  }

  const std::vector<at::Tensor> getOutputTensors() override {
    return inputs;
  }

  void run() override {
    broadcast(inputs[rootTensor]);

    // Copy to non-root tensors
    for (const auto i : c10::irange(inputs.size())) {
      if (i == static_cast<size_t>(rootTensor)) {
        continue;
      }
      inputs[i].copy_(inputs[rootTensor]);
    }
  }
};

class AsyncBroadcastCUDAWork : public AsyncBroadcastWork {
 public:
  AsyncBroadcastCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      int rootRank,
      int rootTensor,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : AsyncBroadcastWork(
            context,
            inputs,
            rootRank,
            rootTensor,
            tag,
            seq,
            timeout) {
    initializeStreamsEvents(inputs, streams, events);

    // Create pinned host side tensors.
    tmp = pinnedLike(inputs[rootTensor]);
    c10::OptionalStreamGuard guard;
    if (context_->rank == rootRank) {
      guard.reset_stream(streams[rootTensor]);
      tmp.copy_(inputs[rootTensor], /* non_blocking */ true);
    }
  }

  void run() override {
    // Synchronize with copy operation if applicable.
    if (context_->rank == rootRank) {
      streams[rootTensor].synchronize();
    }

    // Run broadcast on host side tensors.
    broadcast(tmp);

    // Kick off copy back to the CUDA tensors.
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      inputs[i].copy_(tmp, /* non_blocking */ true);
      events[i].record(streams[i]);
    }
  }

  void synchronize() override {
    // Synchronize with the copy back to CUDA tensors.
    for (const auto i : c10::irange(inputs.size())) {
      c10::Device device = inputs[i].device();
      events[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  at::Tensor tmp;
  std::vector<c10::Stream> streams;
  std::vector<c10::Event> events;
};

} // namespace

c10::intrusive_ptr<Work> ProcessGroupGloo::broadcast(
    std::vector<at::Tensor>& inputs,
    const BroadcastOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupGloo::broadcast: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertRootTensor(
      invalidArgument, opts.rootTensor, static_cast<int64_t>(inputs.size()));
  assertDense(invalidArgument, inputs);
  assertTypeAndSizesMatch(invalidArgument, inputs);

  const auto& device = inputs[0].device();
  switch (device.type()) {
    case at::kCPU:
      break;
    case at::kCUDA:
      // If the user gave us a CUDA tensor then CUDA must be loaded.
      TORCH_INTERNAL_ASSERT(at::hasCUDA());
      break;
    default:
      invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  c10::intrusive_ptr<AsyncBroadcastWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  if (device.type() == at::kCPU) {
    work = c10::make_intrusive<AsyncBroadcastWork>(
        std::move(context),
        inputs,
        opts.rootRank,
        opts.rootTensor,
        tag,
        seq_,
        opts.timeout);
  } else if (device.type() == at::kCUDA) {
    work = c10::make_intrusive<AsyncBroadcastCUDAWork>(
        std::move(context),
        inputs,
        opts.rootRank,
        opts.rootTensor,
        tag,
        seq_,
        opts.timeout);
  } else {
    TORCH_CHECK(false, "Invalid backend");
  }

  enqueue(work);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupGloo::allreduce(
    std::vector<at::Tensor>& inputs,
    const AllreduceOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupGloo::allreduce: " + msg);
  };

  assertNonEmpty(invalidArgument, inputs);
  assertLayoutMatch(invalidArgument, inputs);
  assertTypeAndSizesMatch(invalidArgument, inputs);

  const auto& device = inputs[0].device();
  switch (device.type()) {
    case at::kCPU:
      break;
    case at::kCUDA:
      // If the user gave us a CUDA tensor then CUDA must be loaded.
      TORCH_INTERNAL_ASSERT(at::hasCUDA());
      break;
    default:
      invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  const auto& layout = inputs[0].layout();
  if (layout == c10::kSparse && opts.reduceOp != ReduceOp::SUM) {
    invalidArgument(
        "unsupported reduction operation "
        "(allreduce of sparse tensors only works with ReduceOp.SUM)");
  }

  c10::intrusive_ptr<AsyncWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;

  work = GlooAllreduceRegistry()->Create(
      device.type(), context, inputs, opts.reduceOp, tag, seq_, opts.timeout);

  enqueue(work);
  return work;
}

static c10::intrusive_ptr<ProcessGroupGloo::AsyncWork> makeAllreduceCPUWork(
    std::shared_ptr<gloo::Context> context,
    std::vector<at::Tensor>& inputs,
    ReduceOp reduceOp,
    uint32_t tag,
    uint64_t seq,
    std::chrono::milliseconds timeout) {
  auto layout = inputs[0].layout();

  if (layout == c10::kStrided) {
    return c10::make_intrusive<AsyncAllreduceWork>(
        std::move(context), inputs, reduceOp, tag, seq, timeout);
  } else if (layout == c10::kSparse) {
    return c10::make_intrusive<AsyncSparseAllreduceWork>(
        std::move(context), inputs, tag, seq, timeout);
  } else {
    TORCH_CHECK(false, "ProcessGroupGloo::allreduce: unsupported layout");
  }
}

C10_DEFINE_TYPED_REGISTRY(
    GlooAllreduceRegistry,
    c10::DeviceType,
    ProcessGroupGloo::AsyncWork,
    c10::intrusive_ptr,
    std::shared_ptr<gloo::Context>,
    std::vector<at::Tensor>&,
    ReduceOp,
    uint32_t,
    uint64_t,
    std::chrono::milliseconds)

C10_REGISTER_TYPED_CREATOR(
    GlooAllreduceRegistry,
    at::kCPU,
    makeAllreduceCPUWork)

c10::intrusive_ptr<Work> ProcessGroupGloo::allreduce_sparse(
    std::vector<at::Tensor>& inputs,
    const AllreduceOptions& opts) {
  // all reduce sparse calls into default allreduce which
  // implemented with all_gathering indices and values
  // we do this we do not have a native cuda implementation
  return allreduce(inputs, opts);
}

c10::intrusive_ptr<Work> ProcessGroupGloo::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupGloo::allreduce_coalesced: " + msg);
  };
  assertNonEmpty(invalidArgument, tensors);

  // tensors will be flattened and concatenated (coalesced). This means that
  // input
  // tensors must have the same device, layout and type.
  assertLayoutMatch(invalidArgument, tensors);
  if (!std::all_of(tensors.begin(), tensors.end(), [&](at::Tensor& t) {
        return t.options().type_equal(tensors[0].options());
      })) {
    invalidArgument("tensors must all have the same type");
  }
  if (!std::all_of(tensors.begin(), tensors.end(), [&](at::Tensor& t) {
        return t.device() == tensors[0].device();
      })) {
    invalidArgument("tensors must all be on the same device");
  }

  const c10::Device& device = tensors[0].device();
  const c10::Layout& layout = tensors[0].layout();

  // invalid arguments are detected early here before any calls to nextTag()
  // which result in the collectiveCounter_ being incremented.
  switch (device.type()) {
    case c10::kCPU:
      break;
    default:
      invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  switch (layout) {
    case c10::kStrided:
      break;
    default:
      invalidArgument("unsupported layout");
  }

  c10::intrusive_ptr<AsyncWork> work;
  const uint32_t tag = nextTag();
  std::shared_ptr<gloo::Context> context = getContext(tag);
  ++seq_;
  if (device.type() == c10::kCPU) {
    if (layout == c10::kStrided) {
      work = c10::make_intrusive<AsyncAllreduceCoalescedWork>(
          std::move(context), tensors, opts.reduceOp, tag, seq_, opts.timeout);
    } else {
      invalidArgument("unsupported layout");
    }
  } else {
    TORCH_CHECK(false, "Invalid backend");
  }
  enqueue(work);
  return work;
}

namespace {

class AsyncReduceWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncReduceWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<at::Tensor>& inputs,
      int rootRank,
      int rootTensor,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : ProcessGroupGloo::AsyncWork(
            std::move(context),
            {inputs},
            OpType::REDUCE,
            seq,
            timeout,
            "gloo:reduce",
            inputs),
        inputs(inputs),
        rootRank(rootRank),
        rootTensor(rootTensor),
        reduceOp(std::move(reduceOp)),
        tag(tag) {}

  std::vector<at::Tensor> inputs;
  const int rootRank;
  const int rootTensor;
  const ReduceOp reduceOp;
  const uint32_t tag;

  void reduce(std::vector<at::Tensor>& tensors) {
    auto tensor = tensors[0];
    if (tensor.is_complex()) {
      TORCH_CHECK(
          c10d::isComplexViewAsRealAllowed(reduceOp),
          "reduce does not support",
          reduceOp,
          "on complex tensors");
      tensor = at::view_as_real(tensor);
    }
    gloo::ReduceOptions opts(context_);
    const auto& scalarType = tensor.scalar_type();
    opts.setRoot(rootRank);
    opts.setTag(tag);
    opts.setReduceFunction(getFunction(scalarType, reduceOp));
    opts.setTimeout(timeout_);
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, tensor);
    gloo::reduce(opts);

    // Gloo doesn't support AVG so we use SUM + division.
    if (reduceOp == ReduceOp::AVG) {
      tensors[0] /= context_->size;
    }
  }

  void run() override {
    reduce(inputs);
  }

  const std::vector<at::Tensor> getInputTensors() override {
    return inputs;
  }

  const std::vector<at::Tensor> getOutputTensors() override {
    return inputs;
  }

 protected:
  template <typename T>
  void getFunction(gloo::ReduceOptions::Func& fn, const ReduceOp op) {
    fn = toFunction<T>(op);
  }

  gloo::ReduceOptions::Func getFunction(
      const at::ScalarType& dtype,
      const ReduceOp& op) {
    gloo::ReduceOptions::Func fn;
    GENERATE_ALL_TYPES(dtype, getFunction, fn, op);
    return fn;
  }
};

class AsyncReduceCUDAWork : public AsyncReduceWork {
 public:
  AsyncReduceCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      int rootRank,
      int rootTensor,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : AsyncReduceWork(
            context,
            inputs,
            rootRank,
            rootTensor,
            std::move(reduceOp),
            tag,
            seq,
            timeout) {
    initializeStreamsEvents(inputs, streams, events);

    // Kick off copy from CUDA tensors to pinned CPU tensors.
    tmp.reserve(inputs.size());
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      tmp.push_back(pinnedLike(inputs[i]).copy_(inputs[i], true));
    }
  }

  void run() override {
    // Synchronize with copy operations.
    for (const auto i : c10::irange(inputs.size())) {
      streams[i].synchronize();
    }

    // Run reduce on host side tensors.
    reduce(tmp);

    // Kick off copy back to the CUDA tensors.
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      inputs[i].copy_(tmp[i], /* non_blocking */ true);
      events[i].record(streams[i]);
    }
  }

  void synchronize() override {
    // Synchronize with the copy back to CUDA tensors.
    for (const auto i : c10::irange(inputs.size())) {
      c10::Device device = inputs[i].device();
      events[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  std::vector<at::Tensor> tmp;
  std::vector<c10::Stream> streams;
  std::vector<c10::Event> events;
};

} // namespace

c10::intrusive_ptr<Work> ProcessGroupGloo::reduce(
    std::vector<at::Tensor>& inputs,
    const ReduceOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupGloo::reduce: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);
  assertRootTensor(
      invalidArgument, opts.rootTensor, static_cast<int64_t>(inputs.size()));
  assertSingleElement(invalidArgument, inputs);
  assertDense(invalidArgument, inputs);

  const auto& device = inputs[0].device();
  switch (device.type()) {
    case at::kCPU:
      break;
    case at::kCUDA:
      // If the user gave us a CUDA tensor then CUDA must be loaded.
      TORCH_INTERNAL_ASSERT(at::hasCUDA());
      break;
    default:
      invalidArgument(c10::str("unsupported device type ", device.type()));
  }

  c10::intrusive_ptr<AsyncReduceWork> work;
  auto tag = nextTag();
  auto context = getContext(tag);
  ++seq_;
  if (device.type() == at::kCPU) {
    work = c10::make_intrusive<AsyncReduceWork>(
        std::move(context),
        inputs,
        opts.rootRank,
        opts.rootTensor,
        opts.reduceOp,
        tag,
        seq_,
        opts.timeout);
  } else if (device.type() == at::kCUDA) {
    work = c10::make_intrusive<AsyncReduceCUDAWork>(
        std::move(context),
        inputs,
        opts.rootRank,
        opts.rootTensor,
        opts.reduceOp,
        tag,
        seq_,
        opts.timeout);
  } else {
    TORCH_CHECK(false, "Invalid backend");
  }
  enqueue(work);
  return work;
}

namespace {

class AsyncAllgatherWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncAllgatherWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : ProcessGroupGloo::AsyncWork(
            std::move(context),
            outputs,
            OpType::ALLGATHER,
            seq,
            timeout,
            "gloo:all_gather",
            inputs),
        outputs(outputs),
        inputs(inputs),
        tag(tag) {}

  std::vector<std::vector<at::Tensor>> outputs;
  std::vector<at::Tensor> inputs;
  const uint32_t tag;

  void allgather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs) {
    const auto& scalarType = inputs[0].scalar_type();
    gloo::AllgatherOptions opts(context_);
    opts.setTag(tag);
    opts.setTimeout(timeout_);

    // Use single flattened input tensor.
    at::Tensor flatInputTensor = flattenDenseTensors(inputs);
    GENERATE_ALL_TYPES(scalarType, setInput, opts, flatInputTensor);

    // Use single flat output tensor.
    // The first dimension corresponds to the index into outputs[N],
    // so copying into the actual output later is easy.
    at::Tensor flatOutputTensor = newLikeFlat(outputs[0]);
    GENERATE_ALL_TYPES(scalarType, setOutput, opts, flatOutputTensor);
    gloo::allgather(opts);

    // Unflatten into output tensors.
    for (auto& outputgroup : outputs) {
      for (const auto j : c10::irange(outputgroup.size())) {
        outputgroup[j].copy_(flatOutputTensor[static_cast<int64_t>(j)]);
      }
    }
  }

  const std::vector<at::Tensor> getInputTensors() override {
    return inputs;
  }

  const std::vector<at::Tensor> getOutputTensors() override {
    return {newLikeFlat(outputs[0])};
  }

  void run() override {
    allgather(outputs, inputs);
  }
};

// Note: current CUDA implementation holds the assumption that the
// tensors in the nested output tensor vectors are on the same device.
class AsyncAllgatherCUDAWork : public AsyncAllgatherWork {
 public:
  AsyncAllgatherCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : AsyncAllgatherWork(context, outputs, inputs, tag, seq, timeout) {
    initializeStreamsEvents(inputs, inputStreams, inputEvents);
    initializeStreamsEvents(outputs, outputStreams, outputEvents);

    // Kick off copy from CUDA tensors to pinned CPU tensors.
    tmpInputs.reserve(inputs.size());
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(inputStreams[i]);
      tmpInputs.push_back(pinnedLike(inputs[i]).copy_(inputs[i], true));
    }

    tmpOutputs.resize(outputs.size());
    for (const auto i : c10::irange(outputs.size())) {
      tmpOutputs[i].reserve(outputs[i].size());
      for (const auto j : c10::irange(outputs[i].size())) {
        tmpOutputs[i].push_back(pinnedLike(outputs[i][j]));
      }
    }
  }

  void run() override {
    // Synchronize with copy operations.
    for (const auto i : c10::irange(inputs.size())) {
      inputStreams[i].synchronize();
    }

    for (const auto i : c10::irange(outputs.size())) {
      outputStreams[i].synchronize();
    }

    // Run allgather on host side tensors.
    allgather(tmpOutputs, tmpInputs);

    // Kick off copy back to the CUDA tensors.
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(outputs.size())) {
      guard.reset_stream(outputStreams[i]);
      for (const auto j : c10::irange(outputs[i].size())) {
        outputs[i][j].copy_(tmpOutputs[i][j], /* non_blocking */ true);
      }
      outputEvents[i].record(outputStreams[i]);
    }
  }

  void synchronize() override {
    // Synchronize with the copy back to CUDA tensors.
    for (const auto i : c10::irange(outputs.size())) {
      c10::Device device = outputs[i][0].device();
      outputEvents[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  std::vector<at::Tensor> tmpInputs;
  std::vector<c10::Stream> inputStreams;
  std::vector<c10::Event> inputEvents;

  std::vector<std::vector<at::Tensor>> tmpOutputs;
  std::vector<c10::Stream> outputStreams;
  std::vector<c10::Event> outputEvents;
};

// A work that takes an lambda on construction and calls it on wait.
// It is useful for add a continuation to another work, and/or
// composing multiple works together.
class LambdaWork : public Work {
 public:
  LambdaWork(std::function<void(void)> fn) : fn_(std::move(fn)) {}

  bool wait(std::chrono::milliseconds /* unused */) override {
    fn_();
    return true;
  }

 private:
  std::function<void(void)> fn_;
};

} // namespace

c10::intrusive_ptr<Work> ProcessGroupGloo::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceScatterOptions& opts) {
  std::vector<at::Tensor> outputTensors = {outputTensor};
  std::vector<at::Tensor> inputTensors = {inputTensor};
  return reduce_scatter_tensor_coalesced(outputTensors, inputTensors, opts);
}

c10::intrusive_ptr<Work> ProcessGroupGloo::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const ReduceScatterOptions& opts) {
  if (outputTensors.size() != inputTensors.size()) {
    TORCH_CHECK(
        false, "requires input/output tensor lists to have the same length");
  }
  const auto rank = getRank();
  const auto worldSize = getSize();
  std::vector<at::Tensor> buffers;
  for (const auto i : c10::irange(inputTensors.size())) {
    auto inputShape = inputTensors[i].sizes().vec();
    auto outputShape = outputTensors[i].sizes().vec();
    TORCH_CHECK_EQ(outputTensors[i].dtype(), inputTensors[i].dtype());
    TORCH_CHECK_EQ(outputShape[0] * worldSize, inputShape[0]);
    for (size_t i = 1; i < outputShape.size(); ++i) {
      TORCH_CHECK_EQ(outputShape[i], inputShape[i]);
    }
    buffers.push_back(inputTensors[i].clone());
  }
  std::vector<c10::intrusive_ptr<Work>> works;
  for (const auto i : c10::irange(buffers.size())) {
    std::vector<at::Tensor> inp = {buffers[i]};
    AllreduceOptions arOpts;
    arOpts.reduceOp = opts.reduceOp;
    arOpts.timeout = opts.timeout;
    works.push_back(allreduce(inp, arOpts));
  }
  return c10::make_intrusive<LambdaWork>(
      [rank, worldSize, buffers, outputTensors, works = std::move(works)]() {
        for (const auto i : c10::irange(outputTensors.size())) {
          works[i]->wait();
          outputTensors[i].copy_(buffers[i].chunk(worldSize)[rank]);
        }
      });
}

c10::intrusive_ptr<Work> ProcessGroupGloo::_allgather_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const AllgatherOptions& opts) {
  auto tensor_list = at::chunk(output_tensor, this->getSize(), 0);
  std::vector<std::vector<at::Tensor>> outputs = {tensor_list};
  std::vector<at::Tensor> inputs = {input_tensor};
  return this->allgather(outputs, inputs, opts);
}
// Note: current CUDA implementation holds the assumption that the
// tensors in the nested output tensor vectors are on the same device.
c10::intrusive_ptr<Work> ProcessGroupGloo::allgather(
    std::vector<std::vector<at::Tensor>>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    TORCH_CHECK(false, "ProcessGroupGloo::allgather: " + msg);
  };

  if (inputs.empty()) {
    invalidArgument("requires non-empty input tensor list");
  }

  if (inputs.size() != outputs.size()) {
    invalidArgument(
        "requires input/output tensor lists to have the same length");
  }

  for (const auto i : c10::irange(outputs.size())) {
    const
```



## High-Level Overview


This C++ file contains approximately 15 class(es)/struct(s) and 92 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`, `inline`, `std`, `c10d`

**Classes/Structs**: `addrinfo`, `addrinfo`, `addrinfo`, `AsyncBroadcastWork`, `AsyncBroadcastCUDAWork`, `AsyncReduceWork`, `AsyncReduceCUDAWork`, `AsyncAllgatherWork`, `AsyncAllgatherCUDAWork`, `LambdaWork`, `AsyncAllgatherCoalescedWork`, `AsyncGatherWork`, `AsyncGatherCUDAWork`, `AsyncScatterWork`, `AsyncScatterCUDAWork`, `AsyncAlltoallWork`, `AsyncAlltoallCUDAWork`, `unbound`, `unbound`, `unbound`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/Exception.h`
- `c10/util/error.h`
- `torch/csrc/distributed/c10d/ProcessGroupGloo.hpp`
- `torch/csrc/distributed/c10d/FlightRecorder.hpp`
- `torch/csrc/distributed/c10d/GlooDeviceFactory.hpp`
- `torch/csrc/distributed/c10d/PrefixStore.hpp`
- `torch/csrc/distributed/c10d/ProcessGroup.hpp`
- `torch/csrc/distributed/c10d/ProcessGroupGlooDetail.hpp`
- `torch/csrc/distributed/c10d/Utils.hpp`
- `chrono`
- `exception`
- `gloo/common/win.h`
- `winsock2.h`
- `ws2tcpip.h`
- `netdb.h`
- `sys/socket.h`
- `unistd.h`
- `sys/types.h`
- `type_traits`
- `utility`
- `ATen/ThreadLocalState.h`
- `ATen/native/SparseTensorUtils.h`
- `c10/util/StringUtil.h`
- `c10/util/intrusive_ptr.h`
- `c10/util/irange.h`
- `gloo/config.h`
- `gloo/rendezvous/context.h`
- `gloo/rendezvous/prefix_store.h`


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

- **File Documentation**: `ProcessGroupGloo.cpp_docs.md`
- **Keyword Index**: `ProcessGroupGloo.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
