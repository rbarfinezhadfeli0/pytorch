# Documentation: `torch/csrc/distributed/c10d/reducer.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/reducer.cpp`
- **Size**: 96,801 bytes (94.53 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/distributed/c10d/reducer.hpp>

#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/default_comm_hooks.hpp>

#include <functional>

#include <c10/core/DeviceGuard.h>
#include <c10/core/ScalarType.h>
#include <c10/core/StreamGuard.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>
#include <c10/util/hash.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/utils/grad_layout_contract.h>
#include <torch/csrc/autograd/utils/lambda_post_hook.h>
#include <torch/csrc/distributed/c10d/comm.hpp>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <utility>

namespace c10d {
namespace {

constexpr int kUnsetDivFactor = -1;

// Macro that wraps TORCH_CHECK with DDP logging.
#define REDUCER_CHECK(cond, logger_, ...)             \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {               \
    if (!logger_.expired()) {                         \
      logger_.lock()->set_error_and_log(__VA_ARGS__); \
    }                                                 \
    TORCH_CHECK(false, ##__VA_ARGS__);                \
  }

} // namespace

C10_DEFINE_TYPED_REGISTRY(
    TimerRegistry,
    c10::DeviceType,
    Timer,
    std::unique_ptr,
    c10::Device)

namespace {

class CpuTimer : public Timer {
 public:
  explicit CpuTimer(c10::Device /* unused */) {}

  std::optional<int64_t> measureDifference(Event start, Event end) override {
    int64_t start_time = getTimeRef(start);
    int64_t end_time = getTimeRef(end);
    // If cpu_end_time is not recorded in this iteration,
    // avg_time will return invalid value.
    // For some cases like DDP runs on non-sync mode, backward compute
    // end time can not be recorded in this iteration and thus can not
    // calculate the valid avg_time.
    // In this case, skip calculating the avg_time and return.
    if (end_time < start_time) {
      return std::nullopt;
    }
    return end_time - start_time;
  }
};

C10_REGISTER_TYPED_CLASS(TimerRegistry, c10::kCPU, CpuTimer)

std::vector<at::Tensor> extractTensors(const c10::IValue& result) {
  if (result.isPyObject()) {
    return result.toPyObjectHolder()->extractTensors();
  }
  TORCH_INTERNAL_ASSERT(
      result.isTensor() || result.isTensorList(),
      "expected the hook result is either a Tensor or a TensorList found ",
      result.tagKind());

  if (result.isTensor()) {
    return {result.toTensor()};
  }

  return result.toTensorVector();
}

} // namespace

Reducer::Reducer(
    std::vector<at::Tensor> params,
    std::vector<std::vector<size_t>> bucket_indices,
    c10::intrusive_ptr<c10d::ProcessGroup> process_group,
    std::vector<bool> expect_sparse_gradients,
    int64_t bucket_bytes_cap,
    bool find_unused_parameters,
    bool gradient_as_bucket_view,
    std::unordered_map<size_t, std::string> param_names,
    int64_t first_bucket_bytes_cap,
    bool skip_all_reduce_unused_params,
    bool use_python_reducer)
    : params_(std::move(params)),
      process_group_(std::move(process_group)),
      expect_sparse_gradients_(std::move(expect_sparse_gradients)),
      expect_autograd_hooks_(false),
      require_finalize_(false),
      next_bucket_(0),
      has_marked_unused_parameters_(false),
      find_unused_parameters_(find_unused_parameters),
      gradient_as_bucket_view_(gradient_as_bucket_view),
      local_used_map_reduced_(false),
      num_iterations_(0),
      num_bwd_calls_(0),
      first_autograd_hook_called_(false),
      num_buckets_ready_(0),
      num_buckets_reduced_(0),
      has_rebuilt_bucket_(false),
      bucket_bytes_cap_(bucket_bytes_cap),
      div_factor_(kUnsetDivFactor),
      static_graph_(false),
      skip_all_reduce_unused_params_(skip_all_reduce_unused_params),
      comm_hook_(nullptr),
      ddp_debug_level_(debug_level()),
      param_names_(std::move(param_names)),
      first_bucket_bytes_cap_(first_bucket_bytes_cap),
      use_python_reducer_(use_python_reducer) {
  C10_LOG_API_USAGE_ONCE("torch.distributed.ddp.reducer");
  TORCH_INTERNAL_ASSERT(!params_.empty(), "Expected at least one parameter.");

  if (ddp_debug_level_ != c10d::DebugLevel::Off) {
    LOG(INFO) << "Reducer initialized with bucket_bytes_cap: "
              << bucket_bytes_cap_
              << " first_bucket_bytes_cap: " << first_bucket_bytes_cap;
  }
  // Check whether the module is multi_device_module
  {
    std::set<int> unique_devices;
    for (const auto& v : params_) {
      auto device_idx = static_cast<int>(v.device().index());
      auto [_, inserted] = unique_devices.emplace(device_idx);
      if (inserted) {
        if (unique_devices.size() > 1) {
          is_multi_device_module_ = true;
          break;
        }
      }
    }
  }

  // For CUDA, record events only for single device module.
  c10::Device device = params_[0].device();
  if (!(device.is_cuda() && is_multi_device_module_)) {
    timer_ = TimerRegistry()->Create(device.type(), device);
  }

  // If `expect_sparse_gradients` is not specified, initialize it such that
  // we do not expect sparse gradients for any parameter.
  if (expect_sparse_gradients_.empty()) {
    expect_sparse_gradients_ = std::vector<bool>(params_.size(), false);
  }
  TORCH_INTERNAL_ASSERT(expect_sparse_gradients_.size() == params_.size());

  // Initialize variable bucketing.
  // This can be reinitialized later after capturing runtime information.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    initialize_buckets(std::move(bucket_indices));
  }

  // All variables are expected to have their `grad_fn` set to the gradient
  // accumulation function (since they are leaves in the autograd graph).
  // We store pointers to these functions such that we can check if they are
  // used in an autograd pass. If they are not, we know their grad tensors
  // can be marked as ready for reduction.
  {
    const auto variable_count = params_.size();
    grad_accumulators_.resize(variable_count);
    for (const auto variable_index : c10::irange(variable_count)) {
      auto& variable = params_[variable_index];

      // The gradient accumulator function is lazily initialized once.
      // Therefore we can use its presence in the autograd graph as
      // evidence that the parameter has participated in an iteration.
      auto grad_accumulator = torch::autograd::impl::grad_accumulator(variable);

#ifndef _WIN32
      using torch::distributed::autograd::ThreadLocalDistAutogradContext;
#endif
      // Hook to execute after the gradient accumulator has executed.
      hooks_.emplace_back(
          grad_accumulator->add_post_hook(std::make_unique<
                                          torch::autograd::utils::
                                              LambdaPostHook>(
              [this, variable_index](
                  const torch::autograd::variable_list& outputs,
                  const torch::autograd::variable_list& /* unused */) {
#ifndef _WIN32
                this->rpc_context_.set(
                    ThreadLocalDistAutogradContext::getContextPtr());
#endif
                this->autograd_hook(variable_index);
                return outputs;
              },
              [this](torch::autograd::CompiledNodeArgs& args) {
                TORCH_CHECK(
                    this->use_python_reducer_,
                    "Compiled autograd is not compatible with C++ DDP Reducer, please use torch._dynamo.config.optimize_ddp=\"python_reducer\".");
              })),
          grad_accumulator);

      // Map raw function pointer to parameter index.
      // This is used later on when the autograd graph is traversed
      // to check for parameters for which no gradient is computed, if
      // find_unused_parameters=True.
      // Note that the mapping of gradient accumulator to variable should be
      // one to one as we deduplicate shared parameters before constructing
      // Reducer.
      if (find_unused_parameters_) {
        gradAccToVariableMap_[grad_accumulator.get()] = variable_index;
      }

      numGradHooksTriggeredMap_[variable_index] = 0;

      // The gradient accumulator is stored as weak_ptr in the autograd
      // metadata of the variable, so we have to keep it alive here for
      // the raw pointer to be valid.
      REDUCER_CHECK(
          grad_accumulators_[variable_index] == nullptr,
          logger_,
          c10::str(
              "Reducer tried to register duplicate grad accumulator for variable ",
              variable_index));

      grad_accumulators_[variable_index] = std::move(grad_accumulator);
    }
  }

  // Initialize backward stats vector.
  {
    const auto variable_count = params_.size();
    backward_stats_.resize(variable_count);
  }

  // See Note [Skip allreducing local_used_map_dev]
  if (find_unused_parameters_) {
    initialize_local_used_map();
  }
}

// Note [Skip allreducing local_used_map_dev]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// If find_unused_parameters_ is set to false, there is no need to allreduce
// local_used_map_dev_, because all parameters will be reduced anyway.
// Therefore, we can avoid allocating memory for local_used_map and
// local_used_map_dev_ if find_unused_parameters_ is false.

// Note [DDP Communication Hook]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// If DDP communication hook is not registered, the reducer reduces the buckets
// by just calling allreduce. If registered, it calls the hook and uses future
// work handle. If registered, reducer also skips dividing grads by world size.
// The reason for this is that the communication hook is expected to completely
// override how we perform communication and the user should have complete
// control over how the grads are handled.
//
// DDP communication hook is an enhancement that provides a hook which can be
// used to override how DDP communicates gradients across ranks, this can be
// used for algorithms like Gradient Compression/GossipGrad. This hook can be
// registered from Python API using `register_comm_hook`. `PythonCommHook`
// enables registering a Python hook and is a subclass of `CommHookInterface`.
// Additionally, there are also some built-in C++ hook implementations that can
// be specified by calling `register_builtin_comm_hook` from Python API.

Reducer::~Reducer() noexcept(false) {
  remove_autograd_hooks();
}

bool Reducer::dynamic_graph_find_unused() {
  return !static_graph_ && find_unused_parameters_;
}

bool Reducer::static_graph_first_iteration() {
  return static_graph_ && num_bwd_calls_ == 1;
}

bool Reducer::static_graph_after_first_iteration() {
  return static_graph_ && num_bwd_calls_ > 1;
}

bool Reducer::ddp_graph_static() {
  std::lock_guard<std::mutex> lock(mutex_);
  return ddp_graph_static_;
}

void Reducer::initialize_local_used_map() {
  const auto variable_count = params_.size();
  at::TensorOptions options;
  options = options.dtype(at::kInt);

  // Deliberately don't pin the memory even if local_used_map_dev_ will
  // be cuda. See Note [local_used_map_ -> local_used_map_dev copying]
  local_used_map_ = at::zeros({static_cast<long>(variable_count)}, options);

  // This tensor needs to be on the same device as the replica params because
  // backend such as NCCL may not support CPU tensors, and hence it might not
  // work if we always put it on CPU. The dist backend for MTIA doesn't support
  // int32 allreduce for now, so it has to be placed on CPU.
  options = options.device(
      (params_[0].is_mtia()) ? c10::Device(c10::DeviceType::CPU)
                             : params_[0].device());
  local_used_map_dev_ = at::empty({static_cast<long>(variable_count)}, options);
}

void Reducer::check_grad_layout(
    const at::Tensor& grad,
    const at::Tensor& bucket_view) {
  // Ensure that the gradient type matches the bucket type, or mixed precision
  // type if we are training with mixed precision.
  auto type = mixed_precision_param_dtype_
      ? *mixed_precision_param_dtype_
      : bucket_view.options().dtype().toScalarType();
  REDUCER_CHECK(
      grad.options().dtype().toScalarType() == type,
      logger_,
      c10::str(
          "Expected ", type, ", got ", grad.options().dtype().toScalarType()));

  TORCH_INTERNAL_ASSERT(grad.device() == bucket_view.device());
  TORCH_INTERNAL_ASSERT(grad.numel() == bucket_view.numel());
  // AccumulateGrad doesn't HAVE to obey the grad layout contract.
  // The penalty for disobedience is reduced performance, not numerical
  // death. Warnings here help diagnose poor DDP performance.
  if (grad.strides() != bucket_view.strides()) {
    TORCH_WARN_ONCE(
        "Grad strides do not match bucket view strides. "
        "This may indicate grad was not created according to the "
        "gradient layout contract, or that the param's strides "
        "changed since DDP was constructed.  This is not an error, "
        "but may impair performance.\n"
        "grad.sizes() = ",
        grad.sizes(),
        ", strides() = ",
        grad.strides(),
        "\n",
        "bucket_view.sizes() = ",
        bucket_view.sizes(),
        ", strides() = ",
        bucket_view.strides());
  }
  if (!gradient_as_bucket_view_) {
    TORCH_INTERNAL_ASSERT(!grad.is_alias_of(bucket_view));
  }
}

void Reducer::mark_variable_ready_dense(size_t variable_index) {
  const auto& bucket_index = variable_locators_[variable_index];
  auto& bucket = buckets_[bucket_index.bucket_index];
  auto& variable = bucket.variables[bucket_index.intra_bucket_index];
  auto& bucket_view = bucket.bucket_views_in[bucket_index.intra_bucket_index];

  // Copy the contents of the gradient tensor to the corresponding part of the
  // bucket's flattened gradient tensor.
  // If the gradient is not set, we assume it wasn't computed as part of the
  // current backwards pass, and we zero the part of the bucket it would
  // otherwise hold.
  runGradCallbackForVariable(variable, [&](auto& grad) {
    if (grad.defined()) {
      this->check_grad_layout(grad, bucket_view);
      // When gradient_as_bucket_view_ is false, or even when
      // gradient_as_bucket_view_ is true, in rare cases users may set grad to
      // be None after every iteration. In these cases, grad and bucket_view are
      // pointing to different storages and thus need to copy grads to
      // bucket_view. If gradient_as_bucket_view_ is set as true, let grad point
      // to bucket_view. If grad has already been set as views of buckets in
      // previous iterations, no copy is needed.
      if (!grad.is_alias_of(bucket_view)) {
        if (comm_hook_ == nullptr) {
          auto wrapped = at::native::wrapped_scalar_tensor(1. / div_factor_);
          if (!grad.requires_grad()) {
            // Divides while copying into the bucket view to save one scan over
            // all the input parameters.
            RECORD_FUNCTION(
                "torch::distributed::reducer::mul_out",
                std::vector<c10::IValue>({bucket_view}))
            at::mul_out(bucket_view, grad, wrapped);
          } else {
            // If DDP is running with create_graph=True, gradients require_grad
            // themselves in order to compute higher order derivatives. However,
            // DDP will not sync up these gradients currently (see
            // https://github.com/pytorch/pytorch/issues/63812).
            C10_LOG_EVERY_N(WARNING, 1000)
                << "Using DistributedDataParallel with create_graph=True "
                << " is not well-supported. The higher-order gradient will "
                << " not be synchronized across ranks, and backpropagation "
                << " through all_reduce operations will not occur. If you require "
                << " DDP to work with higher-order gradients for your use case, "
                << " please ping https://github.com/pytorch/pytorch/issues/63929";
            auto div_result = at::mul(grad, wrapped);
            RECORD_FUNCTION(
                "torch::distributed::reducer::copy_",
                std::vector<c10::IValue>({bucket_view}))
            bucket_view.copy_(div_result);
          }
        } else {
          RECORD_FUNCTION(
              "torch::distributed::reducer::copy_",
              std::vector<c10::IValue>({bucket_view}))
          bucket_view.copy_(grad);
        }

        if (gradient_as_bucket_view_) {
          // Let grad point to bucket_view buffer.
          grad = bucket_view;
          // The grad is modified and need to be written back.
          return true;
        }
      } else {
        // If grad and bucket view point to the same storage, no need to copy.
        if (comm_hook_ == nullptr) {
          bucket_view.div_(div_factor_);
        }
      }
    } else {
      // Gradient is undefined. When find_unused_parameters=True, ensure it is
      // not marked as locally used, otherwise we will be allreducing zero's
      // instead of not touching .grad field of parameter.
      if (this->dynamic_graph_find_unused() ||
          this->static_graph_first_iteration()) {
        REDUCER_CHECK(
            local_used_map_[variable_index].item<int>() == 0,
            logger_,
            "Encountered gradient which is undefined, but still allreduced by "
            "DDP reducer. This indicates a bug in DDP implementation, please "
            "report a bug with a repro to PyTorch.");
      }
      bucket_view.zero_();
    }
    // The grad is not modified and doesn't need to be written back.
    return false;
  });
}

void Reducer::mark_variable_ready_sparse(size_t variable_index) {
  const auto& bucket_index = variable_locators_[variable_index];
  auto& bucket = buckets_[bucket_index.bucket_index];
  auto& variable = bucket.variables[bucket_index.intra_bucket_index];

  runGradCallbackForVariable(variable, [&](auto& grad) {
    REDUCER_CHECK(
        grad.defined(), logger_, "Expected sparse gradient to be defined.");
    REDUCER_CHECK(
        grad.options().layout() == c10::kSparse,
        logger_,
        "Expected variable to have sparse gradient.");

    // Copy the indices of sparse metadata
    if (sparse_metadata_) {
      grad = grad.coalesce();
      REDUCER_CHECK(
          !param_names_.empty(), logger_, "No parameter names were found");
      std::string& param_name = param_names_[variable_index];
      auto iter = sparse_metadata_->find(param_name);
      REDUCER_CHECK(
          iter != sparse_metadata_->end(),
          logger_,
          "param: " + param_name + " not found in sparse metadata");
      bucket.sparse_tensor_indices =
          iter->second.to(at::kLong).unsqueeze(0).to(grad.device());
      auto indices = at::searchsorted(
          bucket.sparse_tensor_indices.value(), grad.indices(), false, false);
      // For indices we are using the ones set by sparse_metadata
      grad = at::sparse_coo_tensor(indices, grad.values(), grad.sizes());
    }

    // Sparse tensors cannot be grouped together with other sparse tensors in a
    // single reduction operation like we can for dense tensors. Therefore, the
    // `offsets` and `lengths` vectors in the bucket struct are empty, and
    // there is no pre-existing accumulation tensor.
    // Directly assign the sparse tensor to the `gradients` field.
    bucket.gradients = grad;
    // If no DDP comm hook is registered, the allreduce only sums up the
    // value, and a separate division is required.
    if (comm_hook_ == nullptr) {
      bucket.gradients.div_(div_factor_);
    }
    // The grad is modified in place and needs to be written back.
    return true;
  });
}

std::vector<c10d::GradBucket> Reducer::get_grad_buckets(
    bool return_zero_tensors) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<c10d::GradBucket> gradBuckets;
  gradBuckets.reserve(buckets_.size());
  for (const auto i : c10::irange(buckets_.size())) {
    auto& bucket = buckets_[i];
    auto variables_for_bucket = get_variables_for_bucket(i, bucket);
    gradBuckets.emplace_back(
        i,
        buckets_.size(),
        return_zero_tensors ? at::zeros_like(bucket.gradients)
                            : bucket.gradients,
        bucket.offsets,
        bucket.lengths,
        bucket.sizes_vec,
        variables_for_bucket,
        std::nullopt);
  }
  return gradBuckets;
}

void Reducer::set_forward_pass_work_handle(
    c10::intrusive_ptr<c10d::Work> forwardPassWorkHandle,
    bool useStaticWorldSize) {
  std::lock_guard<std::mutex> lock(mutex_);
  forwardPassWorkHandle_.workHandle = std::move(forwardPassWorkHandle);
  forwardPassWorkHandle_.useStaticWorldSize = useStaticWorldSize;
}

at::Tensor Reducer::get_local_used_map_on_device() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return local_used_map_dev_;
}

void Reducer::push_rebuilt_params_for_all_indices() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!should_rebuild_buckets() || !rebuilt_param_indices_.empty()) {
    return;
  }
  const auto variable_count = params_.size();
  for (const auto variable_index : c10::irange(variable_count)) {
    push_rebuilt_params(variable_index);
  }
}

void Reducer::push_rebuilt_params(const size_t& index) {
  rebuilt_params_.push_back(params_[index]);
  rebuilt_param_indices_.push_back(static_cast<int64_t>(index));
}

void Reducer::set_divide_factor() {
  // If it was scheduled, wait on allreduce in forward pass that tells us
  // division factor based on no. of currently participating processes.
  if (div_factor_ == kUnsetDivFactor) {
    div_factor_ = process_group_->getSize();
    auto& workHandle = forwardPassWorkHandle_.workHandle;
    if (workHandle && !forwardPassWorkHandle_.useStaticWorldSize) {
      workHandle->wait();
      // PyProcessGroup::PyWork doesn't expose value, so fetch it from the
      // future
      auto results = extractTensors(workHandle->getFuture()->value());

      // Guard against the results being empty
      TORCH_INTERNAL_ASSERT(!results.empty());
      at::Tensor& res = results.front();
      div_factor_ = res.item().to<int>();
    }
  }
}

// This is called before training and converts the gradients to the dtype they
// should be reduced in.
void Reducer::set_mixed_precision_param_dtype(c10::ScalarType dtype) {
  mixed_precision_param_dtype_ = dtype;
  for (auto& bucket : buckets_) {
    bucket.gradients = bucket.gradients.to(dtype);
  }
}

// Right now delay_all_reduce is only called when static_graph_=true and
// num_iterations_==1.
void Reducer::delay_all_reduce() {
  std::lock_guard<std::mutex> lock(this->mutex_);

  if (should_collect_runtime_stats()) {
    record_backward_compute_end_time();
    record_backward_comm_start_time();
  }

  // launch all reduce local used map
  all_reduce_local_used_map();

  // prepare to set unused_parameters_, if it is static graph,
  // unused_parameters_ will not change after 1st iteration.
  unused_parameters_.clear();

  require_finalize_ = true;
  // copy all gradients to buckets
  for (const auto variable_index : c10::irange(params_.size())) {
    // set unused_parameters_
    if (numGradHooksTriggeredMap_[variable_index] == 0) {
      unused_parameters_.push_back(variable_index);
    }
    set_divide_factor();
    if (expect_sparse_gradients_[variable_index]) {
      mark_variable_ready_sparse(variable_index);
    } else {
      mark_variable_ready_dense(variable_index);
    }
  }

  // To avoid confusion around why static graph is picking up
  // some parameters as unused on a rank vs not, we log
  // unused parameter names for each rank for better
  // debugability when TORCH_DISTRIBUTED_DEBUG is set to
  // INFO or DETAIL
  if (ddp_debug_level_ != c10d::DebugLevel::Off) {
    // construct one string to output
    std::ostringstream unused_params_stream;

    for (const auto& unused_index : unused_parameters_) {
      auto param_name = param_names_.find(unused_index);
      TORCH_INTERNAL_ASSERT(
          param_name != param_names_.end(),
          "Expected to find parameter name from unused parameters map in debug mode.");
      // Add the param_name
      unused_params_stream << "{" << param_name->second << "," << unused_index
                           << "}";
    }

    // Each rank prints out all the unused parameters detected
    if (!unused_parameters_.empty()) {
      LOG(INFO) << "[Rank " << process_group_->getRank() << "]: "
                << "Parameter(s) (in the format of {param_name, index}): "
                << unused_params_stream.str()
                << " is(are) unused during first iteration. Since"
                << " static_graph=True is enabled for DDP, we expect"
                << " this set of unused parameters to remain consistent"
                << " on this rank throughout the training.";
    }
  }

  // launch all reduces for all buckets
  for (auto& bucket : buckets_) {
    all_reduce_bucket(bucket);
  }

  finalize_backward();
}

void Reducer::set_logger(std::weak_ptr<c10d::Logger> logger) {
  logger_ = std::move(logger);
}

// The function `autograd_hook` is called after the gradient for a
// model parameter has been accumulated into its gradient tensor.
// This function is only to be called from the autograd thread.
void Reducer::autograd_hook(size_t index) {
  std::lock_guard<std::mutex> lock(this->mutex_);
  if (!first_autograd_hook_called_) {
    first_autograd_hook_called_ = true;
    num_bwd_calls_++;
  }

  // See Note [Skip allreducing local_used_map_dev]
  if (dynamic_graph_find_unused() || static_graph_first_iteration()) {
    // Since it gets here, this param has been used for this iteration. We want
    // to mark it in local_used_map_. During no_sync session, the same var can
    // be set multiple times, which is OK as does not affect correctness. As
    // long as it is used once during no_sync session, it is marked as used.
    // Only set it as locally used if the grad is defined. Otherwise, hooks can
    // be fired  with undefined grads, such as when not all outputs are used in
    // DDP when computing loss. In this case, we don't want to mark it as
    // locally used to ensure we don't touch the parameter's .grad field.
    auto& variable = get_param_from_index(index);
    runGradCallbackForVariable(variable, [&](auto& grad) {
      if (grad.defined()) {
        local_used_map_[static_cast<int64_t>(index)] = 1;
      }
      // The gradient is never modified.
      return false;
    });
  }

  if (static_graph_first_iteration()) {
    numGradHooksTriggeredMap_[index] += 1;
    return;
  }

  // Ignore if we don't expect to be called.
  // This may be the case if the user wants to accumulate gradients
  // for number of iterations before reducing them.
  if (!expect_autograd_hooks_) {
    return;
  }

  grad_ready_order_indices_.push_back(static_cast<int64_t>(index));

  // If `find_unused_parameters_` is true there may be model parameters that
  // went unused when computing the model output, they won't be part of the
  // autograd graph, and won't receive gradients. These parameters are
  // discovered in the `prepare_for_backward` function and their indexes stored
  // in the `unused_parameters_` vector.
  if (!has_marked_unused_parameters_) {
    has_marked_unused_parameters_ = true;
    for (const auto& unused_index : unused_parameters_) {
      mark_variable_ready(unused_index);
    }
  }

  // Rebuild bucket only if 1) it is the first time to rebuild bucket 2)
  // static_graph_ is true or find_unused_parameters_ is false,
  // 3) this backward pass needs to run allreduce.
  // Here, we just dump tensors and their parameter indices into
  // rebuilt_params_ and rebuilt_param_indices_ based on gradient arriving
  // order, and then at the end of finalize_backward(), buckets will be
  // rebuilt based on rebuilt_params_ and rebuilt_param_indices_, and then
  // will be broadcasted and initialized.
  // If it is static graph, after 1st iteration, check if a variable
  // is ready for communication based on numGradHooksTriggeredMap_.
  if (static_graph_after_first_iteration()) {
    REDUCER_CHECK(
        numGradHooksTriggeredMapPerIteration_[index] > 0,
        logger_,
        "Your training graph has changed in this iteration, ",
        "e.g., one parameter is unused in first iteration, but ",
        "then got used in the second iteration. this is not ",
        "compatible with static_graph set to True.");
    if (--numGradHooksTriggeredMapPerIteration_[index] == 0) {
      if (should_rebuild_buckets()) {
        push_rebuilt_params(index);
      }
      // Finally mark variable for which this function was originally called.
      mark_variable_ready(index);
    }
  } else {
    if (should_rebuild_buckets()) {
      push_rebuilt_params(index);
    }
    // Finally mark variable for which this function was originally called.
    mark_variable_ready(index);
  }
}

void Reducer::all_reduce_local_used_map() {
  // See Note [Skip allreducing local_used_map_dev]
  // H2D from local_used_map_ to local_used_map_dev_
  if (local_used_map_dev_.is_cuda() || local_used_map_dev_.is_privateuseone()) {
    // Note [local_used_map_ -> local_used_map_dev copying]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // We do async H2D to avoid the blocking overhead. The async copy and
    // allreduce respect the current stream, so will be sequenced
    // correctly.
    //
    // Correct sequencing with respect to host operations is also
    // essential. The H2D copy_ is stream ordered, while the host's
    // changes to local_used_map_ are host ordered. If a large backlog of
    // cuda/privateuseone-stream work pushes the copy_ far into the future, and
    // if no blocking calls occur between now and finalize_backward()** such
    // that finalize_backward() re-zeroes local_used_map_ on the host
    // before the stream executes the copy_, copy_ will read those zeros
    // instead of the values we thought we told it to read here. Copying
    // local_used_map_ to a pinned temporary (which the pinned caching
    // allocator should supply asynchronously) avoids this nasty, rare
    // race condition.
    //
    // ** In the hoped-for case where all params are used, DDP itself
    // won't do any blocking work between now and the re-zeroing, so the
    // danger is real.
    //
    // Defensively ensures local_used_map_tmp is distinct from
    // local_used_map_
    auto local_used_map_tmp = at::native::empty_like(
        local_used_map_,
        c10::optTypeMetaToScalarType(local_used_map_.options().dtype_opt()),
        local_used_map_.options().layout_opt(),
        local_used_map_.options().device_opt(),
        true /* pinned_memory */);
    // Paranoid asserts here because in some workloads, the pinned
    // allocator behaves in a way we don't understand, and may be bugged.
    // See https://github.com/pytorch/pytorch/pull/54474
    TORCH_INTERNAL_ASSERT(local_used_map_tmp.is_pinned());
    TORCH_INTERNAL_ASSERT(
        local_used_map_tmp.data_ptr() != local_used_map_.data_ptr());
    local_used_map_tmp.copy_(local_used_map_);
    local_used_map_dev_.copy_(local_used_map_tmp, true);
  } else if (local_used_map_dev_.is_mtia()) {
    // MTIA probably will have special logic in the future, following code might
    // be changed drastically. Therefore, a new if case is created for MTIA, for
    // now, the implementation is similar to the CUDA/privateuseone one, except
    // for the pin memory step.
    auto local_used_map_tmp = at::native::empty_like(
        local_used_map_,
        c10::optTypeMetaToScalarType(local_used_map_.options().dtype_opt()),
        local_used_map_.options().layout_opt(),
        local_used_map_.options().device_opt());
    local_used_map_tmp.copy_(local_used_map_);
    local_used_map_dev_.copy_(local_used_map_tmp, true);
  } else {
    local_used_map_dev_.copy_(local_used_map_, true);
  }
  std::vector<at::Tensor> temp_local_used_map_dev_vec_ = {local_used_map_dev_};
  local_used_work_ = process_group_->allreduce(temp_local_used_map_dev_vec_);
}

at::Tensor& Reducer::get_param_from_index(size_t index) {
  const auto& bucket_index = variable_locators_[index];
  auto& bucket = buckets_[bucket_index.bucket_index];
  // Cannot simply access variable via `bucket.variables[variable_index]` since
  // return value is used in `runGradCallbackForVariable()` which does not
  // accept const tensors.
  auto& variable = bucket.variables[bucket_index.intra_bucket_index];
  return variable;
}

void Reducer::checkAndRaiseMarkedTwiceError(size_t index) {
  // Something is wrong if all variables contained in this bucket have
  // already been marked as ready.
  // We don't expect the same variable to be marked ready twice.
  bool marked_twice =
      perIterationReadyParams_.find(index) != perIterationReadyParams_.end();

  if (marked_twice) {
    // Report index of param that has been marked twice. In debug mode, also
    // report fully qualified parameter name.
    auto param_name = param_names_.find(index);
    const bool found_param_name = param_name != param_names_.end();
    TORCH_INTERNAL_ASSERT(
        ddp_debug_level_ == c10d::DebugLevel::Off || found_param_name,
        "Expected to find parameter name in debug mode.");
    std::string paramInfo = c10::str(
        "Parameter at index ",
        index,
        found_param_name ? c10::str(" with name ", param_name->second) : "",
        " has been marked as ready twice. This means that multiple autograd engine ",
        " hooks have fired for this particular parameter during this iteration.");
    // param_names_ is empty in debug mode.
    if (!found_param_name) {
      paramInfo += c10::str(
          " You can set the environment variable TORCH_DISTRIBUTED_DEBUG to either",
          " INFO or DETAIL to print parameter names for further debugging.");
    }
    std::string common_error = c10::str(
        "Expected to mark a variable ready only once. ",
        "",
        "This error is caused by one of the following reasons: ",
        "1) Use of a module parameter outside the `forward` function. ",
        "Please make sure model parameters are not shared across multiple ",
        "concurrent forward-backward passes. or try to use _set_static_graph() ",
        "as a workaround if this module graph does not change ",
        "during training loop.",
        "2) Reused parameters in multiple reentrant backward passes. For ",
        "example, if you use multiple `checkpoint` functions to wrap the ",
        "same part of your model, it would result in the same set of ",
        "parameters been used by different reentrant backward passes ",
        "multiple times, and hence marking a variable ready multiple times. ",
        "DDP does not support such use cases in default. You can try to ",
        "use _set_static_graph() as a workaround if your module graph ",
        "does not change over iterations.");

    common_error += c10::str("\n", paramInfo);

    REDUCER_CHECK(
        has_marked_unused_parameters_,
        logger_,
        common_error,
        "3) Incorrect unused parameter detection. The return value of the ",
        "`forward` function is inspected by the distributed data parallel ",
        "wrapper to figure out if any of the module's parameters went ",
        "unused. For unused parameters, DDP would not expect gradients from ",
        "then. However, if an unused parameter becomes part of the autograd ",
        "graph at a later point in time (e.g., in a reentrant backward when ",
        "using `checkpoint`), the gradient will show up unexpectedly. If all ",
        "parameters in the model participate in the backward pass, you can ",
        "disable unused parameter detection by passing the keyword argument ",
        "`find_unused_parameters=False` to ",
        "`torch.nn.parallel.DistributedDataParallel`. If unused parameters ",
        "in the model do not change over iterations, You can try to use ",
        "_set_static_graph() as a workaround if this module graph does not ",
        "change during training loop.");
    REDUCER_CHECK(!has_marked_unused_parameters_, logger_, common_error);
  }
}

void Reducer::mark_variable_ready(size_t variable_index) {
  REDUCER_CHECK(
      variable_index < variable_locators_.size(),
      logger_,
      "Out of range variable index.");

  checkAndRaiseMarkedTwiceError(variable_index);
  perIterationReadyParams_.insert(variable_index);
  backward_stats_[variable_index] =
      current_time_in_nanos() - backward_compute_start_time_;

  // Any time we mark a variable ready (be it in line due to unused parameters,
  // or via an autograd hook), we require a call to the finalize function. If
  // this doesn't happen before the next iteration (or call to
  // `prepare_for_backwards`), we know something is wrong.
  require_finalize_ = true;

  const auto& bucket_index = variable_locators_[variable_index];
  auto& bucket = buckets_[bucket_index.bucket_index];

  set_divide_factor();

  if (bucket.expect_sparse_gradient) {
    mark_variable_ready_sparse(variable_index);
  } else {
    mark_variable_ready_dense(variable_index);
  }

  // TODO(@pietern): Make this work for both CPU/CUDA tensors.
  // When using CPU tensors we don't need to do this.
  // Record event so that we can wait for all of them.
  // auto& event = bucket.events[bucket_index.intra_bucket_index];
  // event.record();

  // Check if this was the final gradient for this bucket.
  if (--bucket.pending == 0) {
    mark_bucket_ready(bucket_index.bucket_index);
  }

  // Run finalizer function and kick off reduction for local_used_map once the
  // final bucket was marked ready.
  if (next_bucket_ == buckets_.size()) {
    if (dynamic_graph_find_unused()) {
      all_reduce_local_used_map();
    }

    torch::autograd::Engine::get_default_engine().queue_callback([this] {
      std::lock_guard<std::mutex> lock(this->mutex_);
      if (should_collect_runtime_stats()) {
        record_backward_compute_end_time();
      }
      // Check that all buckets were completed and had their work kicked off.
      TORCH_INTERNAL_ASSERT(next_bucket_ == buckets_.size());
      if (static_graph_after_first_iteration() && should_rebuild_buckets()) {
        for (const auto& unused_index : unused_parameters_) {
          push_rebuilt_params(unused_index);
        }
      }
      this->finalize_backward();
    });
  }
}

c10::intrusive_ptr<c10::ivalue::Future> Reducer::run_comm_hook(
    GradBucket& grad_bucket) {
  if (comm_hook_ == nullptr) {
    return run_allreduce_hook(grad_bucket);
  } else {
    return comm_hook_->runHook(grad_bucket);
  }
}

c10::intrusive_ptr<c10::ivalue::Future> Reducer::run_allreduce_hook(
    GradBucket& grad_bucket) {
  _AllReduceBySumCommHook allreduce_hook(process_group_);
  return allreduce_hook.runHook(grad_bucket);
}

void Reducer::all_reduce_bucket(Bucket& bucket) {
  auto variables_for_bucket = get_variables_for_bucket(next_bucket_, bucket);
  // TODO(@pietern): Ensure proper synchronization with the CUDA events
  // that recorded copies into this `gradients` tensor. If these copies are
  // executed on non-default streams, the current stream for the device
  // that holds the `gradients` tensor must wait on these events.
  //
  // As long as autograd uses the default stream for every device,
  // these operations are implicitly sequenced, and we don't need to
  // do any extra synchronization here.
  const auto& tensor = bucket.gradients;

  GradBucket grad_bucket(
      next_bucket_,
      buckets_.size(),
      tensor,
      bucket.offsets,
      bucket.lengths,
      bucket.sizes_vec,
      variables_for_bucket,
      bucket.sparse_tensor_indices);
  bucket.future_work = run_comm_hook(grad_bucket);
}

std::vector<at::Tensor> Reducer::get_variables_for_bucket(
    size_t bucket_index,
    const Bucket& bucket) const {
  // Check if we have cached mapping previously.
  if (has_rebuilt_bucket_ &&
      cached_variables_for_bucket_.find(bucket_index) !=
          cached_variables_for_bucket_.end()) {
    return cached_variables_for_bucket_[bucket_index];
  }
  std::vector<at::Tensor> variables_for_bucket;
  variables_for_bucket.reserve(bucket.variable_indices.size());
  for (const auto& variable_index : bucket.variable_indices) {
    // Grab bucket index where gradient is located using variable_locators_.
    auto& bucket_index_for_variable = variable_locators_[variable_index];
    // Grab the actual model parameter.
    auto& variable =
        bucket.variables[bucket_index_for_variable.intra_bucket_index];
    variables_for_bucket.emplace_back(variable);
  }

  if (has_rebuilt_bucket_) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        cached_variables_for_bucket_.find(bucket_index) ==
        cached_variables_for_bucket_.end());
    cached_variables_for_bucket_.insert(
        {bucket_index, std::move(variables_for_bucket)});
    return cached_variables_for_bucket_[bucket_index];
  } else {
    return variables_for_bucket;
  }
}

bool Reducer::is_unused_bucket(Bucket& bucket) {
  for (const auto& variable_index : bucket.variable_indices) {
    if (std::find(
            unused_parameters_.begin(),
            unused_parameters_.end(),
            variable_index) == unused_parameters_.end()) {
      return false;
    }
  }
  return true;
}

bool Reducer::should_skip_all_reduce_bucket(Bucket& bucket) {
  return is_unused_bucket(bucket) && skip_all_reduce_unused_params_;
}

// Called when the bucket at the specified index is ready to be reduced.
void Reducer::mark_bucket_ready(size_t bucket_index) {
  TORCH_INTERNAL_ASSERT(bucket_index >= next_bucket_);

  // Buckets are reduced in sequence. Ignore this bucket if
  // it's not its turn to be reduced.
  if (bucket_index > next_bucket_) {
    return;
  }

  // Keep going, until we either:
  // - have kicked off reduction for all buckets, or
  // - found a bucket that's not yet ready for reduction.
  for (; next_bucket_ < buckets_.size() && buckets_[next_bucket_].pending == 0;
       next_bucket_++) {
    num_buckets_ready_++;
    if (num_buckets_ready_ == 1 && should_collect_runtime_stats()) {
      record_backward_comm_start_time();
    }
    auto& bucket = buckets_[next_bucket_];
    if (!should_skip_all_reduce_bucket(bucket)) {
      all_reduce_bucket(bucket);
      num_buckets_reduced_++;
    }
  }
}

void Reducer::install_futures(
    const c10::List<c10::intrusive_ptr<c10::ivalue::Future>>& futs) {
  // Append instead of overwrite so that this method can be called multiple
  // times in one iteration.
  if (!installed_futures_) {
    installed_futures_ = futs;
  } else {
    installed_futures_->append(futs);
  }
}

void Reducer::initialize_buckets(
    std::vector<std::vector<size_t>> bucket_indices) {
  // If initialize_buckets is called inside DDP constructor, then
  // it does not matter rpc context ptr is nullptr or not, as grad
  // will not be mutated.
  // If initialize_buckets is called during training loop, e.g, inside
  // rebuild_buckets(), since grad could be mutated and be pointed to
  // bucket_view, then it needs to check rpc context ptr is nullptr or not,
  // If rpc context ptr is nullptr, mutate variable.grad(); otherwise,
  // mutate grad in rpc context.
#ifndef _WIN32
  using torch::distributed::autograd::ThreadLocalDistAutogradContext;
  this->rpc_context_.set(ThreadLocalDistAutogradContext::getContextPtr());
#endif

  // This shouldn't be called if we're expecting autograd hooks to fire.
  REDUCER_CHECK(
      !expect_autograd_hooks_,
      logger_,
      "`initialize_buckets` must NOT be called during autograd execution.");

  // Clear current bucket assignment.
  buckets_.clear();
  variable_locators_.clear();

  // Ensure we have a bucket index for every variable.
  variable_locators_.resize(params_.size());

  // Iterate over buckets.
  const auto bucket_count = bucket_indices.size();
  buckets_.reserve(bucket_count);
  for (const auto bucket_index : c10::irange(bucket_count)) {
    Bucket bucket;

    // TODO(@pietern): Validate indices.
    // Must be non-empty, unique, and unique across buckets.
    REDUCER_CHECK(
        !bucket_indices[bucket_index].empty(),
        logger_,
        "Empty bucket specified.");

    // Variables that expect sparse gradients must have their own bucket.
    if (bucket_indices[bucket_index].size() == 1) {
      const auto variable_index = bucket_indices[bucket_index].front();
      bucket.expect_sparse_gradient = expect_sparse_gradients_[variable_index];
    } else {
      for (const auto variable_index : bucket_indices[bucket_index]) {
        REDUCER_CHECK(
            !expect_sparse_gradients_[variable_index],
            logger_,
            "Buckets with more than one variable cannot include variables ",
            "that expect a sparse gradient.");
      }
    }

    if (bucket.expect_sparse_gradient) {
      const auto variable_index = bucket_indices[bucket_index].front();
      const auto& variable = params_[variable_index];
      TORCH_INTERNAL_ASSERT(bucket_indices[bucket_index].size() == 1);
      bucket.variables = {variable};
    } else {
      at::TensorOptions options;
      // The start index of the variable in the flattened tensor.
      size_t offset = 0;

      // Reserve enough space for the per-variable fields stored in the bucket
      // for efficiency.
      const size_t num_variables = bucket_indices[bucket_index].size();
      bucket.variables.reserve(num_variables);
      bucket.offsets.reserve(num_variables);
      bucket.lengths.reserve(num_variables);
      bucket.sizes_vec.reserve(num_variables);

      // Iterate over bucket variables.
      for (const auto variable_index : bucket_indices[bucket_index]) {
        TORCH_INTERNAL_ASSERT(
            variable_index < params_.size(),
            "Out of range variable index specified.");
        const auto& variable = params_[variable_index];
        if (!options.has_device()) {
          options = options.device(variable.device());
        } else {
          REDUCER_CHECK(
              variable.device() == options.device(),
              logger_,
              "All parameters in a bucket must be ",
              "placed on the same device.");
        }
        if (!options.has_dtype()) {
          options = options.dtype(variable.dtype());
        } else {
          REDUCER_CHECK(
              variable.dtype() == options.dtype(),
              logger_,
              "All parameters in a bucket must have the same dtype.");
        }
        const auto length = variable.numel();
        bucket.variables.push_back(variable);
        bucket.offsets.push_back(offset);
        bucket.lengths.push_back(length);
        bucket.sizes_vec.push_back(variable.sizes());
        offset += length;
      }

      // Make gradient type in the reduced precision if mixed precision is
      // enabled. This ensures that the type is correct when e.g. rebuilding
      // buckets.
      if (mixed_precision_param_dtype_.has_value()) {
        options = options.dtype(mixed_precision_param_dtype_);
      }

      // Allocate the bucket's flattened `gradients` tensor.
      auto bucketSize = static_cast<long>(offset);
      // Check if we can use comm-optimized memory pool to allocate tensor
      c10::intrusive_ptr<Backend> backend = nullptr;
      // An environment variable to disable comm-optimized memory pool.
      // Default is 1 for now (disabled).
      // TODO: turn it on by default once we have more confidence on it.
      bool ddpDisableCommMem =
          (getCvarString({"DDP_DISABLE_COMM_MEM"}, "1") == "1");
      try {
        backend = process_group_->getDefaultBackend();
      } catch (...) {
        // Sometimes the backend type can be `UNDEFINED` rather than `NCCL` or
        // `GLOO`. In this case, we just fall back to the regular way of
        // creating tensor
        LOG(INFO)
            << "Reducer: default comm backend not found, skipping bucket memory optimization";
      }
      if (ddpDisableCommMem == 0 && backend != nullptr &&
          backend->supportsTensorAlloc(options.device().index())) {
        // Comm-optimized memory pool is available, use it to allocate tensor
        LOG(INFO)
            << "Reducer: found comm-optimized memory allocator, using it to create bucket";
        bucket.gradients = backend->allocateTensor(bucketSize, options);
      } else {
        // Plain creation of tensor
        LOG(INFO)
            << "Reducer: comm-optimized memory allocator not found, using regular one";
        bucket.gradients = at::empty({bucketSize}, options);
      }

      // Note:  "Gradient Layout Contract"
      //
      // Here, create views into the `gradients` tensor for each variable's
      // grad. Views serve as entry points to `copy_()` each grad's data in/out
      // of the flattened `gradients` tensor.
      //
      // Gradients may have dense memory but non-row-major-contiguous strides
      // (e.g. channels_last or channels_last_3d). For coalesced accesses
      // during copy_s, it's beneficial for each view's layout to match its
      // grad's layout.
      //
      // Specifically, we expect torch/csrc/autograd/functions/accumulate_grad.h
      // produces grads that obey the "Gradient Layout Contract":
      //   (1) if variable.is_non_overlapping_and_dense(), the stashed grad's
      //       strides match variable.
      //   (2) else, stashed grad is rowmajor contiguous.
      // and create views to match.
      //
      // If AccumulateGrad breaks the contract, and produces a grad with an
      // unexpected layout, performance will degrade due to poor memory access
      // patterns when copy_ing grad data in and out of its bucket view.
      // However, numerics remain correct, because the bucket view is the same
      // on either end of the raw allreduce.  bucket_view_in.copy(grad)
      // transposes
      // (+ densifies) to the bucket view's layout, the data is allreduced,
      // then grad.copy_(bucket_view_o
```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 46 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `std`, `Reducer`, `C10_DEFINE_TYPED_REGISTRY`, `c10d`

**Classes/Structs**: `CpuTimer`, `of`, `are`, `one`, `BucketKey`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/c10d/reducer.hpp`
- `torch/csrc/distributed/c10d/Utils.hpp`
- `torch/csrc/distributed/c10d/default_comm_hooks.hpp`
- `functional`
- `c10/core/DeviceGuard.h`
- `c10/core/ScalarType.h`
- `c10/core/StreamGuard.h`
- `c10/util/Exception.h`
- `c10/util/Logging.h`
- `c10/util/hash.h`
- `c10/util/irange.h`
- `torch/csrc/autograd/engine.h`
- `torch/csrc/autograd/function_hook.h`
- `torch/csrc/autograd/functions/accumulate_grad.h`
- `torch/csrc/autograd/profiler.h`
- `torch/csrc/autograd/utils/grad_layout_contract.h`
- `torch/csrc/autograd/utils/lambda_post_hook.h`
- `torch/csrc/distributed/c10d/comm.hpp`
- `torch/csrc/distributed/c10d/logger.hpp`
- `utility`


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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

- **File Documentation**: `reducer.cpp_docs.md`
- **Keyword Index**: `reducer.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
