# Documentation: `docs/torch/csrc/autograd/saved_variable.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/autograd/saved_variable.cpp_docs.md`
- **Size**: 13,867 bytes (13.54 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/autograd/saved_variable.cpp`

## File Metadata

- **Path**: `torch/csrc/autograd/saved_variable.cpp`
- **Size**: 11,208 bytes (10.95 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/autograd/saved_variable.h>

#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/variable.h>

#include <ATen/Tensor.h>

#include <memory>
#include <sstream>

namespace torch::autograd {

SavedVariable::SavedVariable(
    const Variable& variable,
    bool is_output,
    bool is_inplace_on_view) {
  if (variable.defined()) {
    // Note [Inference tensor cannot be saved for backward]
    // Invariant:
    //   You can't save an inference tensor for backwards.
    // If an inference tensor was saved for backward in an autograd session and
    // then you reenter inference mode and make an inplace update to the tensor
    // without bumping version_counter, it'll lead to silent wrong result when
    // you do backward() for the previous autograd session.  Technically we
    // don't have to check here since it'll fail when querying `current_version`
    // on the inference tensor, but we can give a much better error message
    // here.
    //
    // Note in the documentation we say "inference tensor cannot participate
    // in autograd" which is more restrictive than the invariant.  In practice
    // the check is more permissive and only error out when an inference tensor
    // is saved for backward.  Whether a tensor is saved for backward is
    // determined by derivative formula and thus varies op by op, so by saying
    // "no inference tensor in autograd" it's easier for users to understand and
    // follow.
    TORCH_CHECK(
        !variable.is_inference(),
        "Inference tensors cannot be saved for backward. Please do not use "
        "Tensors created in inference mode in computation tracked by autograd. "
        "To work around this, you can make a clone to get a normal tensor and "
        "use it in autograd, or use `torch.no_grad()` instead of "
        "`torch.inference_mode()`.");

    was_default_constructed_ = false;
    saved_version_ = variable._version();
    is_leaf_ = variable.is_leaf();
    is_output_ = is_output;
    is_inplace_on_view_ = is_inplace_on_view;

    if (is_inplace_on_view) {
      TORCH_INTERNAL_ASSERT(!is_leaf_ && is_output);
      weak_grad_fn_ = variable.grad_fn();
    }
    std::unique_ptr<SavedVariableHooks> maybe_hooks =
        at::SavedTensorDefaultHooks::is_enabled() ? get_default_hooks()
                                                  : nullptr;

    // Avoid wrapped numbers from being leaked to the user
    if (maybe_hooks && !variable.unsafeGetTensorImpl()->is_wrapped_number()) {
      save_metadata(variable);
      set_hooks_and_pack_data(std::move(maybe_hooks), variable);
      TORCH_INTERNAL_ASSERT(!data_.defined());
      return;
    }

    // If the variable is a leaf or is not an output, we can safely save the
    // original variable without running the risk of reference cycles.
    // 1. If the variable is not an output, its grad_fn has already been fully
    // created and in particular will be a different Node than the one
    // we are currently constructing (the one that owns this SavedVariable).
    // 2. If the variable is a leaf, it only has weak reference to the
    // grad_accumulator which cannot create a cycle. In those cases, we save the
    // original variable and don't need further processing.
    if (!is_output || is_leaf_) {
      saved_original_ = true;
      data_ = variable;
      return;
    }

    save_metadata(variable);

    // Only do this if we actually need to.
    data_ = variable.tensor_data();
  }
}

void SavedVariable::save_metadata(const Variable& data) {
  // Save output number, version counter and fw_grad if needed

  output_nr_ = data.output_nr();

  if (is_leaf_) {
    grad_accumulator_ = impl::grad_accumulator(data);
    requires_grad_ = data.requires_grad();
  } else if (!is_output_) {
    grad_fn_ = data.grad_fn();
  }

  // TODO(albanD) This needs to be updated when moving to multiple levels
  const auto& fw_grad = data._fw_grad(/* level */ 0);
  if (fw_grad.defined()) {
    fw_grad_ = std::make_shared<ForwardGrad>();
    fw_grad_->set_value(fw_grad, /* level */ 0);
  }
}

std::unique_ptr<SavedVariableHooks> SavedVariable::get_default_hooks() {
  return Engine::get_default_engine().get_default_saved_variable_hooks();
}

void SavedVariable::reset_data() {
  hooks_.reset();
  grad_fn_.reset();
  data_.reset();
}

SavedVariable::SavedVariable(
    const std::optional<Variable>& variable,
    bool is_output,
    bool is_inplace_on_view)
    : SavedVariable(
          variable.has_value() ? *variable : Variable(),
          is_output,
          is_inplace_on_view) {}

Variable SavedVariable::unpack(std::shared_ptr<Node> saved_for) const {
  if (was_default_constructed_) {
    return Variable();
  }

  if (!data_.defined()) {
    TORCH_CHECK(hooks_, ERR_BACKWARD_TWICE);
  }

  // We want grad_fn here to provide the most helpful debug message to the user
  // if versions don't match

  std::shared_ptr<Node> grad_fn;
  if (is_inplace_on_view_) {
    grad_fn = weak_grad_fn_.lock();
  } else if (!hooks_) {
    grad_fn = saved_original_ ? data_.grad_fn() : nullptr;
  } else {
    grad_fn = grad_fn_;
  }

  if (!is_leaf_ && !grad_fn) {
    // This issue was introduced when we added logic to save the original
    // because now we rely on data_.grad_fn(), but can be unreliable if the
    // autograd_meta of that saved tensor is cleared with an in-place detach.
    // As a simple fix, we choose to disallow that behavior here even though
    // it makes behavior inconsistent depending on whether you are saving
    // input or output.
    TORCH_CHECK(
        saved_for,
        "Trying to use a saved tensor that has been detached in-place, i.e. with .detach_()."
        "This is not supported, please use out-of-place `.detach()` instead");
    grad_fn = std::move(saved_for);
  }

  // Only check version counter in the case without hooks
  // If user provides hooks, we can't track versions through the hooks
  if (!hooks_) {
    auto current_version = impl::version_counter(data_).current_version();

    if (saved_version_ != current_version) {
      std::stringstream message;
      message
          << "one of the variables needed for gradient computation has been "
             "modified by an inplace operation: ["
          << data_.toString() << " ";
      if (data_.is_nested()) {
        message << data_._nested_tensor_size() << "]";
      } else {
        message << data_.sizes() << "]";
      }
      if (grad_fn) {
        message << ", which is output " << output_nr_ << " of "
                << grad_fn->name() << ",";
      }
      message << " is at version " << current_version << "; expected version "
              << saved_version_ << " instead.";
      if (!AnomalyMode::is_enabled()) {
        message << " Hint: enable anomaly detection to find the operation "
                   "that failed to compute its gradient, with torch.autograd."
                   "set_detect_anomaly(True).";
      } else {
        message
            << " Hint: the backtrace further above shows the operation "
               "that failed to compute its gradient. The variable in question "
               "was changed in there or anywhere later. Good luck!";
      }
      TORCH_CHECK(false, message.str());
    }
  }

  // The version counter is correct.
  // Additionally, if we deal with a non-leaf variable, we have its correct
  // grad_fn.

  // If we have the original variable, we simply return it
  if (!hooks_ && saved_original_) {
    return data_;
  }

  auto data = hooks_ ? hooks_->call_unpack_hook() : data_;

  if (!grad_fn && !requires_grad_ && !data.requires_grad() &&
      !(fw_grad_ && !fw_grad_->empty())) {
    // Avoid detaching if we don't need to.
    return data;
  }

  // NB: saved views are unpacked as normal Variables (not views) even though
  // they still share the same storage. This works only because we never call
  // in-place functions on unpacked variables.
  Variable var;
  if (grad_fn) {
    var = make_variable(data, Edge(std::move(grad_fn), output_nr_));
  } else {
    var = make_variable(data, requires_grad_);
  }

  impl::set_grad_accumulator(var, grad_accumulator_);
  impl::set_version_counter(var, impl::version_counter(data));

  // NB: var here is never a view so there is no need to make anything special
  // for the case where the saved Tensor was a view. This whole argument relies
  // on the fact that the Tensor returned by this function is never
  // modified in-place.
  if (fw_grad_ && !fw_grad_->empty()) {
    // TODO(albanD) This needs to be updated when moving to multiple levels
    auto new_fw_grad = fw_grad_->value(/* level */ 0);
    var._set_fw_grad(new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
  }

  return var;
}

void SavedVariable::set_hooks_and_pack_data(
    std::unique_ptr<SavedVariableHooks>&& hooks,
    const Variable& data) {
  hooks_ = std::move(hooks);
  at::NoGradGuard guard;
  const auto version = impl::version_counter(data).current_version();
  hooks_->call_pack_hook(saved_original_ ? data.detach() : data);
  TORCH_CHECK(
      version == impl::version_counter(data).current_version(),
      "A saved tensor pack hook is modifying its input in place. "
      "Tensors provided as input to pack hook can not be modified by "
      "in-place operations as this can lead to unexpected side-effects. "
      "Please open an issue if you need to perform in-place operations on "
      "the input to a pack hook.");
}

void SavedVariable::register_hooks(
    std::unique_ptr<SavedVariableHooks>&& hooks) {
  TORCH_INTERNAL_ASSERT(hooks);
  TORCH_CHECK(
      !hooks_,
      "Calling register_hooks on a saved tensor whose hooks have already been set. "
      "Hint: only one pair of hooks is allowed at a time.");
  if (!data_.defined()) {
    if (!was_default_constructed_) {
      TORCH_CHECK(
          false,
          "Calling register_hooks on a saved tensor after it has been freed. "
          "Saved intermediate values of the graph are freed when you call "
          ".backward() or autograd.grad(). Specify retain_graph=True if you "
          "need to backward through the graph a second time or if you need to "
          "access saved variables after calling backward.");
    } else {
      TORCH_CHECK(
          false,
          "Calling register_hooks on a saved tensor with value None is forbidden");
    }
  }
  // If we didn't save the original variable, we already saved metadata
  if (saved_original_) {
    save_metadata(data_);
  }
  set_hooks_and_pack_data(std::move(hooks), data_);
  data_.reset();
}

const char* ERR_BACKWARD_TWICE =
    "Trying to backward through the graph a second time (or directly access saved "
    "tensors after they have already been freed). Saved intermediate values "
    "of the graph are freed when you call .backward() or autograd.grad(). Specify "
    "retain_graph=True if you need to backward through the graph a second time or "
    "if you need to access saved tensors after calling backward.";

} // namespace torch::autograd

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/autograd/saved_variable.h`
- `torch/csrc/autograd/anomaly_mode.h`
- `torch/csrc/autograd/edge.h`
- `torch/csrc/autograd/engine.h`
- `torch/csrc/autograd/function.h`
- `torch/csrc/autograd/grad_mode.h`
- `torch/csrc/autograd/variable.h`
- `ATen/Tensor.h`
- `memory`
- `sstream`


## Code Patterns & Idioms

### Common Patterns

- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes


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

Files in the same folder (`torch/csrc/autograd`):

- [`graph_task.h_docs.md`](./graph_task.h_docs.md)
- [`python_function.cpp_docs.md`](./python_function.cpp_docs.md)
- [`profiler.h_docs.md`](./profiler.h_docs.md)
- [`TraceTypeManual.cpp_docs.md`](./TraceTypeManual.cpp_docs.md)
- [`python_autograd.h_docs.md`](./python_autograd.h_docs.md)
- [`variable_info.cpp_docs.md`](./variable_info.cpp_docs.md)
- [`jit_decomp_interface.h_docs.md`](./jit_decomp_interface.h_docs.md)
- [`input_buffer.cpp_docs.md`](./input_buffer.cpp_docs.md)
- [`python_variable.h_docs.md`](./python_variable.h_docs.md)
- [`python_nn_functions.h_docs.md`](./python_nn_functions.h_docs.md)


## Cross-References

- **File Documentation**: `saved_variable.cpp_docs.md`
- **Keyword Index**: `saved_variable.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/autograd`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/autograd`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Automatic Differentiation**: Uses autograd for gradient computation


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/autograd`):

- [`python_cpp_function.h_kw.md_docs.md`](./python_cpp_function.h_kw.md_docs.md)
- [`anomaly_mode.cpp_kw.md_docs.md`](./anomaly_mode.cpp_kw.md_docs.md)
- [`python_nested_functions_manual.cpp_kw.md_docs.md`](./python_nested_functions_manual.cpp_kw.md_docs.md)
- [`variable_info.h_docs.md_docs.md`](./variable_info.h_docs.md_docs.md)
- [`python_nn_functions.h_docs.md_docs.md`](./python_nn_functions.h_docs.md_docs.md)
- [`python_cpp_function.h_docs.md_docs.md`](./python_cpp_function.h_docs.md_docs.md)
- [`profiler_legacy.cpp_kw.md_docs.md`](./profiler_legacy.cpp_kw.md_docs.md)
- [`python_fft_functions.h_docs.md_docs.md`](./python_fft_functions.h_docs.md_docs.md)
- [`python_autograd.h_kw.md_docs.md`](./python_autograd.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `saved_variable.cpp_docs.md_docs.md`
- **Keyword Index**: `saved_variable.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
