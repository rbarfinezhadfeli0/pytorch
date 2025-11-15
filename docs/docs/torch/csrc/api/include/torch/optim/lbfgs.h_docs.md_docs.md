# Documentation: `docs/torch/csrc/api/include/torch/optim/lbfgs.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/include/torch/optim/lbfgs.h_docs.md`
- **Size**: 5,767 bytes (5.63 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/include/torch/optim/lbfgs.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/optim/lbfgs.h`
- **Size**: 3,430 bytes (3.35 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/nn/module.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <torch/serialize/archive.h>

#include <deque>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace torch::optim {

struct TORCH_API LBFGSOptions : public OptimizerCloneableOptions<LBFGSOptions> {
  LBFGSOptions(double lr = 1);
  TORCH_ARG(double, lr) = 1;
  TORCH_ARG(int64_t, max_iter) = 20;
  TORCH_ARG(std::optional<int64_t>, max_eval) = std::nullopt;
  TORCH_ARG(double, tolerance_grad) = 1e-7;
  TORCH_ARG(double, tolerance_change) = 1e-9;
  TORCH_ARG(int64_t, history_size) = 100;
  TORCH_ARG(std::optional<std::string>, line_search_fn) = std::nullopt;

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const LBFGSOptions& lhs,
      const LBFGSOptions& rhs);
  double get_lr() const override;
  void set_lr(const double lr) override;
};

struct TORCH_API LBFGSParamState
    : public OptimizerCloneableParamState<LBFGSParamState> {
  TORCH_ARG(int64_t, func_evals) = 0;
  TORCH_ARG(int64_t, n_iter) = 0;
  TORCH_ARG(double, t) = 0;
  TORCH_ARG(double, prev_loss) = 0;
  TORCH_ARG(Tensor, d);
  TORCH_ARG(Tensor, H_diag);
  TORCH_ARG(Tensor, prev_flat_grad);
  TORCH_ARG(std::deque<Tensor>, old_dirs);
  TORCH_ARG(std::deque<Tensor>, old_stps);
  TORCH_ARG(std::deque<Tensor>, ro);
  TORCH_ARG(std::optional<std::vector<Tensor>>, al) = std::nullopt;

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const LBFGSParamState& lhs,
      const LBFGSParamState& rhs);
};

class TORCH_API LBFGS : public Optimizer {
 public:
  explicit LBFGS(
      const std::vector<OptimizerParamGroup>& param_groups,
      LBFGSOptions defaults = {})
      : Optimizer(param_groups, std::make_unique<LBFGSOptions>(defaults)) {
    TORCH_CHECK(
        param_groups_.size() == 1,
        "LBFGS doesn't support per-parameter options (parameter groups)");
    if (defaults.max_eval() == std::nullopt) {
      auto max_eval_val = (defaults.max_iter() * 5) / 4;
      static_cast<LBFGSOptions&>(param_groups_[0].options())
          .max_eval(max_eval_val);
      static_cast<LBFGSOptions&>(*defaults_).max_eval(max_eval_val);
    }
    _numel_cache = std::nullopt;
  }
  explicit LBFGS(std::vector<Tensor> params, LBFGSOptions defaults = {})
      : LBFGS({OptimizerParamGroup(std::move(params))}, std::move(defaults)) {}

  Tensor step(LossClosure closure) override;
  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  std::optional<int64_t> _numel_cache;
  int64_t _numel();
  Tensor _gather_flat_grad();
  void _add_grad(const double step_size, const Tensor& update);
  std::tuple<double, Tensor> _directional_evaluate(
      const LossClosure& closure,
      const std::vector<Tensor>& x,
      double t,
      const Tensor& d);
  void _set_param(const std::vector<Tensor>& params_data);
  std::vector<Tensor> _clone_param();

  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(LBFGS);
  }
};
} // namespace torch::optim

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 17 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`, `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/optim`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/nn/module.h`
- `torch/optim/optimizer.h`
- `torch/optim/serialize.h`
- `torch/serialize/archive.h`
- `deque`
- `functional`
- `memory`
- `utility`
- `vector`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/api/include/torch/optim`):

- [`optimizer.h_docs.md`](./optimizer.h_docs.md)
- [`adamw.h_docs.md`](./adamw.h_docs.md)
- [`serialize.h_docs.md`](./serialize.h_docs.md)
- [`adam.h_docs.md`](./adam.h_docs.md)
- [`sgd.h_docs.md`](./sgd.h_docs.md)
- [`adagrad.h_docs.md`](./adagrad.h_docs.md)
- [`rmsprop.h_docs.md`](./rmsprop.h_docs.md)


## Cross-References

- **File Documentation**: `lbfgs.h_docs.md`
- **Keyword Index**: `lbfgs.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/api/include/torch/optim`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/api/include/torch/optim`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/api/include/torch/optim`):

- [`serialize.h_docs.md_docs.md`](./serialize.h_docs.md_docs.md)
- [`rmsprop.h_kw.md_docs.md`](./rmsprop.h_kw.md_docs.md)
- [`adagrad.h_kw.md_docs.md`](./adagrad.h_kw.md_docs.md)
- [`sgd.h_docs.md_docs.md`](./sgd.h_docs.md_docs.md)
- [`lbfgs.h_kw.md_docs.md`](./lbfgs.h_kw.md_docs.md)
- [`sgd.h_kw.md_docs.md`](./sgd.h_kw.md_docs.md)
- [`serialize.h_kw.md_docs.md`](./serialize.h_kw.md_docs.md)
- [`adam.h_kw.md_docs.md`](./adam.h_kw.md_docs.md)
- [`adamw.h_docs.md_docs.md`](./adamw.h_docs.md_docs.md)
- [`rmsprop.h_docs.md_docs.md`](./rmsprop.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `lbfgs.h_docs.md_docs.md`
- **Keyword Index**: `lbfgs.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
