# Documentation: `torch/csrc/api/include/torch/optim/adagrad.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/optim/adagrad.h`
- **Size**: 3,197 bytes (3.12 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <torch/nn/pimpl.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <torch/serialize/archive.h>
#include <torch/types.h>

#include <utility>
#include <vector>

namespace torch::serialize {
class OutputArchive;
class InputArchive;
} // namespace torch::serialize

namespace torch::optim {

struct TORCH_API AdagradOptions
    : public OptimizerCloneableOptions<AdagradOptions> {
  AdagradOptions(double lr = 1e-2);
  TORCH_ARG(double, lr) = 1e-2;
  TORCH_ARG(double, lr_decay) = 0;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(double, initial_accumulator_value) = 0;
  TORCH_ARG(double, eps) = 1e-10;

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const AdagradOptions& lhs,
      const AdagradOptions& rhs);
  double get_lr() const override;
  void set_lr(const double lr) override;
};

struct TORCH_API AdagradParamState
    : public OptimizerCloneableParamState<AdagradParamState> {
  TORCH_ARG(torch::Tensor, sum);
  TORCH_ARG(int64_t, step) = 0;

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const AdagradParamState& lhs,
      const AdagradParamState& rhs);
};

class TORCH_API Adagrad : public Optimizer {
 public:
  explicit Adagrad(
      const std::vector<OptimizerParamGroup>& param_groups,
      AdagradOptions defaults = {})
      : Optimizer(param_groups, std::make_unique<AdagradOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(
        defaults.lr_decay() >= 0,
        "Invalid lr_decay value: ",
        defaults.lr_decay());
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
    TORCH_CHECK(
        defaults.initial_accumulator_value() >= 0,
        "Invalid initial_accumulator_value value: ",
        defaults.initial_accumulator_value());
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());

    for (const auto& group : param_groups_) {
      for (const auto& p : group.params()) {
        auto state = std::make_unique<AdagradParamState>();
        state->step(0);
        state->sum(torch::full_like(
            p.data(),
            defaults.initial_accumulator_value(),
            at::MemoryFormat::Preserve));
        state_[p.unsafeGetTensorImpl()] = std::move(state);
      }
    }
  }

  explicit Adagrad(std::vector<Tensor> params, AdagradOptions defaults = {})
      : Adagrad({OptimizerParamGroup(std::move(params))}, std::move(defaults)) {
  }

  torch::Tensor step(LossClosure closure = nullptr) override;
  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(Adagrad);
  }
};
} // namespace torch::optim

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `OutputArchive`, `InputArchive`, `TORCH_API`, `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/optim`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/nn/pimpl.h`
- `torch/optim/optimizer.h`
- `torch/optim/serialize.h`
- `torch/serialize/archive.h`
- `torch/types.h`
- `utility`
- `vector`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/csrc/api/include/torch/optim`):

- [`optimizer.h_docs.md`](./optimizer.h_docs.md)
- [`adamw.h_docs.md`](./adamw.h_docs.md)
- [`lbfgs.h_docs.md`](./lbfgs.h_docs.md)
- [`serialize.h_docs.md`](./serialize.h_docs.md)
- [`adam.h_docs.md`](./adam.h_docs.md)
- [`sgd.h_docs.md`](./sgd.h_docs.md)
- [`rmsprop.h_docs.md`](./rmsprop.h_docs.md)


## Cross-References

- **File Documentation**: `adagrad.h_docs.md`
- **Keyword Index**: `adagrad.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
