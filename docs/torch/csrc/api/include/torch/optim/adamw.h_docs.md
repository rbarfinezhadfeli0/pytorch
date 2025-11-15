# Documentation: `torch/csrc/api/include/torch/optim/adamw.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/optim/adamw.h`
- **Size**: 2,895 bytes (2.83 KB)
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

#include <utility>
#include <vector>

namespace torch::serialize {
class OutputArchive;
class InputArchive;
} // namespace torch::serialize

namespace torch::optim {

struct TORCH_API AdamWOptions : public OptimizerCloneableOptions<AdamWOptions> {
  AdamWOptions(double lr = 1e-3);
  TORCH_ARG(double, lr) = 1e-3;
  typedef std::tuple<double, double> betas_t;
  TORCH_ARG(betas_t, betas) = std::make_tuple(0.9, 0.999);
  TORCH_ARG(double, eps) = 1e-8;
  TORCH_ARG(double, weight_decay) = 1e-2;
  TORCH_ARG(bool, amsgrad) = false;

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const AdamWOptions& lhs,
      const AdamWOptions& rhs);
  double get_lr() const override;
  void set_lr(const double lr) override;
};

struct TORCH_API AdamWParamState
    : public OptimizerCloneableParamState<AdamWParamState> {
  TORCH_ARG(int64_t, step) = 0;
  TORCH_ARG(torch::Tensor, exp_avg);
  TORCH_ARG(torch::Tensor, exp_avg_sq);
  TORCH_ARG(torch::Tensor, max_exp_avg_sq);

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const AdamWParamState& lhs,
      const AdamWParamState& rhs);
};

class TORCH_API AdamW : public Optimizer {
 public:
  explicit AdamW(
      const std::vector<OptimizerParamGroup>& param_groups,
      AdamWOptions defaults = {})
      : Optimizer(param_groups, std::make_unique<AdamWOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());
    auto betas = defaults.betas();
    TORCH_CHECK(
        0 <= std::get<0>(betas) && std::get<0>(betas) < 1.0,
        "Invalid beta parameter at index 0: ",
        std::get<0>(betas));
    TORCH_CHECK(
        0 <= std::get<1>(betas) && std::get<1>(betas) < 1.0,
        "Invalid beta parameter at index 1: ",
        std::get<1>(betas));
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
  }
  explicit AdamW(std::vector<Tensor> params, AdamWOptions defaults = {})
      : AdamW({OptimizerParamGroup(std::move(params))}, std::move(defaults)) {}

  torch::Tensor step(LossClosure closure = nullptr) override;
  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(AdamW);
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

- `torch/nn/module.h`
- `torch/optim/optimizer.h`
- `torch/optim/serialize.h`
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
- [`lbfgs.h_docs.md`](./lbfgs.h_docs.md)
- [`serialize.h_docs.md`](./serialize.h_docs.md)
- [`adam.h_docs.md`](./adam.h_docs.md)
- [`sgd.h_docs.md`](./sgd.h_docs.md)
- [`adagrad.h_docs.md`](./adagrad.h_docs.md)
- [`rmsprop.h_docs.md`](./rmsprop.h_docs.md)


## Cross-References

- **File Documentation**: `adamw.h_docs.md`
- **Keyword Index**: `adamw.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
