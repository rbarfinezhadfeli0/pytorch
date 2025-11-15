# Documentation: `docs/torch/csrc/api/include/torch/optim/sgd.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/include/torch/optim/sgd.h_docs.md`
- **Size**: 4,924 bytes (4.81 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/include/torch/optim/sgd.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/optim/sgd.h`
- **Size**: 2,622 bytes (2.56 KB)
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
#include <torch/types.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace torch::serialize {
class OutputArchive;
class InputArchive;
} // namespace torch::serialize

namespace torch::optim {

struct TORCH_API SGDOptions : public OptimizerCloneableOptions<SGDOptions> {
  SGDOptions(double lr);
  TORCH_ARG(double, lr);
  TORCH_ARG(double, momentum) = 0;
  TORCH_ARG(double, dampening) = 0;
  TORCH_ARG(double, weight_decay) = 0;
  TORCH_ARG(bool, nesterov) = false;

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const SGDOptions& lhs,
      const SGDOptions& rhs);
  double get_lr() const override;
  void set_lr(const double lr) override;
};

struct TORCH_API SGDParamState
    : public OptimizerCloneableParamState<SGDParamState> {
  TORCH_ARG(torch::Tensor, momentum_buffer);

 public:
  void serialize(torch::serialize::InputArchive& archive) override;
  void serialize(torch::serialize::OutputArchive& archive) const override;
  TORCH_API friend bool operator==(
      const SGDParamState& lhs,
      const SGDParamState& rhs);
};

class TORCH_API SGD : public Optimizer {
 public:
  explicit SGD(
      const std::vector<OptimizerParamGroup>& param_groups,
      SGDOptions defaults)
      : Optimizer(param_groups, std::make_unique<SGDOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());
    TORCH_CHECK(
        defaults.momentum() >= 0,
        "Invalid momentum value: ",
        defaults.momentum());
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
    TORCH_CHECK(
        !defaults.nesterov() ||
            (defaults.momentum() > 0 && defaults.dampening() == 0),
        "Nesterov momentum requires a momentum and zero dampening");
  }

  explicit SGD(std::vector<Tensor> params, SGDOptions defaults)
      : SGD({OptimizerParamGroup(std::move(params))}, std::move(defaults)) {}

  torch::Tensor step(LossClosure closure = nullptr) override;

  void save(serialize::OutputArchive& archive) const override;
  void load(serialize::InputArchive& archive) override;

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(SGD);
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
- `torch/serialize/archive.h`
- `torch/types.h`
- `cstddef`
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
- [`adagrad.h_docs.md`](./adagrad.h_docs.md)
- [`rmsprop.h_docs.md`](./rmsprop.h_docs.md)


## Cross-References

- **File Documentation**: `sgd.h_docs.md`
- **Keyword Index**: `sgd.h_kw.md`
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

Files in the same folder (`docs/torch/csrc/api/include/torch/optim`):

- [`serialize.h_docs.md_docs.md`](./serialize.h_docs.md_docs.md)
- [`rmsprop.h_kw.md_docs.md`](./rmsprop.h_kw.md_docs.md)
- [`adagrad.h_kw.md_docs.md`](./adagrad.h_kw.md_docs.md)
- [`lbfgs.h_kw.md_docs.md`](./lbfgs.h_kw.md_docs.md)
- [`sgd.h_kw.md_docs.md`](./sgd.h_kw.md_docs.md)
- [`serialize.h_kw.md_docs.md`](./serialize.h_kw.md_docs.md)
- [`adam.h_kw.md_docs.md`](./adam.h_kw.md_docs.md)
- [`adamw.h_docs.md_docs.md`](./adamw.h_docs.md_docs.md)
- [`rmsprop.h_docs.md_docs.md`](./rmsprop.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `sgd.h_docs.md_docs.md`
- **Keyword Index**: `sgd.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
