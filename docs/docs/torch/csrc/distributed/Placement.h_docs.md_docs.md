# Documentation: `docs/torch/csrc/distributed/Placement.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/Placement.h_docs.md`
- **Size**: 4,865 bytes (4.75 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/Placement.h`

## File Metadata

- **Path**: `torch/csrc/distributed/Placement.h`
- **Size**: 2,886 bytes (2.82 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
/**
 * The implementations in this file are coupled with
 * torch/distributed/tensor/placement_types.py.
 */

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace torch::distributed {

class Placement {
 public:
  Placement() = default;
  virtual ~Placement() = default;

  Placement(const Placement&) = default;
  Placement& operator=(const Placement&) = default;
  Placement(Placement&&) noexcept = default;
  Placement& operator=(Placement&&) noexcept = default;

  virtual bool is_shard(std::optional<std::int64_t> dim) const {
    return false;
  }

  virtual bool is_replicate() const {
    return false;
  }

  virtual bool is_partial(
      std::optional<std::string_view> reduce_op = std::nullopt) const {
    return false;
  }
};

class Shard : public Placement {
 public:
  std::int64_t dim;
  explicit Shard(std::int64_t dim_) : dim(dim_) {}

  bool is_shard(std::optional<std::int64_t> dim_) const override {
    return !dim_.has_value() || *dim_ == dim;
  }

  bool operator==(const Shard& rhs) const {
    return dim == rhs.dim;
  }

  bool operator!=(const Shard& rhs) const {
    return !operator==(rhs);
  }
};

class StridedShard : public Shard {
 public:
  std::int64_t split_factor;
  explicit StridedShard(std::int64_t dim, std::int64_t split_factor_)
      : Shard(dim), split_factor(split_factor_) {}

  bool operator==(const StridedShard& rhs) const {
    return dim == rhs.dim && split_factor == rhs.split_factor;
  }

  bool operator==(const Shard& rhs) const {
    if (auto* rhs_strided = dynamic_cast<const StridedShard*>(&rhs)) {
      return operator==(*rhs_strided);
    }
    // TODO: this is to avoid extra all-gather in dtensor op dispatch
    // note that sharding prop would not produce _StridedShard and a
    // placement inequality would introduce an all-gather for resharding
    return dim == rhs.dim;
  }

  bool operator!=(const Shard& rhs) const {
    return !operator==(rhs);
  }
};

class Replicate : public Placement {
 public:
  bool is_replicate() const override {
    return true;
  }

  bool operator==(const Replicate& rhs) const {
    return true;
  }

  bool operator!=(const Replicate& rhs) const {
    return false;
  }
};

class Partial : public Placement {
 public:
  std::string reduce_op;

  Partial() : Partial("sum") {}

  explicit Partial(std::optional<std::string> reduce_op_)
      : reduce_op(
            reduce_op_.has_value() ? std::move(*reduce_op_)
                                   : std::string("sum")) {}

  bool is_partial(
      std::optional<std::string_view> op = std::nullopt) const override {
    return !op.has_value() || *op == reduce_op;
  }

  bool operator==(const Partial& rhs) const {
    return reduce_op == rhs.reduce_op;
  }

  bool operator!=(const Partial& rhs) const {
    return !operator==(rhs);
  }
};

} // namespace torch::distributed

```



## High-Level Overview


This C++ file contains approximately 5 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `Placement`, `Shard`, `StridedShard`, `Replicate`, `Partial`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `cstdint`
- `optional`
- `string`
- `string_view`


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

Files in the same folder (`torch/csrc/distributed`):

- [`python_placement.cpp_docs.md`](./python_placement.cpp_docs.md)
- [`python_placement.h_docs.md`](./python_placement.h_docs.md)


## Cross-References

- **File Documentation**: `Placement.h_docs.md`
- **Keyword Index**: `Placement.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/distributed`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/distributed`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/distributed`):

- [`Placement.h_kw.md_docs.md`](./Placement.h_kw.md_docs.md)
- [`python_placement.cpp_kw.md_docs.md`](./python_placement.cpp_kw.md_docs.md)
- [`python_placement.cpp_docs.md_docs.md`](./python_placement.cpp_docs.md_docs.md)
- [`python_placement.h_docs.md_docs.md`](./python_placement.h_docs.md_docs.md)
- [`python_placement.h_kw.md_docs.md`](./python_placement.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `Placement.h_docs.md_docs.md`
- **Keyword Index**: `Placement.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
