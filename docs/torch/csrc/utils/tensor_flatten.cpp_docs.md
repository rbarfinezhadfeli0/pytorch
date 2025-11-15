# Documentation: `torch/csrc/utils/tensor_flatten.cpp`

## File Metadata

- **Path**: `torch/csrc/utils/tensor_flatten.cpp`
- **Size**: 3,821 bytes (3.73 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/utils/tensor_flatten.h>

#include <map>
#include <unordered_map>

namespace torch::utils {

using namespace at;

std::vector<TensorGroup> take_tensors(
    TensorList tensors,
    size_t size_limit,
    bool fine_grained) {
  std::vector<TensorGroup> results;
  // an overapproximation, but at least we won't have to copy stuff around
  results.reserve(tensors.size());
  std::map<int64_t, TensorGroup> groups;
  size_t cur_group_size = 0;

  for (const auto& tensor : tensors) {
    size_t tensor_size = 0;
    if (tensor.is_sparse()) {
      const auto& indices = tensor._indices();
      const auto& values = tensor._values();
      tensor_size = indices.numel() * indices.element_size() +
          values.numel() * indices.element_size();
    } else {
      tensor_size = tensor.numel() * tensor.element_size();
    }

    auto& type_group = groups[static_cast<int64_t>(type_id(tensor))];
    type_group.tensors.push_back(tensor);

    if (fine_grained) {
      cur_group_size += tensor_size;
      // Regardless the type, the current total size exceeds the limit
      if (cur_group_size >= size_limit) {
        // Spill all types to separate groups in results
        for (auto& entry : groups) {
          auto& group = entry.second;
          results.emplace_back(std::move(group));
        }
        cur_group_size = 0;
        groups.clear();
      }
    } else {
      type_group.size += tensor_size;
      if (type_group.size >= size_limit) {
        results.emplace_back();
        std::swap(results.back(), type_group);
      }
    }
  }
  // End case. Look for any remaining groups and return them.
  for (auto& entry : groups) {
    auto& group = entry.second;
    if (group.tensors.empty()) {
      continue;
    }
    results.emplace_back(std::move(group));
  }
  return results;
}

void reorder_tensors_like(std::vector<Tensor>& tensors, TensorList order) {
  AT_ASSERT(tensors.size() == order.size());
  std::unordered_map<size_t, std::vector<size_t>> type_id_to_indices;
  for (size_t i = 0, num_tensors = tensors.size(); i < num_tensors; ++i)
    type_id_to_indices[type_id(tensors[i])].push_back(i);

  std::unordered_map<size_t, size_t> type_id_to_type_used;
  std::vector<Tensor> ordered_tensors;
  ordered_tensors.reserve(tensors.size());
  for (auto& tmpl_tensor : order) {
    size_t tmpl_type_id = type_id(tmpl_tensor);
    auto& indices = type_id_to_indices[tmpl_type_id];
    auto& used = type_id_to_type_used[tmpl_type_id];
    ordered_tensors.push_back(tensors[indices[used++]]);
  }
  std::swap(tensors, ordered_tensors);
}

namespace {

at::Tensor get_indices(const at::Tensor& t) {
  return t._indices();
}

at::Tensor get_values(const at::Tensor& t) {
  return t._values();
}

} // namespace

std::pair<at::Tensor, at::Tensor> flatten_sparse_tensors(
    at::TensorList tensors) {
  auto flat_indices = utils::flatten_dense_tensors(fmap(tensors, &get_indices));
  auto flat_values = utils::flatten_dense_tensors(fmap(tensors, &get_values));
  return std::make_pair(flat_indices, flat_values);
}

std::vector<at::Tensor> unflatten_sparse_tensors(
    const at::Tensor& flat_indices,
    const at::Tensor& flat_values,
    at::TensorList tensors) {
  if (tensors.empty())
    return {};

  auto indices =
      utils::unflatten_dense_tensors(flat_indices, fmap(tensors, &get_indices));
  auto values =
      utils::unflatten_dense_tensors(flat_values, fmap(tensors, &get_values));

  std::vector<at::Tensor> outputs;
  outputs.reserve(tensors.size());
  for (size_t i = 0, num_tensors = tensors.size(); i < num_tensors; ++i) {
    auto& ref_t = tensors[i];
    auto t =
        at::_sparse_coo_tensor_unsafe(indices[i], values[i], ref_t.sizes());
    outputs.emplace_back(t._coalesced_(ref_t.is_coalesced()));
  }
  return outputs;
}

} // namespace torch::utils

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `std`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/utils/tensor_flatten.h`
- `map`
- `unordered_map`


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

Files in the same folder (`torch/csrc/utils`):

- [`tensor_list.h_docs.md`](./tensor_list.h_docs.md)
- [`disable_torch_function.cpp_docs.md`](./disable_torch_function.cpp_docs.md)
- [`tensor_new.cpp_docs.md`](./tensor_new.cpp_docs.md)
- [`tensor_apply.cpp_docs.md`](./tensor_apply.cpp_docs.md)
- [`cpp_stacktraces.cpp_docs.md`](./cpp_stacktraces.cpp_docs.md)
- [`numpy_stub.h_docs.md`](./numpy_stub.h_docs.md)
- [`nested.h_docs.md`](./nested.h_docs.md)
- [`nested.cpp_docs.md`](./nested.cpp_docs.md)
- [`six.h_docs.md`](./six.h_docs.md)
- [`python_scalars.h_docs.md`](./python_scalars.h_docs.md)


## Cross-References

- **File Documentation**: `tensor_flatten.cpp_docs.md`
- **Keyword Index**: `tensor_flatten.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
