# Documentation: `docs/torch/csrc/lazy/core/ir_dump_util.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/lazy/core/ir_dump_util.cpp_docs.md`
- **Size**: 9,990 bytes (9.76 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/lazy/core/ir_dump_util.cpp`

## File Metadata

- **Path**: `torch/csrc/lazy/core/ir_dump_util.cpp`
- **Size**: 7,475 bytes (7.30 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/lazy/core/ir_dump_util.h>

#include <c10/util/irange.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/backend/lowering_context.h>
#include <torch/csrc/lazy/core/ir_util.h>
#include <optional>

#include <regex>
#include <sstream>
#include <unordered_map>

namespace torch::lazy {
namespace {

using NodeIdMap = std::unordered_map<const Node*, size_t>;

struct AttrTag {
  std::string name;
  std::string value;
  std::string::size_type pos = 0;
};

std::string::size_type SkipTagSeparator(
    const std::string& node_string,
    std::string::size_type pos) {
  return node_string.compare(pos, 2, ", ") == 0 ? pos + 2 : pos;
}

std::optional<AttrTag> ParseAttrTag(
    const std::string& node_string,
    std::string::size_type pos) {
  // @lint-ignore-every CLANGTIDY facebook-hte-StdRegexIsAwful
  const std::regex tag_regex("^([a-zA-Z0-9_]+)=");
  std::smatch match;
  // @lint-ignore-every CLANGTIDY facebook-hte-StdRegexIsAwful
  if (!std::regex_search(
          node_string.begin() + static_cast<std::ptrdiff_t>(pos),
          node_string.end(),
          match,
          tag_regex)) {
    return std::nullopt;
  }

  std::string::size_type vpos = match[1].second - node_string.begin() + 1;
  std::optional<char> nested_open;
  std::optional<char> nested_close;
  size_t nest_count = 1;
  AttrTag tag;
  tag.name = match[1].str();
  for (pos = vpos; pos < node_string.size(); ++pos) {
    if (!nested_open.has_value()) {
      if (SkipTagSeparator(node_string, pos) != pos) {
        break;
      }
      // NOLINTNEXTLINE(bugprone-switch-missing-default-case)
      switch (node_string[pos]) {
        case '(':
          nested_open = node_string[pos];
          nested_close = ')';
          break;
        case '[':
          nested_open = node_string[pos];
          nested_close = ']';
          break;
        case '{':
          nested_open = node_string[pos];
          nested_close = '}';
          break;
      }
    } else if (node_string[pos] == nested_close) {
      --nest_count;
      if (nest_count == 0) {
        nest_count = 1;
        nested_open.reset();
        nested_close.reset();
      }
    } else if (node_string[pos] == nested_open) {
      ++nest_count;
    }
  }
  tag.value = node_string.substr(vpos, pos - vpos);
  tag.pos = pos;
  return tag;
}

NodeIdMap GenerateIdMap(c10::ArrayRef<const Node*> post_order) {
  NodeIdMap id_map;
  for (auto node : post_order) {
    TORCH_CHECK(id_map.emplace(node, id_map.size()).second, node->ToString());
  }
  return id_map;
}

std::unordered_map<const Node*, size_t> GetRootsIds(
    c10::ArrayRef<const Node*> roots) {
  std::unordered_map<const Node*, size_t> roots_ids;
  for (const auto i : c10::irange(roots.size())) {
    roots_ids[roots[i]] = i;
  }
  return roots_ids;
}

std::optional<size_t> GetRootNodeId(
    const Node* node,
    const std::unordered_map<const Node*, size_t>& roots_ids) {
  auto it = roots_ids.find(node);
  if (it == roots_ids.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::vector<AttrTag> GetNodeTags(const Node* node) {
  std::string node_string = node->ToString();
  std::string op_string = node->op().ToString();
  std::string::size_type pos = node_string.find(op_string);
  TORCH_CHECK(pos != std::string::npos, node_string, " : ", op_string);
  pos += op_string.size();
  std::vector<AttrTag> tags;
  for (;;) {
    pos = SkipTagSeparator(node_string, pos);
    auto tag = ParseAttrTag(node_string, pos);
    if (!tag) {
      break;
    }
    pos = tag->pos;
    tags.push_back(std::move(*tag));
  }
  return tags;
}

std::string GenerateDotNodeLabel(
    const Node* node,
    const std::unordered_map<const Node*, size_t>& roots_ids) {
  static const size_t kMaxValueSize = 64;
  std::stringstream ss;
  ss << node->op() << "\\n" << node->shape();
  for (auto& tag : GetNodeTags(node)) {
    ss << "\\n" << tag.name << "=";
    if (tag.value.size() < kMaxValueSize) {
      ss << tag.value;
    } else {
      ss << tag.value.substr(0, kMaxValueSize) << "...";
    }
  }
  auto opt_root_id = GetRootNodeId(node, roots_ids);
  if (opt_root_id) {
    ss << "\\nROOT=" << *opt_root_id;
  }
  return ss.str();
}

std::string GenerateDotNodeSpec(
    const Node* node,
    const std::unordered_map<const Node*, size_t>& roots_ids) {
  std::stringstream ss;
  ss << "label=\"" << GenerateDotNodeLabel(node, roots_ids) << "\"";
  return ss.str();
}

std::string GenerateTextNodeSpec(const Node* node, const NodeIdMap& id_map) {
  std::stringstream ss;
  ss << node->shapes() << " " << node->op() << "(";
  size_t count = 0;
  for (auto& output : node->operands()) {
    if (count > 0) {
      ss << ", ";
    }
    ss << "%" << id_map.at(output.node);
    if (output.node->num_outputs() > 1) {
      ss << "." << output.index;
    }
    ++count;
  }
  ss << ")";
  for (auto& tag : GetNodeTags(node)) {
    ss << ", " << tag.name << "=" << tag.value;
  }
  return ss.str();
}

} // namespace

std::string DumpUtil::ToDot(c10::ArrayRef<const Node*> nodes) {
  auto post_order = Util::ComputePostOrder(nodes);
  return PostOrderToDot(post_order, nodes);
}

std::string DumpUtil::PostOrderToDot(
    c10::ArrayRef<const Node*> post_order,
    c10::ArrayRef<const Node*> roots) {
  std::unordered_map<const Node*, size_t> roots_ids = GetRootsIds(roots);
  NodeIdMap id_map = GenerateIdMap(post_order);
  std::stringstream ss;
  ss << "digraph G {\n";
  for (auto node : post_order) {
    ss << "  node" << id_map.at(node) << " ["
       << GenerateDotNodeSpec(node, roots_ids) << "]\n";
  }
  for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
    const Node* node = *it;
    size_t id = id_map.at(node);
    for (const auto i : c10::irange(node->operands().size())) {
      const Output& output = node->operand(i);
      ss << "  node" << id_map.at(output.node) << " -> node" << id;
      if (node->operands().size() > 1) {
        ss << " [label=\"i=" << i;
        if (output.node->num_outputs() > 1) {
          ss << ",o=" << output.index;
        }
        ss << "\"]\n";
      } else {
        if (output.node->num_outputs() > 1) {
          ss << " [label=\"o=" << output.index << "\"]";
        }
        ss << "\n";
      }
    }
  }
  ss << "}\n";
  return ss.str();
}

std::string DumpUtil::ToText(c10::ArrayRef<const Node*> nodes) {
  auto post_order = Util::ComputePostOrder(nodes);
  return PostOrderToText(post_order, nodes);
}

std::string DumpUtil::PostOrderToText(
    c10::ArrayRef<const Node*> post_order,
    c10::ArrayRef<const Node*> roots) {
  std::unordered_map<const Node*, size_t> roots_ids = GetRootsIds(roots);
  NodeIdMap id_map = GenerateIdMap(post_order);
  std::stringstream ss;
  ss << "IR {\n";
  for (auto node : post_order) {
    auto opt_root_id = GetRootNodeId(node, roots_ids);
    ss << "  %" << id_map.at(node) << " = "
       << GenerateTextNodeSpec(node, id_map);
    if (opt_root_id) {
      ss << ", ROOT=" << *opt_root_id;
    }
    ss << ", NodeType=" << typeid(*node).name();
    ss << "\n";
  }
  ss << "}\n";
  return ss.str();
}

std::string DumpUtil::ToBackend(
    c10::ArrayRef<Value> values,
    const BackendDevice& device) {
  auto lowering_ctx = LoweringContext::Create("IrToBackend", device);
  for (auto& ir_value : values) {
    lowering_ctx->AddResult(ir_value);
  }
  auto computation = lowering_ctx->Build();
  return getBackend()->GetComputationBackendText(computation);
}

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `std`

**Classes/Structs**: `AttrTag`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/lazy/core`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/lazy/core/ir_dump_util.h`
- `c10/util/irange.h`
- `torch/csrc/lazy/backend/backend_interface.h`
- `torch/csrc/lazy/backend/lowering_context.h`
- `torch/csrc/lazy/core/ir_util.h`
- `optional`
- `regex`
- `sstream`
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

Files in the same folder (`torch/csrc/lazy/core`):

- [`hash.cpp_docs.md`](./hash.cpp_docs.md)
- [`shape_inference.cpp_docs.md`](./shape_inference.cpp_docs.md)
- [`tensor_impl.h_docs.md`](./tensor_impl.h_docs.md)
- [`helpers.h_docs.md`](./helpers.h_docs.md)
- [`tensor_impl.cpp_docs.md`](./tensor_impl.cpp_docs.md)
- [`ir_metadata.cpp_docs.md`](./ir_metadata.cpp_docs.md)
- [`ir_metadata.h_docs.md`](./ir_metadata.h_docs.md)
- [`trie.cpp_docs.md`](./trie.cpp_docs.md)
- [`cache.h_docs.md`](./cache.h_docs.md)
- [`config.cpp_docs.md`](./config.cpp_docs.md)


## Cross-References

- **File Documentation**: `ir_dump_util.cpp_docs.md`
- **Keyword Index**: `ir_dump_util.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/lazy/core`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/lazy/core`, which is part of the **core PyTorch library**.



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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/lazy/core`):

- [`helpers.cpp_docs.md_docs.md`](./helpers.cpp_docs.md_docs.md)
- [`tensor_util.h_kw.md_docs.md`](./tensor_util.h_kw.md_docs.md)
- [`permutation_util.h_kw.md_docs.md`](./permutation_util.h_kw.md_docs.md)
- [`ir_util.cpp_kw.md_docs.md`](./ir_util.cpp_kw.md_docs.md)
- [`shape_inference.h_kw.md_docs.md`](./shape_inference.h_kw.md_docs.md)
- [`ir_builder.h_docs.md_docs.md`](./ir_builder.h_docs.md_docs.md)
- [`shape_inference.cpp_kw.md_docs.md`](./shape_inference.cpp_kw.md_docs.md)
- [`hash.h_kw.md_docs.md`](./hash.h_kw.md_docs.md)
- [`multi_wait.cpp_kw.md_docs.md`](./multi_wait.cpp_kw.md_docs.md)
- [`lazy_graph_executor.cpp_docs.md_docs.md`](./lazy_graph_executor.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `ir_dump_util.cpp_docs.md_docs.md`
- **Keyword Index**: `ir_dump_util.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
