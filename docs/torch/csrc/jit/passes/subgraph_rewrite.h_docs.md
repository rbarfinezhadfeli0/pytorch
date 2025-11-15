# Documentation: `torch/csrc/jit/passes/subgraph_rewrite.h`

## File Metadata

- **Path**: `torch/csrc/jit/passes/subgraph_rewrite.h`
- **Size**: 4,087 bytes (3.99 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
/** This file defines API for pattern-based subgraph rewrites.
 *
 * The API can be used for finding concrete patterns in the model and replacing
 * the corresponding subgraphs with another subgraph. A special case of such
 * rewrites is fusion, where the new subgraph consists of just a single node.
 *
 * There is a default set of the most common patterns that everyone could use.
 * Alternatively, an arbitrary pattern can be registered.
 */
#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

#include <functional>
#include <unordered_set>
#include <vector>

namespace torch::jit {

// Forward declarations.
struct RewritePatternDescr;
struct Match;

using MatchFilter = std::function<
    bool(const Match&, const std::unordered_map<std::string, Value*>&)>;

/** Run pattern-based subgraph rewrites on all methods in the module.
 *
 * This pass will go through all methods in the module and try to replace all
 * recognized patterns (see SubgraphRewriter::RegisterDefaultPatterns for the
 * list of these patterns).
 */
TORCH_API Module PatternBasedRewrite(const Module& module);

/** A class implementing API for pattern-based subgraph rewrites.
 *
 * To perform pattern-based subgraph rewrites on a module using this API, one
 * needs to create an object of such class, register rewrite patterns and run
 * the transformation pass (`runOnModule`).
 *
 * To use standard patterns, one could use `RegisterDefaultPatterns`.
 *
 * To enable rewrites of custom patterns, the custom patterns must be registered
 * with `RegisterRewritePattern`.
 */
class TORCH_API SubgraphRewriter {
 public:
  // Run pattern-based subgraph rewrite pass on the module.
  Module runOnModule(const Module& module);

  // Run pattern-based subgraph rewrite pass on the graph (used in testing).
  // `filter` is a function that does extra filtering on the match. If it
  // returns false for a given Match, we'll skip the Match. The filter
  // function's arguments consist of a Match and a value map from parsing the
  // pattern graph. Both the Match and the value map are necessary because we
  // need to 1) do extra filtering on the matched result as well as 2) refer to
  // the values in the matched result through the values in the pattern graph.
  void runOnGraph(
      std::shared_ptr<Graph>& graph,
      const std::vector<MatchFilter>& filters);

  void runOnGraph(
      std::shared_ptr<Graph>& graph,
      const MatchFilter& filter =
          [](const Match&, const std::unordered_map<std::string, Value*>&) {
            return true;
          }) {
    runOnGraph(graph, std::vector<MatchFilter>({filter}));
  }

  // Register standard rewrite patterns.
  void RegisterDefaultPatterns();

  /** Register a custom rewrite pattern.
   *
   * The method takes two parameters specifying the pattern:
   * \p PATTERN - IR string representing the pattern subgraph.
   * \p REPLACEMENT - IR string representing the replacement subgraph.
   * \p value name map - vector of pairs mapping values in the replacement graph
   * to the values in the pattern graph. Used for preserving source range info
   * across graph rewrite.
   *
   * See examples of pattern registering in `RegisterDefaultPatterns`.
   */
  void RegisterRewritePattern(
      const std::string& pattern,
      const std::string& replacement,
      const std::vector<std::pair<std::string, std::string>>& value_name_pair =
          {});

 private:
  std::vector<RewritePatternDescr> patterns_;
  std::unordered_set<Node*> nodes_to_delete_;

  void rewriteSinglePatternOnGraph(
      std::shared_ptr<Graph>& graph,
      const RewritePatternDescr& pattern,
      const std::vector<MatchFilter>& filters);

  bool overlapsWithPreviousMatches(const Match* match);
};

/** Rewrite pattern descriptor.
 *
 * This structure is used in the implementation of `SubgraphRewriter` and
 * is not supposed to be used externally.
 */
struct RewritePatternDescr {
  std::string pattern;
  std::string replacement;
  std::unordered_map<std::string, std::string> value_name_map;
};

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `RewritePatternDescr`, `Match`, `implementing`, `TORCH_API`, `RewritePatternDescr`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/passes`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/api/module.h`
- `torch/csrc/jit/ir/ir.h`
- `functional`
- `unordered_set`
- `vector`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/csrc/jit/passes`):

- [`inline_fork_wait.h_docs.md`](./inline_fork_wait.h_docs.md)
- [`subgraph_rewrite.cpp_docs.md`](./subgraph_rewrite.cpp_docs.md)
- [`value_refinement_utils.cpp_docs.md`](./value_refinement_utils.cpp_docs.md)
- [`create_autodiff_subgraphs.cpp_docs.md`](./create_autodiff_subgraphs.cpp_docs.md)
- [`update_differentiable_graph_requires_grad.h_docs.md`](./update_differentiable_graph_requires_grad.h_docs.md)
- [`inplace_check.h_docs.md`](./inplace_check.h_docs.md)
- [`common_subexpression_elimination.h_docs.md`](./common_subexpression_elimination.h_docs.md)
- [`dtype_analysis.cpp_docs.md`](./dtype_analysis.cpp_docs.md)
- [`canonicalize.h_docs.md`](./canonicalize.h_docs.md)
- [`add_if_then_else.h_docs.md`](./add_if_then_else.h_docs.md)


## Cross-References

- **File Documentation**: `subgraph_rewrite.h_docs.md`
- **Keyword Index**: `subgraph_rewrite.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
