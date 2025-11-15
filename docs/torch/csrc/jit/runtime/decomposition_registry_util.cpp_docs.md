# Documentation: `torch/csrc/jit/runtime/decomposition_registry_util.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/runtime/decomposition_registry_util.cpp`
- **Size**: 3,228 bytes (3.15 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp

/**
 * @generated
 * This is an auto-generated file. Please do not modify it by hand.
 * To re-generate, please run:
 * cd ~/pytorch && python torchgen/decompositions/gen_jit_decompositions.py
 */
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/decomposition_registry_util.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace torch::jit {

const std::string decomp_funcs =
    R"(def var_decomposition(input: Tensor,
    dim: Optional[List[int]]=None,
    correction: Union[float, int, NoneType, bool]=None,
    keepdim: bool=False) -> Tensor:
  _0 = uninitialized(float)
  if torch.__is__(dim, None):
    dim0 = annotate(List[int], [])
  else:
    dim0 = unchecked_cast(List[int], dim)
  if torch.eq(torch.len(dim0), 0):
    n = torch.numel(input)
  else:
    n0 = 1
    for _1 in range(torch.len(dim0)):
      dim_i = dim0[_1]
      n1 = torch.mul(n0, (torch.size(input))[dim_i])
      n0 = n1
    n = n0
  mean = torch.mean(input, dim0, True)
  sub = torch.sub(input, mean)
  sq = torch.mul(sub, sub)
  sum = torch.sum(sq, dim0, keepdim)
  if torch.__is__(correction, None):
    denom = float(torch.sub(n, 1))
  else:
    correction0 = unchecked_cast(Union[float, int, bool], correction)
    _2 = isinstance(correction0, int)
    if _2:
      correction1 = unchecked_cast(int, correction0)
      denom0 = float(torch.sub(n, correction1))
    else:
      correction2 = unchecked_cast(Union[float, bool], correction0)
      _3 = isinstance(correction2, float)
      if _3:
        correction3 = unchecked_cast(float, correction2)
        denom2 = torch.sub(float(n), correction3)
        denom1 = denom2
      else:
        ops.prim.RaiseException("correction must be int or float", "builtins.RuntimeError")
        denom1 = _0
      denom0 = denom1
    denom = denom0
  _4 = torch.div(sum, ops.prim.max(0, denom))
  return _4

def var(input: Tensor,
    unbiased: bool=True) -> Tensor:
  if unbiased:
    _0 = 1
  else:
    _0 = 0
  _1 = uninitialized(float)
  n = torch.numel(input)
  mean = torch.mean(input, annotate(List[int], []), True)
  sub = torch.sub(input, mean)
  sq = torch.mul(sub, sub)
  sum = torch.sum(sq, annotate(List[int], []))
  _2 = isinstance(_0, int)
  if _2:
    denom = float(torch.sub(n, _0))
  else:
    correction = unchecked_cast(Union[float, bool], _0)
    _3 = isinstance(correction, float)
    if _3:
      correction0 = unchecked_cast(float, correction)
      denom0 = torch.sub(float(n), correction0)
    else:
      ops.prim.RaiseException("correction must be int or float", "builtins.RuntimeError")
      denom0 = _1
    denom = denom0
  _4 = torch.div(sum, ops.prim.max(0, denom))
  return _4

)";

const std::string& GetSerializedDecompositions() {
  return decomp_funcs;
}

const OperatorMap<std::string>& GetDecompositionMapping() {
  // clang-format off
 static const OperatorMap<std::string> decomposition_mapping {
    {"aten::var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor", "var_decomposition"},
    {"aten::var(Tensor self, bool unbiased=True) -> Tensor", "var"},
  };
  // clang-format on

  return decomposition_mapping;
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/jit_log.h`
- `torch/csrc/jit/passes/inliner.h`
- `torch/csrc/jit/runtime/decomposition_registry_util.h`
- `torch/csrc/jit/runtime/operator.h`


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

Files in the same folder (`torch/csrc/jit/runtime`):

- [`decomposition_registry.h_docs.md`](./decomposition_registry.h_docs.md)
- [`register_distributed_ops.cpp_docs.md`](./register_distributed_ops.cpp_docs.md)
- [`instruction.h_docs.md`](./instruction.h_docs.md)
- [`argument_spec.cpp_docs.md`](./argument_spec.cpp_docs.md)
- [`instruction.cpp_docs.md`](./instruction.cpp_docs.md)
- [`symbolic_script.h_docs.md`](./symbolic_script.h_docs.md)
- [`register_prim_ops_fulljit.cpp_docs.md`](./register_prim_ops_fulljit.cpp_docs.md)
- [`symbolic_shape_registry_util.cpp_docs.md`](./symbolic_shape_registry_util.cpp_docs.md)
- [`interpreter.h_docs.md`](./interpreter.h_docs.md)
- [`logging.h_docs.md`](./logging.h_docs.md)


## Cross-References

- **File Documentation**: `decomposition_registry_util.cpp_docs.md`
- **Keyword Index**: `decomposition_registry_util.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
