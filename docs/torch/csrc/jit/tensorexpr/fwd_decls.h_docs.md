# Documentation: `torch/csrc/jit/tensorexpr/fwd_decls.h`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/fwd_decls.h`
- **Size**: 2,990 bytes (2.92 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <c10/core/ScalarType.h>
#include <memory>

namespace torch::jit::tensorexpr {

template <typename Node>
using NodePtr = std::shared_ptr<Node>;

template <typename To, typename From>
NodePtr<To> to(const NodePtr<From>& x) {
  return std::dynamic_pointer_cast<To>(x);
}

template <typename To, typename From>
NodePtr<To> static_to(NodePtr<From> x) {
  return std::static_pointer_cast<To>(x);
}

template <typename Node, typename... Args>
NodePtr<Node> alloc(Args&&... args) {
  return std::make_shared<Node>(std::forward<Args>(args)...);
}

class Buf;
class Expr;
class Stmt;
class Var;

using BufPtr = NodePtr<Buf>;
using ExprPtr = NodePtr<Expr>;
using StmtPtr = NodePtr<Stmt>;
using VarPtr = NodePtr<Var>;

class ExprHandle;
class VarHandle;
class BufHandle;

class Add;
class And;
class BitCast;
class Broadcast;
class Cast;
class CompareSelect;
class Div;
class IfThenElse;
class Intrinsics;
class Let;
class Load;
class Lshift;
class Max;
class MaxTerm;
class Min;
class MinTerm;
class Mod;
class Mul;
class Or;
class Polynomial;
class Ramp;
class ReduceOp;
class RoundOff;
class Rshift;
class Store;
class Sub;
class Term;
class Xor;
using AddPtr = NodePtr<Add>;
using AndPtr = NodePtr<And>;
using BitCastPtr = NodePtr<BitCast>;
using BroadcastPtr = NodePtr<Broadcast>;
using CastPtr = NodePtr<Cast>;
using CompareSelectPtr = NodePtr<CompareSelect>;
using DivPtr = NodePtr<Div>;
using IfThenElsePtr = NodePtr<IfThenElse>;
using IntrinsicsPtr = NodePtr<Intrinsics>;
using LetPtr = NodePtr<Let>;
using LoadPtr = NodePtr<Load>;
using LshiftPtr = NodePtr<Lshift>;
using MaxPtr = NodePtr<Max>;
using MaxTermPtr = NodePtr<MaxTerm>;
using MinPtr = NodePtr<Min>;
using MinTermPtr = NodePtr<MinTerm>;
using ModPtr = NodePtr<Mod>;
using MulPtr = NodePtr<Mul>;
using OrPtr = NodePtr<Or>;
using PolynomialPtr = NodePtr<Polynomial>;
using RampPtr = NodePtr<Ramp>;
using ReduceOpPtr = NodePtr<ReduceOp>;
using RoundOffPtr = NodePtr<RoundOff>;
using RshiftPtr = NodePtr<Rshift>;
using StorePtr = NodePtr<Store>;
using SubPtr = NodePtr<Sub>;
using TermPtr = NodePtr<Term>;
using XorPtr = NodePtr<Xor>;

class Allocate;
class AtomicAdd;
class Block;
class Cond;
class ExternalCall;
class ExternalCallWithAlloc;
class For;
class Free;
class FreeExt;
class PlacementAllocate;
class SyncThreads;
using AllocatePtr = NodePtr<Allocate>;
using AtomicAddPtr = NodePtr<AtomicAdd>;
using BlockPtr = NodePtr<Block>;
using CondPtr = NodePtr<Cond>;
using ExternalCallPtr = NodePtr<ExternalCall>;
using ExternalCallWithAllocPtr = NodePtr<ExternalCallWithAlloc>;
using ForPtr = NodePtr<For>;
using FreePtr = NodePtr<Free>;
using FreeExtPtr = NodePtr<FreeExt>;
using PlacementAllocatePtr = NodePtr<PlacementAllocate>;
using SyncThreadsPtr = NodePtr<SyncThreads>;

#define IMM_DECLARE(Type, Name) \
  class Name##Imm;              \
  using Name##ImmPtr = NodePtr<Name##Imm>;
AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, IMM_DECLARE)
#undef IMM_DECLARE

} // namespace torch::jit::tensorexpr

```



## High-Level Overview


This C++ file contains approximately 47 class(es)/struct(s) and 1 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `Buf`, `Expr`, `Stmt`, `Var`, `ExprHandle`, `VarHandle`, `BufHandle`, `Add`, `And`, `BitCast`, `Broadcast`, `Cast`, `CompareSelect`, `Div`, `IfThenElse`, `Intrinsics`, `Let`, `Load`, `Lshift`, `Max`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/core/ScalarType.h`
- `memory`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`torch/csrc/jit/tensorexpr`):

- [`codegen.cpp_docs.md`](./codegen.cpp_docs.md)
- [`bounds_overlap.h_docs.md`](./bounds_overlap.h_docs.md)
- [`eval.h_docs.md`](./eval.h_docs.md)
- [`cuda_codegen.cpp_docs.md`](./cuda_codegen.cpp_docs.md)
- [`ir_simplifier.h_docs.md`](./ir_simplifier.h_docs.md)
- [`reduction.h_docs.md`](./reduction.h_docs.md)
- [`kernel.cpp_docs.md`](./kernel.cpp_docs.md)
- [`cuda_codegen.h_docs.md`](./cuda_codegen.h_docs.md)
- [`external_functions_core.cpp_docs.md`](./external_functions_core.cpp_docs.md)
- [`graph_opt.h_docs.md`](./graph_opt.h_docs.md)


## Cross-References

- **File Documentation**: `fwd_decls.h_docs.md`
- **Keyword Index**: `fwd_decls.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
