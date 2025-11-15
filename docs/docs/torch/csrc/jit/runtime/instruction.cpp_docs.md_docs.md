# Documentation: `docs/torch/csrc/jit/runtime/instruction.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/runtime/instruction.cpp_docs.md`
- **Size**: 5,031 bytes (4.91 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/runtime/instruction.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/runtime/instruction.cpp`
- **Size**: 2,470 bytes (2.41 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/irange.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <cstring>
#include <iostream>

namespace torch::jit {
static std::ostream& operator<<(std::ostream& out, OpCode op) {
  switch (op) {
#define OP_STRING(x, _) \
  case x:               \
    return out << #x;
    FORALL_OPCODES(OP_STRING)
#undef OP_STRING
  }
  return out;
}

char const* toString(OpCode op) {
  switch (op) {
#define OP_STRING(x, _) \
  case x:               \
    return #x;
    FORALL_OPCODES(OP_STRING)
#undef OP_STRING
  }
  return nullptr;
}

static const char* OpInfo(OpCode op) {
  switch (op) {
#define OP_INFO(x, info) \
  case x:                \
    return info;
    // NOLINTNEXTLINE(bugprone-branch-clone)
    FORALL_OPCODES(OP_INFO)
#undef OP_INFO
  }
  return nullptr;
}

static constexpr size_t instruction_size = 8;
static_assert(
    sizeof(Instruction) == instruction_size,
    "Instructions should be 8 bytes");
std::ostream& operator<<(std::ostream& out, Instruction inst) {
  // TODO: use op info to print out the op in a more user-friendly way
  auto nargs = std::strlen(OpInfo(inst.op));
  out << inst.op;
  if (nargs > 0) {
    out << " " << inst.X;
  }
  if (nargs > 1) {
    out << " " << inst.N;
  }
  return out;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
static constexpr const char* strOpCode[] = {
#define STR_OP(x, _) #x,
    FORALL_OPCODES(STR_OP)
#undef STR_OP
};

OpCode parseOpCode(const char* str) {
  const int n = sizeof(strOpCode) / sizeof(strOpCode[0]);
  for (const auto i : c10::irange(n)) {
    if (strcmp(strOpCode[i], str) == 0)
      return (OpCode)i;
  }
  return OP;
}

bool isOpSupportedInMobile(OpCode op) {
  // clang-format off
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  static constexpr OpCode supported_ops_in_mobile[] {
      OP, OPN, LOAD, MOVE, STOREN, STORE, DROP, DROPR, LOADC, JF, JMP, LOOP,
      RET, GET_ATTR, SET_ATTR, LIST_CONSTRUCT, TUPLE_CONSTRUCT, WARN,
      INTERFACE_CALL, LIST_UNPACK, TUPLE_SLICE, DICT_CONSTRUCT,
      NAMED_TUPLE_CONSTRUCT, CREATE_OBJECT, ISINSTANCE, CALL,
      RAISE_EXCEPTION, UNCHECKED_CAST, __IS__, UN_INITIALIZED,
      __ISNOT__, FORMAT, DEVICE, DICT_INDEX,
      DTYPE, TUPLE_INDEX, DIM, __NOT__,
      TO_LIST, NUM_TO_TENSOR, IS_CUDA};
  // clang-format on

  for (auto sop : supported_ops_in_mobile) {
    if (op == sop)
      return true;
  }
  return false;
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

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

- `c10/util/irange.h`
- `torch/csrc/jit/runtime/instruction.h`
- `cstring`
- `iostream`


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

Files in the same folder (`torch/csrc/jit/runtime`):

- [`decomposition_registry.h_docs.md`](./decomposition_registry.h_docs.md)
- [`register_distributed_ops.cpp_docs.md`](./register_distributed_ops.cpp_docs.md)
- [`instruction.h_docs.md`](./instruction.h_docs.md)
- [`argument_spec.cpp_docs.md`](./argument_spec.cpp_docs.md)
- [`symbolic_script.h_docs.md`](./symbolic_script.h_docs.md)
- [`register_prim_ops_fulljit.cpp_docs.md`](./register_prim_ops_fulljit.cpp_docs.md)
- [`symbolic_shape_registry_util.cpp_docs.md`](./symbolic_shape_registry_util.cpp_docs.md)
- [`interpreter.h_docs.md`](./interpreter.h_docs.md)
- [`logging.h_docs.md`](./logging.h_docs.md)


## Cross-References

- **File Documentation**: `instruction.cpp_docs.md`
- **Keyword Index**: `instruction.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/runtime`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
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

Files in the same folder (`docs/torch/csrc/jit/runtime`):

- [`register_ops_utils.h_docs.md_docs.md`](./register_ops_utils.h_docs.md_docs.md)
- [`register_c10_ops.cpp_docs.md_docs.md`](./register_c10_ops.cpp_docs.md_docs.md)
- [`exception_message.h_kw.md_docs.md`](./exception_message.h_kw.md_docs.md)
- [`register_prim_ops.cpp_kw.md_docs.md`](./register_prim_ops.cpp_kw.md_docs.md)
- [`autodiff.cpp_kw.md_docs.md`](./autodiff.cpp_kw.md_docs.md)
- [`decomposition_registry_util.h_docs.md_docs.md`](./decomposition_registry_util.h_docs.md_docs.md)
- [`slice_indices_adjust.cpp_docs.md_docs.md`](./slice_indices_adjust.cpp_docs.md_docs.md)
- [`graph_iterator.h_kw.md_docs.md`](./graph_iterator.h_kw.md_docs.md)
- [`shape_function_registry.h_docs.md_docs.md`](./shape_function_registry.h_docs.md_docs.md)
- [`symbolic_script.cpp_docs.md_docs.md`](./symbolic_script.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `instruction.cpp_docs.md_docs.md`
- **Keyword Index**: `instruction.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
