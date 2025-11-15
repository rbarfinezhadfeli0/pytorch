# Keyword Index: `torch/csrc/jit/runtime/register_prim_ops.cpp`

## File Information

- **Original File**: [torch/csrc/jit/runtime/register_prim_ops.cpp](../../../../../torch/csrc/jit/runtime/register_prim_ops.cpp)
- **Documentation**: [`register_prim_ops.cpp_docs.md`](./register_prim_ops.cpp_docs.md)
- **Folder**: `torch/csrc/jit/runtime`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`TORCH_SELECTIVE_SCHEMA`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)

### Functions

- **`aliasAnalysisFromSchema`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`dictClear`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`dictConstructFromList`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`dictContains`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`dictCopy`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`dictDelete`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`dictGet`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`dictItems`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`dictKeys`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`dictLen`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`dictPop`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`dictPopItem`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`dictSetDefault`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`dictSetItem`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`dictUpdate`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`dictValues`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`hashValue`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`if`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`isSortableListOfObjectsOrTuples`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`isSortableTupleType`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`normalizeIndex`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`powWrapper`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`reg`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`sort_op`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`stringFindImpl`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`stringSlice`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)

### Includes

- **`ATen/autocast_mode.h`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`ATen/core/Generator.h`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`algorithm`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`bitset`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`c10/util/Exception.h`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`c10/util/irange.h`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`cctype`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`cmath`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`iostream`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`memory`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`optional`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`ostream`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`stdexcept`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`string`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`torch/csrc/jit/mobile/promoted_prim_ops.h`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`torch/csrc/jit/runtime/custom_operator.h`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`torch/csrc/jit/runtime/operator.h`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`torch/csrc/jit/runtime/register_ops_utils.h`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`torch/csrc/jit/runtime/slice_indices_adjust.h`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`torch/library.h`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`utility`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`vector`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)

### Namespaces

- **`because`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)
- **`torch`**: [register_prim_ops.cpp_docs.md](./register_prim_ops.cpp_docs.md)


## Keyword → Section Map

The following sections in the documentation cover these topics:

- **File Metadata**: Basic file information
- **Original Source**: Complete source code
- **High-Level Overview**: Purpose and role
- **Detailed Analysis**: In-depth code analysis
- **Architecture & Design**: Design patterns and structure
- **Dependencies**: Related modules and imports
- **Performance Considerations**: Efficiency and optimization
- **Security & Safety**: Security analysis
- **Testing & Usage**: How to use and test

---

*Generated by PyTorch Repository Documentation System*
