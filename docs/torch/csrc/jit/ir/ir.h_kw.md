# Keyword Index: `torch/csrc/jit/ir/ir.h`

## File Information

- **Original File**: [torch/csrc/jit/ir/ir.h](../../../../../torch/csrc/jit/ir/ir.h)
- **Documentation**: [`ir.h_docs.md`](./ir.h_docs.md)
- **Folder**: `torch/csrc/jit/ir`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`AliasDb`**: [ir.h_docs.md](./ir.h_docs.md)
- **`Block`**: [ir.h_docs.md](./ir.h_docs.md)
- **`Function`**: [ir.h_docs.md](./ir.h_docs.md)
- **`FunctionSchemaMap`**: [ir.h_docs.md](./ir.h_docs.md)
- **`Graph`**: [ir.h_docs.md](./ir.h_docs.md)
- **`GraphFunction`**: [ir.h_docs.md](./ir.h_docs.md)
- **`MatchedSchema`**: [ir.h_docs.md](./ir.h_docs.md)
- **`MoveSide`**: [ir.h_docs.md](./ir.h_docs.md)
- **`Node`**: [ir.h_docs.md](./ir.h_docs.md)
- **`OperatorMap`**: [ir.h_docs.md](./ir.h_docs.md)
- **`OperatorSet`**: [ir.h_docs.md](./ir.h_docs.md)
- **`ProfileOp`**: [ir.h_docs.md](./ir.h_docs.md)
- **`T`**: [ir.h_docs.md](./ir.h_docs.md)
- **`THPPointer`**: [ir.h_docs.md](./ir.h_docs.md)
- **`TORCH_API`**: [ir.h_docs.md](./ir.h_docs.md)
- **`Use`**: [ir.h_docs.md](./ir.h_docs.md)
- **`Value`**: [ir.h_docs.md](./ir.h_docs.md)
- **`WithCurrentScope`**: [ir.h_docs.md](./ir.h_docs.md)
- **`WithInsertPoint`**: [ir.h_docs.md](./ir.h_docs.md)
- **`Wrap`**: [ir.h_docs.md](./ir.h_docs.md)
- **`for`**: [ir.h_docs.md](./ir.h_docs.md)
- **`indicated`**: [ir.h_docs.md](./ir.h_docs.md)
- **`is`**: [ir.h_docs.md](./ir.h_docs.md)
- **`of`**: [ir.h_docs.md](./ir.h_docs.md)

### Functions

- **`OperatorMap`**: [ir.h_docs.md](./ir.h_docs.md)
- **`clear`**: [ir.h_docs.md](./ir.h_docs.md)
- **`contains`**: [ir.h_docs.md](./ir.h_docs.md)
- **`current_scope`**: [ir.h_docs.md](./ir.h_docs.md)
- **`debugName`**: [ir.h_docs.md](./ir.h_docs.md)
- **`erase`**: [ir.h_docs.md](./ir.h_docs.md)
- **`eraseInput`**: [ir.h_docs.md](./ir.h_docs.md)
- **`eraseOutput`**: [ir.h_docs.md](./ir.h_docs.md)
- **`findAttr`**: [ir.h_docs.md](./ir.h_docs.md)
- **`hasAttribute`**: [ir.h_docs.md](./ir.h_docs.md)
- **`hasAttributeS`**: [ir.h_docs.md](./ir.h_docs.md)
- **`hasAttributes`**: [ir.h_docs.md](./ir.h_docs.md)
- **`hasDebugName`**: [ir.h_docs.md](./ir.h_docs.md)
- **`hasSeenTensor`**: [ir.h_docs.md](./ir.h_docs.md)
- **`hasUses`**: [ir.h_docs.md](./ir.h_docs.md)
- **`inBlockList`**: [ir.h_docs.md](./ir.h_docs.md)
- **`insert`**: [ir.h_docs.md](./ir.h_docs.md)
- **`insertOutput`**: [ir.h_docs.md](./ir.h_docs.md)
- **`isCompleteTensor`**: [ir.h_docs.md](./ir.h_docs.md)
- **`isMemberOf`**: [ir.h_docs.md](./ir.h_docs.md)
- **`is_constant`**: [ir.h_docs.md](./ir.h_docs.md)
- **`iterator`**: [ir.h_docs.md](./ir.h_docs.md)
- **`kind`**: [ir.h_docs.md](./ir.h_docs.md)
- **`kindOf`**: [ir.h_docs.md](./ir.h_docs.md)
- **`kindOfS`**: [ir.h_docs.md](./ir.h_docs.md)
- **`nodes`**: [ir.h_docs.md](./ir.h_docs.md)
- **`notExecutedOp`**: [ir.h_docs.md](./ir.h_docs.md)
- **`numAttributes`**: [ir.h_docs.md](./ir.h_docs.md)
- **`offset`**: [ir.h_docs.md](./ir.h_docs.md)
- **`permuteInputs`**: [ir.h_docs.md](./ir.h_docs.md)
- **`permuteOutputs`**: [ir.h_docs.md](./ir.h_docs.md)
- **`registerOutput`**: [ir.h_docs.md](./ir.h_docs.md)
- **`removeAllInputs`**: [ir.h_docs.md](./ir.h_docs.md)
- **`removeAllOutputs`**: [ir.h_docs.md](./ir.h_docs.md)
- **`replaceOutput`**: [ir.h_docs.md](./ir.h_docs.md)
- **`requires_grad`**: [ir.h_docs.md](./ir.h_docs.md)
- **`reverseIterator`**: [ir.h_docs.md](./ir.h_docs.md)
- **`scope`**: [ir.h_docs.md](./ir.h_docs.md)
- **`scopeName`**: [ir.h_docs.md](./ir.h_docs.md)
- **`setCallStack`**: [ir.h_docs.md](./ir.h_docs.md)
- **`setHasSeenTensor`**: [ir.h_docs.md](./ir.h_docs.md)
- **`setHistoricSchemaName`**: [ir.h_docs.md](./ir.h_docs.md)
- **`setInsertPoint`**: [ir.h_docs.md](./ir.h_docs.md)
- **`setOffset`**: [ir.h_docs.md](./ir.h_docs.md)
- **`setScope`**: [ir.h_docs.md](./ir.h_docs.md)
- **`set_current_scope`**: [ir.h_docs.md](./ir.h_docs.md)
- **`set_op_version`**: [ir.h_docs.md](./ir.h_docs.md)
- **`unique`**: [ir.h_docs.md](./ir.h_docs.md)

### Includes

- **`ATen/Utils.h`**: [ir.h_docs.md](./ir.h_docs.md)
- **`ATen/core/Tensor.h`**: [ir.h_docs.md](./ir.h_docs.md)
- **`ATen/core/dynamic_type.h`**: [ir.h_docs.md](./ir.h_docs.md)
- **`ATen/core/enum_type.h`**: [ir.h_docs.md](./ir.h_docs.md)
- **`ATen/core/functional.h`**: [ir.h_docs.md](./ir.h_docs.md)
- **`ATen/core/interned_strings.h`**: [ir.h_docs.md](./ir.h_docs.md)
- **`ATen/core/ivalue.h`**: [ir.h_docs.md](./ir.h_docs.md)
- **`ATen/core/jit_type.h`**: [ir.h_docs.md](./ir.h_docs.md)
- **`c10/util/ArrayRef.h`**: [ir.h_docs.md](./ir.h_docs.md)
- **`c10/util/Exception.h`**: [ir.h_docs.md](./ir.h_docs.md)
- **`functional`**: [ir.h_docs.md](./ir.h_docs.md)
- **`iosfwd`**: [ir.h_docs.md](./ir.h_docs.md)
- **`optional`**: [ir.h_docs.md](./ir.h_docs.md)
- **`torch/csrc/Export.h`**: [ir.h_docs.md](./ir.h_docs.md)
- **`torch/csrc/jit/ir/attributes.h`**: [ir.h_docs.md](./ir.h_docs.md)
- **`torch/csrc/jit/ir/graph_node_list.h`**: [ir.h_docs.md](./ir.h_docs.md)
- **`torch/csrc/jit/ir/named_value.h`**: [ir.h_docs.md](./ir.h_docs.md)
- **`torch/csrc/jit/ir/scope.h`**: [ir.h_docs.md](./ir.h_docs.md)
- **`torch/csrc/jit/runtime/operator.h`**: [ir.h_docs.md](./ir.h_docs.md)
- **`torch/csrc/utils/python_stub.h`**: [ir.h_docs.md](./ir.h_docs.md)
- **`torch/csrc/utils/schema_info.h`**: [ir.h_docs.md](./ir.h_docs.md)
- **`unordered_set`**: [ir.h_docs.md](./ir.h_docs.md)
- **`vector`**: [ir.h_docs.md](./ir.h_docs.md)

### Namespaces

- **`aten`**: [ir.h_docs.md](./ir.h_docs.md)
- **`attr`**: [ir.h_docs.md](./ir.h_docs.md)
- **`cuda`**: [ir.h_docs.md](./ir.h_docs.md)
- **`prim`**: [ir.h_docs.md](./ir.h_docs.md)
- **`torch`**: [ir.h_docs.md](./ir.h_docs.md)
- **`utils`**: [ir.h_docs.md](./ir.h_docs.md)


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
