# Keyword Index: `aten/src/ATen/core/ivalue.h`

## File Information

- **Original File**: [aten/src/ATen/core/ivalue.h](../../../../../aten/src/ATen/core/ivalue.h)
- **Documentation**: [`ivalue.h_docs.md`](./ivalue.h_docs.md)
- **Folder**: `aten/src/ATen/core`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`Await`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`Capsule`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`ClassType`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`CompAliasedIValues`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`CompIdentityIValues`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`CompilationUnit`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`ComplexHolder`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`ConstantString`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`Dict`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`EnumHolder`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`Function`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`Future`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`GenericDict`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`HashAliasedIValue`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`HashIdentityIValue`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`IListRef`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`IValue`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`Key`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`List`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`Module`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`NullType`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`Object`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`OptionalArray`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`PyObjectHolder`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`RRefInterface`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`StreamData3Holder`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`T`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`TORCH_API`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`Tag`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`TagType`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`Tuple`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`Type`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`Value`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`WeakIValue`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`WeakOrStrongCompilationUnit`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`an`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`c10`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`get`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`object`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`this`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`to`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`type`**: [ivalue.h_docs.md](./ivalue.h_docs.md)

### Functions

- **`destroy`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`hash`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`hashTensor`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`holdingEmptyStrongRef`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`holdingStrongRef`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`holds_empty_strong_ref`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`holds_strong_ref`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isAliasOf`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isAwait`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isBlob`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isBool`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isCapsule`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isComplexDouble`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isDevice`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isDouble`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isEnum`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isFuture`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isGenerator`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isGenericDict`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isInt`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isIntrusivePtr`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isIntrusivePtrConstexpr`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isIntrusivePtrLegacyBehavior`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isList`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isNone`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isObject`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isPtrType`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isPyObject`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isQuantizer`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isRRef`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isSameIdentity`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isScalar`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isStorage`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isStream`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isString`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isSymBool`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isSymFloat`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isSymInt`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isTensor`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isTuple`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`isUnsigned`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`lock`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`tagKind`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`toBool`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`toDevice`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`toDimname`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`toDouble`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`toInt`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`toLayout`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`toMemoryFormat`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`toNone`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`toQScheme`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`toScalar`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`toScalarType`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`toUInt`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`uninitialized`**: [ivalue.h_docs.md](./ivalue.h_docs.md)

### Includes

- **`ATen/core/DimVector.h`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`ATen/core/TensorBody.h`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`ATen/core/blob.h`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`ATen/core/custom_class.h`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`ATen/core/ivalue_inl.h`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`ATen/core/ivalue_to.h`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`ATen/core/jit_type_base.h`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`ATen/core/type_factory.h`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`c10/core/SymBool.h`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`c10/core/SymFloat.h`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`c10/macros/Export.h`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`c10/util/MaybeOwned.h`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`c10/util/intrusive_ptr.h`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`limits`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`type_traits`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`unordered_map`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`unordered_set`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`utility`**: [ivalue.h_docs.md](./ivalue.h_docs.md)

### Namespaces

- **`c10`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`ivalue`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`jit`**: [ivalue.h_docs.md](./ivalue.h_docs.md)
- **`torch`**: [ivalue.h_docs.md](./ivalue.h_docs.md)


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
