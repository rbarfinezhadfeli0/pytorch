# Keyword Index: `torch/csrc/jit/python/script_init.cpp`

## File Information

- **Original File**: [torch/csrc/jit/python/script_init.cpp](../../../../../torch/csrc/jit/python/script_init.cpp)
- **Documentation**: [`script_init.cpp_docs.md`](./script_init.cpp_docs.md)
- **Folder**: `torch/csrc/jit/python`

## Keywords Extracted

This file contains the following key identifiers, symbols, and concepts:


### Class/Structs

- **`DeepCopyMemoTable`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`M`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`PythonResolver`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`VISIBILITY_HIDDEN`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`Work`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`bodies`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`currently`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`inheritance`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`of`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`slot_dict_impl`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`type`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`yet`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)

### Functions

- **`_jit_debug_module_iterators`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`addFunctionToModule`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`bind`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`calcOverloadedFunctionDefaults`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`checkMutableFunctionDefault`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`checkOverloadDecl`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`contains`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`debugMakeList`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`debugMakeNamedList`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`debugMakeSet`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`extra_files_from_python`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`extra_files_to_python`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`getSchemaWithNameAndDefaults`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`getattr`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`initJitScriptBindings`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`isNamedTupleClass`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`ivalue_tags_match`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`mergeDefaultsAndExtraParametersToOverloadDecl`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`pyCompilationUnitDefine`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`pyIValueDeepcopy`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`resolveTypeFromObject`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`script_compile_function`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`script_compile_overloaded_function`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`setattr`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)

### Includes

- **`ATen/ATen.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`ATen/core/function_schema.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`ATen/core/ivalue.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`ATen/core/qualified_name.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`c10/util/Exception.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`c10/util/intrusive_ptr.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`c10/util/irange.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`caffe2/serialize/versions.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`cstddef`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`fmt/format.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`memory`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`pybind11/detail/common.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`pybind11/functional.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`pybind11/pybind11.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`pybind11/pytypes.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`pybind11/stl.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`pybind11/stl_bind.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`sstream`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`string`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/Device.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/DynamicTypes.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/api/include/torch/ordered_dict.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/api/module.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/api/object.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/frontend/ir_emitter.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/frontend/parser.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/frontend/sugared_value.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/frontend/tracer.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/ir/constants.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/ir/graph_utils.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/ir/irparser.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/mobile/code.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/mobile/compatibility/backport.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/mobile/compatibility/model_compatibility.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/mobile/file_format.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/mobile/flatbuffer_loader.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/mobile/import.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/mobile/module.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/mobile/quantization.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/mobile/train/export_data.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/operator_upgraders/upgraders.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/operator_upgraders/upgraders_entry.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/operator_upgraders/utils.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/operator_upgraders/version_map.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/passes/inliner.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/passes/shape_analysis.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/python/module_python.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/python/pybind_utils.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/python/python_dict.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/python/python_ivalue.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/python/python_list.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/python/python_sugared_value.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/python/python_tracer.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/python/script_init.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/runtime/graph_executor.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/runtime/instruction.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/runtime/interpreter.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/runtime/logging.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/serialization/export_bytecode.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/serialization/flatbuffer_serializer.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/serialization/import.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/serialization/import_source.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/serialization/pickle.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/serialization/python_print.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/testing/file_check.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/jit/testing/hooks_for_testing.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch/csrc/utils/pybind.h`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`tuple`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`utility`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`vector`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)

### Namespaces

- **`static`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)
- **`torch`**: [script_init.cpp_docs.md](./script_init.cpp_docs.md)


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
