# Documentation: `test/mobile/test_upgrader_bytecode_table_example.cpp`

## File Metadata

- **Path**: `test/mobile/test_upgrader_bytecode_table_example.cpp`
- **Size**: 24,624 bytes (24.05 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
/**
 * @generated
 * This is an auto-generated file. Please do not modify it by hand.
 * To re-generate, please run:
 * cd ~/pytorch && python torch/csrc/jit/mobile/upgrader_mobile.cpp
 */

#include <torch/csrc/jit/mobile/upgrader_mobile.h>

#include <ATen/core/ivalue.h>
#include <caffe2/serialize/versions.h>
#include <torch/csrc/jit/mobile/type_parser.h>

namespace c10 {
TypePtr parseType(const std::string& pythonStr);
} // namespace c10

namespace torch {
namespace jit {

// clang-format off

// From operator_versions_map

const std::unordered_map<std::string, std::vector<Upgrader>>
getOperatorVersionMapForMobile() {
  static std::unordered_map<std::string, std::vector<Upgrader>>
        operatorVersionMapForMobile({
                {std::string("aten::div.Scalar"),
                    std::vector<Upgrader>({
                        Upgrader({0, 3, "div_Scalar_0_3", 0})
                    })},
                {std::string("aten::div.Tensor"),
                    std::vector<Upgrader>({
                        Upgrader({0, 3, "div_Tensor_0_3", 1})
                    })},
                {std::string("aten::div.out"),
                    std::vector<Upgrader>({
                        Upgrader({0, 3, "div_out_0_3", 4})
                    })},
                {std::string("aten::div_.Scalar"),
                    std::vector<Upgrader>({
                        Upgrader({0, 3, "div__Scalar_0_3", 2})
                    })},
                {std::string("aten::div_.Tensor"),
                    std::vector<Upgrader>({
                        Upgrader({0, 3, "div__Tensor_0_3", 3})
                    })},
                {std::string("aten::linspace"),
                    std::vector<Upgrader>({
                        Upgrader({0, 7, "linspace_0_7", 7})
                    })},
                {std::string("aten::linspace.out"),
                    std::vector<Upgrader>({
                        Upgrader({0, 7, "linspace_out_0_7", 8})
                    })},
      });
  return operatorVersionMapForMobile;
}

const std::vector<ByteCodeFunctionWithOperator>& getUpgraderBytecodeList() {
  auto generate_upgrader_bytecode_list = []() {
    std::vector<ByteCodeFunctionWithOperator> upgrader_function_list({
                   ByteCodeFunctionWithOperator({
                           mobile::Function::registerFunc(
                               "div_Scalar_0_3",
                               std::vector<Instruction>({
                                           Instruction{OpCode::STOREN, 1, 2},
                                           Instruction{OpCode::LOAD, 1, 0},
                                           Instruction{OpCode::OP, 0, 0},
                                           Instruction{OpCode::JF, 3, 0},
                                           Instruction{OpCode::LOADC, 1, 0},
                                           Instruction{OpCode::JMP, 3, 0},
                                           Instruction{OpCode::LOAD, 2, 0},
                                           Instruction{OpCode::ISINSTANCE, 0, 1},
                                           Instruction{OpCode::STORE, 3, 0},
                                           Instruction{OpCode::MOVE, 3, 0},
                                           Instruction{OpCode::JF, 5, 0},
                                           Instruction{OpCode::LOAD, 1, 0},
                                           Instruction{OpCode::LOAD, 2, 0},
                                           Instruction{OpCode::OP, 1, 0},
                                           Instruction{OpCode::JMP, 6, 0},
                                           Instruction{OpCode::LOAD, 1, 0},
                                           Instruction{OpCode::LOAD, 2, 0},
                                           Instruction{OpCode::OP, 2, 0},
                                           Instruction{OpCode::LOADC, 0, 0},
                                           Instruction{OpCode::OP, 3, 0},
                                           Instruction{OpCode::STORE, 4, 0},
                                           Instruction{OpCode::DROPR, 2, 0},
                                           Instruction{OpCode::DROPR, 1, 0},
                                           Instruction{OpCode::MOVE, 4, 0},
                                           Instruction{OpCode::RET, 0, 0},
                                   }), // instructions list,
                               std::vector<c10::IValue>({
                                           c10::IValue("trunc"),
                                           c10::IValue(true),
                                   }), // constants list,
                               std::vector<c10::TypePtr>({
                                       c10::parseType("float"),
                                   }), // types list,
                               4
                           ),
                           std::vector<OperatorString>({
                                   OperatorString({"aten::is_floating_point", "", 1}),
                                   OperatorString({"aten::div", "Scalar", 2}),
                                   OperatorString({"prim::unchecked_cast", "", 1}),
                                   OperatorString({"aten::div", "Scalar_mode", 3}),
                           }), // operators list
                   }),
                   ByteCodeFunctionWithOperator({
                           mobile::Function::registerFunc(
                               "div_Tensor_0_3",
                               std::vector<Instruction>({
                                           Instruction{OpCode::STOREN, 1, 2},
                                           Instruction{OpCode::LOAD, 1, 0},
                                           Instruction{OpCode::OP, 0, 0},
                                           Instruction{OpCode::JF, 3, 0},
                                           Instruction{OpCode::LOADC, 1, 0},
                                           Instruction{OpCode::JMP, 3, 0},
                                           Instruction{OpCode::LOAD, 2, 0},
                                           Instruction{OpCode::OP, 0, 0},
                                           Instruction{OpCode::STORE, 3, 0},
                                           Instruction{OpCode::MOVE, 3, 0},
                                           Instruction{OpCode::JF, 5, 0},
                                           Instruction{OpCode::LOAD, 1, 0},
                                           Instruction{OpCode::LOAD, 2, 0},
                                           Instruction{OpCode::OP, 1, 0},
                                           Instruction{OpCode::JMP, 5, 0},
                                           Instruction{OpCode::LOAD, 1, 0},
                                           Instruction{OpCode::LOAD, 2, 0},
                                           Instruction{OpCode::LOADC, 0, 0},
                                           Instruction{OpCode::OP, 2, 0},
                                           Instruction{OpCode::STORE, 4, 0},
                                           Instruction{OpCode::DROPR, 2, 0},
                                           Instruction{OpCode::DROPR, 1, 0},
                                           Instruction{OpCode::MOVE, 4, 0},
                                           Instruction{OpCode::RET, 0, 0},
                                   }), // instructions list,
                               std::vector<c10::IValue>({
                                           c10::IValue("trunc"),
                                           c10::IValue(true),
                                   }), // constants list,
                               std::vector<c10::TypePtr>(), // types list,
                               4
                           ),
                           std::vector<OperatorString>({
                                   OperatorString({"aten::is_floating_point", "", 1}),
                                   OperatorString({"aten::div", "Tensor", 2}),
                                   OperatorString({"aten::div", "Tensor_mode", 3}),
                           }), // operators list
                   }),
                   ByteCodeFunctionWithOperator({
                           mobile::Function::registerFunc(
                               "div__Scalar_0_3",
                               std::vector<Instruction>({
                                           Instruction{OpCode::STOREN, 1, 2},
                                           Instruction{OpCode::LOAD, 1, 0},
                                           Instruction{OpCode::OP, 0, 0},
                                           Instruction{OpCode::JF, 3, 0},
                                           Instruction{OpCode::LOADC, 1, 0},
                                           Instruction{OpCode::JMP, 3, 0},
                                           Instruction{OpCode::LOAD, 2, 0},
                                           Instruction{OpCode::ISINSTANCE, 0, 1},
                                           Instruction{OpCode::STORE, 3, 0},
                                           Instruction{OpCode::MOVE, 3, 0},
                                           Instruction{OpCode::JF, 5, 0},
                                           Instruction{OpCode::LOAD, 1, 0},
                                           Instruction{OpCode::LOAD, 2, 0},
                                           Instruction{OpCode::OP, 1, 0},
                                           Instruction{OpCode::JMP, 6, 0},
                                           Instruction{OpCode::LOAD, 1, 0},
                                           Instruction{OpCode::LOAD, 2, 0},
                                           Instruction{OpCode::OP, 2, 0},
                                           Instruction{OpCode::LOADC, 0, 0},
                                           Instruction{OpCode::OP, 3, 0},
                                           Instruction{OpCode::STORE, 4, 0},
                                           Instruction{OpCode::DROPR, 2, 0},
                                           Instruction{OpCode::DROPR, 1, 0},
                                           Instruction{OpCode::MOVE, 4, 0},
                                           Instruction{OpCode::RET, 0, 0},
                                   }), // instructions list,
                               std::vector<c10::IValue>({
                                           c10::IValue("trunc"),
                                           c10::IValue(true),
                                   }), // constants list,
                               std::vector<c10::TypePtr>({
                                       c10::parseType("float"),
                                   }), // types list,
                               4
                           ),
                           std::vector<OperatorString>({
                                   OperatorString({"aten::is_floating_point", "", 1}),
                                   OperatorString({"aten::div_", "Scalar", 2}),
                                   OperatorString({"prim::unchecked_cast", "", 1}),
                                   OperatorString({"aten::div_", "Scalar_mode", 3}),
                           }), // operators list
                   }),
                   ByteCodeFunctionWithOperator({
                           mobile::Function::registerFunc(
                               "div__Tensor_0_3",
                               std::vector<Instruction>({
                                           Instruction{OpCode::STOREN, 1, 2},
                                           Instruction{OpCode::LOAD, 1, 0},
                                           Instruction{OpCode::OP, 0, 0},
                                           Instruction{OpCode::JF, 3, 0},
                                           Instruction{OpCode::LOADC, 1, 0},
                                           Instruction{OpCode::JMP, 3, 0},
                                           Instruction{OpCode::LOAD, 2, 0},
                                           Instruction{OpCode::OP, 0, 0},
                                           Instruction{OpCode::STORE, 3, 0},
                                           Instruction{OpCode::MOVE, 3, 0},
                                           Instruction{OpCode::JF, 5, 0},
                                           Instruction{OpCode::LOAD, 1, 0},
                                           Instruction{OpCode::LOAD, 2, 0},
                                           Instruction{OpCode::OP, 1, 0},
                                           Instruction{OpCode::JMP, 5, 0},
                                           Instruction{OpCode::LOAD, 1, 0},
                                           Instruction{OpCode::LOAD, 2, 0},
                                           Instruction{OpCode::LOADC, 0, 0},
                                           Instruction{OpCode::OP, 2, 0},
                                           Instruction{OpCode::STORE, 4, 0},
                                           Instruction{OpCode::DROPR, 2, 0},
                                           Instruction{OpCode::DROPR, 1, 0},
                                           Instruction{OpCode::MOVE, 4, 0},
                                           Instruction{OpCode::RET, 0, 0},
                                   }), // instructions list,
                               std::vector<c10::IValue>({
                                           c10::IValue("trunc"),
                                           c10::IValue(true),
                                   }), // constants list,
                               std::vector<c10::TypePtr>(), // types list,
                               4
                           ),
                           std::vector<OperatorString>({
                                   OperatorString({"aten::is_floating_point", "", 1}),
                                   OperatorString({"aten::div_", "Tensor", 2}),
                                   OperatorString({"aten::div_", "Tensor_mode", 3}),
                           }), // operators list
                   }),
                   ByteCodeFunctionWithOperator({
                           mobile::Function::registerFunc(
                               "div_out_0_3",
                               std::vector<Instruction>({
                                           Instruction{OpCode::STOREN, 1, 3},
                                           Instruction{OpCode::LOAD, 1, 0},
                                           Instruction{OpCode::OP, 0, 0},
                                           Instruction{OpCode::JF, 3, 0},
                                           Instruction{OpCode::LOADC, 1, 0},
                                           Instruction{OpCode::JMP, 3, 0},
                                           Instruction{OpCode::LOAD, 2, 0},
                                           Instruction{OpCode::OP, 0, 0},
                                           Instruction{OpCode::JF, 3, 0},
                                           Instruction{OpCode::LOADC, 1, 0},
                                           Instruction{OpCode::JMP, 3, 0},
                                           Instruction{OpCode::LOAD, 3, 0},
                                           Instruction{OpCode::OP, 0, 0},
                                           Instruction{OpCode::STORE, 4, 0},
                                           Instruction{OpCode::MOVE, 4, 0},
                                           Instruction{OpCode::JF, 6, 0},
                                           Instruction{OpCode::LOAD, 1, 0},
                                           Instruction{OpCode::LOAD, 2, 0},
                                           Instruction{OpCode::LOAD, 3, 0},
                                           Instruction{OpCode::OP, 1, 0},
                                           Instruction{OpCode::JMP, 6, 0},
                                           Instruction{OpCode::LOAD, 1, 0},
                                           Instruction{OpCode::LOAD, 2, 0},
                                           Instruction{OpCode::LOADC, 0, 0},
                                           Instruction{OpCode::LOAD, 3, 0},
                                           Instruction{OpCode::OP, 2, 0},
                                           Instruction{OpCode::STORE, 5, 0},
                                           Instruction{OpCode::DROPR, 3, 0},
                                           Instruction{OpCode::DROPR, 2, 0},
                                           Instruction{OpCode::DROPR, 1, 0},
                                           Instruction{OpCode::MOVE, 5, 0},
                                           Instruction{OpCode::RET, 0, 0},
                                   }), // instructions list,
                               std::vector<c10::IValue>({
                                           c10::IValue("trunc"),
                                           c10::IValue(true),
                                   }), // constants list,
                               std::vector<c10::TypePtr>(), // types list,
                               5
                           ),
                           std::vector<OperatorString>({
                                   OperatorString({"aten::is_floating_point", "", 1}),
                                   OperatorString({"aten::div", "out", 3}),
                                   OperatorString({"aten::div", "out_mode", 4}),
                           }), // operators list
                   }),
                   ByteCodeFunctionWithOperator({
                           mobile::Function::registerFunc(
                               "linspace_0_7",
                               std::vector<Instruction>({
                                           Instruction{OpCode::STOREN, 1, 7},
                                           Instruction{OpCode::LOAD, 3, 0},
                                           Instruction{OpCode::LOADC, 0, 0},
                                           Instruction{OpCode::OP, 0, 0},
                                           Instruction{OpCode::JF, 10, 0},
                                           Instruction{OpCode::LOAD, 1, 0},
                                           Instruction{OpCode::LOAD, 2, 0},
                                           Instruction{OpCode::LOADC, 1, 0},
                                           Instruction{OpCode::LOAD, 4, 0},
                                           Instruction{OpCode::LOAD, 5, 0},
                                           Instruction{OpCode::LOAD, 6, 0},
                                           Instruction{OpCode::LOAD, 7, 0},
                                           Instruction{OpCode::OP, 1, 0},
                                           Instruction{OpCode::JMP, 10, 0},
                                           Instruction{OpCode::LOAD, 1, 0},
                                           Instruction{OpCode::LOAD, 2, 0},
                                           Instruction{OpCode::LOAD, 3, 0},
                                           Instruction{OpCode::OP, 2, 0},
                                           Instruction{OpCode::LOAD, 4, 0},
                                           Instruction{OpCode::LOAD, 5, 0},
                                           Instruction{OpCode::LOAD, 6, 0},
                                           Instruction{OpCode::LOAD, 7, 0},
                                           Instruction{OpCode::OP, 1, 0},
                                           Instruction{OpCode::STORE, 8, 0},
                                           Instruction{OpCode::DROPR, 7, 0},
                                           Instruction{OpCode::DROPR, 6, 0},
                                           Instruction{OpCode::DROPR, 5, 0},
                                           Instruction{OpCode::DROPR, 4, 0},
                                           Instruction{OpCode::DROPR, 2, 0},
                                           Instruction{OpCode::DROPR, 1, 0},
                                           Instruction{OpCode::DROPR, 3, 0},
                                           Instruction{OpCode::MOVE, 8, 0},
                                           Instruction{OpCode::RET, 0, 0},
                                   }), // instructions list,
                               std::vector<c10::IValue>({
                                           c10::IValue(),
                                           c10::IValue(100),
                                   }), // constants list,
                               std::vector<c10::TypePtr>(), // types list,
                               8
                           ),
                           std::vector<OperatorString>({
                                   OperatorString({"aten::__is__", "", 2}),
                                   OperatorString({"aten::linspace", "", 7}),
                                   OperatorString({"prim::unchecked_cast", "", 1}),
                           }), // operators list
                   }),
                   ByteCodeFunctionWithOperator({
                           mobile::Function::registerFunc(
                               "linspace_out_0_7",
                               std::vector<Instruction>({
                                           Instruction{OpCode::STOREN, 1, 4},
                                           Instruction{OpCode::LOAD, 3, 0},
                                           Instruction{OpCode::LOADC, 0, 0},
                                           Instruction{OpCode::OP, 0, 0},
                                           Instruction{OpCode::JF, 7, 0},
                                           Instruction{OpCode::LOAD, 1, 0},
                                           Instruction{OpCode::LOAD, 2, 0},
                                           Instruction{OpCode::LOADC, 1, 0},
                                           Instruction{OpCode::LOAD, 4, 0},
                                           Instruction{OpCode::OP, 1, 0},
                                           Instruction{OpCode::JMP, 7, 0},
                                           Instruction{OpCode::LOAD, 1, 0},
                                           Instruction{OpCode::LOAD, 2, 0},
                                           Instruction{OpCode::LOAD, 3, 0},
                                           Instruction{OpCode::OP, 2, 0},
                                           Instruction{OpCode::LOAD, 4, 0},
                                           Instruction{OpCode::OP, 1, 0},
                                           Instruction{OpCode::STORE, 5, 0},
                                           Instruction{OpCode::DROPR, 4, 0},
                                           Instruction{OpCode::DROPR, 2, 0},
                                           Instruction{OpCode::DROPR, 1, 0},
                                           Instruction{OpCode::DROPR, 3, 0},
                                           Instruction{OpCode::MOVE, 5, 0},
                                           Instruction{OpCode::RET, 0, 0},
                                   }), // instructions list,
                               std::vector<c10::IValue>({
                                           c10::IValue(),
                                           c10::IValue(100),
                                   }), // constants list,
                               std::vector<c10::TypePtr>(), // types list,
                               5
                           ),
                           std::vector<OperatorString>({
                                   OperatorString({"aten::__is__", "", 2}),
                                   OperatorString({"aten::linspace", "out", 4}),
                                   OperatorString({"prim::unchecked_cast", "", 1}),
                           }), // operators list
                   }),
            });
    for (const auto& upgrader_function : upgrader_function_list) {
      for (const auto& op : upgrader_function.operators) {
        upgrader_function.function.append_operator(
            op.name,
            op.overload_name,
            op.num_specified_args,
            caffe2::serialize::kMaxSupportedFileFormatVersion);
      }
    }
    return upgrader_function_list;
  };
  static std::vector<ByteCodeFunctionWithOperator> upgraderBytecodeList =
      generate_upgrader_bytecode_list();
  return upgraderBytecodeList;
}

// clang-format on

} // namespace jit
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 1 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `jit`, `torch`, `c10`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/mobile`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/jit/mobile/upgrader_mobile.h`
- `ATen/core/ivalue.h`
- `caffe2/serialize/versions.h`
- `torch/csrc/jit/mobile/type_parser.h`


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

This is a test file. Run it with:

```bash
python test/mobile/test_upgrader_bytecode_table_example.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/mobile`):

- [`test_upgrader_codegen.py_docs.md`](./test_upgrader_codegen.py_docs.md)
- [`test_quantize_fx_lite_script_module.py_docs.md`](./test_quantize_fx_lite_script_module.py_docs.md)
- [`test_lite_script_module.py_docs.md`](./test_lite_script_module.py_docs.md)
- [`test_lite_script_type.py_docs.md`](./test_lite_script_type.py_docs.md)
- [`test_upgraders.py_docs.md`](./test_upgraders.py_docs.md)
- [`test_bytecode.py_docs.md`](./test_bytecode.py_docs.md)


## Cross-References

- **File Documentation**: `test_upgrader_bytecode_table_example.cpp_docs.md`
- **Keyword Index**: `test_upgrader_bytecode_table_example.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
