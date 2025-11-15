# Documentation: `test/cpp/lite_interpreter_runtime/test_lite_interpreter_runtime.cpp`

## File Metadata

- **Path**: `test/cpp/lite_interpreter_runtime/test_lite_interpreter_runtime.cpp`
- **Size**: 5,814 bytes (5.68 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <ATen/Functions.h>
#include <aten/src/ATen/TensorOperators.h>
#include <gtest/gtest.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>

#include <unordered_set>

namespace torch {
namespace jit {
namespace mobile {

TEST(RunTimeTest, LoadAndForward) {
  // Load check in model: sequence.ptl
  std::string filePath(__FILE__);
  auto testModelFile = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  testModelFile.append("sequence.ptl");

  //  sequence.ptl source code:
  //  class A(torch.nn.Module):
  //    def __init__(self) -> None:
  //      super().__init__()
  //
  //    def forward(self, x):
  //      return x + 1
  //
  //  class B(torch.nn.Module):
  //    def __init__(self) -> None:
  //      super().__init__()
  //
  //    def forward(self, x):
  //      return x + 2
  //
  //  class C(torch.nn.Module):
  //    def __init__(self) -> None:
  //      super().__init__()
  //      self.A0 = A()
  //      self.B0 = B()
  //
  //    def forward(self, x):
  //      return self.A0.forward(self.B0.forward(x))

  Module bc = _load_for_mobile(testModelFile);
  auto forward_method = bc.find_method("forward");
  std::vector<c10::IValue> input{c10::IValue(at::tensor(1))};
  const auto result = bc.forward(input);
  const auto expected_result = c10::IValue(at::tensor(4));
  ASSERT_EQ(result, expected_result);
}

TEST(RunTimeTest, Delegate) {
  std::string filePath(__FILE__);
  auto testModelFile = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  // "delegate_test.ptl" is generated from test/cpp/jit/test_backend.cpp,
  // BackendTest.TestCompiler. This test is on target runtime. It has
  // model running capability, but no compilation and serialization.
  // The mobile model delegated to the "backend_with_compiler_demo" backend
  // The model is from the jit code:
  //  Module m("m");
  //  m.define(R"(
  //    def forward(self, x, h):
  //        return x + h
  //  )");
  testModelFile.append("delegate_test.ptl");
  auto mlm = _load_for_mobile(testModelFile);
  std::vector<IValue> inputs;
  inputs.emplace_back(2.0 * at::ones({}));
  inputs.emplace_back(1.0 * at::ones({}));

  auto mres = mlm.forward(inputs);
  AT_ASSERT(mres.toTensor().equal(3 * at::ones({})));
}

TEST(RunTimeTest, DelegateException) {
  std::string filePath(__FILE__);
  auto testModelFile = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  /*
   * Model: delegated_submodule_with_debug_info.ptl
   * Model structure:
   * def AA(..):
   *   def forward(self, x, y):
   *     return x + y
   *
   * def A(..):
   *   def __init__(..):
   *     self.AA0 = AA()
   *   def forward(self, x, y):
   *     return self.AA0.forward(x, y) + 3
   *
   * def B(..):
   *   def forward(self, x):
   *     return x + 2
   *
   * def C(..):
   *   def __init__(..):
   *     self.A0 = A()
   *     self.B0 = B()
   *   def forward(self, x, y):
   *     return self.A0.forward(x, y) + self.B0.forward(x)
   *
   * std::vector<IValue> inputs;
   * inputs.emplace_back(torch::rand({2, 4}));
   * inputs.emplace_back(torch::rand({13, 9}));
   * Run with inputs and expect exception
   * Error stack trace will look like this:
   * Module hierarchy:top(C).A0(backend_with_compiler_demoLoweredModule).AA0(AA)
   * Traceback of TorchScript (most recent call last):
   *  File "<string>", line 3, in FunctionName_UNKNOWN
   *
   *    def forward(self, x, y):
   *      return self.A0.forward(x, y) + self.B0.forward(x)
   *             ~~~~~~~~~~~~~~~ <--- HERE
   *
   *  File "<string>", line 5, in FunctionName_UNKNOWN
   *                typed_inputs: List[Any] = [x, y, ]
   *                if self.__backend.is_available() :
   *                  _0, = self.__backend.execute(self.__handles["forward"],
   * typed_inputs)
   *                        ~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
   *                  assert isinstance(_0, Tensor)
   *                  return _0
   *  File "<string>", line 3, in FunctionName_UNKNOWN
   *
   *    def forward(self, x, y):
   *      return self.AA0.forward(x, y) + 3
   *             ~~~~~~~~~~~~~~~~ <--- HERE
   *
   *  File "<string>", line 3, in FunctionName_UNKNOWN
   *
   *    def forward(self, x, y):
   *      return x + y
   *             ~~~~~ <--- HERE
   *
   *
   */
  testModelFile.append("delegated_submodule_with_debug_info.ptl");
  auto mlm = _load_for_mobile(testModelFile);
  std::vector<IValue> inputs;
  inputs.emplace_back(torch::rand({2, 4}));
  inputs.emplace_back(torch::rand({13, 9}));

  std::string error_pattern = R"(
  Module hierarchy:top(C)::<unknown>.A0(backend_with_compiler_demoLoweredModule)::forward.AA0(AA)::forward.aten::add
Traceback of TorchScript (most recent call last):
  File "<string>", line 3, in <unknown>

    def forward(self, x, y):
      return self.A0.forward(x, y) + self.B0.forward(x)
             ~~~~~~~~~~~~~~~ <--- HERE

  File "<string>", line 5, in forward
                typed_inputs: List[Any] = [x, y, ]
                if self.__backend.is_available() :
                  _0, = self.__backend.execute(self.__handles["forward"], typed_inputs)
                        ~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
                  assert isinstance(_0, Tensor)
                  return _0
  File "<string>", line 3, in <unknown>

    def forward(self, x, y):
      return self.AA0.forward(x, y) + 3
             ~~~~~~~~~~~~~~~~ <--- HERE

  File "<string>", line 3, in forward

    def forward(self, x, y):
      return x + y
             ~~~~~ <--- HERE
  )";
  ASSERT_THROWS_WITH_MESSAGE(mlm.forward(inputs), error_pattern);
}
} // namespace mobile
} // namespace jit
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 34 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `jit`, `mobile`, `torch`

**Classes/Structs**: `A`, `B`, `C`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/lite_interpreter_runtime`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Functions.h`
- `aten/src/ATen/TensorOperators.h`
- `gtest/gtest.h`
- `test/cpp/jit/test_utils.h`
- `torch/csrc/autograd/generated/variable_factories.h`
- `torch/csrc/jit/api/module.h`
- `torch/csrc/jit/frontend/resolver.h`
- `torch/csrc/jit/mobile/import.h`
- `torch/csrc/jit/mobile/module.h`
- `unordered_set`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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
python test/cpp/lite_interpreter_runtime/test_lite_interpreter_runtime.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/lite_interpreter_runtime`):

- [`resources.h_docs.md`](./resources.h_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`test_mobile_profiler.cpp_docs.md`](./test_mobile_profiler.cpp_docs.md)
- [`main.cpp_docs.md`](./main.cpp_docs.md)


## Cross-References

- **File Documentation**: `test_lite_interpreter_runtime.cpp_docs.md`
- **Keyword Index**: `test_lite_interpreter_runtime.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
