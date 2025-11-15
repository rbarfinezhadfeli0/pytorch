# Documentation: `docs/test/mobile/lightweight_dispatch/test_codegen_unboxing.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/mobile/lightweight_dispatch/test_codegen_unboxing.cpp_docs.md`
- **Size**: 10,030 bytes (9.79 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/mobile/lightweight_dispatch/test_codegen_unboxing.cpp`

## File Metadata

- **Path**: `test/mobile/lightweight_dispatch/test_codegen_unboxing.cpp`
- **Size**: 7,235 bytes (7.07 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/torch.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/module.h>
// Cover codegen'd unboxing logic for these types:
//'Device',
//'Device?',
//'Dimname',
//'Dimname[1]',
//'Dimname[]',
//'Dimname[]?',
//'Generator?',
//'Layout?',
//'MemoryFormat',
//'MemoryFormat?',
//'Scalar',
//'Scalar?',
//'ScalarType',
//'ScalarType?',
//'Scalar[]',
//'Storage',
//'Stream',
//'Tensor',
//'Tensor(a!)',
//'Tensor(a!)[]',
//'Tensor(a)',
//'Tensor(b!)',
//'Tensor(c!)',
//'Tensor(d!)',
//'Tensor?',
//'Tensor?[]',
//'Tensor[]',
//'bool',
//'bool?',
//'bool[2]',
//'bool[3]',
//'bool[4]',
//'float',
//'float?',
//'float[]?',
//'int',
//'int?',
//'int[1]',
//'int[1]?',
//'int[2]',
//'int[2]?',
//'int[3]',
//'int[4]',
//'int[5]',
//'int[6]',
//'int[]',
//'int[]?',
//'str',
//'str?'
namespace torch {
namespace jit {
namespace mobile {
// covers int[], ScalarType?, Layout?, Device?, bool?
TEST(LiteInterpreterTest, Ones) {
  // Load check in model: ModelWithDTypeDeviceLayoutPinMemory.ptl
  auto testModelFile = "ModelWithDTypeDeviceLayoutPinMemory.ptl";

  //  class ModelWithDTypeDeviceLayoutPinMemory(torch.nn.Module):
  //    def forward(self, x: int):
  //        a = torch.ones([3, x], dtype=torch.int64, layout=torch.strided, device="cpu")
  //        return a
  Module bc = _load_for_mobile(testModelFile);
  std::vector<c10::IValue> input{c10::IValue(4)};
  const auto result = bc.forward(input);
  ASSERT_EQ(result.toTensor().size(0), 3);
  ASSERT_EQ(result.toTensor().size(1), 4);
}

TEST(LiteInterpreterTest, Index) {
  // Load check in model: ModelWithTensorOptional.ptl
  auto testModelFile = "ModelWithTensorOptional.ptl";

  //    class ModelWithTensorOptional(torch.nn.Module):
  //      def forward(self, index):
  //        a = torch.zeros(2, 2)
  //        a[0][1] = 1
  //        a[1][0] = 2
  //        a[1][1] = 3
  //        return a[index]
  Module bc = _load_for_mobile(testModelFile);
  int64_t ind_1 = 0;

  const auto result_1 = bc.forward({at::tensor(ind_1)});

  at::Tensor expected = at::empty({1, 2}, c10::TensorOptions(c10::ScalarType::Float));
  expected[0][0] = 0;
  expected[0][1] = 1;

  AT_ASSERT(result_1.toTensor().equal(expected));
}

TEST(LiteInterpreterTest, Gradient) {
  // Load check in model: ModelWithScalarList.ptl
  auto testModelFile = "ModelWithScalarList.ptl";

  //    class ModelWithScalarList(torch.nn.Module):
  //      def forward(self, a: int):
  //        values = torch.tensor([4., 1., 1., 16.], )
  //        if a == 0:
  //          return torch.gradient(values, spacing=torch.scalar_tensor(2., dtype=torch.float64))
  //        elif a == 1:
  //          return torch.gradient(values, spacing=[torch.tensor(1.).item()])
  Module bc = _load_for_mobile(testModelFile);

  const auto result_1 = bc.forward({0});
  at::Tensor expected_1 = at::tensor({-1.5, -0.75, 3.75, 7.5}, c10::TensorOptions(c10::ScalarType::Float));
  AT_ASSERT(result_1.toList().get(0).toTensor().equal(expected_1));

  const auto result_2 = bc.forward({1});
  at::Tensor expected_2 = at::tensor({-3.0, -1.5, 7.5, 15.0}, c10::TensorOptions(c10::ScalarType::Float));
  AT_ASSERT(result_2.toList().get(0).toTensor().equal(expected_2));
}

TEST(LiteInterpreterTest, Upsample) {
  // Load check in model: ModelWithFloatList.ptl
  auto testModelFile = "ModelWithFloatList.ptl";

  // model = torch.nn.Upsample(scale_factor=(2.0,), mode="linear")
  Module bc = _load_for_mobile(testModelFile);

  const auto result_1 = bc.forward({at::ones({1, 2, 3})});
  at::Tensor expected_1 = at::ones({1, 2, 6}, c10::TensorOptions(c10::ScalarType::Float));
  AT_ASSERT(result_1.toTensor().equal(expected_1));
}

TEST(LiteInterpreterTest, IndexTensor) {
  // Load check in model: ModelWithListOfOptionalTensors.ptl
  auto testModelFile = "ModelWithListOfOptionalTensors.ptl";

  // class ModelWithListOfOptionalTensors(torch.nn.Module):
  //   def forward(self, index):
  //      values = torch.tensor([4., 1., 1., 16.], )
  //      return values[[index, torch.tensor(0)]]
  Module bc = _load_for_mobile(testModelFile);
  const auto result_1 = bc.forward({at::tensor({1}, c10::TensorOptions(c10::ScalarType::Long))});

  at::Tensor expected_1 = at::tensor({1.}, c10::TensorOptions(c10::ScalarType::Float));
  AT_ASSERT(result_1.toTensor().equal(expected_1));
}

TEST(LiteInterpreterTest, Conv2d) {
  // Load check in model: ModelWithArrayOfInt.ptl
  auto testModelFile = "ModelWithArrayOfInt.ptl";

  // model = torch.nn.Conv2d(1, 2, (2, 2), stride=(1, 1), padding=(1, 1))
  Module bc = _load_for_mobile(testModelFile);
  const auto result_1 = bc.forward({at::ones({1, 1, 1, 1})});

  ASSERT_EQ(result_1.toTensor().sizes(), c10::IntArrayRef ({1,2,2,2}));
}

TEST(LiteInterpreterTest, AddTensor) {
  // Load check in model: ModelWithTensors.ptl
  auto testModelFile = "ModelWithTensors.ptl";

  //  class ModelWithTensors(torch.nn.Module):
  //    def forward(self, a):
  //      values = torch.ones(size=[2, 3], names=['N', 'C'])
  //      values[0][0] = a[0]
  //      return values
  Module bc = _load_for_mobile(testModelFile);
  const auto result_1 = bc.forward({at::tensor({1, 2, 3}, c10::TensorOptions(c10::ScalarType::Long))});

  at::Tensor expected_1 = at::tensor({2, 3, 4}, c10::TensorOptions(c10::ScalarType::Long));
  AT_ASSERT(result_1.toTensor().equal(expected_1));
}

TEST(LiteInterpreterTest, DivideTensor) {
  // Load check in model: ModelWithStringOptional.ptl
  auto testModelFile = "ModelWithStringOptional.ptl";

  //  class ModelWithStringOptional(torch.nn.Module):
  //    def forward(self, b):
  //      a = torch.tensor(3, dtype=torch.int64)
  //      out = torch.empty(size=[1], dtype=torch.float)
  //      torch.div(b, a, out=out)
  //      return [torch.div(b, a, rounding_mode='trunc'), out]
  Module bc = _load_for_mobile(testModelFile);
  const auto result_1 = bc.forward({at::tensor({-12}, c10::TensorOptions(c10::ScalarType::Long))});

  at::Tensor expected_1 = at::tensor({-4}, c10::TensorOptions(c10::ScalarType::Long));
  at::Tensor expected_2 = at::tensor({-4.}, c10::TensorOptions(c10::ScalarType::Float));
  AT_ASSERT(result_1.toList().get(0).toTensor().equal(expected_1));
  AT_ASSERT(result_1.toList().get(1).toTensor().equal(expected_2));
}

TEST(LiteInterpreterTest, MultipleOps) {
  // Load check in model: ModelWithMultipleOps.ptl
  auto testModelFile = "ModelWithMultipleOps.ptl";

  // class ModelWithMultipleOps(torch.nn.Module):
  //     def __init__(self) -> None:
  //         super().__init__()
  //         self.ops = torch.nn.Sequential(
  //             torch.nn.ReLU(),
  //             torch.nn.Flatten(),
  //         )
  //
  //     def forward(self, x):
  //         x[1] = -2
  //         return self.ops(x)

  Module bc = _load_for_mobile(testModelFile);
  auto b = at::ones({2, 2, 2, 2});
  const auto result = bc.forward({b});

  at::Tensor expected = torch::tensor({{1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 0, 0}}, c10::TensorOptions(c10::ScalarType::Float));
  AT_ASSERT(result.toTensor().equal(expected));
}
} // namespace mobile
} // namespace jit
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 7 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `jit`, `mobile`, `torch`

**Classes/Structs**: `ModelWithDTypeDeviceLayoutPinMemory`, `ModelWithTensorOptional`, `ModelWithScalarList`, `ModelWithListOfOptionalTensors`, `ModelWithTensors`, `ModelWithStringOptional`, `ModelWithMultipleOps`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/mobile/lightweight_dispatch`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `test/cpp/jit/test_utils.h`
- `torch/torch.h`
- `torch/csrc/jit/api/module.h`
- `torch/csrc/jit/frontend/resolver.h`
- `torch/csrc/jit/mobile/import.h`
- `torch/csrc/jit/mobile/module.h`


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
python test/mobile/lightweight_dispatch/test_codegen_unboxing.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/mobile/lightweight_dispatch`):

- [`build.sh_docs.md`](./build.sh_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`lightweight_dispatch_ops.yaml_docs.md`](./lightweight_dispatch_ops.yaml_docs.md)
- [`test_lightweight_dispatch.cpp_docs.md`](./test_lightweight_dispatch.cpp_docs.md)
- [`tests_setup.py_docs.md`](./tests_setup.py_docs.md)


## Cross-References

- **File Documentation**: `test_codegen_unboxing.cpp_docs.md`
- **Keyword Index**: `test_codegen_unboxing.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/mobile/lightweight_dispatch`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/mobile/lightweight_dispatch`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/mobile/lightweight_dispatch/test_codegen_unboxing.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/mobile/lightweight_dispatch`):

- [`tests_setup.py_docs.md_docs.md`](./tests_setup.py_docs.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`test_lightweight_dispatch.cpp_kw.md_docs.md`](./test_lightweight_dispatch.cpp_kw.md_docs.md)
- [`lightweight_dispatch_ops.yaml_kw.md_docs.md`](./lightweight_dispatch_ops.yaml_kw.md_docs.md)
- [`lightweight_dispatch_ops.yaml_docs.md_docs.md`](./lightweight_dispatch_ops.yaml_docs.md_docs.md)
- [`tests_setup.py_kw.md_docs.md`](./tests_setup.py_kw.md_docs.md)
- [`build.sh_docs.md_docs.md`](./build.sh_docs.md_docs.md)
- [`test_lightweight_dispatch.cpp_docs.md_docs.md`](./test_lightweight_dispatch.cpp_docs.md_docs.md)
- [`build.sh_kw.md_docs.md`](./build.sh_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_codegen_unboxing.cpp_docs.md_docs.md`
- **Keyword Index**: `test_codegen_unboxing.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
