# Documentation: `docs/test/cpp/api/jit.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/api/jit.cpp_docs.md`
- **Size**: 6,132 bytes (5.99 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/cpp/api/jit.cpp`

## File Metadata

- **Path**: `test/cpp/api/jit.cpp`
- **Size**: 3,892 bytes (3.80 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <torch/jit.h>
#include <torch/script.h>
#include <torch/types.h>

#include <string>

TEST(TorchScriptTest, CanCompileMultipleFunctions) {
  auto module = torch::jit::compile(R"JIT(
      def test_mul(a, b):
        return a * b
      def test_relu(a, b):
        return torch.relu(a + b)
      def test_while(a, i):
        while bool(i < 10):
          a += a
          i += 1
        return a
      def test_len(a : List[int]):
        return len(a)
    )JIT");
  auto a = torch::ones(1);
  auto b = torch::ones(1);

  ASSERT_EQ(1, module->run_method("test_mul", a, b).toTensor().item<int64_t>());

  ASSERT_EQ(
      2, module->run_method("test_relu", a, b).toTensor().item<int64_t>());

  ASSERT_TRUE(
      0x200 ==
      module->run_method("test_while", a, b).toTensor().item<int64_t>());

  at::IValue list = c10::List<int64_t>({3, 4});
  ASSERT_EQ(2, module->run_method("test_len", list).toInt());
}

TEST(TorchScriptTest, TestNestedIValueModuleArgMatching) {
  auto module = torch::jit::compile(R"JIT(
      def nested_loop(a: List[List[Tensor]], b: int):
        return torch.tensor(1.0) + b
    )JIT");

  auto b = 3;

  torch::List<torch::Tensor> list({torch::rand({4, 4})});

  torch::List<torch::List<torch::Tensor>> list_of_lists;
  list_of_lists.push_back(list);
  module->run_method("nested_loop", list_of_lists, b);

  auto generic_list = c10::impl::GenericList(at::TensorType::get());
  auto empty_generic_list =
      c10::impl::GenericList(at::ListType::create(at::TensorType::get()));
  empty_generic_list.push_back(generic_list);
  module->run_method("nested_loop", empty_generic_list, b);

  auto too_many_lists = c10::impl::GenericList(
      at::ListType::create(at::ListType::create(at::TensorType::get())));
  too_many_lists.push_back(empty_generic_list);
  try {
    module->run_method("nested_loop", too_many_lists, b);
    AT_ASSERT(false);
  } catch (const c10::Error& error) {
    AT_ASSERT(
        std::string(error.what_without_backtrace())
            .find("nested_loop() Expected a value of type 'List[List[Tensor]]'"
                  " for argument 'a' but instead found type "
                  "'List[List[List[Tensor]]]'") == 0);
  };
}

TEST(TorchScriptTest, TestDictArgMatching) {
  auto module = torch::jit::compile(R"JIT(
      def dict_op(a: Dict[str, Tensor], b: str):
        return a[b]
    )JIT");
  c10::Dict<std::string, at::Tensor> dict;
  dict.insert("hello", torch::ones({2}));
  auto output = module->run_method("dict_op", dict, std::string("hello"));
  ASSERT_EQ(1, output.toTensor()[0].item<int64_t>());
}

TEST(TorchScriptTest, TestTupleArgMatching) {
  auto module = torch::jit::compile(R"JIT(
      def tuple_op(a: Tuple[List[int]]):
        return a
    )JIT");

  c10::List<int64_t> int_list({1});
  auto tuple_generic_list = c10::ivalue::Tuple::create({int_list});

  // doesn't fail on arg matching
  module->run_method("tuple_op", tuple_generic_list);
}

TEST(TorchScriptTest, TestOptionalArgMatching) {
  auto module = torch::jit::compile(R"JIT(
      def optional_tuple_op(a: Optional[Tuple[int, str]]):
        if a is None:
          return 0
        else:
          return a[0]
    )JIT");

  auto optional_tuple = c10::ivalue::Tuple::create({2, std::string("hi")});

  ASSERT_EQ(2, module->run_method("optional_tuple_op", optional_tuple).toInt());
  ASSERT_EQ(
      0, module->run_method("optional_tuple_op", torch::jit::IValue()).toInt());
}

TEST(TorchScriptTest, TestPickle) {
  torch::IValue float_value(2.3);

  // TODO: when tensors are stored in the pickle, delete this
  std::vector<at::Tensor> tensor_table;
  auto data = torch::jit::pickle(float_value, &tensor_table);

  torch::IValue ivalue = torch::jit::unpickle(data.data(), data.size());

  double diff = ivalue.toDouble() - float_value.toDouble();
  double eps = 0.0001;
  ASSERT_TRUE(diff < eps && diff > -eps);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/api`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `torch/jit.h`
- `torch/script.h`
- `torch/types.h`
- `string`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp/api/jit.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/api`):

- [`fft.cpp_docs.md`](./fft.cpp_docs.md)
- [`tensor_options.cpp_docs.md`](./tensor_options.cpp_docs.md)
- [`any.cpp_docs.md`](./any.cpp_docs.md)
- [`torch_include.cpp_docs.md`](./torch_include.cpp_docs.md)
- [`rnn.cpp_docs.md`](./rnn.cpp_docs.md)
- [`nn_utils.cpp_docs.md`](./nn_utils.cpp_docs.md)
- [`nested.cpp_docs.md`](./nested.cpp_docs.md)
- [`meta_tensor.cpp_docs.md`](./meta_tensor.cpp_docs.md)
- [`nested_int.cpp_docs.md`](./nested_int.cpp_docs.md)


## Cross-References

- **File Documentation**: `jit.cpp_docs.md`
- **Keyword Index**: `jit.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp/api`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp/api`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/cpp/api/jit.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/api`):

- [`init_baseline.py_kw.md_docs.md`](./init_baseline.py_kw.md_docs.md)
- [`support.cpp_kw.md_docs.md`](./support.cpp_kw.md_docs.md)
- [`memory.cpp_docs.md_docs.md`](./memory.cpp_docs.md_docs.md)
- [`parallel_benchmark.cpp_docs.md_docs.md`](./parallel_benchmark.cpp_docs.md_docs.md)
- [`dataloader.cpp_docs.md_docs.md`](./dataloader.cpp_docs.md_docs.md)
- [`moduledict.cpp_kw.md_docs.md`](./moduledict.cpp_kw.md_docs.md)
- [`support.h_kw.md_docs.md`](./support.h_kw.md_docs.md)
- [`ordered_dict.cpp_docs.md_docs.md`](./ordered_dict.cpp_docs.md_docs.md)
- [`functional.cpp_docs.md_docs.md`](./functional.cpp_docs.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)


## Cross-References

- **File Documentation**: `jit.cpp_docs.md_docs.md`
- **Keyword Index**: `jit.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
