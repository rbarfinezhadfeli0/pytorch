# Documentation: `docs/test/cpp/api/ordered_dict.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/api/ordered_dict.cpp_docs.md`
- **Size**: 9,206 bytes (8.99 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/cpp/api/ordered_dict.cpp`

## File Metadata

- **Path**: `test/cpp/api/ordered_dict.cpp`
- **Size**: 6,968 bytes (6.80 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <test/cpp/api/support.h>
#include <torch/torch.h>

template <typename T>
using OrderedDict = torch::OrderedDict<std::string, T>;

TEST(OrderedDictTest, IsEmptyAfterDefaultConstruction) {
  OrderedDict<int> dict;
  ASSERT_EQ(dict.key_description(), "Key");
  ASSERT_TRUE(dict.is_empty());
  ASSERT_EQ(dict.size(), 0);
}

TEST(OrderedDictTest, InsertAddsElementsWhenTheyAreYetNotPresent) {
  OrderedDict<int> dict;
  dict.insert("a", 1);
  dict.insert("b", 2);
  ASSERT_EQ(dict.size(), 2);
}

TEST(OrderedDictTest, GetReturnsValuesWhenTheyArePresent) {
  OrderedDict<int> dict;
  dict.insert("a", 1);
  dict.insert("b", 2);
  ASSERT_EQ(dict["a"], 1);
  ASSERT_EQ(dict["b"], 2);
}

TEST(OrderedDictTest, GetThrowsWhenPassedKeysThatAreNotPresent) {
  OrderedDict<int> dict;
  dict.insert("a", 1);
  dict.insert("b", 2);
  ASSERT_THROWS_WITH(dict["foo"], "Key 'foo' is not defined");
  ASSERT_THROWS_WITH(dict[""], "Key '' is not defined");
}

TEST(OrderedDictTest, CanInitializeFromList) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_EQ(dict.size(), 2);
  ASSERT_EQ(dict["a"], 1);
  ASSERT_EQ(dict["b"], 2);
}

TEST(OrderedDictTest, InsertThrowsWhenPassedElementsThatArePresent) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_THROWS_WITH(dict.insert("a", 1), "Key 'a' already defined");
  ASSERT_THROWS_WITH(dict.insert("b", 1), "Key 'b' already defined");
}

TEST(OrderedDictTest, FrontReturnsTheFirstItem) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_EQ(dict.front().key(), "a");
  ASSERT_EQ(dict.front().value(), 1);
}

TEST(OrderedDictTest, FrontThrowsWhenEmpty) {
  OrderedDict<int> dict;
  ASSERT_THROWS_WITH(dict.front(), "Called front() on an empty OrderedDict");
}

TEST(OrderedDictTest, BackReturnsTheLastItem) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_EQ(dict.back().key(), "b");
  ASSERT_EQ(dict.back().value(), 2);
}

TEST(OrderedDictTest, BackThrowsWhenEmpty) {
  OrderedDict<int> dict;
  ASSERT_THROWS_WITH(dict.back(), "Called back() on an empty OrderedDict");
}

TEST(OrderedDictTest, FindReturnsPointersToValuesWhenPresent) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_NE(dict.find("a"), nullptr);
  ASSERT_EQ(*dict.find("a"), 1);
  ASSERT_NE(dict.find("b"), nullptr);
  ASSERT_EQ(*dict.find("b"), 2);
}

TEST(OrderedDictTest, FindReturnsNullPointersWhenPasesdKeysThatAreNotPresent) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_EQ(dict.find("bar"), nullptr);
  ASSERT_EQ(dict.find(""), nullptr);
}

TEST(OrderedDictTest, SubscriptOperatorThrowsWhenPassedKeysThatAreNotPresent) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_EQ(dict["a"], 1);
  ASSERT_EQ(dict["b"], 2);
}

TEST(
    OrderedDictTest,
    SubscriptOperatorReturnsItemsPositionallyWhenPassedIntegers) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_EQ(dict[0].key(), "a");
  ASSERT_EQ(dict[0].value(), 1);
  ASSERT_EQ(dict[1].key(), "b");
  ASSERT_EQ(dict[1].value(), 2);
}

TEST(OrderedDictTest, SubscriptOperatorsThrowswhenPassedKeysThatAreNotPresent) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_THROWS_WITH(dict["foo"], "Key 'foo' is not defined");
  ASSERT_THROWS_WITH(dict[""], "Key '' is not defined");
}

TEST(OrderedDictTest, UpdateInsertsAllItemsFromAnotherOrderedDict) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  OrderedDict<int> dict2 = {{"c", 3}};
  dict2.update(dict);
  ASSERT_EQ(dict2.size(), 3);
  ASSERT_NE(dict2.find("a"), nullptr);
  ASSERT_NE(dict2.find("b"), nullptr);
  ASSERT_NE(dict2.find("c"), nullptr);
}

TEST(OrderedDictTest, UpdateAlsoChecksForDuplicates) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  OrderedDict<int> dict2 = {{"a", 1}};
  ASSERT_THROWS_WITH(dict2.update(dict), "Key 'a' already defined");
}

TEST(OrderedDictTest, CanIterateItems) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  auto iterator = dict.begin();
  ASSERT_NE(iterator, dict.end());
  ASSERT_EQ(iterator->key(), "a");
  ASSERT_EQ(iterator->value(), 1);
  ++iterator;
  ASSERT_NE(iterator, dict.end());
  ASSERT_EQ(iterator->key(), "b");
  ASSERT_EQ(iterator->value(), 2);
  ++iterator;
  ASSERT_EQ(iterator, dict.end());
}

TEST(OrderedDictTest, EraseWorks) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}, {"c", 3}};
  dict.erase("b");
  ASSERT_FALSE(dict.contains("b"));
  ASSERT_EQ(dict["a"], 1);
  ASSERT_EQ(dict["c"], 3);
  dict.erase("a");
  ASSERT_FALSE(dict.contains("a"));
  ASSERT_EQ(dict["c"], 3);
  dict.erase("c");
  ASSERT_FALSE(dict.contains("c"));
  ASSERT_TRUE(dict.is_empty());
}

TEST(OrderedDictTest, ClearMakesTheDictEmpty) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_FALSE(dict.is_empty());
  dict.clear();
  ASSERT_TRUE(dict.is_empty());
}

TEST(OrderedDictTest, CanCopyConstruct) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  OrderedDict<int> copy = dict;
  ASSERT_EQ(copy.size(), 2);
  ASSERT_EQ(*copy[0], 1);
  ASSERT_EQ(*copy[1], 2);
}

TEST(OrderedDictTest, CanCopyAssign) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  OrderedDict<int> copy = {{"c", 1}};
  ASSERT_NE(copy.find("c"), nullptr);
  copy = dict;
  ASSERT_EQ(copy.size(), 2);
  ASSERT_EQ(*copy[0], 1);
  ASSERT_EQ(*copy[1], 2);
  ASSERT_EQ(copy.find("c"), nullptr);
}

TEST(OrderedDictTest, CanMoveConstruct) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  OrderedDict<int> copy = std::move(dict);
  ASSERT_EQ(copy.size(), 2);
  ASSERT_EQ(*copy[0], 1);
  ASSERT_EQ(*copy[1], 2);
}

TEST(OrderedDictTest, CanMoveAssign) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  OrderedDict<int> copy = {{"c", 1}};
  ASSERT_NE(copy.find("c"), nullptr);
  copy = std::move(dict);
  ASSERT_EQ(copy.size(), 2);
  ASSERT_EQ(*copy[0], 1);
  ASSERT_EQ(*copy[1], 2);
  ASSERT_EQ(copy.find("c"), nullptr);
}

TEST(OrderedDictTest, CanInsertWithBraces) {
  OrderedDict<std::pair<int, int>> dict;
  dict.insert("a", {1, 2});
  ASSERT_FALSE(dict.is_empty());
  ASSERT_EQ(dict["a"].first, 1);
  ASSERT_EQ(dict["a"].second, 2);
}

TEST(OrderedDictTest, ErrorMessagesIncludeTheKeyDescription) {
  OrderedDict<int> dict("Penguin");
  ASSERT_EQ(dict.key_description(), "Penguin");
  dict.insert("a", 1);
  ASSERT_FALSE(dict.is_empty());
  ASSERT_THROWS_WITH(dict["b"], "Penguin 'b' is not defined");
  ASSERT_THROWS_WITH(dict.insert("a", 1), "Penguin 'a' already defined");
}

TEST(OrderedDictTest, KeysReturnsAllKeys) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_EQ(dict.keys(), std::vector<std::string>({"a", "b"}));
}

TEST(OrderedDictTest, ValuesReturnsAllValues) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_EQ(dict.values(), std::vector<int>({1, 2}));
}

TEST(OrderedDictTest, ItemsReturnsAllItems) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  std::vector<OrderedDict<int>::Item> items = dict.items();
  ASSERT_EQ(items.size(), 2);
  ASSERT_EQ(items[0].key(), "a");
  ASSERT_EQ(items[0].value(), 1);
  ASSERT_EQ(items[1].key(), "b");
  ASSERT_EQ(items[1].value(), 2);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

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
- `test/cpp/api/support.h`
- `torch/torch.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp/api/ordered_dict.cpp
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
- [`jit.cpp_docs.md`](./jit.cpp_docs.md)
- [`nn_utils.cpp_docs.md`](./nn_utils.cpp_docs.md)
- [`nested.cpp_docs.md`](./nested.cpp_docs.md)
- [`meta_tensor.cpp_docs.md`](./meta_tensor.cpp_docs.md)
- [`nested_int.cpp_docs.md`](./nested_int.cpp_docs.md)


## Cross-References

- **File Documentation**: `ordered_dict.cpp_docs.md`
- **Keyword Index**: `ordered_dict.cpp_kw.md`
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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/cpp/api/ordered_dict.cpp_docs.md
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
- [`functional.cpp_docs.md_docs.md`](./functional.cpp_docs.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)


## Cross-References

- **File Documentation**: `ordered_dict.cpp_docs.md_docs.md`
- **Keyword Index**: `ordered_dict.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
