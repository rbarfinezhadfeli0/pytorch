# Documentation: `docs/aten/src/ATen/test/Dict_test.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/test/Dict_test.cpp_docs.md`
- **Size**: 17,443 bytes (17.03 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/test/Dict_test.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/Dict_test.cpp`
- **Size**: 14,916 bytes (14.57 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <ATen/core/Dict.h>
#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <string>

using std::string;
using c10::Dict;

#define ASSERT_EQUAL(t1, t2) ASSERT_TRUE(t1.equal(t2));

TEST(DictTest, givenEmptyDict_whenCallingEmpty_thenReturnsTrue) {
    Dict<int64_t, string> dict;
    EXPECT_TRUE(dict.empty());
}

TEST(DictTest, givenNonemptyDict_whenCallingEmpty_thenReturnsFalse) {
    Dict<int64_t, string> dict;
    dict.insert(3, "value");
    EXPECT_FALSE(dict.empty());
}

TEST(DictTest, givenEmptyDict_whenCallingSize_thenReturnsZero) {
    Dict<int64_t, string> dict;
    EXPECT_EQ(0, dict.size());
}

TEST(DictTest, givenNonemptyDict_whenCallingSize_thenReturnsNumberOfElements) {
    Dict<int64_t, string> dict;
    dict.insert(3, "value");
    dict.insert(4, "value2");
    EXPECT_EQ(2, dict.size());
}

TEST(DictTest, givenNonemptyDict_whenCallingClear_thenIsEmpty) {
  Dict<int64_t, string> dict;
  dict.insert(3, "value");
  dict.insert(4, "value2");
  dict.clear();
  EXPECT_TRUE(dict.empty());
}

TEST(DictTest, whenInsertingNewKey_thenReturnsTrueAndIteratorToNewElement) {
  Dict<int64_t, string> dict;
  std::pair<Dict<int64_t, string>::iterator, bool> result = dict.insert(3, "value");
  EXPECT_TRUE(result.second);
  EXPECT_EQ(3, result.first->key());
  EXPECT_EQ("value", result.first->value());
}

TEST(DictTest, whenInsertingExistingKey_thenReturnsFalseAndIteratorToExistingElement) {
  Dict<int64_t, string> dict;
  dict.insert(3, "old_value");
  std::pair<Dict<int64_t, string>::iterator, bool> result = dict.insert(3, "new_value");
  EXPECT_FALSE(result.second);
  EXPECT_EQ(3, result.first->key());
  EXPECT_EQ("old_value", result.first->value());
}

TEST(DictTest, whenInsertingExistingKey_thenDoesNotModifyDict) {
  Dict<int64_t, string> dict;
  dict.insert(3, "old_value");
  dict.insert(3, "new_value");
  EXPECT_EQ(1, dict.size());
  EXPECT_EQ(3, dict.begin()->key());
  EXPECT_EQ("old_value", dict.begin()->value());
}

TEST(DictTest, whenInsertOrAssigningNewKey_thenReturnsTrueAndIteratorToNewElement) {
  Dict<int64_t, string> dict;
  std::pair<Dict<int64_t, string>::iterator, bool> result = dict.insert_or_assign(3, "value");
  EXPECT_TRUE(result.second);
  EXPECT_EQ(3, result.first->key());
  EXPECT_EQ("value", result.first->value());
}

TEST(DictTest, whenInsertOrAssigningExistingKey_thenReturnsFalseAndIteratorToChangedElement) {
  Dict<int64_t, string> dict;
  dict.insert(3, "old_value");
  std::pair<Dict<int64_t, string>::iterator, bool> result = dict.insert_or_assign(3, "new_value");
  EXPECT_FALSE(result.second);
  EXPECT_EQ(3, result.first->key());
  EXPECT_EQ("new_value", result.first->value());
}

TEST(DictTest, whenInsertOrAssigningExistingKey_thenDoesModifyDict) {
  Dict<int64_t, string> dict;
  dict.insert(3, "old_value");
  dict.insert_or_assign(3, "new_value");
  EXPECT_EQ(1, dict.size());
  EXPECT_EQ(3, dict.begin()->key());
  EXPECT_EQ("new_value", dict.begin()->value());
}

TEST(DictTest, givenEmptyDict_whenIterating_thenBeginIsEnd) {
  Dict<int64_t, string> dict;
  EXPECT_EQ(dict.begin(), dict.end());
}

TEST(DictTest, givenMutableDict_whenIterating_thenFindsElements) {
  Dict<int64_t, string> dict;
  dict.insert(3, "3");
  dict.insert(5, "5");
  bool found_first = false;
  bool found_second = false;
  for (Dict<int64_t, string>::iterator iter = dict.begin(); iter != dict.end(); ++iter) {
    if (iter->key() == 3) {
      EXPECT_EQ("3", iter->value());
      EXPECT_FALSE(found_first);
      found_first = true;
    } else if (iter->key() == 5) {
      EXPECT_EQ("5", iter->value());
      EXPECT_FALSE(found_second);
      found_second = true;
    } else {
      ADD_FAILURE();
    }
  }
  EXPECT_TRUE(found_first);
  EXPECT_TRUE(found_second);
}

TEST(DictTest, givenMutableDict_whenIteratingWithForeach_thenFindsElements) {
  Dict<int64_t, string> dict;
  dict.insert(3, "3");
  dict.insert(5, "5");
  bool found_first = false;
  bool found_second = false;
  for (const auto& elem : dict) {
    if (elem.key() == 3) {
      EXPECT_EQ("3", elem.value());
      EXPECT_FALSE(found_first);
      found_first = true;
    } else if (elem.key() == 5) {
      EXPECT_EQ("5", elem.value());
      EXPECT_FALSE(found_second);
      found_second = true;
    } else {
      ADD_FAILURE();
    }
  }
  EXPECT_TRUE(found_first);
  EXPECT_TRUE(found_second);
}

TEST(DictTest, givenConstDict_whenIterating_thenFindsElements) {
  Dict<int64_t, string> dict_;
  dict_.insert(3, "3");
  dict_.insert(5, "5");
  const Dict<int64_t, string>& dict = dict_;
  bool found_first = false;
  bool found_second = false;
  for (Dict<int64_t, string>::iterator iter = dict.begin(); iter != dict.end(); ++iter) {
    if (iter->key() == 3) {
      EXPECT_EQ("3", iter->value());
      EXPECT_FALSE(found_first);
      found_first = true;
    } else if (iter->key() == 5) {
      EXPECT_EQ("5", iter->value());
      EXPECT_FALSE(found_second);
      found_second = true;
    } else {
      ADD_FAILURE();
    }
  }
  EXPECT_TRUE(found_first);
  EXPECT_TRUE(found_second);
}

TEST(DictTest, givenConstDict_whenIteratingWithForeach_thenFindsElements) {
  Dict<int64_t, string> dict_;
  dict_.insert(3, "3");
  dict_.insert(5, "5");
  const Dict<int64_t, string>& dict = dict_;
  bool found_first = false;
  bool found_second = false;
  for (const auto& elem : dict) {
    if (elem.key() == 3) {
      EXPECT_EQ("3", elem.value());
      EXPECT_FALSE(found_first);
      found_first = true;
    } else if (elem.key() == 5) {
      EXPECT_EQ("5", elem.value());
      EXPECT_FALSE(found_second);
      found_second = true;
    } else {
      ADD_FAILURE();
    }
  }
  EXPECT_TRUE(found_first);
  EXPECT_TRUE(found_second);
}

TEST(DictTest, givenIterator_thenCanModifyValue) {
  Dict<int64_t, string> dict;
  dict.insert(3, "old_value");
  dict.begin()->setValue("new_value");
  EXPECT_EQ("new_value", dict.begin()->value());
}

TEST(DictTest, givenOneElementDict_whenErasingByIterator_thenDictIsEmpty) {
  Dict<int64_t, string> dict;
  dict.insert(3, "3");
  dict.erase(dict.begin());
  EXPECT_TRUE(dict.empty());
}

TEST(DictTest, givenOneElementDict_whenErasingByKey_thenReturnsOneAndDictIsEmpty) {
  Dict<int64_t, string> dict;
  dict.insert(3, "3");
  bool result = dict.erase(3);
  EXPECT_EQ(1, result);
  EXPECT_TRUE(dict.empty());
}

TEST(DictTest, givenOneElementDict_whenErasingByNonexistingKey_thenReturnsZeroAndDictIsUnchanged) {
  Dict<int64_t, string> dict;
  dict.insert(3, "3");
  bool result = dict.erase(4);
  EXPECT_EQ(0, result);
  EXPECT_EQ(1, dict.size());
}

TEST(DictTest, whenCallingAtWithExistingKey_thenReturnsCorrectElement) {
  Dict<int64_t, string> dict;
  dict.insert(3, "3");
  dict.insert(4, "4");
  EXPECT_EQ("4", dict.at(4));
}

TEST(DictTest, whenCallingAtWithNonExistingKey_thenReturnsCorrectElement) {
  Dict<int64_t, string> dict;
  dict.insert(3, "3");
  dict.insert(4, "4");
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_THROW(dict.at(5), std::out_of_range);
}

TEST(DictTest, givenMutableDict_whenCallingFindOnExistingKey_thenFindsCorrectElement) {
  Dict<int64_t, string> dict;
  dict.insert(3, "3");
  dict.insert(4, "4");
  Dict<int64_t, string>::iterator found = dict.find(3);
  EXPECT_EQ(3, found->key());
  EXPECT_EQ("3", found->value());
}

TEST(DictTest, givenMutableDict_whenCallingFindOnNonExistingKey_thenReturnsEnd) {
  Dict<int64_t, string> dict;
  dict.insert(3, "3");
  dict.insert(4, "4");
  Dict<int64_t, string>::iterator found = dict.find(5);
  EXPECT_EQ(dict.end(), found);
}

TEST(DictTest, givenConstDict_whenCallingFindOnExistingKey_thenFindsCorrectElement) {
  Dict<int64_t, string> dict_;
  dict_.insert(3, "3");
  dict_.insert(4, "4");
  const Dict<int64_t, string>& dict = dict_;
  Dict<int64_t, string>::iterator found = dict.find(3);
  EXPECT_EQ(3, found->key());
  EXPECT_EQ("3", found->value());
}

TEST(DictTest, givenConstDict_whenCallingFindOnNonExistingKey_thenReturnsEnd) {
  Dict<int64_t, string> dict_;
  dict_.insert(3, "3");
  dict_.insert(4, "4");
  const Dict<int64_t, string>& dict = dict_;
  Dict<int64_t, string>::iterator found = dict.find(5);
  EXPECT_EQ(dict.end(), found);
}

TEST(DictTest, whenCallingContainsWithExistingKey_thenReturnsTrue) {
  Dict<int64_t, string> dict;
  dict.insert(3, "3");
  dict.insert(4, "4");
  EXPECT_TRUE(dict.contains(3));
}

TEST(DictTest, whenCallingContainsWithNonExistingKey_thenReturnsFalse) {
  Dict<int64_t, string> dict;
  dict.insert(3, "3");
  dict.insert(4, "4");
  EXPECT_FALSE(dict.contains(5));
}

TEST(DictTest, whenCallingReserve_thenDoesntCrash) {
  Dict<int64_t, string> dict;
  dict.reserve(100);
}

TEST(DictTest, whenCopyConstructingDict_thenAreEqual) {
  Dict<int64_t, string> dict1;
  dict1.insert(3, "3");
  dict1.insert(4, "4");

  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  Dict<int64_t, string> dict2(dict1);

  EXPECT_EQ(2, dict2.size());
  EXPECT_EQ("3", dict2.at(3));
  EXPECT_EQ("4", dict2.at(4));
}

TEST(DictTest, whenCopyAssigningDict_thenAreEqual) {
  Dict<int64_t, string> dict1;
  dict1.insert(3, "3");
  dict1.insert(4, "4");

  Dict<int64_t, string> dict2;
  dict2 = dict1;

  EXPECT_EQ(2, dict2.size());
  EXPECT_EQ("3", dict2.at(3));
  EXPECT_EQ("4", dict2.at(4));
}

TEST(DictTest, whenCopyingDict_thenAreEqual) {
  Dict<int64_t, string> dict1;
  dict1.insert(3, "3");
  dict1.insert(4, "4");

  Dict<int64_t, string> dict2 = dict1.copy();

  EXPECT_EQ(2, dict2.size());
  EXPECT_EQ("3", dict2.at(3));
  EXPECT_EQ("4", dict2.at(4));
}

TEST(DictTest, whenMoveConstructingDict_thenNewIsCorrect) {
  Dict<int64_t, string> dict1;
  dict1.insert(3, "3");
  dict1.insert(4, "4");

  Dict<int64_t, string> dict2(std::move(dict1));

  EXPECT_EQ(2, dict2.size());
  EXPECT_EQ("3", dict2.at(3));
  EXPECT_EQ("4", dict2.at(4));
}

TEST(DictTest, whenMoveAssigningDict_thenNewIsCorrect) {
  Dict<int64_t, string> dict1;
  dict1.insert(3, "3");
  dict1.insert(4, "4");

  Dict<int64_t, string> dict2;
  dict2 = std::move(dict1);

  EXPECT_EQ(2, dict2.size());
  EXPECT_EQ("3", dict2.at(3));
  EXPECT_EQ("4", dict2.at(4));
}

TEST(DictTest, whenMoveConstructingDict_thenOldIsUnchanged) {
  Dict<int64_t, string> dict1;
  dict1.insert(3, "3");
  dict1.insert(4, "4");

  Dict<int64_t, string> dict2(std::move(dict1));
  EXPECT_EQ(2, dict1.size());
  EXPECT_EQ("3", dict1.at(3));
  EXPECT_EQ("4", dict1.at(4));
}

TEST(DictTest, whenMoveAssigningDict_thenOldIsUnchanged) {
  Dict<int64_t, string> dict1;
  dict1.insert(3, "3");
  dict1.insert(4, "4");

  Dict<int64_t, string> dict2;
  dict2 = std::move(dict1);
  EXPECT_EQ(2, dict1.size());
  EXPECT_EQ("3", dict1.at(3));
  EXPECT_EQ("4", dict1.at(4));
}

TEST(DictTest, givenIterator_whenPostfixIncrementing_thenMovesToNextAndReturnsOldPosition) {
  Dict<int64_t, string> dict;
  dict.insert(3, "3");
  dict.insert(4, "4");

  Dict<int64_t, string>::iterator iter1 = dict.begin();
  Dict<int64_t, string>::iterator iter2 = iter1++;
  EXPECT_NE(dict.begin()->key(), iter1->key());
  EXPECT_EQ(dict.begin()->key(), iter2->key());
}

TEST(DictTest, givenIterator_whenPrefixIncrementing_thenMovesToNextAndReturnsNewPosition) {
  Dict<int64_t, string> dict;
  dict.insert(3, "3");
  dict.insert(4, "4");

  Dict<int64_t, string>::iterator iter1 = dict.begin();
  Dict<int64_t, string>::iterator iter2 = ++iter1;
  EXPECT_NE(dict.begin()->key(), iter1->key());
  EXPECT_NE(dict.begin()->key(), iter2->key());
}

TEST(DictTest, givenEqualIterators_thenAreEqual) {
  Dict<int64_t, string> dict;
  dict.insert(3, "3");
  dict.insert(4, "4");

  Dict<int64_t, string>::iterator iter1 = dict.begin();
  Dict<int64_t, string>::iterator iter2 = dict.begin();
  EXPECT_TRUE(iter1 == iter2);
  EXPECT_FALSE(iter1 != iter2);
}

TEST(DictTest, givenDifferentIterators_thenAreNotEqual) {
  Dict<int64_t, string> dict;
  dict.insert(3, "3");
  dict.insert(4, "4");

  Dict<int64_t, string>::iterator iter1 = dict.begin();
  Dict<int64_t, string>::iterator iter2 = dict.begin();
  iter2++;

  EXPECT_FALSE(iter1 == iter2);
  EXPECT_TRUE(iter1 != iter2);
}

TEST(DictTest, givenIterator_whenDereferencing_thenPointsToCorrectElement) {
  Dict<int64_t, string> dict;
  dict.insert(3, "3");

  Dict<int64_t, string>::iterator iter = dict.begin();
  EXPECT_EQ(3, (*iter).key());
  EXPECT_EQ("3", (*iter).value());
  EXPECT_EQ(3, iter->key());
  EXPECT_EQ("3", iter->value());
}

TEST(DictTest, givenIterator_whenWritingToValue_thenChangesValue) {
  Dict<int64_t, string> dict;
  dict.insert(3, "3");

  Dict<int64_t, string>::iterator iter = dict.begin();

  (*iter).setValue("new_value");
  EXPECT_EQ("new_value", dict.begin()->value());

  iter->setValue("new_value_2");
  EXPECT_EQ("new_value_2", dict.begin()->value());
}

TEST(ListTestIValueBasedList, givenIterator_whenWritingToValueFromIterator_thenChangesValue) {
  Dict<int64_t, string> dict;
  dict.insert(3, "3");
  dict.insert(4, "4");
  dict.insert(5, "5");

  (*dict.find(3)).setValue(dict.find(4)->value());
  EXPECT_EQ("4", dict.find(3)->value());

  dict.find(3)->setValue(dict.find(5)->value());
  EXPECT_EQ("5", dict.find(3)->value());
}

TEST(DictTest, isReferenceType) {
  Dict<int64_t, string> dict1;
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  Dict<int64_t, string> dict2(dict1);
  Dict<int64_t, string> dict3;
  dict3 = dict1;

  dict1.insert(3, "three");
  EXPECT_EQ(1, dict1.size());
  EXPECT_EQ(1, dict2.size());
  EXPECT_EQ(1, dict3.size());
}

TEST(DictTest, copyHasSeparateStorage) {
  Dict<int64_t, string> dict1;
  Dict<int64_t, string> dict2(dict1.copy());
  Dict<int64_t, string> dict3;
  dict3 = dict1.copy();

  dict1.insert(3, "three");
  EXPECT_EQ(1, dict1.size());
  EXPECT_EQ(0, dict2.size());
  EXPECT_EQ(0, dict3.size());
}

TEST(DictTest, dictTensorAsKey) {
  Dict<at::Tensor, string> dict;
  at::Tensor key1 = at::tensor(3);
  at::Tensor key2 = at::tensor(4);
  dict.insert(key1, "three");
  dict.insert(key2, "four");

  EXPECT_EQ(2, dict.size());

  Dict<at::Tensor, string>::iterator found_key1 = dict.find(key1);
  ASSERT_EQUAL(key1, found_key1->key());
  EXPECT_EQ("three", found_key1->value());

  Dict<at::Tensor, string>::iterator found_nokey1 = dict.find(at::tensor(3));
  Dict<at::Tensor, string>::iterator found_nokey2 = dict.find(at::tensor(5));
  EXPECT_EQ(dict.end(), found_nokey1);
  EXPECT_EQ(dict.end(), found_nokey2);
}

TEST(DictTest, dictEquality) {
  Dict<string, int64_t> dict;
  dict.insert("one", 1);
  dict.insert("two", 2);

  Dict<string, int64_t> dictSameValue;
  dictSameValue.insert("one", 1);
  dictSameValue.insert("two", 2);

  Dict<string, int64_t> dictNotEqual;
  dictNotEqual.insert("foo", 1);
  dictNotEqual.insert("bar", 2);

  Dict<string, int64_t> dictRef = dict;

  EXPECT_EQ(dict, dictSameValue);
  EXPECT_NE(dict, dictNotEqual);
  EXPECT_NE(dictSameValue, dictNotEqual);
  EXPECT_FALSE(dict.is(dictSameValue));
  EXPECT_TRUE(dict.is(dictRef));
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Dict.h`
- `ATen/ATen.h`
- `gtest/gtest.h`
- `gmock/gmock.h`
- `string`


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
python aten/src/ATen/test/Dict_test.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/test`):

- [`operators_test.cpp_docs.md`](./operators_test.cpp_docs.md)
- [`xpu_generator_test.cpp_docs.md`](./xpu_generator_test.cpp_docs.md)
- [`native_test.cpp_docs.md`](./native_test.cpp_docs.md)
- [`reportMemoryUsage.h_docs.md`](./reportMemoryUsage.h_docs.md)
- [`tensor_iterator_test.cpp_docs.md`](./tensor_iterator_test.cpp_docs.md)
- [`memory_overlapping_test.cpp_docs.md`](./memory_overlapping_test.cpp_docs.md)
- [`operator_name_test.cpp_docs.md`](./operator_name_test.cpp_docs.md)
- [`cuda_distributions_test.cu_docs.md`](./cuda_distributions_test.cu_docs.md)
- [`type_test.cpp_docs.md`](./type_test.cpp_docs.md)
- [`allocator_clone_test.h_docs.md`](./allocator_clone_test.h_docs.md)


## Cross-References

- **File Documentation**: `Dict_test.cpp_docs.md`
- **Keyword Index**: `Dict_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python docs/aten/src/ATen/test/Dict_test.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/aten/src/ATen/test`):

- [`cuda_dlconvertor_test.cpp_kw.md_docs.md`](./cuda_dlconvertor_test.cpp_kw.md_docs.md)
- [`cuda_atomic_ops_test.cu_kw.md_docs.md`](./cuda_atomic_ops_test.cu_kw.md_docs.md)
- [`ivalue_test.cpp_kw.md_docs.md`](./ivalue_test.cpp_kw.md_docs.md)
- [`mobile_memory_cleanup.cpp_kw.md_docs.md`](./mobile_memory_cleanup.cpp_kw.md_docs.md)
- [`reportMemoryUsage_test.cpp_docs.md_docs.md`](./reportMemoryUsage_test.cpp_docs.md_docs.md)
- [`cpu_rng_test.cpp_kw.md_docs.md`](./cpu_rng_test.cpp_kw.md_docs.md)
- [`lazy_tensor_test.cpp_kw.md_docs.md`](./lazy_tensor_test.cpp_kw.md_docs.md)
- [`cuda_allocator_test.cpp_docs.md_docs.md`](./cuda_allocator_test.cpp_docs.md_docs.md)
- [`MaybeOwned_test.cpp_docs.md_docs.md`](./MaybeOwned_test.cpp_docs.md_docs.md)
- [`dlconvertor_test.cpp_kw.md_docs.md`](./dlconvertor_test.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `Dict_test.cpp_docs.md_docs.md`
- **Keyword Index**: `Dict_test.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
