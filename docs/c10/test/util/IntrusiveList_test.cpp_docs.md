# Documentation: `c10/test/util/IntrusiveList_test.cpp`

## File Metadata

- **Path**: `c10/test/util/IntrusiveList_test.cpp`
- **Size**: 2,857 bytes (2.79 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <c10/util/IntrusiveList.h>
#include <c10/util/irange.h>

#include <gtest/gtest.h>

namespace {

class ListItem : public c10::IntrusiveListHook {};

template <typename TItem>
void check_containers_equal(
    c10::IntrusiveList<TItem>& c1,
    std::vector<std::unique_ptr<TItem>>& c2) {
  EXPECT_EQ(c1.size(), c2.size());
  {
    auto it = c1.begin();
    for (const auto i : c10::irange(c1.size())) {
      EXPECT_EQ(&*it, c2[i].get());
      EXPECT_EQ(it, c1.iterator_to(*c2[i]));
      ++it;
    }
  }
  {
    auto it = c1.rbegin();
    for (const auto i : c10::irange(c1.size())) {
      EXPECT_EQ(&*it, c2[c2.size() - 1 - i].get());
      ++it;
    }
  }
};

} // namespace

TEST(IntrusiveList, TestInsert) {
  c10::IntrusiveList<ListItem> l;
  std::vector<std::unique_ptr<ListItem>> v;

  auto size = 50;

  for ([[maybe_unused]] const auto i : c10::irange(size)) {
    v.push_back(std::make_unique<ListItem>());
    l.insert(l.end(), *v.back());
    check_containers_equal(l, v);
  }
}

TEST(IntrusiveList, TestUnlink) {
  c10::IntrusiveList<ListItem> l;
  std::vector<std::unique_ptr<ListItem>> v;

  auto size = 50;

  for ([[maybe_unused]] const auto i : c10::irange(size)) {
    v.push_back(std::make_unique<ListItem>());
    l.insert(l.end(), *v.back());
  }

  for ([[maybe_unused]] const auto i : c10::irange(size)) {
    auto first = l.begin();
    EXPECT_TRUE(first->is_linked());
    first->unlink();
    EXPECT_FALSE(first->is_linked());
    v.erase(v.begin());
    check_containers_equal(l, v);
  }
}

TEST(IntrusiveList, TestMoveElement) {
  c10::IntrusiveList<ListItem> l;
  std::vector<std::unique_ptr<ListItem>> v;

  auto size = 5;

  for ([[maybe_unused]] const auto i : c10::irange(size)) {
    v.push_back(std::make_unique<ListItem>());
    l.insert(l.end(), *v.back());
  }

  // move 3rd element to the end of the list
  {
    auto it = l.iterator_to(*v[2]);
    EXPECT_TRUE(it->is_linked());
    l.iterator_to(*v[2])->unlink();
    EXPECT_FALSE(it->is_linked());
    l.insert(l.end(), *v[2]);
  }
  {
    auto it = v.begin() + 2;
    std::rotate(it, it + 1, v.end());
  }

  check_containers_equal(l, v);
}

TEST(IntrusiveList, TestEmpty) {
  c10::IntrusiveList<ListItem> l;
  ListItem i;

  EXPECT_TRUE(l.empty());
  l.insert(l.end(), i);
  EXPECT_FALSE(l.empty());
  l.begin()->unlink();
  EXPECT_TRUE(l.empty());
}
TEST(IntrusiveList, TestUnlinkUnlinked) {
  EXPECT_ANY_THROW(ListItem().unlink());
}

TEST(IntrusiveList, TestInitializerListCtro) {
  ListItem i, j;
  c10::IntrusiveList<ListItem> l({i, j});

  EXPECT_EQ(l.size(), 2);
  EXPECT_EQ(l.iterator_to(i), l.begin());
  EXPECT_EQ(l.iterator_to(j), ++l.begin());
}

TEST(IntrusiveList, TestNullListIterator) {
  auto null_iter = c10::ListIterator<c10::IntrusiveListHook, ListItem>{nullptr};

  EXPECT_ANY_THROW(--null_iter);
  EXPECT_ANY_THROW(++null_iter);
}

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `TEST`

**Classes/Structs**: `ListItem`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/test/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/IntrusiveList.h`
- `c10/util/irange.h`
- `gtest/gtest.h`


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
python c10/test/util/IntrusiveList_test.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`c10/test/util`):

- [`bfloat16_test.cpp_docs.md`](./bfloat16_test.cpp_docs.md)
- [`complex_test_common.h_docs.md`](./complex_test_common.h_docs.md)
- [`TypeIndex_test.cpp_docs.md`](./TypeIndex_test.cpp_docs.md)
- [`generic_math_test.cpp_docs.md`](./generic_math_test.cpp_docs.md)
- [`Half_test.cpp_docs.md`](./Half_test.cpp_docs.md)
- [`nofatal_test.cpp_docs.md`](./nofatal_test.cpp_docs.md)
- [`small_vector_test.cpp_docs.md`](./small_vector_test.cpp_docs.md)
- [`exception_test.cpp_docs.md`](./exception_test.cpp_docs.md)
- [`string_view_test.cpp_docs.md`](./string_view_test.cpp_docs.md)
- [`Enumerate_test.cpp_docs.md`](./Enumerate_test.cpp_docs.md)


## Cross-References

- **File Documentation**: `IntrusiveList_test.cpp_docs.md`
- **Keyword Index**: `IntrusiveList_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
