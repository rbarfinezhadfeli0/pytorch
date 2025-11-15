# Documentation: `docs/aten/src/ATen/test/NamedTensor_test.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/test/NamedTensor_test.cpp_docs.md`
- **Size**: 10,254 bytes (10.01 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/test/NamedTensor_test.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/NamedTensor_test.cpp`
- **Size**: 7,646 bytes (7.47 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorNames.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

using at::Dimname;
using at::DimnameList;
using at::Symbol;
using at::namedinference::TensorName;
using at::namedinference::TensorNames;

static Dimname dimnameFromString(const std::string& str) {
  return Dimname::fromSymbol(Symbol::dimname(str));
}

TEST(NamedTensorTest, isNamed) {
  auto tensor = at::zeros({3, 2, 5, 7});
  ASSERT_FALSE(tensor.has_names());

  tensor = at::zeros({3, 2, 5, 7});
  ASSERT_FALSE(tensor.has_names());

  tensor = at::zeros({3, 2, 5, 7});
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  auto H = dimnameFromString("H");
  auto W = dimnameFromString("W");
  std::vector<Dimname> names = { N, C, H, W };
  at::internal_set_names_inplace(tensor, names);
  ASSERT_TRUE(tensor.has_names());
}

static bool dimnames_equal(at::DimnameList names, at::DimnameList other) {
  if (names.size() != other.size()) {
    return false;
  }
  for (const auto i : c10::irange(names.size())) {
    const auto& name = names[i];
    const auto& other_name = other[i];
    if (name.type() != other_name.type() || name.symbol() != other_name.symbol()) {
      return false;
    }
  }
  return true;
}

TEST(NamedTensorTest, attachMetadata) {
  auto tensor = at::zeros({3, 2, 5, 7});
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  auto H = dimnameFromString("H");
  auto W = dimnameFromString("W");
  std::vector<Dimname> names = { N, C, H, W };

  at::internal_set_names_inplace(tensor, names);

  const auto retrieved_meta = tensor.get_named_tensor_meta();
  ASSERT_TRUE(dimnames_equal(retrieved_meta->names(), names));

  // Test dropping metadata
  tensor.unsafeGetTensorImpl()->set_named_tensor_meta(nullptr);
  ASSERT_FALSE(tensor.has_names());
}

TEST(NamedTensorTest, internalSetNamesInplace) {
  auto tensor = at::zeros({3, 2, 5, 7});
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  auto H = dimnameFromString("H");
  auto W = dimnameFromString("W");
  std::vector<Dimname> names = { N, C, H, W };
  ASSERT_FALSE(tensor.has_names());

  // Set names
  at::internal_set_names_inplace(tensor, names);
  const auto retrieved_names = tensor.opt_names().value();
  ASSERT_TRUE(dimnames_equal(retrieved_names, names));

  // Drop names
  at::internal_set_names_inplace(tensor, std::nullopt);
  ASSERT_TRUE(tensor.get_named_tensor_meta() == nullptr);
  ASSERT_TRUE(tensor.opt_names() == std::nullopt);
}

TEST(NamedTensorTest, empty) {
  auto N = Dimname::fromSymbol(Symbol::dimname("N"));
  auto C = Dimname::fromSymbol(Symbol::dimname("C"));
  auto H = Dimname::fromSymbol(Symbol::dimname("H"));
  auto W = Dimname::fromSymbol(Symbol::dimname("W"));
  std::vector<Dimname> names = { N, C, H, W };

  auto tensor = at::empty({});
  ASSERT_EQ(tensor.opt_names(), std::nullopt);

  tensor = at::empty({1, 2, 3});
  ASSERT_EQ(tensor.opt_names(), std::nullopt);

  tensor = at::empty({1, 2, 3, 4}, names);
  ASSERT_TRUE(dimnames_equal(tensor.opt_names().value(), names));

  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_THROW(at::empty({1, 2, 3}, names), c10::Error);
}

TEST(NamedTensorTest, dimnameToPosition) {
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  auto H = dimnameFromString("H");
  auto W = dimnameFromString("W");
  std::vector<Dimname> names = { N, C, H, W };

  auto tensor = at::empty({1, 1, 1});
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_THROW(dimname_to_position(tensor, N), c10::Error);

  tensor = at::empty({1, 1, 1, 1}, names);
  ASSERT_EQ(dimname_to_position(tensor, H), 2);
}

static std::vector<Dimname> tensornames_unify_from_right(
    DimnameList names,
    DimnameList other_names) {
  auto names_wrapper = at::namedinference::TensorNames(names);
  auto other_wrapper = at::namedinference::TensorNames(other_names);
  return names_wrapper.unifyFromRightInplace(other_wrapper).toDimnameVec();
}

static void check_unify(
    DimnameList names,
    DimnameList other_names,
    DimnameList expected) {
  // Check legacy at::unify_from_right
  const auto result = at::unify_from_right(names, other_names);
  ASSERT_TRUE(dimnames_equal(result, expected));

  // Check with TensorNames::unifyFromRight.
  // In the future we'll merge at::unify_from_right and
  // TensorNames::unifyFromRight, but for now, let's test them both.
  const auto also_result = tensornames_unify_from_right(names, other_names);
  ASSERT_TRUE(dimnames_equal(also_result, expected));
}

static void check_unify_error(DimnameList names, DimnameList other_names) {
  // In the future we'll merge at::unify_from_right and
  // TensorNames::unifyFromRight. For now, test them both.
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_THROW(at::unify_from_right(names, other_names), c10::Error);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_THROW(tensornames_unify_from_right(names, other_names), c10::Error);
}

TEST(NamedTensorTest, unifyFromRight) {
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  auto H = dimnameFromString("H");
  auto W = dimnameFromString("W");
  auto None = dimnameFromString("*");

  std::vector<Dimname> names = { N, C };

  check_unify({ N, C, H, W }, { N, C, H, W }, { N, C, H, W });
  check_unify({ W }, { C, H, W }, { C, H, W });
  check_unify({ None, W }, { C, H, W }, { C, H, W });
  check_unify({ None, None, H, None }, { C, None, W }, { None, C, H, W });

  check_unify_error({ W, H }, { W, C });
  check_unify_error({ W, H }, { C, H });
  check_unify_error({ None, H }, { H, None });
  check_unify_error({ H, None, C }, { H });
}

TEST(NamedTensorTest, alias) {
  // tensor.alias is not exposed in Python so we test its name propagation here
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  std::vector<Dimname> names = { N, C };

  auto tensor = at::empty({2, 3}, std::vector<Dimname>{ N, C });
  auto aliased = tensor.alias();
  ASSERT_TRUE(dimnames_equal(tensor.opt_names().value(), aliased.opt_names().value()));
}

TEST(NamedTensorTest, NoNamesGuard) {
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  std::vector<Dimname> names = { N, C };

  auto tensor = at::empty({2, 3}, names);
  ASSERT_TRUE(at::NamesMode::is_enabled());
  {
    at::NoNamesGuard guard;
    ASSERT_FALSE(at::NamesMode::is_enabled());
    ASSERT_FALSE(tensor.opt_names());
    ASSERT_FALSE(at::impl::get_opt_names(tensor.unsafeGetTensorImpl()));
  }
  ASSERT_TRUE(at::NamesMode::is_enabled());
}

static std::vector<Dimname> nchw() {
  auto N = dimnameFromString("N");
  auto C = dimnameFromString("C");
  auto H = dimnameFromString("H");
  auto W = dimnameFromString("W");
  return { N, C, H, W };
}

TEST(NamedTensorTest, TensorNamePrint) {
  auto names = nchw();
  {
    auto N = TensorName(names, 0);
    ASSERT_EQ(
        c10::str(N),
        "'N' (index 0 of ['N', 'C', 'H', 'W'])");
  }
  {
    auto H = TensorName(names, 2);
    ASSERT_EQ(
        c10::str(H),
        "'H' (index 2 of ['N', 'C', 'H', 'W'])");
  }
}

TEST(NamedTensorTest, TensorNamesCheckUnique) {
  auto names = nchw();
  {
    // smoke test to check that this doesn't throw
    TensorNames(names).checkUnique("op_name");
  }
  {
    std::vector<Dimname> nchh = { names[0], names[1], names[2], names[2] };
    auto tensornames = TensorNames(nchh);
    // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
    ASSERT_THROW(tensornames.checkUnique("op_name"), c10::Error);
  }
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

- `gtest/gtest.h`
- `ATen/ATen.h`
- `ATen/NamedTensorUtils.h`
- `ATen/TensorNames.h`
- `c10/util/Exception.h`
- `c10/util/irange.h`


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
python aten/src/ATen/test/NamedTensor_test.cpp
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

- **File Documentation**: `NamedTensor_test.cpp_docs.md`
- **Keyword Index**: `NamedTensor_test.cpp_kw.md`
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
python docs/aten/src/ATen/test/NamedTensor_test.cpp_docs.md
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

- **File Documentation**: `NamedTensor_test.cpp_docs.md_docs.md`
- **Keyword Index**: `NamedTensor_test.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
