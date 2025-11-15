# Documentation: `docs/test/cpp/api/modulelist.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/api/modulelist.cpp_docs.md`
- **Size**: 11,569 bytes (11.30 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/cpp/api/modulelist.cpp`

## File Metadata

- **Path**: `test/cpp/api/modulelist.cpp`
- **Size**: 9,078 bytes (8.87 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <torch/torch.h>

#include <algorithm>
#include <memory>
#include <vector>

#include <test/cpp/api/support.h>

using namespace torch::nn;
using namespace torch::test;

struct ModuleListTest : torch::test::SeedingFixture {};

TEST_F(ModuleListTest, ConstructsFromSharedPointer) {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };
  ModuleList list(
      std::make_shared<M>(1), std::make_shared<M>(2), std::make_shared<M>(3));
  ASSERT_EQ(list->size(), 3);
}

TEST_F(ModuleListTest, ConstructsFromConcreteType) {
  static int copy_count;

  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    M(const M& other) : torch::nn::Module(other) {
      copy_count++;
    }
    int value;
  };

  copy_count = 0;
  ModuleList list(M(1), M(2), M(3));
  ASSERT_EQ(list->size(), 3);
  // NOTE: The current implementation expects each module to be copied exactly
  // once, which happens when the module is passed into `std::make_shared<T>()`.
  // TODO: Find a way to avoid copying, and then delete the copy constructor of
  // `M`.
  ASSERT_EQ(copy_count, 3);
}

TEST_F(ModuleListTest, ConstructsFromModuleHolder) {
  struct MImpl : torch::nn::Module {
    explicit MImpl(int value_) : value(value_) {}
    int value;
  };

  struct M : torch::nn::ModuleHolder<MImpl> {
    using torch::nn::ModuleHolder<MImpl>::ModuleHolder;
    using torch::nn::ModuleHolder<MImpl>::get;
  };

  ModuleList list(M(1), M(2), M(3));
  ASSERT_EQ(list->size(), 3);
}

TEST_F(ModuleListTest, PushBackAddsAnElement) {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };

  ModuleList list;
  ASSERT_EQ(list->size(), 0);
  ASSERT_TRUE(list->is_empty());
  list->push_back(Linear(3, 4));
  ASSERT_EQ(list->size(), 1);
  list->push_back(std::make_shared<M>(1));
  ASSERT_EQ(list->size(), 2);
  list->push_back(M(2));
  ASSERT_EQ(list->size(), 3);
}

TEST_F(ModuleListTest, Insertion) {
  struct MImpl : torch::nn::Module {
    explicit MImpl(int value_) : value(value_) {}
    int value;
  };
  TORCH_MODULE(M);

  ModuleList list;
  list->push_back(MImpl(1));
  ASSERT_EQ(list->size(), 1);
  list->insert(0, std::make_shared<MImpl>(2));
  ASSERT_EQ(list->size(), 2);
  list->insert(1, M(3));
  ASSERT_EQ(list->size(), 3);
  list->insert(3, M(4));
  ASSERT_EQ(list->size(), 4);
  ASSERT_EQ(list->at<MImpl>(0).value, 2);
  ASSERT_EQ(list->at<MImpl>(1).value, 3);
  ASSERT_EQ(list->at<MImpl>(2).value, 1);
  ASSERT_EQ(list->at<MImpl>(3).value, 4);

  std::unordered_map<size_t, size_t> U = {{0, 2}, {1, 3}, {2, 1}, {3, 4}};
  for (const auto& P : list->named_modules("", false))
    ASSERT_EQ(U[std::stoul(P.key())], P.value()->as<M>()->value);
}

TEST_F(ModuleListTest, AccessWithAt) {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };
  std::vector<std::shared_ptr<M>> modules = {
      std::make_shared<M>(1), std::make_shared<M>(2), std::make_shared<M>(3)};

  ModuleList list;
  for (auto& module : modules) {
    list->push_back(module);
  }
  ASSERT_EQ(list->size(), 3);

  // returns the correct module for a given index
  for (const auto i : c10::irange(modules.size())) {
    ASSERT_EQ(&list->at<M>(i), modules[i].get());
  }

  // throws for a bad index
  ASSERT_THROWS_WITH(list->at<M>(modules.size() + 1), "Index out of range");
  ASSERT_THROWS_WITH(
      list->at<M>(modules.size() + 1000000), "Index out of range");
}

TEST_F(ModuleListTest, AccessWithPtr) {
  struct M : torch::nn::Module {
    explicit M(int value_) : value(value_) {}
    int value;
  };
  std::vector<std::shared_ptr<M>> modules = {
      std::make_shared<M>(1), std::make_shared<M>(2), std::make_shared<M>(3)};

  ModuleList list;
  for (auto& module : modules) {
    list->push_back(module);
  }
  ASSERT_EQ(list->size(), 3);

  // returns the correct module for a given index
  for (const auto i : c10::irange(modules.size())) {
    ASSERT_EQ(list->ptr(i).get(), modules[i].get());
    ASSERT_EQ(list[i].get(), modules[i].get());
    ASSERT_EQ(list->ptr<M>(i).get(), modules[i].get());
  }

  // throws for a bad index
  ASSERT_THROWS_WITH(list->ptr(modules.size() + 1), "Index out of range");
  ASSERT_THROWS_WITH(list->ptr(modules.size() + 1000000), "Index out of range");
}

TEST_F(ModuleListTest, SanityCheckForHoldingStandardModules) {
  ModuleList list(
      Linear(10, 3),
      Conv2d(1, 2, 3),
      Dropout(0.5),
      BatchNorm2d(5),
      Embedding(4, 10),
      LSTM(4, 5));
}

TEST_F(ModuleListTest, ExtendPushesModulesFromOtherModuleList) {
  struct A : torch::nn::Module {};
  struct B : torch::nn::Module {};
  struct C : torch::nn::Module {};
  struct D : torch::nn::Module {};
  ModuleList a(A{}, B{});
  ModuleList b(C{}, D{});
  a->extend(*b);

  ASSERT_EQ(a->size(), 4);
  ASSERT_TRUE(a[0]->as<A>());
  ASSERT_TRUE(a[1]->as<B>());
  ASSERT_TRUE(a[2]->as<C>());
  ASSERT_TRUE(a[3]->as<D>());

  ASSERT_EQ(b->size(), 2);
  ASSERT_TRUE(b[0]->as<C>());
  ASSERT_TRUE(b[1]->as<D>());

  std::vector<std::shared_ptr<A>> c = {
      std::make_shared<A>(), std::make_shared<A>()};
  b->extend(c);

  ASSERT_EQ(b->size(), 4);
  ASSERT_TRUE(b[0]->as<C>());
  ASSERT_TRUE(b[1]->as<D>());
  ASSERT_TRUE(b[2]->as<A>());
  ASSERT_TRUE(b[3]->as<A>());
}

TEST_F(ModuleListTest, HasReferenceSemantics) {
  ModuleList first(Linear(2, 3), Linear(4, 4), Linear(4, 5));
  ModuleList second(first);

  ASSERT_EQ(first.get(), second.get());
  ASSERT_EQ(first->size(), second->size());
  ASSERT_TRUE(std::equal(
      first->begin(),
      first->end(),
      second->begin(),
      [](const std::shared_ptr<Module>& first,
         const std::shared_ptr<Module>& second) {
        return first.get() == second.get();
      }));
}

TEST_F(ModuleListTest, IsCloneable) {
  ModuleList list(Linear(3, 4), Functional(torch::relu), BatchNorm1d(3));
  ModuleList clone = std::dynamic_pointer_cast<ModuleListImpl>(list->clone());
  ASSERT_EQ(list->size(), clone->size());

  for (size_t i = 0; i < list->size(); ++i) {
    // The modules should be the same kind (type).
    ASSERT_EQ(list[i]->name(), clone[i]->name());
    // But not pointer-equal (distinct objects).
    ASSERT_NE(list[i], clone[i]);
  }

  // Verify that the clone is deep, i.e. parameters of modules are cloned too.

  torch::NoGradGuard no_grad;

  auto params1 = list->named_parameters();
  auto params2 = clone->named_parameters();
  ASSERT_EQ(params1.size(), params2.size());
  for (auto& param : params1) {
    ASSERT_FALSE(pointer_equal(param.value(), params2[param.key()]));
    ASSERT_EQ(param->device(), params2[param.key()].device());
    ASSERT_TRUE(param->allclose(params2[param.key()]));
    param->add_(2);
  }
  for (auto& param : params1) {
    ASSERT_FALSE(param->allclose(params2[param.key()]));
  }
}

TEST_F(ModuleListTest, RegistersElementsAsSubmodules) {
  ModuleList list(Linear(10, 3), Conv2d(1, 2, 3), Dropout2d(0.5));

  auto modules = list->children();
  ASSERT_TRUE(modules[0]->as<Linear>());
  ASSERT_TRUE(modules[1]->as<Conv2d>());
  ASSERT_TRUE(modules[2]->as<Dropout2d>());
}

TEST_F(ModuleListTest, NestingIsPossible) {
  ModuleList list(
      (ModuleList(Dropout(), Dropout())),
      (ModuleList(Dropout(), Dropout()), Dropout()));
}

TEST_F(ModuleListTest, CloneToDevice_CUDA) {
  ModuleList list(Linear(3, 4), Functional(torch::relu), BatchNorm1d(3));
  torch::Device device(torch::kCUDA, 0);
  ModuleList clone =
      std::dynamic_pointer_cast<ModuleListImpl>(list->clone(device));
  for (const auto& p : clone->parameters()) {
    ASSERT_EQ(p.device(), device);
  }
  for (const auto& b : clone->buffers()) {
    ASSERT_EQ(b.device(), device);
  }
}

TEST_F(ModuleListTest, PrettyPrintModuleList) {
  ModuleList list(
      Linear(10, 3),
      Conv2d(1, 2, 3),
      Dropout(0.5),
      BatchNorm2d(5),
      Embedding(4, 10),
      LSTM(4, 5));
  ASSERT_EQ(
      c10::str(list),
      "torch::nn::ModuleList(\n"
      "  (0): torch::nn::Linear(in_features=10, out_features=3, bias=true)\n"
      "  (1): torch::nn::Conv2d(1, 2, kernel_size=[3, 3], stride=[1, 1])\n"
      "  (2): torch::nn::Dropout(p=0.5, inplace=false)\n"
      "  (3): torch::nn::BatchNorm2d(5, eps=1e-05, momentum=0.1, affine=true, track_running_stats=true)\n"
      "  (4): torch::nn::Embedding(num_embeddings=4, embedding_dim=10)\n"
      "  (5): torch::nn::LSTM(input_size=4, hidden_size=5, num_layers=1, bias=true, batch_first=false, dropout=0, bidirectional=false)\n"
      ")");
}

TEST_F(ModuleListTest, RangeBasedForLoop) {
  torch::nn::ModuleList mlist(
      torch::nn::Linear(3, 4),
      torch::nn::BatchNorm1d(4),
      torch::nn::Dropout(0.5));

  std::stringstream buffer;
  for (const auto& module : *mlist) {
    module->pretty_print(buffer);
  }
}

TEST_F(ModuleListTest, InvalidAt) {
  torch::nn::ModuleList m(torch::nn::Linear(1, 2));
  ASSERT_THROWS_WITH(
      m->at<torch::nn::Dropout2dImpl>(0), "Unable to cast module");
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 28 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `ModuleListTest`, `M`, `M`, `MImpl`, `M`, `M`, `MImpl`, `M`, `M`, `A`, `B`, `C`, `D`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/api`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `c10/util/irange.h`
- `torch/torch.h`
- `algorithm`
- `memory`
- `vector`
- `test/cpp/api/support.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp/api/modulelist.cpp
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

- **File Documentation**: `modulelist.cpp_docs.md`
- **Keyword Index**: `modulelist.cpp_kw.md`
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

- This file appears to involve **GPU/parallel computing** capabilities.
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
python docs/test/cpp/api/modulelist.cpp_docs.md
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

- **File Documentation**: `modulelist.cpp_docs.md_docs.md`
- **Keyword Index**: `modulelist.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
