# Documentation: `docs/test/cpp/api/grad_mode.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/api/grad_mode.cpp_docs.md`
- **Size**: 4,681 bytes (4.57 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/cpp/api/grad_mode.cpp`

## File Metadata

- **Path**: `test/cpp/api/grad_mode.cpp`
- **Size**: 2,432 bytes (2.38 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
#include <gtest/gtest.h>
#include <test/cpp/api/support.h>
#include <torch/script.h>

using namespace torch::autograd;
using namespace torch::test;

TEST(GradModeTest, TestRequiresGradFunctionalOp) {
  torch::AutoGradMode mode(false);
  for (bool requires_grad : {true, false}) {
    torch::Tensor c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);

    torch::Tensor func_out = c * c;
    ASSERT_FALSE(func_out.requires_grad());
    ASSERT_TRUE(func_out.is_leaf());
  }
}

TEST(GradModeTest, TestRequiresGradInplaceOp) {
  torch::AutoGradMode mode(false);
  for (bool requires_grad : {true, false}) {
    torch::Tensor c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);

    c.mul_(2);
    ASSERT_EQ(c.requires_grad(), requires_grad);
  }
}

TEST(GradModeTest, TestRequiresGradViewOp) {
  torch::AutoGradMode mode(false);
  for (bool requires_grad : {true, false}) {
    torch::Tensor c = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);

    torch::Tensor view_out = c.view({2, 3});
    ASSERT_EQ(view_out.requires_grad(), requires_grad);
    ASSERT_TRUE(view_out.is_leaf());
  }
}

TEST(GradModeTest, TestRequiresGradViewOpExiting) {
  for (bool requires_grad : {true, false}) {
    torch::Tensor s = torch::ones({1, 2, 3}).set_requires_grad(requires_grad);
    torch::Tensor a = s.clone();
    torch::Tensor view_out, tmp;

    {
      torch::AutoGradMode mode(false);
      view_out = a.view(
          {2, 3}); // go through kernels: VariableType, ADInplaceOrView, CPU
      assert_tensor_creation_meta(
          view_out, torch::autograd::CreationMeta::NO_GRAD_MODE);
      ASSERT_EQ(view_out.requires_grad(), requires_grad);
      ASSERT_TRUE(view_out.is_leaf());
    }

    tmp = view_out * view_out;
    ASSERT_EQ(tmp.requires_grad(), requires_grad);
    if (requires_grad) {
      tmp.backward(torch::ones_like(tmp));
      // TODO: this behavior is a side effect of issue #11390.
      ASSERT_FALSE(view_out.grad().defined());
    }

    if (requires_grad) {
      ASSERT_THROWS_WITH(
          view_out.mul_(
              2), // go through kernels: VariableType, ADInplaceOrView, CPU
          "A view was created in no_grad mode and is being modified inplace");
    } else {
      view_out.mul_(2);
    }

    tmp = view_out.view({2, 3});
    ASSERT_EQ(tmp.requires_grad(), requires_grad);
    assert_tensor_creation_meta(
        tmp, torch::autograd::CreationMeta::NO_GRAD_MODE);
  }
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/api`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `test/cpp/api/support.h`
- `torch/script.h`


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
python test/cpp/api/grad_mode.cpp
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

- **File Documentation**: `grad_mode.cpp_docs.md`
- **Keyword Index**: `grad_mode.cpp_kw.md`
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
python docs/test/cpp/api/grad_mode.cpp_docs.md
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

- **File Documentation**: `grad_mode.cpp_docs.md_docs.md`
- **Keyword Index**: `grad_mode.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
