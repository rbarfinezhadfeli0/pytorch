# Documentation: `aten/src/ATen/test/cuda_tensor_interop_test.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/cuda_tensor_interop_test.cpp`
- **Size**: 5,177 bytes (5.06 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/irange.h>
#include <caffe2/core/init.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/context_gpu.h>
#include <caffe2/utils/math.h>

// dumbest possible copies
template<typename T>
T cuda_get(T* addr) {
  T result;
  CUDA_ENFORCE(cudaMemcpy(&result, addr, sizeof(T), cudaMemcpyDefault));
  return result;
}

template<typename T>
void cuda_set(T* addr, T value) {
  CUDA_ENFORCE(cudaMemcpy(addr, &value, sizeof(T), cudaMemcpyDefault));
}

TEST(CUDACaffe2ToPytorch, SimpleLegacy) {
  if (!at::cuda::is_available()) return;
  caffe2::Tensor c2_tensor(caffe2::CUDA);
  c2_tensor.Resize(4, 4);
  auto data = c2_tensor.mutable_data<int64_t>();
  {
    caffe2::CUDAContext context;
    caffe2::math::Set<int64_t>(16, 777, data, &context);
  }
  at::Tensor at_tensor(c2_tensor);
  ASSERT_TRUE(at_tensor.is_cuda());

  auto at_cpu = at_tensor.cpu();
  auto it = at_cpu.data_ptr<int64_t>();
  for (const auto i : c10::irange(16)) {
    ASSERT_EQ(it[i], 777);
  }
}

TEST(CUDACaffe2ToPytorch, Simple) {
  if (!at::cuda::is_available()) return;
  caffe2::Tensor c2_tensor =
      caffe2::empty({4, 4}, at::dtype<int64_t>().device(caffe2::CUDA));
  auto data = c2_tensor.mutable_data<int64_t>();
  {
    caffe2::CUDAContext context;
    caffe2::math::Set<int64_t>(16, 777, data, &context);
  }
  at::Tensor at_tensor(c2_tensor);
  ASSERT_TRUE(at_tensor.is_cuda());

  auto at_cpu = at_tensor.cpu();
  auto it = at_cpu.data_ptr<int64_t>();
  for (const auto i : c10::irange(16)) {
    ASSERT_EQ(it[i], 777);
  }
}

TEST(CUDACaffe2ToPytorch, Op) {
  if (!at::cuda::is_available()) return;
  caffe2::Tensor c2_tensor =
      caffe2::empty({3, 3}, at::dtype<int64_t>().device(caffe2::CUDA));
  auto data = c2_tensor.mutable_data<int64_t>();
  {
    caffe2::CUDAContext context;
    caffe2::math::Set<int64_t>(9, 111, data, &context);
  }
  at::Tensor at_tensor(c2_tensor);
  ASSERT_TRUE(at_tensor.is_cuda());

  ASSERT_EQ(at::sum(at_tensor).item<int64_t>(), 999);
}

TEST(CUDAPytorchToCaffe2, Op) {
  if (!at::cuda::is_available()) return;
  caffe2::Workspace workspace;
  caffe2::NetDef net;

  auto at_tensor_a = at::ones({5, 5}, at::dtype(at::kFloat).device(at::kCUDA));
  auto at_tensor_b = at::ones({5, 5}, at::dtype(at::kFloat).device(at::kCUDA));
  auto at_tensor_c = at::ones({5, 5}, at::dtype(at::kFloat).device(at::kCUDA));

  auto* c2_tensor_a = BlobSetTensor(workspace.CreateBlob("a"), caffe2::Tensor(at_tensor_a));
  auto* c2_tensor_b = BlobSetTensor(workspace.CreateBlob("b"), caffe2::Tensor(at_tensor_b));
  (void)c2_tensor_a;
  (void)c2_tensor_b;

  // Test Alias
  {
    caffe2::Tensor c2_tensor_from_aten(at_tensor_c);
    BlobSetTensor(workspace.CreateBlob("c"), c2_tensor_from_aten.Alias());
  }

  {
    auto op = net.add_op();
    op->set_type("Sum");
    op->add_input("a");
    op->add_input("b");
    op->add_input("c");
    op->add_output("d");
    op->mutable_device_option()->set_device_type(caffe2::PROTO_CUDA);
  }

  workspace.RunNetOnce(net);

  const auto& result = workspace.GetBlob("d")->Get<caffe2::Tensor>();
  ASSERT_EQ(result.GetDeviceType(), caffe2::CUDA);

  auto data = result.data<float>();
  for (const auto i : c10::irange(25)) {
    ASSERT_EQ(cuda_get(data + i), 3.0);
  }
  at::Tensor at_result(result);
  ASSERT_TRUE(at_result.is_cuda());
  ASSERT_EQ(at::sum(at_result).item<float>(), 75);
}

TEST(CUDAPytorchToCaffe2, SharedStorageWrite) {
  if (!at::cuda::is_available()) return;
  auto at_tensor_a = at::ones({5, 5}, at::dtype(at::kFloat).device(at::kCUDA));
  auto at_tensor_b = at_tensor_a.view({25});

  caffe2::Tensor c2_tensor_a(at_tensor_a);
  caffe2::Tensor c2_tensor_b(at_tensor_b);

  // change is visible everywhere
  cuda_set<float>(c2_tensor_a.mutable_data<float>() + 1, 123);
  ASSERT_EQ(cuda_get(c2_tensor_b.mutable_data<float>() + 1), 123);
  ASSERT_EQ(at_tensor_a[0][1].item().to<float>(), 123);
  ASSERT_EQ(at_tensor_b[1].item().to<float>(), 123);
}

TEST(CUDAPytorchToCaffe2, MutualResizes) {
  if (!at::cuda::is_available()) return;
  auto at_tensor = at::ones({5, 5}, at::dtype(at::kFloat).device(at::kCUDA));

  caffe2::Tensor c2_tensor(at_tensor);

  // change is visible
  cuda_set<float>(c2_tensor.mutable_data<float>(), 123);
  ASSERT_EQ(at_tensor[0][0].item().to<float>(), 123);

  // resize PT tensor in smaller direction - storage is preserved
  at_tensor.resize_({4, 4});
  cuda_set<float>(c2_tensor.mutable_data<float>() + 1, 234);
  ASSERT_EQ(at_tensor[0][1].item().to<float>(), 234);

  // resize PT tensor in larger direction - storage is preserved
  at_tensor.resize_({6, 6});
  cuda_set<float>(c2_tensor.mutable_data<float>() + 2, 345);
  ASSERT_EQ(at_tensor[0][2].item().to<float>(), 345);
  ASSERT_EQ(c2_tensor.sizes()[0], 6);
  ASSERT_EQ(c2_tensor.sizes()[1], 6);

  // resize Caffe2 tensor - semantics are to NOT preserve the data, but the
  // TensorImpl is still shared
  c2_tensor.Resize(7, 7);
  cuda_set<float>(c2_tensor.mutable_data<float>() + 3, 456);
  ASSERT_EQ(at_tensor[0][3].item().to<float>(), 456);
  ASSERT_EQ(at_tensor.sizes()[0], 7);
  ASSERT_EQ(at_tensor.sizes()[1], 7);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 11 function(s).

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
- `ATen/cuda/CUDAContext.h`
- `c10/util/irange.h`
- `caffe2/core/init.h`
- `caffe2/core/operator.h`
- `caffe2/core/context_gpu.h`
- `caffe2/utils/math.h`


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
python aten/src/ATen/test/cuda_tensor_interop_test.cpp
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

- **File Documentation**: `cuda_tensor_interop_test.cpp_docs.md`
- **Keyword Index**: `cuda_tensor_interop_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
