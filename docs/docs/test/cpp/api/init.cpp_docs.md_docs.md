# Documentation: `docs/test/cpp/api/init.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/api/init.cpp_docs.md`
- **Size**: 6,548 bytes (6.39 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/cpp/api/init.cpp`

## File Metadata

- **Path**: `test/cpp/api/init.cpp`
- **Size**: 4,269 bytes (4.17 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <torch/torch.h>

#include <test/cpp/api/init_baseline.h>
#include <test/cpp/api/support.h>

#include <functional>
#include <vector>

void check_exact_values(
    const std::vector<torch::Tensor>& parameters,
    const std::vector<std::vector<torch::Tensor>>& expected_parameters) {
  ASSERT_EQ(parameters.size(), expected_parameters.size());

  for (const auto i : c10::irange(parameters.size())) {
    auto layerParameters = parameters[i];
    auto expectedLayerParameters = expected_parameters[i];

    if (static_cast<size_t>(layerParameters.size(0)) !=
        expectedLayerParameters.size()) {
      std::cout << "layer #" << i
                << " layerParameters size: " << layerParameters.size(0)
                << " != "
                << " expectedLayerParameters size: "
                << expectedLayerParameters.size() << std::endl;
      ASSERT_TRUE(false);
    }

    for (const auto p : c10::irange(layerParameters.size(0))) {
      // Always compare using double dtype, regardless of the original dtype of
      // the tensors
      auto tensor = layerParameters[p].to(torch::kFloat64);
      auto expectedTensor = expectedLayerParameters[p].to(torch::kFloat64);

      if (!tensor.allclose(expectedTensor, /*rtol=*/1e-3, /*atol=*/5e-4)) {
        std::cout << "layer " << i << ": " << tensor << " != " << expectedTensor
                  << " (parameter " << p << ")" << std::endl;
        ASSERT_TRUE(false);
      }
    }
  }
}

void check_initializer_against_baseline(
    std::function<void(torch::Tensor)> initializer,
    std::vector<std::vector<torch::Tensor>> expected) {
  torch::manual_seed(0);

  auto layer1 = torch::nn::Linear(7, 15);
  initializer(layer1->weight);
  layer1->to(torch::kFloat64);

  auto layer2 = torch::nn::Linear(15, 15);
  initializer(layer2->weight);
  layer2->to(torch::kFloat64);

  auto layer3 = torch::nn::Linear(15, 2);
  initializer(layer3->weight);
  layer3->to(torch::kFloat64);

  auto parameters = std::vector<torch::Tensor>{
      layer1->weight,
      layer2->weight,
      layer3->weight,
  };

  check_exact_values(parameters, expected);
}

TEST(InitTest, ProducesPyTorchValues_XavierUniform) {
  auto expected = expected_parameters::Xavier_Uniform();
  auto initializer = [](torch::Tensor tensor) {
    torch::nn::init::xavier_uniform_(tensor);
  };
  check_initializer_against_baseline(initializer, expected);
}

TEST(InitTest, ProducesPyTorchValues_XavierNormal) {
  auto expected = expected_parameters::Xavier_Normal();
  auto initializer = [](torch::Tensor tensor) {
    torch::nn::init::xavier_normal_(tensor);
  };
  check_initializer_against_baseline(initializer, expected);
}

TEST(InitTest, ProducesPyTorchValues_KaimingNormal) {
  auto expected = expected_parameters::Kaiming_Normal();
  auto initializer = [](torch::Tensor tensor) {
    torch::nn::init::kaiming_normal_(tensor);
  };
  check_initializer_against_baseline(initializer, expected);
}

TEST(InitTest, ProducesPyTorchValues_KaimingUniform) {
  auto expected = expected_parameters::Kaiming_Uniform();
  auto initializer = [](torch::Tensor tensor) {
    torch::nn::init::kaiming_uniform_(tensor);
  };
  check_initializer_against_baseline(initializer, expected);
}

TEST(InitTest, CanInitializeTensorThatRequiresGrad) {
  auto tensor = torch::empty({3, 4}, torch::requires_grad());
  ASSERT_THROWS_WITH(
      tensor.fill_(1),
      "a leaf Variable that requires grad "
      "is being used in an in-place operation");
  ASSERT_EQ(torch::nn::init::ones_(tensor).sum().item<int32_t>(), 12);
}

TEST(InitTest, CalculateGainWithTanh) {
  double gain = torch::nn::init::calculate_gain(torch::kTanh);
  ASSERT_DOUBLE_EQ(gain, 5.0 / 3.0);
}

TEST(InitTest, CalculateGainWithRelu) {
  double gain = torch::nn::init::calculate_gain(torch::kReLU);
  ASSERT_DOUBLE_EQ(gain, std::sqrt(2.0));
}

TEST(InitTest, CalculateGainWithLeakyRelu) {
  double gain = torch::nn::init::calculate_gain(torch::kLeakyReLU);
  ASSERT_DOUBLE_EQ(gain, std::sqrt(2.0 / (1 + pow(0.01, 2))));
}

TEST(InitTest, CanInitializeCnnWithOrthogonal) {
  torch::nn::Conv2d conv_layer(torch::nn::Conv2dOptions(3, 2, 3).stride(2));
  torch::nn::init::orthogonal_(conv_layer->named_parameters()["weight"]);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 3 function(s).

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
- `c10/util/irange.h`
- `torch/torch.h`
- `test/cpp/api/init_baseline.h`
- `test/cpp/api/support.h`
- `functional`
- `vector`


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
python test/cpp/api/init.cpp
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

- **File Documentation**: `init.cpp_docs.md`
- **Keyword Index**: `init.cpp_kw.md`
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
python docs/test/cpp/api/init.cpp_docs.md
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

- **File Documentation**: `init.cpp_docs.md_docs.md`
- **Keyword Index**: `init.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
