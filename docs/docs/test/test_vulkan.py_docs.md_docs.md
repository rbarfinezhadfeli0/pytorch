# Documentation: `docs/test/test_vulkan.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_vulkan.py_docs.md`
- **Size**: 9,854 bytes (9.62 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_vulkan.py`

## File Metadata

- **Path**: `test/test_vulkan.py`
- **Size**: 6,829 bytes (6.67 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: mobile"]

import unittest
import torch
from torch.nn import functional as F

from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing import FileCheck
import io

@unittest.skipUnless(torch.is_vulkan_available(),
                     "Vulkan backend must be available for these tests.")
class TestVulkanRewritePass(TestCase):
    @staticmethod
    def validate_transformed_module(
            # To please flake
            self,
            pattern_count_map,
            data_shape,
            prepack_removal=False,
            fuse_clamping_ops=False):
        module_instance = self
        scripted_model = torch.jit.script(module_instance)
        scripted_model.eval()
        input_data = torch.normal(1, 20, size=data_shape)
        scripted_model(input_data)
        torch._C._jit_pass_vulkan_insert_prepacked_ops(scripted_model._c)
        if fuse_clamping_ops or prepack_removal:
            scripted_model._c = torch._C._freeze_module(scripted_model._c)
        if fuse_clamping_ops:
            torch._C._jit_pass_vulkan_fuse_clamp_w_prepacked_conv(scripted_model._c)
        if prepack_removal:
            torch._C._jit_pass_vulkan_fold_prepacking_ops(scripted_model._c)

        buffer = io.BytesIO()
        torch.jit.save(scripted_model, buffer)
        buffer.seek(0)
        deserialized_scripted_model = torch.jit.load(buffer)
        for pattern, v in pattern_count_map.items():
            if (v == 0):
                FileCheck().check(pattern).run(deserialized_scripted_model.graph)
            elif (v == -1):
                FileCheck().check_not(pattern).run(deserialized_scripted_model.graph)
            else:
                FileCheck().check_count(pattern, v, exactly=True).run(deserialized_scripted_model.graph)

    def test_conv(self):
        # Conv params
        batch_size = 2
        input_channels_per_group = 6
        height = 16
        width = 16
        output_channels_per_group = 6
        groups = 4
        kernel_h = kernel_w = 3
        stride_h = stride_w = 1
        pad_h = pad_w = 1
        dilation = 1
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        strides = (stride_h, stride_w)
        paddings = (pad_h, pad_w)
        dilations = (dilation, dilation)
        conv_weight_shape = (output_channels, input_channels_per_group, kernel_h, kernel_w)
        conv_bias_shape = (output_channels)

        class Conv2D(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(conv_weight_shape), requires_grad=False)
                self.bias = torch.nn.Parameter(torch.rand(conv_bias_shape), requires_grad=False)
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                return F.conv2d(x, self.weight, self.bias,
                                self.strides, self.paddings, self.dilations, self.groups)

        data_shape = (batch_size, input_channels, height, width)
        pattern_count_map = {"Tensor = aten::conv2d": -1,
                             "vulkan_prepack::conv2d_clamp_prepack": 1,
                             "vulkan_prepack::conv2d_clamp_run": 1}
        TestVulkanRewritePass.validate_transformed_module(Conv2D(), pattern_count_map, data_shape)

        class Conv2DRelu(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(conv_weight_shape), requires_grad=False)
                self.bias = torch.nn.Parameter(torch.rand(conv_bias_shape), requires_grad=False)
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                o = F.conv2d(x, self.weight, self.bias,
                             self.strides, self.paddings, self.dilations, self.groups)
                o = F.relu(o)
                return o

        data_shape = (batch_size, input_channels, height, width)
        pattern_count_map = {"Tensor = aten::conv2d": -1,
                             "vulkan_prepack::conv2d_clamp_prepack": 1,
                             "vulkan_prepack::conv2d_clamp_run": 1}
        TestVulkanRewritePass.validate_transformed_module(
            Conv2DRelu(), pattern_count_map, data_shape)

        pattern_count_map["aten::relu"] = 1
        pattern_count_map["vulkan_prepack::conv2d_clamp_prepack"] = -1
        TestVulkanRewritePass.validate_transformed_module(
            Conv2DRelu(),
            pattern_count_map,
            data_shape,
            prepack_removal=True)
        pattern_count_map["aten::relu"] = -1
        TestVulkanRewritePass.validate_transformed_module(
            Conv2DRelu(),
            pattern_count_map,
            data_shape,
            prepack_removal=True,
            fuse_clamping_ops=True)


        class Conv2DHardtanh(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(conv_weight_shape), requires_grad=False)
                self.bias = torch.nn.Parameter(torch.rand(conv_bias_shape), requires_grad=False)
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                o = F.conv2d(x, self.weight, self.bias,
                             self.strides, self.paddings, self.dilations, self.groups)
                o = F.hardtanh(o)
                return o

        data_shape = (batch_size, input_channels, height, width)
        pattern_count_map = {"Tensor = aten::conv2d": -1,
                             "vulkan_prepack::conv2d_clamp_prepack": 1,
                             "vulkan_prepack::conv2d_clamp_run": 1}
        TestVulkanRewritePass.validate_transformed_module(Conv2DHardtanh(), pattern_count_map, data_shape)
        pattern_count_map["aten::hardtanh"] = 1
        pattern_count_map["vulkan_prepack::conv2d_clamp_prepack"] = -1
        TestVulkanRewritePass.validate_transformed_module(
            Conv2DHardtanh(),
            pattern_count_map,
            data_shape,
            prepack_removal=True)
        pattern_count_map["aten::hardtanh"] = -1
        TestVulkanRewritePass.validate_transformed_module(
            Conv2DRelu(),
            pattern_count_map,
            data_shape,
            prepack_removal=True,
            fuse_clamping_ops=True)

if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 4 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestVulkanRewritePass`, `Conv2D`, `Conv2DRelu`, `Conv2DHardtanh`

**Functions defined**: `validate_transformed_module`, `test_conv`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`

**Key imports**: unittest, torch, functional as F, TestCase, run_tests, FileCheck, io


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `torch`
- `torch.nn`: functional as F
- `torch.testing._internal.common_utils`: TestCase, run_tests
- `torch.testing`: FileCheck
- `io`


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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/test_vulkan.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_mkldnn.py_docs.md`](./test_mkldnn.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `test_vulkan.py_docs.md`
- **Keyword Index**: `test_vulkan.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test`, which is part of the **testing infrastructure**.



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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_vulkan.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_tensorexpr.py_docs.md_docs.md`](./test_tensorexpr.py_docs.md_docs.md)
- [`pytest_shard_custom.py_docs.md_docs.md`](./pytest_shard_custom.py_docs.md_docs.md)
- [`test_weak.py_kw.md_docs.md`](./test_weak.py_kw.md_docs.md)
- [`test_view_ops.py_kw.md_docs.md`](./test_view_ops.py_kw.md_docs.md)
- [`test_varlen_attention.py_kw.md_docs.md`](./test_varlen_attention.py_kw.md_docs.md)
- [`test_namedtensor.py_docs.md_docs.md`](./test_namedtensor.py_docs.md_docs.md)
- [`test_binary_ufuncs.py_docs.md_docs.md`](./test_binary_ufuncs.py_docs.md_docs.md)
- [`test_ops_gradients.py_kw.md_docs.md`](./test_ops_gradients.py_kw.md_docs.md)
- [`test_torchfuzz_repros.py_docs.md_docs.md`](./test_torchfuzz_repros.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_vulkan.py_docs.md_docs.md`
- **Keyword Index**: `test_vulkan.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
