# Documentation: `test/jit/test_optimize_for_mobile_preserve_debug_info.py`

## File Metadata

- **Path**: `test/jit/test_optimize_for_mobile_preserve_debug_info.py`
- **Size**: 9,657 bytes (9.43 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: mobile"]

import torch
import torch._C
import torch.nn.functional as F
from torch.testing._internal.common_utils import raise_on_run_directly, skipIfNoXNNPACK
from torch.testing._internal.jit_utils import JitTestCase


class TestOptimizeForMobilePreserveDebugInfo(JitTestCase):
    def check_replacement(
        self,
        model,
        replacements,
        jit_pass,
    ):
        """
        model: Model which optimization is performed on
        replacements: Dict mapping from nodes' kinds in the optimized model
            to the kinds of nodes they replaced in the original model
        jit_pass: Function to perform optimization
        """

        original_kinds = set(replacements.values())
        original_source_ranges = {
            node.kind(): node.sourceRange()
            for node in model.graph.nodes()
            if node.kind() in original_kinds
        }

        jit_pass(model._c)

        for node in model.graph.nodes():
            if node.kind() in replacements:
                self.assertEqual(
                    node.sourceRange(),
                    original_source_ranges[replacements[node.kind()]],
                )

    @skipIfNoXNNPACK
    def test_replace_conv1d_with_conv2d(self):
        class TestConv1d(torch.nn.Module):
            def __init__(self, weight, bias):
                super().__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                return F.conv1d(x, self.weight, self.bias)

        self.check_replacement(
            model=torch.jit.script(
                TestConv1d(
                    weight=torch.rand(3, 3, 3),
                    bias=torch.rand(3),
                ),
            ),
            replacements={
                "prim::ListUnpack": "aten::conv1d",
                "prim::ListConstruct": "aten::conv1d",
                "aten::unsqueeze": "aten::conv1d",
                "aten::conv2d": "aten::conv1d",
                "aten::squeeze": "aten::conv1d",
            },
            jit_pass=torch._C._jit_pass_transform_conv1d_to_conv2d,
        )

    @skipIfNoXNNPACK
    def test_insert_pre_packed_linear_before_inline_and_conv_2d_op(self):
        class TestPrepackedLinearBeforeInlineAndConv2dOp(torch.nn.Module):
            def __init__(
                self,
                linear_weight,
                linear_bias,
                conv2d_weight,
                conv2d_bias,
                conv_transpose2d_weight,
                conv_transpose2d_bias,
            ):
                super(
                    TestPrepackedLinearBeforeInlineAndConv2dOp,
                    self,
                ).__init__()
                self.linear_weight = linear_weight.float()
                self.linear_bias = linear_bias.float()
                self.conv2d_weight = conv2d_weight.float()
                self.conv2d_bias = conv2d_bias.float()
                self.conv_transpose2d_weight = conv_transpose2d_weight.float()
                self.conv_transpose2d_bias = conv_transpose2d_bias.float()

            def forward(self, x):
                linear_res = F.linear(
                    x.float(),
                    self.linear_weight,
                    self.linear_bias,
                )
                conv2d_res = F.conv2d(
                    input=linear_res.unsqueeze(dim=0).float(),
                    weight=self.conv2d_weight,
                    bias=self.conv2d_bias,
                )
                return F.conv_transpose2d(
                    input=conv2d_res,
                    weight=self.conv_transpose2d_weight,
                    bias=self.conv_transpose2d_bias,
                )

        in_channels = 6
        iW = 5
        out_channels = 6
        kH = 2
        kW = 3

        self.check_replacement(
            model=torch.jit.script(
                TestPrepackedLinearBeforeInlineAndConv2dOp(
                    linear_weight=torch.rand(iW, 3),
                    linear_bias=torch.rand(iW),
                    conv2d_weight=torch.rand(out_channels, in_channels, kH, kW),
                    conv2d_bias=torch.rand(out_channels),
                    conv_transpose2d_weight=torch.rand(
                        out_channels,
                        in_channels,
                        kH,
                        kW,
                    ),
                    conv_transpose2d_bias=torch.rand(out_channels),
                ),
            ),
            replacements={
                "prepacked::linear_clamp_prepack": "aten::linear",
                "prepacked::linear_clamp_run": "aten::linear",
                "prepacked::conv2d_clamp_prepack": "aten::conv2d",
                "prepacked::conv2d_clamp_run": "aten::conv2d",
                "prepacked::conv2d_transpose_clamp_prepack": "aten::conv_transpose2d",
                "prepacked::conv2d_transpose_clamp_run": "aten::conv_transpose2d",
            },
            jit_pass=torch._C._jit_pass_insert_prepacked_ops,
        )

    @skipIfNoXNNPACK
    def test_insert_pre_packed_linear_op(self):
        self.check_replacement(
            model=torch.jit.trace(torch.nn.Linear(5, 4), torch.rand(3, 2, 5)),
            replacements={
                "prepacked::linear_clamp_prepack": "aten::linear",
                "prepacked::linear_clamp_run": "aten::linear",
            },
            jit_pass=torch._C._jit_pass_insert_prepacked_ops,
        )

    def run_test_fuse_activation_with_pack_ops_linear_conv2d(
        self,
        linear_activation,
        linear_activation_kind,
        conv2d_activation,
        conv2d_activation_kind,
    ):
        class TestFuseActivationLinearConv2d(torch.nn.Module):
            def __init__(
                self,
                linear_weight,
                linear_bias,
                conv2d_weight,
                conv2d_bias,
            ):
                super().__init__()
                self.linear_weight = linear_weight
                self.linear_bias = linear_bias
                self.conv2d_weight = conv2d_weight
                self.conv2d_bias = conv2d_bias

            def forward(self, x):
                x = F.linear(
                    input=x,
                    weight=self.linear_weight,
                    bias=self.linear_bias,
                )
                x = linear_activation(x)
                x = F.conv2d(
                    input=x.unsqueeze(dim=0),
                    weight=self.conv2d_weight,
                    bias=self.conv2d_bias,
                )
                return conv2d_activation(x)

        linear_in_features = 5
        linear_out_features = 4
        conv2d_in_channels = 3
        conv2d_out_channels = 4
        conv2d_kernel = 2
        x_shape = (3, 2, 5)

        model = torch.jit.trace(
            TestFuseActivationLinearConv2d(
                linear_weight=torch.nn.Parameter(
                    data=torch.rand(
                        linear_out_features,
                        linear_in_features,
                    ),
                    requires_grad=False,
                ),
                linear_bias=torch.nn.Parameter(
                    data=torch.rand(linear_out_features),
                    requires_grad=False,
                ),
                conv2d_weight=torch.rand(
                    conv2d_out_channels,
                    conv2d_in_channels,
                    conv2d_kernel,
                    conv2d_kernel,
                ),
                conv2d_bias=torch.rand(conv2d_out_channels),
            ),
            torch.rand(x_shape),
        )

        torch._C._jit_pass_insert_prepacked_ops(model._c)

        self.check_replacement(
            model=model,
            replacements={
                "prepacked::linear_clamp_prepack": "prepacked::linear_clamp_prepack",
                "prepacked::linear_clamp_run": linear_activation_kind,
                "prepacked::conv2d_clamp_prepack": "prepacked::conv2d_clamp_prepack",
                "prepacked::conv2d_clamp_run": conv2d_activation_kind,
            },
            jit_pass=torch._C._jit_pass_fuse_clamp_w_prepacked_linear_conv,
        )

    @skipIfNoXNNPACK
    def test_fuse_activation_with_pack_ops_linear_conv2d_1(self):
        self.run_test_fuse_activation_with_pack_ops_linear_conv2d(
            linear_activation=F.hardtanh,
            linear_activation_kind="aten::hardtanh",
            conv2d_activation=F.hardtanh_,
            conv2d_activation_kind="aten::hardtanh_",
        )

    @skipIfNoXNNPACK
    def test_fuse_activation_with_pack_ops_linear_conv2d_2(self):
        self.run_test_fuse_activation_with_pack_ops_linear_conv2d(
            linear_activation=F.hardtanh_,
            linear_activation_kind="aten::hardtanh_",
            conv2d_activation=F.hardtanh,
            conv2d_activation_kind="aten::hardtanh",
        )

    @skipIfNoXNNPACK
    def test_fuse_activation_with_pack_ops_linear_conv2d_3(self):
        self.run_test_fuse_activation_with_pack_ops_linear_conv2d(
            linear_activation=F.relu,
            linear_activation_kind="aten::relu",
            conv2d_activation=F.relu_,
            conv2d_activation_kind="aten::relu_",
        )

    @skipIfNoXNNPACK
    def test_fuse_activation_with_pack_ops_linear_conv2d_4(self):
        self.run_test_fuse_activation_with_pack_ops_linear_conv2d(
            linear_activation=F.relu_,
            linear_activation_kind="aten::relu_",
            conv2d_activation=F.relu,
            conv2d_activation_kind="aten::relu",
        )


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview

"""        model: Model which optimization is performed on        replacements: Dict mapping from nodes' kinds in the optimized model            to the kinds of nodes they replaced in the original model        jit_pass: Function to perform optimization

This Python file contains 4 class(es) and 15 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestOptimizeForMobilePreserveDebugInfo`, `TestConv1d`, `TestPrepackedLinearBeforeInlineAndConv2dOp`, `TestFuseActivationLinearConv2d`

**Functions defined**: `check_replacement`, `test_replace_conv1d_with_conv2d`, `__init__`, `forward`, `test_insert_pre_packed_linear_before_inline_and_conv_2d_op`, `__init__`, `forward`, `test_insert_pre_packed_linear_op`, `run_test_fuse_activation_with_pack_ops_linear_conv2d`, `__init__`, `forward`, `test_fuse_activation_with_pack_ops_linear_conv2d_1`, `test_fuse_activation_with_pack_ops_linear_conv2d_2`, `test_fuse_activation_with_pack_ops_linear_conv2d_3`, `test_fuse_activation_with_pack_ops_linear_conv2d_4`

**Key imports**: torch, torch._C, torch.nn.functional as F, raise_on_run_directly, skipIfNoXNNPACK, JitTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._C`
- `torch.nn.functional as F`
- `torch.testing._internal.common_utils`: raise_on_run_directly, skipIfNoXNNPACK
- `torch.testing._internal.jit_utils`: JitTestCase


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

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/jit/test_optimize_for_mobile_preserve_debug_info.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/jit`):

- [`test_dataclasses.py_docs.md`](./test_dataclasses.py_docs.md)
- [`test_recursive_script.py_docs.md`](./test_recursive_script.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_python_builtins.py_docs.md`](./test_python_builtins.py_docs.md)
- [`test_functional_blocks.py_docs.md`](./test_functional_blocks.py_docs.md)
- [`test_hooks_modules.py_docs.md`](./test_hooks_modules.py_docs.md)
- [`mydecorator.py_docs.md`](./mydecorator.py_docs.md)
- [`test_union.py_docs.md`](./test_union.py_docs.md)
- [`test_python_bindings.py_docs.md`](./test_python_bindings.py_docs.md)
- [`test_parametrization.py_docs.md`](./test_parametrization.py_docs.md)


## Cross-References

- **File Documentation**: `test_optimize_for_mobile_preserve_debug_info.py_docs.md`
- **Keyword Index**: `test_optimize_for_mobile_preserve_debug_info.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
