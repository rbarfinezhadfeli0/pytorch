# Documentation: `test/quantization/jit/test_quantize_jit.py`

## File Metadata

- **Path**: `test/quantization/jit/test_quantize_jit.py`
- **Size**: 143,767 bytes (140.40 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: quantization"]
# ruff: noqa: F841

# torch
import io
import itertools
import unittest

import torch
import torch.jit
import torch.jit.quantized
import torch.nn as nn
import torch.nn.functional as F

# torch.ao.quantization
from torch.ao.quantization import (
    default_dynamic_qconfig,
    default_histogram_observer,
    default_observer,
    default_per_channel_weight_observer,
    default_qconfig,
    default_weight_observer,
    float16_dynamic_qconfig,
    fuse_modules,
    get_default_qconfig,
    per_channel_dynamic_qconfig,
    PlaceholderObserver,
    QConfig,
    quantize,
    quantize_dynamic,
    quantize_dynamic_jit,
    quantize_jit,
)

# torch.ao.quantization.quantize_jit
from torch.ao.quantization.quantize_jit import (
    convert_dynamic_jit,
    convert_jit,
    fuse_conv_bn_jit,
    prepare_dynamic_jit,
    prepare_jit,
    script_qconfig,
)
from torch.jit._recursive import wrap_cpp_module
from torch.testing import FileCheck

# Annotated models
from torch.testing._internal.common_quantization import (
    AnnotatedConvBnModel,
    AnnotatedConvModel,
    AnnotatedConvTransposeModel,
    AnnotatedNestedModel,
    AnnotatedSingleLayerLinearModel,
    AnnotatedSkipQuantModel,
    ConvBnModel,
    ConvModel,
    ConvTransposeModel,
    default_per_channel_qconfig,
    get_script_module,
    NestedModel,
    QuantizationTestCase,
    SingleLayerLinearModel,
    skipIfNoFBGEMM,
    SkipQuantModel,
    test_only_eval_fn,
)

# Testing utils
from torch.testing._internal.common_quantized import (
    override_qengines,
    qengine_is_fbgemm,
    qengine_is_qnnpack,
)
from torch.testing._internal.common_utils import (
    raise_on_run_directly,
    set_default_dtype,
)
from torch.testing._internal.jit_utils import (
    attrs_with_prefix,
    get_forward,
    get_forward_graph,
)


# Standard library


class TestQuantizeJitPasses(QuantizationTestCase):
    """Test graph mode quantization passes used by quantize_jit"""

    def test_skip_dequant_constant_prop(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3).float()

            def forward(self, x):
                return self.conv(x)

        m = torch.jit.script(M())
        observer = default_per_channel_weight_observer.with_args(ch_axis=1)
        qconfig_dict = {"": QConfig(activation=default_observer, weight=observer)}
        m = prepare_jit(m, qconfig_dict)
        data = torch.randn(1, 3, 10, 10, dtype=torch.float)

        m(data)
        m = convert_jit(m, debug=True)

        freezed = torch.jit.freeze(m)
        freezed(data)

        # After freezing, weight becomes Constant.
        # We have this pattern in the original graph: Constant f32_weight -> quant -> dequant
        # After skipping dequant during Constant Propagation, the resulting graph will be:
        # Constant int8_weight -> dequant
        FileCheck().check_count("aten::quantize_per_tensor", 2, exactly=True).run(
            freezed.graph
        )
        FileCheck().check_count("aten::quantize_per_channel", 0, exactly=True).run(
            freezed.graph
        )
        FileCheck().check_count("aten::dequantize", 3, exactly=True).run(freezed.graph)
        FileCheck().check("aten::quantize_per_tensor").check_next(
            "aten::dequantize"
        ).check_not("aten::quantize_per_channel").check("aten::dequantize").check_next(
            "aten::conv2d"
        ).check_next("aten::quantize_per_tensor").check_next("aten::dequantize").run(
            freezed.graph
        )

    def test_foldbn_trivial(self):
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}
        conv_module = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        # Test trivial case
        class TestModule(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.conv = conv_module[dim](1, 20, 5, 1)
                self.bn = bn_module[dim](num_features=20)
                self.bn.eps = 0.0023

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        options = itertools.product([True, False], [2, 3])
        data = {2: torch.rand(1, 1, 6, 6), 3: torch.rand(1, 1, 6, 6, 6)}
        # Check that the transformation doesn't change numerics
        for tracing, dim in options:
            eager = TestModule(dim).eval()
            x = data[dim]
            scripted_or_traced = get_script_module(eager, tracing, x).eval()
            # Check that in the original script module's forward we have two
            # CallMethod nodes. One of them should be for conv.forward and the other
            # for bn.forward.
            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', 2, exactly=True
            ).run(str(get_forward(scripted_or_traced._c).graph))

            # Run FoldConvBatchnorm pass.
            scripted_or_traced = fuse_conv_bn_jit(scripted_or_traced)

            # Check that after the pass one of the CallMethods is gone (supposedly,
            # the bn.forward).
            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', 1, exactly=True
            ).run(str(get_forward_graph(scripted_or_traced._c)))

            # Check that the transformation doesn't change numerics
            self.assertEqual(eager(x), scripted_or_traced(x))

    def test_foldbn_trivial_nobias(self):
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}
        conv_module = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        # Test trivial case
        class TestModule(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.conv = conv_module[dim](1, 20, 5, 1, bias=False)
                self.bn = bn_module[dim](num_features=20)
                # to make sure new bias is not zero
                self.bn.eps = 0.0027
                self.bn.bias = torch.nn.Parameter(torch.rand([20]))

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        options = itertools.product([True, False], [2, 3])
        data = {2: torch.rand(1, 1, 6, 6), 3: torch.rand(1, 1, 6, 6, 6)}
        for tracing, dim in options:
            eager = TestModule(dim).eval()
            x = data[dim]
            scripted_or_traced = get_script_module(eager, tracing, x).eval()
            # Check that in the original script module's forward we have two
            # CallMethod nodes. One of them should be for conv.forward and the other
            # for bn.forward.
            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', 2, exactly=True
            ).run(str(get_forward_graph(scripted_or_traced._c)))

            # Run FoldConvBatchnorm pass.
            scripted_or_traced = fuse_conv_bn_jit(scripted_or_traced)

            # Check that after the pass one of the CallMethods is gone (supposedly,
            # the bn.forward).
            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', 1, exactly=True
            ).run(str(get_forward_graph(scripted_or_traced._c)))

            # Check that the transformation doesn't change numerics
            self.assertEqual(eager(x), scripted_or_traced(x))

    def test_foldbn_in_submodule(self):
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}
        conv_module = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        # Test that we find Conv-BN patterns in submodules
        class SubModule(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.conv = conv_module[dim](1, 20, 5, 1)
                self.bn = bn_module[dim](num_features=20)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return x

        class TestModule(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.sub = SubModule(dim)

            def forward(self, x):
                x = self.sub(x)
                return x

        options = itertools.product([True, False], [2, 3])
        data = {2: torch.rand(1, 1, 10, 10), 3: torch.rand(1, 1, 10, 10, 10)}
        for tracing, dim in options:
            eager = TestModule(dim).eval()
            x = data[dim]
            scripted_or_traced = get_script_module(eager, tracing, x).eval()
            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', 2, exactly=True
            ).run(str(get_forward_graph(scripted_or_traced.sub._c)))

            scripted_or_traced = fuse_conv_bn_jit(scripted_or_traced)

            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', 1, exactly=True
            ).run(str(get_forward_graph(scripted_or_traced.sub._c)))

            self.assertEqual(eager(x), scripted_or_traced(x))

    def test_foldbn_shared_classtype(self):
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}
        conv_module = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        class TestModule(torch.nn.Module):
            def __init__(self, dim, bias=False):
                super().__init__()
                self.conv1 = conv_module[dim](5, 5, 3, bias=bias)
                self.bn1 = bn_module[dim](num_features=5)
                self.bn1.running_mean.fill_(-0.2)
                self.bn1.bias = torch.nn.Parameter(torch.rand([5]))
                # to make sure new bias is not zero
                self.bn1.eps = 0.0023
                self.conv2 = conv_module[dim](5, 5, 3, bias=bias)
                self.bn2 = bn_module[dim](num_features=5)
                self.bn2.eps = 0.0029
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)
                return x

        options = itertools.product([True, False], [2, 2], [True, False])
        data = {2: torch.rand(1, 5, 6, 6), 3: torch.rand(1, 5, 6, 6, 6)}
        for tracing, dim, bias in options:
            eager = TestModule(dim, bias).eval()
            x = data[dim]
            scripted_or_traced = get_script_module(eager, tracing, x)
            folded = fuse_conv_bn_jit(scripted_or_traced)
            self.assertEqual(eager(x), scripted_or_traced(x))

    def test_foldbn_no_fusion(self):
        """Test that we don't fuse the cases when module type does not match"""

        class CustomConv(torch.nn.Module):
            def forward(self, x):
                return x

        class CustomBn(torch.nn.Module):
            def forward(self, x):
                return x

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = CustomConv()
                self.bn = CustomBn()

            def forward(self, x):
                return self.bn(self.conv(x))

        m = torch.jit.script(M())
        m = fuse_conv_bn_jit(m)
        FileCheck().check_count("prim::CallMethod", 2, exactly=True).run(m.graph)

    @set_default_dtype(torch.double)
    def test_foldbn_complex_cases(self):
        # This test case attempt to try combinations of conv2d/conv3d with bias/nobias
        # as well as BatchNorm with affine/no-affine along with varying the
        # number of layers.
        # this only works when default dtype is double
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}
        conv_module = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        class SubModule(torch.nn.Module):
            def __init__(self, dim, num_blocks, enable_bias, enable_affine):
                super().__init__()
                layers = []
                for _ in range(num_blocks):
                    layers.append(conv_module[dim](20, 20, 5, 1, bias=enable_bias))
                    bn_obj = bn_module[dim](num_features=20, affine=enable_affine)
                    if enable_affine:
                        bn_obj.weight = torch.nn.Parameter(
                            torch.rand_like(bn_obj.weight)
                        )
                        bn_obj.bias = torch.nn.Parameter(torch.rand_like(bn_obj.bias))
                    bn_obj.running_mean = torch.rand_like(bn_obj.running_mean)
                    bn_obj.running_var = torch.rand_like(bn_obj.running_var)
                    layers.append(bn_obj)
                self.layers = nn.Sequential(*layers)

            def forward(self, x):
                return self.layers(x)

        class TestModule(torch.nn.Module):
            def __init__(self, dim, num_blocks, enable_bias, enable_affine):
                super().__init__()
                self.sub = SubModule(dim, num_blocks, enable_bias, enable_affine)

            def forward(self, x):
                x = self.sub(x)
                return x

        options = itertools.product(
            [True, False], [2, 3], [True, False], [True, False], [1, 2]
        )
        data = {2: torch.rand(1, 20, 10, 10), 3: torch.rand(1, 20, 10, 10, 10)}
        for tracing, dim, enable_bias, enable_bn_affine, num_layers in options:
            eager = TestModule(dim, num_layers, enable_bias, enable_bn_affine).eval()
            x = data[dim]
            scripted_or_traced = get_script_module(eager, tracing, x).eval()

            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', num_layers * 2, exactly=True
            ).run(str(get_forward_graph(scripted_or_traced.sub.layers._c)))

            scripted_or_traced = fuse_conv_bn_jit(scripted_or_traced)

            FileCheck().check_count(
                'prim::CallMethod[name="forward"]', num_layers, exactly=True
            ).run(str(get_forward_graph(scripted_or_traced.sub.layers._c)))

            self.assertEqual(eager(x), scripted_or_traced(x))

    def test_fuse_linear(self):
        class FunctionalLinear(torch.nn.Module):
            def __init__(self, weight, bias):
                super().__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                res = torch.matmul(x, self.weight.t())
                if self.bias is not None:
                    res.add_(self.bias)
                return res

        x1 = torch.rand(3)
        w1 = torch.rand(5, 3)
        b1 = torch.rand(5)

        x2 = torch.rand(5, 5)
        w2 = torch.rand(5, 5)
        b2 = torch.rand(5)

        x3 = torch.rand(5, 5, 5)
        w3 = torch.rand(5, 5)
        b3 = torch.rand(5)
        for has_bias, (x, weight, b) in itertools.product(
            [True, False], [(x1, w1, b1), (x2, w2, b2), (x3, w3, b3)]
        ):
            bias = b if has_bias else None
            model = torch.jit.trace(FunctionalLinear(weight, bias), [x])
            for node in model.graph.nodes():
                if node.kind() == "aten::matmul":
                    source_range_1 = node.sourceRange()
            torch._C._jit_pass_fuse_linear(model.graph)
            for node in model.graph.nodes():
                if node.kind() == "aten::linear":
                    source_range_2 = node.sourceRange()
            FileCheck().check("aten::linear").run(model.graph)
            check_not = ["aten::matmul", "aten::addmm", "aten::add_", "aten::t("]
            for cn in check_not:
                FileCheck().check_not(cn).run(model.graph)
            # make sure it runs
            self.assertTrue(source_range_1 == source_range_2)
            model(x)

        # check matmuls are not fused
        class Matmul(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x):
                return torch.matmul(x, self.weight)

        x = torch.rand(5, 6, 5)
        w = torch.rand(5, 5, 100)
        model = torch.jit.trace(Matmul(w), [x])
        torch._C._jit_pass_fuse_linear(model.graph)
        # check 3d matmul is not fused
        FileCheck().check("aten::matmul").run(model.graph)
        FileCheck().check_not("aten::linear").run(model.graph)
        # make sure it runs
        model(x)

    def test_insert_observers(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)

            def forward(self, x):
                return self.conv(x)

        m = torch.jit.script(M())
        qconfig_dict = {"": default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        # for input and output of conv
        assert len(attrs_with_prefix(m, "_observer_")) == 2
        # for weight
        assert len(attrs_with_prefix(m.conv, "_observer_")) == 1

    def test_insert_observers_interface(self):
        @torch.jit.interface
        class SubInterface(torch.nn.Module):
            def addOne(self, inp) -> torch.Tensor:
                pass

        class Sub(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = torch.nn.Linear(5, 5)

            def addOne(self, inp):
                return self.fc(inp) + 1

            def forward(self, x):
                return self.addOne(x)

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)
                self.sub = Sub()

            def forward(self, x):
                return self.sub(self.conv(x))

        m = torch.jit.script(M())
        qconfig_dict = {"sub.conv": default_qconfig}
        m = prepare_jit(m, qconfig_dict)

    def test_insert_observers_interface_unshare_type(self):
        @torch.jit.interface
        class OperatorIf(nn.Module):
            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                pass

        class Operator(nn.Module):
            def __init__(self, a):
                super().__init__()
                self.a = a

            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                return self.a * (inp + self.a)

        class Inner(nn.Module):
            op: OperatorIf

            def __init__(self, op):
                super().__init__()
                self.op = op

            def forward(self, inp):
                return self.op(inp)

        class Outer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner_a = Inner(Operator(1))
                self.inner_b = Inner(Operator(3.0))

            def forward(self, inp):
                return self.inner_a(inp) + self.inner_b(inp)

        qconfig_dict = {"inner_a": default_qconfig, "inner_b": default_qconfig}

        eager_model = Outer()
        for tracing in [True, False]:
            x = torch.rand(3)
            script_model = get_script_module(eager_model, tracing, x)
            # make sure it runs
            prepare_jit(script_model, qconfig_dict)

    def test_insert_observers_child_qconfig(self):
        class Sub(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                return self.fc(x)

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)
                self.sub = Sub()

            def forward(self, x):
                return self.sub(self.conv(x))

        m = torch.jit.script(M())
        qconfig_dict = {"sub.fc": default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        # input and output of sub
        assert len(attrs_with_prefix(m, "_observer_")) == 2
        # not quantized
        assert len(attrs_with_prefix(m.conv, "_observer_")) == 0
        # no observers since we observe in the outer most call site
        assert len(attrs_with_prefix(m.sub, "_observer_")) == 0
        # weight of linear
        assert len(attrs_with_prefix(m.sub.fc, "_observer_")) == 1

    @unittest.skipUnless(
        "fbgemm" in torch.backends.quantized.supported_engines,
        " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
        " with instruction set support avx2 or newer.",
    )
    def test_insert_observers_skip_values(self):
        class ConvFunctionalReLU(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)

            def forward(self, x):
                return F.relu(self.conv(x))

        class ConvReLUModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(x))

        class AddReLUModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = torch.nn.ReLU()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                out = self.conv(x)
                out += x
                return self.relu(out)

        class AddFunctionalReLU(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                out = self.conv(x)
                out += x
                return F.relu(out)

        def attrs_with_prefix(module, prefix):
            return [x for x, _ in module._modules._c.items() if x.startswith(prefix)]

        qconfig_dict = {"": default_qconfig}
        m = torch.jit.script(ConvFunctionalReLU())
        m = prepare_jit(m, qconfig_dict)
        # observer for weight of conv
        assert len(attrs_with_prefix(m.conv, "_observer_")) == 1
        # observer for input of conv and output of relu
        assert len(attrs_with_prefix(m, "_observer_")) == 2

        m = torch.jit.script(ConvReLUModule())
        m = prepare_jit(m, qconfig_dict)
        # observer for input of conv and output of relu
        assert len(attrs_with_prefix(m, "_observer_")) == 2
        # observer for weight of conv
        assert len(attrs_with_prefix(m.conv, "_observer_")) == 1
        # observer for output of relu
        assert len(attrs_with_prefix(m.relu, "_observer_")) == 0

        m = torch.jit.script(AddReLUModule())
        qconfig_dict = {"": default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        assert len(attrs_with_prefix(m, "_observer")) == 3
        assert len(attrs_with_prefix(m.relu, "_observer")) == 0
        FileCheck().check("aten::add_").check_not(
            'Observer = prim::GetAttr[name="_observer_'
        ).check("ReLU = prim::GetAttr").run(str(get_forward_graph(m._c)))

        m = torch.jit.script(AddFunctionalReLU())
        qconfig_dict = {"": default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        assert len(attrs_with_prefix(m, "_observer")) == 3
        FileCheck().check("aten::add_").check_not(
            'Observer = prim::GetAttr[name="_observer_'
        ).check("CallFunction").check('Observer = prim::GetAttr[name="_observer_').run(
            str(get_forward_graph(m._c))
        )

    def test_insert_observers_weight_dtype(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)

            def forward(self, x):
                return F.relu(self.conv(x))

        m = torch.jit.script(M())
        qconfig_dict = {"": default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        activation_dtypes = {
            obs.getattr("dtype")
            for x, obs in m._modules._c.items()
            if x.startswith("_observer_")
        }
        weight_dtypes = {
            obs.getattr("dtype")
            for x, obs in m.conv._modules._c.items()
            if x.startswith("_observer_")
        }
        assert len(activation_dtypes) == 1, "Expected to have 1 activation dtype"
        assert len(weight_dtypes) == 1, "Expected to have 1 weight dtype"
        assert next(iter(activation_dtypes)) != next(iter(weight_dtypes)), (
            "Expected activation dtype to "
        )
        " be different from wegiht dtype"

    def test_insert_observers_for_reused_weight(self):
        class M(torch.nn.Module):
            def forward(self, x, y, weight):
                x = F.conv2d(x, weight)
                y = F.conv2d(y, weight)
                return x + y

        m = torch.jit.script(M()).eval()
        m = prepare_jit(m, {"": default_qconfig})
        # 3 for x, y, weight, one for output of each F.conv2d and one for output of add
        assert len(attrs_with_prefix(m, "_observer")) == 6

    def test_insert_observers_shared_class_type(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 5, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 5, 3).float()

            def forward(self, x):
                return self.conv2(self.conv1(x))

        m = torch.jit.script(M())
        qconfig_dict = {"": default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        # conv1 and conv2 shares the same type, we need to
        # make sure we didn't quantize the type twice
        conv1_observers = attrs_with_prefix(m.conv1, "_observer_")
        conv2_observers = attrs_with_prefix(m.conv2, "_observer_")
        assert len(conv1_observers) == 1, "Expected to have 1 observer submodules"
        assert len(conv2_observers) == 1, "Expected to have 1 observer submodules"
        assert conv1_observers == conv2_observers, (
            "Expect conv1 and conv2 to have same observers since the class type is shared"
        )

    def test_insert_observers_for_general_ops(self):
        """Make sure we skip observers for ops that doesn't require
        observation, e.g. flatten
        """

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                x = self.conv(x)
                x = torch.flatten(x)
                return x

        m = torch.jit.script(M())
        qconfig_dict = {"": default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        # input and output of conv
        assert len(attrs_with_prefix(m, "_observer_")) == 2
        FileCheck().check('Observer = prim::GetAttr[name="_observer_').check(
            'prim::GetAttr[name="conv"]'
        ).check("prim::CallMethod").check(
            'Observer = prim::GetAttr[name="_observer_'
        ).check("aten::flatten").check_not(
            'Observer = prim::GetAttr[name="_observer_'
        ).run(m.graph)

    # TODO: this is too long, split this to test_insert_observers.py and remove
    # insrt_observers prefix
    def test_insert_observers_propagate_observed(self):
        """Make sure we propagate observed property through general ops"""

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                x = self.conv1(x)
                x = torch.flatten(x)
                # we don't want to insert observer for input of self.conv2
                # because output of self.conv1 is already observed
                x = self.conv2(x)
                return x

        m = torch.jit.script(M())
        qconfig_dict = {"": default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        # input and output of conv
        assert len(attrs_with_prefix(m, "_observer_")) == 3
        FileCheck().check('Observer = prim::GetAttr[name="_observer_').check(
            'prim::GetAttr[name="conv1"]'
        ).check("prim::CallMethod").check(
            'Observer = prim::GetAttr[name="_observer_'
        ).check("aten::flatten").check_not(
            'Observer = prim::GetAttr[name="_observer_'
        ).check('prim::GetAttr[name="conv2"]').check(
            'Observer = prim::GetAttr[name="_observer_'
        ).run(m.graph)

    def test_insert_observers_propagate_observed_in_submodule(self):
        """Make sure we propagate observed property through general ops"""

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()
                self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                x = self.conv1(x)
                x = self.avgpool(x)
                # we don't want to insert observer for input of self.conv2
                # because output of self.conv1 is already observed
                x = self.conv2(x)
                return x

        m = torch.jit.script(M())
        qconfig_dict = {"": default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        # input and output of conv
        assert len(attrs_with_prefix(m, "_observer_")) == 3
        FileCheck().check('Observer = prim::GetAttr[name="_observer_').check(
            'prim::GetAttr[name="conv1"]'
        ).check("prim::CallMethod").check(
            'Observer = prim::GetAttr[name="_observer_'
        ).check("prim::CallMethod").check_not(
            'Observer = prim::GetAttr[name="_observer_'
        ).check('prim::GetAttr[name="conv2"]').check(
            'Observer = prim::GetAttr[name="_observer_'
        ).run(m.graph)

    def test_insert_observers_propagate_observed_for_function(self):
        def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
            batchsize, num_channels, height, width = x.data.size()
            channels_per_group = num_channels // groups
            # reshape
            x = x.view(batchsize, groups, channels_per_group, height, width)
            x = torch.transpose(x, 1, 2).contiguous()
            # flatten
            x = x.view(batchsize, -1, height, width)
            return x

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 1).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 1).float()

            def forward(self, x):
                x = self.conv1(x)
                x = channel_shuffle(x, 1)
                x = self.conv2(x)
                return x

        data = [
            (
                torch.rand((1, 3, 10, 10), dtype=torch.float),
                torch.randint(0, 1, (1,), dtype=torch.long),
            )
            for _ in range(2)
        ]
        m = torch.jit.script(M()).eval()
        m = prepare_jit(m, {"": default_qconfig})
        # we want to test that channel_shuffle is going to pass
        # the observed property from the output of conv1 to input of conv2
        # so that we don't insert observers for input of conv2
        assert (
            len(
                attrs_with_prefix(
                    m,
                    "_observer_",
                )
            )
            == 3
        )

    def test_insert_observers_for_if(self):
        class QuantProp(torch.nn.Module):
            def __init__(self, use_skip):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()
                self.use_skip = use_skip

            def forward(self, x):
                if self.use_skip:
                    x = self.conv(x)
                    return torch.reshape(x, x.shape)
                else:
                    x = self.conv(x)
                    return torch.reshape(x, x.shape)

        class Res(torch.nn.Module):
            def __init__(self, use_skip):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()
                self.use_skip = use_skip

            def forward(self, x):
                if self.use_skip:
                    return self.conv(x)
                else:
                    return self.conv(x)

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.quant_prop = QuantProp(True)
                self.res = Res(False)

            def forward(self, x):
                x = self.quant_prop(x)
                x = self.res(x)
                return x

        data = [torch.rand(1, 3, 10, 10, dtype=torch.float)]
        result = {False: [1, 2, 2], True: [2, 1, 0]}
        for tracing in [True, False]:
            if tracing:
                m = torch.jit.trace(M(), data).eval()
            else:
                m = torch.jit.script(M()).eval()
            m = prepare_jit(m, {"": default_qconfig})
            assert (
                len(
                    attrs_with_prefix(
                        m,
                        "_observer_",
                    )
                )
                == result[tracing][0]
            )
            assert (
                len(
                    attrs_with_prefix(
                        m.quant_prop,
                        "_observer_",
                    )
                )
                == result[tracing][1]
            )
            assert (
                len(
                    attrs_with_prefix(
                        m.res,
                        "_observer_",
                    )
                )
                == result[tracing][2]
            )

    def test_insert_observers_for_nested_if(self):
        class Res(torch.nn.Module):
            def __init__(self, use_skip):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()
                self.cond = use_skip
                self.use_skip = use_skip

            def forward(self, x):
                if self.use_skip:
                    if self.cond:
                        return self.conv(x)
                    else:
                        return self.conv(x)
                else:
                    return self.conv(x)

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.res1 = Res(True)
                self.res2 = Res(False)

            def forward(self, x):
                x = self.res1(x)
                x = self.res2(x)
                return x

        data = torch.rand((1, 3, 10, 10), dtype=torch.float)
        result = {True: 3, False: 1}
        for tracing in [True, False]:
            if tracing:
                m = torch.jit.trace(M(), data).eval()
            else:
                m = torch.jit.script(M()).eval()
            m = prepare_jit(m, {"": default_qconfig})
            assert len(attrs_with_prefix(m, "_observer_")) == result[tracing]

    def test_insert_observers_for_if_consistent_observation(self):
        """check quantization for if works as long as
        output of all branches are quantized/observed consistently
        """

        class M(torch.nn.Module):
            def __init__(self, cond):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()
                self.cond = cond

            def forward(self, x):
                x = self.conv(x)
                # x is already observed
                if self.cond:
                    x = torch.flatten(x)
                return x

        class M2(torch.nn.Module):
            def __init__(self, cond):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()
                self.cond = cond

            def forward(self, x):
                x = self.conv1(x)
                if self.cond:
                    x = self.conv2(x)
                    # x will be observed in the branch
                else:
                    x = torch.flatten(x)
                # since output for both branch are quantized
                # the if node is quantized consistently
                return x

        data = torch.rand((1, 3, 5, 5), dtype=torch.float)
        options = list(itertools.product([True, False], [True, False]))
        for cond, tracing in options:
            if tracing:
                m = torch.jit.trace(M(cond), data)
            else:
                m = torch.jit.script(M(cond))
            m = prepare_jit(m, {"": default_qconfig})
            assert len(attrs_with_prefix(m, "_observer_")) == 2

        for cond, tracing in options:
            if tracing:
                m = torch.jit.trace(M2(cond), data)
            else:
                m = torch.jit.script(M2(cond))
            m = prepare_jit(m, {"": default_qconfig})
            num_observers = 2 if tracing and not cond else 3
            assert len(attrs_with_prefix(m, "_observer_")) == num_observers

    def test_insert_quant_dequant(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3).float()

            def forward(self, x):
                return self.conv(x)

        for is_per_channel in [True, False]:
            m = torch.jit.script(M())
            observer = (
                default_per_channel_weight_observer.with_args(ch_axis=1)
                if is_per_channel
                else default_observer
            )
            qconfig_dict = {"": QConfig(activation=observer, weight=observer)}
            m = prepare_jit(m, qconfig_dict)
            data = torch.randn(1, 3, 10, 10, dtype=torch.float)

            m(data)
            m = convert_jit(m, debug=True)
            assert len(m._modules._c.items()) == 1, (
                "Expected to have single submodule of conv"
            )
            # make sure the quantized model is executable
            m(data)
            quant_func = (
                "aten::quantize_per_channel"
                if is_per_channel
                else "aten::quantize_per_tensor"
            )
            FileCheck().check_count(quant_func, 3, exactly=True).run(m.graph)

    def test_insert_quant_dequant_shared_class_type(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                return self.conv2(self.conv1(x))

        for is_per_channel in [True, False]:
            m = torch.jit.script(M())
            observer = (
                default_per_channel_weight_observer.with_args(ch_axis=1)
                if is_per_channel
                else default_observer
            )
            qconfig = QConfig(activation=observer, weight=observer)
            qconfig_dict = {"": qconfig}
            m = prepare_jit(m, qconfig_dict)
            # observers for input, output and value between conv1/conv2
            assert len(attrs_with_prefix(m, "_observer_")) == 3, (
                "Expected to have 3 obervers"
            )
            # observer for weight
            assert len(attrs_with_prefix(m.conv1, "_observer_")) == 1, (
                "Expected to have 1 obervers"
            )
            # observer for weight
            assert len(attrs_with_prefix(m.conv2, "_observer_")) == 1, (
                "Expected to have 1 obervers"
            )

            data = torch.randn(1, 3, 10, 10, dtype=torch.float)
            m(data)
            m = convert_jit(m, debug=True)
            m(data)
            assert m.conv1._c._type() == m.conv2._c._type()

            # check all observers have been removed
            assert len(attrs_with_prefix(m, "_observer_")) == 0, (
                "Expected to have 0 obervers"
            )
            assert len(attrs_with_prefix(m.conv1, "_observer_")) == 0, (
                "Expected to have 0 obervers"
            )
            assert len(attrs_with_prefix(m.conv2, "_observer_")) == 0, (
                "Expected to have 0 obervers"
            )

            quant_func = (
                "aten::quantize_per_channel"
                if is_per_channel
                else "aten::quantize_per_tensor"
            )
            for module in ["conv1", "conv2"]:
                conv = m._c.getattr(module)
                # quantize weight
                FileCheck().check(quant_func).check_next("aten::dequantize").check(
                    'prim::CallMethod[name="_conv_forward"]'
                ).check("return").run(get_forward_graph(conv))
                # no quantize node in _conv_forward
                FileCheck().check_not(quant_func).check("aten::conv2d").check_not(
                    quant_func
                ).check("return").run(conv._get_method("_conv_forward").graph)

    def test_dedup_module_uses(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.relu(x)
                x -= 0.5
                return self.relu(x)

        data = torch.randn((2, 2))
        m = torch.jit.script(M())
        ref_res = m(data)
        assert (
            len([x for x, _ in m._modules._c.items() if x.startswith("relu")]) == 1
        ), "Expected to have 1 relu modules after dedup module uses"
        torch._C._jit_pass_dedup_module_uses(m._c)
        m = torch.jit._recursive.wrap_cpp_module(m._c)
        res = m(data)
        assert (
            len([x for x, _ in m._modules._c.items() if x.startswith("relu")]) == 2
        ), "Expected to have 2 relu modules after dedup module uses"
        self.assertEqual(res, ref_res)

    def test_replicate_dequantize(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()

            def forward(self, x):
                x = torch.dequantize(x)
                r = self.conv(x)
                r += x
                return r

        x = torch.randn([1, 3, 10, 10], dtype=torch.float)
        x = torch.quantize_per_tensor(x, 0.5, 1, torch.quint8)
        m = torch.jit.script(M())
        ref_res = m(x)
        FileCheck().check_count("aten::dequantize", 1, exactly=True).run(m.graph)
        torch._C._jit_pass_replicate_dequantize(m.graph)
        FileCheck().check_count("aten::dequantize", 2, exactly=True).run(m.graph)
        res = get_forward(m._c)(x)
        self.assertEqual(res, ref_res)

    def test_replicate_dequantize_in_block(self):
        class M(torch.nn.Module):
            def __init__(self, cond):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()

                self.cond = cond

            def forward(self, x):
                x = torch.dequantize(x)
                if self.cond:
                    x = self.conv(x)
                else:
                    x = x + 3
                return x

        x = torch.randn([1, 3, 10, 10], dtype=torch.float)
        x = torch.quantize_per_tensor(x, 0.5, 1, torch.quint8)
        m = torch.jit.script(M(True))
        ref_res = m(x)
        FileCheck().check_count("aten::dequantize", 1, exactly=True).run(m.graph)
        torch._C._jit_pass_replicate_dequantize(m.graph)
        FileCheck().check_count("aten::dequantize", 2, exactly=True).run(m.graph)
        # check dequantize is right before CallMethod of conv
        FileCheck().check("aten::dequantize").check_next("CallMethod").run(m.graph)
        # check dequantize is right before add
        FileCheck().check("aten::dequantize").check("aten::dequantize").check_next(
            "aten::add"
        ).run(m.graph)
        res = get_forward(m._c)(x)
        self.assertEqual(res, ref_res)

    def test_swap_functional_linear(self):
        # TODO: This pass replaces any function called "linear" with "aten::linear"
        # No longer necessary, and also quite surprising
        def linear(input, weight, bias):
            return torch.nn.functional.linear(input, weight, bias)

        class M(torch.nn.Module):
            def forward(self, x, weight, bias):
                x = torch.dequantize(x)
                weight = torch.dequantize(weight)
                x = linear(x, weight, bias)
                x = torch.quantize_per_tensor(
                    x, scale=1.0, zero_point=0, dtype=torch.quint8
                )
                return x

        x = torch.rand((10, 5), dtype=torch.float)
        x = torch.quantize_per_tensor(x, scale=0.5, zero_point=1, dtype=torch.quint8)
        weight = torch.rand((5, 5), dtype=torch.float)
        weight = torch.quantize_per_tensor(
            weight, scale=0.5, zero_point=1, dtype=torch.qint8
        )
        bias = torch.rand((5), dtype=torch.float)
        m = torch.jit.script(M())
        ref_res = m(x, weight, bias)
        FileCheck().check("CallFunction").run(m.graph)
        torch._C._jit_pass_swap_functional_linear(m.graph)
        FileCheck().check("aten::linear").check_not("CallFunction").run(m.graph)
        res = m(x, weight, bias)
        self.assertEqual(res, ref_res)

    def test_replicate_quantize_for_if(self):
        """We want to move quantize nodes for output of prim::If
        inside the prim::If blocks so that we can match quantization
        patterns.
        """

        class Res(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 1).float()
                self.use_skip = True

            def forward(self, x: torch.Tensor, cond: bool) -> torch.Tensor:
                # to avoid being frozen
                self.use_skip = cond
                if self.use_skip:
                    return self.conv(x)
                else:
                    return self.conv2(x)

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.res1 = Res()
                self.res2 = Res()

            def forward(self, x):
                x = self.res1(x, True)
                x = self.res2(x, False)
                return x

        data = [[torch.rand((1, 3, 10, 10), dtype=torch.float)]]
        qconfig_dict = {"": default_qconfig}
        m = torch.jit.script(M()).eval()
        m = quantize_jit(m, qconfig_dict, test_only_eval_fn, [data])
        # make sure patterns in both branches are fused
        FileCheck().check_count("quantized::conv2d(", 4, exactly=True).run(m.graph)

    def test_finalize_for_linear(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                return self.fc(x)

        data = [[torch.rand((1, 5), dtype=torch.float)]]
        qconfig_dict = {"": default_qconfig}
        model = torch.jit.script(M()).eval()
        model = quantize_jit(model, qconfig_dict, test_only_eval_fn, [data])
        # make sure there is only one quantize_per_tensor for input
        # and linear_prepack is folded
        FileCheck().check_count("aten::quantize_per_tensor", 1, exactly=True).check_not(
            "quantized::linear_prepack"
        ).check("quantized::linear").run(model.graph)

    def test_inplace_option(self):
        for tracing in [True, False]:
            model = get_script_module(
                torch.nn.Conv2d(3, 3, 3).float(), tracing, self.img_data_2d[0][0]
            )
            qconfig_dict = {"": default_qconfig}
            quantize_jit(
                model, qconfig_dict, test_only_eval_fn, [self.img_data_2d], inplace=True
            )
            FileCheck().check("quantized::conv2d").run(model.graph)

            FileCheck().check_not("aten::conv2d").run(model.graph)

    def test_finalize_debug(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()
                self.avgpool = torch.nn.AvgPool2d(3)

            def forward(self, x):
                x = self.conv(x)
                x = self.avgpool(x)
                return x

        data = [[torch.rand((1, 3, 10, 10), dtype=torch.float)]]
        qconfig_dict = {"": default_qconfig}
        model = torch.jit.script(M()).eval()
        model = quantize_jit(model, qconfig_dict, test_only_eval_fn, [data], debug=True)
        FileCheck().check_not("quantized::conv2d").check("aten::conv2d").check(
            "aten::avg_pool2d"
        ).check("aten::q_scale").check_next("aten::q_zero_point").check_next(
            "prim::dtype"
        ).check_next("aten::quantize_per_tensor").check("aten::dequantize").run(
            model.graph
        )

    def test_module_list(self):
        class SimpleLinearLayer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                return self.fc(x)

        class ComplexModel(torch.nn.Module):
            def __init__(self) -> None:
             
```



## High-Level Overview


This Python file contains 152 class(es) and 370 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestQuantizeJitPasses`, `M`, `TestModule`, `TestModule`, `SubModule`, `TestModule`, `TestModule`, `CustomConv`, `CustomBn`, `M`, `SubModule`, `TestModule`, `FunctionalLinear`, `Matmul`, `M`, `SubInterface`, `Sub`, `M`, `OperatorIf`, `Operator`

**Functions defined**: `test_skip_dequant_constant_prop`, `__init__`, `forward`, `test_foldbn_trivial`, `__init__`, `forward`, `test_foldbn_trivial_nobias`, `__init__`, `forward`, `test_foldbn_in_submodule`, `__init__`, `forward`, `__init__`, `forward`, `test_foldbn_shared_classtype`, `__init__`, `forward`, `test_foldbn_no_fusion`, `forward`, `forward`

**Key imports**: io, itertools, unittest, torch, torch.jit, torch.jit.quantized, torch.nn as nn, torch.nn.functional as F, wrap_cpp_module, FileCheck


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/quantization/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `io`
- `itertools`
- `unittest`
- `torch`
- `torch.jit`
- `torch.jit.quantized`
- `torch.nn as nn`
- `torch.nn.functional as F`
- `torch.jit._recursive`: wrap_cpp_module
- `torch.testing`: FileCheck
- `warnings`


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

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/quantization/jit/test_quantize_jit.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/quantization/jit`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_ondevice_quantization.py_docs.md`](./test_ondevice_quantization.py_docs.md)
- [`test_fusion_passes.py_docs.md`](./test_fusion_passes.py_docs.md)
- [`test_deprecated_jit_quant.py_docs.md`](./test_deprecated_jit_quant.py_docs.md)


## Cross-References

- **File Documentation**: `test_quantize_jit.py_docs.md`
- **Keyword Index**: `test_quantize_jit.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
