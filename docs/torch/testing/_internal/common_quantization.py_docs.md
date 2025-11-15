# Documentation: `torch/testing/_internal/common_quantization.py`

## File Metadata

- **Path**: `torch/testing/_internal/common_quantization.py`
- **Size**: 115,812 bytes (113.10 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# mypy: ignore-errors

r"""Importing this file includes common utility methods and base classes for
checking quantization api and properties of resulting modules.
"""

import torch
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from functorch.experimental import control_flow
from torch.ao.nn.intrinsic import _FusedModule
from torch.ao.quantization import (
    convert,
    default_dynamic_qat_qconfig,
    default_dynamic_qconfig,
    default_dynamic_quant_observer,
    default_embedding_qat_qconfig,
    default_observer,
    default_per_channel_qconfig,
    default_qconfig,
    default_symmetric_qnnpack_qat_qconfig,
    default_weight_observer,
    DeQuantStub,
    float_qparams_weight_only_qconfig,
    get_default_qat_qconfig,
    get_default_qat_qconfig_mapping,
    get_default_qconfig,
    get_default_qconfig_mapping,
    PerChannelMinMaxObserver,
    propagate_qconfig_,
    QConfig,
    QConfigMapping,
    quantize,
    quantize_dynamic_jit,
    quantize_jit,
    QuantStub,
    QuantType,
    QuantWrapper,
)
from torch.ao.quantization.backend_config import get_executorch_backend_config
from torch.ao.quantization.quantization_mappings import (
    get_default_dynamic_quant_module_mappings,
    get_default_qat_module_mappings,
    get_default_qconfig_propagation_list,
)
from torch.ao.quantization.quantize_pt2e import (
    _convert_to_reference_decomposed_fx,
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)

from torch.export import export
from torch.jit.mobile import _load_for_lite_interpreter
from torch.testing._internal.common_quantized import override_quantized_engine
from torch.testing._internal.common_utils import TEST_WITH_ROCM, TestCase

try:
    from torch.ao.ns.fx.ns_types import NSSingleResultValuesType, NSSubgraph

    # graph mode quantization based on fx
    from torch.ao.quantization.quantize_fx import (
        convert_fx,
        convert_to_reference_fx,
        prepare_fx,
        prepare_qat_fx,
    )
    from torch.fx import GraphModule
    from torch.fx.graph import Node

    HAS_FX = True
except ImportError:
    HAS_FX = False

import contextlib
import copy
import functools
import io
import os

import unittest
from typing import Any, Optional, Union
from collections.abc import Callable

import numpy as np
import torch._dynamo as torchdynamo
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
import torch.ao.quantization.quantizer.xpu_inductor_quantizer as xpuiq
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.ao.quantization.quantizer.xpu_inductor_quantizer import XPUInductorQuantizer
from torch.testing import FileCheck


class NodeSpec:
    """Used for checking GraphModule Node"""

    def __init__(self, op, target):
        """
        op: call_function | call_module
        target:
          for call_function, target would be a function
          for call_module, target would be the type of PyTorch module
        """
        self.op = op
        self.target = target

    @classmethod
    def call_function(cls, target):
        return NodeSpec("call_function", target)

    @classmethod
    def call_method(cls, target):
        return NodeSpec("call_method", target)

    @classmethod
    def call_module(cls, target):
        return NodeSpec("call_module", target)

    def __hash__(self):
        return hash((self.op, self.target))

    def __eq__(self, other):
        if not isinstance(other, NodeSpec):
            return NotImplemented

        return self.op == other.op and self.target == other.target

    def __repr__(self):
        return repr(self.op) + " " + repr(self.target)


def get_supported_device_types():
    return (
        ["cpu", "cuda"] if torch.cuda.is_available() and not TEST_WITH_ROCM else ["cpu"]
    )


def test_only_eval_fn(model, calib_data):
    r"""
    Default evaluation function takes a torch.utils.data.Dataset or a list of
    input Tensors and run the model on the dataset
    """
    for inp in calib_data:
        model(*inp)


_default_loss_fn = torch.nn.CrossEntropyLoss()


def test_only_train_fn(model, train_data, loss_fn=_default_loss_fn):
    r"""
    Default train function takes a torch.utils.data.Dataset and train the model
    on the dataset
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss, correct, total = 0, 0, 0
    for _ in range(10):
        model.train()

        for data, target in train_data:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return train_loss, correct, total


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    for cnt, (image, target) in enumerate(data_loader, start=1):
        print(".", end="")
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy(output, target, topk=(1, 5))
        if cnt >= ntrain_batches:
            return
    return


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def ddp_cleanup():
    dist.destroy_process_group()


def run_ddp(rank, world_size, prepared):
    ddp_setup(rank, world_size)
    prepared.cuda()
    prepared = torch.nn.parallel.DistributedDataParallel(prepared, device_ids=[rank])
    prepared.to(rank)
    model_with_ddp = prepared
    optimizer = torch.optim.SGD(model_with_ddp.parameters(), lr=0.0001)
    train_one_epoch(model_with_ddp, criterion, optimizer, dataset, rank, 1)  # noqa: F821
    ddp_cleanup()


def convert_dynamic(module):
    convert(module, get_default_dynamic_quant_module_mappings(), inplace=True)


def prepare_dynamic(model, qconfig_dict=None):
    propagate_qconfig_(model, qconfig_dict)


def _make_conv_test_input(
    batch_size,
    in_channels_per_group,
    input_feature_map_size,
    out_channels_per_group,
    groups,
    kernel_size,
    X_scale,
    X_zero_point,
    W_scale,
    W_zero_point,
    use_bias,
    use_channelwise,
):
    in_channels = in_channels_per_group * groups
    out_channels = out_channels_per_group * groups

    (X_value_min, X_value_max) = (0, 4)
    X_init = torch.randint(
        X_value_min,
        X_value_max,
        (
            batch_size,
            in_channels,
        )
        + input_feature_map_size,
    )
    X = X_scale * (X_init - X_zero_point).float()
    X_q = torch.quantize_per_tensor(
        X, scale=X_scale, zero_point=X_zero_point, dtype=torch.quint8
    )

    W_scale = W_scale * out_channels
    W_zero_point = W_zero_point * out_channels
    # Resize W_scale and W_zero_points arrays equal to out_channels
    W_scale = W_scale[:out_channels]
    W_zero_point = W_zero_point[:out_channels]
    # For testing, we use small values for weights and for activations so that
    # no overflow occurs in vpmaddubsw instruction. If the overflow occurs in
    # qconv implementation and if there is no overflow.
    # In reference we can't exactly match the results with reference.
    # Please see the comment in qconv implementation file
    #   aten/src/ATen/native/quantized/cpu/qconv.cpp for more details.
    (W_value_min, W_value_max) = (-5, 5)
    # The operator expects them in the format
    # (out_channels, in_channels/groups,) + kernel_size
    W_init = torch.randint(
        W_value_min,
        W_value_max,
        (
            out_channels,
            in_channels_per_group,
        )
        + kernel_size,
    )
    b_init = torch.randint(0, 10, (out_channels,))

    if use_channelwise:
        W_shape = (-1, 1) + (1,) * len(kernel_size)
        W_scales_tensor = torch.tensor(W_scale, dtype=torch.float)
        W_zero_points_tensor = torch.tensor(W_zero_point, dtype=torch.float)
        W = (
            W_scales_tensor.reshape(*W_shape)
            * (W_init.float() - W_zero_points_tensor.reshape(*W_shape)).float()
        )
        b = X_scale * W_scales_tensor * b_init.float()
        W_q = torch.quantize_per_channel(
            W,
            W_scales_tensor.double(),
            W_zero_points_tensor.long(),
            0,
            dtype=torch.qint8,
        )
    else:
        W = W_scale[0] * (W_init - W_zero_point[0]).float()
        b = X_scale * W_scale[0] * b_init.float()
        W_q = torch.quantize_per_tensor(
            W, scale=W_scale[0], zero_point=W_zero_point[0], dtype=torch.qint8
        )

    return (X, X_q, W, W_q, b if use_bias else None)


def _make_conv_add_extra_input_tensor(scale, zero_point, sizes):
    (X_value_min, X_value_max) = (0, 4)
    X_init = torch.randint(
        X_value_min,
        X_value_max,
        sizes,  # Infer the size of tensor to do the add
    )
    X = scale * (X_init - zero_point).float()
    X_q = torch.quantize_per_tensor(
        X, scale=scale, zero_point=zero_point, dtype=torch.quint8
    )
    return X, X_q


def skipIfNoFBGEMM(fn):
    reason = "Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs with instruction set support AVX2 or newer."
    if isinstance(fn, type):
        if "fbgemm" not in torch.backends.quantized.supported_engines:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "fbgemm" not in torch.backends.quantized.supported_engines:
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)

    return wrapper


def skipIfNoQNNPACK(fn):
    reason = "Quantized operations require QNNPACK."
    if isinstance(fn, type):
        if "qnnpack" not in torch.backends.quantized.supported_engines:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "qnnpack" not in torch.backends.quantized.supported_engines:
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)

    return wrapper


def withQNNPACKBackend(fn):
    # TODO(future PR): consider combining with skipIfNoQNNPACK,
    # will require testing of existing callsites
    reason = "Quantized operations require QNNPACK."
    if isinstance(fn, type):
        if "qnnpack" not in torch.backends.quantized.supported_engines:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "qnnpack" not in torch.backends.quantized.supported_engines:
            raise unittest.SkipTest(reason)
        with override_quantized_engine("qnnpack"):
            fn(*args, **kwargs)

    return wrapper


def skipIfNoONEDNN(fn):
    reason = "Quantized operations require ONEDNN."
    if isinstance(fn, type):
        if "onednn" not in torch.backends.quantized.supported_engines:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "onednn" not in torch.backends.quantized.supported_engines:
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)

    return wrapper


def skipIfNoONEDNNBF16(fn):
    reason = "Quantized operations require BF16 support."
    if isinstance(fn, type):
        if not torch.ops.mkldnn._is_mkldnn_bf16_supported():
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not torch.ops.mkldnn._is_mkldnn_bf16_supported():
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)

    return wrapper


def skipIfNoX86(fn):
    reason = "Quantized operations require X86."
    if isinstance(fn, type):
        if "x86" not in torch.backends.quantized.supported_engines:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if "x86" not in torch.backends.quantized.supported_engines:
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)

    return wrapper


def skipIfNoDynamoSupport(fn):
    reason = "dynamo doesn't support."
    if isinstance(fn, type):
        if not torchdynamo.is_dynamo_supported():
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not torchdynamo.is_dynamo_supported():
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)

    return wrapper


def skipIfNoInductorSupport(fn):
    reason = "inductor doesn't support."
    if isinstance(fn, type):
        if not torchdynamo.is_inductor_supported():
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not torchdynamo.is_inductor_supported():
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)

    return wrapper


try:
    import torchvision  # noqa: F401

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skip_if_no_torchvision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")


def get_script_module(model, tracing, data):
    return torch.jit.trace(model, data) if tracing else torch.jit.script(model)


def lengths_to_offsets(t, offset_type=np.int64, use_begin_offset=True):
    """
    Convert lengths to offsets for embedding_bag
    """
    tt = np.zeros((t.shape[0] + 1,), dtype=offset_type)
    tt[1:] = t
    tt = torch.from_numpy(np.cumsum(tt, dtype=offset_type))
    if use_begin_offset:
        return tt[:-1]
    return tt[1:]


def _group_quantize_tensor(w, n_bit=4, q_group_size=16):
    assert w.dim() == 2
    w = w.transpose(0, 1).contiguous()
    assert q_group_size > 1
    assert w.shape[-1] % q_group_size == 0

    to_quant = w.reshape(-1, q_group_size)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    min_int = 0
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    assert torch.isnan(scales).sum() == 0

    zeros = min_val + scales * (2 ** (n_bit - 1))
    assert torch.isnan(zeros).sum() == 0

    out = to_quant.sub(min_val).div(scales).round().clamp_(min_int, max_int)
    assert torch.isnan(out).sum() == 0

    out = out.to(dtype=torch.int32).reshape(w.shape)
    if out.device != torch.device("cpu"):
        out = (out[::, ::2] << 4 | out[::, 1::2]).to(torch.uint8)

    # Scales and zeros for the same q-group should be contiguous, so we can
    # load as a 32-bit word
    scales = scales.view(w.shape[0], -1)
    zeros = zeros.view(w.shape[0], -1)
    scales_and_zeros = (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )

    return out, scales_and_zeros


def _group_quantize_tensor_symmetric(w, n_bit=4, groupsize=32):
    # W is of shape [K x N]
    # We transpose W as Quantization is applied on [N x K]
    w = w.transpose(0, 1).contiguous()
    assert w.dim() == 2
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    # Calculate scale and zeros
    to_quant = w.reshape(-1, groupsize)
    max_val = to_quant.abs().amax(dim=1, keepdim=True)
    eps = torch.finfo(max_val.dtype).eps
    max_int = 2 ** (n_bit - 1) - 1  # For 4-bit, this is 7
    scales = max_val.clamp(min=eps) / max_int
    zeros = torch.zeros_like(scales)

    # Quantize the weight
    scales = scales.to(torch.float32).reshape(w.shape[0], -1)
    zeros = zeros.to(torch.float32).reshape(w.shape[0], -1)
    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)
    max_int = 2**n_bit - 1
    w_int8 = to_quant.div(scales).add(8.5).to(torch.int8).clamp(max=max_int)
    # We pack 2 signed int4 values in unsigned uint8 container.
    # This reduces the weight size by half and improves load perf
    out_uint8 = (w_int8[::, 1::2] << 4 | w_int8[::, ::2]).to(torch.uint8)

    scales_and_zeros = scales.squeeze().contiguous()

    return out_uint8, scales_and_zeros


def _dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # source: https://github.com/meta-pytorch/gpt-fast/blob/main/quantize.py
    # default setup for affine quantization of activations
    x_dtype = x.dtype
    x = x.float()
    eps = torch.finfo(torch.float32).eps

    # get min and max
    min_val, max_val = torch.aminmax(x, dim=1)

    # calculate scales and zero_points based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
    device = min_val_neg.device

    # reference: https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scales is the same dtype as the original tensor
    scales = torch.clamp(scales, min=eps).to(x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scales/zp
    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points.unsqueeze(-1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(target_dtype)

    return quant, scales.to(x_dtype), zero_points


# QuantizationTestCase used as a base class for testing quantization on modules
class QuantizationTestCase(TestCase):
    def setUp(self):
        super().setUp()
        self.calib_data = [[torch.rand(2, 5, dtype=torch.float)] for _ in range(2)]
        self.train_data = [
            [
                torch.rand(2, 5, dtype=torch.float),
                torch.randint(0, 1, (2,), dtype=torch.long),
            ]
            for _ in range(2)
        ]
        self.img_data_1d = [[torch.rand(2, 3, 10, dtype=torch.float)] for _ in range(2)]
        self.img_data_2d = [
            [torch.rand(1, 3, 10, 10, dtype=torch.float)] for _ in range(2)
        ]
        self.img_data_3d = [
            [torch.rand(1, 3, 5, 5, 5, dtype=torch.float)] for _ in range(2)
        ]
        self.img_data_1d_train = [
            [
                torch.rand(2, 3, 10, dtype=torch.float),
                torch.randint(0, 1, (1,), dtype=torch.long),
            ]
            for _ in range(2)
        ]
        self.img_data_2d_train = [
            [
                torch.rand(1, 3, 10, 10, dtype=torch.float),
                torch.randint(0, 1, (1,), dtype=torch.long),
            ]
            for _ in range(2)
        ]
        self.img_data_3d_train = [
            [
                torch.rand(1, 3, 5, 5, 5, dtype=torch.float),
                torch.randint(0, 1, (1,), dtype=torch.long),
            ]
            for _ in range(2)
        ]

        self.img_data_dict = {
            1: self.img_data_1d,
            2: self.img_data_2d,
            3: self.img_data_3d,
        }

        # Quant types that produce statically quantized ops
        self.static_quant_types = [QuantType.STATIC, QuantType.QAT]
        # All quant types for (fx based) graph mode quantization
        self.all_quant_types = [QuantType.DYNAMIC, QuantType.STATIC, QuantType.QAT]

    def checkNoPrepModules(self, module):
        r"""Checks the module does not contain child
        modules for quantization preparation, e.g.
        quant, dequant and observer
        """
        self.assertFalse(hasattr(module, "quant"))
        self.assertFalse(hasattr(module, "dequant"))

    def checkNoQconfig(self, module):
        r"""Checks the module does not contain qconfig"""
        self.assertFalse(hasattr(module, "qconfig"))

        for child in module.children():
            self.checkNoQconfig(child)

    def checkHasPrepModules(self, module):
        r"""Checks the module contains child
        modules for quantization preparation, e.g.
        quant, dequant and observer
        """
        self.assertTrue(hasattr(module, "module"))
        self.assertTrue(hasattr(module, "quant"))
        self.assertTrue(hasattr(module, "dequant"))

    def checkObservers(
        self, module, propagate_qconfig_list=None, prepare_custom_config_dict=None
    ):
        r"""Checks the module or module's leaf descendants
        have observers in preparation for quantization
        """
        if propagate_qconfig_list is None:
            propagate_qconfig_list = get_default_qconfig_propagation_list()
        if prepare_custom_config_dict is None:
            prepare_custom_config_dict = {}
        float_to_observed_module_class_mapping = prepare_custom_config_dict.get(
            "float_to_observed_custom_module_class", {}
        )

        # check if a module is a leaf module, ignoring activation_post_process attribute
        def is_leaf_module(module):
            submodule_name_count = 0
            for name, _ in module.named_children():
                if name != "activation_post_process":
                    submodule_name_count += 1
            return submodule_name_count == 0

        if (
            hasattr(module, "qconfig")
            and module.qconfig is not None
            and (
                (
                    is_leaf_module(module)
                    and not isinstance(module, torch.nn.Sequential)
                    and type(module) in propagate_qconfig_list
                )
                or type(module) in float_to_observed_module_class_mapping
            )
            and not isinstance(module, torch.ao.quantization.DeQuantStub)
        ):
            self.assertTrue(
                hasattr(module, "activation_post_process"),
                "module: " + str(type(module)) + " do not have observer",
            )
        # we don't need to check observers for child modules of the
        # qat modules
        if (
            type(module) not in get_default_qat_module_mappings().values()
            and type(module) not in float_to_observed_module_class_mapping.values()
            and not isinstance(module, _FusedModule)
        ):
            for child in module.children():
                if type(child) is nn.Dropout:
                    continue
                self.checkObservers(
                    child, propagate_qconfig_list, prepare_custom_config_dict
                )

    def checkQuantDequant(self, mod):
        r"""Checks that mod has nn.Quantize and
        nn.DeQuantize submodules inserted
        """
        self.assertEqual(type(mod.quant), nnq.Quantize)
        self.assertEqual(type(mod.dequant), nnq.DeQuantize)

    def checkWrappedQuantizedLinear(self, mod):
        r"""Checks that mod has been swapped for an nnq.Linear
        module, the bias is qint32, and that the module
        has Quantize and DeQuantize submodules
        """
        self.assertEqual(type(mod.module), nnq.Linear)
        self.checkQuantDequant(mod)

    def checkQuantizedLinear(self, mod):
        self.assertEqual(type(mod), nnq.Linear)

    def checkDynamicQuantizedLinear(self, mod, dtype):
        r"""Checks that mod has been swapped for an nnqd.Linear
        module, the bias is float.
        """
        self.assertEqual(type(mod), nnqd.Linear)
        self.assertEqual(mod._packed_params.dtype, dtype)

    def checkDynamicQuantizedLinearRelu(self, mod, dtype):
        r"""Checks that mod has been swapped for an nnqd.Linear
        module, the bias is float.
        """
        self.assertEqual(type(mod), nniqd.LinearReLU)
        self.assertEqual(mod._packed_params.dtype, dtype)

    def check_eager_serialization(self, ref_model, loaded_model, x):
        # Check state dict serialization and torch.save APIs
        model_dict = ref_model.state_dict()
        b = io.BytesIO()
        torch.save(model_dict, b)
        b.seek(0)
        # weights_only=False as we sometimes get a ScriptObject here (weird)
        loaded_dict = torch.load(b, weights_only=False)
        loaded_model.load_state_dict(loaded_dict)
        ref_out = ref_model(*x)
        load_out = loaded_model(*x)

        def check_outputs(ref_out, load_out):
            self.assertEqual(ref_out[0], load_out[0])
            if isinstance(ref_out[1], tuple):
                self.assertEqual(ref_out[1][0], load_out[1][0])
                self.assertEqual(ref_out[1][1], load_out[1][1])
            else:
                self.assertEqual(ref_out[1], load_out[1])

        check_outputs(ref_out, load_out)
        b = io.BytesIO()
        torch.save(ref_model, b)
        b.seek(0)
        # weights_only=False as this is legacy code that saves the model
        loaded = torch.load(b, weights_only=False)
        load_out = loaded(*x)
        check_outputs(ref_out, load_out)

    def check_weight_bias_api(self, ref_model, weight_keys, bias_keys):
        weight = ref_model.get_weight()
        bias = ref_model.get_bias()
        self.assertEqual(weight_keys ^ weight.keys(), set())
        self.assertEqual(bias_keys ^ bias.keys(), set())

    def checkDynamicQuantizedLSTM(self, mod, reference_module_type, dtype):
        r"""Checks that mod has been swapped for an nnqd.LSTM type
        module, the bias is float.
        """
        wt_dtype_map = {
            torch.qint8: "quantized_dynamic",
            torch.float16: "quantized_fp16",
        }
        self.assertEqual(type(mod), reference_module_type)
        for packed_params in mod._all_weight_values:
            self.assertEqual(
                packed_params.param.__getstate__()[0][0], wt_dtype_map[dtype]
            )

    def checkLinear(self, mod):
        self.assertEqual(type(mod), torch.nn.Linear)

    def checkDynamicQuantizedModule(self, mod, reference_module_type, dtype):
        r"""Checks that mod has been swapped for an nnqd.Linear
        module, the bias is float.
        """
        wt_dtype_map = {
            torch.qint8: "quantized_dynamic",
            torch.float16: "quantized_fp16",
        }
        self.assertEqual(type(mod), reference_module_type)
        if hasattr(mod, "_all_weight_values"):
            for packed_params in mod._all_weight_values:
                self.assertEqual(
                    packed_params.param.__getstate__()[0][0], wt_dtype_map[dtype]
                )

    def checkScriptable(self, orig_mod, calib_data, check_save_load=False):
        scripted = torch.jit.script(orig_mod)
        self._checkScriptable(orig_mod, scripted, calib_data, check_save_load)

        # Use first calib_data entry as trace input
        traced = torch.jit.trace(orig_mod, calib_data[0])
        self._checkScriptable(orig_mod, traced, calib_data, check_save_load)

    # Call this twice: once for a scripted module and once for a traced module
    def _checkScriptable(self, orig_mod, script_mod, calib_data, check_save_load):
        self._checkModuleCorrectnessAgainstOrig(orig_mod, script_mod, calib_data)

        # Test save/load
        buffer = io.BytesIO()
        torch.jit.save(script_mod, buffer)

        buffer.seek(0)
        loaded_mod = torch.jit.load(buffer)
        # Pending __get_state_ and __set_state__ support
        # See tracking task https://github.com/pytorch/pytorch/issues/23984
        if check_save_load:
            self._checkModuleCorrectnessAgainstOrig(orig_mod, loaded_mod, calib_data)

    def _checkModuleCorrectnessAgainstOrig(self, orig_mod, test_mod, calib_data):
        for inp in calib_data:
            ref_output = orig_mod(*inp)
            scripted_output = test_mod(*inp)
            self.assertEqual(scripted_output, ref_output)

    def checkGraphModeOp(
        self,
        module,
        inputs,
        quantized_op,
        tracing=False,
        debug=False,
        check=True,
        eval_mode=True,
        dynamic=False,
        qconfig=None,
    ):
        if debug:
            print("Testing:", str(module))
        qconfig_dict = {"": get_default_qconfig(torch.backends.quantized.engine)}

        if eval_mode:
            module = module.eval()
        if dynamic:
            qconfig_dict = {"": default_dynamic_qconfig if qconfig is None else qconfig}
        model = get_script_module(module, tracing, inputs[0]).eval()
        if debug:
            print("input graph:", model.graph)
        models = {}
        outputs = {}
        for debug in [True, False]:
            if dynamic:
                models[debug] = quantize_dynamic_jit(model, qconfig_dict, debug=debug)
                # make sure it runs
                outputs[debug] = models[debug](inputs)
            else:
                # module under test can contain in-place ops, and we depend on
                # input data staying constant for comparisons
                inputs_copy = copy.deepcopy(inputs)
                models[debug] = quantize_jit(
                    model,
                    qconfig_dict,
                    test_only_eval_fn,
                    [inputs_copy],
                    inplace=False,
                    debug=debug,
                )
                # make sure it runs
                outputs[debug] = models[debug](*inputs[0])

        if debug:
            print("debug graph:", models[True].graph)
            print("non debug graph:", models[False].graph)

        if check:
            # debug and non-debug option should have the same numerics
            self.assertEqual(outputs[True], outputs[False])

            # non debug graph should produce quantized op
            FileCheck().check(quantized_op).run(models[False].graph)

        return models[False]

    def checkGraphModuleNodes(
        self,
        graph_module,
        expected_node=None,
        expected_node_occurrence=None,
        expected_node_list=None,
    ):
        """Check if GraphModule contains the target node
        Args:
            graph_module: the GraphModule instance we want to check
            expected_node, expected_node_occurrence, expected_node_list:
               see docs for checkGraphModeFxOp
        """
        nodes_in_graph = {}
        node_list = []
        modules = dict(graph_module.named_modules(remove_duplicate=False))
        for node in graph_module.graph.nodes:
            n = None
            if node.op == "call_function" or node.op == "call_method":
                n = NodeSpec(node.op, node.target)
            elif node.op == "call_module":
                n = NodeSpec(node.op, type(modules[node.target]))

            if n is not None:
                node_list.append(n)
                if n in nodes_in_graph:
                    nodes_in_graph[n] += 1
                else:
                    nodes_in_graph[n] = 1

        if expected_node is not None:
            self.assertTrue(
                expected_node in nodes_in_graph,
                "node:" + str(expected_node) + " not found in the graph module",
            )

        if expected_node_occurrence is not None:
            for expected_node, occurrence in expected_node_occurrence.items():
                if occurrence != 0:
                    self.assertTrue(
                        expected_node in nodes_in_graph,
                        "Check failed for node:" + str(expected_node) + " not found",
                    )
                    self.assertTrue(
                        nodes_in_graph[expected_node] == occurrence,
                        "Check failed for node:"
                        + str(expected_node)
                        + " Expected occurrence:"
                        + str(occurrence)
                        + " Found occurrence:"
                        + str(nodes_in_graph[expected_node]),
                    )
                else:
                    self.assertTrue(
                        expected_node not in nodes_in_graph,
                        "Check failed for node:"
                        + str(expected_node)
                        + " expected no occurrence but found",
                    )

        if expected_node_list is not None:
            cur_index = 0
            for n in node_list:
                if cur_index == len(expected_node_list):
                    return
                if n == expected_node_list[cur_index]:
                    cur_index += 1
            self.assertTrue(
                cur_index == len(expected_node_list),
                "Check failed for graph:"
                + self.printGraphModule(graph_module, print_str=False)
                + "Expected ordered list:"
                + str(expected_node_list),
            )

    def printGraphModule(self, graph_module, print_str=True):
        modules = dict(graph_module.named_modules(remove_duplicate=False))
        node_infos = []
        for n in graph_module.graph.nodes:
            node_info = " ".join(map(repr, [n.op, n.name, n.target, n.args, n.kwargs]))
            if n.op == "call_module":
                node_info += " module type: " + repr(type(modules[n.target]))
            node_infos.append(node_info)
        str_to_print = "\n".join(node_infos)
        if print_str:
            print(str_to_print)
        return str_to_print

    if HAS_FX:

        def assert_types_for_matched_subgraph_pairs(
            self,
            matched_subgraph_pairs: dict[str, tuple[NSSubgraph, NSSubgraph]],
            expected_types: dict[
                str, tuple[tuple[Callable, Callable], tuple[Callable, Callable]]
            ],
            gm_a: GraphModule,
            gm_b: GraphModule,
        ) -> None:
            """
            Verifies that the types specified in expected_types match
            the underlying objects pointed to by the nodes in matched_subgraph_pairs.

            An example successful test case:

              matched_subgraph_pairs = {'x0': (graph_a_conv_0_node, graph_b_conv_0_node)}
              expected_types = {'x0': (nn.Conv2d, nnq.Conv2d)}

            The function tests for key equivalence, and verifies types with
            instance checks.
            """

            def _get_underlying_op_type(
                node: Node, gm: GraphModule
            ) -> Union[Callable, str]:
                if node.op == "call_module":
                    mod = getattr(gm, node.target)
                    return type(mod)
                else:
                    assert node.op in ("call_function", "call_method")
                    return node.target

            self.assertTrue(
                len(matched_subgraph_pairs) == len(expected_types),
                f"Expected length of results to match, but got {len(matched_subgraph_pairs)} and {len(expected_types)}",
            )
            for k, v in expected_types.items():
                expected_types_a, expected_types_b = v
                exp_type_start_a, exp_type_end_a = expected_types_a
                exp_type_start_b, exp_type_end_b = expected_types_b
                subgraph_a, subgraph_b = matched_subgraph_pairs[k]

                act_type_start_a = _get_underlying_op_type(subgraph_a.start_node, gm_a)
                act_type_start_b = _get_underlying_op_type(subgraph_b.start_node, gm_b)
                act_type_end_a = _get_underlying_op_type(subgraph_a.end_node, gm_a)
                act_type_end_b = _get_underlying_op_type(subgraph_b.end_node, gm_b)
                types_match = (
                    (exp_type_start_a is act_type_start_a)
                    and (exp_type_end_a is act_type_end_a)
                    and (exp_type_start_b is act_type_start_b)
                    and (exp_type_end_b is act_type_end_b)
                )
                self.assertTrue(
                    types_match,
                    f"Type mismatch at {k}: expected {(exp_type_start_a, exp_type_end_a, exp_type_start_b, exp_type_end_b)}, "
                    f"got {(act_type_start_a, act_type_end_a, act_type_start_b, act_type_end_b)}",
                )

        def assert_ns_compare_dict_valid(
            self,
            act_compare_dict: dict[str, dict[str, dict[str, Any]]],
        ) -> None:
            """
            Verifies that the act_compare_dict (output of Numeric Suite APIs) is valid:
            1. for each layer, results are recorded for two models
            2. number of seen tensors match
            3. shapes of each pair of seen tensors match
            """
            for layer_name, result_type_to_data in act_compare_dict.items():
                for result_type, layer_data in result_type_to_data.items():
                    self.assertTrue(
                        len(layer_data) == 2,
                        f"Layer {layer_name} does not have exactly two model results.",
                    )
                    model_name_0, model_name_1 = layer_data.keys()
                    for res_idx in range(len(layer_data[model_name_0])):
                        layer_data_0 = layer_data[model_name_0][res_idx]
                        layer_data_1 = layer_data[model_name_1][res_idx]
                        self.assertTrue(
                            layer_data_0["type"] == layer_data_0["type"],
                            f"Layer {layer_name}, {model_name_0} and {model_name_1} do not have the same type.",
                        )

                        self.assertTrue(
                            len(layer_data_0["values"]) == len(layer_data_1["values"]),
                            f"Layer {layer_name}, {model_name_0} and {model_name_1} do not have the same number of seen Tensors.",
                        )

                        # F.conv1d weight has rank 3, and toq.conv1d unpacked weight
                        # has rank 4. For now, skip the length check for conv1d only.
                        is_weight_functional_conv1d = (
                            result_type == NSSingleResultValuesType.WEIGHT.value
                            and (
                                "conv1d" in layer_data_0["prev_node_target_type"]
                                or "conv1d" in layer_data_1["prev_node_target_type"]
                            )
                        )
                        if not is_weight_functional_conv1d:
                            for idx in range(len(layer_data_0["values"])):
                                values_0 = layer_data_0["values"][idx]
                                values_1 = layer_data_1["values"][idx]
                                if isinstance(values_0, torch.Tensor):
                                    self.assertTrue(
                                        values_0.shape == values_1.shape,
                                        f"Layer {layer_name}, {model_name_0} and {model_name_1} "
                                        + f"have a shape mismatch at idx {idx}.",
                                    )
                                elif isinstance(values_0, list):
                                    values_0 = values_0[0]
                                    values_1 = values_1[0]
                                    self.assertTrue(
                                        values_0.shape == values_1.shape,
                                        f"Layer {layer_name}, {model_name_0} and {model_name_1} "
                                        + f"have a shape mismatch at idx {idx}.",
                                    )
                                else:
                                    assert isinstance(
                                        values_0, tuple
                                    ), f"unhandled type {type(values_0)}"
                                    assert len(values_0) == 2
                                    assert len(values_0[1]) == 2
                                    assert values_0[0].shape == values_1[0].shape
                                    assert values_0[1][0].shape == values_1[1][0].shape
                                    assert values_0[1][1].shape == values_1[1][1].shape

                        # verify that ref_node_name is valid
                        ref_node_name_0 = layer_data_0["ref_node_name"]
                        ref_node_name_1 = layer_data_1["ref_node_name"]
                        prev_node_name_0 = layer_data_0["prev_node_name"]
                        prev_node_name_1 = layer_data_1["prev_node_name"]
                        if (
                            layer_data_0["type"]
                            == NSSingleResultValuesType.NODE_OUTPUT.value
                        ):
                            self.assertTrue(ref_node_name_0 == prev_node_name_0)
                            self.assertTrue(ref_node_name_1 == prev_node_name_1)
                        elif (
                            layer_data_0["type"]
                            == NSSingleResultValuesType.NODE_INPUT.value
                        ):
                            self.assertTrue(ref_node_name_0 != prev_node_name_0)
                            self.assertTrue(ref_node_name_1 != prev_node_name_1)

        def checkGraphModeFxOp(
            self,
            model,
            inputs,
            quant_type,
            expected_node=None,
            expected_node_occurrence=None,
            expected_node_list=None,
            is_reference=False,
            print_debug_info=False,
            custom_qconfig_dict=None,
            prepare_expected_node=None,
            prepare_expected_node_occurrence=None,
            prepare_expected_node_list=None,
            prepare_custom_config=None,
            backend_config=None,
        ):
            """Quantizes model with graph mode quantization on fx and check if the
            quantized model contains the quantized_node

            Args:
                model: floating point torch.nn.Module
                inputs: one positional sample input arguments for model
                expected_node: NodeSpec
                    e.g. NodeSpec.call_function(torch.quantize_per_tensor)
                expected_node_occurrence: a dict from NodeSpec to
                    expected number of occurrences (int)
                    e.g. {NodeSpec.call_function(torch.quantize_per_tensor) : 1,
                            NodeSpec.call_method('dequantize'): 1}
                expected_node_list: a list of NodeSpec, used to check the order
                    of the occurrence of Node
                    e.g. [NodeSpec.call_function(torch.quantize_per_tensor),
                            NodeSpec.call_module(nnq.Conv2d),
                            NodeSpec.call_function(F.hardtanh_),
                            NodeSpec.call_method('dequantize')]
                is_reference: if True, enables reference mode
                print_debug_info: if True, prints debug info
                custom_qconfig_dict: overrides default qconfig_dict
                prepare_expected_node: same as expected_node, but for prepare
                prepare_expected_node_occurrence: same as
                    expected_node_occurrence, but for prepare
                prepare_expected_node_list: same as expected_node_list, but
                    for prepare

            Returns:
                A dictionary with the following structure:
               {
                   "prepared": ...,  # the prepared model
                   "quantized": ...,  # the quantized non-reference model
                   "quantized_reference": ...,  # the quantized reference model
                   "result": ...,  # the result for either quantized or
                                   # quantized_reference model depending on the
                                   # is_reference argument
               }
            """
            # TODO: make img_data a single example instead of a list
            if type(inputs) is list:
                inputs = inputs[0]

            if quant_type == QuantType.QAT:
                qconfig_mapping = get_default_qat_qconfig_mapping(
                    torch.backends.quantized.engine
                )
                model.train()
            elif quant_type == QuantType.STATIC:
                qconfig_mapping = get_default_qconfig_mapping(
                    torch.backends.quantized.engine
                )
                model.eval()
            else:
                qconfig = default_dynamic_qconfig
                qconfig_mapping = QConfigMapping().set_global(qconfig)
                model.eval()

            if quant_type == QuantType.QAT:
                prepare = prepare_qat_fx
            else:
                prepare = prepare_fx

            # overwrite qconfig_dict with custom_qconfig_dict
            if custom_qconfig_dict is not None:
                assert type(custom_qconfig_dict) in (
                    QConfigMapping,
                    dict,
                ), "custom_qconfig_dict should be a QConfigMapping or a dict"
                if isinstance(custom_qconfig_dict, QConfigMapping):
                    qconfig_mapping = custom_qconfig_dict
                else:
                    qconfig_mapping = QConfigMapping.from_dict(custom_qconfig_dict)
            prepared = prepare(
                model,
                qconfig_mapping,
                example_inputs=inputs,
                prepare_custom_config=prepare_custom_config,
                backend_config=backend_config,
            )
            if quant_type != QuantType.DYNAMIC:
                prepared(*inputs)

            if print_debug_info:
                print()
                print("quant type:\n", quant_type)
                print("original model:\n", model)
                print()
                print("prepared model:\n", prepared)

            self.checkGraphModuleNodes(
                prepared,
                prepare_expected_node,
                prepare_expected_node_occurrence,
                prepare_expected_node_list,
            )

            prepared_copy = copy.deepcopy(prepared)
            qgraph = convert_fx(copy.deepcopy(prepared))
            qgraph_reference = convert_to_reference_fx(copy.deepcopy(prepared))
            result = qgraph(*inputs)
            result_reference = qgraph_reference(*inputs)
            qgraph_copy = copy.deepcopy(qgraph)
            qgraph_reference_copy = copy.deepcopy(qgraph_reference)

            qgraph_to_check = qgraph_reference if is_reference else qgraph
            if print_debug_info:
                print()
                print("quantized model:\n", qgraph_to_check)
                self.printGraphModule(qgraph_to_check)
                print()
            self.checkGraphModuleNodes(
                qgraph_to_check,
                expected_node,
                expected_node_occurrence,
                expected_node_list,
            )
            return {
                "prepared": prepared_copy,
                "quantized": qgraph_copy,
                "quantized_reference": qgraph_reference_copy,
                "quantized_output": result,
                "quantized_reference_output": result_reference,
            }

    def checkEmbeddingSerialization(
        self,
        qemb,
        num_embeddings,
        embedding_dim,
        indices,
        offsets,
        set_qconfig,
        is_emb_bag,
        dtype=torch.quint8,
    ):
        # Test serialization of dynamic EmbeddingBag module using state_dict
        if is_emb_bag:
            inputs = [indices, offsets]
        else:
            inputs = [indices]
        emb_dict = qemb.state_dict()
        b = io.BytesIO()
        torch.save(emb_dict, b)
        b.seek(0)
        loaded_dict = torch.load(b)
        embedding_unpack = torch.ops.quantized.embedding_bag_unpack
        # Check unpacked weight values explicitly
        for key in emb_dict:
            if isinstance(emb_dict[key], torch._C.ScriptObject):
                assert isinstance(loaded_dict[key], torch._C.ScriptObject)
                emb_
```



## High-Level Overview

r"""Importing this file includes common utility methods and base classes forchecking quantization api and properties of resulting modules.

This Python file contains 108 class(es) and 325 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `NodeSpec`, `AverageMeter`, `QuantizationTestCase`, `QuantizationLiteTestCase`, `PT2EQuantizationTestCase`, `M`, `SingleLayerLinearModel`, `AnnotatedSingleLayerLinearModel`, `SingleLayerLinearDynamicModel`, `LinearAddModel`, `RNNDynamicModel`, `RNNCellDynamicModel`, `LSTMwithHiddenDynamicModel`, `ConvModel`, `ConvTransposeModel`, `AnnotatedConvModel`, `AnnotatedConvTransposeModel`, `ConvBnModel`, `AnnotatedConvBnModel`, `ConvBnReLUModel`

**Functions defined**: `__init__`, `call_function`, `call_method`, `call_module`, `__hash__`, `__eq__`, `__repr__`, `get_supported_device_types`, `test_only_eval_fn`, `test_only_train_fn`, `__init__`, `reset`, `update`, `__str__`, `accuracy`, `train_one_epoch`, `ddp_setup`, `ddp_cleanup`, `run_ddp`, `convert_dynamic`

**Key imports**: torch, torch.ao.nn.intrinsic.quantized.dynamic as nniqd, torch.ao.nn.quantized as nnq, torch.ao.nn.quantized.dynamic as nnqd, torch.distributed as dist, torch.nn as nn, torch.nn.functional as F, control_flow, _FusedModule, get_executorch_backend_config


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/testing/_internal`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.ao.nn.intrinsic.quantized.dynamic as nniqd`
- `torch.ao.nn.quantized as nnq`
- `torch.ao.nn.quantized.dynamic as nnqd`
- `torch.distributed as dist`
- `torch.nn as nn`
- `torch.nn.functional as F`
- `functorch.experimental`: control_flow
- `torch.ao.nn.intrinsic`: _FusedModule
- `torch.ao.quantization.backend_config`: get_executorch_backend_config
- `torch.export`: export
- `torch.jit.mobile`: _load_for_lite_interpreter
- `torch.testing._internal.common_quantized`: override_quantized_engine
- `torch.testing._internal.common_utils`: TEST_WITH_ROCM, TestCase
- `torch.ao.ns.fx.ns_types`: NSSingleResultValuesType, NSSubgraph
- `torch.fx`: GraphModule
- `torch.fx.graph`: Node
- `contextlib`
- `copy`
- `functools`
- `io`
- `os`
- `unittest`
- `typing`: Any, Optional, Union
- `collections.abc`: Callable
- `numpy as np`
- `torch._dynamo as torchdynamo`
- `torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq`
- `torch.ao.quantization.quantizer.xpu_inductor_quantizer as xpuiq`
- `torch.ao.quantization.quantizer.x86_inductor_quantizer`: X86InductorQuantizer


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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
python torch/testing/_internal/common_quantization.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/testing/_internal`):

- [`common_jit.py_docs.md`](./common_jit.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`autograd_function_db.py_docs.md`](./autograd_function_db.py_docs.md)
- [`custom_op_db.py_docs.md`](./custom_op_db.py_docs.md)
- [`subclasses.py_docs.md`](./subclasses.py_docs.md)
- [`logging_utils.py_docs.md`](./logging_utils.py_docs.md)
- [`two_tensor.py_docs.md`](./two_tensor.py_docs.md)
- [`autocast_test_lists.py_docs.md`](./autocast_test_lists.py_docs.md)
- [`hypothesis_utils.py_docs.md`](./hypothesis_utils.py_docs.md)
- [`common_mkldnn.py_docs.md`](./common_mkldnn.py_docs.md)


## Cross-References

- **File Documentation**: `common_quantization.py_docs.md`
- **Keyword Index**: `common_quantization.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
