# Documentation: `test/dynamo/test_modules.py`

## File Metadata

- **Path**: `test/dynamo/test_modules.py`
- **Size**: 112,141 bytes (109.51 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]
# ruff: noqa: F841

import collections
import copy
import itertools
import os
import tempfile
import traceback
import types
import unittest
from copy import deepcopy
from functools import partial
from typing import NamedTuple
from unittest.mock import patch

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch.nn.functional as F
from torch._dynamo.debug_utils import same_two_models
from torch._dynamo.eval_frame import unsupported
from torch._dynamo.mutation_guard import GenerationTracker
from torch._dynamo.testing import expectedFailureDynamic, same
from torch._dynamo.utils import ifdynstaticdefault
from torch._dynamo.variables.torch_function import TensorWithTFOverrideVariable
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import skipIfHpu


try:
    from . import test_functions
except ImportError:
    import test_functions


_variable = 0
_variable1 = 0


def update_global():
    global _variable, _variable1
    _variable += 1
    _variable1 += 1


class BasicModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        return F.relu(self.linear1(x)) * self.scale


class FnMember(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.activation = F.relu

    def forward(self, x):
        x = self.linear1(x)
        if self.activation:
            x = self.activation(x)
        return x


class FnMemberCmp(torch.nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.activation = activation

    def forward(self, x):
        x = self.linear1(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.activation is None:
            x = torch.sigmoid(x)
        return x


class SubmoduleExample(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x * self.scale


class IsTrainingCheck(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)
        self.train(True)

    def forward(self, x):
        if self.training:
            mod = self.linear1
        else:
            mod = self.linear2
        return F.relu(mod(x))


class IsEvalCheck(IsTrainingCheck):
    def __init__(self) -> None:
        super().__init__()
        self.train(False)


class ModuleMethodCall(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    def call_and_scale(self, mod, x):
        x = mod(x)
        return x * self.scale

    def forward(self, x):
        x1 = self.call_and_scale(self.layer1, x)
        x2 = self.call_and_scale(self.layer2, x)
        return x1 + x2


class UnsupportedMethodCall(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.scale = torch.randn(1, 10)

    def call_and_scale(self, mod, x):
        x = mod(x)
        x = x * self.scale
        return unsupported(x, x)

    def forward(self, x):
        x1 = self.call_and_scale(self.layer1, x)
        return x + x1


class UnsupportedModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        x = self.layer1(x) * self.scale
        return unsupported(x, x)


class UnsupportedModuleCall(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mod = UnsupportedModule()

    def forward(self, x):
        return 1 + self.mod(x * 1.5)


class ModuleWithStaticForward(torch.nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class ModuleCallModuleWithStaticForward(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mod = ModuleWithStaticForward()

    def forward(self, x):
        return self.mod(x)


class ModuleStaticMethodCall(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    @staticmethod
    def call_and_scale(scale, mod, x):
        x = mod(x)
        return x * scale

    def forward(self, x):
        x1 = self.call_and_scale(self.scale, self.layer1, x)
        x2 = self.call_and_scale(self.scale, self.layer2, x)
        return x1 + x2


class ModuleClassMethodCall(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    @classmethod
    def call_and_scale(cls, scale, mod, x):
        x = mod(x)
        return x * scale

    def forward(self, x):
        x1 = self.call_and_scale(self.scale, self.layer1, x)
        x2 = self.call_and_scale(self.scale, self.layer2, x)
        return x1 + x2


class ModuleProperty(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.randn(1, 10)

    @property
    def scale_alias(self):
        return self.scale

    def forward(self, x):
        return x * self.scale_alias


class NestedModuleList(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(3):
            self.layers.append(
                torch.nn.ModuleList(
                    [
                        torch.nn.Linear(10, 10),
                        torch.nn.ReLU(),
                    ]
                )
            )

    def forward(self, x):
        for layer, act in self.layers:
            x = act(layer(x))
        return x


class ConstLoop(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.count = 3

    def forward(self, x):
        for _ in range(self.count):
            x = torch.sigmoid(self.linear1(x))
        return x


class ViaModuleCall(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)

    def forward(self, x):
        return test_functions.constant3(torch.sigmoid(self.linear1(x)), x)


class IsNoneLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = None
        self.train(True)

    def forward(self, x):
        if self.layer1 is not None:
            x = self.layer1(x)
        if self.layer2 is not None:
            x = self.layer2(x)
        return x


class LayerList(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ModuleList(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
            ]
        )

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)

        for layer in self.layers:
            x = layer(x)

        for layer, val in zip(self.layers, (x, x, x, x)):
            x = layer(x) + val

        for layer, val in zip(self.layers, (1, 2, 3, 4)):
            x = layer(x) + val

        for idx, layer in enumerate(self.layers):
            x = layer(x) * idx

        for idx, layer in enumerate(self.layers[::-1]):
            x = layer(x) * idx

        return x


class CustomGetItemModuleList(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
            ]
        )

    def __getitem__(self, idx: int):
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)

    def forward(self, x):
        for i in range(len(self)):
            x = self[i](x)

        return x


class ModuleDict(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleDict(
            {
                "0": torch.nn.Linear(10, 10),
            }
        )

    def forward(self, x):
        # TODO(future PR): handle more logic
        x = self.layers["0"](x)
        return x


class ParameterDict(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ParameterDict(
            {
                "0": torch.nn.Parameter(torch.randn(10, 10)),
            }
        )

    def forward(self, x):
        x = self.layers["0"].mm(x)
        return x


class CustomGetItemParameterDict(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ParameterDict(
            {
                "0": torch.nn.Parameter(torch.randn(10, 10)),
            }
        )

    def __getitem__(self, key: str) -> torch.nn.Module:
        return self.layers[key]

    def forward(self, x):
        x = self["0"].mm(x)
        return x


class CustomGetItemModuleDict(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleDict(
            {
                "0": torch.nn.Linear(10, 10),
            }
        )

    def __getitem__(self, key: str) -> torch.nn.Module:
        return self.layers[key]

    def forward(self, x):
        x = self["0"](x)
        return x


class TensorList(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = (
            torch.randn((1, 10)),
            torch.randn((10, 1)),
            torch.randn((1, 10)),
            torch.randn((10, 1)),
        )

    def forward(self, x):
        for layer in self.layers:
            x = x * layer
        return x


class Children(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(10, 10)
        self.l2 = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(10, 10)
        self.l4 = torch.nn.ReLU()

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class NamedChildren(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(10, 10)
        self.l2 = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(10, 10)
        self.l4 = torch.nn.ReLU()

    def forward(self, x):
        for _, block in self.named_children():
            x = block(x)
        return x


class IntArg(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)

    def forward(self, x, offset=1):
        x = F.relu(self.layer1(x)) + offset
        return x


class Seq(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class Cfg:
    def __init__(self) -> None:
        self.val = 0.5
        self.count = 3


class CfgModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = Cfg()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        for _ in range(self.cfg.count):
            x = self.layer(x + self.cfg.val)
        return x


class StringMember(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.mode = "some_string"

    def forward(self, x):
        if self.mode == "some_string":
            return F.relu(self.linear1(x))


class _Block(torch.nn.Module):
    def forward(self, x):
        return 1.5 * torch.cat(x, 1)


class _DenseBlock(torch.nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            self.add_module(f"denselayer{i + 1:d}", _Block())

    def forward(self, init_features):
        features = [init_features]
        for layer in self.values():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNetBlocks(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = _DenseBlock()

    def forward(self, x):
        return self.layers(x)


class MaterializedModule(torch.nn.Module):
    """Once the below lazy module is initialized with its first input,
    it is transformed into this module."""

    param: Parameter

    def __init__(self) -> None:
        super().__init__()
        self.register_parameter("param", None)

    def forward(self, x):
        return x


class LazyModule(LazyModuleMixin, MaterializedModule):
    param: UninitializedParameter
    cls_to_become = MaterializedModule

    def __init__(self) -> None:
        super().__init__()
        self.param = UninitializedParameter()

    def initialize_parameters(self, x):
        # force graph break to ensure this was not inlined
        torch._dynamo.graph_break()
        self.param.materialize(x.shape)


class LazyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.LazyLinear(10)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.LazyLinear(1)
        self.relu2 = torch.nn.ReLU()

    def forward(self, input):
        x = self.relu1(self.fc1(input))
        y = self.relu2(self.fc2(x))
        return y


class MyInput(NamedTuple):
    x: dict[str, dict[str, torch.Tensor]]
    y: torch.Tensor


class LazyLayerWithNamedTupleInput(LazyModuleMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def initialize_parameters(self, input):
        with torch.no_grad():
            self._param = torch.nn.Parameter(
                torch.empty(input.x["a"][0].shape).fill_(0.5)
            )

    def forward(self, input):
        input = input.x["a"]
        x = 0
        for i in range(len(input)):
            x = x + input[i]
        return x


class LazyModuleWithNamedTupleInput(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = LazyLayerWithNamedTupleInput()

    def forward(self, input):
        return self.layer(input)


class LazyLayerWithListInput(LazyModuleMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def initialize_parameters(self, input):
        with torch.no_grad():
            self._param = torch.nn.Parameter(torch.empty(input[0].shape).fill_(0.5))

    def forward(self, input):
        x = 0
        for i in range(len(input)):
            x = x + input[i]
        return x


class LazyModuleWithListInput(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = LazyLayerWithListInput()

    def forward(self, input):
        return self.layer(input[:-1])


class LazyModuleWithLazySubmodule(LazyModuleMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def initialize_parameters(self, input):
        with torch.no_grad():
            self.layer = LazyLayerWithListInput()

    def forward(self, x):
        return self.layer(x)


class LazyLayerWithInputs(LazyModuleMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def initialize_parameters(self, x, y):
        with torch.no_grad():
            self._param_x = torch.nn.Parameter(torch.empty(x[0].shape).fill_(0.5))
            self._param_y = torch.nn.Parameter(torch.empty(y[0].shape).fill_(0.5))

    def forward(self, x, y):
        res_x = 0
        for i in range(len(x)):
            res_x = res_x + x[i]
        res_y = 0
        for i in range(len(y)):
            res_y = res_y + y[i]
        return res_x + res_y


class LazyModuleKwArgs(LazyModuleMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def initialize_parameters(self, *args, **kwargs):
        with torch.no_grad():
            self.layer = LazyLayerWithInputs()

    def forward(self, x, y):
        return self.layer(x, y=y)


class LazyModuleBadInferParams(LazyModuleMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def initialize_parameters(self, *args, **kwargs):
        self.foo += 1

    def forward(self, x, y):
        return self.layer(x, y=y)


class LazyParentModule(LazyModuleMixin, torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def impl(self, x):
        return x.cos() + self._val


class LazyChildModuleNoClsToBecome(LazyParentModule):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return super().impl(x.sin())

    def initialize_parameters(self, input):
        self._val = torch.nn.Parameter(torch.ones(2, 2))


def requires_grad1(module: torch.nn.Module, recurse: bool = False) -> bool:
    requires_grad = any(p.requires_grad for p in module.parameters(recurse))
    return requires_grad


def requires_grad2(module: torch.nn.Module, recurse: bool = False) -> bool:
    requires_grad = any(p.requires_grad for p in module.parameters(recurse))
    return requires_grad


class ParametersModule1(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.nn.Parameter(torch.randn(1, 10))

    def forward(self, x):
        if not requires_grad1(self):
            return F.relu(self.linear1(x)) * self.scale
        else:
            return x + 1


class ParametersModule2(ParametersModule1):
    def forward(self, x):
        if not requires_grad2(self):
            return F.relu(self.linear1(x)) * self.scale
        else:
            return x + 1


class ParametersModule3(ParametersModule1):
    def forward(self, x):
        ones = torch.ones(10, dtype=next(self.parameters()).dtype)
        return F.relu(self.linear1(x)) * self.scale + ones


class ParametersModule4(ParametersModule1):
    def forward(self, x):
        ones = torch.ones(10, dtype=next(self.parameters(recurse=False)).dtype)
        return F.relu(self.linear1(x)) * self.scale + ones


class ParametersModule5(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.nn.Parameter(torch.randn(10, 10))
        self.scale_dup = self.scale

    def forward(self, x):
        counter = 0
        for _param in self.parameters():
            counter += 1

        return x * self.scale * counter


class SuperModule(BasicModule):
    def forward(self, x):
        x = super().forward(x)
        return x + 10.0


class SuperModule2(BasicModule):
    def forward(self, x):
        return BasicModule.forward(self, x)


class ComplicatedSuperParent(torch.nn.Module):
    @classmethod
    def custom_add(cls, x):
        x = x + x
        return x


class SuperChildCallsClassMethod(ComplicatedSuperParent):
    @classmethod
    def child_func(cls, x):
        x = super().custom_add(x)
        return x

    def forward(self, x):
        x = self.child_func(x)
        return x


class HasAttrModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.randn(1, 10))

    def forward(self, x):
        x = F.relu(x)
        if hasattr(self, "scale"):
            x *= self.scale
        if hasattr(self, "scale2"):
            x *= self.scale2
        return x


class EnumValues(torch.nn.ModuleDict):
    def __init__(
        self,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            self.add_module(f"denselayer{i + 1:d}", _Block())

    def forward(self, init_features):
        features = [init_features]
        for layer in self.values():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class AccessByKeys(torch.nn.ModuleDict):
    def __init__(
        self,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            self.add_module(f"denselayer{i + 1:d}", _Block())

    def forward(self, init_features):
        features = [init_features]
        for k in self.keys():
            new_features = self[k](features)
            features.append(new_features)
        return torch.cat(features, 1)


class CallForwardDirectly(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        return x


class ConvCallForwardDirectly(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)

    def forward(self, x):
        return self.layer.forward(x)


class ConvTransposeCallForwardDirectly(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.ConvTranspose2d(4, 4, 4)

    def forward(self, x):
        return self.layer.forward(x)


class ConvCallSuperForwardDirectly(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs,
        )

    def forward(self, inputs, mask=None):
        outputs = super().forward(inputs)
        return outputs


class ConvTransposeCallSuperForwardDirectly(torch.nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs,
        )

    def forward(self, x):
        if x.numel() > 0:
            return super().forward(x)
        output_shape = [
            ((i - 1) * d - 2 * p + (di * (k - 1) + 1) + op)
            for i, p, di, k, d, op in zip(
                x.shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride,
                self.output_padding,
            )
        ]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)  # noqa: F821


class ModuleNameString(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)

    def forward(self, x):
        if self.__class__.__name__ == "ABC":
            return 10
        if self.linear1.__class__.__name__ == "Linear":
            return F.relu(self.linear1(x) + 10)
        return 11


class SelfMutatingModule(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.counter = 0

    def forward(self, x):
        result = self.layer(x) + self.counter
        self.counter += 1
        return F.relu(result)


class ModuleAttributePrecedenceBase(torch.nn.Module):
    def linear(self, x, flag=None):
        if flag:
            return x * 2.0
        return x * 3.0


class ModuleAttributePrecedence(ModuleAttributePrecedenceBase):
    def __init__(self) -> None:
        super().__init__()
        self.activation = torch.nn.ReLU()
        self.linear = torch.nn.Linear(10, 10)
        self.initializer = torch.ones([10, 10])
        self.scale = 0.5

    def activation(self, x):
        return x * 1.2

    def initializer(self):
        return torch.zeros([10, 10])

    def scale(self):
        return 2.0

    def forward(self, x):
        # object attribute takes precedence unless it's a nn.Module
        return self.activation(self.linear(self.initializer + x)) * self.scale


class ModuleForwardHasGraphBreak(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.layer3 = torch.nn.Sequential(BasicModule(), BasicModule())
        self.layer4 = torch.nn.ModuleList(
            [
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 10),
                torch.nn.ReLU(),
            ]
        )
        self.layer5 = torch.nn.ModuleDict(
            {
                "0": torch.nn.Linear(10, 10),
            }
        )
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        """
        This is used to test if the results of functions like `named_parameters`
        can be reconstructed correctly after graph break.

        https://github.com/pytorch/torchdynamo/issues/1931
        """
        x = self.layer1(x)
        params1 = dict(self.named_parameters())
        params2 = list(self.parameters())
        buffers1 = dict(self.named_buffers())
        buffers2 = list(self.buffers())
        modules1 = dict(self.named_modules())
        modules2 = list(self.modules())
        torch._dynamo.graph_break()
        y = modules2
        y = modules1
        y = buffers2
        y = buffers1
        y = params2
        y = params1
        x = (
            self.layer2(x)
            + y["layer3.1.linear1.weight"]
            + y["layer4.2.weight"]
            + y["layer5.0.weight"]
        )
        return x * self.scale


class ModuleGuardNameIsValid(torch.nn.ModuleDict):
    # Guard names should be valid python identifier as we use eval() to get
    # corresponding guard value. Some guard names come from source(module path)
    # where special symbols are valid. But they are not valid python identifier,
    # we should identify these pattern and rewrite them with getattr.
    def __init__(self) -> None:
        super().__init__()
        for i in range(2):
            self.add_module(f"l@yer-{i + 1:d}", BasicModule())

    def forward(self, x):
        for layer in self.values():
            x = layer(x)
        return x


class SequentialWithDuplicatedModule(torch.nn.Module):
    # Sequential module(self.layer) contains three duplicated ReLU module.
    def __init__(self) -> None:
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            self.relu,
            torch.nn.Linear(20, 20),
            self.relu,
            torch.nn.Linear(20, 10),
            self.relu,
        )

    def forward(self, x):
        return self.layer(x)


class SequentialWithDuplicatedModule2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.layer = torch.nn.Sequential(
            collections.OrderedDict(
                [
                    ("linear1", torch.nn.Linear(10, 20)),
                    ("relu1", self.relu),
                    ("linear2", torch.nn.Linear(20, 20)),
                    ("relu2", self.relu),
                    ("linear3", torch.nn.Linear(20, 10)),
                    ("relu3", self.relu),
                ]
            )
        )

    def forward(self, x):
        return self.layer(x)


class ModuleComparison(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer0 = torch.nn.Linear(10, 10)
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = torch.nn.Linear(10, 10)

    @property
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2]

    def forward(self, x):
        for layer in self.encoder_layers:
            output = layer(x)
            if layer is None or layer == self.layer0:
                output = F.relu6(output)
            else:
                output = F.relu(output)
        return output


class ModulePatch1(torch.nn.Module):
    pass


class ModulePatch2(torch.nn.Module):
    def forward(self, x):
        return x - 1


class UnspecNonInlinableModule(torch.nn.Module):
    torchdynamo_force_dynamic = True  # forced to be a UnspecializedNNModule

    def forward(self, x):
        if x.sum() > 0:
            return x + 1
        else:
            return x - 1


class UnspecNonInlinableToplevelModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.m = UnspecNonInlinableModule()

    def forward(self, x):
        return self.m(x)


class ModuleWithIntAttr(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 4)
        self.step = 10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 1
        return self.layer(x) + self.step


class UnspecInlinableModule(torch.nn.Module):
    torchdynamo_force_dynamic = True  # forced to be a UnspecializedNNModule

    def forward(self, x):
        return torch.sin(x)


class UnspecModuleWithIntAttr(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = UnspecInlinableModule()
        self.step = 10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 1
        return self.layer(x) + self.step


def make_test(fn, expected_ops=None):
    def test_fn(self):
        return torch._dynamo.testing.standard_test(
            self, fn=fn, nargs=1, expected_ops=expected_ops
        )

    fn.eval()
    return test_fn


def temporary_tensor_subclass(torch_function=None):
    class TensorProxy(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if torch_function is not None:
                torch_function()
            return super().__torch_function__(func, types, args, kwargs)

    return TensorProxy


class NNModuleTests(torch._dynamo.test_case.TestCase):
    test_seq = make_test(Seq())
    test_basicmodule1 = make_test(BasicModule())
    test_basicmodule2 = make_test(BasicModule())
    test_submodules1 = make_test(SubmoduleExample())
    test_submodules2 = make_test(SubmoduleExample())
    test_modulemethod1 = make_test(ModuleMethodCall())
    test_modulemethod2 = make_test(ModuleMethodCall())
    test_module_call_module_with_static_forward = make_test(
        ModuleCallModuleWithStaticForward()
    )
    test_module_static_method = make_test(ModuleStaticMethodCall())
    test_fnmember = make_test(FnMember())
    test_fnmembercmp1 = make_test(FnMemberCmp(F.relu))
    test_fnmembercmp2 = make_test(FnMemberCmp(None))
    test_constloop = make_test(ConstLoop())
    test_istraining1 = make_test(IsTrainingCheck())
    test_istraining2 = make_test(IsTrainingCheck())
    test_iseval1 = make_test(IsEvalCheck())
    test_iseval2 = make_test(IsEvalCheck())
    test_viamodulecall = make_test(ViaModuleCall())
    test_isnonelayer = make_test(IsNoneLayer())
    test_layerlist = make_test(LayerList())
    test_tensorlist = make_test(TensorList())
    test_intarg = make_test(IntArg())
    test_cfgmod = make_test(CfgModule())
    test_stringmember = make_test(StringMember())
    test_modulelist = make_test(ModuleList())
    test_modulelist_nested = make_test(NestedModuleList())
    test_modulelist_custom = make_test(CustomGetItemModuleList())
    test_moduledict = make_test(ModuleDict())
    test_moduledict_custom = make_test(CustomGetItemModuleDict())
    test_parameterdict = make_test(ParameterDict())
    test_parameterdict_custom = make_test(CustomGetItemParameterDict())
    test_super1 = make_test(SuperModule())
    test_super2 = make_test(SuperModule2())
    test_super_class_method = make_test(SuperChildCallsClassMethod())
    test_children = make_test(Children())
    test_named_children = make_test(NamedChildren())
    test_densenet = make_test(DenseNetBlocks())
    test_parameters1 = make_test(ParametersModule1())
    test_parameters2 = make_test(ParametersModule2())
    test_parameters3 = make_test(ParametersModule3(), expected_ops=5)
    test_parameters4 = make_test(ParametersModule4())
    test_parameters5 = make_test(ParametersModule5())
    test_hasattr = make_test(HasAttrModule())
    test_enumvalues = make_test(EnumValues())
    test_access_by_keys = make_test(AccessByKeys())
    test_module_class_method = make_test(ModuleClassMethodCall())
    test_module_property = make_test(ModuleProperty())
    test_forward_directly = make_test(CallForwardDirectly())
    test_module_name_string = make_test(ModuleNameString())
    test_module_attribute_precedence = make_test(ModuleAttributePrecedence())
    test_module_guard_name_is_valid = make_test(ModuleGuardNameIsValid())
    test_sequential_with_duplicated_module = make_test(SequentialWithDuplicatedModule())
    test_sequential_with_duplicated_module2 = make_test(
        SequentialWithDuplicatedModule2()
    )
    test_module_comparison = make_test(ModuleComparison())

    def test_inject_module_parameters(self):
        from collections import OrderedDict

        class ZeROOrderedDict(OrderedDict):
            def __init__(self, parent_module=None, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._parent_module = parent_module

            def __getitem__(self, key):
                param = super().__getitem__(key)
                return param

        def inject_parameters(module, cls):
            for m in module.modules():
                if cls == ZeROOrderedDict:
                    new_param = cls(parent_module=m)
                else:
                    new_param = cls()

                for key, param in m._parameters.items():
                    new_param[key] = param
                m._parameters = new_param

        model = ParametersModule5()
        inject_parameters(model, ZeROOrderedDict)
        model = torch.compile(model, backend="inductor")
        x = torch.ones(10)
        # model can be compiled without error
        y = model(x)

    def test_module_forward_has_graph_break(self):
        m = ModuleForwardHasGraphBreak()
        x = torch.rand([10, 10])
        ref = m(x)
        opt_m = torch.compile(m, backend="eager")
        res = opt_m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_unsupportedmethod(self):
        m = UnsupportedMethodCall()
        i = torch.randn(10)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_m = torch.compile(m, backend=cnt)
        r = opt_m(i)
        self.assertTrue(torch._dynamo.testing.same(r, m(i)))
        self.assertEqual(cnt.op_count, 5)

    def test_unsupportedmodule(self):
        m = UnsupportedModuleCall()
        i = torch.randn(10)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_m = torch.compile(m, backend=cnt)
        r = opt_m(i)
        self.assertTrue(torch._dynamo.testing.same(r, m(i)))
        self.assertEqual(cnt.op_count, 6)

    @patch.object(torch._dynamo.config, "allow_unspec_int_on_nn_module", True)
    def test_self_mutating1(self):
        m1 = torch.nn.Linear(10, 10)
        m2 = SelfMutatingModule(m1)
        m3 = SelfMutatingModule(m1)
        m4 = SelfMutatingModule(m1)
        i = torch.randn(10)
        out2 = [m2(i), m2(i), m2(i)]
        cnt = torch._dynamo.testing.CompileCounter()
        opt_m3 = torch._dynamo.optimize_assert(cnt)(m3)
        opt_m4 = torch._dynamo.optimize_assert(cnt)(m4)
        out3 = [opt_m3(i), opt_m3(i), opt_m3(i)]
        out4 = [opt_m4(i), opt_m4(i), opt_m4(i)]
        self.assertTrue(torch._dynamo.testing.same(out2, out3))
        self.assertTrue(torch._dynamo.testing.same(out2, out4))
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(cnt.frame_count, """2""")
        else:
            self.assertExpectedInline(cnt.frame_count, """1""")

    def test_nn_module_setattr(self):
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.var = 0

        @torch.compile(backend="eager", dynamic=False)
        def f(x, m):
            return x + m.var

        inp = torch.ones(3)
        m = Mod()

        self.assertEqual(f(inp, m), inp)
        # In 3.13.0, setattr will not fire a __dict__'s watchers,
        # so guards may not be invalidated.
        m.var = 1
        # should trigger a recompile
        self.assertEqual(f(inp, m), inp + 1)

    @patch.object(torch._dynamo.config, "raise_on_ctx_manager_usage", False)
    def test_generation_tag(self):
        cnt = torch._dynamo.testing.CompileCounter()

        # guarantee that we have installed
        # the generation tagging function
        with torch._dynamo.optimize_assert(cnt):
            pass

        m1 = torch.nn.Linear(10, 10)
        prev_generation = GenerationTracker.get_generation_value(m1)
        cur_generation = prev_generation + 1

        with torch._dynamo.optimize_assert(cnt):
            m2 = torch.nn.Linear(10, 10)

        self.assertEqual(GenerationTracker.get_generation_value(m1), prev_generation)
        self.assertEqual(GenerationTracker.get_generation_value(m2), cur_generation)
        # check that newly constructed instances
        # also have the same generation (even if copied from an old instance)
        m3 = deepcopy(m1)
        self.assertEqual(GenerationTracker.get_generation_value(m3), cur_generation)

    def test_simple_torch_function(self):
        def foo(x):
            # function call, twice to test wrapping
            x = F.sigmoid(x)
            x = F.sigmoid(x)
            # method call, twice to test wrapping
            x = x.sigmoid()
            x = x.sigmoid()
            return x

        TensorProxy = temporary_tensor_subclass()
        x = torch.randn(1).as_subclass(TensorProxy)
        cnt = torch._dynamo.testing.CompileCounter()
        out1 = foo(x)
        opt_foo = torch.compile(foo, backend=cnt, fullgraph=True)
        out2 = opt_foo(x)

        self.assertEqual(cnt.op_count, 4)
        self.assertTrue(torch._dynamo.testing.same(out1, out2))

    def test_torch_function_with_closure(self):
        def run():
            def foo(x):
                # function call, twice to test wrapping
                x = F.sigmoid(x)
                x = F.sigmoid(x)
                # method call, twice to test wrapping
                x = x.sigmoid()
                x = x.sigmoid()
                return x

            counter = 0

            def function():
                nonlocal counter
                # for now, only support reads from closure cells
                # TODO(future PR): support writes as well
                counter + 1

            TensorProxy = temporary_tensor_subclass(function)
            x = torch.randn(1).as_subclass(TensorProxy)
            x = torch.randn(1)
            cnt = torch._dynamo.testing.CompileCounter()
            out1 = foo(x)
            opt_foo = torch.compile(foo, backend=cnt, fullgraph=True)
            out2 = opt_foo(x)

            self.assertEqual(cnt.op_count, 4)
            self.assertTrue(torch._dynamo.testing.same(out1, out2))

        run()

    def test_torch_mangled_class_name(self):
        original = TensorWithTFOverrideVariable.global_mangled_class_name
        results = []

        def instrumented(self, tx):
            result = original(self, tx)
            results.append(result)
            return result

        TensorWithTFOverrideVariable.global_mangled_class_name = instrumented

        def one_break(x):
            x = F.sigmoid(x)
            print()  # force break
            x = x.sigmoid()
            return x

        try:
            TensorProxy = temporary_tensor_subclass()
            x = torch.randn(1).as_subclass(TensorProxy)
            x1 = one_break(x)

            cnt = torch._dynamo.testing.CompileCounter()
            opt_one_break = torch.compile(one_break, backend=cnt)
            x2 = opt_one_break(x)

            self.assertTrue(torch._dynamo.testing.same(x1, x2))
            self.assertEqual(cnt.frame_count, 2)
            self.assertEqual(cnt.op_count, 2)

            compile_ids = set()
            for r in results:
                # A mangled classname looks like __subclass_TensorProxy_94524181138240_c0
                # where the last segment contains the compile_id.
                prefix = "__subclass_TensorProxy_"
                before, sep, after = r.partition(prefix)
                self.assertEqual(before, "")
                self.assertEqual(sep, prefix)

                class_type_id, compile_id = after.split("_")
                self.assertTrue(class_type_id.isnumeric())
                self.assertTrue(compile_id.startswith("c"))

                cid = compile_id[1:]
                self.assertTrue(cid.isnumeric())
                compile_ids.add(cid)

            self.assertEqual(len(compile_ids), 3)

        finally:
            TensorWithTFOverrideVariable.global_mangled_class_name = original

    def test_nn_moduledict_contains(self):
        class M(torch.nn.Module):
            def __init__(self, module_dict):
                super().__init__()
                self.module_dict = module_dict

            def forward(self, x):
                if "foo" in self.module_dict:
                    x = torch.mul(x, 1.0)
                x = torch.add(x, 1.0)
                return x

        module_dict = torch.nn.ModuleDict({"foo": torch.nn.Conv2d(1, 1, 1)})
        m = M(module_dict)
        data = torch.randn(1)
        out1 = m(data)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_m = torch._dynamo.optimize(cnt, nopython=True)(m)
        out2 = opt_m(data)
        self.assertEqual(cnt.op_count, 2)
        self.assertTrue(torch._dynamo.testing.same(out1, out2))

        module_dict = torch.nn.ModuleDict({"bar": torch.nn.Conv2d(1, 1, 1)})
        m = M(module_dict)
        data = torch.randn(1)
        out1 = m(data)
        cnt = torch._dynamo.testing.CompileCounter()
        torch._dynamo.reset()
        opt_m = torch._dynamo.optimize(cnt, nopython=True)(m)
        out2 = opt_m(data)

        self.assertEqual(cnt.op_count, 1)
        self.assertTrue(torch._dynamo.testing.same(out1, out2))

    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
    @expectedFailureDynamic
    def test_lazy_module1(self):
        input_shape = (16, 3, 6, 7, 8)

        cnt = torch._dynamo.testing.CompileCounter()
        module = LazyModule()

        def test_static_module():
            input = torch.ones(*input_shape)
            module(input)

        # test no graph break
        opt_test_static_module = torch.compile(
            test_static_module, backend=cnt, fullgraph=True
        )
        opt_test_static_module()

        self.assertTrue(
            isinstance(module, MaterializedModule),
            "Module should be transformed to an instance of MaterializedModule.",
        )
        self.assertEqual(module.param.shape, input_shape)

        # test when mapped to UnspecializedNNModule
        module = LazyModule()

        def test_unspecialized():
            nonlocal module
            module = LazyModule()
            input = torch.ones(*input_shape)
            module(input)

        opt_test_unspecialized = torch.compile(test_unspecialized, backend=cnt)
        opt_test_unspecialized()

        self.assertTrue(
            isinstance(module, MaterializedModule),
            "Module should be transformed to an instance of MaterializedModule.",
        )
        self.assertEqual(module.param.shape, input_shape)

        # test with a static module in torch.*
        module = torch.nn.modules.LazyBatchNorm3d(
            affine=False, track_running_stats=False
        )

        cnt = torch._dynamo.testing.CompileCounter()

        torch._dynamo.reset()

        def test_torch_static():
            input = torch.ones(*input_shape)
            return module(input)  # fully materialized

        # test no graph break
        opt_test_torch_static = torch.compile(
            test_torch_static, backend=cnt, fullgraph=True
        )
        opt_test_torch_static()
        out = opt_test_torch_static()

        self.assertTrue(same(out, module(torch.ones(*input_shape))))

        self.assertTrue(
            isinstance(module, torch.nn.modules.batchnorm.BatchNorm3d),
            "Module should be transformed to an instance of BatchNorm3d.",
        )
        self.assertEqual(cnt.frame_count, 1, "No guards should have triggered.")

    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
    @expectedFailureDynamic
    def test_lazy_module2(self):
        # Test FX graph 'call_module' works well if argument is lazy module
        m = LazyMLP()
        x = torch.rand([10, 10])
        opt_m = torch.compile(m, backend="eager", fullgraph=True)
        # We should run compile mode firstly, otherwise the module
        # would be initialized when running eager mode.
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))

    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
    @expectedFailureDynamic
    def test_lazy_module4(self):
        m = LazyMLP()
        x = torch.rand([10, 10])
        cnt = torch._dynamo.testing.CompileCounter()
        opt_m = torch.compile(m, backend=cnt, fullgraph=True)
        # first iteration
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))
        # input shape changed and second iteration
        x = torch.rand([20, 20])
        try:
            opt_m(x)
        except RuntimeError:
            self.assertIn("must have same reduction dim", traceback.format_exc())

    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
    @expectedFailureDynamic
    def test_lazy_module5(self):
        # Test lazy module works well with list/tuple input
        m = LazyModuleWithListInput()
        x = [torch.rand([5, 5])] * 3 + [None]
        opt_m = torch.compile(m, backend="eager", fullgraph=True)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))

    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
    @expectedFailureDynamic
    def test_lazy_module6(self):
        # Test new lazy submodule in lazy module's initialize_parameters
        m = LazyModuleWithLazySubmodule()
        x = [torch.rand([5, 5])] * 3
        opt_m = torch.compile(m, backend="eager", fullgraph=True)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))

    # RuntimeError: SymIntArrayRef expected to contain only concrete integers
    @expectedFailureDynamic
    def test_lazy_module7(self):
        # Test lazy module works well with namedtuple/dict input
        m = LazyModuleWithNamedTupleInput()
        x = MyInput(
            x={"a": [torch.rand([5, 5])] * 3, "b": torch.rand([5, 5])},
            y=torch.rand([5, 5]),
        )
        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_lazy_module_no_cls_to_become(self):
        # make sure super() works in the case where cls_to_become is None
        m = LazyChildModuleNoClsToBecome()
        x = torch.rand(2, 2)
        opt_m = torch.compile(m, backend="eager", fullgraph=True)
        res = opt_m(x)
        ref = m(x)
        self.assertTrue(torch.allclose(ref, res))

    def test_lazy_module_kwargs(self):
        m = LazyModuleKwArgs()
        x = [torch.rand([5, 5])] * 3
        y = [torch.rand([5, 5])] * 2
        opt_m = torch.compile(backend="eager", fullgraph=True)(m)
        exp_res = m(x, y)
        self.assertTrue(torch.allclose(exp_res, opt_m(x, y)))

    def test_lazy_module_bad_params(self):
        m = LazyModuleBadInferParams()
        x = [torch.rand([5, 5])] * 3
        y = [torch.rand([5, 5])] * 2
        # Note that this raises from within dynamo code, with no exception handling.
        with self.assertRaises(AttributeError) as cm:
            opt_m = torch.compile(backend="eager")(m)
            exp_res = opt_m(x, y)

    def test_lazy_module_bad_params_call_function(self):
        class holder:
            x = LazyModuleBadInferParams()

            def apply(self, x, y):
                self.x(x, y)

        def m(x, y):
            h = holder()
            return h.apply(x, y)

        x = [torch.rand([5, 5])] * 3
        y = [torch.rand([5, 5])] * 2
        opt_m = torch.compile(backend="eager")(m)
        with self.assertRaises(AttributeError):
            exp_res = opt_m
```



## High-Level Overview


This Python file contains 149 class(es) and 432 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BasicModule`, `FnMember`, `FnMemberCmp`, `SubmoduleExample`, `IsTrainingCheck`, `IsEvalCheck`, `ModuleMethodCall`, `UnsupportedMethodCall`, `UnsupportedModule`, `UnsupportedModuleCall`, `ModuleWithStaticForward`, `ModuleCallModuleWithStaticForward`, `ModuleStaticMethodCall`, `ModuleClassMethodCall`, `ModuleProperty`, `NestedModuleList`, `ConstLoop`, `ViaModuleCall`, `IsNoneLayer`, `LayerList`

**Functions defined**: `update_global`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `__init__`, `call_and_scale`, `forward`, `__init__`, `call_and_scale`, `forward`, `__init__`, `forward`

**Key imports**: collections, copy, itertools, os, tempfile, traceback, types, unittest, deepcopy, partial


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `collections`
- `copy`
- `itertools`
- `os`
- `tempfile`
- `traceback`
- `types`
- `unittest`
- `functools`: partial
- `typing`: NamedTuple
- `unittest.mock`: patch
- `torch`
- `torch._dynamo.test_case`
- `torch._dynamo.testing`
- `torch.nn.functional as F`
- `torch._dynamo.debug_utils`: same_two_models
- `torch._dynamo.eval_frame`: unsupported
- `torch._dynamo.mutation_guard`: GenerationTracker
- `torch._dynamo.utils`: ifdynstaticdefault
- `torch._dynamo.variables.torch_function`: TensorWithTFOverrideVariable
- `torch.nn.modules.lazy`: LazyModuleMixin
- `torch.nn.parameter`: Parameter, UninitializedParameter
- `torch.testing._internal.common_device_type`: instantiate_device_type_tests
- `torch.testing._internal.common_utils`: skipIfHpu
- `.`: test_functions
- `test_functions`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
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
python test/dynamo/test_modules.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/dynamo`):

- [`test_guard_serialization.py_docs.md`](./test_guard_serialization.py_docs.md)
- [`test_subgraphs.py_docs.md`](./test_subgraphs.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_unspec.py_docs.md`](./test_unspec.py_docs.md)
- [`test_trace_rules.py_docs.md`](./test_trace_rules.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_pre_dispatch.py_docs.md`](./test_pre_dispatch.py_docs.md)
- [`test_autograd_function.py_docs.md`](./test_autograd_function.py_docs.md)
- [`test_optimizers.py_docs.md`](./test_optimizers.py_docs.md)
- [`test_callback.py_docs.md`](./test_callback.py_docs.md)


## Cross-References

- **File Documentation**: `test_modules.py_docs.md`
- **Keyword Index**: `test_modules.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
