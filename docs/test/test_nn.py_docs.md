# Documentation: test_nn.py

## File Metadata
- **Path**: `test/test_nn.py`
- **Size**: 648914 bytes
- **Lines**: 13828
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# Owner(s): ["module: nn"]
# ruff: noqa: F841

import contextlib
import math
import random
import unittest
import io
import itertools
import warnings
import os
import pickle
import re
from copy import deepcopy
from itertools import product
from functools import partial
from collections import OrderedDict
from unittest import SkipTest

import torch
from torch import inf, nan
import torch.autograd.forward_ad as fwAD
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils import clip_grad_norm_, clip_grad_value_, clip_grads_with_norm_, get_total_norm
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.utils.fusion import fuse_conv_bn_weights
from torch.nn.utils.fusion import fuse_linear_bn_weights
from torch.nn import Buffer, Parameter
from torch.nn.parallel._functions import Broadcast
from torch.testing._internal.common_dtype import integral_types, get_all_math_dtypes, floating_types
from torch.testing._internal.common_utils import dtype_name, freeze_rng_state, run_tests, TestCase, \
    skipIfNoLapack, skipIfRocm, MI300_ARCH, skipIfRocmArch, \
    TEST_NUMPY, TEST_SCIPY, TEST_WITH_CROSSREF, TEST_WITH_ROCM, \
    download_file, get_function_arglist, load_tests, skipIfMPS, \
    IS_PPC, \
    parametrize as parametrize_test, subtest, instantiate_parametrized_tests, \
    skipIfTorchDynamo, gcIfJetson, set_default_dtype
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU, TEST_CUDNN, \
    _get_torch_rocm_version
from torch.testing._internal.common_nn import NNTestCase, NewModuleTest, CriterionTest, \
    module_tests, criterion_tests, loss_reference_fns, _create_basic_net, \
    ctcloss_reference, get_new_module_tests, single_batch_reference_fn, _test_bfloat16_ops, _test_module_empty_input
from torch.testing._internal.common_device_type import dtypesIfMPS, instantiate_device_type_tests, dtypes, \
    dtypesIfCUDA, precisionOverride, onlyCUDA, onlyCPU, \
    skipCUDAIfRocm, skipCUDAIf, skipCUDAIfNotRocm, \
    onlyNativeDeviceTypes, deviceCountAtLeast, largeTensorTest, expectedFailureMeta, expectedFailureMPS, \
    skipMeta, get_all_device_types

from hypothesis import given
import torch.testing._internal.hypothesis_utils as hu
from torch.testing._internal.common_utils import _assertGradAndGradgradChecks, gradcheck, gradgradcheck, \
    GRADCHECK_NONDET_TOL
from torch.testing._internal.common_utils import dtype2prec_DONTUSE
from torch.testing._internal.common_cuda import tf32_on_and_off, tf32_off, tf32_on
from torch.types import _TensorOrTensors
from torch.testing._internal.common_mkldnn import reduced_f32_on_and_off

AMPERE_OR_ROCM = TEST_WITH_ROCM or torch.cuda.is_tf32_supported()

if TEST_WITH_ROCM:
    os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC"] = "1"
    os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC_BATCHNORM"] = "1"

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

if TEST_SCIPY:
    import scipy.signal
    import scipy.ndimage

if TEST_NUMPY:
    import numpy as np


# WARNING: If you add a new top-level test case to this file, you MUST
# update test/run_test.py to list it, otherwise it will NOT be run in
# CI.

class TestNN(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def _forward(self, module, input: _TensorOrTensors):
        with freeze_rng_state():
            if isinstance(input, tuple):
                return module(*input)
            else:
                return module(input)

    def _backward(self, module, input: _TensorOrTensors, output, grad_output, create_graph=False):
        output.backward(grad_output, retain_graph=True, create_graph=create_graph)
        if isinstance(input, tuple):
            return tuple(i.grad.data if i.grad is not None else None for i in input)
        else:
            return input.grad.data if input.grad is not None else None

    def _forward_criterion(self, criterion, input, target, extra_args=None):
        if extra_args is None:
            extra_args = ()
        if isinstance(input, tuple):
            args = input + (target,) + extra_args
            output = criterion(*args)
        else:
            output = criterion(input, target, *extra_args)
        return output

    def _backward_criterion(self, criterion, input, output, target, gradOutput=None, extra_args=None):
        if extra_args is None:
            extra_args = ()
        input_tuple = input if isinstance(input, tuple) else (input,)
        output_tuple = output if isinstance(output, tuple) else (output,)
        for i in input_tuple:
            if i.grad is not None:
                i.grad.data.zero_()
        args = input_tuple + (target,) + extra_args
        if gradOutput is None:
            gradOutput = torch.ones(())
        criterion(*args).backward(gradOutput.to(output_tuple[0]))
        if isinstance(input, tuple):
            return tuple(i.grad.data for i in input)
        else:
            return input.grad.data

    def _zero_grad_parameters(self, module):
        for p in module.parameters():
            if p.grad is not None:
                with torch.no_grad():
                    p.grad.zero_()
                p.grad.detach_()

    def _get_parameters(self, module):
        params = []
        d_params = []
        for p in module.parameters():
            params.append(p)
            d_params.append(p.grad)
        return params, d_params

    def test_parse_to(self):
        # Test for buggy use of THPMemoryFormat_New
        self.assertEqual(
            repr(torch._C._nn._parse_to(memory_format=torch.contiguous_format)[3]),
            "torch.contiguous_format"
        )

    def test_requires_grad_(self):
        m = _create_basic_net()[-1]
        assert len(list(m.buffers())) > 0, 'invalid test'
        assert all(not b.requires_grad for b in m.buffers()) > 0, 'invalid test'
        assert len(list(m.parameters())) > 0, 'invalid test'
        assert all(p.requires_grad for p in m.parameters()) > 0, 'invalid test'
        for requires_grad in (False, True):
            self.assertIs(m.requires_grad_(requires_grad), m)
            for p in m.parameters():
                self.assertEqual(p.requires_grad, requires_grad)
            for b in m.buffers():
                self.assertFalse(b.requires_grad)

    def test_module_backcompat(self):
        from torch.serialization import SourceChangeWarning
        path = download_file('https://download.pytorch.org/test_data/linear.pt')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SourceChangeWarning)
            # weights_only=False as this is legacy code that saves the model
            m = torch.load(path, weights_only=False)
        input = torch.randn(2, 3, dtype=torch.float)
        self.assertEqual(m(input).size(), (2, 5))

    def test_module_super_init(self):
        class MyMixin:
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.mixin_init = True

        class MyModuleWithMixinBefore(MyMixin, nn.Module):
            pass

        class MyModuleWithMixinAfter(nn.Module, MyMixin):
            pass

        self.assertTrue(hasattr(MyModuleWithMixinBefore(), 'mixin_init'))
        self.assertFalse(hasattr(MyModuleWithMixinAfter(), 'mixin_init'))

        nn.Module.call_super_init = True
        self.assertTrue(hasattr(MyModuleWithMixinBefore(), 'mixin_init'))
        self.assertTrue(hasattr(MyModuleWithMixinAfter(), 'mixin_init'))
        nn.Module.call_super_init = False

        MyModuleWithMixinBefore.call_super_init = True
        MyModuleWithMixinAfter.call_super_init = True
        self.assertTrue(hasattr(MyModuleWithMixinBefore(), 'mixin_init'))
        self.assertTrue(hasattr(MyModuleWithMixinAfter(), 'mixin_init'))
        MyModuleWithMixinBefore.call_super_init = False
        MyModuleWithMixinAfter.call_super_init = False

    def test_share_memory(self):
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.p = nn.Parameter(torch.eye(5))
                self.par = nn.ParameterList()
                self.par.append(nn.Parameter(torch.randn(10)))

            def forward(self, inp):
                # NB: dead code
                return inp.clone()

        net = Net()
        for p in net.parameters():
            self.assertFalse(p.storage().is_shared())
        for b in net.buffers():
            self.assertFalse(b.storage().is_shared())
        net.share_memory()
        for p in net.parameters():
            self.assertTrue(p.storage().is_shared())
        for b in net.buffers():
            self.assertTrue(b.storage().is_shared())

    def test_to(self):
        m = nn.Linear(3, 5)
        self.assertIs(m, m.to('cpu'))
        self.assertIs(m, m.to('cpu', dtype=torch.float32))
        self.assertEqual(m.double(), m.to(torch.float64))
        self.assertRaises(RuntimeError, lambda: m.to('cpu', copy=True))

        if torch.cuda.is_available():
            for cuda in ['cuda', 'cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1']:
                m2 = m.cuda(device=cuda)
                self.assertIs(m2, m2.to(cuda))
                self.assertEqual(m, m2.to('cpu'))
                self.assertEqual(m2, m.to(cuda))
                self.assertIs(m2, m2.to(dtype=torch.float32))
                self.assertEqual(m2.double(), m2.to(dtype=torch.float64))

    def test_zero_grad(self):
        i = torch.randn(2, 5, requires_grad=True)
        module = nn.Linear(5, 5)
        for p in module.parameters():
            p.requires_grad = False
        module.zero_grad()

        module.weight.requires_grad = True
        module.zero_grad()
        self.assertIsNone(module.weight.grad)  # uninitialized grad

        module(i).sum().backward()
        self.assertIsNotNone(module.weight.grad)
        self.assertGreater(module.weight.grad.data.abs().sum(), 0)
        module.zero_grad()
        self.assertIsNone(module.weight.grad)

        module.bias.requires_grad = True
        module.zero_grad()
        self.assertIsNone(module.weight.grad)
        self.assertIsNone(module.bias.grad)
        module(i).sum().backward()
        self.assertIsNotNone(module.weight.grad)
        self.assertIsNotNone(module.bias.grad)
        self.assertGreater(module.weight.grad.data.abs().sum(), 0)
        self.assertGreater(module.bias.grad.data.abs().sum(), 0)
        module.zero_grad(set_to_none=False)   # Force set to zeros.
        self.assertEqual(module.weight.grad.data, module.weight.data.clone().zero_())
        self.assertEqual(module.bias.grad.data, module.bias.data.clone().zero_())

        module.zero_grad()
        self.assertIsNone(module.weight.grad)
        self.assertIsNone(module.bias.grad)

    def test_no_grad(self):
        for dtype in [torch.bfloat16, torch.float, torch.double]:
            module = nn.Conv2d(2, 5, kernel_size=3, padding=1).to(dtype)
            input = torch.randn(1, 2, 10, 10).to(dtype)
            x = input
            y = input.clone()

            output = module(x)
            self.assertTrue(output.requires_grad)
            output.backward(torch.ones(1, 5, 10, 10))

            with torch.no_grad():
                output2 = module(y)
                self.assertFalse(output2.requires_grad)
                self.assertRaises(RuntimeError, lambda: output2.backward(torch.ones(1, 5, 10, 10)))

    def test_parameters_and_named_parameters(self):
        def names(named_parameters):
            return [k for k, _ in named_parameters]

        l, n, s = _create_basic_net()

        self.assertEqual(len(list(l.parameters())), 1)
        self.assertEqual(
            names(l.named_parameters()),
            ['layer_dummy_param'])

        self.assertEqual(len(list(n.parameters())), 2)
        self.assertEqual(
            names(n.named_parameters()),
            ['dummy_param', 'l1.layer_dummy_param'])

        self.assertEqual(len(list(n.parameters(recurse=False))), 1)
        self.assertEqual(
            names(n.named_parameters(recurse=False)),
            ['dummy_param'])

        self.assertEqual(len(list(s.parameters())), 2)
        self.assertEqual(
            names(s.named_parameters()),
            ['0.dummy_param', '0.l1.layer_dummy_param'])

    def test_named_parameters_remove_duplicate(self):
        def names(named_parameters):
            return [k for k, _ in named_parameters]

        class M1(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param1 = nn.Parameter(torch.empty(3, 3))
                self.param2 = self.param1

        m1 = M1()
        self.assertEqual(names(m1.named_parameters()),
                         ["param1"])
        self.assertEqual(names(m1.named_parameters(remove_duplicate=False)),
                         ["param1", "param2"])

        class M2(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mod1 = nn.Linear(3, 4, bias=False)
                self.mod2 = self.mod1

        m2 = M2()
        self.assertEqual(names(m2.named_parameters()),
                         ["mod1.weight"])
        self.assertEqual(names(m2.named_parameters(remove_duplicate=False)),
                         ["mod1.weight", "mod2.weight"])

    def test_buffers_and_named_buffers(self):
        def names(named_buffers):
            return [k for k, _ in named_buffers]

        l, n, s = _create_basic_net()

        self.assertEqual(len(list(l.buffers())), 1)
        self.assertEqual(
            names(l.named_buffers()),
            ['layer_dummy_buf'])

        self.assertEqual(len(list(n.buffers())), 2)
        self.assertEqual(
            names(n.named_buffers()),
            ['dummy_buf', 'l1.layer_dummy_buf'])

        self.assertEqual(len(list(n.buffers(recurse=False))), 1)
        self.assertEqual(
            names(n.named_buffers(recurse=False)),
            ['dummy_buf'])

        self.assertEqual(len(list(s.buffers())), 2)
        self.assertEqual(
            names(s.named_buffers()),
            ['0.dummy_buf', '0.l1.layer_dummy_buf'])

        # test remove_duplicate
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer1 = Buffer(torch.empty(3, 5))
                self.buffer2 = self.buffer1

        m = M()
        self.assertEqual(names(m.named_buffers()),
                         ["buffer1"])
        self.assertEqual(names(m.named_buffers(remove_duplicate=False)),
                         ["buffer1", "buffer2"])

    def test_buffer_bad_module_subclass(self):
        class MyBadModule(nn.Linear):
            def __init__(self) -> None:
                super().__init__(2, 2)
                self.bar = Buffer(torch.rand(2, 2))

            def register_buffer(self, name, value):
                # persistent is explicitly missing!
                super().register_buffer(name, value, True)

        foo = MyBadModule()
        self.assertIsNotNone(foo.bar)

    def test_call_supports_python_dict_output(self):
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l1 = nn.Linear(10, 20)
                self.register_backward_hook(self.hook)
                self.check_backward_hook_flag = False

            def hook(self, module, grad_out, grad_in):
                self.check_backward_hook_flag = True

            def forward(self, inputs):
                return {"output": self.l1(inputs).sum()}

        net = Net()
        model_output = net(torch.randn([5, 10]))
        model_output["output"].backward()
        self.assertTrue(net.check_backward_hook_flag)

    def test_children(self):
        l1 = nn.Linear(2, 2)
        l2 = nn.Linear(2, 2)
        l3 = nn.Linear(2, 2)
        l4 = nn.Linear(2, 2)
        subnet = nn.Sequential(l3, l4)
        s = nn.Sequential(l1, l2, l1, l2, subnet)
        self.assertEqual(list(s.children()), [l1, l2, subnet])

    def test_train_errors_for_invalid_mode(self):
        class SubclassNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l1 = nn.Linear(2, 2)

            def forward(self, inputs):
                return self.l1(inputs)

        subclass_net = SubclassNet()
        sequential_net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

        error_modes = ["invalid_str", torch.device('cpu')]
        modules_to_check = [subclass_net, sequential_net]

        for error_mode, module in itertools.product(error_modes, modules_to_check):
            with self.assertRaises(ValueError):
                module.train(error_mode)

    def test_dir(self):
        linear = nn.Linear(2, 2)
        linear._test_submodule = nn.Linear(2, 2)
        linear._test_parameter = Parameter(torch.empty(2, 2))
        linear._test_buffer = Buffer(torch.empty(2, 2))
        keys = dir(linear)
        self.assertIn('_test_submodule', keys)
        self.assertIn('_test_parameter', keys)
        self.assertIn('_test_buffer', keys)

        for key in keys:
            self.assertTrue(hasattr(linear, key))

    def test_repr(self):
        # no extra information or sub-modules
        empty_sequential = nn.Sequential()
        expected_repr_empty = 'Sequential()'
        self.assertEqual(repr(empty_sequential), expected_repr_empty)

        # one liner extra information
        linear = nn.Linear(1, 1)
        expected_repr_linear = 'Linear(in_features=1, out_features=1, bias=True)'
        self.assertEqual(repr(linear), expected_repr_linear)

        # sub-modules repr
        sequential = nn.Sequential(linear)
        expected_repr_sequential = 'Sequential(\n' \
            '  (0): Linear(in_features=1, out_features=1, bias=True)\n' \
            ')'
        self.assertEqual(repr(sequential), expected_repr_sequential)

    def test_dir_digit(self):
        model = nn.Sequential(nn.Linear(2, 2))
        keys = dir(model)
        self.assertNotIn('0', keys)

    def test_named_children(self):
        l1 = nn.Linear(2, 2)
        l2 = nn.Linear(2, 2)
        l3 = nn.Linear(2, 2)
        l4 = nn.Linear(2, 2)
        subnet = nn.Sequential(l3, l4)
        s = nn.Sequential()
        with self.assertRaises(KeyError):
            s.add_module('', l1)
        with self.assertRaises(KeyError):
            s.add_module('name.with.dot', l1)
        s.add_module('layer1', l1)
        s.add_module('layer2', l2)
        s.add_module('layer3', l1)
        s.add_module('layer4', l2)
        s.add_module('subnet', subnet)
        self.assertEqual(list(s.named_children()), [('layer1', l1), ('layer2', l2), ('subnet', subnet)])

    def test_modules(self):
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l1 = l
                self.l2 = l
                self.param = torch.empty(3, 5)

        l = nn.Linear(10, 20)
        n = Net()
        s = nn.Sequential(n, n, n, n)
        self.assertEqual(list(s.modules()), [s, n, l])

    def test_named_modules(self):
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l1 = l
                self.l2 = l
                self.param = torch.empty(3, 5)
                self.block = block
        l = nn.Linear(10, 20)
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(10, 20)
        block = nn.Sequential()
        block.add_module('linear1', l1)
        block.add_module('linear2', l2)
        n = Net()
        s = nn.Sequential(n, n)
        self.assertEqual(list(s.named_modules()), [('', s), ('0', n), ('0.l1', l),
                                                   ('0.block', block), ('0.block.linear1', l1),
                                                   ('0.block.linear2', l2)])
        # test the option to not remove duplicate module instances
        self.assertEqual(list(s.named_modules(remove_duplicate=False)), [
            ('', s), ('0', n), ('0.l1', l), ('0.l2', l),
            ('0.block', block), ('0.block.linear1', l1),
            ('0.block.linear2', l2),
            ('1', n), ('1.l1', l), ('1.l2', l),
            ('1.block', block), ('1.block.linear1', l1),
            ('1.block.linear2', l2)])

    def test_register_buffer_raises_error_if_name_is_not_string(self):
        m = nn.Module()
        expected_error = 'buffer name should be a string. Got '
        with self.assertRaisesRegex(TypeError, expected_error + 'int'):
            m.register_buffer(1, torch.rand(5))
        with self.assertRaisesRegex(TypeError, expected_error + 'NoneType'):
            m.register_buffer(None, torch.rand(5))

    def test_register_buffer_raises_error_if_attr_exists(self):
        m = nn.Module()
        m.attribute_name = 5
        with self.assertRaises(KeyError):
            m.register_buffer('attribute_name', torch.rand(5))

        with self.assertRaises(KeyError):
            m.attribute_name = Buffer(torch.rand(5))

        del m.attribute_name
        m.register_parameter('attribute_name', nn.Parameter())
        with self.assertRaises(KeyError):
            m.register_buffer('attribute_name', torch.rand(5))

        del m.attribute_name
        m.add_module('attribute_name', nn.Module())
        with self.assertRaises(KeyError):
            m.register_buffer('attribute_name', torch.rand(5))

    def test_register_buffer_raises_error_if_not_tensor(self):
        m = nn.Module()
        with self.assertRaises(TypeError):
            m.register_buffer('attribute_name', 5)

    def test_register_buffer_allows_overwriting_with_same_name(self):
        m = nn.Module()
        buffer1 = torch.rand(5)
        buffer2 = buffer1 + 5
        buffer3 = None
        m.register_buffer('buffer_name', buffer1)
        self.assertEqual(m.buffer_name, buffer1)
        m.register_buffer('buffer_name', buffer2)
        self.assertEqual(m.buffer_name, buffer2)
        m.register_buffer('buffer_name', buffer3)
        self.assertEqual(m.buffer_name, buffer3)
        m.buffer_name = Buffer(buffer1)
        self.assertEqual(m.buffer_name, Buffer(buffer1))
        m.buffer_name = Buffer(buffer2)
        self.assertEqual(m.buffer_name, Buffer(buffer2))
        m.buffer_name = Buffer(buffer3)
        self.assertEqual(m.buffer_name, Buffer(buffer3))

    def test_register_buffer_allows_tensor_like_object(self):
        class TensorLike:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                raise NotImplementedError(f"TensorLike.__torch_function__: {func}")

        buffer1 = TensorLike()
        buffer2 = TensorLike()
        m = nn.Module()
        m.register_buffer('buffer_name', buffer1)
        self.assertEqual(m.buffer_name, buffer1)
        self.assertEqual(m.get_buffer('buffer_name'), buffer1)
        m.buffer_name = buffer2
        self.assertEqual(m.buffer_name, buffer2)
        self.assertEqual(m.get_buffer('buffer_name'), buffer2)

    def test_get_buffer(self):
        m = nn.Module()
        buffer1 = torch.randn(2, 3)
        buffer2 = torch.randn(4, 5)
        m.foo = Buffer(buffer1)
        m.register_buffer('bar', buffer2)
        self.assertEqual(buffer1, m.get_buffer('foo'))
        self.assertEqual(buffer2, m.get_buffer('bar'))

    def test_get_buffer_from_submodules(self):
        class MyModule(nn.Module):
            def __init__(self, foo, bar):
                super().__init__()
                self.sub = Sub(foo, bar)

        class Sub(nn.Module):
            def __init__(self, foo, bar):
                super().__init__()
                self.foo = Buffer(foo)
                self.subsub = SubSub(bar)

        class SubSub(nn.Module):
            def __init__(self, bar):
                super().__init__()
                self.bar = Buffer(bar)

        foo = torch.randn(2, 3)
        bar = torch.randn(4, 5)
        m = MyModule(foo, bar)
        self.assertEqual(foo, m.get_buffer('sub.foo'))
        self.assertEqual(bar, m.get_buffer('sub.subsub.bar'))

    def test_buffer_not_persistent(self):
        m = nn.Module()
        m.buf = nn.Buffer(torch.rand(5), persistent=False)
        self.assertTrue(len(list(m.buffers())) == 1)
        self.assertTrue(len(m.state_dict()) == 0)

    def test_buffer_not_persistent_del(self):
        m = nn.Module()
        m.buf = nn.Buffer(torch.rand(5), persistent=False)
        del m.buf
        self.assertTrue(len(list(m.buffers())) == 0)

    def test_buffer_not_persistent_overwrite(self):
        m = nn.Module()
        m.buf = nn.Buffer(torch.rand(5), persistent=False)
        m.buf = nn.Buffer(torch.rand(5))

        # can we overwrite a non-persistent buffer with a persistent one?
        self.assertTrue(len(list(m.buffers())) == 1)
        self.assertTrue(len(m.state_dict()) == 1)

        # can we overwrite a persistent buffer with a non-persistent one?
        m.buf = nn.Buffer(torch.rand(5), persistent=False)
        self.assertTrue(len(list(m.buffers())) == 1)
        self.assertTrue(len(m.state_dict()) == 0)

    def test_buffer_not_persistent_assign(self):
        m = nn.Module()
        m.buf = nn.Buffer(torch.rand(5), persistent=False)
        self.assertTrue(len(list(m.buffers())) == 1)
        self.assertTrue(len(m.state_dict()) == 0)

        # Assigning None removes the buffer but if we then assign a new Tensor
        # to the same property, it should still be marked as a buffer.
        m.buf = None
        self.assertTrue(len(list(m.buffers())) == 0)
        self.assertTrue(len(m.state_dict()) == 0)
        m.buf = torch.rand(5)
        self.assertTrue(len(list(m.buffers())) == 1)
        self.assertTrue(len(m.state_dict()) == 0)

        # Assigning a Parameter removes the buffer.
        m.buf = nn.Parameter(torch.rand(5))
        self.assertTrue(len(list(m.buffers())) == 0)
        self.assertTrue(len(m.state_dict()) == 1)

    def test_buffer_not_persistent_load(self):
        m = nn.Module()
        m.buf = nn.Buffer(torch.rand(5), persistent=False)
        m.load_state_dict({})

    def test_register_parameter_raises_error_if_name_is_not_string(self):
        m = nn.Module()
        expected_error = 'parameter name should be a string. Got '
        with self.assertRaisesRegex(TypeError, expected_error + 'int'):
            m.register_parameter(1, nn.Parameter())
        with self.assertRaisesRegex(TypeError, expected_error + 'NoneType'):
            m.register_parameter(None, nn.Parameter())

    def test_register_parameter_raises_error_if_attr_exists(self):
        m = nn.Module()
        m.attribute_name = 5
        with self.assertRaises(KeyError):
            m.register_parameter('attribute_name', nn.Parameter())

        del m.attribute_name
        m.register_buffer('attribute_name', torch.rand(5))
        with self.assertRaises(KeyError):
            m.register_parameter('attribute_name', nn.Parameter())

        del m.attribute_name
        m.attribute_name = Buffer(torch.rand(5))
        with self.assertRaises(KeyError):
            m.register_parameter('attribute_name', nn.Parameter())

        del m.attribute_name
        m.add_module('attribute_name', nn.Module())
        with self.assertRaises(KeyError):
            m.register_parameter('attribute_name', nn.Parameter())

    def test_register_parameter_allows_overwriting_with_same_name(self):
        m = nn.Module()
        param1 = nn.Parameter(torch.rand(5))
        param2 = nn.Parameter(param1.data + 5)
        param3 = None
        m.register_parameter('param_name', param1)
        self.assertEqual(m.param_name, param1)
        m.register_parameter('param_name', param2)
        self.assertEqual(m.param_name, param2)
        m.register_parameter('param_name', param3)
        self.assertEqual(m.param_name, param3)

    def test_add_module_raises_error_if_attr_exists(self):
        methods_to_test = ['add_module', 'register_module']
        for fn in methods_to_test:
            m = nn.Module()
            m.attribute_name = 5
            with self.assertRaises(KeyError):
                getattr(m, fn)('attribute_name', nn.Module())

            del m.attribute_name
            m.register_buffer('attribute_name', torch.rand(5))
            with self.assertRaises(KeyError):
                getattr(m, fn)('attribute_name', nn.Module())

            del m.attribute_name
            m.register_parameter('attribute_name', nn.Parameter())
            with self.assertRaises(KeyError):
                getattr(m, fn)('attribute_name', nn.Module())

    @unittest.expectedFailure
    def test_getattr_with_property(self):
        class Model(nn.Module):
            @property
            def some_property(self):
                return self.something_that_doesnt_exist

        model = Model()

        with self.assertRaisesRegex(
                AttributeError,
                r"'Model' object has no attribute 'something_that_doesnt_exist'"):
            model.some_property

    def test_Sequential_getitem(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3, l4)
        self.assertIs(n[0], l1)
        self.assertIs(n[1], l2)
        self.assertIs(n[2], l3)
        self.assertIs(n[3], l4)
        self.assertIs(n[torch.tensor(3, dtype=torch.int64)], l4)
        self.assertEqual(n[1:], nn.Sequential(l2, l3, l4))
        self.assertEqual(n[3:], nn.Sequential(l4))
        self.assertEqual(n[:-1], nn.Sequential(l1, l2, l3))
        self.assertEqual(n[:-3], nn.Sequential(l1))
        self.assertEqual(n[::-1], nn.Sequential(l4, l3, l2, l1))

    def test_Sequential_setitem(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3)
        n[0] = l4
        n[-1] = l4
        n[torch.tensor(1, dtype=torch.int16)] = l1
        self.assertIs(n[0], l4)
        self.assertIs(n[1], l1)
        self.assertIs(n[2], l4)

    def test_Sequential_setitem_named(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(OrderedDict([
            ('linear1', l1),
            ('linear2', l2),
            ('linear3', l3),
        ]))

        n[0] = l4
        n[-1] = l4
        self.assertEqual(n.linear1, l4)
        self.assertEqual(n.linear3, l4)

    def test_Sequential_delitem(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3, l4)
        del n[-1]
        self.assertEqual(n, nn.Sequential(l1, l2, l3))
        del n[1::2]
        self.assertEqual(n, nn.Sequential(l1, l3))

    def test_Sequential_add(self):
        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 4)
        l4 = nn.Linear(4, 5)
        n = nn.Sequential(l1, l2)
        other = nn.Sequential(l3, l4)
        self.assertEqual(n + other, nn.Sequential(l1, l2, l3, l4))

    def test_Sequential_iadd(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3)
        n2 = nn.Sequential(l4)
        n += n2
        n2 += n
        self.assertEqual(n, nn.Sequential(l1, l2, l3, l4))
        self.assertEqual(n2, nn.Sequential(l4, l1, l2, l3, l4))

    def test_Sequential_mul(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3, l4)
        n2 = n * 2
        self.assertEqual(n2, nn.Sequential(l1, l2, l3, l4, l1, l2, l3, l4))

    def test_Sequential_rmul(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3, l4)
        n2 = 2 * n
        self.assertEqual(n2, nn.Sequential(l1, l2, l3, l4, l1, l2, l3, l4))

    def test_Sequential_imul(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3, l4)
        n *= 2
        self.assertEqual(n, nn.Sequential(l1, l2, l3, l4, l1, l2, l3, l4))
        n *= 2
        self.assertEqual(
            n,
            nn.Sequential(l1, l2, l3, l4, l1, l2, l3, l4, l1, l2, l3, l4, l1, l2, l3, l4)
        )

    def test_Sequential_append(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n = nn.Sequential(l1, l2, l3)
        n2 = n.append(l4)
        self.assertEqual(n, nn.Sequential(l1, l2, l3, l4))
        self.assertEqual(n2, nn.Sequential(l1, l2, l3, l4))
        self.assertEqual(nn.Sequential(l1).append(l2).append(l4), nn.Sequential(l1, l2, l4))

    def test_Sequential_pop(self):
        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 4)
        l4 = nn.Linear(4, 5)
        n1 = nn.Sequential(l1, l2, l3, l4)
        self.assertEqual(l4, n1.pop(3))
        n2 = nn.Sequential(l1, l2, l3)
        self.assertEqual(n1, n2)
        # check order of the index
        for k, mod in zip(range(len(n1)), n1):
            self.assertIs(n1[k], mod)

    def test_Sequential_insert(self):
        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 4)

        n1 = nn.Sequential(l1, l2, l3)
        module_1 = nn.Linear(4, 5)
        n2 = nn.Sequential(l1, module_1, l2, l3)
        self.assertEqual(n1.insert(1, module_1), n2)

        # test for negative support
        n3 = nn.Sequential(l1, l2, l3)
        module_2 = nn.Linear(5, 6)
        n4 = nn.Sequential(l1, module_2, l2, l3)
        self.assertEqual(n3.insert(-2, module_2), n4)

    def test_Sequential_insert_fail_case(self):
        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 4)

        module = nn.Linear(5, 6)

        # test for error case
        n1 = nn.Sequential(l1, l2, l3)
        with self.assertRaises(IndexError):
            n1.insert(-5, module)

        with self.assertRaises(AssertionError):
            n1.insert(1, [nn.Linear(6, 7)])

    def test_Sequential_extend(self):
        l1 = nn.Linear(10, 20)
        l2 = nn.Linear(20, 30)
        l3 = nn.Linear(30, 40)
        l4 = nn.Linear(40, 50)
        n1 = nn.Sequential(l1, l2)
        n2 = nn.Sequential(l3, l4)
        n3 = nn.Sequential(l1, l2)
        for l in n2:
            n1.append(l)
        n3.extend(n2)
        self.assertEqual(n3, n1)

    def test_ModuleList(self):
        modules = [nn.ReLU(), nn.Linear(5, 5)]
        module_list = nn.ModuleList(modules)

        def check():
            self.assertEqual(len(module_list), len(modules))
            for m1, m2 in zip(modules, module_list):
                self.assertIs(m1, m2)
            for m1, m2 in zip(modules, module_list.children()):
                self.assertIs(m1, m2)
            for i in range(len(modules)):
                self.assertIs(module_list[i], modules[i])

        check()
        modules += [nn.Conv2d(3, 4, 3)]
        module_list += [modules[-1]]
        check()
        modules = modules + [nn.Conv2d(3, 4, 3, bias=False), nn.GELU()]
        module_list = module_list + nn.ModuleList(modules[-2:])
        check()
        modules.insert(1, nn.Linear(3, 2))
        module_list.insert(1, modules[1])
        check()
        modules.append(nn.Tanh())
        module_list.append(modules[-1])
        check()
        next_modules = [nn.Linear(5, 5), nn.Sigmoid()]
        modules.extend(next_modules)
        module_list.extend(next_modules)
        check()
        modules[2] = nn.Conv2d(5, 3, 2)
        module_list[2] = modules[2]
        check()
        modules[-1] = nn.Conv2d(5, 2, 1)
        module_list[-1] = modules[-1]
        check()
        idx = torch.tensor(2, dtype=torch.int32)
        modules[2] = nn.Conv2d(5, 3, 2)
        module_list[idx] = modules[2]
        self.assertIs(module_list[idx], modules[2])
        check()
        self.assertEqual(module_list[1:], nn.ModuleList(modules[1:]))
        self.assertEqual(module_list[3:], nn.ModuleList(modules[3:]))
        self.assertEqual(module_list[:-1], nn.ModuleList(modules[:-1]))
        self.assertEqual(module_list[:-3], nn.ModuleList(modules[:-3]))
        self.assertEqual(module_list[::-1], nn.ModuleList(modules[::-1]))
        del module_list[-1]
        self.assertEqual(module_list, nn.ModuleList(modules[:-1]))
        del module_list[1::2]
        self.assertEqual(module_list, nn.ModuleList(modules[:-1][0::2]))

        with self.assertRaises(TypeError):
            module_list += nn.ReLU()
        with self.assertRaises(TypeError):
            module_list.extend(nn.ReLU())

        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 2)
        l4 = nn.Linear(2, 3)
        subnet = nn.Sequential(l3, l4)
        s = nn.Sequential(
            OrderedDict([
                ("layer1", l1),
                ("layer2", l2),
                ("layer3", l3),
                ("layer4", l4),
                ("subnet_layer", subnet)
            ])
        )
        modules = list(s.modules())
        module_list = nn.ModuleList()
        module_list.extend(s.modules())
        check()

        modules = [nn.ReLU(), nn.Linear(5, 5), nn.Conv2d(3, 4, 3)]
        module_list = nn.ModuleList(modules)
        self.assertEqual(modules.pop(1), module_list.pop(1))
        self.assertEqual(modules, module_list)
        # check order of the index
        for k, mod in zip(range(len(module_list)), module_list):
            self.assertIs(module_list[k], mod)

        # verify the right exception is thrown when trying to "forward" through a ModuleList
        self.assertRaises(NotImplementedError, module_list)
        self.assertRaises(NotImplementedError, module_list, torch.rand(1, 3))

    def test_ModuleDict(self):
        modules = OrderedDict([
            ('act', nn.ReLU()),
            ('conv', nn.Conv2d(10, 10, 5)),
            ('fc', nn.Linear(5, 5)),
        ])

        module_dict = nn.ModuleDict(modules)

        def check():
            self.assertEqual(len(module_dict), len(modules))
            for k1, m2 in zip(modules, module_dict.children()):
                self.assertIs(modules[k1], m2)
            for k1, k2 in zip(modules, module_dict):
                self.assertIs(modules[k1], module_dict[k2])
            for k in module_dict:
                self.assertIs(module_dict[k], modules[k])
            for k in module_dict:
                self.assertIs(module_dict[k], modules[k])
            for k, v in module_dict.items():
                self.assertIs(modules[k], v)
            for k1, m2 in zip(modules, module_dict.values()):
                self.assertIs(modules[k1], m2)
            for k in modules:
                self.assertTrue(k in module_dict)
        check()

        modules['conv'] = nn.Conv2d(3, 4, 3)
        module_dict['conv'] = modules['conv']
        check()

        next_modules = [
            ('fc2', nn.Linear(5, 5)),
            ('act', nn.Sigmoid()),
        ]
        modules.update(next_modules)
        module_dict.update(next_modules)
        check()

        next_modules = OrderedDict([
            ('fc3', nn.Linear(5, 5)),
            ('act2', nn.Sigmoid()),
        ])
        modules.update(next_modules)
        module_dict.update(next_modules)
        check()

        next_modules = {
            'fc4': nn.Linear(5, 5),
            'act3': nn.Sigmoid()
        }
        modules.update(next_modules.items())
        module_dict.update(next_modules)
        check()

        next_modules = nn.ModuleDict([
            ('fc5', nn.Linear(5, 5)),
            ('act4', nn.Sigmoid()),
        ])
        modules.update(next_modules)
        module_dict.update(next_modules)
        check()

        del module_dict['fc']
        del modules['fc']
        check()

        with self.assertRaises(TypeError):
            module_dict.update(nn.ReLU())

        with self.assertRaises(TypeError):
            module_dict.update([nn.ReLU()])

        with self.assertRaises(ValueError):
            module_dict.update([[nn.ReLU()]])

        with self.assertRaises(TypeError):
            module_dict[1] = nn.ReLU()

        s = nn.Sequential(modules)
        module_dict = nn.ModuleDict(s.named_children())
        check()

        c = module_dict.pop('conv')
        self.assertIs(c, modules['conv'])
        modules.pop('conv')
        check()

        module_dict.clear()
        self.assertEqual(len(module_dict), 0)
        modules.clear()
        check()

        # verify the right exception is thrown when trying to "forward" through a ModuleDict
        self.assertRaises(NotImplementedError, module_dict)
        self.assertRaises(NotImplementedError, module_dict, torch.rand(1, 3))

    @skipIfTorchDynamo()
    def test_ParameterList(self):
        def make_param():
            return Parameter(torch.randn(2, 2))
        parameters = [make_param(), make_param()]
        param_list = nn.ParameterList(parameters)

        def check():
            self.assertEqual(len(parameters), len(param_list))
            for p1, p2 in zip(parameters, param_list):
                self.assertIs(p1, p2)
            for p1, p2 in zip(filter(lambda x: isinstance(x, Parameter), parameters), param_list.parameters()):
                self.assertIs(p1, p2)
            for i in range(len(parameters)):
                self.assertIs(parameters[i], param_list[i])

        check()
        parameters += [make_param()]
        param_list += [parameters[-1]]
        check()
        parameters.append(make_param())
        param_list.append(parameters[-1])
        check()
        next_params = [make_param(), make_param()]
        parameters.extend(next_params)
        param_list.extend(next_params)
        check()
        parameters[2] = make_param()
        param_list[2] = parameters[2]
        check()
        parameters[-1] = make_param()
        param_list[-1] = parameters[-1]
        check()
        idx = torch.tensor(2, dtype=torch.int32)
        parameters[2] = make_param()
        param_list[idx] = parameters[2]
        self.assertIs(param_list[idx], parameters[2])
        check()
        self.assertEqual(param_list[1:], nn.ParameterList(parameters[1:]))
        self.assertEqual(param_list[3:], nn.ParameterList(parameters[3:]))
        self.assertEqual(param_list[:-1], nn.ParameterList(parameters[:-1]))
        self.assertEqual(param_list[:-3], nn.ParameterList(parameters[:-3]))
        self.assertEqual(param_list[::-1], nn.ParameterList(parameters[::-1]))

        with self.assertRaises(TypeError):
            param_list += make_param()
        with self.assertRaises(TypeError):
            param_list.extend(make_param())

        l1 = nn.Linear(1, 2)
        l2 = nn.Linear(2, 3)
        l3 = nn.Linear(3, 2)
        l4 = nn.Linear(2, 3)
        subnet = nn.Sequential(l3, l4)
        s = nn.Sequential(
            OrderedDict([
                ("layer1", l1),
                ("layer2", l2),
                ("layer3", l3),
                ("layer4", l4),
                ("subnet_layer", subnet)
            ])
        )
        parameters = list(s.parameters())
        param_list = nn.ParameterList()
        param_list.extend(s.parameters())
        check()

        param_list.append(torch.rand(2, 2))
        self.assertIsInstance(param_list[-1], Parameter)
        parameters.append(param_list[-1])

        param_list.extend([torch.rand(2, 2), "foo"])
        self.assertIsInstance(param_list[-2], Parameter)
        self.assertIsInstance(param_list[-1], str)
        parameters.extend(param_list[-2:])

        param_list += ["bar", torch.rand(2, 2)]
        self.assertIsInstance(param_list[-2], str)
        self.assertIsInstance(param_list[-1], Parameter)
        parameters += param_list[-2:]
        check()

    def test_ParameterList_meta(self):
        p = torch.nn.Parameter(torch.empty(1, device='meta'))
        self.assertExpectedInline(str(p), """\
Parameter containing:
tensor(..., device='meta', size=(1,), requires_grad=True)""")
        pl = torch.nn.ParameterList([p])
        self.assertExpectedInline(str(pl), """ParameterList(  (0): Parameter containing: [torch.float32 of size 1])""")

    def test_ParameterList_replication(self):
        # The actual replication code from DP cannot be used on CPU so doing it manually here
        def make_param():
            return Parameter(torch.randn(2, 2))
        parameters = [make_param(), make_param()]
        param_list = nn.ParameterList(parameters)

        new_param_list = param_list._replicate_for_data_parallel()

        for n, p in param_list.named_parameters():
            # Do a view here so that we can check the base later
            setattr(new_param_list, n, p.view_as(p))

        for p, p2 in zip(param_list, new_param_list):
            self.assertEqual(p, p2)
            self.assertIsNotNone(p2.grad_fn)
            self.assertIs(p2._base, p)

    def test_ParameterDict(self):
        parameters = OrderedDict([
            ('p1', Parameter(torch.randn(10, 10))),
            ('p2', Parameter(torch.randn(10, 10))),
            ('p3', Parameter(torch.randn(10, 10))),
        ])

        parameter_dict = nn.ParameterDict(parameters)

        def check():
            self.assertEqual(len(parameter_dict), len(parameters))
            for (k1, (k2, m2)) in zip(parameters, parameter_dict.named_parameters()):
                self.assertEqual(k1, k2)
                self.assertIs(parameters[k1], m2)
            for k1, k2 in zip(parameters, parameter_dict):
                self.assertIs(parameters[k1], parameter_dict[k2])
            for k in parameter_dict:
                self.assertIs(parameter_dict[k], parameters[k])
            for k in parameter_dict:
                self.assertIs(parameter_dict[k], parameters[k])
            for k, v in parameter_dict.items():
                self.assertIs(v, parameters[k])
            for k1, m2 in zip(parameters, parameter_dict.values()):
                self.assertIs(parameters[k1], m2)
            for k in parameters:
                self.assertTrue(k in parameter_dict)

        check()

        parameters['p4'] = Parameter(torch.randn(10, 10))
        parameter_dict['p4'] = parameters['p4']
        check()

        next_parameters = [
            ('p5', Parameter(torch.randn(10, 10))),
            ('p2', Parameter(torch.randn(10, 10))),
        ]
        parameters.update(next_parameters)
        parameter_dict.update(next_parameters)
        check()

        next_parameters = OrderedDict([
            ('p6', Parameter(torch.randn(10, 10))),
            ('p5', Parameter(torch.randn(10, 10))),
        ])
        parameters.update(next_parameters)
        parameter_dict.update(next_parameters)
        check()

        next_parameters = {
            'p8': Parameter(torch.randn(10, 10)),
            'p7': Parameter(torch.randn(10, 10))
        }
        parameters.update(sorted(next_parameters.items()))
        parameter_dict.update(next_parameters)
        check()

        next_parameters = nn.ParameterDict([
            ('p10', Parameter(torch.randn(10, 10))),
            ('p9', Parameter(torch.randn(10, 10))),
        ])
        parameters.update(next_parameters)
        parameter_dict.update(next_parameters)
        check()

        del parameter_dict['p3']
        del parameters['p3']
        check()

        with self.assertRaises(TypeError):
            parameter_dict.update(1)

        with self.assertRaises(TypeError):
            parameter_dict.update([1])

        with self.assertRaises(ValueError):
            parameter_dict.update(Parameter(torch.randn(10, 10)))

        p_pop = parameter_dict.pop('p4')
        self.assertIs(p_pop, parameters['p4'])
        parameters.pop('p4')
        check()

        # Check reverse works
        forward = list(iter(parameter_dict))
        backward = list(reversed(parameter_dict))
        self.assertEqual(len(forward), len(backward))
        n = len(forward)
        for i in range(n):
            self.assertIs(forward[i], backward[n - i - 1])
        check()

        # Check copy works
        copy = parameter_dict.copy()

        # Check all keys are present and have shallow copied values
        for key in parameter_dict:
            self.assertTrue(key in copy)
            self.assertEqual(parameter_dict[key], copy[key])
            self.assertIs(parameter_dict[key], copy[key])
        check()

        parameter_dict["p20"] = Parameter(torch.randn(10, 10))
        copy["p21"] = Parameter(torch.randn(9, 10))

        self.assertTrue("p20" in parameter_dict)
        self.assertFalse("p20" in copy)
        self.assertFalse("p21" in parameter_dict)
        self.assertTrue("p21" in copy)
        parameter_dict.pop("p20")
        check()

        p = Parameter(torch.randn(10, 10))
        parameter_dict['p12'] = p
        p_popitem = parameter_dict.popitem()
        self.assertEqual(p_popitem[0], 'p12')
        self.assertIs(p_popitem[1], p)
        check()

        # Unit test for set_default
        # 1. Ensure parameter is correctly inserted when
        #    the key is not present in `ParameterDict`
        assert 'p11' not in parameter_dict
        assert 'p11' not in parameters
        parameters['p11'] = Parameter(torch.randn(10, 10))
        p_setdefault = parameter_dict.setdefault('p11', parameters['p11'])
        self.assertIs(p_setdefault, parameters['p11'])
        self.assertIs(p_setdefault, parameter_dict['p11'])
        check()
        # 2. Ensure parameter is NOT inserted when the
        #    key is already present in `ParameterDict`
        p = Parameter(torch.randn(10, 10))
        self.assertFalse(parameter_dict.setdefault('p11', p) is p)
        check()
        # 3. Ensure `None` is inserted when the key is not
        #    present in `Parameter` and parameter is not specified
        self.assertIs(parameter_dict.setdefault('p26'), None)
        del parameter_dict['p26']
        check()

        parameters2 = OrderedDict([
            ('p13', Parameter(torch.randn(10, 10))),
            ('p2', Parameter(torch.randn(10, 10))),
            ('p3', Parameter(torch.randn(10, 10))),
        ])
        parameter_dict2 = nn.ParameterDict(parameters2)
        parameters.update(parameters2)
        parameter_dict |= parameter_dict2
        check()

        parameters2 = OrderedDict()
        parameter_dict2 = nn.ParameterDict(parameters2)
        parameters.update(parameters2)
        parameter_dict |= parameter_dict2
        check()

        parameters2 = OrderedDict([
            ('p14', Parameter(torch.randn(10, 10))),
            ('p15', Parameter(torch.randn(10, 10))),
            ('p13', Parameter(torch.randn(10, 10))),
        ])
        parameter_dict2 = nn.ParameterDict(parameters2)
        parameters.update(parameters2)
        parameter_dict |= parameter_dict2
        check()

        # Check __or__ and __ror__ works
        parameters2 = OrderedDict([
            ('p20', Parameter(torch.randn(10, 10))),
            ('p21', Parameter(torch.randn(10, 10))),
            ('p22', Parameter(torch.randn(10, 10))),
        ])
        parameter_dict2 = nn.ParameterDict(parameters2)
        parameters.update(parameters2)
        parameter_dict = parameter_dict | parameter_dict2
        check()

        parameters2 = OrderedDict([
            ('p23', Parameter(torch.randn(10, 10))),
            ('p24', Parameter(torch.randn(10, 10))),
            ('p25', Parameter(torch.randn(10, 10))),
        ])
        parameter_dict2 = nn.ParameterDict(parameters2)
        parameters2.update(parameters)
        parameters = parameters2
        parameter_dict = parameter_dict2 | parameter_dict
        check()

        parameters['p17'] = Parameter(torch.randn(10, 10))
        parameter_dict['p17'] = parameters['p17']
        self.assertIs(parameters['p17'], parameter_dict.get('p17'))
        temp_param = Parameter(torch.randn(10, 10))
        self.assertIs(parameters['p17'], parameter_dict.get('p17', temp_param))
        self.assertIs(None, parameter_dict.get('p18'))
        self.assertIs(temp_param, parameter_dict.get('p18', temp_param))
        check()

        parameter_dict.clear()
        self.assertEqual(len(parameter_dict), 0)
        parameters.clear()
        check()

        parameter_dict2 = parameter_dict.fromkeys(['p19', 'p20'])
        self.assertEqual({'p19': None, 'p20': None}, parameter_dict2)
        check()

        parameter_dict2 = parameter_dict.fromkeys(['p19', 'p20'], temp_param)
        self.assertEqual({'p19': temp_param, 'p20': temp_param}, parameter_dict2)
        check()

        parameter_dict['p21'] = torch.rand(2, 2)
        self.assertIsInstance(parameter_dict['p21'], Parameter)
        parameters['p21'] = parameter_dict['p21']

        parameter_dict.update({'p22': torch.rand(2, 2), 'foo': 'bar'})
        self.assertIsInstance(parameter_dict['p22'], Parameter)
        self.assertIsInstance(parameter_dict['foo'], str)
        parameters['p22'] = parameter_dict['p22']
        parameters['foo'] = parameter_dict['foo']

    def test_ParameterDict_replication(self):
        # The actual replication code from DP cannot be used on CPU so doing it manually here
        def make_param():
            return Parameter(torch.randn(2, 2))
        parameters = {"foo": make_param(), "bar": make_param()}
        param_dict = nn.ParameterDict(parameters)

        new_param_dict = param_dict._replicate_for_data_parallel()

        for n, p in param_dict.named_parameters():
            # Do a view here so that we can check the base later
            setattr(new_param_dict, n, p.view_as(p))

        for (k, p), (k2, p2) in zip(param_dict.items(), new_param_dict.items()):
            self.assertEqual(k, k2)
            self.assertEqual(p, p2)
            self.assertIsNotNone(p2.grad_fn)
            self.assertIs(p2._base, p)

        self.assertEqual(param_dict["foo"], new_param_dict["foo"])

    def test_add_module(self):
        methods_to_test = ['add_module', 'register_module']
        for fn in methods_to_test:
            l = nn.Linear(10, 20)
            net = nn.Module()
            net.l = l
            net.l2 = l
            getattr(net, fn)('empty', None)
            self.assertEqual(net.l, l)
            self.assertEqual(net.l2, l)
            self.assertEqual(net.empty, None)
            getattr(net, fn)('l3', l)
            self.assertEqual(net.l3, l)
            l3 = nn.Linear(20, 10)
            getattr(net, fn)('l', l3)
            self.assertEqual(net.l, l3)
            self.assertRaises(TypeError, lambda: getattr(net, fn)('x', 'non-module'))
            self.assertRaisesRegex(TypeError, 'module name should be a string. Got int',
                                   lambda: getattr(net, fn)(1, l))
            self.assertRaisesRegex(TypeError, 'module name should be a string. Got NoneType',
                                   lambda: getattr(net, fn)(None, l))

    def test_set_submodule(self):
        # test the docstring example
        A = nn.Module()
        A.set_submodule("net_b", nn.Module())
        A.set_submodule("net_b.net_c", nn.Module())
        A.set_submodule("net_b.net_c.conv", nn.Conv2d(3, 3, 3))
        A.set_submodule("net_b.linear", nn.Linear(3, 3))
        new_linear = nn.Linear(1, 1)
        A.set_submodule("net_b.net_c.conv", new_linear)
        self.assertEqual(A.get_submodule("net_b.net_c.conv"), new_linear)
        new_linear = nn.Linear(1, 2)
        A.set_submodule("net_b.net_c.conv", new_linear, True)
        self.assertEqual(A.get_submodule("net_b.net_c.conv"), new_linear)
        new_conv = nn.Conv2d(1, 1, 1)
        self.assertRaises(AttributeError, A.set_submodule, "net_b.conv", new_conv, True)
        A.set_submodule("net_b.conv", new_conv)
        self.assertEqual(A.get_submodule("net_b.conv"), new_conv)

        # more tests
        net = nn.Module()
        net.t = nn.Module()
        l = nn.Linear(1, 2)
        target = "t.l"
        net.t.l = l
        self.assertEqual(net.get_submodule(target), l)
        l2 = nn.Linear(2, 1)
        net.set_submodule(target, l2)
        self.assertEqual(net.get_submodule(target), l2)
        self.assertRaises(ValueError, net.set_submodule, "", l)
        self.assertRaises(AttributeError, net.set_submodule, "a.l", l)
        self.assertRaises(AttributeError, net.set_submodule, "0", l, True)
        net.set_submodule("0", l, False)
        self.assertEqual(net.get_submodule("0"), l)
        l3 = nn.Linear(1, 1)
        net.set_submodule("0", l3, True)
        self.assertEqual(net.get_submodule("0"), l3)
        net.foo = "bar"
        self.assertRaises(AttributeError, net.set_submodule, "foo", l)
        self.assertRaises(ValueError, net.set_submodule, "t.l", "bazz")

    def test_module_to_argparse(self):
        net = nn.Sequential(nn.Linear(3, 3))
        cpu = torch.device('cpu')
        with self.assertRaises(TypeError):
            net.to(cpu, True)
        with self.assertRaises(TypeError):
            net.to(torch.long)
        with self.assertRaises(TypeError):
            net.to(None, True)
        with self.assertRaises(TypeError):
            net.to(cpu, torch.long, True)
        with self.assertRaises(TypeError):
            net.to(cpu, dtype=torch.long, non_blocking=True)
        with self.assertRaises(TypeError):
            net.to([])
        with self.assertRaises(TypeError):
            net.to({}, non_blocking=True)
        with self.assertRaises(TypeError):
            net.to(torch.tensor(3, dtype=torch.long), non_blocking=True)
        with self.assertRaises(TypeError):
            net.to(cpu, torch.tensor(3, dtype=torch.long), non_blocking=True)

    def test_RNN_nonlinearity(self):
        rnn = torch.nn.RNN(1, 10)
        self.assertEqual(rnn.nonlinearity, 'tanh')

        rnn = torch.nn.RNN(1, 10, nonlinearity='relu')
        self.assertEqual(rnn.nonlinearity, 'relu')

        with self.assertRaisesRegex(ValueError, 'Unknown nonlinearity'):
            rnn = torch.nn.RNN(1, 10, nonlinearity='garbage')

    def test_RNN_nonlinearity_passed_as_arg(self):
        rnn = torch.nn.RNN(2, 3, 1, 'relu')
        self.assertEqual(rnn.nonlinearity, 'relu')

    def test_module_apply_inplace_op(self):
        def add_one_inplace(t):
            return t.add_(1.0)

        # Test that applying an in-place operation to a module would bump
        # the module's parameters' version counter.
        m = nn.Linear(20, 10)
        pvm = m.weight.mul(m.weight)
        m_weight_version_saved = m.weight._version
        m = m._apply(add_one_inplace)
        self.assertGreater(m.weight._version, m_weight_version_saved)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            pvm.backward(torch.randn(10, 20))

        # Test that applying an in-place operation to a module would bump
        # the module's parameters' gradients' version counter.
        m = nn.Linear(20, 10)
        m.weight.grad = torch.randn(10, 20).requires_grad_()
        pgm = m.weight.grad.mul(m.weight.grad)
        m_weight_grad_version_saved = m.weight.grad._version
        m = m._apply(add_one_inplace)
        self.assertGreater(m.weight.grad._version, m_weight_grad_version_saved)
        with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
            pgm.backward(torch.randn(10, 20))

    def test_overwrite_module_params_on_conversion(self):
        # Test that if the conversion function passed to `module._apply()`
        # changes the TensorImpl type of `module`'s parameters, the `module`'s
        # parameters are always overwritten, regardless of the value of
        # `torch.__future__.get_overwrite_module_params_on_conversion()`.
        m = nn.Linear(20, 10)
        m.weight.grad = torch.randn(10, 20)
        weight_ref = m.weight
        weight_grad_ref = m.weight.grad
        m = m._apply(lambda t: torch.sparse_coo_tensor(torch.zeros([2, 1]), torch.ones([1]), torch.Size([10, 20])))
        self.assertNotEqual(weight_ref.layout, m.weight.layout)
        self.assertNotEqual(weight_grad_ref.layout, m.weight.grad.layout)

        # Test that under the current default settings
        # (`torch.__future__.get_overwrite_module_params_on_conversion() == False`),
        # a view to a module's parameters is not pointing to the same storage as
        # its base variable after converting the module to a different dtype.
        m = nn.Linear(20, 10).float()
        mw = m.weight[:]
        m.double()
        with torch.no_grad():
            mw[0][0] = 5
        self.assertTrue(mw[0][0].dtype == torch.float)
        self.assertTrue(mw._base[0][0].dtype == torch.double)

        try:
            torch.__future__.set_overwrite_module_params_on_conversion(True)

            # Test that if `torch.__future__.get_overwrite_module_params_on_conversion() == True`,
            # a view to a module's parameters is still pointing to the same storage as
            # its base variable after converting the module to a different dtype.
            m = nn.Linear(20, 10).float()
            mw = m.weight[:]
            m.double()
            with torch.no_grad():
                mw[0][0] = 5
            self.assertTrue(mw[0][0] == mw._base[0][0])

            # Test that if `torch.__future__.get_overwrite_module_params_on_conversion() == True`,
            # `float_module.double()` doesn't preserve previous references to
            # `float_module`'s parameters or gradients.
            m = nn.Linear(20, 10).float()
            m.weight.grad = torch.randn(10, 20).float()
            weight_ref = m.weight
            weight_grad_ref = m.weight.grad
            m.double()
            self.assertNotEqual(weight_ref.dtype, m.weight.dtype)
            self.assertNotEqual(weight_grad_ref.dtype, m.weight.grad.dtype)

            def add_one_inplace(t):
                return t.add_(1.0)

            # Test that if `torch.__future__.get_overwrite_module_params_on_conversion() == True`,
            # applying an in-place operation to a module would bump the module's
            # original parameters' version counter.
            m = nn.Linear(20, 10)
            pvm = m.weight.mul(m.weight)
            weight_ref = m.weight
            m_weight_version_saved = weight_ref._version
            m = m._apply(add_one_inplace)
            # Test that the in-place operation bumps the original parameter's version counter
            self.assertGreater(weight_ref._version, m_weight_version_saved)
            with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
                pvm.backward(torch.randn(10, 20))

            # Test that if `torch.__future__.get_overwrite_module_params_on_conversion() == True`,
            # applying an in-place operation to a module would bump the module's
            # original parameters' gradients' version counter.
            m = nn.Linear(20, 10)
            m.weight.grad = torch.randn(10, 20).requires_grad_()
            pgm = m.weight.grad.mul(m.weight.grad)
            weight_grad_ref = m.weight.grad
            m_weight_grad_version_saved = weight_grad_ref._version
            m = m._apply(add_one_inplace)
            self.assertGreater(weight_grad_ref._version, m_weight_grad_version_saved)
            with self.assertRaisesRegex(RuntimeError, "modified by an inplace operation"):
                pgm.backward(torch.randn(10, 20))

            # Test that if `torch.__future__.get_overwrite_module_params_on_conversion() == True`,
            # applying an out-of-place operation to a module doesn't bump
            # the module's original parameters' version counter.
            m = nn.Linear(20, 10)
            weight_ref = m.weight
            m_weight_version_saved = weight_ref._version
            m = m._apply(lambda t: torch.randn(t.shape))
            self.assertEqual(weight_ref._version, m_weight_version_saved)

            # Test that if `torch.__future__.get_overwrite_module_params_on_conversion() == True`,
            # applying an out-of-place operation to a module doesn't bump
            # the module's original parameters' gradients' version counter.
            m = nn.Linear(20, 10)
            m.weight.grad = torch.randn(10, 20).requires_grad_()
            weight_grad_ref = m.weight.grad
            m_weight_grad_version_saved = weight_grad_ref._version
            m = m._apply(lambda t: torch.randn(t.shape))
            self.assertEqual(weight_grad_ref._version, m_weight_grad_version_saved)
        finally:
            torch.__future__.set_overwrite_module_params_on_conversion(False)

    def test_swap_module_params_poisons_acc_grad(self):
        try:
            torch.__future__.set_swap_module_params_on_conversion(True)
            # (1) backward cannot be run after _apply
            # forward will init AccumulateGrad nodes, which bumps use_count of parameters' at::Tensors
            # additionally, if any Tensors are saved for backward, their use_count will be bumped
            m = torch.nn.Linear(2, 3)
            inp = torch.randn(2, 2)
            out = m(inp)
            m.half()
            self.assertTrue(all(p.dtype == torch.float16 for p in m.parameters()))
            with self.assertRaisesRegex(RuntimeError, "Trying to execute AccumulateGrad node that was poisoned by swap_tensors"):
                out.sum().backward()
            # (2) _apply can be run after backward()
            # After running backward, all the references generated by "save for backward" will be cleared
            # So the use_count will be 2 (1 from Tensor itself, and 1 from AccumulateGrad node), swap_tensors
            # should allow this.
            inp2 = torch.randn(2, 2, dtype=torch.half)
            out2 = m(inp2)
            out2.sum().backward()
            m.float()
            self.assertTrue(all(p.dtype == torch.float32 for p in m.parameters()))
            out3 = m(inp)
        finally:
            torch.__future__.set_swap_module_params_on_conversion(False)

    def test_type(self):
        l = nn.Linear(10, 20)
        net = nn.Module()
        net.l = l
        net.l2 = l
        net.add_module('empty', None)
        net.indices = Buffer(torch.LongTensor(1))
        net.float()
        self.assertIsInstance(l.weight.data, torch.FloatTensor)
        self.assertIsInstance(l.bias.data, torch.FloatTensor)
        self.assertIsInstance(net.indices, torch.LongTensor)
        net.double()
        self.assertIsInstance(l.weight.data, torch.DoubleTensor)
        self.assertIsInstance(l.bias.data, torch.DoubleTensor)
        self.assertIsInstance(net.indices, torch.LongTensor)
        net.to(torch.half)
        self.assertIsInstance(l.weight.data, torch.HalfTensor)
        self.assertIsInstance(l.bias.data, torch.HalfTensor)
        self.assertIsInstance(net.indices, torch.LongTensor)
        if TEST_CUDA:
            net.float().cuda()
            self.assertIsInstance(l.weight.data, torch.cuda.FloatTensor)
            self.assertIsInstance(l.bias.data, torch.cuda.FloatTensor)
            self.assertIsInstance(net.indices, torch.cuda.LongTensor)
            net.cpu()
            self.assertIsInstance(l.weight.data, torch.FloatTensor)
            self.assertIsInstance(l.bias.data, torch.FloatTensor)
            self.assertIsInstance(net.indices, torch.LongTensor)
            net.to("cuda", torch.double, True)
            self.assertIsInstance(l.weight.data, torch.cuda.DoubleTensor)
            self.assertIsInstance(l.bias.data, torch.cuda.DoubleTensor)
            self.assertIsInstance(net.indices, torch.cuda.LongTensor)
            net.to(torch.empty(1, device="cuda:0", dtype=torch.half))
            self.assertIsInstance(l.weight.data, torch.cuda.HalfTensor)
            self.assertIsInstance(l.bias.data, torch.cuda.HalfTensor)
            self.assertIsInstance(net.indices, torch.cuda.LongTensor)
        net.to(torch.device("cpu"), non_blocking=True)
        self.assertIsInstance(l.weight.data, torch.HalfTensor)
        self.assertIsInstance(l.bias.data, torch.HalfTensor)
        self.assertIsInstance(net.indices, torch.LongTensor)
        net.to(torch.float)
        self.assertIsInstance(l.weight.data, torch.FloatTensor)
        self.assertIsInstance(l.bias.data, torch.FloatTensor)
        net.to(torch.DoubleTensor(1))
        self.assertIsInstance(l.weight.data, torch.DoubleTensor)
        self.assertIsInstance(l.bias.data, torch.DoubleTensor)
        if TEST_CUDA:
            net.to(device='cuda', dtype=torch.float)
            self.assertIsInstance(l.weight.data, torch.cuda.FloatTensor)
            self.assertIsInstance(l.bias.data, torch.cuda.FloatTensor)

    def test_non_leaf_parameters(self):
        l1 = nn.Linear(10, 10)
        l2 = nn.Linear(10, 10)

        def assign_weight():
            l2.weight = l1.weight + 2

        self.assertRaises(TypeError, assign_weight)
        # This should work though
        l2.weight = Parameter(torch.randn(10, 10))

    def test_parameters_to_vector(self):
        conv1 = nn.Conv2d(3, 10, 5)
        fc1 = nn.Linear(10, 20)
        model = nn.Sequential(conv1, fc1)

        vec = parameters_to_vector(model.parameters())
        self.assertEqual(vec.size(0), 980)

    def test_vector_to_parameters(self):
        conv1 = nn.Conv2d(3, 10, 5)
        fc1 = nn.Linear(10, 20)
        model = nn.Sequential(conv1, fc1)

        vec = torch.arange(0., 980)
        vector_to_parameters(vec, model.parameters())

        sample = next(model.parameters())[0, 0, 0]
        self.assertTrue(torch.equal(sample.data, vec.data[:5]))

    def test_rnn_weight_norm(self):
        def check_weight_norm(l, name, num_params):
            # This Module has 4 or 5 parameters called:
            # 'weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0', weight_hr_l0

            # Applying weight norm on one of them causes it to become a tensor
            l = torch.nn.utils.weight_norm(l, name=name)
            self.assertEqual(
                sum(isinstance(p, torch.nn.Parameter) for p in l._flat_weights),
                num_params - 1,
            )

            # Removing the weight norm reparameterization restores the Parameter
            l = torch.nn.utils.remove_weight_norm(l, name=name)
            self.assertEqual(
                sum(isinstance(p, torch.nn.Parameter) for p in l._flat_weights),
                num_params,
            )

            # Make sure that, upon removal of the reparameterization, the
            # `._parameters` and `.named_parameters` contain the right params.
            # Specifically, the original weight ('weight_ih_l0') should be placed
            # back in the parameters, while the reparameterization components
            # ('weight_ih_l0_v' and 'weight_ih_l0_g') should be removed.
            self.assertTrue(name in l._parameters)
            self.assertIsNotNone(l._parameters[name])
            self.assertTrue(name + '_v' not in l._parameters)
            self.assertTrue(name + '_g' not in l._parameters)
            self.assertTrue(name in dict(l.named_parameters()))
            self.assertIsNotNone(dict(l.named_parameters())[name])
            self.assertTrue(name + '_v' not in dict(l.named_parameters()))
            self.assertTrue(name + '_g' not in dict(l.named_parameters()))

        check_weight_norm(torch.nn.LSTM(32, 32), 'weight_ih_l0', 4)
        check_weight_norm(torch.nn.LSTM(32, 32, proj_size=16), 'weight_hr_l0', 5)


    def test_weight_norm(self):
        for dtype in [torch.float, torch.bfloat16, torch.float16]:
            input = torch.randn(3, 4, dtype=dtype)
            m = nn.Linear(4, 5).to(dtype=dtype)
            expected_output = m(input)

            # add weight normalization
            m = torch.nn.utils.weight_norm(m)
            self.assertEqual(m.weight_v.size(), m.weight.size())
            self.assertEqual(m.weight_g.size(), (5, 1))
            self.assertEqual(m(input), expected_output, atol=dtype2prec_DONTUSE[dtype], rtol=0)

            # remove weight norm
            m = torch.nn.utils.remove_weight_norm(m)
            self.assertFalse(hasattr(m, 'weight_g'))
            self.assertFalse(hasattr(m, 'weight_v'))
            self.assertEqual(m(input), expected_output, atol=dtype2prec_DONTUSE[dtype], rtol=0)

            # test with dim=1
            m = torch.nn.utils.weight_norm(m, dim=1)
            self.assertEqual(m.weight_v.size(), m.weight.size())
            self.assertEqual(m.weight_g.size(), (1, 4))
            self.assertEqual(m(input), expected_output, atol=dtype2prec_DONTUSE[dtype], rtol=0)

            # test with dim=None
            m = nn.Linear(4, 5).to(dtype=dtype)
            expected_output = m(input)
            m = torch.nn.utils.weight_norm(m, dim=None)
            self.assertEqual(m(input), expected_output)

            with self.assertRaisesRegex(RuntimeError, 'register two weight_norm hooks'):
                m = torch.nn.utils.weight_norm(m)
                m = torch.nn.utils.weight_norm(m)

        # For float16, the forward of the Module doesn't work but we must still be able
        # to register the weight norm as this is often done before sending the Module to
        # CUDA.
        m = nn.Linear(4, 5, dtype=torch.float16)
        m = torch.nn.utils.weight_norm(m)

    def test_parameterlistdict_setting_attributes(self):
        with warnings.catch_warnings(record=True) as w:
            mod = nn.ParameterList(map(nn.Parameter, [torch.rand(2), torch.rand(2)]))
        self.assertTrue(len(w) == 0)

        with warnings.catch_warnings(record=True) as w:
            mod.train()
            mod.eval()
        self.assertTrue(len(w) == 0)

        with warnings.catch_warnings(record=True) as w:
            mod = nn.ParameterDict({"a": nn.Parameter(torch.rand(2)), "b": nn.Parameter(torch.rand(2))})
        self.assertTrue(len(w) == 0)

        with warnings.catch_warnings(record=True) as w:
            mod.train()
            mod.eval()
        self.assertTrue(len(w) == 0)

    def test_parameterlistdict_pickle(self):
        m = nn.ParameterList(map(nn.Parameter, [torch.rand(2), torch.rand(2)]))
        with warnings.catch_warnings(record=True) as w:
            m = pickle.loads(pickle.dumps(m))
        self.assertTrue(len(w) == 0)

        # Test whether loading from older checkpoints works without triggering warnings
        m = nn.ParameterList(map(nn.Parameter, [torch.rand(2), torch.rand(2)]))
        del m._forward_pre_hooks, m._state_dict_hooks, m._load_state_dict_pre_hooks, m._non_persistent_buffers_set
        with warnings.catch_warnings(record=True) as w:
            m = pickle.loads(pickle.dumps(m))
        self.assertTrue(len(w) == 0)

        m = nn.ParameterDict({"a": nn.Parameter(torch.rand(2)), "b": nn.Parameter(torch.rand(2))})
        with warnings.catch_warnings(record=True) as w:
            m = pickle.loads(pickle.dumps(m))
        self.assertTrue(len(w) == 0)

        # Test whether loading from older checkpoints works without triggering warnings
        m = nn.ParameterDict({"a": nn.Parameter(torch.rand(2)), "b": nn.Parameter(torch.rand(2))})
        del m._forward_pre_hooks, m._state_dict_hooks, m._load_state_dict_pre_hooks, m._non_persistent_buffers_set
        with warnings.catch_warnings(record=True) as w:
            m = pickle.loads(pickle.dumps(m))
        self.assertTrue(len(w) == 0)

    def test_weight_norm_pickle(self):
        m = torch.nn.utils.weight_norm(nn.Linear(5, 7))
        m = pickle.loads(pickle.dumps(m))
        self.assertIsInstance(m, nn.Linear)

    @skipIfTorchDynamo("TorchDynamo fails here for unknown reasons")
    @set_default_dtype(torch.double)
    def test_spectral_norm(self):
        input = torch.randn(3, 5)
        m = nn.Linear(5, 7)
        m = torch.nn.utils.spectral_norm(m)

        self.assertEqual(m.weight_u.size(), torch.Size([m.weight.size(0)]))
        # weight_orig should be trainable
        self.assertTrue(hasattr(m, 'weight_orig'))
        self.assertTrue('weight_orig' in m._parameters)
        # weight_u should be just a reused buffer
        self.assertTrue(hasattr(m, 'weight_u'))
        self.assertTrue('weight_u' in m._buffers)
        self.assertTrue('weight_v' in m._buffers)
        # weight should be a plain attribute, not counted as a buffer or a param
        self.assertFalse('weight' in m._buffers)
        self.assertFalse('weight' in m._parameters)
        # it should also be sharing storage as `weight_orig`
        self.assertEqual(m.weight_orig.storage(), m.weight.storage())
        self.assertEqual(m.weight_orig.size(), m.weight.size())
        self.assertEqual(m.weight_orig.stride(), m.weight.stride())

        m = torch.nn.utils.remove_spectral_norm(m)
        self.assertFalse(hasattr(m, 'weight_orig'))
        self.assertFalse(hasattr(m, 'weight_u'))
        # weight should be converted back as a parameter
        self.assertTrue(hasattr(m, 'weight'))
        self.assertTrue('weight' in m._parameters)

        with self.assertRaisesRegex(RuntimeError, 'register two spectral_norm hooks'):
            m = torch.nn.utils.spectral_norm(m)
            m = torch.nn.utils.spectral_norm(m)

        # test correctness in training/eval modes and cpu/multi-gpu settings
        for apply_dp in (True, False):
            if apply_dp:
                if not TEST_MULTIGPU:
                    continue
                device = torch.device('cuda:0')

                def maybe_wrap(m):
                    return torch.nn.DataParallel(m, [0, 1])
            else:
                device = torch.device('cpu')

                def maybe_wrap(m):
                    return m

            for requires_grad in (True, False):
                m = nn.Linear(3, 4).to(device)
                m.weight.requires_grad_(requires_grad)
                m = torch.nn.utils.spectral_norm(m)
                wrapped_m = maybe_wrap(m)
                self.assertTrue(hasattr(m, 'weight_u'))
                u0 = m.weight_u.clone()
                v0 = m.weight_v.clone()

                # TEST TRAINING BEHAVIOR

                # assert that u and v are updated
                input = torch.randn(2, 3, device=device)
                out = wrapped_m(input)
                self.assertNotEqual(u0, m.weight_u)
                self.assertNotEqual(v0, m.weight_v)

                # assert that backprop reaches weight_orig
                # can't use gradcheck because the function changes as we
                # activate through it in training mode
                if requires_grad:
                    torch.autograd.grad(out.sum(), m.weight_orig)

                # test backward works with multiple forwards
                # it uses training mode so we need to reset `u` and `v` vectors
                # to same value at beginning for finite difference test to pass
                saved_u = m.weight_u.clone()
                saved_v = m.weight_v.clone()

                def fn(input):
                    m.weight_u.data.copy_(saved_u)
                    m.weight_v.data.copy_(saved_v)
                    out0 = wrapped_m(input)
                    out1 = wrapped_m(input)
                    return out0 + out1

                gradcheck(fn, (input.clone().requires_grad_(),), check_batched_grad=False)

                # test removing
                pre_remove_out = wrapped_m(input)
                m = torch.nn.utils.remove_spectral_norm(m)
                self.assertEqual(wrapped_m(input), pre_remove_out)

                m = torch.nn.utils.spectral_norm(m)
                for _ in range(3):
                    pre_remove_out = wrapped_m(input)
                m = torch.nn.utils.remove_spectral_norm(m)
                self.assertEqual(wrapped_m(input), pre_remove_out)

                # TEST EVAL BEHAVIOR

                m = torch.nn.utils.spectral_norm(m)
                wrapped_m(input)
                last_train_out = wrapped_m(input)
                last_train_u = m.weight_u.clone()
                last_train_v = m.weight_v.clone()
                wrapped_m.zero_grad()
                wrapped_m.eval()

                eval_out0 = wrapped_m(input)
                # assert eval gives same result as last training iteration
                self.assertEqual(eval_out0, last_train_out)
                # assert doing more iteration in eval don't change things
                self.assertEqual(eval_out0, wrapped_m(input))
                self.assertEqual(last_train_u, m.weight_u)
                self.assertEqual(last_train_v, m.weight_v)

                # FIXME: the code below is flaky when executed with DataParallel
                # see https://github.com/pytorch/pytorch/issues/13818
                if apply_dp:
                    continue

                # test backward works with multiple forwards in mixed training
                # and eval modes
                # it uses training mode so we need to reset `u` and `v` vectors
                # to same value at beginning for finite difference test to pass
                saved_u = m.weight_u.clone()
                saved_v = m.weight_v.clone()

                def fn(input):
                    m.weight_u.data.copy_(saved_u)
                    m.weight_v.data.copy_(saved_v)
                    wrapped_m.train()
                    out0 = wrapped_m(input)
                    wrapped_m.eval()
                    out1 = wrapped_m(input)
                    wrapped_m.train()
                    out2 = wrapped_m(input)
                    wrapped_m.eval()
                    out3 = wrapped_m(input)
                    return out0 + out1 + out2 + out3

                gradcheck(fn, (input.clone().requires_grad_(),))

                # assert that backprop reaches weight_orig in eval
                if requires_grad:
                    def fn(weight):
                        return wrapped_m(input)

                    gradcheck(fn, (m.weight_orig,))

    @skipIfNoLapack
    def test_spectral_norm_load_state_dict(self):
        inp = torch.randn(2, 3)
        for activate_times in (0, 3):
            # Test backward compatibility
            # At version None -> 1: weight becomes not a buffer and v vector becomes a buffer
            m = nn.Linear(3, 5)
            snm = torch.nn.utils.spectral_norm(m)
            snm.train()
            for _ in range(activate_times):
                snm(inp)

            version_latest_ref_state_dict = deepcopy(snm.state_dict())
            self.assertEqual({'weight_orig', 'bias', 'weight_u', 'weight_v'}, set(version_latest_ref_state_dict.keys()))

            # test that non-strict loading works
            non_strict_state_dict = deepcopy(version_latest_ref_state_dict)
            non_strict_state_dict['nonsense'] = 'nonsense'
            with self.assertRaisesRegex(RuntimeError, r'Unexpected key\(s\) in state_dict: "nonsense"'):
                snm.load_state_dict(non_strict_state_dict, strict=True)
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['weight_orig']
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['weight_u']
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['weight_v']
            snm.load_state_dict(non_strict_state_dict, strict=False)
            non_strict_state_dict['weight'] = snm.weight.detach().clone()  # set W as a buffer
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict._metadata['']['spectral_norm']       # remove metadata info
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['weight']                            # remove W buffer
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict['bias']
            snm.load_state_dict(non_strict_state_dict, strict=False)

            # craft a version None state_dict
            version_none_state_dict = deepcopy(version_latest_ref_state_dict)
            self.assertIn('spectral_norm', version_none_state_dict._metadata[''])
            del version_none_state_dict._metadata['']['spectral_norm']       # remove metadata info
            del version_none_state_dict['weight_v']                          # remove v vector
            version_none_state_dict['weight'] = snm.weight.detach().clone()  # set W as a buffer

            # normal state_dict
            for version_latest_with_metadata in [True, False]:
                version_latest_state_dict = deepcopy(version_latest_ref_state_dict)

                if not version_latest_with_metadata:
                    # We want to still load a user-crafted state_dict, one without metadata
                    del version_latest_state_dict._metadata['']['spectral_norm']

                # test that re-wrapping does not matter
                m = torch.nn.utils.remove_spectral_norm(snm)
                snm = torch.nn.utils.spectral_norm(m)

                snm.load_state_dict(version_latest_ref_state_dict)
                with torch.no_grad():
                    snm.eval()
                    out0_eval = snm(inp)
                    snm.train()
                    out1_train = snm(inp)
                    out2_train = snm(inp)
                    snm.eval()
                    out3_eval = snm(inp)

                # test that re-wrapping does not matter
                m = torch.nn.utils.remove_spectral_norm(snm)
                snm = torch.nn.utils.spectral_norm(m)

                snm.load_state_dict(version_none_state_dict)
                if activate_times > 0:
                    # since in loading version None state dict, we assume that the
                    # values in the state dict have gone through at lease one
                    # forward, we only test for equivalence when activate_times > 0.
                    with torch.no_grad():
                        snm.eval()
                        self.assertEqual(out0_eval, snm(inp))
                        snm.train()
                        self.assertEqual(out1_train, snm(inp))
                        self.assertEqual(out2_train, snm(inp))
                        snm.eval()
                        self.assertEqual(out3_eval, snm(inp))

                # test that re-wrapping does not matter
                m = torch.nn.utils.remove_spectral_norm(snm)
                snm = torch.nn.utils.spectral_norm(m)

                # Test normal loading
                snm.load_state_dict(version_latest_state_dict)
                with torch.no_grad():
                    snm.eval()
                    self.assertEqual(out0_eval, snm(inp))
                    snm.train()
                    self.assertEqual(out1_train, snm(inp))
                    self.assertEqual(out2_train, snm(inp))
                    snm.eval()
                    self.assertEqual(out3_eval, snm(inp))

    def test_spectral_norm_dim(self):
        inp = torch.randn(2, 3, 10, 12)
        m = nn.ConvTranspose2d(3, 4, (5, 6))
        m = torch.nn.utils.spectral_norm(m)
        # this should not run into incompatible shapes
        x = m(inp)
        # check that u refers to the same dimension
        self.assertEqual(m.weight_u.shape, m.weight_orig[0, :, 0, 0].shape)

    def test_spectral_norm_forward(self):
        input = torch.randn(3, 5)
        m = nn.Linear(5, 7)
        m = torch.nn.utils.spectral_norm(m)
        # naive forward
        _weight, _bias, _u = m.weight_orig, m.bias, m.weight_u
        _weight_mat = _weight.view(_weight.size(0), -1)
        _v = torch.mv(_weight_mat.t(), _u)
        _v = F.normalize(_v, dim=0, eps=1e-12)
        _u = torch.mv(_weight_mat, _v)
        _u = F.normalize(_u, dim=0, eps=1e-12)
        _weight.data /= torch.dot(_u, torch.matmul(_weight_mat, _v))
        out_hat = torch.nn.functional.linear(input, _weight, _bias)
        expect_out = m(input)
        self.assertEqual(expect_out, out_hat)

    def test_spectral_norm_pickle(self):
        m = torch.nn.utils.spectral_norm(nn.Linear(5, 7))
        m = pickle.loads(pickle.dumps(m))
        self.assertIsInstance(m, nn.Linear)

    def test_threshold_int(self):
        x = torch.tensor([-3, -2, -1, 0, 1, 2, 3])
        expected = torch.tensor([99, 99, 99, 99, 1, 2, 3])
        self.assertEqual(F.threshold(x, 0, 99), expected)

    def test_threshold_bfloat16_half(self):
        x = torch.randn(100)
        for dtype in [torch.bfloat16, torch.half]:
            for threshold in [0, -0.5, 0.5, float('inf'), float('-inf'), float('nan')]:
                expected = F.threshold(x, threshold, 0).to(dtype=dtype).float()
                res_bf16 = F.threshold(x.to(dtype=dtype), threshold, 0).float()
                self.assertEqual(res_bf16, expected)

    @unittest.skipUnless('fbgemm' in torch.backends.quantized.supported_engines,
                         'Linear_FP16_weight requires FBGEMM. FBGEMM is only optimized for CPUs'
                         ' with instruction set support avx2 or newer.')
    def test_fb_fc_packed(self):
        X = np.random.rand(16, 16).astype(np.float32) - 0.5
        W = np.random.rand(16, 16).astype(np.float32) - 0.5
        b = np.random.rand(16).astype(np.float32) - 0.5

        def fc_op(X, W, b):
            return np.dot(X, W.T) + b

        x_tensor = torch.tensor(X)
        w_tensor = torch.tensor(W)
        b_tensor = torch.tensor(b)
        packed_w_tensor = torch.fbgemm_pack_gemm_matrix_fp16(w_tensor)
        actual_output = torch.fbgemm_linear_fp16_weight(x_tensor, packed_w_tensor, b_tensor)
        expected_output = fc_op(X, W, b)
        torch.testing.assert_close(torch.from_numpy(expected_output), actual_output.cpu(), atol=1e-3, rtol=1e-3)

    def test_pad_scalar_error(self):
        inputs = torch.tensor(0., requires_grad=True)
        self.assertRaises(RuntimeError, lambda: F.pad(inputs, (1, 1)))
        self.assertRaises(RuntimeError, lambda: F.pad(inputs, (1,)))

    def test_nested_tensor_from_mask(self):
        N, L, D = 10, 12, 14

        input = torch.rand(N, L, D)
        mask = torch.ones(N, L, dtype=torch.bool)
        # Leave first row be all True to maintain the nt's size unchanged
        for i in range(1, N):
            end = torch.randint(1, L, size=()).item()
            mask[i, end:] = False

        nt = torch._nested_tensor_from_mask(input, mask)
        input_convert = nt.to_padded_tensor(0.)
        input.masked_fill_(mask.reshape(N, L, 1).logical_not(), 0.)

        self.assertEqual(input, input_convert)

    def test_nested_tensor_from_mask_error(self):
        N, L, D = 10, 12, 14

        input = torch.rand(N, L, D)
        # Mask is not bool
        mask = torch.zeros(N, L, dtype=torch.float)
        self.assertRaises(RuntimeError, lambda: torch._nested_tensor_from_mask(input, mask))

        # Mask size is not 2
        mask = torch.zeros(N, L, D, dtype=torch.bool)
        self.assertRaises(RuntimeError, lambda: torch._nested_tensor_from_mask(input, mask))

        # Input size is not 3
        mask = torch.zeros(N, L, dtype=torch.bool)
        input = torch.rand(N, L)
        self.assertRaises(RuntimeError, lambda: torch._nested_tensor_from_mask(input, mask))

        # Mask size does not match input
        mask = torch.zeros(N + 1, L + 1, dtype=torch.bool)
        input = torch.rand(N, L, D)
        self.assertRaises(RuntimeError, lambda: torch._nested_tensor_from_mask(input, mask))

        # Mask is not padding format
        mask = torch.ones(N, L, dtype=torch.bool)
        mask[0, 0] = False
        mask[0, 2] = False
        self.assertRaises(RuntimeError, lambda: torch._nested_tensor_from_mask(input, mask))

    def test_normalize(self):
        inputs = torch.randn(1, 3, 4, 4, requires_grad=True, dtype=torch.double)
        self.assertTrue(gradcheck(lambda x: F.normalize(x, p=1, dim=-1), (inputs,)))
        self.assertTrue(gradcheck(lambda x: F.normalize(x, p=2, dim=-2), (inputs,)))

        inputs = torch.randn((), requires_grad=True)
        self.assertTrue(gradcheck(lambda x: F.normalize(x, p=1, dim=-1), (inputs,)))

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    # Skip the test for ROCm as per https://github.com/pytorch/pytorch/issues/53190
    @skipIfRocm
    def test_broadcast_double_backwards_gpu(self):
        tensors = (torch.randn(4, 4, device='cuda', requires_grad=True, dtype=torch.double),
                   torch.randn(4, 4, device='cuda', requires_grad=True, dtype=torch.double),
                   torch.randn(4, 4, device='cuda', requires_grad=True, dtype=torch.double))
        # TODO(#50743): the following segfaults with check_batched_grad=True
        _assertGradAndGradgradChecks(self, lambda *i: Broadcast.apply((0, 1), *i), tensors,
                                     check_batched_grad=False)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_broadcast_not_requiring_grad(self):
        variables = [
            torch.randn(1, 2, device='cuda', requires_grad=True),
            torch.randn(1, 2, device='cuda', requires_grad=False),
            torch.randn(1, 2, device='cuda', requires_grad=False),
            torch.randn(1, 2, device='cuda', requires_grad=True),
            torch.randn(1, 2, device='cuda', requires_grad=True),
        ]
        broadcasted_variables = Broadcast.apply((0, 1), *variables)
        for output_idx, broadcasted_var in enumerate(broadcasted_variables):
            input_var = variables[output_idx % len(variables)]
            self.assertEqual(input_var.requires_grad, broadcasted_var.requires_grad)

    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    def test_broadcast_no_grad(self):
        x = torch.randn(1, 2, dtype=torch.float32, requires_grad=True, device='cuda')
        with torch.no_grad():
            broadcasted = Broadcast.apply((0, 1), x)
        self.assertTrue(x.requires_grad)
        for output in broadcasted:
            self.assertFalse(output.requires_grad)

    def test_state_dict(self):
        l = nn.Linear(5, 5)
        block = nn.Module()
        block.conv = nn.Conv2d(3, 3, 3, bias=False)
        net = nn.Module()
        net.linear1 = l
        net.linear2 = l
        net.bn = nn.BatchNorm2d(2)
        net.block = block
        net.add_module('empty', None)

        state_dict = net.state_dict()
        self.assertEqual(len(state_dict), 10)
        self.assertEqual(len(state_dict._metadata), 6)
        self.assertIn('', state_dict._metadata)
        self.assertIn('linear1', state_dict._metadata)
        self.assertIn('linear1.weight', state_dict)
        self.assertIn('linear1.bias', state_dict)
        self.assertIn('linear2', state_dict._metadata)
        self.assertIn('linear2.weight', state_dict)
        self.assertIn('linear2.bias', state_dict)
        self.assertIn('block', state_dict._metadata)
        self.assertIn('block.conv', state_dict._metadata)
        self.assertIn('block.conv.weight', state_dict)
        self.assertIn('block.conv.weight', state_dict)
        self.assertNotIn('block.conv.bias', state_dict)
        self.assertIn('bn', state_dict._metadata)
        self.assertIn('bn.weight', state_dict)
        self.assertIn('bn.bias', state_dict)
        self.assertIn('bn.running_var', state_dict)
        self.assertIn('bn.running_mean', state_dict)
        self.assertIn('bn.num_batches_tracked', state_dict)
        self.assertFalse(any(k.startswith('empty') for k in state_dict))
        for k, v in state_dict.items():
            param = net
            for component in k.split('.'):
                param = getattr(param, component)
                if isinstance(param, Parameter):
                    param = param.data
            self.assertEqual(v.data_ptr(), param.data_ptr())

        l = nn.Linear(5, 5)
        state_dict = l.state_dict()
        self.assertEqual(len(state_dict), 2)
        self.assertEqual(len(state_dict._metadata), 1)
        self.assertIn('', state_dict._metadata)
        self.assertTrue(state_dict._metadata['']['version'] >= 0)
        self.assertEqual(state_dict['weight'].data_ptr(), l.weight.data_ptr())
        self.assertEqual(state_dict['bias'].data_ptr(), l.bias.data_ptr())

        # Reference https://github.com/pytorch/pytorch/pull/75507#issuecomment-1110291545
        self.assertNotWarn(lambda: l.state_dict(destination={}), "Should not warn kwarg destination w/o _metadata")

    def test_extra_state(self):

        class SubModule(torch.nn.Module):
            def __init__(self, foo):
                super().__init__()
                self.foo = foo

            def get_extra_state(self):
                return {
                    'foo': self.foo
                }

            def set_extra_state(self, state):
                self.foo = state['foo']

        class MyModule(torch.nn.Module):
            def __init__(self, foo, bar):
                super().__init__()
                self.sub = SubModule(foo)
                self.bar = bar

            def get_extra_state(self):
                return {
                    'bar': self.bar
                }

            def set_extra_state(self, state):
                self.bar = state['bar']

        # Ensure state_dict contains the extra state by loading it into another module.
        m = MyModule(3, 'something')
        m2 = MyModule(5, 'something else')
        m2.load_state_dict(m.state_dict())
        self.assertEqual(m.state_dict(), m2.state_dict())
        self.assertEqual(m2.bar, m.bar)
        self.assertEqual(m2.sub.foo, m.sub.foo)

    def test_extra_state_non_dict(self):

        class MyModule(torch.nn.Module):
            def __init__(self, foo):
                super().__init__()
                self.foo = foo

            def get_extra_state(self):
                return self.foo

            def set_extra_state(self, state):
                self.foo = state

        # Test various types of extra state.
        for state in ('something', 5, MyModule(3)):
            m = MyModule(state)
            m2 = MyModule('something else')
            m2.load_state_dict(m.state_dict())
            self.assertEqual(m.state_dict(), m2.state_dict())
            self.assertEqual(m.foo, m2.foo)

    def test_extra_state_missing_set_extra_state(self):

        class MyModule(torch.nn.Module):
            def get_extra_state(self):
                return {
                    'foo': 5
                }

        m = MyModule()
        with self.assertRaisesRegex(RuntimeError, 'Unexpected key'):
            m.load_state_dict(m.state_dict())

    def test_extra_state_missing_get_extra_state(self):

        class MyModule(torch.nn.Module):
            def set_extra_state(self):
                pass

        m = MyModule()
        with self.assertRaisesRegex(RuntimeError, 'Missing key'):
            m.load_state_dict(m.state_dict())

    @skipIfTorchDynamo("TorchDynamo fails here for unknown reasons")
    def test_parameter_assignment(self):
        l = nn.Linear(5, 5)

        def num_params():
            return len(list(l.parameters()))

        self.assertEqual(num_params(), 2)

        new_param = Parameter(torch.randn(5, 5))
        l.param_name = new_param
        self.assertEqual(num_params(), 3)
        self.assertObjectIn(new_param, l.parameters())

        var = torch.randn(5, 5)
        l.var_name = var
        self.assertEqual(num

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 40 class(es): TestNN, MyMixin, MyModuleWithMixinBefore, MyModuleWithMixinAfter, Net, M1, M2, M, MyBadModule, Net, SubclassNet, Net, Net, TensorLike, MyModule, Sub, SubSub, Model, SubModule, MyModule

### Functions
This file defines 752 function(s): _forward, _backward, _forward_criterion, _backward_criterion, _zero_grad_parameters, _get_parameters, test_parse_to, test_requires_grad_, test_module_backcompat, test_module_super_init, __init__, test_share_memory, __init__, forward, test_to, test_zero_grad, test_no_grad, test_parameters_and_named_parameters, names, test_named_parameters_remove_duplicate, names, __init__, __init__, test_buffers_and_named_buffers, names, __init__, test_buffer_bad_module_subclass, __init__, register_buffer, test_call_supports_python_dict_output


## Key Components

The file contains 48512 words across 13828 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 648914 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
