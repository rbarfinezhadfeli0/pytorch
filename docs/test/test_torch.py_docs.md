# Documentation: test_torch.py

## File Metadata
- **Path**: `test/test_torch.py`
- **Size**: 473746 bytes
- **Lines**: 10799
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
# Owner(s): ["module: tests"]

import torch
import torch.utils.data
import numpy as np

import contextlib
import gc
import io
import inspect
import itertools
import math
import random
import re
import copy
import os
import tempfile
import unittest
import warnings
import types
import pickle
import textwrap
import subprocess
import weakref
import sys
import copyreg
from torch import inf, nan
from itertools import product, combinations, permutations, chain
from functools import partial
from torch import multiprocessing as mp
from torch.testing import make_tensor
from torch.testing._internal.common_optimizers import (
    optim_db, optims, _get_optim_inputs_including_global_cliquey_kwargs)

from torch.testing._internal.common_utils import (  # type: ignore[attr-defined]
    MI300_ARCH, TEST_WITH_TORCHINDUCTOR, TEST_WITH_ROCM, run_tests, IS_JETSON,
    IS_FILESYSTEM_UTF8_ENCODING,
    IS_SANDCASTLE, IS_FBCODE, IS_REMOTE_GPU, skipIfRocmArch, skipIfTorchInductor, load_tests, slowTest, slowTestIf,
    skipIfCrossRef, TEST_WITH_CROSSREF, skipIfTorchDynamo, skipRocmIfTorchInductor, set_default_dtype,
    skipCUDAMemoryLeakCheckIf, BytesIOContext,
    skipIfRocm, skipIfNoSciPy, TemporaryFileName, TemporaryDirectoryName,
    wrapDeterministicFlagAPITest, DeterministicGuard, CudaSyncGuard,
    bytes_to_scalar, parametrize, skipIfMPS, noncontiguous_like,
    AlwaysWarnTypedStorageRemoval, TEST_WITH_TORCHDYNAMO, xfailIfTorchDynamo,
    xfailIfS390X, set_warn_always_context)
from multiprocessing.reduction import ForkingPickler
from torch.testing._internal.common_device_type import (
    expectedFailureMeta,
    expectedFailureXLA,
    instantiate_device_type_tests,
    onlyCUDA, onlyCPU,
    dtypes, dtypesIfCUDA, dtypesIfCPU, deviceCountAtLeast,
    skipMeta, PYTORCH_CUDA_MEMCHECK, largeTensorTest, onlyNativeDeviceTypes, skipCUDAIfNotRocm,
    get_all_device_types, skipXLA)
import torch.backends.quantized
import torch.testing._internal.data
from torch.testing._internal.common_cuda import (
    tf32_on_and_off, TEST_CUDNN, TEST_MULTIGPU,
    _create_scaling_case, _create_scaling_models_optimizers)
from torch.testing._internal.common_mkldnn import reduced_f32_on_and_off
from torch.testing._internal.common_dtype import (
    floating_types_and, get_all_math_dtypes, all_types_and_complex_and, complex_types,
    all_types_and, floating_types, floating_and_complex_types, integral_types_and,
    get_all_qint_dtypes, all_types_complex_float8_and,
)
from torch.testing._internal.two_tensor import TwoTensor
from torch.testing._internal.common_utils import IS_WINDOWS

if TEST_WITH_TORCHINDUCTOR:
    from torch._inductor.test_case import TestCase
else:
    from torch.testing._internal.common_utils import TestCase  # type: ignore[assignment]


# Protects against includes accidentally setting the default dtype
assert torch.get_default_dtype() is torch.float32

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

AMPERE_OR_ROCM = TEST_WITH_ROCM or torch.cuda.is_tf32_supported()

@contextlib.contextmanager
def torch_vital_set(value):
    stash = None
    if 'TORCH_VITAL' in os.environ:
        stash = os.environ['TORCH_VITAL']
    os.environ['TORCH_VITAL'] = value
    try:
        yield
    finally:
        if stash:
            os.environ['TORCH_VITAL'] = stash
        else:
            del os.environ['TORCH_VITAL']

# Tests Vital Signs for Torch
# FIXME: document or deprecate whatever this is
class TestBasicVitalSigns(TestCase):
    def test_basic_vitals(self):
        with torch_vital_set(''):
            self.assertFalse(torch.vitals_enabled())
        with torch_vital_set('ON'):
            self.assertTrue(torch.vitals_enabled())

    def test_basic_vitals_read_write(self):
        with torch_vital_set('ON'):
            self.assertTrue(torch.vitals_enabled())
            # This tests the code path of setting a vital
            self.assertTrue(torch.set_vital('Dataloader', 'basic_unit_test', 'TEST_VALUE_STRING'))
            self.assertIn('TEST_VALUE_STRING', torch.read_vitals())
            self.assertIn('CUDA.used', torch.read_vitals())

    def test_dataloader_vitals(self):
        with torch_vital_set('ON'):
            inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
            tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
            dataset = torch.utils.data.TensorDataset(inps, tgts)
            torch.utils.data.DataLoader(dataset, batch_size=2)
            self.assertIn('Dataloader.enabled\t\t True', torch.read_vitals())

# FIXME: document or deprecate whatever this is
class TestVitalSignsCuda(TestCase):
    @onlyCUDA
    def test_cuda_vitals_gpu_only(self, device):
        with torch_vital_set('ON'):
            self.assertIn('CUDA.used\t\t true', torch.read_vitals())


is_cuda_sm86 = torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (8, 6)

class TestTorchDeviceType(TestCase):
    exact_dtype = True

    # TODO: move all tensor creation to common ops
    def _rand_shape(self, dim, min_size, max_size):
        shape = []
        for _ in range(dim):
            shape.append(random.randint(min_size, max_size))
        return tuple(shape)

    # Validates that mathematical constants are defined properly, as required by
    # the Python Array API (https://data-apis.org/array-api/latest/API_specification/constants.html)
    @onlyCPU
    def test_constants(self, device):
        self.assertIsInstance(torch.e, float)
        self.assertEqual(torch.e, math.e, atol=0, rtol=0)

        self.assertIsInstance(torch.pi, float)
        self.assertEqual(torch.pi, math.pi, atol=0, rtol=0)

        self.assertIsInstance(torch.nan, float)
        self.assertEqual(torch.nan, math.nan, equal_nan=True)

        self.assertIsInstance(torch.inf, float)
        self.assertEqual(torch.inf, math.inf)

    @onlyNativeDeviceTypes
    @slowTestIf(IS_WINDOWS)
    @dtypes(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64,
            torch.bool, torch.float32, torch.complex64, torch.float64,
            torch.complex128, torch.uint16, torch.uint32, torch.uint64)
    def test_bytes_to_scalar(self, device, dtype):
        def rand_byte():
            if dtype == torch.bool:
                return torch.randint(0, 2, ()).item()
            else:
                return torch.randint(0, 256, ()).item()

        element_size = torch._utils._element_size(dtype)

        for _ in range(10):
            bytes_list = [rand_byte() for _ in range(element_size)]
            scalar = bytes_to_scalar(bytes_list, dtype, device)
            self.assertEqual(scalar.storage().untyped().tolist(), bytes_list)

    # For testing in64 support in upsample_nearest3d
    @onlyCUDA
    @largeTensorTest('56GB', device='cuda')
    @dtypes(torch.bfloat16)
    @unittest.skipIf(IS_JETSON, "Large tensor tests are too large for Jetson.")
    def test_int64_upsample3d(self, device, dtype):
        x = torch.ones((1, 256, 16, 720, 1280), dtype=dtype, device=device)
        try:
            torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    @dtypes(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64,
            torch.bool, torch.float32, torch.complex64, torch.float64,
            torch.complex128, torch.uint16, torch.uint32, torch.uint64)
    @slowTestIf(IS_WINDOWS)
    def test_storage(self, device, dtype):
        v = make_tensor((3, 5), dtype=dtype, device=device, low=-9, high=9)
        self.assertEqual(v.storage()[0], v[0][0])
        self.assertEqual(v.storage()[14], v[2][4])
        v_s = v.storage()

        for el_num in range(v.numel()):
            dim0 = el_num // v.size(1)
            dim1 = el_num % v.size(1)
            self.assertEqual(
                v_s[el_num],
                v[dim0][dim1])

        v_s_byte = v.storage().untyped()
        el_size = v.element_size()

        for el_num in range(v.numel()):
            start = el_num * el_size
            end = start + el_size
            dim0 = el_num // v.size(1)
            dim1 = el_num % v.size(1)
            self.assertEqual(
                bytes_to_scalar(v_s_byte[start:end], dtype, device),
                v[dim0][dim1])

    @onlyNativeDeviceTypes
    @dtypes(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64,
            torch.bool, torch.float32, torch.complex64, torch.float64,
            torch.complex128, torch.quint8, torch.qint8, torch.qint32,
            torch.quint4x2)
    @slowTestIf(IS_WINDOWS)
    def test_storage_setitem(self, device, dtype):
        # Skip quantized dtypes for CUDA, since they're not supported
        if torch.device(device).type == 'cuda':
            if dtype in [torch.quint8, torch.qint8, torch.qint32, torch.quint4x2]:
                return

        storage_type_name = torch.storage._dtype_to_storage_type_map()[dtype]
        if torch.device(device).type == 'cuda':
            storage_type = eval('torch.cuda.' + storage_type_name)
        else:
            storage_type = eval('torch.' + storage_type_name)

        N = 10

        s = storage_type(N)
        s[:] = 0
        l = [0] * N
        self.assertEqual(s, storage_type(l))

        for i in range(N):
            s[i] = i
            l[i] = i

        self.assertEqual(s, storage_type(l))

        l[2:7] = [1] * 5
        s[2:7] = 1
        self.assertEqual(s, storage_type(l))

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    @onlyNativeDeviceTypes
    @slowTestIf(IS_WINDOWS)
    def test_storage_use_count(self, device):
        a = torch.randn(10, device=device)
        prev_cf = torch._C._storage_Use_Count(a.untyped_storage()._cdata)
        self.assertEqual(prev_cf, 1)
        b = a.view(2, 5)
        self.assertEqual(torch._C._storage_Use_Count(b.untyped_storage()._cdata), prev_cf + 1)

    @xfailIfTorchDynamo
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @slowTestIf(IS_WINDOWS)
    def test_tensor_storage_type(self, device, dtype):
        a = make_tensor((10,), dtype=dtype, device=device, low=-9, high=9)

        module = torch.cuda if (torch.device(device).type == 'cuda') else torch
        expected_storage_type = getattr(module, torch.storage._dtype_to_storage_type_map()[dtype])

        self.assertEqual(a.storage_type(), expected_storage_type)

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16, torch.uint16, torch.uint32, torch.uint64))
    @slowTestIf(IS_WINDOWS)
    def test_tensor_from_storage(self, device, dtype):
        a = make_tensor((4, 5, 3), dtype=dtype, device=device, low=-9, high=9)
        a_s = a.storage()
        b = torch.tensor(a_s, device=device, dtype=dtype).reshape(a.size())
        self.assertEqual(a, b)
        c = torch.tensor(a_s.untyped(), device=device, dtype=dtype).reshape(a.size())
        self.assertEqual(a, c)

        for error_dtype in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16):
            if error_dtype == dtype:
                continue
            with self.assertRaisesRegex(RuntimeError, r'Expected a Storage of type'):
                error_storage = a.to(error_dtype).storage()
                torch.tensor(error_storage, device=device, dtype=dtype)

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @slowTestIf(IS_WINDOWS)
    def test_set_storage(self, device, dtype):
        a = make_tensor((4, 5, 3), dtype=dtype, device=device, low=-9, high=9)
        a_s = a.storage()
        b = torch.tensor([], device=device, dtype=dtype).set_(a_s).reshape(a.size())
        self.assertEqual(a, b)
        c = torch.tensor([], device=device, dtype=dtype).set_(a_s.untyped()).reshape(a.size())
        self.assertEqual(a, c)

        for error_dtype in all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16):
            if error_dtype == dtype:
                continue
            with self.assertRaisesRegex(RuntimeError, r'Expected a Storage of type'):
                error_storage = a.to(error_dtype).storage()
                b = torch.tensor([], device=device, dtype=dtype).set_(error_storage)

    def _check_storage_meta(self, s, s_check):
        self.assertTrue(
            isinstance(s, (torch.UntypedStorage, torch.TypedStorage)) and
            isinstance(s_check, type(s)),
            (
                's and s_check must both be one of UntypedStorage or '
                'TypedStorage, but got'
                f' {type(s).__name__} and {type(s_check).__name__}'))

        self.assertEqual(s.device.type, 'meta')
        self.assertEqual(s.nbytes(), s_check.nbytes())
        self.assertEqual(s.size(), s_check.size())
        self.assertEqual(s.data_ptr(), 0)

        with self.assertRaisesRegex(NotImplementedError, r'Not available'):
            s[0]

        if isinstance(s, torch.TypedStorage):
            self.assertEqual(s.dtype, s_check.dtype)
            self._check_storage_meta(s.untyped(), s_check.untyped())

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @slowTestIf(IS_WINDOWS)
    def test_typed_storage_meta(self, device, dtype):
        args_list = [
            [],
            [0],
            [100],
            [[1, 2, 3, 4, 5, 6]],
        ]
        for args in args_list:
            s_check = torch.TypedStorage(*args, dtype=dtype, device=device)
            s = torch.TypedStorage(*args, dtype=dtype, device='meta')
            self._check_storage_meta(s, s_check)

    @onlyNativeDeviceTypes
    @slowTestIf(IS_WINDOWS)
    def test_untyped_storage_meta(self, device):
        args_list = [
            [],
            [0],
            [100],
            [[1, 2, 3, 4, 5, 6]],
        ]
        for args in args_list:
            s_check = torch.UntypedStorage(*args, device=device)
            s = torch.UntypedStorage(*args, device='meta')
            self._check_storage_meta(s, s_check)

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @slowTestIf(IS_WINDOWS)
    def test_storage_meta_from_tensor(self, device, dtype):
        t_check = make_tensor((4, 5, 3), dtype=dtype, device=device, low=-9, high=9)
        t = t_check.to('meta')

        s_check = t_check.storage()
        s = t.storage()
        self._check_storage_meta(s, s_check)

    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @slowTestIf(IS_WINDOWS)
    def test_storage_meta_errors(self, device, dtype):
        s0 = torch.TypedStorage([1, 2, 3, 4], device='meta', dtype=dtype)

        with self.assertRaisesRegex(NotImplementedError, r'Cannot copy out'):
            s0.cpu()

        with self.assertRaisesRegex(RuntimeError, r'only available on CPU'):
            s0._share_fd_cpu_()

        with self.assertRaisesRegex(RuntimeError, r'only available on CPU'):
            s0._share_filename_cpu_()

        if torch.cuda.is_available():
            with self.assertRaisesRegex(NotImplementedError, r'Cannot copy out'):
                s0.cuda()

            with self.assertRaisesRegex(RuntimeError, r'only available on CUDA'):
                s0._share_cuda_()

            with self.assertRaisesRegex(TypeError, r"cannot pin 'torch.storage.UntypedStorage' only CPU memory can be pinned"):
                s0.pin_memory()

        with self.assertRaisesRegex(RuntimeError, r'only available on CPU'):
            s0.share_memory_()

        with self.assertRaisesRegex(NotImplementedError, r'Not available'):
            s0.tolist()

        with tempfile.NamedTemporaryFile() as f:
            with self.assertRaisesRegex(NotImplementedError, r'Cannot copy out'):
                s0._write_file(f, True, True, s0.element_size())

        for device in ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']:
            s1 = torch.TypedStorage([1, 2, 3, 4], device=device, dtype=dtype)

            with self.assertRaisesRegex(NotImplementedError, r'Cannot copy out'):
                s1.copy_(s0)

    @onlyCPU
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @slowTestIf(IS_WINDOWS)
    def test_storage_meta_ok(self, device, dtype):
        s0 = torch.TypedStorage([1, 2, 3, 4], device='meta', dtype=dtype)

        # This is OK, it changes the meta storage size without allocating
        s0.resize_(10)

    @onlyCUDA
    def test_module_share_memory(self):
        # Test fix for issue #80733
        # See https://github.com/pytorch/pytorch/issues/80733
        model = torch.nn.Linear(3, 1)
        _model_cuda = model.to('cuda')
        model.share_memory()

    @dtypes(torch.float32, torch.complex64)
    @slowTestIf(IS_WINDOWS)
    def test_deepcopy(self, device, dtype):
        from copy import deepcopy
        a = torch.randn(5, 5, dtype=dtype, device=device)
        b = torch.randn(5, 5, dtype=dtype, device=device)
        c = a.view(25)
        q = [a, [a.storage(), b.storage()], b, c]
        w = deepcopy(q)
        self.assertEqual(w[0], q[0], atol=0, rtol=0)
        self.assertEqual(w[1][0], q[1][0], atol=0, rtol=0)
        self.assertEqual(w[1][1], q[1][1], atol=0, rtol=0)
        self.assertEqual(w[1], q[1], atol=0, rtol=0)
        self.assertEqual(w[2], q[2], atol=0, rtol=0)

        # Check that deepcopy preserves sharing
        w[0].add_(1)
        for i in range(a.numel()):
            self.assertEqual(w[1][0][i], q[1][0][i] + 1)
        self.assertEqual(w[3], c + 1)
        w[2].sub_(1)
        for i in range(a.numel()):
            self.assertEqual(w[1][1][i], q[1][1][i] - 1)

        # Check that deepcopy preserves attributes
        a.foo = 3
        self.assertEqual(deepcopy(a).foo, 3)

    @dtypes(torch.float32, torch.complex64)
    @slowTestIf(IS_WINDOWS)
    def test_deepcopy_scalar(self, device, dtype):
        from copy import deepcopy
        a = torch.tensor(5, dtype=dtype, device=device)
        self.assertEqual(a.size(), deepcopy(a).size())
        self.assertEqual(a, deepcopy(a))

    def check_internal_mem_overlap(self, inplace_op, num_inputs,
                                   dtype, device,
                                   expected_failure=False):
        if isinstance(inplace_op, str):
            inplace_op = getattr(torch.Tensor, inplace_op)
        input = torch.randn(1, dtype=dtype, device=device).expand(3, 3)
        inputs = [input] + [torch.randn_like(input)
                            for i in range(num_inputs - 1)]
        if not expected_failure:
            with self.assertRaisesRegex(RuntimeError, 'single memory location'):
                inplace_op(*inputs)
        else:
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, 'single memory location'):
                    inplace_op(*inputs)

    def unary_check_input_output_mem_overlap(self, data, sz, op,
                                             expected_failure=False):

        def _test(op, output, input):
            output_exp = torch.empty_like(output)
            op(input, out=output_exp)
            self.assertEqual(op(input, out=output), output_exp, msg=op.__name__)

        # output is identical to input:
        _test(op, output=data[0:sz], input=data[0:sz])
        # output and input are independent:
        _test(op, output=data[0:sz], input=data[sz:2 * sz])
        # output partially overlaps with input:
        if not expected_failure:
            with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                _test(op, data[0:sz], data[1:sz + 1])
        else:
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                    _test(op, data[0:sz], data[1:sz + 1])
        # output is transpose of input:
        length = int(math.sqrt(sz))
        input = data[:length**2].view([length, length])
        out = input.t()
        if not expected_failure:
            with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                _test(op, out, input)
        else:
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                    _test(op, out, input)

    def ternary_check_input_output_mem_overlap(self, op, device,
                                               expected_failure=False):
        sz = 9
        data = torch.randn(2 * sz, device=device)
        other1 = torch.randn(sz, device=device)
        other2 = torch.randn(sz, device=device)

        self.unary_check_input_output_mem_overlap(
            data, sz, lambda input, out:
                op(input, other1.view(input.shape), other2.view(input.shape), out=out),
            expected_failure=expected_failure)

        self.unary_check_input_output_mem_overlap(
            data, sz, lambda input, out:
                op(other1.view(input.shape), input, other2.view(input.shape), out=out),
            expected_failure=expected_failure)

        self.unary_check_input_output_mem_overlap(
            data, sz, lambda input, out:
                op(other1.view(input.shape), other2.view(input.shape), input, out=out),
            expected_failure=expected_failure)

    def _select_broadcastable_dims(self, dims_full=None):
        # select full dimensionality
        if dims_full is None:
            dims_full = []
            ndims = random.randint(1, 4)
            dims_full = [random.randint(1, 8) for _ in range(ndims)]
        else:
            ndims = len(dims_full)

        # select actual dimensions for ops:
        # larger: full ndims, individual sizes may be reduced
        # smaller: possibly reduced ndims, sizes may be reduced
        smaller_ndims = random.randint(1, ndims)
        dims_small = []
        dims_large = []
        for i in range(ndims - 1, -1, -1):
            j = random.randint(1, 3)
            if j == 1:  # no reduced singleton dimension
                ds = dims_full[i]
                dl = dims_full[i]
            elif j == 2:  # larger may have reduced singleton dimension
                ds = dims_full[i]
                dl = 1 if len(dims_small) < smaller_ndims else dims_full[i]
            elif j == 3:  # smaller may have reduced singleton dimension
                ds = 1
                dl = dims_full[i]
            dims_large = [dl] + dims_large
            if len(dims_small) < smaller_ndims:
                dims_small = [ds] + dims_small
        return (dims_small, dims_large, dims_full)

    # collected tests of ops that used scalar_check in Declarations.cwrap for
    # correctness
    def test_scalar_check(self, device):
        zero_d = torch.randn((), device=device)
        one_d = torch.randn((1,), device=device)

        # remainder
        self.assertEqual((), torch.remainder(zero_d, zero_d).shape)
        self.assertEqual((), torch.remainder(zero_d, 2).shape)
        self.assertEqual((1,), torch.remainder(zero_d, one_d).shape)
        self.assertEqual((1,), torch.remainder(one_d, zero_d).shape)

        # fmod
        self.assertEqual((), torch.fmod(zero_d, zero_d).shape)
        self.assertEqual((), torch.fmod(zero_d, 2).shape)
        self.assertEqual((1,), torch.fmod(zero_d, one_d).shape)
        self.assertEqual((1,), torch.fmod(one_d, zero_d).shape)

        # exp, cos, cosh, tan, atan, tanh, erf, erfc, reciprocal
        self.assertEqual((), torch.exp(zero_d).shape)
        self.assertEqual((), torch.cos(zero_d).shape)
        self.assertEqual((), torch.cosh(zero_d).shape)
        self.assertEqual((), torch.tan(zero_d).shape)
        self.assertEqual((), torch.atan(zero_d).shape)
        self.assertEqual((), torch.acosh(zero_d).shape)
        self.assertEqual((), torch.asinh(zero_d).shape)
        self.assertEqual((), torch.atanh(zero_d).shape)
        self.assertEqual((), torch.tanh(zero_d).shape)
        self.assertEqual((), torch.erf(zero_d).shape)
        self.assertEqual((), torch.erfc(zero_d).shape)
        self.assertEqual((), torch.reciprocal(zero_d).shape)
        self.assertEqual((1,), torch.exp(one_d).shape)
        self.assertEqual((1,), torch.cos(one_d).shape)
        self.assertEqual((1,), torch.cosh(one_d).shape)
        self.assertEqual((1,), torch.tan(one_d).shape)
        self.assertEqual((1,), torch.atan(one_d).shape)
        self.assertEqual((1,), torch.acosh(one_d).shape)
        self.assertEqual((1,), torch.asinh(one_d).shape)
        self.assertEqual((1,), torch.atanh(one_d).shape)
        self.assertEqual((1,), torch.tanh(one_d).shape)
        self.assertEqual((1,), torch.erf(one_d).shape)
        self.assertEqual((1,), torch.erfc(one_d).shape)
        self.assertEqual((1,), torch.reciprocal(one_d).shape)

        # clamp
        self.assertEqual((), torch.clamp(zero_d, min=0, max=1).shape)
        self.assertEqual((), torch.clamp(zero_d, min=0).shape)
        self.assertEqual((), torch.clamp(zero_d, max=1).shape)
        self.assertEqual((1,), torch.clamp(one_d, min=0, max=1).shape)
        self.assertEqual((1,), torch.clamp(one_d, min=0).shape)
        self.assertEqual((1,), torch.clamp(one_d, max=1).shape)

        # cumsum, cumprod, cummax, cummin
        self.assertEqual((), torch.logcumsumexp(zero_d, 0).shape)
        self.assertEqual((), torch.cumsum(zero_d, 0).shape)
        self.assertEqual((), torch.cumprod(zero_d, 0).shape)
        self.assertEqual((), torch.cummax(zero_d, 0)[0].shape)
        self.assertEqual((), torch.cummin(zero_d, 0)[0].shape)

        # sort, topk
        self.assertEqual([(), ()], [x.shape for x in torch.sort(zero_d, 0, False)])
        self.assertEqual([(), ()], [x.shape for x in torch.sort(zero_d, 0, True)])
        self.assertEqual([(), ()], [x.shape for x in torch.topk(zero_d, 1, 0, False)])
        self.assertEqual([(), ()], [x.shape for x in torch.topk(zero_d, 1, 0, True)])

        # max, min
        self.assertEqual((), torch.max(zero_d, zero_d).shape)
        self.assertEqual((1,), torch.max(one_d, zero_d).shape)
        self.assertEqual((1,), torch.max(zero_d, one_d).shape)
        self.assertEqual((), torch.min(zero_d, zero_d).shape)
        self.assertEqual((1,), torch.min(one_d, zero_d).shape)
        self.assertEqual((1,), torch.min(zero_d, one_d).shape)

        zero_d_int = torch.tensor(1, device=device)
        one_d_int = torch.tensor([1], device=device)

        # lshift, rshift
        self.assertEqual((), (zero_d_int >> zero_d_int).shape)
        self.assertEqual((), (zero_d_int >> 1).shape)
        self.assertEqual((1,), (one_d_int >> zero_d_int).shape)
        self.assertEqual((1,), (zero_d_int >> one_d_int).shape)
        self.assertEqual((1,), (one_d_int >> 1).shape)

        self.assertEqual((), (zero_d_int << zero_d_int).shape)
        self.assertEqual((), (zero_d_int << 1).shape)
        self.assertEqual((1,), (one_d_int << zero_d_int).shape)
        self.assertEqual((1,), (zero_d_int << one_d_int).shape)
        self.assertEqual((1,), (one_d_int << 1).shape)

        # or
        self.assertEqual((), (zero_d_int | zero_d_int).shape)
        self.assertEqual((), (zero_d_int | 1).shape)
        self.assertEqual((1,), (one_d_int | zero_d_int).shape)
        self.assertEqual((1,), (zero_d_int | one_d_int).shape)
        self.assertEqual((1,), (one_d_int | 1).shape)

        # and
        self.assertEqual((), (zero_d_int & zero_d_int).shape)
        self.assertEqual((), (zero_d_int & 1).shape)
        self.assertEqual((1,), (one_d_int & zero_d_int).shape)
        self.assertEqual((1,), (zero_d_int & one_d_int).shape)
        self.assertEqual((1,), (one_d_int & 1).shape)

        # clone
        self.assertEqual((), zero_d.clone().shape)

        zero_d_bool = torch.tensor(True, device=device)
        one_d_bool = torch.tensor([True], device=device)

        # masked_select
        self.assertEqual((1,), torch.masked_select(zero_d_bool, zero_d_bool).shape)
        self.assertEqual((1,), torch.masked_select(zero_d_bool, one_d_bool).shape)
        self.assertEqual((1,), torch.masked_select(one_d_bool, zero_d_bool).shape)

        torch.tensor(1, dtype=torch.uint8, device=device)
        torch.tensor([1], dtype=torch.uint8, device=device)

        # mode
        self.assertEqual([(), ()], [x.shape for x in torch.mode(zero_d, dim=0, keepdim=True)])
        self.assertEqual([(), ()], [x.shape for x in torch.mode(zero_d, dim=0, keepdim=False)])
        self.assertEqual([(1,), (1,)], [x.shape for x in torch.mode(one_d, dim=0, keepdim=True)])
        self.assertEqual([(), ()], [x.shape for x in torch.mode(one_d, dim=0, keepdim=False)])

        # max
        self.assertEqual([(), ()], [x.shape for x in torch.max(zero_d, dim=0, keepdim=True)])
        self.assertEqual([(), ()], [x.shape for x in torch.max(zero_d, dim=0, keepdim=False)])
        self.assertEqual([(1,), (1,)], [x.shape for x in torch.max(one_d, dim=0, keepdim=True)])
        self.assertEqual([(), ()], [x.shape for x in torch.max(one_d, dim=0, keepdim=False)])

        # amax
        self.assertEqual((), torch.amax(zero_d, dim=0, keepdim=True).shape)
        self.assertEqual((), torch.amax(zero_d, dim=0, keepdim=False).shape)
        self.assertEqual((1,), torch.amax(one_d, dim=0, keepdim=True).shape)
        self.assertEqual((), torch.amax(one_d, dim=0, keepdim=False).shape)

        # min
        self.assertEqual([(), ()], [x.shape for x in torch.min(zero_d, dim=0, keepdim=True)])
        self.assertEqual([(), ()], [x.shape for x in torch.min(zero_d, dim=0, keepdim=False)])
        self.assertEqual([(1,), (1,)], [x.shape for x in torch.min(one_d, dim=0, keepdim=True)])
        self.assertEqual([(), ()], [x.shape for x in torch.min(one_d, dim=0, keepdim=False)])

        # amin
        self.assertEqual((), torch.amin(zero_d, dim=0, keepdim=True).shape)
        self.assertEqual((), torch.amin(zero_d, dim=0, keepdim=False).shape)
        self.assertEqual((1,), torch.amin(one_d, dim=0, keepdim=True).shape)
        self.assertEqual((), torch.amin(one_d, dim=0, keepdim=False).shape)

        # set_
        zero_d_clone = zero_d.clone()
        one_d_clone = one_d.clone()
        self.assertEqual((), zero_d_clone.set_(one_d.storage(), 0, (), ()).shape)
        self.assertEqual((1,), zero_d_clone.set_(one_d.storage(), 0, (1,), (1,)).shape)
        self.assertEqual((), one_d_clone.set_(one_d.storage(), 0, (), ()).shape)
        self.assertEqual((1,), one_d_clone.set_(one_d.storage(), 0, (1,), (1,)).shape)

        self.assertEqual((), zero_d.clone().set_(zero_d).shape)
        self.assertEqual((), one_d.clone().set_(zero_d).shape)
        self.assertEqual((1,), zero_d.clone().set_(one_d).shape)
        self.assertEqual((1,), one_d.clone().set_(one_d).shape)

        # take
        self.assertEqual((), torch.randn((2, 3), device=device).take(zero_d_int).shape)
        self.assertEqual((1,), torch.randn((2, 3), device=device).take(one_d_int).shape)

        # gather
        self.assertEqual((), torch.gather(zero_d, 0, torch.zeros((), dtype=torch.int64, device=device)).shape)
        self.assertEqual((1,), torch.gather(zero_d, 0, torch.zeros((1,), dtype=torch.int64, device=device)).shape)
        self.assertEqual((), torch.gather(one_d, 0, torch.zeros((), dtype=torch.int64, device=device)).shape)
        self.assertEqual((1,), torch.gather(one_d, 0, torch.zeros((1,), dtype=torch.int64, device=device)).shape)

        # normal
        # std must be >= 0
        zero_d_ge_0 = torch.rand((), device=device)
        # documentation says out shape matches shape of mean
        self.assertEqual((), torch.normal(zero_d, zero_d_ge_0).shape)
        self.assertEqual((1,), torch.normal(one_d, zero_d_ge_0).shape)
        self.assertEqual((), torch.normal(1, zero_d_ge_0).shape)
        self.assertEqual((), torch.normal(zero_d, 1).shape)
        self.assertEqual((1,), torch.normal(one_d, 1).shape)
        # TODO: this behavior differs on CPU and GPU, see https://github.com/pytorch/pytorch/issues/30480.
        # self.assertEqual((), torch.normal(zero_d, one_d).shape)
        # self.assertEqual((), torch.normal(1, one_d).shape)

        # convolutions.  Yes, we are testing nn.functional here; seems justified
        # given its similar to the other tests
        w = torch.randn(2, 1, 3, 3, device=device).div_(2).requires_grad_()
        self.assertRaises(RuntimeError, lambda: torch.nn.functional.conv2d(zero_d, w, groups=1))
        self.assertRaises(RuntimeError, lambda: torch.nn.functional.conv2d(zero_d, w, groups=2))

        # nll_loss -- verify input can't be 0-dimensional.
        self.assertRaises(ValueError, lambda: torch.nn.functional.nll_loss(zero_d, zero_d, reduction='none'))
        self.assertRaises(ValueError, lambda: torch.nn.functional.nll_loss(zero_d, one_d, reduction='none'))
        # verify output is 0-dimensional when reduction != 'none'
        for (input, target) in ((torch.randn(1, 1, device=device), torch.tensor([0], device=device)),
                                (torch.randn(1, 1, 1, 1, device=device), torch.tensor([[[0]]], device=device))):
            self.assertEqual((), torch.nn.functional.nll_loss(input, target, reduction='mean').shape)
            self.assertEqual((), torch.nn.functional.nll_loss(input, target, reduction='sum').shape)

    # Test that `torch._check_tensor_all` raises errors in the correct cases
    def test_check_tensor_all(self, device):
        default_message = 'Expected cond to be True'
        check_fn = torch._check_tensor_all
        expected_error = RuntimeError

        # cond must be a tensor
        with self.assertRaisesRegex(TypeError, 'cond must be a tensor'):
            check_fn(True)

        # cond tensor must be boolean
        with self.assertRaisesRegex(TypeError, 'cond tensor must have dtype torch.bool'):
            check_fn(torch.ones(1, device=device))

        test_sizes = [
            (),
            (1,),
            (10,),
            (1, 1),
            (1, 10),
            (10, 1),
            (10, 10),
            (1, 1, 1),
            (10, 1, 1),
            (1, 10, 1),
            (10, 10, 10),
        ]
        for size in test_sizes:
            t_all_true = torch.ones(size, dtype=torch.bool, device=device)
            t_all_false = torch.zeros(size, dtype=torch.bool, device=device)

            # Should not raise error
            check_fn(t_all_true)

            with self.assertRaisesRegex(expected_error, default_message):
                check_fn(t_all_false)

            if t_all_true.numel() > 1:
                t_all_true_but_one = t_all_true.clone()
                # Choose a random element to set to false
                idx = (random.choice(range(dim_size)) for dim_size in size)
                t_all_true_but_one[(..., *idx)] = False

                with self.assertRaisesRegex(expected_error, default_message):
                    check_fn(t_all_true_but_one)

            # Test a simple failure message
            message = 'message'
            with self.assertRaisesRegex(expected_error, message):
                check_fn(t_all_false, lambda: message)

            # Test message with tensor
            def message():
                return torch.arange(4)

            with self.assertRaisesRegex(expected_error, re.escape(str(message()))):
                check_fn(t_all_false, message)

            # Test format string message
            def message():
                return f"{'test'} {[1, 2, 'a', True]} {True} {100} {torch.arange(4)}"

            with self.assertRaisesRegex(expected_error, re.escape(str(message()))):
                check_fn(t_all_false, message)

    # Test that `TORCH_CHECK_TENSOR_ALL` raises errors that propagate from C++ to Python
    def test_check_tensor_internal(self, device):
        test_sizes = [
            (),
            (1,),
            (10,),
            (1, 1),
            (1, 10),
            (10, 1),
            (10, 10),
            (1, 1, 1),
            (10, 1, 1),
            (1, 10, 1),
            (10, 10, 10),
        ]
        for size in test_sizes:
            t_all_true = torch.ones(size, dtype=torch.bool, device=device)
            t_all_false = torch.zeros(size, dtype=torch.bool, device=device)

            # Should not raise error
            torch._test_check_tensor(t_all_true)

            with self.assertRaisesRegex(RuntimeError, "Test message for TORCH_CHECK_TENSOR_ALL"):
                torch._test_check_tensor(t_all_false)

            if t_all_true.numel() > 1:
                t_all_true_but_one = t_all_true.clone()
                # Choose a random element to set to false
                idx = (random.choice(range(dim_size)) for dim_size in size)
                t_all_true_but_one[(..., *idx)] = False

                with self.assertRaisesRegex(RuntimeError, "Test message for TORCH_CHECK_TENSOR_ALL"):
                    torch._test_check_tensor(t_all_true_but_one)

    # Uses mismatched arange out size to trigger a warning
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    @unittest.skipIf(TEST_WITH_CROSSREF, "crossref perturbs line numbering")
    def test_cpp_warnings_have_python_context(self, device):
        # Creates long string in advance to avoid a too-long Python line
        s = ".+Triggered internally at.+RangeFactories.+"
        # nvfuser deprecation warning filter
        warnings.filterwarnings("ignore", "torch::jit::fuser::cuda", UserWarning)

        def cpp_warn_fn():
            out = torch.empty((5,))
            torch.arange(0, 3, out=out)
            return out

        # Checks eager-mode cpp warning
        with warnings.catch_warnings(record=True) as w:
            cpp_warn_fn()
            frameinfo = inspect.getframeinfo(inspect.currentframe())
            warning = w[0]

            # Checks for cpp context in the warning message
            escaped_warning_message = str(warning.message).encode('unicode_escape')
            self.assertTrue(re.search(s, repr(escaped_warning_message), re.IGNORECASE) is not None)

            # Checks the Python features of the warning
            # Note: the eager mode warning refers to the line in the function
            # that throws the warning.
            self.assertEqual(frameinfo.lineno - 6, warning.lineno)
            self.assertEqual(len(w), 1)

        # Checks jitted cpp warning
        with warnings.catch_warnings(record=True) as w:
            scripted_cpp_warn_fn = torch.jit.script(cpp_warn_fn)
            scripted_cpp_warn_fn()
            warning = w[0]

            # Checks for cpp context in the warning message
            escaped_warning_message = str(warning.message).encode('unicode_escape')
            self.assertTrue(re.search(s, repr(escaped_warning_message), re.IGNORECASE) is not None)

            # Checks the Python features of the warning
            # Note: the jitted warning's lineno refers to the call to the jitted
            # function, which in our test suite has a layer of indirection
            # that makes checking the Python lineno fragile
            self.assertEqual(len(w), 1)

        # Checks jitted Python warning
        def warn_fn():
            warnings.warn("Warning!")

        # The jit mimics an eager-mode Python warning in this case
        with warnings.catch_warnings(record=True) as w:
            scripted_warn_fn = torch.jit.script(warn_fn)
            scripted_warn_fn()
            frameinfo = inspect.getframeinfo(inspect.currentframe())
            warning = w[0]

            self.assertTrue(re.search('Warning!', str(warning.message)) is not None)

            # Checks the Python features of the warning
            self.assertEqual(frameinfo.lineno - 6, warning.lineno)
            self.assertEqual(len(w), 1)

    # FIXME: move to test_testing
    @onlyCPU
    def test_warn_always_caught(self, device):
        # Check that we can catch a TORCH_WARN_ONCE warning twice
        # since assertWarnsOnceRegex uses set_warn_always(True) which changes
        # TORCH_WARN_ONCE to TORCH_WARN
        a = np.arange(10)
        a.flags.writeable = False
        with self.assertWarnsOnceRegex(UserWarning, '.*non-writable.*'):
            torch.from_numpy(a)

        # OK, got it once, now try again
        with self.assertWarnsOnceRegex(UserWarning, '.*non-writable.*'):
            torch.from_numpy(a)

        # Make sure emitting two warnings will pass the assertWarnsOnceRegex
        # context manager
        with self.assertWarnsOnceRegex(UserWarning, '.*non-writable.*'):
            torch.from_numpy(a)
            torch.from_numpy(a)

    @onlyNativeDeviceTypes
    def test_complex_half_experimental_warning(self, device):
        msg = 'ComplexHalf support is experimental'
        with self.assertWarnsOnceRegex(UserWarning, msg):
            t = torch.randn(3, dtype=torch.chalf, device=device)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.rand(3, dtype=torch.chalf, device=device)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.empty(3, dtype=torch.chalf, device=device)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.ones(3, dtype=torch.chalf, device=device)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.zeros(3, dtype=torch.chalf, device=device)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.randn_like(t)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.rand_like(t)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.empty_like(t)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.ones_like(t)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.zeros_like(t)

        with self.assertWarnsOnceRegex(UserWarning, msg):
            # t + 1 allocates a new tensor for result using empty
            t + 1

    @onlyCUDA
    def test_dtypetensor_warnings(self, device):
        msg = 'The torch.cuda.*DtypeTensor constructors are no longer recommended'
        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.cuda.FloatTensor([0])

        with self.assertWarnsOnceRegex(UserWarning, msg):
            torch.cuda.DoubleTensor([0])

    def test_set_default_tensor_type_warnings(self, device):
        msg = '.*is deprecated as of PyTorch 2.1, please use torch.set_default_dtype().*'
        default_type = torch.tensor([]).type()
        try:
            with self.assertWarnsOnceRegex(UserWarning, msg):
                torch.set_default_tensor_type(torch.FloatTensor)

            if torch.cuda.is_available():
                with self.assertWarnsOnceRegex(UserWarning, msg):
                    torch.set_default_tensor_type(torch.cuda.FloatTensor)
        finally:
            torch.set_default_tensor_type(default_type)

    # TODO: this test should be in test_nn.py
    def test_conv_transposed_backward_agnostic_to_memory_format(self, device):
        in_channels = 64
        out_channels = 128
        scale_factor = 8
        batch_size = 8
        length = 16

        conv = torch.nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=scale_factor * 2, stride=scale_factor).to(device)
        layer_norm = torch.nn.LayerNorm(out_channels).to(device)

        input_ = torch.randn(batch_size, in_channels, length).to(device).contiguous()
        input_ = conv(input_).contiguous()
        input_ = layer_norm(input_.transpose(1, 2).contiguous()).contiguous()
        input_.sum().backward()

        # 3d
        conv = torch.nn.ConvTranspose3d(3, 3, kernel_size=3).to(device)
        input = torch.randn(batch_size, 3, length, length, length, device=device)
        out = conv(input)
        out.backward(torch.ones_like(out).transpose(-2, -1))

    # TODO: this test should be in test_nn.py
    @onlyCUDA
    @largeTensorTest('12GB')
    def test_conv_transposed_large(self, device):
        # ConvTranspose3d works for large input tensors (gh-32866)
        in_channels = 64
        out_channels = 128
        kernel_size = 5

        conv = torch.nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=2, padding=2, output_padding=1).to(device)

        x = torch.rand([1, 64, 8, 128, 172]).to(device)
        conv(x)

    def test_is_set_to(self, device):
        t1 = torch.empty(3, 4, 9, 10, device=device)
        t2 = torch.empty(3, 4, 9, 10, device=device)
        t3 = torch.tensor([], device=device).set_(t1)
        t4 = t3.clone().resize_(12, 90)
        self.assertFalse(t1.is_set_to(t2))
        self.assertTrue(t1.is_set_to(t3))
        self.assertTrue(t3.is_set_to(t1), "is_set_to should be symmetric")
        self.assertFalse(t1.is_set_to(t4))
        self.assertFalse(torch.tensor([]).is_set_to(torch.tensor([])),
                         "Tensors with no storages should not appear to be set "
                         "to each other")

        t1 = torch.tensor([True, True], dtype=torch.bool, device=device)
        t2 = torch.tensor([0], dtype=torch.bool, device=device).set_(t1)
        self.assertTrue(t1.is_set_to(t2))

        # test that sizes must match
        t1 = torch.empty([2, 3, 4], device=device)
        t2 = t1.view(4, 3, 2)
        self.assertFalse(t1.is_set_to(t2))
        self.assertFalse(t2.is_set_to(t1))

        # test that legacy empty size behavior used to be respected (i.e. all
        # empty tensors were logically collapsed to size [0]).
        t1 = torch.empty([2, 5, 0], device=device)
        t2 = t1.view([0])
        self.assertFalse(t1.is_set_to(t2))
        self.assertFalse(t2.is_set_to(t1))

    # See https://github.com/pytorch/pytorch/issues/72650
    @skipIfMPS
    @skipMeta
    @parametrize(
        "fn",
        [
            "dist", "atan2", "pow", "lerp", "add", "sub", "mul", "div", "fmod", "remainder", "eq", "ge", "gt", "le",
            "lt", "max", "min", "ne", "addcdiv", "addcmul", "masked_scatter", "masked_select", "masked_fill", "map",
            "map2", "copy",
        ],
    )
    def test_broadcast(self, fn, device):
        # functions with three tensor arguments
        fns_3_args = {"map2"}
        fns_value_kwarg = {"addcdiv", "addcmul"}

        (dims_small, dims_large, dims_full) = self._select_broadcastable_dims()
        full1d = torch.randn(*dims_full, device=device).flatten().float()
        small = torch.randn(*dims_small, device=device).float()
        large = torch.randn(*dims_large, device=device).float()
        small_expanded = small.expand(*dims_full)
        large_expanded = large.expand(*dims_full)
        small2 = None
        small2_expanded = None
        if fn in fns_3_args or fn in fns_value_kwarg:
            # create another smaller tensor
            (dims_small2, _, _) = self._select_broadcastable_dims(dims_full)
            small2 = torch.randn(*dims_small2, device=device).float()
            small2_expanded = small2.expand(*dims_full)

        if small.is_cuda and fn in ['map', 'map2']:
            # map and map2 are not implemented on CUDA tensors
            return

        if hasattr(large_expanded, fn):
            # run through tensor versions of functions
            # and verify fully expanded inputs give same results
            expanded = {large: large_expanded, small: small_expanded, small2: small2_expanded}

            def tensorfn(myfn, t1, t2):
                if fn == "lerp":
                    return myfn(t1, 0.5)
                elif fn == "masked_select":
                    return myfn(t1 < 0)
                elif fn == "masked_scatter":
                    return myfn(t1 < 0.5, full1d)
                elif fn == "masked_fill":
                    return myfn(t1 < 0.5, 1.0)
                elif fn in fns_3_args:
                    return myfn(1, t1, t2)
                elif fn in fns_value_kwarg:
                    return myfn(t1, t2, value=1)
                else:
                    return myfn(t1)

            # test various orders
            for first, second, third in [(large, small, small2), (small, large, small2),
                                         (small2, small, large), (small2, large, small)]:
                if first is None:
                    break  # ignore last iter when small2 is None
                method_expanded = getattr(expanded[first], fn)
                method = getattr(first, fn)
                r1 = tensorfn(method_expanded, expanded[second], expanded[third])
                r2 = tensorfn(method, second, third)
                self.assertEqual(r1, r2)

        # now for torch. versions of functions
        if hasattr(torch, fn):
            fntorch = getattr(torch, fn)
            expanded = {large: large_expanded, small: small_expanded, small2: small2_expanded}

            def torchfn(t1, t2, t3):
                if fn == "lerp":
                    return fntorch(t1, t2, 0.5)
                elif fn == "masked_select":
                    return fntorch(t1, t2 < 0)
                elif fn == "masked_scatter":
                    return fntorch(t1, t2 < 0.5, full1d)
                elif fn == "masked_fill":
                    return fntorch(t1, t2 < 0.5, 1.0)
                elif fn in fns_3_args:
                    return fntorch(t1, 1.0, t2, t3)
                elif fn in fns_value_kwarg:
                    return fntorch(t1, t2, t3, value=1.0)
                else:
                    return fntorch(t1, t2)

            # test various orders
            for first, second, third in [(large, small, small2), (small, large, small2),
                                         (small2, small, large), (small2, large, small)]:
                if first is None:
                    break  # ignore last iter when small2 is None
                r1 = torchfn(expanded[first], expanded[second], expanded[third])
                r2 = torchfn(first, second, third)
                self.assertEqual(r1, r2)

        # now for in place functions
        # in-place tensor is not broadcastable; test only guaranteed
        # to work by broadcasting other argument(s)
        if not hasattr(large_expanded, fn + "_"):
            return

        # need to clone largeExpanded so we can reuse, since functions are in-place
        large_expanded_clone = large_expanded.clone()

        def tensorfn_inplace(t0, t1, t2=None):
            t0_fn = getattr(t0, fn + "_")
            if fn == "lerp":
                return t0_fn(t1, 0.5)
            elif fn == "masked_scatter":
                return t0_fn(t1 < 0.5, full1d)
            elif fn == "masked_fill":
                return t0_fn(t1 < 0.5, 1.0)
            elif fn == "map":
                return t0_fn(t1, lambda x, y: x + y)
            elif fn == "map2":
                return t0_fn(t1, t2, lambda x, y, z: x + y + z)
            elif fn in fns_3_args:
                return t0_fn(1.0, t1, t2)
            elif fn in fns_value_kwarg:
                return t0_fn(t1, t2, value=1.0)
            else:
                return t0_fn(t1)
        # in-place pointwise operations don't actually work if the in-place
        # tensor is 0-strided (numpy has the same issue)
        if (0 not in large_expanded.stride() and 0 not in large_expanded_clone.stride()):
            r1 = tensorfn_inplace(large_expanded, small_expanded, small2_expanded)
            r2 = tensorfn_inplace(large_expanded_clone, small, small2)
            self.assertEqual(r1, r2)

        def broadcastable(t0, t1, t2=None):
            try:
                t1.expand_as(t0)
                if t2 is not None:
                    t2.expand_as(t0)
            except RuntimeError:
                return False
            return True

        def _test_in_place_broadcastable(t0, t1, t2=None):
            if not broadcastable(t0, t1, t2):
                same_size = t0.numel() == t1.numel() and (t0.numel() == t2.numel() if t2 is not None else True)
                if not same_size:
                    # Functionalization converts the inplace to an out-of-place, which causes us to error.
                    # We should fix this, but "error probably on bad inputs" isn't a hi-pri PT2 item.
                    if not TEST_WITH_TORCHINDUCTOR:
                        self.assertRaises(RuntimeError, lambda: tensorfn_inplace(t0, t1, t2))
            else:
                tensorfn_inplace(t0, t1, t2)

        if fn not in fns_3_args and fn not in fns_value_kwarg:
            _test_in_place_broadcastable(small, large_expanded)
            _test_in_place_broadcastable(small, large)
        else:
            _test_in_place_broadcastable(small2, small_expanded, large_expanded)
            _test_in_place_broadcastable(small2, small, large)

    @onlyCPU
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    @dtypes(*get_all_qint_dtypes())
    def test_nondeterministic_resize_quantized(self, device, dtype):
        a = torch.tensor([-1, 0, 1, 2, 3], dtype=torch.float, device=device)
        b = torch.quantize_per_tensor(a, 0.1, 10, dtype)
        self.check_nondeterministic_alert(
            lambda: b.resize_((10,)),
            'quantized_resize_cpu_')

    @skipXLA
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16, torch.uint16, torch.uint32, torch.uint64))
    def test_deterministic_resize(self, device, dtype):
        test_cases = [
            # size, stride, resize_size
            ((10,), (1,), (5,)),
            ((10,), (0,), (10,)),
            ((10,), (1,), (20,)),
            ((2, 3, 4), None, (2, 3, 4)),
            ((2, 3, 4), None, (6, 3, 4)),
            ((2, 3, 4), None, (2, 5, 4)),
            ((2, 3, 4), None, (2, 3, 6)),
            ((2, 3, 4), None, (3, 4, 5)),
            ((2, 3, 4), (1, 4, 12), (2, 3, 4)),
            ((2, 3, 4), (1, 4, 12), (4, 3, 4)),
            ((2, 3, 4), (1, 4, 12), (2, 4, 4)),
            ((2, 3, 4), (1, 4, 12), (2, 3, 5)),
            ((2, 3, 4), (1, 4, 12), (3, 4, 5)),
            ((2, 3, 4), (1, 0, 1), (2, 4, 5)),
        ]

        for size, stride, resize_size in test_cases:
            if stride is None:
                a = torch.zeros(size, dtype=dtype, device=device)
            else:
                a = torch.empty_strided(size, stride, dtype=dtype, device=device).fill_(0)
            old_storage = a.untyped_storage().clone()
            with DeterministicGuard(True, fill_uninitialized_memory=True):
                a.resize_(resize_size)

            new_storage = a.untyped_storage()

            # If storage size was increased, check that the new section is
            # filled with NaN/MAX_INT. Otherwise, check that the storages are
            # equal.
            old_tensor = torch.tensor(old_storage, dtype=dtype)
            old_numel = old_tensor.numel()
            new_tensor = torch.tensor(new_storage, dtype=dtype)
            new_numel = new_tensor.numel()

            if new_numel > old_numel:
                self.assertEqual(new_tensor[:old_numel], old_tensor)
                fill_section = new_tensor[old_numel:]

                if dtype.is_floating_point or dtype.is_complex:
                    self.assertTrue(fill_section.isnan().all())
                else:
                    if dtype == torch.bool:
                        max_val = True
                    else:
                        max_val = torch.iinfo(dtype).max
                    self.assertTrue(fill_section.eq(max_val).all())
            else:
                self.assertEqual(old_tensor, new_tensor)

    # When deterministic algorithms are enabled, `torch.empty` should fill floating
    # point tensors with NaN and integer tensors with MAX_INT
    @skipXLA
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    @dtypes(*all_types_and_complex_and(
        torch.half, torch.bool, torch.bfloat16, torch.uint16, torch.uint32, torch.uint64, torch.complex32))
    def test_deterministic_empty(self, device, dtype):
        gen_fns = [
            lambda: torch.empty(10, 9, device=device, dtype=dtype),
            lambda: torch.empty(10, 9, out=torch.zeros(1, device=device, dtype=dtype)),
            lambda: torch.empty_like(torch.zeros(10, 9, device=device, dtype=dtype)),
            lambda: torch.empty_like(torch.zeros(10, 9, device=device, dtype=dtype), memory_format=torch.contiguous_format),
            lambda: torch.empty_strided((10, 9), (1, 5), device=device, dtype=dtype),
            lambda: torch.empty_permuted((2, 3, 5), (1, 0, 2), device=device, dtype=dtype),
        ]

        for gen_fn in gen_fns:
            with DeterministicGuard(True, fill_uninitialized_memory=True):
                res = gen_fn()

            if dtype.is_floating_point or dtype.is_complex:
                self.assertTrue(res.isnan().all())
            else:
                if dtype == torch.bool:
                    max_val = True
                else:
                    max_val = torch.iinfo(dtype).max
                self.assertTrue(res.eq(max_val).all())

    # FIXME: update OpInfos to support "nondeterministic samples" and port these tests
    #   to that architecture
    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_AvgPool3d(self, device):
        module = torch.nn.AvgPool3d(3)
        input = torch.randn(2, 3, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'avg_pool3d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_AdaptiveAvgPool2d(self, device):
        module = torch.nn.AdaptiveAvgPool2d(3)
        input = torch.randn(2, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'adaptive_avg_pool2d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_AdaptiveAvgPool3d(self, device):
        module = torch.nn.AdaptiveAvgPool3d(3)
        input = torch.randn(2, 3, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'adaptive_avg_pool3d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_MaxPool3d(self, device):
        module = torch.nn.MaxPool3d(3)
        input = torch.randn(2, 3, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'max_pool3d_with_indices_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_AdaptiveMaxPool2d(self, device):
        module = torch.nn.AdaptiveMaxPool2d(3)
        input = torch.randn(2, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'adaptive_max_pool2d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_FractionalMaxPool2d(self, device):
        module = torch.nn.FractionalMaxPool2d(2, output_ratio=0.5)
        input = torch.randn(2, 3, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'fractional_max_pool2d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_FractionalMaxPool3d(self, device):
        module = torch.nn.FractionalMaxPool3d(2, output_ratio=0.5)
        input = torch.randn(2, 3, 3, 3, 3, requires_grad=True, device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'fractional_max_pool3d_backward_cuda',
            torch.device(device).type == 'cuda')

    @dtypes(*floating_types_and(torch.half))
    @onlyNativeDeviceTypes
    def test_nondeterministic_alert_MaxUnpool1d(self, device, dtype):
        if dtype == torch.half and torch.device(device).type == 'cpu':
            self.skipTest('float16 not implemented on CPU')

        module = torch.nn.MaxUnpool1d(3, 1)
        input = torch.randn(1, 1, 7, dtype=dtype, device=device)
        indices = torch.zeros_like(input, dtype=torch.long, device=device)

        self.check_nondeterministic_alert(
            lambda: module(input, indices),
            'max_unpooling2d_forward_out')

    @dtypes(*floating_types_and(torch.half))
    @onlyNativeDeviceTypes
    def test_nondeterministic_alert_MaxUnpool2d(self, device, dtype):
        if dtype == torch.half and torch.device(device).type == 'cpu':
            self.skipTest('float16 not implemented on CPU')

        module = torch.nn.MaxUnpool2d(3, 1)
        input = torch.randn(1, 1, 7, 7, dtype=dtype, device=device)
        indices = torch.zeros_like(input, dtype=torch.long, device=device)

        self.check_nondeterministic_alert(
            lambda: module(input, indices),
            'max_unpooling2d_forward_out')

    @dtypes(*floating_types_and(torch.half))
    @onlyNativeDeviceTypes
    def test_nondeterministic_alert_MaxUnpool3d(self, device, dtype):
        if dtype == torch.half and torch.device(device).type == 'cpu':
            self.skipTest('float16 not implemented on CPU')

        module = torch.nn.MaxUnpool3d(3, 1)
        input = torch.randn(1, 1, 7, 7, 7, dtype=dtype, device=device)
        indices = torch.zeros_like(input, dtype=torch.long, device=device)

        self.check_nondeterministic_alert(
            lambda: module(input, indices),
            'max_unpooling3d_forward_out')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_interpolate_linear(self, device):
        input = torch.randn(1, 2, 4, device=device, requires_grad=True)
        res = torch.nn.functional.interpolate(
            input,
            size=12,
            mode='linear',
            align_corners=False)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad),
            'upsample_linear1d_backward_out_cuda',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_interpolate_bilinear(self, device):
        input = torch.randn(1, 2, 4, 4, device=device, requires_grad=True)
        res = torch.nn.functional.interpolate(
            input,
            size=12,
            mode='bilinear',
            align_corners=False)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad),
            'upsample_bilinear2d_backward_out_cuda',
            torch.device(device).type == 'cuda')

    def test_no_nondeterministic_alert_interpolate_bilinear(self, device):
        input = torch.randn(1, 2, 4, 4, device=device, requires_grad=True)

        def fn():
            res = torch.nn.functional.interpolate(
                input,
                size=12,
                mode='bilinear',
                align_corners=False)
            grad = torch.ones_like(res)
            return res.backward(grad)

        self.check_nondeterministic_alert(
            fn,
            'upsample_bilinear2d_backward_out_cuda',
            False)

    def test_no_nondeterministic_alert_interpolate_trilinear(self, device):
        input = torch.randn(1, 2, 4, 4, 4, device=device, requires_grad=True)

        def fn():
            res = torch.nn.functional.interpolate(
                input,
                size=12,
                mode='trilinear',
                align_corners=False)
            grad = torch.ones_like(res)
            return res.backward(grad)

        self.check_nondeterministic_alert(
            fn,
            'upsample_trilinear3d_backward_out_cuda',
            False)

    @skipIfTorchInductor("aot-autograd issue")
    def test_deterministic_replication_pad2d(self, device):
        test_cases = [
            # size, padding
            [(1, 2, 4, 4), (0, 0, 0, 0)],
            [(1, 2, 4, 4), (3, 4, 5, 6)],
            [(3, 8, 7), (0, 0, 0, 0)],
            [(3, 8, 7), (4, 3, 2, 7)],
        ]

        if torch.device(device).type != 'xla':
            test_cases += [
                [(4, 3, 5, 10), (-9, 4, 5, 6)],
                [(3, 8, 7), (-4, -2, -2, -3)],
            ]

        for size, padding in test_cases:
            input = torch.randn(*size, device=device, requires_grad=True)
            grad = None
            with DeterministicGuard(True):
                res = torch.nn.functional.pad(
                    input,
                    padding,
                    mode='replicate')
                res.backward(torch.ones_like(res))
                if grad is None:
                    grad = input.grad
                else:
                    self.assertEqual(grad, input.grad, atol=0, rtol=0)
                input.grad = None

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_deterministic_interpolate_bilinear(self, device):
        input = torch.randn(1, 2, 4, 4, device=device, requires_grad=True)
        grad = None
        with DeterministicGuard(True):
            for _ in range(5):
                res = torch.nn.functional.interpolate(
                    input,
                    size=12,
                    mode='bilinear',
                    align_corners=False)
                res.backward(torch.ones_like(res))
                if grad is None:
                    grad = input.grad
                else:
                    self.assertEqual(grad, input.grad, atol=0, rtol=0)
                input.grad = None

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_interpolate_bicubic(self, device):
        input = torch.randn(1, 2, 4, 4, device=device, requires_grad=True)
        res = torch.nn.functional.interpolate(
            input,
            size=12,
            mode='bicubic',
            align_corners=False)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad),
            'upsample_bicubic2d_backward_out_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_interpolate_trilinear(self, device):
        input = torch.randn(1, 2, 4, 4, 4, device=device, requires_grad=True)
        res = torch.nn.functional.interpolate(
            input,
            size=12,
            mode='trilinear',
            align_corners=False)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad),
            'upsample_trilinear3d_backward_out_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_ReflectionPad1d(self, device):
        module = torch.nn.ReflectionPad1d((1, 2))
        input = torch.randn(2, 3, 8, device=device, requires_grad=True)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'reflection_pad1d_backward_out_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_ReflectionPad3d(self, device):
        module = torch.nn.ReflectionPad3d((1, 2, 3, 4, 5, 6))
        input = torch.randn(2, 3, 8, 8, 8, device=device, requires_grad=True)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'reflection_pad3d_backward_out_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_ReplicationPad1d(self, device):
        module = torch.nn.ReplicationPad1d((1, 2))
        input = torch.randn(2, 3, 4, device=device, requires_grad=True)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'replication_pad1d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_ReplicationPad2d(self, device):
        module = torch.nn.ReplicationPad2d((1, 2, 3, 4))
        input = torch.randn(2, 3, 4, 4, device=device, requires_grad=True)
        res = module(input)
        grad = torch.ones_like(res)

        # Nondeterministic alert should only be raised if the forward call was
        # nondeterministic
        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'replication_pad2d_backward_cuda',
            torch.device(device).type == 'cuda')

        with DeterministicGuard(True):
            res = module(input)

        grad = torch.ones_like(res)

        # If the forward call was deterministic, nondeterministic alert should
        # not be raised
        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'replication_pad2d_backward_cuda',
            False)

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_ReplicationPad3d(self, device):
        module = torch.nn.ReplicationPad3d((1, 2, 3, 4, 5, 6))
        input = torch.randn(2, 3, 4, 4, 4, device=device, requires_grad=True)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'replication_pad3d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfTorchDynamo("Warning is not raised.")
    def test_nondeterministic_alert_NLLLoss(self, device):
        module = torch.nn.NLLLoss()
        input = torch.randn(2, 3, 5, 5, device=device)
        target = torch.rand(2, 5, 5, device=device).mul(3).floor().long()


        self.check_nondeterministic_alert(
            lambda: module(input, target),
            'nll_loss2d_forward_out_cuda_template',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_CTCLoss(self, device):
        module = torch.nn.CTCLoss()
        input = torch.randn(50, 3, 15, device=device, requires_grad=True)
        target = torch.randint(0, 14, (3, 30), device=device)
        input_lengths = [50, 50, 50]
        target_lengths = [30, 25, 20]
        res = module(input, target, input_lengths, target_lengths)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'ctc_loss_backward_gpu',
            torch.device(device).type == 'cuda')

    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_EmbeddingBag_max(self, device):
        module = torch.nn.EmbeddingBag(
            4, 3, None, 2., False, 'max',
            _weight=torch.randn(4, 3, device=device, requires_grad=True))
        input = torch.randint(0, 3, (4, 3), device=device)
        res = module(input)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'embedding_bag_backward_cuda_max',
            torch.device(device).type == 'cuda')

    @skipIfRocmArch(MI300_ARCH)
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    @onlyCUDA
    def test_deterministic_cumsum(self, device):
        test_cases = [
            # size, dim
            [(1025,), 0],
            [(8193,), 0],
            [(8191,), 0],
            [(128256,), 0],
            [(1282560,), 0],
            [(12825600,), 0],
        ]
        for size, dim in test_cases:
            input = 100 * torch.rand(*size, device=device)
            with DeterministicGuard(True):
                res0 = input.cumsum(dim)
                for _ in range(3):
                    res1 = input.cumsum(dim)
                    self.assertEqual(res0, res1, atol=0, rtol=0)

            res_cpu = input.cpu().cumsum(dim)
            self.assertEqual(res0, res_cpu, atol=1e-3, rtol=1e-2)
            # test double complex that has fewer threads than CTAs
            # that's a problem for large GPUS (H100 with 132 sms)
            # test is very tailored for that and will provide 0 signal
            # on smaller GPUs
            num_sms = 132
            elems_per_cta = 256 * 16
            N = num_sms * elems_per_cta
            input = torch.rand(N, dtype=torch.complex128, device=device)
            with DeterministicGuard(True):
                res0 = input.cumsum(dim)
                for _ in range(3):
                    res1 = input.cumsum(dim)
                    self.assertEqual(res0, res1, atol=0, rtol=0)

            res_cpu = input.cpu().cumsum(dim)
            self.assertEqual(res0, res_cpu, atol=1e-3, rtol=1e-2)

    @onlyCUDA
    @largeTensorTest('49GB')
    def test_cumsum_64bit_indexing(self, device):
        b = torch.ones(2 * 4096 * 8, 100000, dtype=torch.float, device='cuda')
        b /= 100000
        d = b.cumsum(dim=-1)
        chunk = 2**30 // b.shape[-1]
        for i in range(0, b.shape[0], chunk):
            end = min(i + chunk, b.shape[0])
            b[i:end, :].cumsum_(dim=-1)
        # cheat a bit to avoid OOM
        self.assertEqual(b[0, :], d[0, :], atol=3e-5, rtol=3e-5)
        self.assertEqual(b[-1, :], d[-1, :], atol=3e-5, rtol=3e-5)

    @onlyCUDA
    @largeTensorTest('48GB')
    def test_cumsum_outer_dim_64bit_indexing(self, device):
        x = torch.zeros(309504, 1, 16384, device=device)
        torch.exp(x)
        cumsum = torch.cumsum(x, dim=1)
        self.assertEqual(cumsum.max().item(), 0., atol=0., rtol=0.)

    @expectedFailureMeta  # expected a non-determinitic error, but it was not raised
    @onlyNativeDeviceTypes
    def test_nondeterministic_alert_put(self, device):
        a = torch.randn(10, device=device)
        indices = torch.tensor([0, 0], device=device)
        values = torch.tensor([0., 1.], device=device)

        for op_call in [torch.Tensor.put, torch.Tensor.put_]:
            self.check_nondeterministic_alert(
                lambda: op_call(a, indices, values, accumulate=False),
                'put_')

    # warn_only=False correctly raises RuntimeError: put_ does not have a deterministic implementation
    # warn_only=True logs warning from the FallbackKernel: torch.ops.aten.put_.default, instead of as UserWarning:
    # [W Context.cpp:%(lineno)] Warning: put_ does not have a deterministic implementation
    @skipIfTorchInductor("warning is logged from the FallbackKernel: torch.ops.aten.put_.default when warn_only=True")
    def test_nondeterministic_alert_put_accumulate(self, device):
        a = torch.randn(10, device=device)
        indices = torch.tensor([0, 0], device=device)
        values = torch.tensor([0., 1.], device=device)

        for op_call in [torch.Tensor.put, torch.Tensor.put_]:
            self.check_nondeterministic_alert(
                lambda: op_call(a, indices, values, accumulate=True),
                'put_',
                torch.device(device).type == 'cuda')

    @dtypes(torch.float32)
    @dtypesIfCUDA(torch.float32, torch.int32)
    @skipIfMPS
    def test_nondeterministic_alert_histc(self, device, dtype):
        a = torch.tensor([], device=device, dtype=dtype)
        for op_call in [torch.histc, torch.Tensor.histc]:
            self.check_nondeterministic_alert(
                lambda: op_call(a, min=0, max=3),
                '_histc_cuda with floating point input',
                torch.device(device).type == 'cuda' and dtype.is_floating_point)

    @skipIfMPS
    def test_nondeterministic_alert_bincount(self, device):
        a = torch.tensor([], device=device, dtype=torch.long)
        weights = torch.tensor([], device=device)

        for op_call in [torch.bincount, torch.Tensor.bincount]:
            # Error should only be raised when device is CUDA and weights are
            # given
            self.check_nondeterministic_alert(
                lambda: op_call(a, weights),
                '_bincount_cuda',
                torch.device(device).type == 'cuda')

            self.check_nondeterministic_alert(
                lambda: op_call(a),
                '_bincount_cuda',
                False)

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_grid_sample_2d(self, device):
        input = torch.empty(1, 1, 2, 2, device=device, requires_grad=True)
        grid = torch.empty(1, 1, 1, 2, device=device)
        res = torch.nn.functional.grid_sample(input, grid, align_corners=False)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'grid_sampler_2d_backward_cuda',
            torch.device(device).type == 'cuda')

    @skipIfMPS
    @skipIfTorchInductor("https://github.com/pytorch/pytorch/issues/113707")
    def test_nondeterministic_alert_grid_sample_3d(self, device):
        input = torch.empty(1, 1, 2, 2, 2, device=device, requires_grad=True)
        grid = torch.empty(1, 1, 1, 2, 3, device=device)
        res = torch.nn.functional.grid_sample(input, grid, align_corners=False)
        grad = torch.ones_like(res)

        self.check_nondeterministic_alert(
            lambda: res.backward(grad, retain_graph=True),
            'grid_sampler_3d_backward_cuda',
            torch.device(device).type == 'cuda')

    def test_invalid_shapes_grid_sampler(self, device):
        make_arg = partial(
            make_tensor, device=device, dtype=torch.float64, requires_grad=True)

        inputs = (
            # input, grid
            ((5, 5, 5, 5, 5,), (1, 1, 1, 4, 4,)),  # 3d
            ((5, 5, 5, 5,), (1, 1, 4, 4,)),  # 2d
        )

        interpolation_mode = 0
        padding_mode = 0
        align_corners = True

        err = "expected grid and input to have same batch size"

        for input, grid in inputs:
            input = make_arg(input)
            grid = make_arg(grid, low=-1, high=1)

            # Wrapper for the 2d, 3d, and cuDNN functions listed below.
            with self.assertRaisesRegex(RuntimeError, err):
                torch.grid_sampler(
                    input, grid, interpolation_mode, padding_mode,
                    align_corners)

            # Expects 2d input.
            with self.assertRaisesRegex(RuntimeError, err):
                torch.grid_sampler_2d(
                    input, grid, interpolation_mode, padding_mode,
                    align_corners)

            # Expects 3d input.
            with self.assertRaisesRegex(RuntimeError, err):
                torch.grid_sampler_3d(
                    input, grid, interpolation_mode, padding_mode,
                    align_corners)

            # Expects 2d input.
            with self.assertRaisesRegex(RuntimeError, err):
                torch._grid_sampler_2d_cpu_fallback(
                    input, grid, interpolation_mode, padding_mode,
                    align_corners)

            # Expects 2d input, on CUDA.
            # Doesn't work on CPU and ROCm.
            if device != 'cpu' and TEST_CUDNN and not TEST_WITH_ROCM:
                with self.assertRaisesRegex(RuntimeError, err):
                    torch.cudnn_grid_sampler(input, grid)

    def test_dist(self, device):
        def run_test(x, y):
            for p in [0, 1, 2, 3, 4, inf, -inf]:
                dist_xy = torch.dist(x, y, p)
                dist_xy_norm = torch.norm(x - y, p)
                self.assertEqual(dist_xy, dist_xy_norm)

        run_test(torch.randn(5, device=device), torch.randn(5, device=device))

        x = torch.zeros(3, device=device)
        y = torch.zeros(3, device=device)
        y[1] = 1.
        run_test(x, y)

    # Ensures that median throws nondeterministic alerts in the correct cases
    @dtypes(torch.double)
    def test_nondeterministic_alert_median(self, device, dtype):
        def test_func(call_type):
            S = 10
            a = torch.randn(S, device=device)
            if call_type == 'function':
                torch.median(a)
            elif call_type == 'function with indices':
                torch.median(a, 0)
            elif call_type == 'method':
                a.median()
            elif call_type == 'method with indices':
                a.median(0)
            elif call_type == 'out with indices':
                result = torch.empty_like(a)
                indices = torch.empty((), dtype=torch.long, device=device)
                torch.median(a, 0, out=(result, indices))
            else:
                self.fail(f"'{call_type}' is not a valid call type")

        def test_func_expect_error(call_type, should_error):
            self.check_nondeterministic_alert(
                lambda: test_func(call_type),
                'median CUDA with indices output',
                should_error)

        is_cuda = torch.device(device).type == 'cuda'

        test_func_expect_error('function', False)
        test_func_expect_error('function with indices', is_cuda)
        test_func_expect_error('method', False)
        test_func_expect_error('method with indices', is_cuda)
        test_func_expect_error('out with indices', is_cuda)

    # FIXME: move to test_scatter_gather_ops
    def _test_gather_backward_one_dim(self, device, deterministic: bool = False) -> None:
        with DeterministicGuard(deterministic):
            m = random.randint(2000, 3000)
            elems = random.randint(10 * m, 20 * m)
            dim = 0
            src = torch.randn(m, device=device, requires_grad=True)
            idx = torch.randint(m, (elems,), device=device)
            res = torch.gather(src, dim, idx)
            weight = torch.rand_like(res, device=device) * 10 ** 6
            res.backward(weight)
            assert src.grad is not None
            grad = src.grad.detach().clone()

            if torch.device(device).type == 'cuda':
                for _ in range(2):
                    src.grad.data.zero_()
                    res = torch.gather(src, dim, idx)
                    res.backward(weight)
                    self.assertEqual(src.grad, grad, atol=0, rtol=0)
            else:
                expected = torch.zeros_like(src, device=device)
                for i in range(elems):
                    expected[idx[i]] += weight[i]
                self.assertEqual(grad, expected, atol=0, rtol=0)

    # FIXME: move to test_scatter_gather_ops
    @onlyNativeDeviceTypes
    def test_gather_backward_deterministic_path(self, device) -> None:
        self._test_gather_backward_one_dim(device, True)

    # FIXME: move to test_scatter_gather_ops
    @onlyCPU
    def test_gather_backward_one_dim(self, device) -> None:
        self._test_gather_backward_one_dim(device, False)

    # FIXME: move to test_scatter_gather_ops
    @onlyNativeDeviceTypes
    def test_scatter_add_one_dim_deterministic(self, device) -> None:
        with DeterministicGuard(True):
            m = random.randint(20, 30)
            elems = random.randint(2000 * m, 3000 * m)
            dim = 0
            src = torch.randn(elems, device=device)
            idx = torch.randint(m, (elems,), device=device)

            x = torch.zeros(m, device=device)
            res = x.scatter_add(dim, idx, src)

            # Checking if scatter_add is deterministic
            for _ in range(5):
                res_next = x.scatter_add(dim, idx, src)
                self.assertEqual(res, res_next, atol=0, rtol=0)
                res = res_next

            expected = torch.zeros(m, device=device)
            for i in range(elems):
                expected[idx[i]] += src[i]

            self.assertEqual(res, expected, atol=1e-4, rtol=1e-5)

    # FIXME: move to test_scatter_gather_ops
    @onlyNativeDeviceTypes
    def test_scatter_zero_size_index(self, device) -> None:
        null_index = torch.zeros((0, 4), dtype=torch.int64)
        null_arr = torch.zeros((0, 4))
        original = torch.arange(4, dtype=torch.float32)
        result = original.scatter(0, null_index, null_arr)
        self.assertEqual(result, original, atol=0, rtol=0)

    @onlyCUDA
    @skipIfTorchInductor("FIXME")
    def test_sync_warning(self, device):

        def _sync_raises_helper(f, level):
            with CudaSyncGuard(level):
                if level == 1:
                    with self.assertWarnsRegex(UserWarning, "called a synchronizing "):
                        f()
                elif level == 2:
                    with self.assertRaisesRegex(RuntimeError, "called a synchronizing "):
                        f()

        def _no_sync_helper(f, level):
            with CudaSyncGuard(level):
                f()

        def _ind_put_fn(x, ind, val):
            x[ind] = val
            return x

        def _ind_get_fn(x, ind):
            return x[ind]

        def _cond_fn(x):
            if x:  # taking boolean value of a tensor synchronizes
                return x
            else:
                return 2 * x

        # prepare inputs for subsequent ops
        size = 4
        x = torch.rand(size, device=device)
        y = torch.rand((), device=device)
        ind = torch.randint(size, (3,), device=device)
        ind_cpu = ind.cpu()
        repeats = torch.full((1,), 2, device=device)
        mask = torch.randint(2, (size,), device=device, dtype=bool)
        mask_cpu = mask.cpu()
        expect_no_sync = (lambda: _ind_put_fn(x, mask, 1.),
                          lambda: _ind_put_fn(x, mask_cpu, y),
                          lambda: _ind_put_fn(x, ind, y),
                          lambda: _ind_get_fn(x, mask_cpu),
                          lambda: _ind_get_fn(x, ind),
                          lambda: torch.nn.functional.one_hot(ind, num_classes=size),
                          lambda: torch.randperm(20000, device=device),
                          lambda: torch.repeat_interleave(x, 2, output_size=2 * size),
                          lambda: torch.repeat_interleave(x, repeats, output_size=2 * size),
                          lambda: torch.any(y))
        expect_sync = (lambda: _ind_put_fn(x, mask, y),
                       lambda: _ind_put_fn(x, ind_cpu, y),
                       lambda: _ind_get_fn(x, mask),
                       lambda: _ind_get_fn(x, ind_cpu),
                       lambda: x.nonzero(),
                       lambda: _cond_fn(y),
                       lambda: torch.nn.functional.one_hot(ind),
                       lambda: torch.repeat_interleave(x, repeats))
        for f, level in product(expect_no_sync, (1, 2)):
            _no_sync_helper(f, level)
        for f, level in product(expect_sync, (1, 2)):
            _sync_raises_helper(f, level)


    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    @skipIfMPS
    def test_log_normal(self, device, dtype):
        a = torch.tensor([10], dtype=dtype, device=device).log_normal_()
        self.assertEqual(a.dtype, dtype)
        self.assertEqual(a.size(), torch.Size([1]))

    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    @skipIfMPS
    def test_geometric(self, device, dtype):
        a = torch.tensor([10], dtype=dtype, device=device).geometric_(0.5)
        self.assertEqual(a.dtype, dtype)
        self.assertEqual(a.size(), torch.Size([1]))

    @skipIfMPS
    def test_repeat_interleave(self, device):
        y = torch.tensor([[1, 2], [3, 4]], device=device)
        # exercise single argument function signature
        temp = y.repeat_interleave(2)
        self.assertEqual(torch.Size([8]), temp.size())

        for dtype in [torch.int, torch.long]:
            lengths = torch.tensor([1, 2], dtype=dtype, device=device)
            output_size = torch.sum(lengths)
            a = torch.repeat_interleave(
                y,
                lengths,
                dim=0,
            )
            self.assertEqual(a.dtype, y.dtype)
            self.assertEqual(a.size(), torch.Size([3, 2]))

            a_with_output = torch.repeat_interleave(
                y,
                lengths,
                dim=0,
                output_size=output_size,
            )
            self.assertEqual(a_with_output.dtype, y.dtype)
            self.assertEqual(a_with_output.size(), torch.Size([3, 2]))

    @dtypes(*floating_types())
    @dtypesIfCPU(*floating_types_and(torch.bfloat16, torch.half))
    @dtypesIfCUDA(*floating_types_and(torch.half))
    def test_bernoulli_p(self, device, dtype):
        for trivial_p in ([0, 1], [1, 0, 1, 1, 0, 1]):
            x = torch.tensor(trivial_p, dtype=dtype, device=device)
            self.assertEqual(x.bernoulli().tolist(), trivial_p)

        def isBinary(t):
            return torch.ne(t, 0).mul_(torch.ne(t, 1)).sum().item() == 0

        p = torch.rand(5, 5, dtype=dtype, device=device)
        self.assertTrue(isBinary(p.bernoulli()))

        p = torch.rand(5, dtype=dtype, device=device).expand(5, 5)
        self.assertTrue(isBinary(p.bernoulli()))

        p = torch.rand(5, 5, dtype=dtype, device=device)
        torch.bernoulli(torch.rand_like(p), out=p)
        self.assertTrue(isBinary(p))

    # RngUniform not implemented for Integral type in XLA test
    @dtypes(*floating_types())
    @dtypesIfCPU(*all_types_and(torch.bool, torch.half))
    @dtypesIfCUDA(*all_types_and(torch.bool, torch.half))
    def test_bernoulli_self(self, device, dtype):

        def isBinary(t):
            return torch.ne(t, 0).mul_(torch.ne(t, 1)).sum().item() == 0

        t = torch.empty(10, 10, dtype=dtype, device=device)

        t.fill_(2)
        t.bernoulli_(0.5)
        self.assertTrue(isBinary(t))

        for p_dtype in floating_types_and(*[torch.half] if device.startswith('cuda') else []):
            p = torch.rand(10, dtype=p_dtype, device=device).expand(10, 10)
            t.fill_(2)
            t.bernoulli_(p)
            self.assertTrue(isBinary(t))

            t.fill_(2)
            torch.bernoulli(torch.rand_like(t, dtype=p_dtype), out=t)
            self.assertTrue(isBinary(t))

            t.fill_(2)
            t.bernoulli_(torch.rand_like(t, dtype=p_dtype))
            self.assertTrue(isBinary(t))

    @slowTest
    @dtypes(*floating_types_and(torch.half))
    @dtypesIfCUDA(*floating_types_and(torch.half))
    def test_bernoulli_edge_cases(self, device, dtype):
        # Need to draw a lot of samples to cover every random floating point number.
        a = torch.zeros(10000, 10000, dtype=dtype, device=device)  # probability of drawing "1" is 0
        num_ones = (torch.bernoulli(a) == 1).sum()
        self.assertEqual(num_ones, 0)

        b = torch.ones(10000, 10000, dtype=dtype, device=device)  # probability of drawing "1" is 1
        num_zeros = (torch.bernoulli(b) == 0).sum()
        self.assertEqual(num_zeros, 0)

    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    @skipIfMPS
    def test_exponential(self, device, dtype):
        a = torch.tensor([10], dtype=dtype, device=device).exponential_(0.5)
        self.assertEqual(a.dtype, dtype)
        self.assertEqual(a.size(), torch.Size([1]))

        # Tests extremal behavior
        t = torch.empty((1,), device=device, dtype=dtype).exponential_(float('inf'))
        self.assertTrue(t.item() == 0)

        # Tests that negative lambda fails
        with self.assertRaises(RuntimeError):
            torch.empty((1,), device=device, dtype=dtype).exponential_(-0.5)

    @onlyCUDA
    @dtypes(torch.half, torch.float)
    def test_exponential_no_zero(self, device, dtype):
        # naively, 0 in exponential can be generated with probability 2^-24
        # so we need more samples to check if it's not generated
        # instead of doing one
        # don't test CPU, that would be a long test
        x = torch.empty(50000000, device=device, dtype=dtype).exponential_()
        self.assertTrue(x.min() > 0)

    def _generate_correlation_tensors(self, device, dtype):
        yield make_tensor((0, 0), dtype=dtype, device=device)
        yield make_tensor((1, 0), dtype=dtype, device=device)
        yield make_tensor((0, 1), dtype=dtype, device=device)
        yield make_tensor((2,), dtype=dtype, device=device)
        yield make_tensor((2, 1), dtype=dtype, device=device)
        yield make_tensor((2, 2), dtype=dtype, device=device)
        yield make_tensor((2, 3), dtype=dtype, device=device)
        yield make_tensor((5, 10), dtype=dtype, device=device)
        yield make_tensor((5, 10), dtype=dtype, device=device, noncontiguous=True)
        if dtype != torch.int:
            yield torch.tensor([0, -2, nan, 10.2, inf], dtype=dtype, device=device)

    @tf32_on_and_off(0.005)
    @onlyNativeDeviceTypes
    @dtypes(torch.int, torch.float, torch.cfloat)
    def test_corrcoef(self, device, dtype):
        for x in self._generate_correlation_tensors(device, dtype):
            res = torch.corrcoef(x)
            ref = np.corrcoef(x.cpu().numpy())
            self.assertEqual(res, ref, atol=1e-04, rtol=1e-03, exact_dtype=False)

    @skipRocmIfTorchInductor
    @dtypes(torch.int, torch.float, torch.cfloat)
    def test_cov(self, device, dtype):
        def check(t, correction=1, fweights=None, aweights=None):
            res = torch.cov(t, correction=correction, fweights=fweights, aweights=aweights)
            t = t.cpu().numpy()
            fweights = fweights.cpu().numpy() if fweights is not None else None
            aweights = aweights.cpu().numpy() if aweights is not None else None
            ref = np.cov(t, ddof=correction, fweights=fweights, aweights=aweights)
            self.assertEqual(res, ref, atol=1e-05, rtol=1e-05, exact_dtype=False)

        for x in self._generate_correlation_tensors(device, dtype):
            check(x)
            num_observations = x.numel() if x.ndim < 2 else x.size(1)
            if num_observations > 0:
                fweights = torch.randint(1, 10, (num_observations,), device=device)
                aweights = make_tensor((num_observations,), dtype=torch.float, device=device, low=1)
                for correction, fw, aw in product([0, 1, 2], [None, fweights], [None, aweights]):
                    check(x, correction, fw, aw)

    @skipIfNoSciPy
    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_uniform_kstest(self, device, dtype):
        from scipy import stats
        size = 1000
        for from_ in [-42, 0, 4.2]:
            for to_ in [-4.2, 0, 42]:
                if to_ > from_:
                    t = torch.empty(size, dtype=dtype, device=device).uniform_(from_, to_)
                    res = stats.kstest(t.cpu().to(torch.double), 'uniform', args=(from_, (to_ - from_)))
                    self.assertTrue(res.statistic < 0.1)

    @skipIfNoSciPy
    @dtypes(*floating_types_and(torch.half))
    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    def test_normal_kstest(self, device, dtype):
        from scipy import stats
        size = 1000
        for mean in [-10, 0, 50]:
            for std in [1, 5, 10]:
                t = torch.empty(size, dtype=dtype, device=device).normal_(mean=mean, std=std)
                res = stats.kstest(t.cpu().to(torch.double), 'norm', args=(mean, std))
                self.assertTrue(res.statistic < 0.1)

    @skipIfMPS
    @skipIfNoSciPy
    @skipRocmIfTorchInductor
    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_lognormal_kstest(self, device, dtype):
        from scipy import stats
        size = 1000
        for mean in [-3, 0, 7]:
            for std in [1, 5, 7]:
                t = torch.empty(size, dtype=dtype, device=device).log_normal_(mean=mean, std=std)
                res = stats.kstest(t.cpu().to(torch.double), 'lognorm', args=(std, 0, math.exp(mean)))
                if dtype == torch.half:
                    self.assertTrue(res.statistic < 0.3)
                else:
                    self.assertTrue(res.statistic < 0.1)

    @skipIfMPS
    @skipIfNoSciPy
    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_exponential_kstest(self, device, dtype):
        from scipy import stats
        size = 1000
        for lambd in [0.5, 1.0, 5.0]:
            t = torch.empty(size, dtype=dtype, device=device).exponential_(lambd=lambd)
            res = stats.kstest(t.cpu().to(torch.double), 'expon', args=(0, 1 / lambd,))
            self.assertTrue(res.statistic < 0.1)

    @skipIfMPS
    @skipIfNoSciPy
    @skipRocmIfTorchInductor
    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_cauchy_kstest(self, device, dtype):
        from scipy import stats
        size = 1000
        for median in [-10, 0, 50]:
            for sigma in [0.5, 1.0, 10.0]:
                t = torch.empty(size, dtype=dtype, device=device).cauchy_(median=median, sigma=sigma)
                res = stats.kstest(t.cpu().to(torch.double), 'cauchy', args=(median, sigma))
                self.assertTrue(res.statistic < 0.1)

    @slowTest
    @onlyCUDA
    @dtypes(torch.bfloat16, torch.float32)
    def test_cauchy_no_inf(self, device, dtype):
        # torch.float16 will have `inf` because of its smaller range.
        for _ in range((2**16) * 2):
            x = torch.empty((2**16), dtype=dtype, device=device)
            x.cauchy_()
            self.assertFalse(x.isinf().sum())

    @dtypes(*floating_types_and(torch.half, torch.bfloat16))
    def test_cauchy(self, device, dtype):
        a = torch.tensor([10], dtype=dtype, device=device).cauchy_(0.0, 0.5)
        self.assertEqual(a.dtype, dtype)
        self.assertEqual(a.size(), torch.Size([1]))

        # Tests extremal behavior
        t = torch.empty((1,), device=device, dtype=dtype).cauchy_(float('inf'), 0.5)
        sel

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 47 class(es): TestBasicVitalSigns, TestVitalSignsCuda, TestTorchDeviceType, _PlaceHolderOptimizer, Optimizer1, Optimizer2, TestDevicePrecision, Tracker, TestTorch, in, in, in, in, in, DummySequence, Wildcard, Foo1, Foo2, def, SubTensor

### Functions
This file defines 597 function(s): torch_vital_set, test_basic_vitals, test_basic_vitals_read_write, test_dataloader_vitals, test_cuda_vitals_gpu_only, _rand_shape, test_constants, test_bytes_to_scalar, rand_byte, test_int64_upsample3d, test_storage, test_storage_setitem, test_storage_use_count, test_tensor_storage_type, test_tensor_from_storage, test_set_storage, _check_storage_meta, test_typed_storage_meta, test_untyped_storage_meta, test_storage_meta_from_tensor, test_storage_meta_errors, test_storage_meta_ok, test_module_share_memory, test_deepcopy, test_deepcopy_scalar, check_internal_mem_overlap, unary_check_input_output_mem_overlap, _test, ternary_check_input_output_mem_overlap, _select_broadcastable_dims


## Key Components

The file contains 37288 words across 10799 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 473746 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
