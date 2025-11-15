# Documentation: test_tensor_creation_ops.py

## File Metadata
- **Path**: `test/test_tensor_creation_ops.py`
- **Size**: 205285 bytes
- **Lines**: 4438
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# Owner(s): ["module: tensor creation"]
# ruff: noqa: F841

import torch
import numpy as np

import sys
import math
import warnings
import unittest
from itertools import product, combinations, combinations_with_replacement, permutations
import random
import tempfile
from typing import Any

from torch.testing import make_tensor
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    do_test_empty_full,
    TEST_WITH_ROCM,
    suppress_warnings,
    torch_to_numpy_dtype_dict,
    numpy_to_torch_dtype_dict,
    slowTest,
    set_default_dtype,
    set_default_tensor_type,
    TEST_SCIPY,
    IS_PPC,
    IS_WINDOWS,
    IS_FBCODE,
    IS_SANDCASTLE,
    IS_S390X,
    IS_ARM64,
    parametrize,
    xfailIfTorchDynamo,
)
from torch.testing._internal.common_device_type import (
    expectedFailureMeta, instantiate_device_type_tests, deviceCountAtLeast, onlyNativeDeviceTypes,
    onlyCPU, largeTensorTest, precisionOverride, dtypes,
    onlyCUDA, skipCPUIf, dtypesIfCUDA, dtypesIfCPU, skipMeta)
from torch.testing._internal.common_dtype import (
    all_types_and_complex, all_types_and_complex_and, all_types_and, floating_and_complex_types, complex_types,
    floating_types, floating_and_complex_types_and, integral_types, integral_types_and, get_all_dtypes,
    float_to_corresponding_complex_type_map, all_types_complex_float8_and
)

from torch.utils.dlpack import to_dlpack

# TODO: replace with make_tensor
def _generate_input(shape, dtype, device, with_extremal):
    if shape == ():
        x = torch.tensor((), dtype=dtype, device=device)
    else:
        if dtype.is_floating_point or dtype.is_complex:
            # work around torch.randn not being implemented for bfloat16
            if dtype == torch.bfloat16:
                x = torch.randn(*shape, device=device) * random.randint(30, 100)
                x = x.to(torch.bfloat16)
            else:
                x = torch.randn(*shape, dtype=dtype, device=device) * random.randint(30, 100)
            x[torch.randn(*shape) > 0.5] = 0
            if with_extremal and dtype.is_floating_point:
                # Use extremal values
                x[torch.randn(*shape) > 0.5] = float('nan')
                x[torch.randn(*shape) > 0.5] = float('inf')
                x[torch.randn(*shape) > 0.5] = float('-inf')
            elif with_extremal and dtype.is_complex:
                x[torch.randn(*shape) > 0.5] = complex('nan')
                x[torch.randn(*shape) > 0.5] = complex('inf')
                x[torch.randn(*shape) > 0.5] = complex('-inf')
        elif dtype == torch.bool:
            x = torch.zeros(shape, dtype=dtype, device=device)
            x[torch.randn(*shape) > 0.5] = True
        else:
            x = torch.randint(15, 100, shape, dtype=dtype, device=device)

    return x


# TODO: replace with make_tensor
def _rand_shape(dim, min_size, max_size):
    shape = []
    for _ in range(dim):
        shape.append(random.randint(min_size, max_size))
    return tuple(shape)

# Test suite for tensor creation ops
#
# Includes creation functions like torch.eye, random creation functions like
#   torch.rand, and *like functions like torch.ones_like.
# DOES NOT INCLUDE view ops, which are tested in TestViewOps (currently in
#   test_torch.py) OR numpy interop (which is also still tested in test_torch.py)
#
# See https://pytorch.org/docs/main/torch.html#creation-ops

class TestTensorCreation(TestCase):
    exact_dtype = True

    @onlyCPU
    @dtypes(torch.float)
    def test_diag_embed(self, device, dtype):
        x = torch.arange(3 * 4, dtype=dtype, device=device).view(3, 4)
        result = torch.diag_embed(x)
        expected = torch.stack([torch.diag(r) for r in x], 0)
        self.assertEqual(result, expected)

        result = torch.diag_embed(x, offset=1, dim1=0, dim2=2)
        expected = torch.stack([torch.diag(r, 1) for r in x], 1)
        self.assertEqual(result, expected)

    def test_cat_mem_overlap(self, device):
        x = torch.rand((1, 3), device=device).expand((6, 3))
        y = torch.rand((3, 3), device=device)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            torch.cat([y, y], out=x)

    @onlyNativeDeviceTypes
    def test_vander(self, device):
        x = torch.tensor([1, 2, 3, 5], device=device)

        self.assertEqual((0, 0), torch.vander(torch.tensor([]), 0).shape)

        with self.assertRaisesRegex(RuntimeError, "N must be non-negative."):
            torch.vander(x, N=-1)

        with self.assertRaisesRegex(RuntimeError, "x must be a one-dimensional tensor."):
            torch.vander(torch.stack((x, x)))

    @onlyNativeDeviceTypes
    @dtypes(torch.bool, torch.uint8, torch.int8, torch.short, torch.int, torch.long,
            torch.float, torch.double,
            torch.cfloat, torch.cdouble)
    def test_vander_types(self, device, dtype):
        if dtype is torch.uint8:
            # Note: no negative uint8 values
            X = [[1, 2, 3, 5], [0, 1 / 3, 1, math.pi, 3 / 7]]
        elif dtype is torch.bool:
            # Note: see https://github.com/pytorch/pytorch/issues/37398
            # for why this is necessary.
            X = [[True, True, True, True], [False, True, True, True, True]]
        elif dtype in [torch.cfloat, torch.cdouble]:
            X = [[1 + 1j, 1 + 0j, 0 + 1j, 0 + 0j],
                 [2 + 2j, 3 + 2j, 4 + 3j, 5 + 4j]]
        else:
            X = [[1, 2, 3, 5], [-math.pi, 0, 1 / 3, 1, math.pi, 3 / 7]]

        N = [None, 0, 1, 3]
        increasing = [False, True]

        for x, n, inc in product(X, N, increasing):
            numpy_dtype = torch_to_numpy_dtype_dict[dtype]
            pt_x = torch.tensor(x, device=device, dtype=dtype)
            np_x = np.array(x, dtype=numpy_dtype)

            pt_res = torch.vander(pt_x, increasing=inc) if n is None else torch.vander(pt_x, n, inc)
            np_res = np.vander(np_x, n, inc)

            self.assertEqual(
                pt_res,
                torch.from_numpy(np_res),
                atol=1e-3,
                rtol=0,
                exact_dtype=False)

    def test_cat_all_dtypes_and_devices(self, device):
        for dt in all_types_and_complex_and(
            torch.half,
            torch.bool,
            torch.bfloat16,
            torch.chalf,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
        ):
            x = torch.tensor([[1, 2], [3, 4]], dtype=dt, device=device)

            expected1 = torch.tensor([[1, 2], [3, 4], [1, 2], [3, 4]], dtype=dt, device=device)
            self.assertEqual(torch.cat((x, x), 0), expected1)

            expected2 = torch.tensor([[1, 2, 1, 2], [3, 4, 3, 4]], dtype=dt, device=device)
            self.assertEqual(torch.cat((x, x), 1), expected2)

    def test_fill_all_dtypes_and_devices(self, device):
        for dt in all_types_complex_float8_and(torch.half, torch.bool, torch.bfloat16, torch.chalf):
            for x in [torch.tensor((10, 10), dtype=dt, device=device),
                      torch.empty(10000, dtype=dt, device=device)]:  # large tensor
                numel = x.numel()
                bound_dtypes = (torch.uint8, torch.int8, torch.float8_e4m3fn,
                                torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz)
                bound = 100 if dt in bound_dtypes else 2000
                for n in range(-bound, bound, bound // 10):
                    x.fill_(n)
                    self.assertEqual(x, torch.tensor([n] * numel, dtype=dt, device=device))
                    self.assertEqual(dt, x.dtype)

    def test_roll(self, device):
        numbers = torch.arange(1, 9, device=device)

        single_roll = numbers.roll(1, 0)
        expected = torch.tensor([8, 1, 2, 3, 4, 5, 6, 7], device=device)
        self.assertEqual(single_roll, expected, msg=f"{single_roll} did not equal expected result")

        roll_backwards = numbers.roll(-2, 0)
        expected = torch.tensor([3, 4, 5, 6, 7, 8, 1, 2], device=device)
        self.assertEqual(roll_backwards, expected, msg=f"{roll_backwards} did not equal expected result")

        data = numbers.view(2, 2, 2)
        rolled = data.roll(1, 0)
        expected = torch.tensor([5, 6, 7, 8, 1, 2, 3, 4], device=device).view(2, 2, 2)
        self.assertEqual(expected, rolled, msg=f"{rolled} did not equal expected result: {expected}")

        data = data.view(2, 4)
        # roll a loop until back where started
        loop_rolled = data.roll(2, 0).roll(4, 1)
        self.assertEqual(data, loop_rolled, msg=f"{loop_rolled} did not equal the original: {data}")
        # multiple inverse loops
        self.assertEqual(data, data.roll(-20, 0).roll(-40, 1))
        self.assertEqual(torch.tensor([8, 1, 2, 3, 4, 5, 6, 7], device=device), numbers.roll(1, 0))

        # test non-contiguous
        # strided equivalent to numbers.as_strided(size=(4, 2), stride=(1, 4))
        strided = numbers.view(2, 4).transpose(0, 1)
        self.assertFalse(strided.is_contiguous(), "this test needs a non-contiguous tensor")
        expected = torch.tensor([4, 8, 1, 5, 2, 6, 3, 7]).view(4, 2)
        rolled = strided.roll(1, 0)
        self.assertEqual(expected, rolled,
                         msg=f"non contiguous tensor rolled to {rolled} instead of {expected} ")

        # test roll with no dimension specified
        expected = numbers.roll(1, 0).view(2, 4)
        self.assertEqual(expected, data.roll(1), msg="roll with no dims should flatten and roll.")
        self.assertEqual(expected, data.roll(1, dims=None), msg="roll with no dims should flatten and roll.")

        # test roll over multiple dimensions
        expected = torch.tensor([[7, 8, 5, 6], [3, 4, 1, 2]], device=device)
        double_rolled = data.roll(shifts=(2, -1), dims=(1, 0))
        self.assertEqual(double_rolled, expected,
                         msg=f"should be able to roll over two dimensions, got {double_rolled}")

        self.assertRaisesRegex(RuntimeError, "required", lambda: data.roll(shifts=(), dims=()))
        self.assertRaisesRegex(RuntimeError, "required", lambda: data.roll(shifts=(), dims=1))
        # shifts/dims should align
        self.assertRaisesRegex(RuntimeError, "align", lambda: data.roll(shifts=(1, 2), dims=(1,)))
        self.assertRaisesRegex(RuntimeError, "align", lambda: data.roll(shifts=(1,), dims=(1, 2)))

        # test bool tensor
        t = torch.zeros(6, dtype=torch.bool, device=device)
        t[0] = True
        t[3] = True
        self.assertEqual(torch.tensor([False, True, False, False, True, False]), t.roll(1, 0))

        # test complex tensor
        t = torch.tensor([1, 2 + 1j, 3.5, 4. + 2j, 5j, 6.], device=device)
        t[0] = 1 + 0.5j
        t[3] = 4.
        expected = torch.tensor([6., 1 + 0.5j, 2 + 1j, 3.5, 4., 5j], device=device)
        self.assertEqual(expected, t.roll(1, 0))

    def test_diagflat(self, device):
        dtype = torch.float32
        # Basic sanity test
        x = torch.randn((100,), dtype=dtype, device=device)
        result = torch.diagflat(x)
        expected = torch.diag(x)
        self.assertEqual(result, expected)

        # Test offset
        x = torch.randn((100,), dtype=dtype, device=device)
        result = torch.diagflat(x, 17)
        expected = torch.diag(x, 17)
        self.assertEqual(result, expected)

        # Test where input has more than one dimension
        x = torch.randn((2, 3, 4), dtype=dtype, device=device)
        result = torch.diagflat(x)
        expected = torch.diag(x.contiguous().view(-1))
        self.assertEqual(result, expected)

        # Noncontig input
        x = torch.randn((2, 3, 4), dtype=dtype, device=device).transpose(2, 0)
        self.assertFalse(x.is_contiguous())
        result = torch.diagflat(x)
        expected = torch.diag(x.contiguous().view(-1))
        self.assertEqual(result, expected)

        # Complex number support
        result = torch.diagflat(torch.ones(4, dtype=torch.complex128))
        expected = torch.eye(4, dtype=torch.complex128)
        self.assertEqual(result, expected)

    def test_block_diag(self, device):
        def block_diag_workaround(*arrs):
            arrs_expanded = []
            for a in arrs:
                if a.dim() == 2:
                    arrs_expanded.append(a)
                elif a.dim() == 1:
                    arrs_expanded.append(a.expand(1, a.size(0)))
                elif a.dim() == 0:
                    arrs_expanded.append(a.expand(1, 1))
            shapes = torch.tensor([a.shape for a in arrs_expanded], device=device)
            out = torch.zeros(
                torch.sum(shapes, dim=0).tolist(),
                dtype=arrs_expanded[0].dtype,
                device=device
            )
            r, c = 0, 0
            for i, (rr, cc) in enumerate(shapes):
                out[r:r + rr, c:c + cc] = arrs_expanded[i]
                r += rr
                c += cc
            return out

        tensors = [
            torch.rand((2, 2), device=device),
            torch.rand((2, 3), device=device),
            torch.rand(10, device=device),
            torch.rand((8, 1), device=device),
            torch.rand(1, device=device)[0]
        ]
        result = torch.block_diag(*tensors)
        result_check = block_diag_workaround(*tensors)
        self.assertEqual(result, result_check)

        tensor = torch.rand(1, device=device)[0]
        result = torch.block_diag(tensor)
        result_check = tensor.expand(1, 1)
        self.assertEqual(result, result_check)

        tensor = torch.rand(10, device=device)
        result = torch.block_diag(tensor)
        result_check = tensor.expand(1, tensor.size(0))
        self.assertEqual(result, result_check)

        result = torch.block_diag()
        result_check = torch.empty(1, 0, device=device)
        self.assertEqual(result, result_check)
        self.assertEqual(result.device.type, 'cpu')

        test_dtypes = [
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128
        ]
        # Test pairs of different dtypes
        for dtype1 in test_dtypes:
            for dtype2 in test_dtypes:
                a = torch.tensor(1, device=device, dtype=dtype1)
                b = torch.tensor(2, device=device, dtype=dtype2)
                result = torch.block_diag(a, b)
                result_dtype = torch.result_type(a, b)
                result_check = torch.tensor([[1, 0], [0, 2]], device=device, dtype=result_dtype)
                self.assertEqual(result, result_check)

        with self.assertRaisesRegex(
            RuntimeError,
            "torch.block_diag: Input tensors must have 2 or fewer dimensions. Input 1 has 3 dimensions"
        ):
            torch.block_diag(torch.tensor(5), torch.tensor([[[6]]]))

        with self.assertRaisesRegex(
            RuntimeError,
            "torch.block_diag: Input tensors must have 2 or fewer dimensions. Input 0 has 4 dimensions"
        ):
            torch.block_diag(torch.tensor([[[[6]]]]))

        if device != 'cpu':
            with self.assertRaisesRegex(
                RuntimeError,
                (
                    "torch.block_diag: input tensors must all be on the same device."
                    " Input 0 is on device cpu and input 1 is on device "
                )
            ):
                torch.block_diag(torch.ones(2, 2).cpu(), torch.ones(2, 2, device=device))

    @unittest.skipIf(not TEST_SCIPY, "Scipy not found")
    def test_block_diag_scipy(self, device):
        import scipy.linalg
        scipy_tensors_list = [
            [
                1,
                [2],
                [],
                [3, 4, 5],
                [[], []],
                [[6], [7.3]]
            ],
            [
                [[1, 2], [3, 4]],
                [1]
            ],
            [
                [[4, 9], [7, 10]],
                [4.6, 9.12],
                [1j + 3]
            ],
            []
        ]

        expected_torch_types = [
            torch.float32,
            torch.int64,
            torch.complex64,
            torch.float32
        ]

        expected_scipy_types = [
            torch.float64,
            # windows scipy block_diag returns int32 types
            torch.int32 if IS_WINDOWS else torch.int64,
            torch.complex128,
            torch.float64
        ]

        for scipy_tensors, torch_type, scipy_type in zip(scipy_tensors_list, expected_torch_types, expected_scipy_types):
            torch_tensors = [torch.tensor(t, device=device) for t in scipy_tensors]
            torch_result = torch.block_diag(*torch_tensors)
            self.assertEqual(torch_result.dtype, torch_type)

            scipy_result = torch.tensor(
                scipy.linalg.block_diag(*scipy_tensors),
                device=device
            )
            self.assertEqual(scipy_result.dtype, scipy_type)
            scipy_result = scipy_result.to(torch_type)

            self.assertEqual(torch_result, scipy_result)

    @onlyNativeDeviceTypes
    @dtypes(torch.half, torch.float32, torch.float64)
    def test_torch_complex(self, device, dtype):
        real = torch.tensor([1, 2], device=device, dtype=dtype)
        imag = torch.tensor([3, 4], device=device, dtype=dtype)
        z = torch.complex(real, imag)
        complex_dtype = float_to_corresponding_complex_type_map[dtype]
        self.assertEqual(torch.tensor([1.0 + 3.0j, 2.0 + 4.0j], dtype=complex_dtype), z)

    @onlyNativeDeviceTypes
    @dtypes(torch.float32, torch.float64)
    def test_torch_polar(self, device, dtype):
        abs = torch.tensor([1, 2, -3, -4.5, 1, 1], device=device, dtype=dtype)
        angle = torch.tensor([math.pi / 2, 5 * math.pi / 4, 0, -11 * math.pi / 6, math.pi, -math.pi],
                             device=device, dtype=dtype)
        z = torch.polar(abs, angle)
        complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        self.assertEqual(torch.tensor([1j, -1.41421356237 - 1.41421356237j, -3,
                                       -3.89711431703 - 2.25j, -1, -1],
                                      dtype=complex_dtype),
                         z, atol=1e-5, rtol=1e-5)

    @onlyNativeDeviceTypes
    @dtypes(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.complex64, torch.complex128, torch.bool)
    def test_torch_complex_floating_dtype_error(self, device, dtype):
        for op in (torch.complex, torch.polar):
            a = torch.tensor([1, 2], device=device, dtype=dtype)
            b = torch.tensor([3, 4], device=device, dtype=dtype)
            error = r"Expected both inputs to be Half, Float or Double tensors but " \
                    r"got [A-Za-z]+ and [A-Za-z]+"
            with self.assertRaisesRegex(RuntimeError, error):
                op(a, b)

    @onlyNativeDeviceTypes
    @dtypes(torch.float32, torch.float64)
    def test_torch_complex_same_dtype_error(self, device, dtype):

        def dtype_name(dtype):
            return 'Float' if dtype == torch.float32 else 'Double'

        for op in (torch.complex, torch.polar):
            other_dtype = torch.float64 if dtype == torch.float32 else torch.float32
            a = torch.tensor([1, 2], device=device, dtype=dtype)
            b = torch.tensor([3, 4], device=device, dtype=other_dtype)
            error = f"Expected object of scalar type {dtype_name(dtype)} but got scalar type " \
                    f"{dtype_name(other_dtype)} for second argument"
            with self.assertRaisesRegex(RuntimeError, error):
                op(a, b)

    @onlyNativeDeviceTypes
    @dtypes(torch.float32, torch.float64)
    def test_torch_complex_out_dtype_error(self, device, dtype):

        def dtype_name(dtype):
            return 'Float' if dtype == torch.float32 else 'Double'

        def complex_dtype_name(dtype):
            return 'ComplexFloat' if dtype == torch.complex64 else 'ComplexDouble'

        for op in (torch.complex, torch.polar):
            a = torch.tensor([1, 2], device=device, dtype=dtype)
            b = torch.tensor([3, 4], device=device, dtype=dtype)
            out = torch.zeros(2, device=device, dtype=dtype)
            expected_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
            error = f"Expected object of scalar type {complex_dtype_name(expected_dtype)} but got scalar type " \
                    f"{dtype_name(dtype)} for argument 'out'"
            with self.assertRaisesRegex(RuntimeError, error):
                op(a, b, out=out)

    def test_cat_empty_legacy(self, device):
        # FIXME: this is legacy behavior and should be removed
        # when we support empty tensors with arbitrary sizes
        dtype = torch.float32

        x = torch.randn((4, 3, 32, 32), dtype=dtype, device=device)
        empty = torch.randn((0,), dtype=dtype, device=device)

        res1 = torch.cat([x, empty], dim=1)
        res2 = torch.cat([empty, x], dim=1)
        self.assertEqual(res1, res2)

        res1 = torch.cat([empty, empty], dim=1)
        self.assertEqual(res1, empty)

    def test_cat_empty(self, device):
        dtype = torch.float32

        x = torch.randn((4, 3, 32, 32), dtype=dtype, device=device)
        empty = torch.randn((4, 0, 32, 32), dtype=dtype, device=device)

        res1 = torch.cat([x, empty], dim=1)
        res2 = torch.cat([empty, x], dim=1)
        self.assertEqual(res1, res2)

        res1 = torch.cat([empty, empty], dim=1)
        self.assertEqual(res1, empty)

    def test_concat_empty_list_error(self, device):
        # Regression test for https://github.com/pytorch/pytorch/issues/155306
        msg = "expected a non-empty list of Tensors"
        with self.assertRaisesRegex(ValueError, msg):
            torch.concat([], dim='N')
        with self.assertRaisesRegex(ValueError, msg):
            torch.concatenate([], dim='N')

    def test_cat_out(self, device):
        x = torch.zeros((0), device=device)
        y = torch.randn((4, 6), device=device)

        w = y.view(-1).clone()
        a = torch.cat([w[:2], w[4:6]])
        b = torch.cat([w[:2], w[4:6]], out=w[6:10])
        self.assertEqual(a, b)
        self.assertEqual(a, w[6:10])
        self.assertEqual(w[:6], y.view(-1)[:6])

        # Case:
        # Reference: https://github.com/pytorch/pytorch/issues/49878
        for dim in [0, 1]:
            x = torch.zeros((10, 5, 2), device=device)

            random_length = random.randint(1, 4)
            y = x.narrow(dim, 0, x.shape[dim] - random_length)
            val = torch.full_like(y[0], 3., device=device)

            if dim == 0:
                self.assertTrue(y.is_contiguous())
            else:
                self.assertFalse(y.is_contiguous())

            torch.cat((val[None],) * y.shape[0], dim=0, out=y)

            expected_y = torch.cat((val[None],) * y.shape[0], dim=0)
            expected_x = torch.zeros((10, 5, 2), device=device)
            if dim == 0:
                expected_x[:x.shape[dim] - random_length, :, :] = expected_y
            elif dim == 1:
                expected_x[:, :x.shape[dim] - random_length, :] = expected_y

            self.assertEqual(y, expected_y)
            self.assertEqual(x, expected_x)

    @dtypes(*all_types_and_complex(), torch.uint16, torch.uint32, torch.uint64)
    def test_cat_out_fast_path_dim0_dim1(self, device, dtype):
        int_types = integral_types_and(torch.uint16, torch.uint32, torch.uint64)
        x = torch.zeros((0), device=device, dtype=dtype)
        if dtype in int_types:
            y = torch.randint(low=0, high=100, size=(4, 6), device=device, dtype=dtype)
        else:
            y = torch.randn((4, 6), device=device, dtype=dtype)
        # Test concat on dimension 0
        w = y.view(-1).clone()
        a = torch.cat([w[:2], w[4:6]])
        b = torch.cat([w[:2], w[4:6]], out=w[6:10])
        # Note that there is no guarantee that slicing here will result in
        # contiguous tensors
        self.assertEqual(a, b)
        self.assertEqual(a, w[6:10])
        self.assertEqual(w[:6], y.view(-1)[:6])
        # If inputs are contiguous tensors, then fast concat paths will be invoked
        a_fastcat = torch.cat([w[:2].contiguous(), w[4:6].contiguous()])
        self.assertEqual(a_fastcat, a)
        # Test concat on dimension 1
        w = y.clone()
        w_slices = torch.tensor_split(w, (2, 4), dim=1)
        # Note that the tensor in w_slices[] here may not be a contiguous
        # tensor and we need to make sure this is not broken by fast concat
        b = torch.cat([w_slices[0], w_slices[1]], dim=1)
        expected_b = torch.index_select(w, 1, torch.tensor([0, 1, 2, 3], device=device))
        self.assertEqual(b, expected_b)
        # If inputs are contiguous tensors, then fast concat paths will be invoked
        b_fastcat = torch.cat([w_slices[0].contiguous(), w_slices[1].contiguous()], dim=1)
        self.assertEqual(b_fastcat, expected_b)
        # Finally, we need to make sure backward is not broken
        # Integral types will not have grad
        if dtype not in int_types:
            a = torch.randn((4, 3), device=device, dtype=dtype, requires_grad=True)
            b = torch.randn((2, 3), device=device, dtype=dtype, requires_grad=True)
            c = torch.randn((5, 3), device=device, dtype=dtype, requires_grad=True)
            d = torch.randn((5, 2), device=device, dtype=dtype, requires_grad=True)
            expected_a_grad = torch.ones((4, 3), device=device, dtype=dtype)
            expected_b_grad = torch.ones((2, 3), device=device, dtype=dtype)
            expected_c_grad = torch.ones((5, 3), device=device, dtype=dtype)
            expected_d_grad = torch.ones((5, 2), device=device, dtype=dtype)
            # All the new tensors should be contiguous here. Let us make sure
            # to explicitly set them contiguous to enforce fast cat
            dim0_cat = torch.cat([a.contiguous(), b.contiguous()], dim=0)
            if dtype in complex_types():
                dim0_cat.sum().abs().backward()
                self.assertEqual(a.grad.abs(), expected_a_grad.abs())
                self.assertEqual(b.grad.abs(), expected_b_grad.abs())
            else:
                dim0_cat.sum().backward()
                self.assertEqual(a.grad, expected_a_grad)
                self.assertEqual(b.grad, expected_b_grad)
            dim1_cat = torch.cat([c.contiguous(), d.contiguous()], dim=1)
            if dtype in complex_types():
                dim1_cat.sum().abs().backward()
                self.assertEqual(c.grad.abs(), expected_c_grad.abs())
                self.assertEqual(d.grad.abs(), expected_d_grad.abs())
            else:
                dim1_cat.sum().backward()
                self.assertEqual(c.grad, expected_c_grad)
                self.assertEqual(d.grad, expected_d_grad)

    def test_cat_out_channels_last(self, device):
        x = torch.randn((4, 3, 8, 8))
        y = torch.randn(x.shape)
        res1 = torch.cat((x, y))
        z = res1.clone().contiguous(memory_format=torch.channels_last)
        res2 = torch.cat((x, y), out=z)
        self.assertEqual(res1, res2)

    @onlyNativeDeviceTypes
    def test_cat_in_channels_last(self, device):
        for dim in range(4):
            x = torch.randn((4, 15, 8, 8), device=device)
            y = torch.randn(x.shape, device=device)
            res1 = torch.cat((x, y), dim=dim)
            x = x.clone().contiguous(memory_format=torch.channels_last)
            y = y.clone().contiguous(memory_format=torch.channels_last)
            res2 = torch.cat((x, y), dim=dim)
            self.assertTrue(res2.is_contiguous(memory_format=torch.channels_last))
            self.assertEqual(res1, res2)

            # Size larger than grain size.
            x = torch.randn((4, 15, 256, 256), device=device)
            y = torch.randn(x.shape, device=device)
            res1 = torch.cat((x, y), dim=dim)
            x = x.clone().contiguous(memory_format=torch.channels_last)
            y = y.clone().contiguous(memory_format=torch.channels_last)
            res2 = torch.cat((x, y), dim=dim)
            self.assertTrue(res2.is_contiguous(memory_format=torch.channels_last))
            self.assertEqual(res1, res2)

    @onlyNativeDeviceTypes
    def test_cat_preserve_channels_last(self, device):
        x = torch.randn((4, 3, 8, 8), device=device)
        y = torch.randn(x.shape, device=device)
        res1 = torch.cat((x, y))
        res2 = torch.cat((x.contiguous(memory_format=torch.channels_last), y.contiguous(memory_format=torch.channels_last)))
        self.assertEqual(res1, res2)
        self.assertTrue(res2.is_contiguous(memory_format=torch.channels_last))
        # discontiguous channels-last inputs
        x = torch.arange(24, dtype=torch.float, device=device).reshape(2, 2, 3, 2).to(memory_format=torch.channels_last)
        x1 = x[:, :, :2]
        x2 = x[:, :, 1:]
        res1 = torch.cat((x1, x2), dim=-1)
        res2 = torch.cat((x1.contiguous(), x2.contiguous()), dim=-1)
        self.assertEqual(res1, res2)
        self.assertTrue(res1.is_contiguous(memory_format=torch.channels_last))

    @onlyCUDA
    def test_cat_channels_last_large_inputs(self, device):
        num_tensors = 130
        inputs_cuda = [
            torch.randn((2, 3, 4, 4), device=device).contiguous(memory_format=torch.channels_last)
            for _ in range(num_tensors)
        ]
        inputs_cpu = [t.cpu() for t in inputs_cuda]

        result = torch.cat(inputs_cuda, dim=1)
        expected = torch.cat(inputs_cpu, dim=1)

        self.assertEqual(result.cpu(), expected)
        self.assertTrue(result.is_contiguous(memory_format=torch.channels_last))

    @onlyCUDA
    def test_cat_out_memory_format(self, device):
        inp_size = (4, 4, 4, 4)
        expected_size = (8, 4, 4, 4)
        a_cuda = torch.randn(inp_size, device=device).contiguous(memory_format=torch.channels_last)
        a_cpu = torch.randn(inp_size, device='cpu').contiguous(memory_format=torch.channels_last)
        b_cuda = torch.randn(inp_size, device=device).contiguous(memory_format=torch.contiguous_format)
        b_cpu = torch.randn(inp_size, device='cpu').contiguous(memory_format=torch.contiguous_format)
        c_cuda = torch.randn(inp_size, device=device).contiguous(memory_format=torch.channels_last)

        # Case 1: if out= is the correct shape then the memory format of out= is respected

        out_cuda = torch.empty(expected_size, device=device).contiguous(memory_format=torch.contiguous_format)
        res1_cuda = torch.cat((a_cuda, b_cuda), out=out_cuda)

        out_cpu = torch.empty(expected_size, device='cpu').contiguous(memory_format=torch.contiguous_format)
        res1_cpu = torch.cat((a_cpu, b_cpu), out=out_cpu)

        self.assertTrue(res1_cuda.is_contiguous(memory_format=torch.contiguous_format))
        self.assertTrue(res1_cpu.is_contiguous(memory_format=torch.contiguous_format))

        # Case 2: if out= is not the correct shape then the output it is resized internally
        # - For both CPU and CUDA variants, it only propagates memory format if all the tensors have
        #   the same memory format, otherwise it just uses contiguous_format as a default

        out_cuda = torch.empty((0), device=device).contiguous(memory_format=torch.contiguous_format)
        # a_cuda and b_cuda have different memory_format
        res2_cuda = torch.cat((a_cuda, b_cuda), out=out_cuda)

        out_cpu = torch.empty((0), device='cpu').contiguous(memory_format=torch.contiguous_format)
        res2_cpu = torch.cat((a_cpu, b_cpu), out=out_cpu)

        self.assertTrue(res2_cuda.is_contiguous(memory_format=torch.contiguous_format))
        self.assertTrue(res2_cpu.is_contiguous(memory_format=torch.contiguous_format))

        out_cuda = torch.empty((0), device=device).contiguous(memory_format=torch.contiguous_format)
        # a_cuda and c_cuda have same memory_format
        res3_cuda = torch.cat((a_cuda, c_cuda), out=out_cuda)

        self.assertTrue(res3_cuda.is_contiguous(memory_format=torch.channels_last))

    @onlyCUDA
    def test_cat_stack_cross_devices(self, device):
        cuda = torch.randn((3, 3), device=device)
        cpu = torch.randn((3, 3), device='cpu')

        # Stack
        with self.assertRaisesRegex(RuntimeError,
                                    "Expected all tensors to be on the same device"):
            torch.stack((cuda, cpu))
        with self.assertRaisesRegex(RuntimeError,
                                    "Expected all tensors to be on the same device"):
            torch.stack((cpu, cuda))

    # TODO: reconcile with other cat tests
    # TODO: Compare with a NumPy reference instead of CPU
    @onlyCUDA
    def test_cat(self, device):
        SIZE = 10
        for dim in range(-3, 3):
            pos_dim = dim if dim >= 0 else 3 + dim
            x = torch.rand(13, SIZE, SIZE, device=device).transpose(0, pos_dim)
            y = torch.rand(17, SIZE, SIZE, device=device).transpose(0, pos_dim)
            z = torch.rand(19, SIZE, SIZE, device=device).transpose(0, pos_dim)

            res1 = torch.cat((x, y, z), dim)
            self.assertEqual(res1.narrow(pos_dim, 0, 13), x, atol=0, rtol=0)
            self.assertEqual(res1.narrow(pos_dim, 13, 17), y, atol=0, rtol=0)
            self.assertEqual(res1.narrow(pos_dim, 30, 19), z, atol=0, rtol=0)

        x = torch.randn(20, SIZE, SIZE, device=device)
        self.assertEqual(torch.cat(torch.split(x, 7)), x)
        self.assertEqual(torch.cat(torch.chunk(x, 7)), x)

        y = torch.randn(1, SIZE, SIZE, device=device)
        z = torch.cat([x, y])
        self.assertEqual(z.size(), (21, SIZE, SIZE))

    # TODO: update this test to compare against NumPy instead of CPU
    @onlyCUDA
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_device_rounding(self, device, dtype):
        # test half-to-even
        a = [-5.8, -3.5, -2.3, -1.5, -0.5, 0.5, 1.5, 2.3, 3.5, 5.8]
        res = [-6., -4., -2., -2., 0., 0., 2., 2., 4., 6.]

        a_tensor = torch.tensor(a, device=device).round()
        res_tensor = torch.tensor(res, device='cpu')
        self.assertEqual(a_tensor, res_tensor)

    # Note: This test failed on XLA since its test cases are created by empty_strided which
    #       doesn't support overlapping sizes/strides in XLA impl
    @onlyNativeDeviceTypes
    def test_like_fn_stride_proparation_vs_tensoriterator_unary_op(self, device):
        # Test like functions against tensoriterator based unary operator (exp) to
        # make sure the returned tensor from like function follows the same stride propergation
        # rule as what tensoriterator does for unary operator. The like function's  output strides
        # is computed on CPU side always, no need to test GPU here.

        def compare_helper_(like_fn, t):
            te = torch.exp(t)
            tl = like_fn(t)
            self.assertEqual(te.stride(), tl.stride())
            self.assertEqual(te.size(), tl.size())

        like_fns = [
            lambda t, **kwargs: torch.zeros_like(t, **kwargs),
            lambda t, **kwargs: torch.ones_like(t, **kwargs),
            lambda t, **kwargs: torch.randint_like(t, 10, 100, **kwargs),
            lambda t, **kwargs: torch.randint_like(t, 100, **kwargs),
            lambda t, **kwargs: torch.randn_like(t, **kwargs),
            lambda t, **kwargs: torch.rand_like(t, **kwargs),
            lambda t, **kwargs: torch.full_like(t, 7, **kwargs),
            lambda t, **kwargs: torch.empty_like(t, **kwargs)]

        # dense non-overlapping tensor,
        # non-dense non-overlapping sliced tensor
        # non-dense non-overlapping gapped tensor
        # non-dense non-overlapping 0 strided tensor
        # non-dense overlapping general tensor
        # non-dense overlapping sliced tensor
        # non-dense overlapping gapped tensor
        # non-dense overlapping 0 strided tensor
        # non-dense overlapping equal strides
        tset = (
            torch.randn(4, 3, 2, device=device),
            torch.randn(4, 3, 2, device=device)[:, :, ::2],
            torch.empty_strided((4, 3, 2), (10, 3, 1), device=device).fill_(1.0),
            torch.empty_strided((4, 3, 2), (10, 0, 3), device=device).fill_(1.0),
            torch.empty_strided((4, 3, 2), (10, 1, 2), device=device).fill_(1.0),
            torch.empty_strided((4, 3, 2), (4, 2, 1), device=device)[:, :, ::2].fill_(1.0),
            torch.empty_strided((4, 3, 2), (10, 1, 1), device=device).fill_(1.0),
            torch.empty_strided((4, 1, 1, 2), (10, 0, 0, 2), device=device).fill_(1.0),
            torch.empty_strided((4, 2, 3), (10, 3, 3), device=device).fill_(1.0))

        for like_fn in like_fns:
            for t in tset:
                for p in permutations(range(t.dim())):
                    tp = t.permute(p)
                    compare_helper_(like_fn, tp)

    def _hvd_split_helper(self, torch_fn, np_fn, op_name, inputs, device, dtype, dim):
        dimension_error_message = op_name + " requires a tensor with at least "
        divisibiliy_error_message = op_name + " attempted to split along dimension "

        for shape, arg in inputs:
            direction = dim - (len(shape) == 1 and dim == 1)
            bound = dim + 2 * (dim == 0) + (dim == 2)
            error_expected = len(shape) < bound or (not isinstance(arg, list) and shape[direction] % arg != 0)

            t = make_tensor(shape, dtype=dtype, device=device)
            t_np = t.cpu().numpy()

            if not error_expected:
                self.assertEqual(torch_fn(t, arg), np_fn(t_np, arg))
            else:
                self.assertRaises(RuntimeError, lambda: torch_fn(t, arg))
                self.assertRaises(ValueError, lambda: np_fn(t, arg))
                expected_error_message = dimension_error_message if len(shape) < bound else divisibiliy_error_message
                self.assertRaisesRegex(RuntimeError, expected_error_message, lambda: torch_fn(t, arg))

    @onlyNativeDeviceTypes
    @dtypes(torch.long, torch.float32, torch.complex64)
    def test_hsplit(self, device, dtype):
        inputs = (
            ((), 3),
            ((), [2, 4, 6]),
            ((6,), 2),
            ((6,), 4),
            ((6,), [2, 5]),
            ((6,), [7, 9]),
            ((3, 8), 4),
            ((3, 8), 5),
            ((3, 8), [1, 5]),
            ((3, 8), [3, 8]),
            ((5, 5, 5), 2),
            ((5, 5, 5), [1, 4]),
            ((5, 0, 5), 3),
            ((5, 5, 0), [2, 6]),
        )
        self._hvd_split_helper(torch.hsplit, np.hsplit, "torch.hsplit", inputs, device, dtype, 1)

    @onlyNativeDeviceTypes
    @dtypes(torch.long, torch.float32, torch.complex64)
    def test_vsplit(self, device, dtype):
        inputs = (
            ((6,), 2),
            ((6,), 4),
            ((6, 5), 2),
            ((6, 5), 4),
            ((6, 5), [1, 2, 3]),
            ((6, 5), [1, 5, 9]),
            ((6, 5, 5), 2),
            ((6, 0, 5), 2),
            ((5, 0, 5), [1, 5]),
        )
        self._hvd_split_helper(torch.vsplit, np.vsplit, "torch.vsplit", inputs, device, dtype, 0)

    @onlyNativeDeviceTypes
    @dtypes(torch.long, torch.float32, torch.complex64)
    def test_dsplit(self, device, dtype):
        inputs = (
            ((6,), 4),
            ((6, 6), 3),
            ((5, 5, 6), 2),
            ((5, 5, 6), 4),
            ((5, 5, 6), [1, 2, 3]),
            ((5, 5, 6), [1, 5, 9]),
            ((5, 5, 0), 2),
            ((5, 0, 6), 4),
            ((5, 0, 6), [1, 2, 3]),
            ((5, 5, 6), [1, 5, 9]),
        )
        self._hvd_split_helper(torch.dsplit, np.dsplit, "torch.dsplit", inputs, device, dtype, 2)

    def _test_special_stacks(self, dim, at_least_dim, torch_fn, np_fn, device, dtype):
        # Test error for non-tuple argument
        t = torch.randn(10)
        with self.assertRaisesRegex(TypeError, "must be tuple of Tensors, not Tensor"):
            torch_fn(t)
        # Test error for a single array
        with self.assertRaisesRegex(TypeError, "must be tuple of Tensors, not Tensor"):
            torch_fn(t)

        # Test 0-D
        num_tensors = random.randint(1, 5)
        input_t = [torch.tensor(random.uniform(0, 10), device=device, dtype=dtype) for i in range(num_tensors)]
        actual = torch_fn(input_t)
        expected = np_fn([input.cpu().numpy() for input in input_t])
        self.assertEqual(actual, expected)

        for ndims in range(1, 5):
            base_shape = list(_rand_shape(ndims, min_size=1, max_size=5))
            for i in range(ndims):
                shape = list(base_shape)
                num_tensors = random.randint(1, 5)
                torch_input = []
                # Create tensors with shape being different along one axis only
                for _ in range(num_tensors):
                    shape[i] = random.randint(1, 5)
                    torch_input.append(_generate_input(tuple(shape), dtype, device, with_extremal=False))

                # Determine if input tensors have valid dimensions.
                valid_dim = True
                for k in range(len(torch_input) - 1):
                    for tdim in range(ndims):
                        # Test whether all tensors have the same shape except in concatenating dimension
                        # Unless the number of dimensions is less than the corresponding at_least function dimension
                        # Since the original concatenating dimension would shift after applying at_least and would no
                        # longer be the concatenating dimension
                        if (ndims < at_least_dim or tdim != dim) and torch_input[k].size()[tdim] != torch_input[k + 1].size()[tdim]:
                            valid_dim = False

                # Special case for hstack is needed since hstack works differently when ndims is 1
                if valid_dim or (torch_fn is torch.hstack and ndims == 1):
                    # Valid dimensions, test against numpy
                    np_input = [input.cpu().numpy() for input in torch_input]
                    actual = torch_fn(torch_input)
                    expected = np_fn(np_input)
                    self.assertEqual(actual, expected)
                else:
                    # Invalid dimensions, test for error
                    with self.assertRaisesRegex(RuntimeError, "Sizes of tensors must match except in dimension"):
                        torch_fn(torch_input)
                    with self.assertRaises(ValueError):
                        np_input = [input.cpu().numpy() for input in torch_input]
                        np_fn(np_input)

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half))
    def test_hstack_column_stack(self, device, dtype):
        ops = ((torch.hstack, np.hstack), (torch.column_stack, np.column_stack))
        for torch_op, np_op in ops:
            self._test_special_stacks(1, 1, torch_op, np_op, device, dtype)

        # Test torch.column_stack with combinations of 1D and 2D tensors input
        one_dim_tensor = torch.arange(0, 10).to(dtype=dtype, device=device)
        two_dim_tensor = torch.arange(0, 100).to(dtype=dtype, device=device).reshape(10, 10)
        inputs = two_dim_tensor, one_dim_tensor, two_dim_tensor, one_dim_tensor
        torch_result = torch.column_stack(inputs)

        np_inputs = [input.cpu().numpy() for input in inputs]
        np_result = np.column_stack(np_inputs)

        self.assertEqual(np_result,
                         torch_result)

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half))
    def test_vstack_row_stack(self, device, dtype):
        ops = ((torch.vstack, np.vstack), (torch.row_stack, np.vstack))
        for torch_op, np_op in ops:
            self._test_special_stacks(0, 2, torch_op, np_op, device, dtype)
            for _ in range(5):
                # Test dimension change for 1D tensor of size (N) and 2D tensor of size (1, N)
                n = random.randint(1, 10)
                input_a = _generate_input((n,), dtype, device, with_extremal=False)
                input_b = _generate_input((1, n), dtype, device, with_extremal=False)
                torch_input = [input_a, input_b]
                np_input = [input.cpu().numpy() for input in torch_input]
                actual = torch_op(torch_input)
                expected = np_op(np_input)
                self.assertEqual(actual, expected)

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half))
    def test_dstack(self, device, dtype):
        self._test_special_stacks(2, 3, torch.dstack, np.dstack, device, dtype)
        for _ in range(5):
            # Test dimension change for 1D tensor of size (N), 2D tensor of size (1, N), and 3D tensor of size (1, N, 1)
            n = random.randint(1, 10)
            input_a = _generate_input((n,), dtype, device, with_extremal=False)
            input_b = _generate_input((1, n), dtype, device, with_extremal=False)
            input_c = _generate_input((1, n, 1), dtype, device, with_extremal=False)
            torch_input = [input_a, input_b, input_c]
            np_input = [input.cpu().numpy() for input in torch_input]
            actual = torch.dstack(torch_input)
            expected = np.dstack(np_input)
            self.assertEqual(actual, expected)

            # Test dimension change for 2D tensor of size (M, N) and 3D tensor of size (M, N, 1)
            m = random.randint(1, 10)
            n = random.randint(1, 10)
            input_a = _generate_input((m, n), dtype, device, with_extremal=False)
            input_b = _generate_input((m, n, 1), dtype, device, with_extremal=False)
            torch_input = [input_a, input_b]
            np_input = [input.cpu().numpy() for input in torch_input]
            actual = torch.dstack(torch_input)
            expected = np.dstack(np_input)
            self.assertEqual(actual, expected)

    @dtypes(torch.int32, torch.int64)
    def test_large_linspace(self, device, dtype):
        start = torch.iinfo(dtype).min
        end = torch.iinfo(dtype).max & ~0xfff
        steps = 15
        x = torch.linspace(start, end, steps, dtype=dtype, device=device)
        self.assertGreater(x[1] - x[0], (end - start) / steps)

    @dtypes(torch.float32, torch.float64)
    def test_unpack_double(self, device, dtype):
        # Reference: https://github.com/pytorch/pytorch/issues/33111
        vals = (2 ** 24 + 1, 2 ** 53 + 1,
                np.iinfo(np.int64).max, np.iinfo(np.uint64).max, np.iinfo(np.uint64).max + 1,
                -1e500, 1e500)
        for val in vals:
            t = torch.tensor(val, dtype=dtype, device=device)
            a = np.array(val, dtype=torch_to_numpy_dtype_dict[dtype])
            self.assertEqual(t, torch.from_numpy(a))

    def _float_to_int_conversion_helper(self, vals, device, dtype, refs=None):
        if refs is None:
            a = np.array(vals, dtype=np.float32).astype(torch_to_numpy_dtype_dict[dtype])
            refs = torch.from_numpy(a)
        t = torch.tensor(vals, device=device, dtype=torch.float).to(dtype)
        self.assertEqual(refs, t.cpu())

    # Checks that float->integer casts don't produce undefined behavior errors.
    # Note: In C++, casting from a floating value to an integral dtype
    # is undefined if the floating point value is not within the integral
    # dtype's dynamic range. This can (and should) cause undefined behavior
    # errors with UBSAN. These casts are deliberate in PyTorch, however, and
    # NumPy may have the same behavior.
    @onlyNativeDeviceTypes
    @unittest.skipIf(IS_PPC, "Test is broken on PowerPC, see https://github.com/pytorch/pytorch/issues/39671")
    @dtypes(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_float_to_int_conversion_finite(self, device, dtype):
        min = torch.finfo(torch.float).min
        max = torch.finfo(torch.float).max

        # Note: CUDA max float -> integer conversion is divergent on some dtypes
        vals = (min, -2, -1.5, -.5, 0, .5, 1.5, 2, max)
        refs = None
        if self.device_type == 'cuda':
            if torch.version.hip:
                # HIP min float -> int64 conversion is divergent
                vals = (-2, -1.5, -.5, 0, .5, 1.5, 2)
            else:
                vals = (min, -2, -1.5, -.5, 0, .5, 1.5, 2)
        elif dtype == torch.uint8:
            # Note: CPU max float -> uint8 conversion is divergent
            vals = (min, -2, -1.5, -.5, 0, .5, 1.5, 2)
            # Note: numpy -2.0 or -1.5 -> uint8 conversion is undefined
            #       see https://github.com/pytorch/pytorch/issues/97794
            refs = (0, 254, 255, 0, 0, 0, 1, 2)
        elif dtype == torch.int16:
            # CPU min and max float -> int16 conversion is divergent.
            vals = (-2, -1.5, -.5, 0, .5, 1.5, 2)

        self._float_to_int_conversion_helper(vals, device, dtype, refs)

    # Note: CUDA will fail this test on most dtypes, often dramatically.
    # Note: This test validates undefined behavior consistency in float-to-ints casts
    # NB: torch.uint16, torch.uint32, torch.uint64 excluded as this
    # nondeterministically fails, warning "invalid value encountered in cast"
    @onlyCPU
    @unittest.skipIf(IS_S390X, "Test fails for int16 on s390x. Needs investigation.")
    @dtypes(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_float_to_int_conversion_nonfinite(self, device, dtype):
        vals = (float('-inf'), float('inf'), float('nan'))

        if dtype == torch.bool:
            refs = (True, True, True)
        elif IS_ARM64:
            refs = (torch.iinfo(dtype).min, torch.iinfo(dtype).max, 0)
            if dtype in (torch.int8, torch.int16):
                refs = (0, -1, 0)
        else:
            refs = (0, 0, 0)
            if dtype in (torch.int32, torch.int64):
                refs = (torch.iinfo(dtype).min, ) * 3
        self._float_to_int_conversion_helper(vals, device, dtype, refs)

    @onlyNativeDeviceTypes
    def test_complex_type_conversions(self, device):
        dtypes = [torch.float, torch.complex64, torch.complex128]
        for from_type in dtypes:
            for to_type in dtypes:
                from_tensor = torch.randn(4, dtype=from_type, device=device)
                to_tensor = from_tensor.to(to_type)
                if from_type.is_complex and not to_type.is_complex:
                    self.assertEqual(torch.real(from_tensor), to_tensor, exact_dtype=False)
                elif not from_type.is_complex and to_type.is_complex:
                    self.assertEqual(from_tensor, torch.real(to_tensor), exact_dtype=False)
                    self.assertEqual(torch.zeros_like(torch.imag(to_tensor)), torch.imag(to_tensor), exact_dtype=False)
                else:
                    self.assertEqual(from_tensor, to_tensor, exact_dtype=False)

    @slowTest
    @onlyCPU
    def test_cat_big(self, device):
        SIZE1 = 6500
        SIZE2 = 4500
        concat_list = []
        concat_list.append(torch.ones((SIZE1, 1024 * 512), dtype=torch.uint8, device=device))
        concat_list.append(torch.ones((SIZE2, 1024 * 512), dtype=torch.uint8, device=device))
        result = torch.cat(concat_list)
        self.assertEqual(result.size(0), SIZE1 + SIZE2)

    @onlyCPU
    @dtypes(torch.half, torch.double, torch.int)
    def test_cat2(self, device, dtype):
        SIZE = 10
        for dim in range(-3, 3):
            pos_dim = dim if dim >= 0 else 3 + dim
            x = torch.randint(low=-100, high=100, size=(13, SIZE, SIZE), device=device).to(dtype).transpose(0, pos_dim)
            y = torch.randint(low=-100, high=100, size=(17, SIZE, SIZE), device=device).to(dtype).transpose(0, pos_dim)
            z = torch.randint(low=-100, high=100, size=(19, SIZE, SIZE), device=device).to(dtype).transpose(0, pos_dim)

            res1 = torch.cat((x, y, z), dim)
            self.assertEqual(res1.narrow(pos_dim, 0, 13), x, atol=0, rtol=0)
            self.assertEqual(res1.narrow(pos_dim, 13, 17), y, atol=0, rtol=0)
            self.assertEqual(res1.narrow(pos_dim, 30, 19), z, atol=0, rtol=0)

        x = torch.randint(low=-100, high=100, size=(20, SIZE, SIZE), device=device).to(dtype)
        self.assertEqual(torch.cat(torch.split(x, 7)), x)
        self.assertEqual(torch.cat(torch.chunk(x, 7)), x)

        y = torch.randint(low=-100, high=100, size=(1, SIZE, SIZE), device=device).to(dtype)
        z = torch.cat([x, y])
        self.assertEqual(z.size(), (21, SIZE, SIZE))

    @dtypes(torch.float)
    def test_cat_size1(self, device, dtype):
        # create a tensor that has aligned stride along dim - 1 dimension
        # but catted slice size is not aligned
        x1 = torch.randn(16, 16, device=device, dtype=dtype)[:1, :1]
        xref = x1.clone().view(-1).view(x1.shape)
        # make sure output size is aligned, need at least 4 elements for this
        res = torch.cat([x1, x1, x1, x1], dim=-1)
        ref = torch.cat([xref, xref, xref, xref], dim=-1)
        self.assertEqual(res, ref)

    @dtypes(torch.float)
    def test_cat_trailing_dim(self, device, dtype):
        x1 = torch.randn(16, 16, 23, device=device, dtype=dtype)
        x2 = torch.rand_like(x1)
        res = torch.cat([x1, x2], dim=1)
        ref = torch.cat([x1.cpu(), x2.cpu()], dim=1)
        self.assertEqual(res, ref)

    @dtypes(torch.float)
    def test_cat_misaligned(self, device, dtype):
        x1 = torch.randn(14, device=device, dtype=dtype)[2:]
        x2 = torch.rand_like(x1)
        res = torch.cat([x1, x2], dim=-1)
        ref = torch.cat([x1.cpu(), x2.cpu()], dim=-1)
        self.assertEqual(res, ref)

    @dtypes(torch.float)
    def test_cat_multi_batch(self, device, dtype):
        xs = [torch.randn(16, 16, device=device, dtype=dtype) for _ in range(130)]
        xs_cpu = [x.cpu() for x in xs]
        res = torch.cat(xs, dim=-1)
        ref = torch.cat(xs_cpu, dim=-1)
        self.assertEqual(res, ref)
        xs = [torch.randn(16, 15, 15, device=device, dtype=dtype) for _ in range(130)]
        xs[128] = torch.randn(15, 15, 15, device=device, dtype=dtype)
        xs[129] = torch.randn(17, 15, 15, device=device, dtype=dtype)
        xs_cpu = [x.cpu() for x in xs]
        res = torch.cat(xs, dim=0)
        ref = torch.cat(xs_cpu, dim=0)
        self.assertEqual(res, ref)

    @dtypes(torch.float)
    @largeTensorTest("16GB")
    def test_cat_large_tensor(self, device, dtype):
        N = 2 ** 32 // dtype.itemsize
        inps = [torch.randn(N, device=device, dtype=dtype), torch.randn(N // 128, device=device, dtype=dtype)]
        res = torch.cat(inps, dim=0)
        ref = torch.cat([x.cpu() for x in inps])
        self.assertEqual(res, ref)

    # FIXME: Create an OpInfo-based tensor creation method test that verifies this for all tensor
    #   creation methods and verify all dtypes and layouts
    @dtypes(torch.bool, torch.uint8, torch.int16, torch.int64, torch.float16, torch.float32, torch.complex64)
    def test_zeros_dtype_layout_device_match(self, device, dtype):
        layout = torch.strided
        t = torch.zeros((2, 3), device=device, dtype=dtype, layout=layout)
        self.assertIs(dtype, t.dtype)
        self.assertIs(layout, t.layout)
        self.assertEqual(torch.device(device), t.device)

    def test_stack(self, device):
        for dtype in (torch.half, torch.double, torch.int):
            x = torch.randint(low=-100, high=100, size=(2, 3, 4), device=device, dtype=dtype)
            y = torch.randint(low=-100, high=100, size=(2, 3, 4), device=device, dtype=dtype)
            z = torch.randint(low=-100, high=100, size=(2, 3, 4), device=device, dtype=dtype)
            for dim in range(4):
                res = torch.stack((x, y, z), dim)
                res_neg = torch.stack((x, y, z), dim - 4)
                expected_size = x.size()[:dim] + (3,) + x.size()[dim:]
                self.assertEqual(res, res_neg)
                self.assertEqual(res.size(), expected_size)
                self.assertEqual(res.select(dim, 0), x, atol=0, rtol=0)
                self.assertEqual(res.select(dim, 1), y, atol=0, rtol=0)
                self.assertEqual(res.select(dim, 2), z, atol=0, rtol=0)

    def test_stack_out(self, device):
        for dtype in (torch.half, torch.double, torch.int):
            x = torch.randint(low=-100, high=100, size=(2, 3, 4), device=device, dtype=dtype)
            y = torch.randint(low=-100, high=100, size=(2, 3, 4), device=device, dtype=dtype)
            z = torch.randint(low=-100, high=100, size=(2, 3, 4), device=device, dtype=dtype)
            for dim in range(4):
                expected_size = x.size()[:dim] + (3,) + x.size()[dim:]
                res_out = x.new(expected_size)
                res_neg_out = x.new(expected_size)
                res_out_dp = res_out.data_ptr()
                res_out_neg_dp = res_neg_out.data_ptr()
                torch.stack((x, y, z), dim, out=res_out)
                torch.stack((x, y, z), dim - 4, out=res_neg_out)
                self.assertEqual(res_out, res_neg_out)
                self.assertEqual(res_out.size(), expected_size)
                self.assertEqual(res_out_dp, res_out.data_ptr())
                self.assertEqual(res_out_neg_dp, res_neg_out.data_ptr())
                self.assertEqual(res_out.select(dim, 0), x, atol=0, rtol=0)
                self.assertEqual(res_out.select(dim, 1), y, atol=0, rtol=0)
                self.assertEqual(res_out.select(dim, 2), z, atol=0, rtol=0)

    def test_repeat_interleave(self, device):
        x = torch.tensor([0, 1, 2, 3], device=device)
        expected = torch.tensor([1, 2, 2, 3, 3, 3], device=device)
        self.assertEqual(torch.repeat_interleave(x), expected)

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.arange(4, device=device).reshape(2, 2))

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.arange(4.0, device=device))

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(torch.tensor([1, 2, -1, 3, 4], device=device))

        y = torch.tensor([[1, 2], [3, 4]], device=device)

        y1_v1 = torch.repeat_interleave(y, 2)
        y1_v2 = torch.repeat_interleave(y, torch.tensor(2, device=device))
        y1_v3 = torch.repeat_interleave(y, torch.tensor([2], device=device))
        y1_expect = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4], device=device)
        self.assertEqual(y1_v1, y1_expect)
        self.assertEqual(y1_v2, y1_expect)
        self.assertEqual(y1_v3, y1_expect)

        y2 = torch.repeat_interleave(y, 3, dim=1)
        y2_expect = torch.tensor([[1, 1, 1, 2, 2, 2],
                                  [3, 3, 3, 4, 4, 4]], device=device)
        self.assertEqual(y2, y2_expect)

        y3 = torch.repeat_interleave(y, torch.tensor([1, 2], device=device), dim=0)
        y3_expect = torch.tensor([[1, 2],
                                  [3, 4],
                                  [3, 4]], device=device)
        self.assertEqual(y3, y3_expect)

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(y, torch.tensor([1, 2, 3], device=device), dim=0)

        with self.assertRaises(RuntimeError):
            torch.repeat_interleave(y, torch.arange(9, device=device).reshape(3, 3), dim=0)

        # test zero sized dimension
        x = torch.zeros((5, 0), device=device)
        y = torch.repeat_interleave(x, repeats=3, dim=1)
        self.assertEqual(y, x.new_zeros(5, 0, device=device))

        x = torch.tensor([], dtype=torch.int64, device=device)
        y = torch.repeat_interleave(x, x)
        self.assertEqual(y, x)

    def test_new_methods_requires_grad(self, device):
        size = (10,)
        test_cases = [
            # method name, args
            ('new_full', [size, 1]),
            ('new_empty', [size]),
            ('new_zeros', [size]),
            ('new_ones', [size]),
        ]
        for method_name, args in test_cases:
            x = torch.randn(size, device=device)
            for requires_grad in [True, False]:
                x_new = x.__getattribute__(method_name)(*args, requires_grad=requires_grad)
                self.assertEqual(x_new.requires_grad, requires_grad)
            x = torch.randint(10, size, device=device)
            with self.assertRaisesRegex(
                    RuntimeError,
                    r'Only Tensors of floating point and complex dtype can require gradients'):
                x_new = x.__getattribute__(method_name)(*args, requires_grad=True)

    def test_tensor_from_sequence(self, device):
        class MockSequence:
            def __init__(self, lst):
                self.lst = lst

            def __len__(self):
                return len(self.lst)

            def __getitem__(self, item):
                raise TypeError

        class GoodMockSequence(MockSequence):
            def __getitem__(self, item):
                return self.lst[item]

        bad_mock_seq = MockSequence([1.0, 2.0, 3.0])
        good_mock_seq = GoodMockSequence([1.0, 2.0, 3.0])
        with self.assertRaisesRegex(ValueError, 'could not determine the shape'):
            torch.tensor(bad_mock_seq, device=device)
        self.assertEqual(torch.tensor([1.0, 2.0, 3.0], device=device), torch.tensor(good_mock_seq, device=device))

    def test_simple_scalar_cast(self, device):
        ok = [torch.tensor([1.5], device=device), torch.zeros(1, 1, 1, 1, device=device)]
        ok_values = [1.5, 0]

        not_ok = map(torch.Tensor, [[], [1, 2], [[1, 2], [3, 4]]])

        for tensor, value in zip(ok, ok_values):
            self.assertEqual(int(tensor), int(value))
            self.assertEqual(float(tensor), float(value))
            self.assertEqual(complex(tensor), complex(value))

        self.assertEqual(complex(torch.tensor(1.5j)), 1.5j)

        for tensor in not_ok:
            self.assertRaises(ValueError, lambda: int(tensor))
            self.assertRaises(ValueError, lambda: float(tensor))
            self.assertRaises(ValueError, lambda: complex(tensor))

        self.assertRaises(RuntimeError, lambda: float(torch.tensor(1.5j)))
        self.assertRaises(RuntimeError, lambda: int(torch.tensor(1.5j)))

    def test_offset_scalar_cast(self, device):
        x = torch.tensor([1., 2., 3.], device=device)
        y = x[2:]
        self.assertEqual(int(y), 3)

    def test_meshgrid_empty(self):
        with self.assertRaisesRegex(RuntimeError,
                                    'expects a non-empty TensorList'):
            torch.meshgrid()

    def test_meshgrid_unsupported_indexing(self):
        with self.assertRaisesRegex(RuntimeError,
                                    'indexing must be one of "xy" or "ij"'):
            torch.meshgrid(torch.tensor([1, 2]), indexing='')

    def test_meshgrid_non_1d_tensor(self):
        with self.assertRaisesRegex(RuntimeError,
                                    'Expected 0D or 1D tensor'):
            torch.meshgrid(torch.tensor([[1, 2], [3, 4]]))

    def test_meshgrid_inconsistent_dtype(self):
        with self.assertRaisesRegex(
                RuntimeError, 'expects all tensors to have the same dtype'):
            torch.meshgrid(torch.tensor([1], dtype=torch.int),
                           torch.tensor([2], dtype=torch.float))

    def test_meshgrid_inconsistent_device(self):
        with self.assertRaisesRegex(
                RuntimeError, 'expects all tensors to have the same device'):
            torch.meshgrid(torch.tensor([1], device='cpu'),
                           torch.tensor([2], device='meta'))

    def test_meshgrid_warns_if_no_indexing(self):
        with self.assertWarnsOnceRegex(
                UserWarning, '.*will be required to pass the indexing arg.*'):
            torch.meshgrid(torch.tensor([1, 2]))

    def test_meshgrid_default_indexing(self, device):
        a = torch.tensor(1, device=device)
        b = torch.tensor([1, 2, 3], device=device)
        c = torch.tensor([1, 2], device=device)
        grid_a, grid_b, grid_c = torch.meshgrid([a, b, c])
        self.assertEqual(grid_a.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_b.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_c.shape, torch.Size([1, 3, 2]))
        grid_a2, grid_b2, grid_c2 = torch.meshgrid(a, b, c)
        self.assertEqual(grid_a2.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_b2.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_c2.shape, torch.Size([1, 3, 2]))
        expected_grid_a = torch.ones(1, 3, 2, dtype=torch.int64, device=device)
        expected_grid_b = torch.tensor([[[1, 1],
                                         [2, 2],
                                         [3, 3]]], device=device)
        expected_grid_c = torch.tensor([[[1, 2],
                                         [1, 2],
                                         [1, 2]]], device=device)
        self.assertTrue(grid_a.equal(expected_grid_a))
        self.assertTrue(grid_b.equal(expected_grid_b))
        self.assertTrue(grid_c.equal(expected_grid_c))
        self.assertTrue(grid_a2.equal(expected_grid_a))
        self.assertTrue(grid_b2.equal(expected_grid_b))
        self.assertTrue(grid_c2.equal(expected_grid_c))

    def test_meshgrid_xy_indexing(self, device):
        a = torch.tensor(1, device=device)
        b = torch.tensor([1, 2, 3], device=device)
        c = torch.tensor([1, 2], device=device)
        grid_a, grid_b, grid_c = torch.meshgrid([a, b, c], indexing='xy')
        self.assertEqual(grid_a.shape, torch.Size([3, 1, 2]))
        self.assertEqual(grid_b.shape, torch.Size([3, 1, 2]))
        self.assertEqual(grid_c.shape, torch.Size([3, 1, 2]))
        grid_a2, grid_b2, grid_c2 = torch.meshgrid(a, b, c, indexing='xy')
        self.assertEqual(grid_a2.shape, torch.Size([3, 1, 2]))
        self.assertEqual(grid_b2.shape, torch.Size([3, 1, 2]))
        self.assertEqual(grid_c2.shape, torch.Size([3, 1, 2]))
        expected_grid_a = torch.ones(3, 1, 2, dtype=torch.int64, device=device)
        expected_grid_b = torch.tensor([[[1, 1]],
                                        [[2, 2]],
                                        [[3, 3]]], device=device)
        expected_grid_c = torch.tensor([[[1, 2]],
                                        [[1, 2]],
                                        [[1, 2]]], device=device)
        self.assertTrue(grid_a.equal(expected_grid_a))
        self.assertTrue(grid_b.equal(expected_grid_b))
        self.assertTrue(grid_c.equal(expected_grid_c))
        self.assertTrue(grid_a2.equal(expected_grid_a))
        self.assertTrue(grid_b2.equal(expected_grid_b))
        self.assertTrue(grid_c2.equal(expected_grid_c))

    def test_meshgrid_ij_indexing(self, device):
        a = torch.tensor(1, device=device)
        b = torch.tensor([1, 2, 3], device=device)
        c = torch.tensor([1, 2], device=device)
        grid_a, grid_b, grid_c = torch.meshgrid([a, b, c], indexing='ij')
        self.assertEqual(grid_a.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_b.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_c.shape, torch.Size([1, 3, 2]))
        grid_a2, grid_b2, grid_c2 = torch.meshgrid(a, b, c, indexing='ij')
        self.assertEqual(grid_a2.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_b2.shape, torch.Size([1, 3, 2]))
        self.assertEqual(grid_c2.shape, torch.Size([1, 3, 2]))
        expected_grid_a = torch.ones(1, 3, 2, dtype=torch.int64, device=device)
        expected_grid_b = torch.tensor([[[1, 1],
                                         [2, 2],
                                         [3, 3]]], device=device)
        expected_grid_c = torch.tensor([[[1, 2],
                                         [1, 2],
                                         [1, 2]]], device=device)
        self.assertTrue(grid_a.equal(expected_grid_a))
        self.assertTrue(grid_b.equal(expected_grid_b))
        self.assertTrue(grid_c.equal(expected_grid_c))
        self.assertTrue(grid_a2.equal(expected_grid_a))
        self.assertTrue(grid_b2.equal(expected_grid_b))
        self.assertTrue(grid_c2.equal(expected_grid_c))

    def test_meshgrid_ij_indexing_is_default(self, device):
        a = torch.tensor(1, device=device)
        b = torch.tensor([1, 2, 3], device=device)
        c = torch.tensor([1, 2], device=device)
        grid_a, grid_b, grid_c = torch.meshgrid(a, b, c, indexing='ij')
        grid_a2, grid_b2, grid_c2 = torch.meshgrid(a, b, c)
        self.assertTrue(grid_a.equal(grid_a2))
        self.assertTrue(grid_b.equal(grid_b2))
        self.assertTrue(grid_c.equal(grid_c2))

    @skipMeta
    def test_meshgrid_vs_numpy(self, device):
        # Shapes to the random tensors. Each line is a test case, and
        # each list within that line is the shape of a single
        # tensor. The shapes are restricted to 0D (represented by [])
        # and 1D tensors.
        cases = [
            [[]],
            [[1], [1], [1]],
            [[], [], []],
            [[3], [5], [7]],
            [[3], [], [7]],
            [[11], [13]],
            [[15]],
        ]

        # We also need to test the different indexing modes. We can't
        # just enumerate them because we don't presently support the
        # same modes as numpy.meshgrid, nor does our default
        # correspond to their default.
        #
        # TODO Eliminate this and replace it with a list of all
        # supported indexing modes when we have full compatibility.
        indexing_correspondence = [
            # No indexing in PyTorch corresponds to "ij" indexing in
            # NumPy.
            ({}, {'indexing': 'ij'}),

            # No indexing in NumPy corresponds to "xy" indexing in
            # PyTorch.
            ({'indexing': 'xy'}, {}),

            # "ij" and "xy" are implemented identically in both.
            ({'indexing': 'ij'}, {'indexing': 'ij'}),
            ({'indexing': 'xy'}, {'indexing': 'xy'}),
        ]
        for shapes, (torch_kwargs, numpy_kwargs) in product(cases, indexing_correspondence):
            with self.subTest(shapes=shapes, torch_kwargs=torch_kwargs, numpy_kwargs=numpy_kwargs):
                tensors = [make_tensor(shape, device=device, dtype=torch.int) for shape in shapes]
                torch_grids = torch.meshgrid(*tensors, **torch_kwargs)
                numpy_grids = np.meshgrid(*(tensor.cpu().numpy() for tensor in tensors), **numpy_kwargs)
                self.assertEqual(torch_grids, numpy_grids)


    def test_cartesian_prod(self, device):
        a = torch.tensor([1], device=device)
        b = torch.tensor([1, 2, 3], device=device)
        c = torch.tensor([1, 2], device=device)
        prod = torch.cartesian_prod(a, b, c)
        expected = torch.tensor(list(product([a], b, c)), device=device)
        self.assertEqual(expected, prod)

        # test 0 size input
        d = torch.empty(0, dtype=b.dtype, device=device)
        prod = torch.cartesian_prod(a, b, c, d)
        expected = torch.empty(0, 4, dtype=b.dtype, device=device)
        self.assertEqual(expected, prod)

        # test single input
        prod = torch.cartesian_prod(b)
        self.assertEqual(b, prod)

    def test_combinations(self, device):
        a = torch.tensor([1, 2, 3], device=device)

        c = torch.combinations(a, r=0)
        expected = torch.empty(0, dtype=a.dtype, device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, r=1)
        expected = torch.tensor(list(combinations(a, r=1)), device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, r=1, with_replacement=True)
        expected = torch.tensor(list(combinations_with_replacement(a, r=1)), device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a)
        expected = torch.tensor(list(combinations(a, r=2)), device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, with_replacement=True)
        expected = torch.tensor(list(combinations_with_replacement(a, r=2)), device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, r=3)
        expected = torch.tensor(list(combinations(a, r=3)), device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, r=4)
        expected = torch.empty(0, 4, dtype=a.dtype, device=device)
        self.assertEqual(c, expected)

        c = torch.combinations(a, r=5)
        expected = torch.empty(0, 5, dtype=a.dtype, device=device)
        self.assertEqual(c, expected)

        # test empty input
        a = torch.empty(0, device=device)
        c1 = torch.combinations(a)
        c2 = torch.combinations(a, with_replacement=True)
        expected = torch.empty(0, 2, dtype=a.dtype, device=device)
        self.assertEqual(c1, expected)
        self.assertEqual(c2, expected)

    @skipMeta
    def test_linlogspace_mem_overlap(self, device):
        x = torch.rand(1, device=device).expand(10)
        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            torch.linspace(1, 10, 10, out=x)

        with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
            torch.logspace(1, 10, 10, out=x)

    def test_ctor_with_numpy_array(self, device):
        correct_dtypes = [
            np.double,
            float,
            np.float16,
            np.int64,
            np.int32,
            np.int16,
            np.int8,
            np.uint8,
            bool,
        ]

        incorrect_byteorder = '>' if sys.byteorder == 'little' else '<'
        incorrect_dtypes = [incorrect_byteorder + t for t in ['d', 'f']]

        for dtype in correct_dtypes:
            array = np.array([1, 2, 3, 4], dtype=dtype)

            # Upcast
            tensor = torch.DoubleTensor(array).to(device)
            for i in range(len(array)):
                self.assertEqual(tensor[i], array[i])

            # Downcast (sometimes)
            tensor = torch.FloatTensor(array).to(device)
            for i in range(len(array)):
                self.assertEqual(tensor[i], array[i])

            tensor = torch.HalfTensor(array).to(device)
            for i in range(len(array)):
                self.assertEqual(tensor[i], array[i])

    @dtypes(torch.float, torch.double, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_random(self, device, dtype):
        # This test is flaky with p<=(2/(ub-lb))^200=6e-36
        t = torch.empty(200, dtype=dtype, device=device)
        lb = 1
        ub = 4

        t.fill_(-1)
        t.random_(lb, ub)
        self.assertEqual(t.min(), lb)
        self.assertEqual(t.max(), ub - 1)

        t.fill_(-1)
        t.random_(ub)
        self.assertEqual(t.min(), 0)
        self.assertEqual(t.max(), ub - 1)

    def test_random_bool(self, device):
        size = 2000
        t = torch.empty(size, dtype=torch.bool, device=device)

        t.fill_(False)
        t.random_()
        self.assertEqual(t.min(), False)
        self.assertEqual(t.max(), True)
        self.assertTrue(0.4 < (t.eq(True)).to(torch.int).sum().item() / size < 0.6)

        t.fill_(True)
        t.random_()
        self.assertEqual(t.min(), False)
        self.assertEqual(t.max(), True)
        self.assertTrue(0.4 < (t.eq(True)).to(torch.int).sum().item() / size < 0.6)

    def test_random_from_to_bool(self, device):
        size = 2000

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        min_val = 0
        max_val = 1

        froms = [int64_min_val, -42, min_val - 1, min_val, max_val, max_val + 1, 42]
        tos = [-42, min_val - 1, min_val, max_val, max_val + 1, 42, int64_max_val]

        for from_ in froms:
            for to_ in tos:
                t = torch.empty(size, dtype=torch.bool, device=device)
                if to_ > from_:
                    if not (min_val <= from_ <= max_val):
                        self.assertRaisesRegex(
                            RuntimeError,
                            "from is out of bounds",
                            lambda: t.random_(from_, to_)
                        )
                    elif not (min_val <= (to_ - 1) <= max_val):
                        self.assertRaisesRegex(
                            RuntimeError,
                            "to - 1 is out of bounds",
                            lambda: t.random_(from_, to_)
                        )
                    else:
                        t.random_(from_, to_)
                        range_ = to_ - from_
                        delta = 1
                        self.assertTrue(from_ <= t.to(torch.int).min() < (from_ + delta))
                        self.assertTrue((to_ - delta) <= t.to(torch.int).max() < to_)
                else:
                    self.assertRaisesRegex(
                        RuntimeError,
                        "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(to_),
                        lambda: t.random_(from_, to_)
                    )

    # NB: uint64 is broken because its max value is not representable in
    # int64_t, but this is what random expects
    @dtypes(*all_types_and(torch.bfloat16, torch.half, torch.uint16, torch.uint32))
    def test_random_full_range(self, device, dtype):
        size = 2000
        alpha = 0.1

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        if dtype == torch.double:
            fp_limit = 2**53
        elif dtype == torch.float:
            fp_limit = 2**24
        elif dtype == torch.half:
            fp_limit = 2**11
        elif dtype == torch.bfloat16:
            fp_limit = 2**8
        else:
            fp_limit = 0

        t = torch.empty(size, dtype=dtype, device=device)

        if dtype in [torch.float, torch.double, torch.half, torch.bfloat16]:
            from_ = int(max(-fp_limit, int64_min_val))
            to_inc_ = int(min(fp_limit, int64_max_val))
        else:
            from_ = int(max(torch.iinfo(dtype).min, int64_min_val))
            to_inc_ = int(min(torch.iinfo(dtype).max, int64_max_val))
        range_ = to_inc_ - from_ + 1

        t.random_(from_, None)
        delta = max(1, alpha * range_)
        self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
        self.assertTrue((to_inc_ - delta) < t.to(torch.double).max() <= to_inc_)

    # NB: uint64 is broken because its max value is not representable in
    # int64_t, but this is what random expects
    @dtypes(*all_types_and(torch.bfloat16, torch.half, torch .uint16, torch.uint32))
    def test_random_from_to(self, device, dtype):
        size = 2000
        alpha = 0.1

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        if dtype in [torch.float, torch.double, torch.half]:
            min_val = int(max(torch.finfo(dtype).min, int64_min_val))
            max_val = int(min(torch.finfo(dtype).max, int64_max_val))
            froms = [min_val, -42, 0, 42]
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.bfloat16:
            min_val = int64_min_val
            max_val = int64_max_val
            froms = [min_val, -42, 0, 42]
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.uint8:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            froms = [int64_min_val, -42, min_val - 1, min_val, 42, max_val, max_val + 1]
            tos = [-42, min_val - 1, min_val, 42, max_val, max_val + 1, int64_max_val]
        elif dtype == torch.int64:
            min_val = int64_min_val
            max_val = int64_max_val
            froms = [min_val, -42, 0, 42]
            tos = [-42, 0, 42, max_val]
        else:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            froms = [int64_min_val, min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1]
            tos = [min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1, int64_max_val]

        if dtype == torch.double:
            fp_limit = 2**53
        elif dtype == torch.float:
            fp_limit = 2**24
        elif dtype == torch.half:
            fp_limit = 2**11
        elif dtype == torch.bfloat16:
            fp_limit = 2**8
        else:
            fp_limit = 0

        for from_ in froms:
            for to_ in tos:
                t = torch.empty(size, dtype=dtype, device=device)
                if to_ > from_:
                    if not (min_val <= from_ <= max_val):
                        self.assertRaisesRegex(
                            RuntimeError,
                            "from is out of bounds",
                            lambda: t.random_(from_, to_)
                        )
                    elif not (min_val <= (to_ - 1) <= max_val):
                        self.assertRaisesRegex(
                            RuntimeError,
                            "to - 1 is out of bounds",
                            lambda: t.random_(from_, to_)
                        )
                    else:
                        if dtype.is_floating_point and (
                                not (-fp_limit <= from_ <= fp_limit) or not (-fp_limit <= (to_ - 1) <= fp_limit)):
                            if not (-fp_limit <= from_ <= fp_limit):
                                self.assertWarnsRegex(UserWarning, "from is out of bounds",
                                                      lambda: t.random_(from_, to_))
                            if not (-fp_limit <= (to_ - 1) <= fp_limit):
                                self.assertWarnsRegex(UserWarning, "to - 1 is out of bounds",
                                                      lambda: t.random_(from_, to_))
                        else:
                            t.random_(from_, to_)
                            range_ = to_ - from_
                            delta = max(1, alpha * range_)
                            if dtype == torch.bfloat16:
                                # Less strict checks because of rounding errors
                                # TODO investigate rounding errors
                                self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                                self.assertTrue((to_ - delta) < t.to(torch.double).max() <= to_)
                            else:
                                self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                                self.assertTrue((to_ - delta) <= t.to(torch.double).max() < to_)
                else:
                    self.assertRaisesRegex(
                        RuntimeError,
                        "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(to_),
                        lambda: t.random_(from_, to_)
                    )

    @dtypes(*all_types_and(torch.bfloat16, torch.half, torch.uint16, torch.uint32))
    def test_random_to(self, device, dtype):
        size = 2000
        alpha = 0.1

        int64_min_val = torch.iinfo(torch.int64).min
        int64_max_val = torch.iinfo(torch.int64).max

        if dtype in [torch.float, torch.double, torch.half]:
            min_val = int(max(torch.finfo(dtype).min, int64_min_val))
            max_val = int(min(torch.finfo(dtype).max, int64_max_val))
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.bfloat16:
            min_val = int64_min_val
            max_val = int64_max_val
            tos = [-42, 0, 42, max_val >> 1]
        elif dtype == torch.uint8:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            tos = [-42, min_val - 1, min_val, 42, max_val, max_val + 1, int64_max_val]
        elif dtype == torch.int64:
            min_val = int64_min_val
            max_val = int64_max_val
            tos = [-42, 0, 42, max_val]
        else:
            min_val = torch.iinfo(dtype).min
            max_val = torch.iinfo(dtype).max
            tos = [min_val - 1, min_val, -42, 0, 42, max_val, max_val + 1, int64_max_val]

        from_ = 0
        for to_ in tos:
            t = torch.empty(size, dtype=dtype, device=device)
            if to_ > from_:
                if not (min_val <= (to_ - 1) <= max_val):
                    self.assertRaisesRegex(
                        RuntimeError,
                        "to - 1 is out of bounds",
                        lambda: t.random_(from_, to_)
                    )
                else:
                    t.random_(to_)
                    range_ = to_ - from_
                    delta = max(1, alpha * range_)
                    if dtype == torch.bfloat16:
                        # Less strict checks because of rounding errors
                        # TODO investigate rounding errors
                        self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                        self.assertTrue((to_ - delta) < t.to(torch.double).max() <= to_)
                    else:
                        self.assertTrue(from_ <= t.to(torch.double).min() < (from_ + delta))
                        self.assertTrue((to_ - delta) <= t.to(torch.double).max() < to_)
            else:
                self.assertRaisesRegex(
                    RuntimeError,
                    "random_ expects 'from' to be less than 'to', but got from=" + str(from_) + " >= to=" + str(to_),
                    lambda: t.random_(from_, to_)
                )

    @dtypes(*all_types_and(torch.bfloat16, torch.half))
    def test_random_default(self, device, dtype):
        size = 2000
        alpha = 0.1

        if dtype == torch.float:
            to_inc = 1 << 24
        elif dtype == torch.double:
            to_inc = 1 << 53
        elif dtype == torch.half:
            to_inc = 1 << 11
        elif dtype == torch.bfloat16:
            to_inc = 1 << 8
        else:
            to_inc = torch.iinfo(dtype).max

        t = torch.empty(size, dtype=dtype, device=device)
        t.random_()
        self.assertTrue(0 <= t.to(torch.double).min() < alpha * to_inc)
        self.assertTrue((to_inc - alpha * to_inc) < t.to(torch.double).max() <= to_inc)

    # TODO: this test should be updated
    @onlyNativeDeviceTypes
    def test_empty_full(self, device):
        torch_device = torch.device(device)
        device_type = torch_device.type

        dtypes = get_all_dtypes(include_half=False, include_bfloat16=False, include_complex32=True)
        if device_type == 'cpu':
            do_test_empty_full(self, dtypes, torch.strided, torch_device)
        if device_type == 'cuda':
            do_test_empty_full(self, dtypes, torch.strided, None)
            do_test_empty_full(self, dtypes, torch.strided, torch_device)

    # TODO: this test should be updated
    @suppress_warnings
    @onlyNativeDeviceTypes
    @deviceCountAtLeast(1)
    def test_tensor_device(self, devices):
        device_type = torch.device(devices[0]).type
        if device_type == 'cpu':
            self.assertEqual('cpu', torch.tensor(5).device.type)
            self.assertEqual('cpu',
                             torch.ones((2, 3), dtype=torch.float32, device='cpu').device.type)
            self.assertEqual('cpu',
                             torch.ones((2, 3), dtype=torch.float32, device='cpu:0').device.type)
            self.assertEqual('cpu',
                             torch.tensor(torch.ones((2, 3), dtype=torch.float32), device='cpu:0').device.type)
            self.assertEqual('cpu', torch.tensor(np.random.randn(2, 3), device='cpu').device.type)
        if device_type == 'cuda':
            self.assertEqual('cuda:0', str(torch.tensor(5).cuda(0).device))
            self.assertEqual('cuda:0', str(torch.tensor(5).cuda('cuda:0').device))
            self.assertEqual('cuda:0',
                             str(torch.tensor(5, dtype=torch.int64, device=0).device))
            self.assertEqual('cuda:0',
                             str(torch.tensor(5, dtype=torch.int64, device='cuda:0').device))
            self.assertEqual('cuda:0',
                             str(torch.tensor(torch.ones((2, 3), dtype=torch.float32), device='cuda:0').device))

            self.assertEqual('cuda:0', str(torch.tensor(np.random.randn(2, 3), device='cuda:0').device))

            for device in devices:
                with torch.cuda.device(device):
                    device_string = 'cuda:' + str(torch.cuda.current_device())
                    self.assertEqual(device_string,
                                     str(torch.tensor(5, dtype=torch.int64, device='cuda').device))

            with self.assertRaises(RuntimeError):
                torch.tensor(5).cuda('cpu')
            with self.assertRaises(RuntimeError):
                torch.tensor(5).cuda('cpu:0')

            if len(devices) > 1:
                self.assertEqual('cuda:1', str(torch.tensor(5).cuda(1).device))
                self.assertEqual('cuda:1', str(torch.tensor(5).cuda('cuda:1').device))
                self.assertEqual('cuda:1',
                                 str(torch.tensor(5, dtype=torch.int64, device=1).device))
                self.assertEqual('cuda:1',
                                 str(torch.tensor(5, dtype=torch.int64, device='cuda:1').device))
                self.assertEqual('cuda:1',
                                 str(torch.tensor(torch.ones((2, 3), dtype=torch.float32),
                                     device='cuda:1').device))

                self.assertEqual('cuda:1',
                                 str(torch.tensor(np.random.randn(2, 3), device='cuda:1').device))

    # TODO: this test should be updated
    @onlyNativeDeviceTypes
    def test_as_strided_neg(self, device):
        error = r'as_strided: Negative strides are not supported at the ' \
                r'moment, got strides: \[-?[0-9]+(, -?[0-9]+)*\]'
        with self.assertRaisesRegex(RuntimeError, error):
            torch.as_strided(torch.ones(3, 3, device=device), (1, 1), (2, -1))
        with self.assertRaisesRegex(RuntimeError, error):
            torch.as_strided(torch.ones(14, device=device), (2,), (-11,))

    # TODO: this test should be updated
    def test_zeros(self, device):
        res1 = torch.zeros(100, 100, device=device)
        res2 = torch.tensor((), device=device)
        torch.zeros(100, 100, device=device, out=res2)

        self.assertEqual(res1, res2)

        boolTensor = torch.zeros(2, 2, device=device, dtype=torch.bool)
        expected = torch.tensor([[False, False], [False, False]],
                                device=device, dtype=torch.bool)
        self.assertEqual(boolTensor, expected)

        halfTensor = torch.zeros(1, 1, device=device, dtype=torch.half)
        expected = torch.tensor([[0.]], device=device, dtype=torch.float16)
        self.assertEqual(halfTensor, expected)

        bfloat16Tensor = torch.zeros(1, 1, device=device, dtype=torch.bfloat16)
        expected = torch.tensor([[0.]], device=device, dtype=torch.bfloat16)
        self.assertEqual(bfloat16Tensor, expected)

        complexTensor = torch.zeros(2, 2, device=device, dtype=torch.complex64)
        expected = torch.tensor([[0., 0.], [0., 0.]], device=device, dtype=torch.complex64)
        self.assertEqual(complexTensor, expected)

        complexHalfTensor = torch.zeros(2, 2, device=device, dtype=torch.complex32)
        expected = torch.tensor([[0., 0.], [0., 0.]], device=device, dtype=torch.complex32)
        self.assertEqual(complexHalfTensor, expected)

    def test_zeros_bounds_checking(self, device):
        # Test negative large integer
        with self.assertRaisesRegex(RuntimeError, r"zeros: Dimension size must be non-negative."):
            torch.zeros(-6744789213055875072, device=device)

    # TODO: this test should be updated
    def test_zeros_out(self, device):
        shape = (3, 4)
        out = torch.zeros(shape, device=device)
        torch.zeros(shape, device=device, out=out)

        # change the dtype, layout, device
        with self.assertRaises(RuntimeError):
            torch.zeros(shape, device=device, dtype=torch.int64, out=out)
        with self.assertRaises(RuntimeError):
            torch.zeros(shape, device=device, layout=torch.sparse_coo, out=out)

        # leave them the same
        self.assertEqual(torch.zeros(shape, device=device),
                         torch.zeros(shape, device=device, dtype=out.dtype, out=out))
        self.assertEqual(torch.zeros(shape, device=device),
                         torch.zeros(shape, device=device, layout=torch.strided, out=out))
        self.assertEqual(torch.zeros(shape, device=device),
                         torch.zeros(shape, device=device, out=out))

    # TODO: this test should be updated
    def test_ones(self, device):
        res1 = torch.ones(100, 100, device=device)
        res2 = torch.tensor((), device=device)
        torch.ones(100, 100, device=device, out=res2)
        self.assertEqual(res1, res2)

        # test boolean tensor
        res1 = torch.ones(1, 2, device=device, dtype=torch.bool)
        expected = torch.tensor([[True, True]], device=device, dtype=torch.bool)
        self.assertEqual(res1, expected)

        # test chalf
        self.assertEqual(torch.ones(100, 100, device=device, dtype=torch.chalf),
                         torch.ones(100, 100, device=device, dtype=torch.cfloat), exact_dtype=False)

    # TODO: this test should be updated
    @onlyCPU
    def test_constructor_dtypes(self, device):
        self.assertIs(torch.tensor([]).dtype, torch.get_default_dtype())

        self.assertIs(torch.uint8, torch.ByteTensor.dtype)
        self.assertIs(torch.float32, torch.FloatTensor.dtype)
        self.assertIs(torch.float64, torch.DoubleTensor.dtype)

        with set_default_tensor_type('torch.FloatTensor'):
            self.assertIs(torch.float32, torch.get_default_dtype())
            self.assertIs(torch.FloatStorage, torch.Storage)

        # only floating-point types are supported as the default type
        self.assertRaises(TypeError, lambda: torch.set_default_tensor_type('torch.IntTensor'))

        with set_default_dtype(torch.float64):
            self.assertIs(torch.float64, torch.get_default_dtype())
            self.assertIs(torch.DoubleStorage, torch.Storage)

        with set_default_tensor_type(torch.FloatTensor):
            self.assertIs(torch.float32, torch.get_default_dtype())
            self.assertIs(torch.FloatStorage, torch.Storage)

        if torch.cuda.is_available():
            with set_default_tensor_type(torch.cuda.FloatTensor):
                self.assertIs(torch.float32, torch.get_default_dtype())
                self.assertIs(torch.float32, torch.cuda.FloatTensor.dtype)
                self.assertIs(torch.cuda.FloatStorage, torch.Storage)

                with set_default_dtype(torch.float64):
                    self.assertIs(torch.float64, torch.get_default_dtype())
                    self.assertIs(torch.cuda.DoubleStorage, torch.Storage)

        # don't allow passing dtype to set_default_tensor_type
        self.assertRaises(TypeError, lambda: torch.set_default_tensor_type(torch.float32))

        # don't allow passing dtype to set_default_dtype
        for t in all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.qint8):
            # only floating-point types are supported as the default type
            if t in (
                    torch.half,
                    torch.float,
                    torch.double,
                    torch.bfloat16):
                with set_default_dtype(t):
                    pass
            else:
                self.assertRaises(TypeError, lambda: torch.set_default_dtype(t))

    # TODO: this test should be updated
    @onlyCPU
    def test_constructor_device_legacy(self, device):
        self.assertRaises(RuntimeError, lambda: torch.FloatTensor(device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.FloatTensor(torch.Size([2, 3, 4]), device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.FloatTensor((2.0, 3.0), device='cuda'))

        self.assertRaises(RuntimeError, lambda: torch.Tensor(device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.Tensor(torch.Size([2, 3, 4]), device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.Tensor((2.0, 3.0), device='cuda'))

        # Tensor constructor/new with Tensor argument shouldn't work with device specified
        i = torch.tensor([1], device='cpu')
        self.assertRaises(RuntimeError, lambda: torch.Tensor(i, device='cpu'))
        self.assertRaises(RuntimeError, lambda: i.new(i, device='cpu'))
        self.assertRaises(RuntimeError, lambda: torch.Tensor(i, device='cuda'))
        self.assertRaises(RuntimeError, lambda: i.new(i, device='cuda'))

        x = torch.randn((3,), device='cpu')
        self.assertRaises(RuntimeError, lambda: x.new(device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new((2.0, 3.0), device='cuda'))

        if torch.cuda.is_available():
            self.assertRaises(RuntimeError, lambda: torch.cuda.FloatTensor(device='cpu'))
            self.assertRaises(RuntimeError, lambda: torch.cuda.FloatTensor(torch.Size([2, 3, 4]), device='cpu'))
            self.assertRaises(RuntimeError, lambda: torch.cuda.FloatTensor((2.0, 3.0), device='cpu'))

            # Tensor constructor/new with Tensor argument shouldn't work with device specified
            i = torch.tensor([1], device='cuda')
            self.assertRaises(RuntimeError, lambda: torch.Tensor(i, device='cuda'))
            self.assertRaises(RuntimeError, lambda: i.new(i, device='cuda'))
            self.assertRaises(RuntimeError, lambda: torch.Tensor(i, device='cpu'))
            self.assertRaises(RuntimeError, lambda: i.new(i, device='cpu'))

            with set_default_tensor_type(torch.cuda.FloatTensor):
                self.assertRaises(RuntimeError, lambda: torch.Tensor(device='cpu'))
                self.assertRaises(RuntimeError, lambda: torch.Tensor(torch.Size([2, 3, 4]), device='cpu'))
                self.assertRaises(RuntimeError, lambda: torch.Tensor((2.0, 3.0), device='cpu'))
            x = torch.randn((3,), device='cuda')
            self.assertRaises(RuntimeError, lambda: x.new(device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new((2.0, 3.0), device='cpu'))

    # TODO: this test should be updated
    @suppress_warnings
    @onlyCPU
    def test_tensor_factory(self, device):
        # TODO: This test probably doesn't make too much sense now that
        # torch.tensor has been established for a while; it makes more
        # sense to test the legacy behavior in terms of the new behavior
        expected = torch.Tensor([1, 1])
        # test data
        res1 = torch.tensor([1, 1])
        self.assertEqual(res1, expected, exact_dtype=False)

        res1 = torch.tensor([1, 1], dtype=torch.int)
        self.assertEqual(res1, expected, exact_dtype=False)
        self.assertIs(torch.int, res1.dtype)

        # test copy
        res2 = torch.tensor(expected)
        self.assertEqual(res2, expected)
        res2[1] = 2
        self.assertEqual(expected, torch.ones_like(expected))

        res2 = torch.tensor(expected, dtype=torch.int)
        self.assertEqual(res1, expected, ex

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 7 class(es): TestTensorCreation, MockSequence, GoodMockSequence, TestRandomTensorCreation, TestLikeTensorCreation, TestBufferProtocol, TestAsArray

### Functions
This file defines 231 function(s): _generate_input, _rand_shape, test_diag_embed, test_cat_mem_overlap, test_vander, test_vander_types, test_cat_all_dtypes_and_devices, test_fill_all_dtypes_and_devices, test_roll, test_diagflat, test_block_diag, block_diag_workaround, test_block_diag_scipy, test_torch_complex, test_torch_polar, test_torch_complex_floating_dtype_error, test_torch_complex_same_dtype_error, dtype_name, test_torch_complex_out_dtype_error, dtype_name, complex_dtype_name, test_cat_empty_legacy, test_cat_empty, test_concat_empty_list_error, test_cat_out, test_cat_out_fast_path_dim0_dim1, test_cat_out_channels_last, test_cat_in_channels_last, test_cat_preserve_channels_last, test_cat_channels_last_large_inputs


## Key Components

The file contains 16315 words across 4438 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 205285 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
