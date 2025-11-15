# Documentation: `test/test_view_ops.py`

## File Metadata

- **Path**: `test/test_view_ops.py`
- **Size**: 81,490 bytes (79.58 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: tests"]
import random
import unittest
from functools import partial
from itertools import combinations, permutations, product

import numpy as np

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfMPS,
    expectedFailureMPS,
    instantiate_device_type_tests,
    onlyCPU,
    onlyNativeDeviceTypes,
    skipLazy,
    skipMeta,
    skipXLA,
)
from torch.testing._internal.common_dtype import (
    all_mps_types_and,
    all_types_and,
    all_types_and_complex_and,
    complex_types,
    floating_and_complex_types_and,
    integral_types_and,
)
from torch.testing._internal.common_utils import (
    gradcheck,
    gradgradcheck,
    IS_FBCODE,
    numpy_to_torch_dtype_dict,
    run_tests,
    skipIfTorchDynamo,
    suppress_warnings,
    TestCase,
)


# TODO: replace this with make_tensor() in common_utils.py
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
                x = torch.randn(*shape, dtype=dtype, device=device) * random.randint(
                    30, 100
                )
            x[torch.randn(*shape) > 0.5] = 0
            if with_extremal and dtype.is_floating_point:
                # Use extremal values
                x[torch.randn(*shape) > 0.5] = float("nan")
                x[torch.randn(*shape) > 0.5] = float("inf")
                x[torch.randn(*shape) > 0.5] = float("-inf")
            elif with_extremal and dtype.is_complex:
                x[torch.randn(*shape) > 0.5] = complex("nan")
                x[torch.randn(*shape) > 0.5] = complex("inf")
                x[torch.randn(*shape) > 0.5] = complex("-inf")
        elif dtype == torch.bool:
            x = torch.zeros(shape, dtype=dtype, device=device)
            x[torch.randn(*shape) > 0.5] = True
        else:
            x = torch.randint(15, 100, shape, dtype=dtype, device=device)

    return x


# TODO: replace this with make_tensor() in common_utils.py
def _rand_shape(dim, min_size, max_size):
    shape = []
    for _ in range(dim):
        shape.append(random.randint(min_size, max_size))
    return tuple(shape)


# TODO: refactor tests to avoid this function
# Converts half/bfloat16 dtype to float when device is cpu
def _convert_t(dtype, device):
    if device == "cpu" and dtype in {torch.half, torch.bfloat16}:
        return torch.float
    return dtype


# TODO: replace this with make_tensor() in common_utils.py
# Returns a tensor of the requested shape, dtype, and device
# Requesting a half CPU tensor returns a float CPU tensor with
# values representable by a half.
# Initialization uses randint for non-float types and randn for float types.
def _make_tensor(shape, dtype, device, fill_ones=False) -> torch.Tensor:
    # Returns a tensor filled with ones
    if fill_ones:
        return torch.ones(*shape, dtype=_convert_t(dtype, device), device=device)

    # Returns a tensor with random integer values
    if not (dtype.is_floating_point or dtype.is_complex):
        t = torch.randint(0, 10, shape, device=device)
        if dtype != torch.uint8:
            t = t - 5  # generate negative values also
        return t.to(_convert_t(dtype, device))

    # Populates the CPU tensor with floats representable as half/bfloat16
    if dtype == torch.half and device == "cpu":
        return torch.randn(*shape, dtype=torch.float, device=device).half().float()
    if dtype == torch.bfloat16 and device == "cpu":
        return torch.randn(*shape, dtype=torch.float, device=device).bfloat16().float()

    # Default: returns a tensor with random float values
    return torch.randn(shape, dtype=dtype, device=device).to(dtype=dtype)


# Tests ops and indexing to ensure they return views (and new tensors) as
# appropriate.
class TestViewOps(TestCase):
    exact_dtype = True

    def is_view_of(self, base, other):
        if (
            not other._is_view()
            or other is base
            or other._base is not base
            or base.device != other.device
        ):
            return False
        # Note: only validates storage on native device types
        # because some accelerators, like XLA, do not expose storage
        if base.device.type == "cpu" or base.device.type == "cuda":
            if base.untyped_storage().data_ptr() != other.untyped_storage().data_ptr():
                return False

        return True

    # Returns true if v1 and v2 are views of the same base
    def is_view_of_same_base(self, v1, v2):
        if not v1._is_view() or v1 is v2:
            return False
        return self.is_view_of(v1._base, v2)

    # Performs transpose if contiguous=True, else returns the input tensor as is
    def _do_transpose(self, x, contiguous=False, dim0=0, dim1=1):
        if contiguous:
            return x
        else:
            return x.transpose(dim0, dim1)

    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    def test_conj_self(self, device, dtype):
        t = torch.ones(5, 5, device=device)
        s = t.conj()
        self.assertTrue(s is t)

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool))
    @dtypesIfMPS(*integral_types_and(torch.cfloat, torch.float, torch.half, torch.bool))
    def test_view_dtype_new(self, device, dtype):
        dtypes = {value: key for (key, value) in numpy_to_torch_dtype_dict.items()}
        if device.startswith("mps"):
            del dtypes[torch.float64]
        del dtypes[torch.bool]

        def generate_inputs():
            yield make_tensor((4, 4, 64), dtype=dtype, device=device, low=-5, high=5)
            yield make_tensor(
                (4, 4, 64), dtype=dtype, device=device, low=-5, high=5
            ).permute(1, 0, 2)
            yield make_tensor(
                (4, 64, 4), dtype=dtype, device=device, low=-5, high=5
            ).permute(2, 0, 1)
            yield make_tensor(
                (1, 5, 1), dtype=dtype, device=device, low=-5, high=5
            ).expand(5, 5, 64)
            yield make_tensor((2, 5, 256), dtype=dtype, device=device, low=-5, high=5)[
                1::2, 1:, ::2
            ]
            yield make_tensor((0, 5, 64), dtype=dtype, device=device, low=-5, high=5)
            yield make_tensor((), dtype=dtype, device=device, low=-5, high=5)

        def calc_expected_size_and_stride(a, view_dtype):
            dtype_size = torch._utils._element_size(a.dtype)
            view_dtype_size = torch._utils._element_size(view_dtype)

            if dtype_size == view_dtype_size:
                return a.size(), a.stride()

            elif dtype_size > view_dtype_size:
                size_ratio = dtype_size // view_dtype_size

                view_size = list(a.size())
                view_size[-1] = view_size[-1] * size_ratio

                view_stride = [stride * size_ratio for stride in a.stride()]
                view_stride[-1] = 1
                return torch.Size(view_size), tuple(view_stride)

            else:
                size_ratio = view_dtype_size // dtype_size

                view_size = list(a.size())
                view_size[-1] = view_size[-1] // size_ratio

                view_stride = [stride // size_ratio for stride in a.stride()]
                view_stride[-1] = 1
                return torch.Size(view_size), tuple(view_stride)

        for a in generate_inputs():
            a_np = a.cpu().numpy()
            a_np_contiguous = a.cpu().contiguous().numpy()

            for view_dtype, np_view_dtype in dtypes.items():
                equal_element_size = torch._utils._element_size(
                    dtype
                ) == torch._utils._element_size(view_dtype)

                if not equal_element_size and a.dim() == 0:
                    with self.assertRaisesRegex(
                        RuntimeError, r"self.dim\(\) cannot be 0"
                    ):
                        a.view(view_dtype)
                    continue

                if not equal_element_size and a.stride(-1) != 1:
                    with self.assertRaisesRegex(
                        RuntimeError, r"self.stride\(-1\) must be 1"
                    ):
                        a.view(view_dtype)
                    continue

                a_view = a.view(view_dtype)
                self.assertEqual(a_view.dtype, view_dtype)
                self.assertEqual(a.data_ptr(), a_view.data_ptr())

                expected_size, expected_stride = calc_expected_size_and_stride(
                    a, view_dtype
                )
                self.assertEqual(a_view.size(), expected_size)
                self.assertEqual(a_view.stride(), expected_stride)

                self.assertEqual(a_view.view(dtype), a, rtol=0, atol=0)

                # NumPy's dtype view requires contiguous input if target
                # dtype is a different size
                if equal_element_size:
                    a_np_view = a_np.view(np_view_dtype)

                else:
                    a_np_view = a_np_contiguous.view(np_view_dtype)

                self.assertEqual(a_view, a_np_view)

        # Test that requires_grad is dropped for floating point casts,
        # because view(dtype) does not support backward yet
        # TODO: Remove this when autograd support is added
        if dtype.is_floating_point or dtype.is_complex:
            for view_dtype in floating_and_complex_types_and(
                torch.half, torch.bfloat16
            ):
                t = make_tensor(
                    (5, 5, 64),
                    dtype=dtype,
                    device=device,
                    low=-5,
                    high=5,
                    requires_grad=True,
                )
                self.assertFalse(t.view(view_dtype).requires_grad)

    # Test the extra error checks that happen when the view dtype
    # has a greater element size than the original dtype
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    @dtypesIfMPS(*all_mps_types_and(torch.bool))
    def test_view_dtype_upsize_errors(self, device, dtype):
        dtype_size = torch._utils._element_size(dtype)

        for view_dtype in all_types_and_complex_and(
            torch.half, torch.bfloat16, torch.bool
        ):
            view_dtype_size = torch._utils._element_size(view_dtype)
            if view_dtype_size <= dtype_size:
                continue

            size_ratio = view_dtype_size // dtype_size
            a = make_tensor(
                (4, 4, size_ratio + 1), dtype=dtype, device=device, low=-5, high=5
            )
            with self.assertRaisesRegex(
                RuntimeError, rf"self.size\(-1\) must be divisible by {size_ratio}"
            ):
                a.view(view_dtype)

            with self.assertRaisesRegex(
                RuntimeError,
                rf"self.storage_offset\(\) must be divisible by {size_ratio}",
            ):
                a[:, :, 1:].view(view_dtype)

            a = make_tensor(
                (4, 4, size_ratio), dtype=dtype, device=device, low=-5, high=5
            )
            a = a.as_strided((4, 4, size_ratio), (size_ratio, 1, 1))
            with self.assertRaisesRegex(
                RuntimeError, rf"self.stride\(1\) must be divisible by {size_ratio}"
            ):
                a.view(view_dtype)

    @onlyNativeDeviceTypes
    def test_view_as_complex(self, device):
        def fn(contiguous_input=True, dim0=0, dim1=1):
            t = torch.randn(3, 2, 2, device=device)
            c_t = t[:, :, 0] + 1j * t[:, :, 1]

            input = self._do_transpose(t, contiguous_input, dim0, dim1)

            if input.size()[-1] != 2:
                self.assertRaisesRegex(
                    RuntimeError,
                    "Tensor must have a last dimension of size 2",
                    lambda: torch.view_as_complex(input),
                )
                return

            if input.stride()[-1] != 1:
                self.assertRaisesRegex(
                    RuntimeError,
                    "Tensor must have a last dimension with stride 1",
                    lambda: torch.view_as_complex(input),
                )
                return

            res = torch.view_as_complex(input)
            self.assertEqual(res, self._do_transpose(c_t, contiguous_input, dim0, dim1))
            self.assertTrue(self.is_view_of(t, res))

        fn()
        fn(contiguous_input=False)
        # RuntimeError since in this case the last dim of input would not be of size 2
        fn(contiguous_input=False, dim0=0, dim1=2)
        # RuntimeError since in this case the last dim of input would not have stride 1
        fn(contiguous_input=False, dim0=1, dim1=2)

        # RuntimeError since in this case the stride of non-last dim of input would not be of size 2
        x = torch.randn(3, 3, device=device)
        t = torch.as_strided(x, (2, 2), (1, 1))
        self.assertRaisesRegex(
            RuntimeError,
            "Tensor must have a stride divisible by 2 for all but last dimension",
            lambda: torch.view_as_complex(t),
        )

        # tensor with zero elements
        x = torch.tensor([], device=device)  # torch.Size([0])
        self.assertRaisesRegex(
            RuntimeError,
            "Tensor must have a last dimension of size 2",
            lambda: torch.view_as_complex(x),
        )

        # zero dimension tensor
        z = torch.tensor(2.0)
        self.assertRaisesRegex(
            RuntimeError,
            "Input tensor must have one or more dimensions",
            lambda: torch.view_as_complex(z),
        )

        y = x.reshape(0, 2)  # torch.Size([0, 2])
        res = torch.view_as_complex(y)
        self.assertTrue(self.is_view_of(x, res))
        self.assertEqual(res.shape, torch.Size([0]))

    @onlyNativeDeviceTypes
    @dtypes(*complex_types(), torch.complex32)
    @dtypesIfMPS(torch.cfloat, torch.chalf)
    def test_view_as_real(self, device, dtype):
        def fn(contiguous_input=True):
            t = torch.randn(3, 4, dtype=dtype, device=device)
            input = self._do_transpose(t, contiguous_input)
            res = torch.view_as_real(input)
            self.assertEqual(res[:, :, 0], input.real)
            self.assertEqual(res[:, :, 1], input.imag)
            self.assertTrue(self.is_view_of(t, res))

        fn()
        fn(contiguous_input=False)

        # tensor with zero elements
        x = torch.tensor([], dtype=dtype, device=device)
        res = torch.view_as_real(x)
        self.assertTrue(self.is_view_of(x, res))
        self.assertEqual(res.shape, torch.Size([0, 2]))

        # tensor with zero dim
        x = torch.tensor(2 + 3j, dtype=dtype, device=device)
        res = torch.view_as_real(x)
        self.assertTrue(self.is_view_of(x, res))
        self.assertEqual(res.shape, torch.Size([2]))

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    @dtypesIfMPS(*all_mps_types_and(torch.bool))
    def test_view_tensor_split(self, device, dtype):
        a = make_tensor((40, 30), dtype=dtype, device=device, low=-9, high=9)
        a_split_dim0 = a.tensor_split(7, 0)
        for a_split_dim0_tensor in a_split_dim0:
            self.assertTrue(self.is_view_of(a, a_split_dim0_tensor))
        a_split_dim1 = a.tensor_split(7, 1)
        for a_split_dim1_tensor in a_split_dim1:
            self.assertTrue(self.is_view_of(a, a_split_dim1_tensor))

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    @dtypesIfMPS(*all_mps_types_and(torch.cfloat, torch.bool))
    def test_view_tensor_hsplit(self, device, dtype):
        t = make_tensor((4, 4, 4), dtype=dtype, device=device, low=-9, high=9)
        t_hsplit = torch.hsplit(t, 2)
        for t_hsplit_tensor in t_hsplit:
            self.assertTrue(self.is_view_of(t, t_hsplit_tensor))
        t[2, 2, 2] = 7
        self.assertEqual(t_hsplit[1][2, 0, 2], t[2, 2, 2])

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    @dtypesIfMPS(*all_mps_types_and(torch.cfloat, torch.bool))
    def test_view_tensor_vsplit(self, device, dtype):
        t = make_tensor((4, 4, 4), dtype=dtype, device=device, low=-9, high=9)
        t_vsplit = torch.vsplit(t, 2)
        for t_vsplit_tensor in t_vsplit:
            self.assertTrue(self.is_view_of(t, t_vsplit_tensor))
        t[2, 2, 2] = 7
        self.assertEqual(t_vsplit[1][0, 2, 2], t[2, 2, 2])

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    @dtypesIfMPS(*all_mps_types_and(torch.cfloat, torch.bool))
    def test_view_tensor_dsplit(self, device, dtype):
        t = make_tensor((4, 4, 4), dtype=dtype, device=device, low=-9, high=9)
        t_dsplit = torch.dsplit(t, 2)
        for t_dsplit_tensor in t_dsplit:
            self.assertTrue(self.is_view_of(t, t_dsplit_tensor))
        t[2, 2, 2] = 7
        self.assertEqual(t_dsplit[1][2, 2, 0], t[2, 2, 2])

    @onlyNativeDeviceTypes
    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    @dtypesIfMPS(*all_mps_types_and(torch.bool))
    def test_imag_noncomplex(self, device, dtype):
        t = torch.ones((5, 5), dtype=dtype, device=device)

        with self.assertRaises(RuntimeError):
            torch.imag(t)

    @onlyNativeDeviceTypes
    @dtypes(*complex_types())
    @dtypesIfMPS(torch.cfloat)
    def test_real_imag_view(self, device, dtype):
        def compare_with_numpy(contiguous_input=True):
            t = torch.randn(3, 3, dtype=dtype, device=device)
            if not contiguous_input:
                u = t.T
            else:
                u = t

            re = u.real
            exp = torch.from_numpy(u.cpu().numpy().real).to(device=device)
            self.assertEqual(re, exp)
            # for the case of contiguous_input, t=u
            # for the case of non contiguous_input, the base still remains
            # t since we are performing a view operation to make the input non-contiguous
            self.assertTrue(self.is_view_of(t, re))

            im = u.imag
            exp = torch.from_numpy(u.cpu().numpy().imag).to(device=device)
            self.assertEqual(im, exp)
            self.assertTrue(self.is_view_of(t, im))

        compare_with_numpy()
        compare_with_numpy(contiguous_input=False)

        # ensure storage offset is being correctly set
        a = torch.randn(10, dtype=dtype)
        self.assertEqual(a[5:].real, a.real[5:])
        self.assertEqual(a[5:].imag, a.imag[5:])

    @onlyNativeDeviceTypes
    @expectedFailureMPS
    @dtypes(*complex_types())
    def test_conj_imag_view(self, device, dtype) -> None:
        t = _make_tensor((4, 5), dtype, device)
        t_numpy_conj = torch.from_numpy(t.cpu().numpy().conj()).to(device=device)
        v = t.conj()
        self.assertTrue(self.is_view_of(t, v))
        self.assertEqual(v, t_numpy_conj)

        if t.is_complex():
            v_imag = v.imag
            self.assertTrue(self.is_view_of(t, v_imag))
            self.assertEqual(v_imag, t_numpy_conj.imag)
            self.assertTrue(v_imag.is_neg())

    @onlyNativeDeviceTypes
    def test_conj_view_with_shared_memory(self, device) -> None:
        a = _make_tensor((4, 5), torch.cfloat, device)
        b = a.conj()
        c = a.conj()

        self.assertEqual(torch.add(a, b), a.add_(b))
        self.assertEqual(torch.add(b, c), torch.add(b, c, out=a))
        self.assertEqual(torch.add(b, c), b.add_(c))

    @onlyNativeDeviceTypes
    @dtypes(
        *product(
            complex_types(),
            all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
        )
    )
    @dtypesIfMPS(
        *product(
            [torch.cfloat, torch.chalf],
            all_mps_types_and(torch.cfloat, torch.chalf, torch.bool),
        )
    )
    @suppress_warnings
    def test_set_real_imag(self, device, dtypes):
        x = torch.randn(10, dtype=dtypes[0], device=device)

        new_real = _make_tensor((10,), dtypes[1], device)
        new_imag = _make_tensor((10,), dtypes[1], device)

        x.real = new_real
        x.imag = new_imag

        if dtypes[1].is_complex:
            self.assertEqual(x.real, new_real.real, exact_dtype=False)
            self.assertEqual(x.imag, new_imag.real, exact_dtype=False)

        else:
            self.assertEqual(x.real, new_real, exact_dtype=False)
            self.assertEqual(x.imag, new_imag, exact_dtype=False)

    def test_diagonal_view(self, device) -> None:
        t = torch.ones((5, 5), device=device)
        v = torch.diagonal(t)
        self.assertTrue(self.is_view_of(t, v))

        v[0] = 0
        self.assertEqual(t[0, 0], v[0])

        t = torch.ones((3, 3, 3), device=device)
        v = torch.diagonal(t, offset=1, dim1=1, dim2=2)
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0] = 0
        self.assertEqual(t[0, 0, 1], v[0, 0])

    def test_select_view(self, device) -> None:
        t = torch.ones((5, 5), device=device)
        v = t.select(0, 2)
        self.assertTrue(self.is_view_of(t, v))

        v[0] = 0
        self.assertEqual(t[2, 0], v[0])

    # Lazy hasn't implemented unbind yet.
    @skipLazy
    def test_unbind_view(self, device) -> None:
        t = torch.zeros((5, 5), device=device)
        tup = torch.unbind(t)

        for idx, v in enumerate(tup):
            self.assertTrue(self.is_view_of(t, v))

            v[0] = idx + 1
            self.assertEqual(t[idx, 0], v[0])

    # TODO: opinfo this or move to unbind's test suite
    def test_unbind(self):
        stacked = torch.randn(3, 10, 10, requires_grad=True)
        x, y, z = stacked.unbind()
        grad = torch.randn(3, 10, 10)
        torch.autograd.backward([x, y, z], grad.unbind())
        self.assertEqual(stacked.grad, grad)
        # check that it works with only one gradient provided (#9977)
        for i in range(3):
            stacked = torch.randn(3, 10, 10, requires_grad=True)
            outs = stacked.unbind()
            gi = grad.unbind()[i]
            (g,) = torch.autograd.grad(outs[i], stacked, gi)
            g_expected = torch.stack(
                [gi if j == i else torch.zeros_like(gi) for j in range(3)], dim=0
            )
            self.assertEqual(g, g_expected)
        # Check with gradcheck
        stacked = torch.randn(3, 10, 10, dtype=torch.double, requires_grad=True)
        gradcheck(lambda x: x.unbind(), (stacked,), check_forward_ad=True)

    # TODO: Fix this test for LTC. There is an interaction with dynamic shapes here that is broken,
    # causing asserts to trigger.
    @skipLazy
    def test_expand_view(self, device) -> None:
        t = torch.ones((5, 1), device=device)
        v = t.expand(5, 5)
        self.assertTrue(self.is_view_of(t, v))

        v[2, 2] = 0
        self.assertEqual(t[2, 0], v[2, 2])

    def test_expand_as_view(self, device):
        t = torch.ones((5, 1), device=device)
        e = torch.empty((5, 5), device=device)
        v = t.expand_as(e)
        self.assertTrue(self.is_view_of(t, v))

        v[2, 2] = 0
        self.assertEqual(t[2, 0], v[2, 2])

    def test_narrow_view(self, device):
        t = torch.ones((5, 5), device=device)
        v = torch.narrow(t, 1, 2, 2)
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0] = 0
        self.assertEqual(t[0, 2], v[0, 0])

    def test_permute_view(self, device) -> None:
        t = torch.ones((5, 5), device=device)
        v = t.permute(1, 0)
        self.assertTrue(self.is_view_of(t, v))

        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

    def test_transpose_view(self, device):
        for fn in (torch.swapdims, torch.swapaxes, torch.transpose):
            t = torch.ones((5, 5), device=device)
            v = fn(t, 0, 1)
            self.assertTrue(self.is_view_of(t, v))

            v[0, 1] = 0
            self.assertEqual(t[1, 0], v[0, 1])

    def test_transpose_inplace_view(self, device):
        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.swapdims_(0, 1)
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.swapaxes_(0, 1)
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.transpose_(0, 1)
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

    def test_t_view(self, device):
        t = torch.ones((5, 5), device=device)
        v = t.t()
        self.assertTrue(self.is_view_of(t, v))

        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

    def test_t_inplace_view(self, device):
        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.t_()
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertEqual(t[1, 0], v[0, 1])

    def test_T_view(self, device):
        for op in ("T", "H", "mT", "mH"):
            t = torch.ones((5, 5), device=device)
            v = getattr(t, op)
            self.assertTrue(self.is_view_of(t, v))

            v[0, 1] = 0
            self.assertEqual(t[1, 0], v[0, 1])

    def test_unfold_view(self, device):
        t = torch.ones(10, device=device)
        v = t.unfold(0, 3, 2)
        self.assertTrue(self.is_view_of(t, v))

        v[1, 0] = 0
        self.assertEqual(t[2], v[1, 0])

    def test_squeeze_view(self, device):
        t = torch.ones(5, 1, 5, device=device)
        v = torch.squeeze(t)
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertEqual(t, v._base)

    def test_squeeze_inplace_view(self, device):
        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.squeeze_()
        self.assertTrue(self.is_view_of(t, v))
        v[0, 1] = 0
        self.assertEqual(t, v._base)

    def test_unsqueeze_view(self, device):
        t = torch.ones(5, 5, device=device)
        v = torch.unsqueeze(t, 1)
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0, 1] = 0
        self.assertEqual(t[0, 1], v[0, 0, 1])

    def test_unsqueeze_inplace_view(self, device):
        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.unsqueeze_(1)
        self.assertTrue(self.is_view_of(t, v))
        v[0, 0, 1] = 0
        self.assertEqual(t[0, 1], v[0, 0, 1])

    def test_as_strided_view(self, device):
        t = torch.ones(5, 5, device=device)
        v = torch.as_strided(t, (25,), (1,))
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_as_strided_inplace_view(self, device):
        t = torch.ones(5, 5, device=device)
        v = t.view_as(t)
        v = v.as_strided_((25,), (1,))
        self.assertTrue(self.is_view_of(t, v))
        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_as_strided_gradients(self):
        def test(x, prepro_fn, size, strides, offset=None):
            x = x.to(torch.double).detach().requires_grad_()

            # Check that forward will **not** resize storage because it may
            # cause NaN in output and fail numerical Jacobian check consequently
            with torch.no_grad():
                y = prepro_fn(x) if prepro_fn is not None else x
                max_offset = sum((si - 1) * st for si, st in zip(size, strides))
                max_offset += offset if offset is not None else y.storage_offset()
                assert max_offset < len(y.storage()), "test case resizes storage"

            def closure(x):
                if prepro_fn is not None:
                    x = prepro_fn(x)
                return x.as_strided(size, strides, offset)

            gradcheck(closure, [x], check_forward_ad=True)
            gradgradcheck(closure, [x])

        # test
        test(torch.arange(0, 25), lambda x: x.view(5, 5), [3, 3], [6, 2], 2)

        # test crazy stride at dim with size 1 case
        test(torch.randn(12), None, [1, 2, 1, 5], [0, 5, 100, 1], 2)

        # test expand case
        test(torch.randn(5), None, [3, 3, 3], [0, 1, 0], 2)
        test(torch.randn(5), None, [3, 3, 3], [0, 0, 0], 4)
        test(torch.randn(5), lambda x: x.expand(5, 5), [5, 5], [0, 1], 0)

        # test non-expand overlapping case
        test(torch.randn(35), None, [6, 6], [5, 1], 2)
        test(torch.randn(15), None, [3, 2], [3, 6], 2)

        # test transpose case
        test(torch.randn(3, 4), None, [4, 3], [1, 4])

        # test "getting things outside the input" case
        x = torch.randn(6, 2)
        test(x[3:], None, [3, 2], [2, 1], 0)  # should be all zeros
        self.assertEqual(x[3:].as_strided([3, 2], [2, 1], 0), x[:3])

        # test select on expanded input case
        test(torch.randn(2, 3), lambda x: x.expand(10, 2, 3), [2, 3], [3, 1], 0)

    def test_view_view(self, device):
        t = torch.ones(5, 5, device=device)
        v = t.view(25)
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_view_as_view(self, device):
        t = torch.ones(5, 5, device=device)
        e = torch.empty((25,))
        v = t.view_as(e)
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_contiguous_self(self, device):
        t = torch.ones(5, 5, device=device)
        s = t.contiguous()
        self.assertTrue(s is t)

    @skipMeta
    # self.is_view_of reports false positives for lazy
    @skipLazy
    def test_contiguous_nonview(self, device):
        t = torch.ones(5, 5, device=device)
        nv = t.t().contiguous()
        self.assertTrue(not self.is_view_of(t, nv))

        nv[0, 0] = 0
        self.assertNotEqual(t[0, 0], nv[0, 0])

    def test_reshape_view(self, device):
        t = torch.ones(5, 5, device=device)
        v = torch.reshape(t, (25,))
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    def test_reshape_as_view(self, device):
        t = torch.ones(5, 5, device=device)
        e = torch.empty((25,), device=device)
        v = t.reshape_as(e)
        self.assertTrue(self.is_view_of(t, v))

        v[6] = 0
        self.assertEqual(t[1, 1], v[6])

    @skipMeta
    # self.is_view_of reports false positives for lazy
    @skipLazy
    def test_reshape_nonview(self, device):
        t = torch.ones(5, 5, device=device)
        nv = torch.reshape(t.t(), (25,))
        self.assertTrue(not self.is_view_of(t, nv))

        nv[6] = 0
        self.assertNotEqual(t[1, 1], nv[6])

    # This test use as_strided to construct a tensor with overlapping memory,
    # which is not handled by the functionalization pass.
    @skipLazy
    @skipXLA
    def test_flatten_view(self, device):
        def test_writes_propagate(t, v):
            idx_t = (0,) * t.ndim
            idx_v = (0,) * v.ndim
            v[idx_v] = 0
            self.assertEqual(t[idx_t], v[idx_v])

        t = torch.ones(1, 2, 3, 4, device=device)
        v = t.flatten()
        self.assertTrue(self.is_view_of(t, v))
        test_writes_propagate(t, v)

        # zero-dimensional tensor
        t = torch.tensor(1, device=device)
        v = t.flatten()
        test_writes_propagate(t, v)
        self.assertTrue(self.is_view_of(t, v))

        t = torch.ones(1, 2, 3, 4, device=device).transpose(2, 3)
        v = t.flatten(0, 1)
        test_writes_propagate(t, v)
        self.assertTrue(self.is_view_of_same_base(t, v))

        # stride[i] = stride[i + 1] * size[i + 1] is satisfied for 3 groups:
        t = torch.ones(720, device=device).as_strided(
            (2, 3, 2, 3, 5, 4), (6, 2, 15, 5, 1, 0)
        )
        #               [--1--|---2---|-3-] [--1--|----2---|-3-]
        v1 = t.flatten(0, 1)
        v2 = v1.flatten(1, 3)
        v3 = v2.flatten(2, 2)
        test_writes_propagate(t, v1)
        self.assertTrue(self.is_view_of_same_base(t, v1))
        test_writes_propagate(t, v2)
        self.assertTrue(self.is_view_of_same_base(t, v2))
        test_writes_propagate(t, v3)
        self.assertTrue(self.is_view_of_same_base(t, v3))

    @onlyNativeDeviceTypes
    def test_flatten_nonview(self, device):
        def assert_is_nonview(t, nv):
            idx_t = (0,) * t.ndim
            idx_nv = (0,) * nv.ndim
            self.assertTrue(not nv._is_view())
            nv[idx_nv] = 0
            if device != "meta":
                self.assertNotEqual(t[idx_t], nv[idx_nv])

        t = torch.ones(2, 3, 2, 3, device=device).transpose(2, 3)
        nv = t.flatten(1, 3)
        assert_is_nonview(t, nv)

        t = torch.ones(2, 2, device=device).T
        nv = t.flatten()
        assert_is_nonview(t, nv)

        # flatten returns the original object if start_dim=end_dim
        t = torch.ones(2, 2, device=device)
        nv = t.flatten(1, 1)
        self.assertTrue(t is nv)

    def test_basic_indexing_slice_view(self, device):
        t = torch.ones(5, 5, device=device)
        v = t[:2, :3]
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0] = 0
        self.assertEqual(t[0, 0], v[0, 0])

    def test_basic_indexing_ellipses_view(self, device):
        t = torch.ones(5, 5, device=device)
        v = t[..., :2]
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0] = 0
        self.assertEqual(t[0, 0], v[0, 0])

    def test_basic_indexing_newaxis_view(self, device):
        t = torch.ones(5, 5, device=device)
        v = t[None, :2, 3]
        self.assertTrue(self.is_view_of(t, v))

        v[0, 0] = 0
        self.assertEqual(t[0, 3], v[0, 0])

    def test_advanced_indexing_nonview(self, device):
        t = torch.ones(3, 3, device=device)
        rows = torch.tensor([[0, 0], [2, 2]], device=device)
        cols = torch.tensor([[0, 1], [2, 2]], device=device)
        nv = t[rows, cols]
        self.assertTrue(not self.is_view_of(t, nv))

        nv[1, 1] = 0
        self.assertNotEqual(t[2, 2], nv[1, 1])

    @unittest.skipIf(
        IS_FBCODE, "TorchScript backend not yet supported in FBCODE/OVRSOURCE builds"
    )
    def test_advanced_indexing_assignment(self, device):
        t = torch.ones(3, 3, device=device)
        rows = torch.tensor([[0, 0], [2, 2]], device=device)
        cols = torch.tensor([[0, 1], [2, 2]], device=device)
        t[rows, cols] = 0
        self.assertEqual(t[2, 2], 0)

    @unittest.skip("See https://github.com/pytorch/pytorch/pull/32720")
    def test_chunk_view(self, device):
        t = torch.zeros(3, 3, device=device)
        l = torch.chunk(t, 3)

        for idx, v in enumerate(l):
            self.assertTrue(self.is_view_of(t, v))

            v[0, 0] = idx + 1
            self.assertEqual(t[idx, 0], v[0, 0])

    @unittest.skip("See https://github.com/pytorch/pytorch/pull/32720")
    def test_split_view(self, device):
        t = torch.zeros(3, 3, device=device)
        l = torch.split(t, [1, 1, 1])

        for idx, v in enumerate(l):
            self.assertTrue(self.is_view_of(t, v))

            v[0, 0] = idx + 1
            self.assertEqual(t[idx, 0], v[0, 0])

    def test_movedim_view(self, device):
        def run_test(device, op):
            t = torch.zeros(3, 3, device=device)
            out = op(t)

            self.assertTrue(self.is_view_of(t, out))

            # Randomly change values in output
            # and verify that original is changed
            # as well.
            for _ in range(3):
                idx_1, idx_2 = random.randint(0, 2), random.randint(0, 2)
                out[idx_1, idx_2] = random.random()
                self.assertEqual(t[idx_2, idx_1], out[idx_1, idx_2])

        for fn in [torch.movedim, torch.moveaxis]:
            op = partial(fn, source=(0, 1), destination=(1, 0))
            run_test(device, op)

            op = partial(fn, source=0, destination=1)
            run_test(device, op)

    # Testing that the generated view_copy kernel and its derivative are implemented correctly
    def test_view_copy(self, device):
        a = torch.randn(4, device=device, requires_grad=True)
        a_ref = a.detach().clone().requires_grad_()
        a_view = a_ref.view(2, 2)
        a_view_copy = torch.view_copy(a, (2, 2))

        # view_copy ops don't preserve view relationship
        self.assertTrue(self.is_view_of(a_ref, a_view))
        self.assertFalse(self.is_view_of(a, a_view_copy))

        a_view_copy.sum().backward()
        a_view.sum().backward()

        # forward and backward give the same shape + result
        self.assertEqual(a_view_copy, a_view)
        self.assertEqual(a.grad, a_ref.grad)

    # Testing that the output of a view_copy kernel (by default) is contiguous.
    def test_view_copy_output_contiguous(self, device):
        a = torch.randn(4, 4, 4, 4, device=device).to(memory_format=torch.channels_last)
        b = torch.ops.aten.slice_copy(a, 0, 0, 2)
        self.assertTrue(b.is_contiguous())

    def test_view_copy_out(self, device):
        a = torch.randn(2, 2, device=device)
        out = torch.empty(2, device=device)

        torch.diagonal_copy(a, out=out)
        expected = torch.diagonal_copy(a)

        self.assertEqual(expected, out)

        a = torch.randn(4, device=device)
        out1 = torch.empty(2, device=device)
        out2 = torch.empty(2, device=device)

        torch.split_copy(a, 2, out=(out1, out2))
        expected1, expected2 = torch.split_copy(a, 2)

        self.assertEqual(expected1, out1)
        self.assertEqual(expected2, out2)


class TestOldViewOps(TestCase):
    def test_ravel(self, device):
        def _test_ravel(tensors, size, nc=False):
            for src in tensors:
                # Continuous Tensor -> View
                flat = src.ravel()
                self.assertEqual(flat.shape, torch.Size([size]))
                self.assertEqual(src.view(-1), flat)
                self.assertIs(flat._base, src)
                self.assertTrue(flat.is_contiguous())

                # Non-continuous Tensor -> Copy
                if nc:
                    nc_src = src.t()
                    nc_flat = nc_src.ravel()
                    self.assertEqual(nc_flat.shape, torch.Size([size]))
                    self.assertEqual(nc_src.contiguous().view(-1), nc_flat)
                    self.assertIsNot(nc_flat._base, src)
                    self.assertTrue(nc_flat.is_contiguous())

        # Test that flatten returns 1-dim tensor when given a 0-dim tensor
        zero_dim_tensor = torch.tensor(123, device=device)
        flat0 = zero_dim_tensor.ravel()
        one_dim_tensor = torch.tensor([123], device=device)
        flat1 = zero_dim_tensor.ravel()
        nc_ones_tensor = torch.ones(10, device=device)[::2]
        flat2 = nc_ones_tensor.ravel()

        self.assertEqual(zero_dim_tensor.shape, torch.Size([]))
        self.assertEqual(flat0.shape, torch.Size([1]))
        self.assertEqual(one_dim_tensor.shape, torch.Size([1]))
        self.assertEqual(flat1.shape, torch.Size([1]))
        self.assertEqual(nc_ones_tensor.shape, torch.Size([5]))
        self.assertEqual(flat2.shape, torch.Size([5]))
        self.assertEqual(flat0, one_dim_tensor)
        self.assertEqual(flat0, flat1)
        self.assertEqual(flat0.shape, flat1.shape)
        self.assertTrue(flat0.is_contiguous())
        self.assertTrue(flat1.is_contiguous())
        self.assertTrue(flat2.is_contiguous())

        # Test both float tensor and quantized tensor
        tensors = [
            torch.randn(5, 5, 5, 5, device=device),
            torch._empty_affine_quantized(
                [5, 5, 5, 5], scale=2, zero_point=3, dtype=torch.quint8, device=device
            ),
        ]
        _test_ravel(tensors, 625)

        tensors = [
            torch.randn(0, 2, 3, device=device),
            torch.randn(3, 0, 2, device=device),
            torch._empty_affine_quantized(
                [0, 2, 3], scale=2, zero_point=3, dtype=torch.quint8, device=device
            ),
            torch._empty_affine_quantized(
                [3, 0, 2], scale=2, zero_point=3, dtype=torch.quint8, device=device
            ),
        ]
        _test_ravel(tensors, 0)

        tensors = [
            torch.randn(5, 5, device=device),
            torch._empty_affine_quantized(
                [5, 5], scale=2, zero_point=3, dtype=torch.quint8, device=device
            ),
        ]
        _test_ravel(tensors, 25, True)

    # TODO: this should be refactored into the view ops test suite
    def test_empty_reshape(self, device):
        x = torch.randn(0, 6, device=device)
        self.assertEqual((1, 0, 6, 1, 1), x.reshape(1, 0, 6, 1, 1).shape)
        # should be viewable -- i.e. data_ptr is the same.
        self.assertEqual(x.data_ptr(), x.reshape(1, 0, 6, 1, 1).data_ptr())

        # match NumPy semantics -- don't infer the size of dimension with a degree of freedom
        self.assertRaises(RuntimeError, lambda: x.reshape(0, -1))

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_expand(self, device):
        tensor = torch.rand(1, 8, 1, device=device)
        tensor2 = torch.rand(5, device=device)
        template = torch.rand(4, 8, 5, device=device)
        target = template.size()
        self.assertEqual(tensor.expand_as(template).size(), target)
        self.assertEqual(tensor.expand(4, 8, 5).size(), target)
        self.assertEqual(tensor.expand(target).size(), target)
        self.assertEqual(tensor2.expand_as(template).size(), target)
        self.assertEqual(tensor2.expand(4, 8, 5).size(), target)
        self.assertEqual(tensor2.expand(target).size(), target)

        # test double expand
        self.assertEqual(tensor2.expand(1, 5).expand(2, 2, 5), tensor2.repeat(2, 2, 1))

        # test non-contiguous
        noncontig = torch.randn(5, 2, 1, 3, device=device)[:, 0]
        self.assertFalse(noncontig.is_contiguous())
        self.assertEqual(
            noncontig.expand(2, 5, 4, 3), noncontig.contiguous().repeat(2, 1, 4, 1)
        )

        # make sure it's compatible with unsqueeze
        expanded = tensor2.expand(1, 1, 5)
        unsqueezed = tensor2.unsqueeze(0).unsqueeze(1)
        self.assertEqual(expanded, unsqueezed)
        self.assertEqual(expanded.stride(), unsqueezed.stride())

        # test -1 as target size
        self.assertEqual(tensor.expand(4, -1, 5), tensor.expand(4, 8, 5))
        self.assertRaises(RuntimeError, lambda: tensor2.expand(-1, -1))

        # test expanding empty to empty
        self.assertEqual(
            torch.zeros(0, device=device).expand((0,)), torch.zeros(0, device=device)
        )

    # TODO: this should be refactored into the view ops test suite
    def test_view_empty(self, device):
        x = torch.randn(0, 6, device=device)
        self.assertEqual((1, 0, 6, 1, 1), x.view(1, 0, 6, 1, 1).shape)

    # TODO: this should be refactored into the view ops test suite
    @onlyNativeDeviceTypes
    def test_reshape(self, device):
        x = torch.randn(3, 3, device=device)
        self.assertEqual(x.data_ptr(), x.reshape(-1).data_ptr())
        self.assertEqual(x.data_ptr(), x.reshape(1, 9, 1).data_ptr())
        self.assertEqual(torch.reshape(x, (9,)), x.reshape(9))
        self.assertRaises(RuntimeError, lambda: x.reshape(-1, -1))

        y = torch.randn(4, 4, 4, device=device)[:, 0, :]
        # .data_ptr() on meta tensors is always 0 so they are equal regardless of the reshape
        if device != "meta":
            self.assertNotEqual(y.data_ptr(), y.reshape(-1).data_ptr())
        self.assertEqual(y.contiguous().view(-1), y.reshape(-1))
        self.assertEqual(y.reshape(2, 2, 4).data_ptr(), y.data_ptr())

        s = torch.randn((), device=device)
        self.assertEqual(s.data_ptr(), s.reshape(()).data_ptr())
        self.assertEqual(s.reshape(-1).shape, (1,))
        self.assertRaises(RuntimeError, lambda: s.reshape(2))

        empty = torch.tensor([], device=device)
        self.assertEqual(empty, empty.reshape(-1))
        self.assertEqual(empty, empty.reshape([0]))
        # TODO: fix these once we have multi-dimensional empty tensors
        self.assertEqual(empty.reshape([0, 1]).shape, (0, 1))
        self.assertEqual(empty.reshape([1, -1]).shape, (1, 0))
        self.assertRaises(RuntimeError, lambda: empty.reshape(1))

        x = torch.randn(3, 3, device=device)
        self.assertEqual(x.data_ptr(), x.reshape_as(torch.rand(9)).data_ptr())
        self.assertEqual(x.data_ptr(), x.reshape_as(torch.rand(1, 9, 1)).data_ptr())
        self.assertRaises(
            RuntimeError, lambda: x.reshape_as(torch.rand(10, device=device))
        )

    def test_flatten(self, device):
        # Test that flatten returns 1-dim tensor when given a 0-dim tensor
        zero_dim_tensor = torch.tensor(123, device=device)
        flat0 = zero_dim_tensor.flatten()
        one_dim_tensor = torch.tensor([123], device=device)
        flat1 = zero_dim_tensor.flatten()

        self.assertEqual(zero_dim_tensor.shape, torch.Size([]))
        self.assertEqual(flat0.shape, torch.Size([1]))
        self.assertEqual(one_dim_tensor.shape, torch.Size([1]))
        self.assertEqual(flat1.shape, torch.Size([1]))
        self.assertEqual(flat0, one_dim_tensor)
        self.assertEqual(flat0, flat1)
        self.assertEqual(flat0.shape, flat1.shape)

        # Test both float tensor and quantized tensor
        tensors = [
            torch.randn(5, 5, 5, 5, device=device),
            torch._empty_affine_quantized(
                [5, 5, 5, 5], scale=2, zero_point=3, dtype=torch.quint8, device=device
            ),
        ]
        for src in tensors:
            flat = src.flatten(0, -1)
            self.assertEqual(flat.shape, torch.Size([625]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(0, 2)
            self.assertEqual(flat.shape, torch.Size([125, 5]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(0, 1)
            self.assertEqual(flat.shape, torch.Size([25, 5, 5]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(1, 2)
            self.assertEqual(flat.shape, torch.Size([5, 25, 5]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(2, 3)
            self.assertEqual(flat.shape, torch.Size([5, 5, 25]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(-2, -1)
            self.assertEqual(flat.shape, torch.Size([5, 5, 25]))
            self.assertEqual(src.view(-1), flat.view(-1))

            flat = src.flatten(2, 2)
            self.assertEqual(flat, src)

            # out of bounds index
            with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                src.flatten(5, 10)

            # invalid start and end
            with self.assertRaisesRegex(
                RuntimeError, "start_dim cannot come after end_dim"
            ):
                src.flatten(2, 0)

    # TODO: update to work on CUDA, too
    @onlyCPU
    def test_narrow(self, device):
        x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.assertEqual(x.narrow(0, 0, 1), torch.tensor([[0, 1, 2]]))
        self.assertEqual(x.narrow(0, 0, 2), torch.tensor([[0, 1, 2], [3, 4, 5]]))
        self.assertEqual(x.narrow(0, 1, 1), torch.tensor([[3, 4, 5]]))
        self.assertEqual(x.narrow(0, -1, 1), torch.tensor([[6, 7, 8]]))
        self.assertEqual(x.narrow(0, -2, 2), torch.tensor([[3, 4, 5], [6, 7, 8]]))
        self.assertEqual(
            x.narrow(0, -3, 3), torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        )
        self.assertEqual(x.narrow(-1, -1, 1), torch.tensor([[2], [5], [8]]))
        self.assertEqual(x.narrow(-2, -1, 1), torch.tensor([[6, 7, 8]]))

    # TODO: update to work on CUDA, too
    @onlyCPU
    def test_narrow_tensor(self, device):
        x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        self.assertEqual(x.narrow(0, torch.tensor(0), 1), torch.tensor([[0, 1, 2]]))
        with self.assertRaises(Exception):
            x.narrow(0, torch.tensor(0.0), 1)
        with self.assertRaises(Exception):
            x.narrow(0, torch.tensor([0]), 1)
        with self.assertRaises(Exception):
            x.narrow(0, torch.tensor([0, 1]), 1)

    # TODO: make work on CUDA, too
    @onlyCPU
    def test_t(self, device):
        # Test 0D tensors
        x = torch.randn(())
        self.assertEqual(x, x.t())
        x = x.to_sparse()
        self.assertEqual(x, x.t())

        # Test 1D tensors
        x = torch.arange(4)
        self.assertEqual(x, x.t())
        x = x.to_sparse()
        self.assertEqual(x, x.t())

        # Test 2D tensors
        x = torch.rand((2, 2))
        self.assertEqual(x.t(), x.transpose(0, 1))
        x = x.to_sparse()
        self.assertEqual(x.t(), x.transpose(0, 1))

        # Test 3D tensor
        x = torch.rand((2, 2, 2))
        with self.assertRaisesRegex(
            RuntimeError, "expects a tensor with <= 2 dimensions, but self is 3D"
        ):
            x.t()
        x = x.to_sparse()
        with self.assertRaisesRegex(
            RuntimeError, "expects a tensor with <= 2 sparse and 0 dense dimensions"
        ):
            x.t()

    @onlyCPU
    def test_split(self, device):
        tensor 
```



## High-Level Overview


This Python file contains 2 class(es) and 118 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestViewOps`, `TestOldViewOps`

**Functions defined**: `_generate_input`, `_rand_shape`, `_convert_t`, `_make_tensor`, `is_view_of`, `is_view_of_same_base`, `_do_transpose`, `test_conj_self`, `test_view_dtype_new`, `generate_inputs`, `calc_expected_size_and_stride`, `test_view_dtype_upsize_errors`, `test_view_as_complex`, `fn`, `test_view_as_real`, `fn`, `test_view_tensor_split`, `test_view_tensor_hsplit`, `test_view_tensor_vsplit`, `test_view_tensor_dsplit`

**Key imports**: random, unittest, partial, combinations, permutations, product, numpy as np, torch, make_tensor


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `random`
- `unittest`
- `functools`: partial
- `itertools`: combinations, permutations, product
- `numpy as np`
- `torch`
- `torch.testing`: make_tensor


## Code Patterns & Idioms

### Common Patterns

- **Automatic Differentiation**: Uses autograd for gradient computation


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
python test/test_view_ops.py
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

- **File Documentation**: `test_view_ops.py_docs.md`
- **Keyword Index**: `test_view_ops.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
