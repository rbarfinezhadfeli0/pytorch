# Documentation: `docs/test/test_sparse.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_sparse.py_docs.md`
- **Size**: 54,019 bytes (52.75 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_sparse.py`

## File Metadata

- **Path**: `test/test_sparse.py`
- **Size**: 262,504 bytes (256.35 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# Owner(s): ["module: sparse"]
# ruff: noqa: F841

import torch
import itertools
import functools
import operator
import random
import unittest
from torch.testing import make_tensor
from torch.testing._internal.common_utils import TestCase, run_tests, do_test_dtypes, \
    load_tests, TEST_NUMPY, TEST_SCIPY, IS_WINDOWS, gradcheck, coalescedonoff, \
    DeterministicGuard, first_sample, TEST_WITH_CROSSREF, TEST_WITH_ROCM, skipIfTorchDynamo, \
    parametrize, subtest, is_coalesced_indices, suppress_warnings, instantiate_parametrized_tests, \
    skipIfCrossRef
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_mps import mps_ops_modifier
from numbers import Number
from typing import Any
from packaging import version
from torch.testing._internal.common_cuda import \
    (SM53OrLater, SM80OrLater, TEST_MULTIGPU)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops, dtypes, dtypesIfCUDA, dtypesIfMPS, onlyCPU, onlyCUDA, precisionOverride,
     deviceCountAtLeast, OpDTypes, onlyNativeDeviceTypes, skipCUDAIf, expectedFailureMPS,
     expectedFailureMPSComplex, largeTensorTest)
from torch.testing._internal.common_methods_invocations import \
    (op_db, reduction_ops, sparse_unary_ufuncs, sparse_masked_reduction_ops, binary_ufuncs)
from torch.testing._internal.common_dtype import (
    all_types, all_types_and_complex, all_mps_types, all_types_and_complex_and, floating_and_complex_types,
    floating_and_complex_types_and, integral_types, floating_types_and,
)
from torch.testing._internal.opinfo.definitions.sparse import validate_sample_input_sparse
from torch.testing._internal.opinfo.refs import (
    ElementwiseBinaryPythonRefInfo,
    ReductionPythonRefInfo
)

def _op_supports_any_sparse(op):
    return (op.supports_sparse
            or op.supports_sparse_csr
            or op.supports_sparse_csc
            or op.supports_sparse_bsr
            or op.supports_sparse_bsc)


reduction_ops_with_sparse_support = [
    op for op in reduction_ops if 'masked.' not in op.name and
    _op_supports_any_sparse(op) and not isinstance(op, ReductionPythonRefInfo)]

binary_ufuncs_with_sparse_support = [
    op for op in binary_ufuncs if _op_supports_any_sparse(op) and
    not isinstance(op, ElementwiseBinaryPythonRefInfo)]

like_fns_with_sparse_support = [op for op in op_db if _op_supports_any_sparse(op) and '_like' in op.name]

if TEST_SCIPY:
    import scipy.sparse

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

# batched grad doesn't support sparse
gradcheck = functools.partial(gradcheck, check_batched_grad=False)

CUSPARSE_SPMM_COMPLEX128_SUPPORTED = (
    IS_WINDOWS and torch.version.cuda
) or (not IS_WINDOWS and not TEST_WITH_ROCM)

HIPSPARSE_SPMM_COMPLEX128_SUPPORTED = torch.version.hip and version.parse(torch.version.hip.split("-")[0]) >= version.parse("6.0")

def all_sparse_layouts(test_name='layout', include_strided=False):
    return parametrize(test_name, [
        subtest(torch.strided, name='Strided'),
        subtest(torch.sparse_coo, name='SparseCOO'),
        subtest(torch.sparse_csr, name='SparseCSR'),
        subtest(torch.sparse_csc, name='SparseCSC'),
        subtest(torch.sparse_bsr, name='SparseBSR'),
        subtest(torch.sparse_bsc, name='SparseBSC'),
    ][(0 if include_strided else 1):])

def gradcheck_semantics(test_name='gradcheck'):
    gradcheck_sparse = functools.partial(gradcheck, masked=False)
    gradcheck_masked = functools.partial(gradcheck, masked=True)
    gradcheck_sparse.masked = False
    gradcheck_masked.masked = True
    return parametrize(test_name, [
        subtest(gradcheck_sparse, name='sparse'),
        subtest(gradcheck_masked, name='masked')])


class CrossRefSparseFakeMode(torch._subclasses.CrossRefFakeMode):
    def __init__(self) -> None:
        super().__init__(
            self.ignore_op, check_strides=False,
            check_aliasing=False,
        )  # TODO: enable stride/alias checking

    # empty_like excluded for now due to sparse complex
    # aten._to_dense.default this one is getting called with csc
    @staticmethod
    def ignore_op(func):
        return func in (
            torch.ops.aten.empty_like.default,
            torch.ops.aten.set_.source_Storage_storage_offset,
            torch.ops.aten.sspaddmm.out,
            torch.ops.aten._spdiags.default,
            torch.ops.aten._to_dense.default,
            torch.ops.aten.indices.default,
            torch.ops.aten._indices.default,
            torch.ops.aten.values.default,
            torch.ops.aten._values.default,
        )

class TestSparseLegacyAndDeprecation(TestCase):

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_legacy_warnings(self):

        def f1():
            "torch.sparse.SparseTensor() is deprecated."\
                "  Please use torch.sparse_coo_tensor((0,), dtype=)"
            x_ref = torch.sparse_coo_tensor((0,), dtype=torch.float64)
            x = torch.sparse.DoubleTensor()
            self.assertEqual(x, x_ref)

        def f2():
            "torch.sparse.SparseTensor(cdata=x._cdata) is deprecated."\
                "  Please use torch.sparse_coo_tensor(x._indices(), x._values(), x.shape)"
            x_ref = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64).to_sparse()
            x = torch.sparse.DoubleTensor(cdata=x_ref._cdata)
            y = torch.sparse_coo_tensor(x._indices(), x._values(), x.shape)
            self.assertEqual(x, x_ref)
            self.assertEqual(y, x_ref)

        def f3():
            "torch.sparse.SparseTensor(indices, values, *, device=) is deprecated."\
                "  Please use torch.sparse_coo_tensor(indices, values, dtype=, device=)"
            x_ref = torch.sparse_coo_tensor([[0, 0, 1, 1], [0, 1, 0, 1]], [1, 2, 3, 4], dtype=torch.float64)
            x = torch.sparse.DoubleTensor(torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]]),
                                          torch.tensor([1, 2, 3, 4], dtype=torch.float64))
            self.assertEqual(x, x_ref)

        def f4():
            "torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated."\
                "  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=)"
            x_ref = torch.sparse_coo_tensor([[0, 0, 1, 1], [0, 1, 0, 1]], [1, 2, 3, 4], (2, 3), dtype=torch.float64)
            x = torch.sparse.DoubleTensor(torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]]),
                                          torch.tensor([1, 2, 3, 4], dtype=torch.float64), (2, 3))
            self.assertEqual(x, x_ref)

        def f5():
            "torch.sparse.SparseTensor(shape, *, device=) is deprecated."\
                "  Please use torch.sparse_coo_tensor(shape, dtype=, device=)"
            x_ref = torch.sparse_coo_tensor((2, 3), dtype=torch.float64)
            x = torch.sparse.DoubleTensor(2, 3)
            self.assertEqual(x, x_ref)

        for test_f in [f1, f2, f3, f4, f5]:

            with self.assertWarns(UserWarning, msg=test_f.__doc__) as cm:
                test_f()
                test_f()

            # Check warn-once:
            self.assertEqual(len(cm.warnings), 1)


class TestSparseBase(TestCase):
    def run(self, result=None):
        if TEST_WITH_CROSSREF:
            with CrossRefSparseFakeMode():
                return super().run(result)
        else:
            return super().run(result)

class TestSparse(TestSparseBase):

    def setUp(self):
        super().setUp()

        self.index_tensor = lambda *args, **kwargs: torch.tensor(*args, **kwargs, dtype=torch.int64)

        def sparse_empty_factory(*args, **kwargs):
            kwargs['layout'] = kwargs.get('layout', torch.sparse_coo)
            return torch.empty(*args, **kwargs)
        self.sparse_empty = sparse_empty_factory

        def sparse_tensor_factory(*args, **kwargs):
            return torch.sparse_coo_tensor(*args, **kwargs)
        self.sparse_tensor = sparse_tensor_factory

    def _gen_sparse(self, sparse_dim, nnz, with_size, dtype, device, coalesced):
        if isinstance(with_size, Number):
            with_size = [with_size] * sparse_dim

        x, i, v = self.genSparseTensor(with_size, sparse_dim, nnz, not coalesced, dtype=dtype, device=device)

        if not coalesced:
            self.assert_uncoalesced(x)

        return x, i, v

    def assert_uncoalesced(self, x):
        """
        Test if a CPU tensor is uncoalesced.  This is used to ensure
        correctness of the uncoalesced tensor generation algorithm.
        """
        assert not x.is_coalesced()
        existing_indices = set()
        indices = x._indices()
        for i in range(x._nnz()):
            index = str(indices[:, i])
            if index in existing_indices:
                return True
            else:
                existing_indices.add(index)

    def test_negative_indices(self):
        indices = torch.tensor([[0, 1, -1], [2, 0, 1]])
        values = torch.tensor([1, 2, 3])
        shape = torch.Size([3, 3])
        self.assertRaisesRegex(RuntimeError, "found negative index", lambda: torch.sparse_coo_tensor(indices, values, shape))

    def randn(self, *args, **kwargs):
        """
        Variant of torch.randn that also works in the TEST_CUDA case.
        """
        # TODO: Put this in torch.cuda.randn
        return torch.empty(*args, **kwargs).normal_()

    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    def test_print_coalesced(self, device, dtype):
        self._test_print(device, dtype, True)

    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    def test_print_uncoalesced(self, device, dtype):
        self._test_print(device, dtype, False)

    def _test_print(self, device, dtype, coalesced):
        shape_sparse_dim_nnz = [
            ((), 0, 2),
            ((0,), 0, 10),
            ((2,), 0, 3),
            ((100, 3), 1, 3),
            ((100, 20, 3), 2, 0),
            ((10, 0, 3), 0, 3),
            ((10, 0, 3), 0, 0),
        ]
        printed = []
        for shape, sparse_dim, nnz in shape_sparse_dim_nnz:
            indices_shape = torch.Size((sparse_dim, nnz))
            values_shape = torch.Size((nnz,) + shape[sparse_dim:])
            printed.append(f"# shape: {torch.Size(shape)}")
            printed.append(f"# nnz: {nnz}")
            printed.append(f"# sparse_dim: {sparse_dim}")
            printed.append(f"# indices shape: {indices_shape}")
            printed.append(f"# values shape: {values_shape}")

            indices = torch.arange(indices_shape.numel(), dtype=self.index_tensor(0).dtype,
                                   device=device).view(indices_shape)
            for d in range(sparse_dim):
                indices[d].clamp_(max=(shape[d] - 1))  # make it valid index
            if not coalesced and indices.numel() > 0:
                indices[:, -1] = indices[:, 0]  # make it uncoalesced
            values_numel = values_shape.numel()
            values = torch.arange(values_numel, dtype=dtype,
                                  device=device).view(values_shape).div_(values_numel / 2.)
            sp_tensor = self.sparse_tensor(indices, values, shape, dtype=dtype, device=device)

            dtypes = [torch.int32]
            if values.dtype == torch.double:
                dtypes.append(torch.float)
            else:
                dtypes.append(torch.double if values.device != torch.device("mps:0") else torch.float32)
            for dtype in dtypes:
                printed.append(f"########## {dtype} ##########")
                x = sp_tensor.detach().to(dtype)
                printed.append("# sparse tensor")
                printed.append(str(x))
                if x.dtype.is_floating_point:
                    printed.append("# after requires_grad_")
                    printed.append(str(x.requires_grad_()))
                    printed.append("# after addition")
                    printed.append(str(x + x))
                printed.append("# _indices")
                printed.append(str(x._indices()))
                printed.append("# _values")
                printed.append(str(x._values()))
            printed.append('')
        self.assertExpected('\n'.join(printed))

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_basic(self, device, dtype, coalesced):
        def test_shape(sparse_dims, nnz, with_size):
            if isinstance(with_size, Number):
                with_size = [with_size] * sparse_dims
            x, i, v = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)
            self.assertEqual(i, x._indices())
            self.assertEqual(v, x._values())
            self.assertEqual(x.ndimension(), len(with_size))
            self.assertEqual(x.coalesce()._nnz(), nnz if x.is_coalesced() else nnz // 2)
            self.assertEqual(list(x.size()), with_size)

            # Test .indices() and .values()
            if not coalesced:
                with self.assertRaisesRegex(RuntimeError, "Cannot get indices on an uncoalesced tensor"):
                    x.indices()
                with self.assertRaisesRegex(RuntimeError, "Cannot get values on an uncoalesced tensor"):
                    x.values()
            else:
                self.assertEqual(x.indices(), x._indices())
                self.assertEqual(x.values(), x._values())

        test_shape(3, 10, 100)
        test_shape(3, 10, [100, 100, 100])
        test_shape(3, 10, [100, 100, 100, 5, 5, 5, 0])
        test_shape(3, 0, [0, 0, 100, 5, 5, 5, 0])

        # Make sure that coalesce handles duplicate indices correctly
        i = self.index_tensor([[9, 0, 0, 0, 8, 1, 1, 1, 2, 7, 2, 2, 3, 4, 6, 9]], device=device)
        v = torch.tensor([[idx**2, idx] for idx in range(i.size(1))], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([10, 2]), dtype=dtype, device=device)
        self.assertEqual(x.coalesce()._nnz(), 9)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble, torch.bfloat16)
    @dtypesIfMPS(torch.float32, torch.complex64)
    @precisionOverride({torch.bfloat16: 1e-2})
    def test_coalesce(self, device, dtype, coalesced):

        def _test_coalesce(t):
            tc = t.coalesce()
            self.assertEqual(tc.to_dense(), t.to_dense())
            self.assertTrue(tc.is_coalesced())
            # Our code below doesn't work when nnz is 0, because
            # then it's a 0D tensor, not a 2D tensor.
            if t._nnz() == 0:
                self.assertEqual(t._indices(), tc._indices())
                self.assertEqual(t._values(), tc._values())
                return tc

            value_map: dict[Any, Any] = {}
            for idx, val in zip(t._indices().t(), t._values()):
                idx_tup = tuple(idx.tolist())
                if idx_tup in value_map:
                    value_map[idx_tup] += val
                else:
                    value_map[idx_tup] = val.clone() if isinstance(val, torch.Tensor) else val

            new_indices = sorted(value_map.keys())
            _new_values = [value_map[idx] for idx in new_indices]
            if t._values().ndimension() < 2:
                new_values = t._values().new(_new_values)
            else:
                new_values = torch.stack(_new_values)

            new_indices = t._indices().new(new_indices).t()
            tg = t.new(new_indices, new_values, t.size())

            self.assertEqual(tc._indices(), tg._indices())
            self.assertEqual(tc._values(), tg._values())

            if t.is_coalesced():
                self.assertEqual(tc._indices(), t._indices())
                self.assertEqual(tc._values(), t._values())

        for empty_i, empty_v, empty_nnz in itertools.product([True, False], repeat=3):
            sparse_size = [] if empty_i else [2, 1]
            dense_size = [1, 0, 2] if empty_v else [1, 2]
            nnz = 0 if empty_nnz else 5

            t, _, _ = self._gen_sparse(len(sparse_size), nnz, sparse_size + dense_size, dtype, device, coalesced)
            _test_coalesce(t)  # this tests correctness

    @onlyCUDA
    @largeTensorTest("30GB", "cuda")
    @skipCUDAIf(not SM80OrLater and not TEST_WITH_ROCM, "CUDA capability < SM80 and not ROCM")
    @dtypes(torch.float)
    def test_coalesce_accepts_large_tensor(self, device, dtype):
        N = 22500000
        NNZ = 272500000
        rows = torch.randint(0, N, (NNZ,), dtype=torch.int64, device=device)
        cols = torch.randint(0, N, (NNZ,), dtype=torch.int64, device=device)
        indices = torch.stack([rows, cols], dim=0)
        values = torch.randn(NNZ, dtype=dtype, device=device)
        sparse_matrix = torch.sparse_coo_tensor(indices, values, size=(N, N), dtype=torch.float32, device=device)
        sparse_matrix = sparse_matrix.coalesce()

    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/89395")
    def test_coalesce_reference_cycle(self, device, dtype):
        # Test coalesce doesn't create autograd graph cycles (gh-52253)

        # Sanity check that the helper class works as expected
        t = torch.rand(2)
        t_ref = torch._C._WeakTensorRef(t)
        self.assertFalse(t_ref.expired())

        del t
        self.assertTrue(t_ref.expired())

        def test_sparse_sum():
            i = torch.tensor([[0], [4]], dtype=torch.long, device=device)
            v = torch.tensor([[[-0.4567, -1.8797, 0.0380, 1.4316]]],
                             dtype=dtype, device=device)
            S = torch.sparse_coo_tensor(i, v)
            S = S.coalesce()
            S.requires_grad_(True)
            S2 = S.coalesce()
            self.assertTrue(S2.is_coalesced())
            return torch._C._WeakTensorRef(S2)

        ref = test_sparse_sum()
        self.assertTrue(ref.expired())

    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    def test_ctor_large_sizes(self, device, dtype):
        # Test that integer overflow is detected when computing numel
        # of a sparse tensor with large dimensions (gh-57416). Notice
        # that numel is computed internally when constructing a
        # tensor, hence the overflow may appear during the tensor
        # construction step.
        N = 100000
        indices = torch.tensor([[N, N - 1]] * 4, dtype=torch.int64, device=device)
        values = torch.tensor([1, 2], dtype=dtype, device=device)
        self.assertRaises(RuntimeError,
                          lambda: torch.sparse_coo_tensor(
                              indices, values, (N + 1,) * 4, device=device))

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_ctor_size_checks(self, device, dtype):
        indices = self.index_tensor([
            [0, 0, 0],
            [0, 3, 0],
            [0, 0, 0],
            [0, 0, 0],
        ], device=device)
        values = torch.tensor([2, 1, 3, 4], dtype=dtype, device=device)

        # indices inconsistent with size
        self.assertRaises(
            RuntimeError,
            lambda: self.sparse_tensor(indices, values, torch.Size([2, 1, 1])))

        # values inconsistent with size
        values = torch.tensor([
            [2, 1, 2, 1],
            [1, 0, 5, 2],
        ], dtype=dtype, device=device)
        self.assertRaises(
            RuntimeError,
            lambda: self.sparse_tensor(indices, values, torch.Size([2, 4, 2, 1])))

    @expectedFailureMPS
    @coalescedonoff
    @dtypes(torch.double)
    @dtypesIfMPS(torch.float32)
    def test_ctor_is_coalesced_with_gradcheck(self, device, dtype, coalesced):
        for sparse_size, nnz in (((3, 3), 5), ((2, 3, 1, 5), 11)):
            t, _, _ = self._gen_sparse(len(sparse_size), nnz, sparse_size, dtype, device, coalesced)
            self.assertEqual(t.is_coalesced(), coalesced)

            def func(indices, values, shape, is_coalesced):
                if shape is None:
                    s = torch.sparse_coo_tensor(indices, values, check_invariants=True, is_coalesced=is_coalesced)
                else:
                    s = torch.sparse_coo_tensor(indices, values, shape, check_invariants=True, is_coalesced=is_coalesced)
                self.assertEqual(s.is_coalesced(), is_coalesced)
                return s.to_dense(masked_grad=False)

            for shape in {t.shape, None}:
                if coalesced:
                    torch.autograd.gradcheck(func, (t._indices(), t._values().requires_grad_(True), shape, False))
                    torch.autograd.gradcheck(func, (t._indices(), t._values().requires_grad_(True), shape, True))
                else:
                    torch.autograd.gradcheck(func, (t._indices(), t._values().requires_grad_(True), shape, False))
                    with self.assertRaisesRegex(RuntimeError,
                                                "cannot set is_coalesced to true if indices correspond to uncoalesced COO tensor"):
                        torch.autograd.gradcheck(func, (t._indices(), t._values().requires_grad_(True), shape, True))

    @dtypes(*floating_and_complex_types_and(torch.float16, torch.bfloat16))
    @dtypesIfMPS(*all_mps_types())
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupported triggers assertion error")
    @gradcheck_semantics()
    def test_to_dense_with_gradcheck(self, device, dtype, gradcheck):

        def test_tensor(x, res):
            x.to_dense()  # Tests triple to_dense for memory corruption
            x.to_dense()
            x.to_dense()
            dense_x = x.to_dense()
            safe_dense_x = self.safeToDense(x)
            dense_x = dense_x.to(res.dtype)
            safe_dense_x = safe_dense_x.to(res.dtype)
            self.assertEqual(res, dense_x)
            self.assertEqual(res, safe_dense_x)

            # Only run autograd test for float64
            if x.dtype != torch.float64:
                return

            def fn(x):
                return x.to_dense(masked_grad=gradcheck.masked)
            x.requires_grad_(True)
            gradcheck(fn, (x,))

        values_types = [torch.double, torch.cdouble] if device != "mps:0" else [torch.float32, torch.complex64]
        for value_type in values_types:
            i = self.index_tensor([
                [0, 1, 2, 2],
                [0, 0, 0, 3],
                [0, 0, 1, 4],
            ], device=device)
            # we don't have to_dense for half types on CPU because it is implemented
            # with a slower add_ operation
            v = torch.tensor([2, 1, 3, 4], dtype=dtype, device=device)
            x = self.sparse_tensor(i, v, torch.Size([3, 4, 5]), dtype=value_type, device=device)
            res = torch.tensor([
                [[2, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]],
                [[1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]],
                [[0, 3, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 4]],
            ], dtype=dtype, device=device)

            test_tensor(x, res)
            test_tensor(res, res)

            i = self.index_tensor([
                [0, 1, 2, 2],
                [0, 0, 0, 3],
                [0, 0, 1, 4],
            ], device=device)
            v = torch.empty(4, 0, dtype=dtype, device=device)
            x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 0]), dtype=value_type, device=device)
            res = torch.empty((3, 4, 5, 0), dtype=dtype, device=device)
            test_tensor(x, res)

    @coalescedonoff
    @dtypes(torch.float16, torch.bfloat16, torch.float64, torch.int, torch.cfloat, torch.cdouble)
    @dtypesIfMPS(torch.float16, torch.bfloat16, torch.float32, torch.int, torch.cfloat)
    def test_to_sparse(self, device, dtype, coalesced):
        shape = [5, 2, 10, 4]
        max_nnz = 1
        dtypes = [torch.double, torch.cdouble] if device != "mps:0" else [torch.float32, torch.complex64]
        for value_type in dtypes:
            for dim, dim_sz in enumerate(shape, 1):
                max_nnz *= dim_sz
                rnnz = torch.randint(2, max_nnz, (1,)).item()
                for nnz in [0, 1, rnnz]:
                    expected, _, _ = self._gen_sparse(dim, nnz, shape, dtype=value_type, device=device,
                                                      coalesced=coalesced)
                    expected = expected.to(dtype)

                    d = expected.to_dense()
                    result = d.to_sparse(dim)
                    self.assertEqual(d, result.to_dense())
                    self.assertEqual(expected.size(), result.size())
                    self.assertEqual(dim, result.sparse_dim())

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_sparse_bool(self, device, dtype):
        a = torch.tensor([True, False], dtype=dtype, device=device).to(torch.bool)
        b = a.to_sparse().to_dense()
        self.assertEqual(a, b)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/108667")
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_scalar(self, device, dtype):
        # tensor with value
        a = self.sparse_tensor(self.index_tensor([], device=device).unsqueeze(1), 12.3, [], dtype=dtype, device=device)
        self.assertEqual(1, a._values().numel())
        self.assertEqual(a, a.clone())
        a_coalesced = a.coalesce()
        self.assertTrue(a_coalesced.is_coalesced())
        self.assertEqual(torch.tensor(12.3, dtype=dtype, device=device), a.to_dense())
        self.assertEqual(a, a.to_dense().to_sparse())

        # tensor with multiple values
        a = self.sparse_tensor(self.index_tensor([], device=device).unsqueeze(1).expand(0, 2),
                               [12.3, 12.3], [], dtype=dtype, device=device)
        self.assertEqual(2, a._values().numel())
        self.assertEqual(a, a.clone())
        a_coalesced = a.coalesce()
        self.assertTrue(a_coalesced.is_coalesced())
        self.assertEqual(torch.tensor(12.3 * 2, dtype=dtype, device=device), a.to_dense())
        self.assertEqual(a.coalesce(), a.coalesce().to_dense().to_sparse())

        # tensor without value
        a = self.sparse_empty((), dtype=dtype, device=device)
        self.assertEqual(0, a._values().numel())
        self.assertEqual(a, a.clone())
        a_coalesced = a.coalesce()
        self.assertTrue(a_coalesced.is_coalesced())
        self.assertEqual(torch.tensor(0, dtype=dtype, device=device), a.to_dense())
        self.assertEqual(a, a.to_dense().to_sparse())

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_shared(self, device, dtype):
        i = self.index_tensor([[2]], device=device)
        v = torch.tensor([5], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3]))
        v[0] = 6
        self.assertEqual(torch.tensor([0, 0, 6], dtype=dtype, device=device), self.safeToDense(x))
        i[0][0] = 0
        self.assertEqual(torch.tensor([6, 0, 0], dtype=dtype, device=device), self.safeToDense(x))

        i = self.index_tensor([[2]], device=device)
        v = torch.empty((1, 0), dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 0]))
        i[0][0] = 0
        self.assertEqual(torch.empty((3, 0), dtype=dtype, device=device), self.safeToDense(x))

    @expectedFailureMPS
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupported triggers assertion error")
    @gradcheck_semantics()
    def test_to_dense_hybrid(self, device, dtype, gradcheck):

        def test_tensor(x, res):
            x.to_dense()  # Tests double to_dense for memory corruption
            x.to_dense()
            x.to_dense()
            self.assertEqual(res, x.to_dense())
            self.assertEqual(res, self.safeToDense(x))

            def fn(x):
                return x.to_dense(masked_grad=gradcheck.masked)
            x.requires_grad_(True)
            gradcheck(fn, (x,))

        i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
        ], device=device)
        v = torch.tensor([[2, 3], [1, 2], [3, 4], [4, 5]], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 2]))
        res = torch.tensor([
            [[2, 3],
             [0, 0],
             [0, 0],
             [0, 0]],
            [[1, 2],
             [0, 0],
             [0, 0],
             [0, 0]],
            [[3, 4],
             [0, 0],
             [0, 0],
             [4, 5]],
        ], dtype=dtype, device=device)
        test_tensor(x, res)

        i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
        ], device=device)
        v = torch.empty((4, 2, 0), dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 2, 0]))
        res = torch.empty((3, 4, 2, 0), dtype=dtype, device=device)
        test_tensor(x, res)

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_contig(self, device, dtype):
        def test_tensor(x, exp_i, exp_v):
            x = x.coalesce()
            self.assertEqual(exp_i, x._indices())
            self.assertEqual(exp_v, x._values())

        i = self.index_tensor([
            [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ], device=device)
        v = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([100, 100]))
        exp_i = self.index_tensor([
            [0, 1, 6, 14, 27, 35, 39, 40, 66, 71],
            [31, 92, 65, 50, 34, 62, 22, 56, 74, 89],
        ], device=device)
        exp_v = torch.tensor([2, 1, 6, 4, 10, 3, 5, 9, 8, 7], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ], device=device)
        v = torch.tensor([3, 2, 4, 1], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5]))
        exp_i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ], device=device)
        exp_v = torch.tensor([2, 1, 3, 4], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ], device=device)
        v = torch.empty([4, 0], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 0]))
        exp_i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ], device=device)
        exp_v = torch.empty([4, 0], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        # Duplicate indices
        i = self.index_tensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ], device=device)
        v = torch.tensor([3, 2, 4, 1], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5]))
        exp_i = self.index_tensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ], device=device)
        exp_v = torch.tensor([6, 4], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ], device=device)
        v = torch.empty([4, 0], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 0]))
        exp_i = self.index_tensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ], device=device)
        exp_v = torch.empty([2, 0], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_contig_hybrid(self, device, dtype):
        def test_tensor(x, exp_i, exp_v):
            x = x.coalesce()
            self.assertEqual(exp_i, x._indices())
            self.assertEqual(exp_v, x._values())

        i = self.index_tensor([
            [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ], device=device)
        v = torch.tensor([
            [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
            [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
        ], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([100, 100, 2]))
        exp_i = self.index_tensor([
            [0, 1, 6, 14, 27, 35, 39, 40, 66, 71],
            [31, 92, 65, 50, 34, 62, 22, 56, 74, 89],
        ], device=device)
        exp_v = torch.tensor([
            [2, 3], [1, 2], [6, 7], [4, 5], [10, 11],
            [3, 4], [5, 6], [9, 10], [8, 9], [7, 8],
        ], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ], device=device)
        v = torch.tensor([[3, 3, 3], [2, 2, 2], [4, 4, 4], [1, 1, 1]], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 3]))
        exp_i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ], device=device)
        exp_v = torch.tensor([[2, 2, 2], [1, 1, 1], [3, 3, 3], [4, 4, 4]], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ], device=device)
        v = torch.empty([4, 3, 0], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 3, 0]))
        exp_i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ], device=device)
        exp_v = torch.empty([4, 3, 0], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        # Duplicate indices
        i = self.index_tensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ], device=device)
        v = torch.tensor([[3, 2, 3], [2, 1, 1], [4, 3, 4], [1, 1, 1]], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 3]))
        exp_i = self.index_tensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ], device=device)
        exp_v = torch.tensor([[6, 4, 5], [4, 3, 4]], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

        i = self.index_tensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ], device=device)
        v = torch.empty([4, 3, 0], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 3, 0]))
        exp_i = self.index_tensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ], device=device)
        exp_v = torch.empty([2, 3, 0], dtype=dtype, device=device)
        test_tensor(x, exp_i, exp_v)

    @coalescedonoff
    @dtypesIfMPS(torch.float32, torch.complex64)
    @dtypes(torch.double, torch.cdouble)
    def test_clone(self, device, dtype, coalesced):
        def test_shape(sparse_dims, nnz, with_size):
            x = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]
            if not coalesced:
                self.assertFalse(x.is_coalesced())
                y = x.clone()
                self.assertFalse(y.is_coalesced())
            x = x.coalesce()
            self.assertTrue(x.is_coalesced())
            y = x.clone()
            self.assertTrue(y.is_coalesced())

        test_shape(4, 20, 5)
        test_shape(3, 10, [100, 100, 100, 5, 5, 5, 0])
        test_shape(3, 0, [0, 0, 100, 5, 5, 5, 0])

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble, torch.bfloat16)
    @dtypesIfMPS(torch.float32, torch.complex64, torch.bfloat16)
    @precisionOverride({torch.bfloat16: 2e-2})
    def test_Sparse_to_Sparse_copy_(self, device, dtype, coalesced):
        # This is for testing torch.copy_(SparseTensor, SparseTensor)
        sparse_dims = 3
        nnz = 10
        sizes = [2, 3, 4, 5]  # hybrid sparse
        x1, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
        x2, _, _ = self._gen_sparse(sparse_dims, nnz + 10, sizes, dtype, device, coalesced)

        # test copy
        x2_dense = x2.to_dense()
        x1.copy_(x2)
        self.assertEqual(x2_dense, x1.to_dense())

        # test type conversion (when x1.copy_(x2), x1.dtype should stay the same)
        x1 = x1.to(torch.float32)

        x2 = x2.to(torch.float16)
        x1_dtype = x1.dtype
        x1.copy_(x2)
        self.assertEqual(x1_dtype, x1.dtype)

        x2 = x2.to(torch.float64) if device != "mps:0" else x2.to(torch.float32)
        x1_dtype = x1.dtype
        x1.copy_(x2)
        self.assertEqual(x1_dtype, x1.dtype)

        # test no broadcast
        self.assertRaises(RuntimeError, lambda: x1.copy_(x2.narrow_copy(0, 0, 1)))

        # test raise error on copy_() between dense and sparse Tensors
        self.assertRaises(RuntimeError, lambda: x1.copy_(torch.randn(5, 5)))

        # test autograd
        x1, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
        x2, _, _ = self._gen_sparse(sparse_dims, nnz + 10, sizes, dtype, device, coalesced)
        x2.requires_grad_(True)
        x1.copy_(x2)
        y = x1 * 2
        x2_clone = x2.clone()
        y.backward(x2_clone)
        expected_grad = x2_clone * 2
        self.assertEqual(expected_grad.to_dense(), x2.grad.to_dense())
        self.assertEqual(None, x1.grad)

    @coalescedonoff
    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    @dtypes(torch.double, torch.cdouble)
    def test_Sparse_to_Sparse_copy_multi_gpu(self, device, dtype, coalesced):
        # This is for testing torch.copy_(SparseTensor, SparseTensor) across GPU devices
        sparse_dims = 3
        nnz = 10
        sizes = [2, 3, 4, 5]  # hybrid sparse
        x1, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
        x2, _, _ = self._gen_sparse(sparse_dims, nnz + 10, sizes, dtype, device, coalesced)
        x1 = x1.to('cuda:0')

        def test_cross_device(x1, x2):
            x1_device = x1.device
            x1.copy_(x2)
            self.assertEqual(x2.to('cuda:0').to_dense(), x1.to_dense())
            self.assertEqual(x1_device, x1.device)

        test_cross_device(x1, x2.to('cuda:1'))  # test across gpu devices
        test_cross_device(x1, x2.to('cpu'))  # test between cpu and gpu

        # test autograd
        x2 = x2.to('cuda:1')
        x2.requires_grad_(True)
        x1.copy_(x2)
        y = x1 * 2
        x2_clone = x2.clone().to('cuda:0')
        y.backward(x2_clone)
        expected_grad = x2_clone * 2
        self.assertEqual(expected_grad.to_dense(), x2.grad.to('cuda:0').to_dense())
        self.assertEqual(None, x1.grad)

    @onlyCUDA
    def test_cuda_empty(self, device):
        def test_tensor(x):
            y = x.to(device)
            self.assertEqual(x.sparse_dim(), y.sparse_dim())
            self.assertEqual(x.dense_dim(), y.dense_dim())
            x = y.cpu()
            self.assertEqual(y.sparse_dim(), x.sparse_dim())
            self.assertEqual(y.dense_dim(), x.dense_dim())

        x = torch.sparse_coo_tensor((2, 3, 4), dtype=torch.float32)
        test_tensor(x)

        x = torch.sparse_coo_tensor((2, 3, 4), dtype=torch.float16)
        test_tensor(x)

        x = torch.sparse_coo_tensor((2, 3, 4), dtype=torch.float16)
        test_tensor(x)

        x = torch.sparse_coo_tensor((2, 3, 4, 0), dtype=torch.float32)
        test_tensor(x)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_transpose(self, device, dtype, coalesced):
        def test_shape(sparse_dims, nnz, with_size):
            x = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]
            y = self.safeToDense(x)

            for i, j in itertools.combinations(range(4), 2):
                x = x.transpose_(i, j)
                y = y.transpose(i, j)
                self.assertEqual(self.safeToDense(x), y)

                x = x.transpose(i, j)
                y = y.transpose(i, j)
                self.assertEqual(self.safeToDense(x), y)

        test_shape(4, 6, 3)
        test_shape(4, 3, [7, 7, 7, 3, 3, 3, 0])
        test_shape(4, 0, [0, 0, 7, 3, 3, 3, 0])

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    @expectedFailureMPS
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupported triggers assertion error")
    @gradcheck_semantics()
    def test_permute(self, device, dtype, coalesced, gradcheck):
        # trivial checks
        s = torch.rand(3, 3, 3, device=device, dtype=dtype).to_sparse()
        with self.assertRaisesRegex(RuntimeError, "does not match the length"):
            s.permute(dims=(1, 0))
        with self.assertRaisesRegex(RuntimeError, "duplicate dims"):
            s.permute(dims=(1, 1, 1))
        # Calling permute on a sparse tensor with an empty tuple used to segfault,
        # see https://github.com/pytorch/pytorch/issues/116325
        x = torch.rand((), device=device, dtype=dtype).to_sparse()
        x.permute(())
        self.assertEqual(len(x.values()), 1)

        def test_shape(sparse_dims, nnz, with_size):
            ndim = len(with_size)
            valid_sparse_dims = torch.arange(-ndim, -ndim + sparse_dims)
            valid_dense_dims = torch.arange(-ndim + sparse_dims, 0)

            for dims in itertools.permutations(range(-ndim, 0)):
                s = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]
                d = self.safeToDense(s)

                dims_sparse, _ = torch.tensor(dims[:sparse_dims]).sort()
                dims_dense, _ = torch.tensor(dims[sparse_dims:]).sort()

                if (valid_sparse_dims == dims_sparse).all() and (valid_dense_dims == dims_dense).all():
                    # if valid permutation, test for correctness
                    s_permuted = s.permute(dims)
                    self.assertEqual(s_permuted, d.permute(dims))

                    # if s is coalesced, and perm does not touch 0-dim,
                    # the result has to be coalesced as well
                    if dims[0] == 0:
                        self.assertEqual(s_permuted.is_coalesced(), s.is_coalesced())
                    else:
                        self.assertFalse(s_permuted.is_coalesced())

                    gradcheck(lambda t: t.permute(dims).to_dense(masked_grad=gradcheck.masked), s.requires_grad_())
                else:
                    # otherwise check if exception is thrown
                    fail_message = "transpositions between sparse and dense dimensions are not allowed"
                    with self.assertRaisesRegex(RuntimeError, fail_message):
                        s.permute(dims)

        test_shape(2, 3, [2, 3, 4, 5])
        test_shape(2, 3, [2, 2, 0])
        # if nnz=0, it is not true that t == t.to_dense().to_sparse()
        # unless t.sparse_dim == t.dim (i.e. t is not hybrid)
        test_shape(3, 0, [0, 0, 2])

    @coalescedonoff
    @onlyCPU
    @dtypes(torch.double)
    def test_coalesce_transpose_mm(self, device, dtype, coalesced):
        def test_shape(di, dj, dk, nnz):
            x, _, _ = self._gen_sparse(2, nnz, [dj, di], dtype, device, coalesced)
            y = torch.randn(dj, dk, dtype=dtype, device=device)

            x_coalesced = x.coalesce()
            self.assertTrue(x_coalesced.is_coalesced())

            x_coalesced_t = x_coalesced.t()
            # Transpose is `colasced`-preserving if the indices tensor is empty.
            self.assertEqual(x_coalesced_t.is_coalesced(), di * nnz == 0)

            res = torch.mm(x_coalesced_t, y)
            expected = torch.mm(self.safeToDense(x_coalesced_t), y)
            self.assertEqual(res, expected)

        test_shape(10, 20, 30, 20)
        test_shape(0, 20, 30, 0)
        test_shape(10, 0, 30, 0)
        test_shape(10, 20, 0, 0)
        test_shape(10, 20, 0, 20)

    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1166")
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_t_empty(self, device, dtype):
        def test_in_place(x):
            shape_original = x.shape
            x.t_()
            self.assertEqual(torch.Size([shape_original[1], shape_original[0]]), x.size())
            self.assertEqual(0, x._indices().numel())
            self.assertEqual(0, x._values().numel())
            self.assertEqual(x.sparse_dim(), 2)
            self.assertEqual(x.dense_dim(), 0)

        def test_not_in_place(x):
            shape_original = x.shape
            y = x.t()
            self.assertEqual(torch.Size([shape_original[1], shape_original[0]]), y.size())
            self.assertEqual(0, y._indices().numel())
            self.assertEqual(0, y._values().numel())
            self.assertEqual(x.sparse_dim(), 2)
            self.assertEqual(x.dense_dim(), 0)

        x = self.sparse_empty(2, 3, dtype=dtype, device=device)
        test_in_place(x)
        test_not_in_place(x)

        x = self.sparse_empty(2, 0, dtype=dtype, device=device)
        test_in_place(x)
        test_not_in_place(x)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_add_zeros(self, device, dtype, coalesced):
        def test_shape(sparse_dims, nnz, sizes):
            x, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
            zeros = torch.sparse_coo_tensor(sizes, device=x.device)
            r1 = zeros + x
            r2 = x + zeros
            self.assertEqual(r1, x)
            self.assertEqual(r2, x)

        test_shape(1, 20, [1])
        test_shape(4, 20, [3, 17, 19, 5])
        test_shape(2, 20, [3, 17, 19, 5])
        test_shape(2, 20, [3, 17, 19, 0])

    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_add_sub_nnz(self, device, dtype):
        # nnz should not grow unbounded (gh-34964)
        x = torch.randn(10, dtype=dtype, device=device).to_sparse()
        x.add_(x)
        x.add_(x)
        self.assertLessEqual(x._nnz(), 10)

        x.sub_(2 * x)
        x.sub_(2 * x)
        self.assertLessEqual(x._nnz(), 10)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_cat(self, device, dtype, coalesced):
        # shapes: list of tuples (sparse_dims, nnz, sizes)
        def test_shapes(shapes, dim, fail_message=None):
            inputs = [self._gen_sparse(shape[0], shape[1], shape[2], dtype, device, coalesced)[0]
                      for shape in shapes]
            if fail_message:
                with self.assertRaisesRegex(RuntimeError, fail_message):
                    torch.cat(inputs, dim)
            else:
                result = torch.cat(inputs, dim)
                dense_result = torch.cat([t.to_dense() for t in inputs], dim)
                self.assertEqual(dense_result, result.to_dense())

        test_shapes(
            [(3, 10, [2, 3, 4]), (3, 10, [2, 1, 4]), (3, 10, [2, 4, 4])], 1)

        # mismatched sizes
        test_shapes([(3, 10, [2, 3, 4]), (3, 10, [2, 1, 4])], 0,
                    "All tensors must have the same shape: \\[2, 3, 4].*\\[2, 1, 4]")
        # hybrid sparse/dense
        test_shapes(
            [(2, 10, [2, 3, 4]), (2, 10, [2, 1, 4]), (2, 10, [2, 4, 4])], 1)
        # cat along dense dim
        test_shapes([(2, 10, [2, 3, 4]), (2, 10, [2, 3, 7])], 2)
        test_shapes([(1, 10, [2, 3, 4]), (1, 10, [2, 3, 4])], 1)
        test_shapes([(1, 10, [2, 3, 4]), (1, 10, [2, 3, 4])], 2)
        # mismatched dimensions
        test_shapes([(2, 10, [2, 3, 4]), (3, 10, [2, 3, 4])], 0,
                    "All tensors must have the same.*2, 1, but tensor at position 1 has 3, 0.")
        # wrapped dimension
        test_shapes(
            [(3, 10, [2, 3, 4]), (3, 10, [2, 1, 4]), (3, 10, [2, 4, 4])], -2)

        # sparse with dense
        sp = self._gen_sparse(3, 10, [2, 3, 4], dtype, device, coalesced)[0]
        dn = sp.to_dense()
        with self.assertRaisesRegex(RuntimeError,
                                    "Concatenating sparse tensors, but a dense tensor was found at position 1."):
            torch.cat((sp, dn))

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypesIfMPS(torch.float32, torch.complex64)
    def test_unsqueeze(self, device, dtype, coalesced):
        def test_shape(sparse_dims, nnz, sizes, unsqueeze_dim, fail_message=None):
            x, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
            if fail_message:
                with self.assertRaisesRegex(IndexError, fail_message):
                    torch.unsqueeze(x, u
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
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_sparse.py_docs.md
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

- **File Documentation**: `test_sparse.py_docs.md_docs.md`
- **Keyword Index**: `test_sparse.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
