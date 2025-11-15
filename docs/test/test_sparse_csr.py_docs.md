# Documentation: `test/test_sparse_csr.py`

## File Metadata

- **Path**: `test/test_sparse_csr.py`
- **Size**: 213,333 bytes (208.33 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# Owner(s): ["module: sparse"]
# ruff: noqa: F841

import torch
import random
import io
import itertools
import unittest
import functools
from contextlib import redirect_stderr
from torch.testing import make_tensor, FileCheck
from torch.testing._internal.common_cuda import SM53OrLater, SM80OrLater, TEST_CUSPARSE_GENERIC
from torch.testing._internal.common_utils import \
    (TEST_WITH_TORCHINDUCTOR, TEST_WITH_ROCM, TEST_CUDA_CUDSS, TEST_SCIPY, TEST_NUMPY, TEST_MKL, IS_WINDOWS, TestCase,
     run_tests, load_tests, coalescedonoff, parametrize, subtest, skipIfTorchDynamo,
     skipIfRocmVersionLessThan, IS_FBCODE, IS_REMOTE_GPU, suppress_warnings)
from torch.testing._internal.common_device_type import \
    (ops, instantiate_device_type_tests, dtypes, OpDTypes, dtypesIfCUDA, onlyCPU, onlyCUDA, skipCUDAIfNoSparseGeneric,
     precisionOverride, skipMeta, skipCUDAIf, skipCUDAIfRocm, skipCPUIfNoMklSparse, largeTensorTest)
from torch.testing._internal.common_methods_invocations import \
    (op_db, sparse_csr_unary_ufuncs, ReductionOpInfo)
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_dtype import (
    floating_types, all_types_and_complex_and, floating_and_complex_types, floating_types_and,
    all_types_and_complex, floating_and_complex_types_and)
from torch.testing._internal.opinfo.definitions.linalg import sample_inputs_linalg_solve
from torch.testing._internal.opinfo.definitions.sparse import validate_sample_input_sparse
from test_sparse import CUSPARSE_SPMM_COMPLEX128_SUPPORTED, HIPSPARSE_SPMM_COMPLEX128_SUPPORTED
import operator

if TEST_SCIPY:
    import scipy.sparse as sp

if TEST_NUMPY:
    import numpy as np
# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

no_mkl_sparse = IS_WINDOWS or not TEST_MKL


def _check_cusparse_spgemm_available():
    # cusparseSpGEMM was added in 11.0
    return not TEST_WITH_ROCM


_sparse_csr_ops = list(filter(lambda op: op.supports_sparse_csr, op_db))
_sparse_compressed_ops = list(filter(lambda op: (op.supports_sparse_csr or op.supports_sparse_csc
                                                 or op.supports_sparse_bsr or op.supports_sparse_bsc), op_db))
binary_functions_with_dense_output = ['mm', 'mv', ]
binary_ops_with_dense_output = list(filter(lambda op: op.name in binary_functions_with_dense_output, op_db))

UNARY_EWISE_CSR_ALLOW_AUTOGRAD = [
    'abs',
    'conj_physical',
    'deg2rad',
    'neg',
    'positive',
    'frac',
    'nn.functional.relu',
    'log1p',
    'rad2deg'
]

# This should be just an import from test_linalg instead of code duplication
# but https://github.com/pytorch/pytorch/pull/63511#discussion_r733989701
def _test_addmm_addmv(
    test_case,
    f,
    t,
    m,
    v,
    *,
    alpha=None,
    beta=None,
    transpose_out=False,
    layout=torch.strided,
    mode=None
):
    """
    Unified test for checking `f(t, m, v, alpha=alpha, beta=beta)` computation,
    where f is `torch.addmv` or `torch.addmm`.
    `transpose_out` controls whether the out argument is in column-major order.
    `layout` controls whether `m` is converted to specified layout or not.
    Custom behaviour is implemented only for torch.sparse_csr layout.
    """
    dtype = t.dtype
    numpy_dtype = dtype
    if dtype in {torch.bfloat16}:
        numpy_dtype = torch.float
    if dtype.is_complex:
        alpha = 0.9 + 0.3j if alpha is None else alpha
        beta = 0.5 + 0.6j if beta is None else beta
    else:
        alpha = 1.2 if alpha is None else alpha
        beta = 0.8 if beta is None else beta

    def convert_layout(mat):
        if layout == torch.sparse_csr:
            return mat.to_sparse_csr()
        elif layout == torch.sparse_csc:
            return mat.to_sparse_csc()
        else:
            assert mat.layout == layout
            return mat

    if mode == "all_sparse":
        res1 = f(*map(convert_layout, (t, m, v)), alpha=alpha, beta=beta)
        test_case.assertEqual(res1.layout, layout)
        res1 = res1.to_dense()
    elif mode == "dense_result":
        res1 = f(t, convert_layout(m), convert_layout(v), alpha=alpha, beta=beta)
    else:
        res1 = f(t, convert_layout(m), v, alpha=alpha, beta=beta)
    res2 = torch.full_like(res1, float('nan'))
    if transpose_out:
        res2 = res2.t().clone(memory_format=torch.contiguous_format).t()
    f(t, convert_layout(m), v, alpha=alpha, beta=beta, out=res2)
    res3 = alpha * (m.to(numpy_dtype).cpu().numpy() @ v.to(numpy_dtype).cpu().numpy())
    if beta != 0:
        res3 += (beta * t).to(numpy_dtype).cpu().numpy()
    res3 = torch.from_numpy(res3).to(dtype)
    test_case.assertEqual(res1, res2)
    test_case.assertEqual(res1, res3)


class TestSparseCSRSampler(TestCase):

    def test_make_crow_indices(self):
        # Here we test the correctness of the crow_indices algorithm
        # and testing it on CPU and with int32 dtype will be
        # sufficient.
        device = torch.device('cpu')
        index_dtype = torch.int32
        for n_rows in range(1, 10):
            for n_cols in range(1, 10):
                for nnz in range(n_rows * n_cols + 1):
                    crow_indices = self._make_crow_indices(
                        n_rows, n_cols, nnz,
                        device=device, dtype=index_dtype)
                    self.assertEqual(len(crow_indices), n_rows + 1)
                    counts = crow_indices[1:] - crow_indices[:-1]
                    self.assertEqual(counts.sum(), nnz)
                    self.assertGreaterEqual(counts.min(), 0)
                    self.assertLessEqual(counts.max(), n_cols)


def all_sparse_compressed_layouts(test_name='layout'):
    return parametrize(test_name, [
        subtest(torch.sparse_csr, name='SparseCSR'),
        subtest(torch.sparse_csc, name='SparseCSC'),
        subtest(torch.sparse_bsr, name='SparseBSR'),
        subtest(torch.sparse_bsc, name='SparseBSC')])


def sparse_compressed_nonblock_layouts(test_name='layout'):
    return parametrize(test_name, [
        subtest(torch.sparse_csr, name='SparseCSR'),
        subtest(torch.sparse_csc, name='SparseCSC')])


sparse_compressed_indices_methods = {
    torch.sparse_csr: (torch.Tensor.crow_indices, torch.Tensor.col_indices),
    torch.sparse_csc: (torch.Tensor.ccol_indices, torch.Tensor.row_indices),
    torch.sparse_bsr: (torch.Tensor.crow_indices, torch.Tensor.col_indices),
    torch.sparse_bsc: (torch.Tensor.ccol_indices, torch.Tensor.row_indices),
}


def batched_nonbatched(test_name='batched'):
    return parametrize(test_name, [
        subtest(True, name="Batched"),
        subtest(False, name="NonBatched")
    ])


def hybrid_nonhybrid(test_name='hybrid'):
    return parametrize(test_name, [
        subtest(True, name="Hybrid"),
        subtest(False, name="NonHybrid")
    ])


class TestSparseCompressed(TestCase):
    """Testing sparse compressed (CSR, CSC, BSR, BSC) tensor generic features.
    """

    def genTensor(self, size, nnz, *, layout, device=None, dtype=torch.float, index_dtype=torch.int64):
        if device is None:
            device = self.device_type
        return self.genSparseCompressedTensor(size, nnz, device=device, dtype=dtype, index_dtype=index_dtype, layout=layout)

    @all_sparse_compressed_layouts()
    @onlyCPU
    def test_layout(self, layout):
        self.assertIn(str(layout), {'torch.sparse_csr', 'torch.sparse_csc', 'torch.sparse_bsr', 'torch.sparse_bsc'})
        self.assertEqual(type(layout), torch.layout)

    @parametrize('shape_and_device_inference', [subtest(False, name='_'), subtest(True, name='shape_and_device_inference')])
    @parametrize('use_factory_function', [subtest(False, name='_'), subtest(True, name='factory')])
    @parametrize('input_kind', [subtest('tensor', name='from_tensor'), subtest('list', name='from_list')])
    @all_sparse_compressed_layouts()
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_sparse_compressed_constructor(self, layout, device, dtype,
                                           use_factory_function, shape_and_device_inference, input_kind):
        if input_kind == 'list' and shape_and_device_inference:
            if torch.device(device).type == 'cuda':
                # list inputs to factory/constructor function without
                # specifying device will result a sparse compressed tensor
                # on CPU. So, skip testing against cuda device as unused.
                self.skipTest("nothing to test")
            if dtype not in {torch.float32, torch.complex64, torch.int64, torch.bool}:
                self.skipTest("dtype not supported with list values")

        expected_devices = [torch.device(device)]
        if TEST_CUDA and torch.device(device).type == 'cuda' and torch.cuda.device_count() >= 2 and not shape_and_device_inference:
            expected_devices.append(torch.device('cuda:1'))

        factory_function = {
            torch.sparse_csr: torch.sparse_csr_tensor,
            torch.sparse_csc: torch.sparse_csc_tensor,
            torch.sparse_bsr: torch.sparse_bsr_tensor,
            torch.sparse_bsc: torch.sparse_bsc_tensor,
        }[layout]
        compressed_indices_mth, plain_indices_mth = sparse_compressed_indices_methods[layout]
        if input_kind == 'list':
            index_dtypes = [torch.int64]
        else:
            index_dtypes = [torch.int32, torch.int64]
        if dtype.is_floating_point or dtype.is_complex:
            requires_grad_lst = [False, True]
        else:
            requires_grad_lst = [False]
        for index_dtype in index_dtypes:
            for expected_device in expected_devices:
                for (compressed_indices, plain_indices, values), kwargs in self.generate_simple_inputs(
                        layout, device=expected_device, dtype=dtype, index_dtype=index_dtype,
                        # skip zero-sized tensors for list inputs:
                        enable_zero_sized=input_kind != 'list',
                        output_tensor=False):
                    size = kwargs['size']
                    if shape_and_device_inference and 0 in size:
                        # skip shape inference for zero-sized tensor
                        # inputs because (i) the shape determined from
                        # an empty list is ambiguous, and (ii) the
                        # size of the plain dimension defined as
                        # max(plain_indices) is undefined if
                        # plain_indices has no values
                        continue
                    compressed_indices_expect = compressed_indices
                    plain_indices_expect = plain_indices
                    values_expect = values

                    if input_kind == 'list':
                        compressed_indices = compressed_indices.tolist()
                        plain_indices = plain_indices.tolist()
                        values = values.tolist()

                    for requires_grad in requires_grad_lst:
                        if use_factory_function:
                            if shape_and_device_inference:
                                sparse = factory_function(
                                    compressed_indices, plain_indices, values, requires_grad=requires_grad)
                            else:
                                sparse = factory_function(
                                    compressed_indices, plain_indices, values, size,
                                    dtype=dtype, device=expected_device, requires_grad=requires_grad)
                        else:
                            if shape_and_device_inference:
                                sparse = torch.sparse_compressed_tensor(
                                    compressed_indices, plain_indices, values,
                                    layout=layout, requires_grad=requires_grad)
                            else:
                                sparse = torch.sparse_compressed_tensor(
                                    compressed_indices, plain_indices, values, size,
                                    dtype=dtype, layout=layout, device=expected_device, requires_grad=requires_grad)

                        self.assertEqual(layout, sparse.layout)
                        self.assertEqual(size, sparse.shape)
                        self.assertEqual(compressed_indices_expect, compressed_indices_mth(sparse))
                        self.assertEqual(plain_indices_expect, plain_indices_mth(sparse))
                        self.assertEqual(values_expect, sparse.values())
                        self.assertEqual(sparse.device, sparse.values().device)
                        self.assertEqual(sparse.device, expected_device)
                        self.assertEqual(sparse.values().requires_grad, requires_grad)
                        self.assertEqual(sparse.requires_grad, requires_grad)
                        self.assertFalse(compressed_indices_mth(sparse).requires_grad)
                        self.assertFalse(plain_indices_mth(sparse).requires_grad)

    @skipMeta
    @sparse_compressed_nonblock_layouts()
    @dtypes(*all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half))
    def test_empty(self, layout, device, dtype):
        ns = [5, 2, 0]
        batch_shapes = [(), (2,), (2, 3)]
        compressed_dim = {
            torch.sparse_csr: -2,
            torch.sparse_csc: -1,
        }[layout]
        compressed_indices_mth, plain_indices_mth = sparse_compressed_indices_methods[layout]
        for m, n, b in itertools.product(ns, ns, batch_shapes):
            shape = (*b, m, n)
            with torch.sparse.check_sparse_tensor_invariants(enable=False):
                # torch.empty may return invalid sparse compressed tensors
                result = torch.empty(shape, dtype=dtype, device=device, layout=layout)
            self.assertEqual(result.shape, shape)
            self.assertEqual(result.dtype, dtype)
            self.assertEqual(result.device, torch.device(device))
            self.assertEqual(result.layout, layout)
            self.assertEqual(compressed_indices_mth(result).shape, (*b, shape[compressed_dim] + 1,))
            self.assertEqual(plain_indices_mth(result).shape, (*b, 0,))
            self.assertEqual(result.values().shape, (*b, 0,))
            self.assertEqual(result._nnz(), 0)
            self.assertEqual(compressed_indices_mth(result).device, torch.device(device))
            self.assertEqual(plain_indices_mth(result).device, torch.device(device))
            self.assertEqual(result.values().device, torch.device(device))
            self.assertEqual(compressed_indices_mth(result).dtype, torch.int64)
            self.assertEqual(plain_indices_mth(result).dtype, torch.int64)
            self.assertEqual(result.values().dtype, dtype)

    @skipMeta
    @sparse_compressed_nonblock_layouts()
    @dtypes(*all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16))
    def test_empty_errors(self, layout, device, dtype):
        with self.assertRaisesRegex(RuntimeError,
                                    "torch.empty: Only batched sparse compressed \\(non-block\\) tensors are supported"
                                    ", but got size"):
            torch.empty((5,), dtype=dtype, device=device, layout=layout)

    @skipMeta
    @all_sparse_compressed_layouts()
    @dtypes(*all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half))
    def test_sparse_compressed_tensor_with_dims(self, layout, device, dtype):

        def get_sparse_compressed_tensor_properties(s):
            if layout in {torch.sparse_csr, torch.sparse_bsr}:
                compressed_indices, plain_indices = s.crow_indices(), s.col_indices()
            else:
                compressed_indices, plain_indices = s.ccol_indices(), s.row_indices()
            values = s.values()
            return dict(shape=s.shape, dtype=s.dtype, device=s.device, nnz=s._nnz(), layout=s.layout,
                        compressed_indices_shape=compressed_indices.shape,
                        compressed_indices_dtype=compressed_indices.dtype,
                        compressed_indices_device=compressed_indices.device,
                        plain_indices_shape=plain_indices.shape,
                        plain_indices_dtype=plain_indices.dtype,
                        plain_indices_device=plain_indices.device,
                        values_shape=values.shape,
                        values_dtype=values.dtype,
                        values_device=values.device)

        for index_dtype in [torch.int32, torch.int64]:
            for t in self.generate_simple_inputs(layout, device=device, dtype=dtype, index_dtype=index_dtype):
                dense_dim = t.dense_dim()
                sparse_dim = t.sparse_dim()
                batch_dim = t.ndim - sparse_dim - dense_dim
                nnz = t.values().shape[batch_dim]
                if layout in {torch.sparse_bsr, torch.sparse_bsc}:
                    blocksize = t.values().shape[batch_dim + 1: batch_dim + 1 + sparse_dim]
                else:
                    blocksize = ()

                e = torch.ops.aten._sparse_compressed_tensor_with_dims(nnz, dense_dim, t.shape, blocksize, index_dtype,
                                                                       dtype=dtype, layout=layout, device=device)

                e_prop, t_prop = get_sparse_compressed_tensor_properties(e), get_sparse_compressed_tensor_properties(t)
                for k, v in e_prop.items():
                    self.assertEqual(v, t_prop[k], lambda msg: f'{msg} when comparing {k}, expected {t_prop[k]}, got {v}')

    @skipMeta
    @all_sparse_compressed_layouts()
    @dtypes(*all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16))
    def test_clone(self, layout, device, dtype):
        for sparse in self.generate_simple_inputs(
                layout, device=device, dtype=dtype, index_dtype=torch.int32):
            cloned_sparse = sparse.clone()
            self.assertEqual(sparse, cloned_sparse)

    @all_sparse_compressed_layouts()
    def test_print(self, layout, device):
        compressed_indices_mth, plain_indices_mth = sparse_compressed_indices_methods[layout]
        printed = []
        for enable_hybrid in [False, True]:
            # using local patterns for test_print stability
            patterns = [
                # 2 x 3 batch of 3 x 2 tensors, trivial blocksize, non-hybrid/hybrid:
                ([[[[1, 2, 0],
                    [1, 0, 3]],
                   [[1, 2, 3],
                    [1, 0, 0]],
                   [[1, 0, 0],
                    [1, 2, 3]]],
                  [[[0, 2, 0],
                    [1, 2, 3]],
                   [[1, 0, 3],
                    [1, 2, 0]],
                   [[1, 2, 3],
                    [0, 2, 0]]]], [(2, 1)], [(), (4,)] if enable_hybrid else [()]),
                # tensor with non-trivial blocksize, non-hybrid/hybrid:
                ([[0, 1, 0, 2, 0, 2],
                  [0, 1, 0, 0, 2, 0],
                  [3, 3, 3, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 5, 0, 6, 6, 6],
                  [5, 0, 5, 6, 6, 6],
                  [0, 0, 0, 0, 8, 8],
                  [7, 7, 7, 0, 8, 8]], [(2, 3)], [(), (4, 2)] if enable_hybrid else [()]),
            ]
            for index_dtype in [torch.int32, torch.int64]:
                for dtype in [torch.float32, torch.float64]:
                    for (compressed_indices, plain_indices, values), kwargs in self.generate_simple_inputs(
                            layout, device=device, dtype=dtype, index_dtype=index_dtype, enable_hybrid=enable_hybrid,
                            enable_non_contiguous_indices=False, enable_non_contiguous_values=False,
                            enable_zero_sized=False, output_tensor=False, patterns=patterns):
                        size = tuple(kwargs['size'])
                        block_ndim = 2 if layout in {torch.sparse_bsr, torch.sparse_bsc} else 0
                        base_ndim = 2
                        batch_ndim = compressed_indices.dim() - 1
                        dense_ndim = values.dim() - batch_ndim - block_ndim - 1
                        if enable_hybrid and dense_ndim == 0:
                            # non-hybrid cases are covered by the enable_hybrid==False loop
                            continue
                        batchsize = size[:batch_ndim]
                        basesize = size[batch_ndim:batch_ndim + base_ndim]
                        densesize = size[batch_ndim + base_ndim:]
                        assert len(densesize) == dense_ndim
                        printed.append(f"########## {dtype}/{index_dtype}/size={batchsize}+{basesize}+{densesize} ##########")
                        x = torch.sparse_compressed_tensor(compressed_indices,
                                                           plain_indices,
                                                           values, size, dtype=dtype, layout=layout, device=device)
                        printed.append("# sparse tensor")
                        printed.append(str(x))
                        printed.append(f"# _{compressed_indices_mth.__name__}")
                        printed.append(str(compressed_indices_mth(x)))
                        printed.append(f"# _{plain_indices_mth.__name__}")
                        printed.append(str(plain_indices_mth(x)))
                        printed.append("# _values")
                        printed.append(str(x.values()))
                        printed.append('')
                    printed.append('')
        orig_maxDiff = self.maxDiff
        self.maxDiff = None
        try:
            self.assertExpected('\n'.join(printed))
            self.maxDiff = orig_maxDiff
        except Exception:
            self.maxDiff = orig_maxDiff
            raise

    @skipMeta
    @all_sparse_compressed_layouts()
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_copy(self, layout, device, dtype):

        def run_test(shape, blocksize, nnz, index_type):
            a = self.genSparseCompressedTensor(shape, nnz, dtype=dtype, layout=layout, device=device,
                                               index_dtype=index_dtype, blocksize=blocksize)
            b = self.genSparseCompressedTensor(shape, nnz, dtype=dtype, layout=layout, device=device,
                                               index_dtype=index_dtype, blocksize=blocksize)

            a.copy_(b)

            self.assertEqual(a, b)

        ns = [(9, 3), (2, 1), (0, 0)]  # (number of dimensions, the corresponding block size)
        batch_shapes = [(), (2,), (2, 3)]
        for ((m, bm), (n, bn), b), index_dtype in zip(itertools.product(ns, ns, batch_shapes), [torch.int32, torch.int64]):
            blocksize = (bm, bn) if layout in {torch.sparse_bsr, torch.sparse_bsc} else ()
            run_test((*b, m, n), blocksize, 0, index_dtype)
            run_test((*b, m, n), blocksize, m * n, index_dtype)

    @skipMeta
    @all_sparse_compressed_layouts()
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_copy_errors(self, layout, device, dtype):
        blocksize = (2, 3) if layout in {torch.sparse_bsr, torch.sparse_bsc} else ()
        nnz = 6 if layout in {torch.sparse_bsr, torch.sparse_bsc} else 1
        shape1 = (2 * 6, 3 * 6) if layout in {torch.sparse_bsr, torch.sparse_bsc} else (2, 3)
        for index_dtype in [torch.int32, torch.int64]:
            a = self.genSparseCompressedTensor(shape1, 0, dtype=dtype, layout=layout, device=device,
                                               index_dtype=index_dtype, blocksize=blocksize)

            with self.assertRaisesRegex(RuntimeError,
                                        "copy of sparse compressed tensors having different layouts is not supported."):
                a.copy_(torch.empty(a.shape, dtype=dtype, device=device))

            b = self.genSparseCompressedTensor(shape1, nnz, dtype=dtype, layout=layout, device=device,
                                               index_dtype=index_dtype, blocksize=blocksize)
            assert a._nnz() != b._nnz(), (a._nnz(), b._nnz())
            with self.assertRaisesRegex(RuntimeError,
                                        "only sparse compressed tensors with the same number of specified elements are supported."):
                a.copy_(b)

            shape2 = tuple(reversed(shape1))
            c = self.genSparseCompressedTensor(shape2, nnz, dtype=dtype, layout=layout, device=device,
                                               index_dtype=index_dtype, blocksize=blocksize)
            with self.assertRaisesRegex(
                    RuntimeError,
                    "expected shapes of self and src to match along dimension"):
                b.copy_(c)

            if blocksize:
                blocksize1 = tuple(reversed(blocksize))
                d = self.genSparseCompressedTensor(shape1, nnz, dtype=dtype, layout=layout, device=device,
                                                   index_dtype=index_dtype, blocksize=blocksize1)
                with self.assertRaisesRegex(RuntimeError,
                                            "copy of sparse compressed tensors having different block sizes is not supported"):
                    b.copy_(d)

    def _smallest_divisor(self, n):
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return i
        return n

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    @all_sparse_compressed_layouts()
    @ops(_sparse_compressed_ops)
    @precisionOverride({torch.bfloat16: 1e-2, torch.float16: 1e-2})
    def test_consistency(self, layout, device, dtype, op):
        """Checks that the op on a strided and on a sparse tensors will
        produce the same results.
        """
        if not op.supports_sparse_layout(layout):
            self.skipTest(f"{op.name} does not support input with {layout} layout")

        # FIXME: remove in followup once integer support is landed for segment_reduce
        if (layout == torch.sparse_csr and not dtype.is_floating_point
                and op.name in ('masked.mean', 'masked.amax', 'masked.amin')):
            self.skipTest(f"{op.name} does not support input with {layout} layout and {dtype} dtype")

        require_mask = isinstance(op, ReductionOpInfo) and 'masked.' in op.name

        samples = []
        for sample in op.sample_inputs(device, dtype):
            if sample.input.ndim < 2:
                continue
            dense_dim = sample.input.ndim - 2
            blocksize = (tuple(map(self._smallest_divisor, sample.input.shape[:2]))
                         if layout in {torch.sparse_bsr, torch.sparse_bsc} else None)

            def _to_sparse(x):
                if isinstance(x, torch.Tensor):
                    if blocksize is None:
                        if x.ndim != sample.input.ndim:
                            return x
                    elif x.ndim != sample.input.ndim + 2 or x.shape[-3] % blocksize[0] or x.shape[-2] % blocksize[1]:
                        return x
                    return x.clone().to_sparse(layout=layout, blocksize=blocksize, dense_dim=dense_dim)
                return x

            sparse_sample = sample.transform(_to_sparse)
            # Some strided samples (with inf, nan elements) appear to share
            # storage, so we must clone:
            sample = sample.transform(lambda x: (x.clone() if isinstance(x, torch.Tensor) else x))

            if validate_sample_input_sparse(op, sparse_sample, check_validate=False) is not sparse_sample:
                # that is, the validation returns the sparse sample
                # wrapped within ErrorInput instance
                continue
            samples.append((sample, sparse_sample))

        # Fail early to prevent silent success with this test
        if len(samples) == 0:
            raise ValueError("Expected at least one 2 or higher D tensor in samples.")

        # Re-define atol and rtol for operations that result values
        # are random (and hence, non-comparable) be we still want to
        # check the shape, dtype, etc attributes of the results:
        atol = rtol = None
        if op.name == 'randn_like':
            atol = 1e300
            rtol = 1

        for sample, sparse_sample in samples:
            expected = op(sample.input, *sample.args, **sample.kwargs)
            assert torch.is_tensor(expected)
            output = op(sparse_sample.input, *sparse_sample.args, **sparse_sample.kwargs)
            assert torch.is_tensor(output)
            strided_output = output.to_dense()
            if require_mask and sample.kwargs.get('mask') is not None:
                output_mask = torch.masked._output_mask(op.op, sample.input, *sample.args, **sample.kwargs)
                expected.masked_fill_(~output_mask, 0)
            self.assertEqual(strided_output, expected, atol=atol, rtol=rtol)

    @skipMeta
    @all_sparse_compressed_layouts()
    @all_sparse_compressed_layouts('layout2')
    @dtypes(*all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16))
    def test_empty_like(self, layout, layout2, device, dtype):
        for sparse in self.generate_simple_inputs(layout):
            if layout == layout2:
                result = torch.empty_like(sparse, layout=layout2)
                compressed_indices_mth, plain_indices_mth = sparse_compressed_indices_methods[result.layout]
                torch._validate_sparse_compressed_tensor_args(compressed_indices_mth(result),
                                                              plain_indices_mth(result),
                                                              result.values(),
                                                              result.shape,
                                                              result.layout)
                self.assertEqual(sparse.shape, result.shape)
            else:
                self.assertRaisesRegex(
                    RuntimeError,
                    "empty_like with different sparse layout is not supported",
                    lambda: torch.empty_like(sparse, layout=layout2)
                )

    @skipMeta
    @all_sparse_compressed_layouts()
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_validate(self, layout, device, dtype):
        def make_zero_batched(t):
            return torch.empty(*((0,) + t.shape), dtype=t.dtype, device=t.device)

        for index_dtype in [torch.int32, torch.int64]:
            for (compressed_indices, plain_indices, values), kwargs in self.generate_simple_inputs(
                    layout, device=device, dtype=dtype, index_dtype=index_dtype, output_tensor=False):
                size = kwargs['size']
                torch._validate_sparse_compressed_tensor_args(compressed_indices, plain_indices, values, size, layout)

                # check empty batch
                torch._validate_sparse_compressed_tensor_args(
                    *(make_zero_batched(t) for t in (compressed_indices, plain_indices, values)),
                    (0,) + size,
                    layout
                )

            compressed_indices = torch.tensor([0, 0], dtype=index_dtype)
            plain_indices = torch.tensor([], dtype=index_dtype)
            torch._validate_compressed_sparse_indices(layout in {torch.sparse_csr, torch.sparse_bsr},
                                                      compressed_indices, plain_indices, 1, 1, 0)

    def _generate_invalid_input(self, layout, device):
        from functools import partial

        def shape(shape, basedim=0):
            blocksize = (1, 1)
            if layout is torch.sparse_csc:
                shape = shape[:basedim] + (shape[basedim + 1], shape[basedim]) + shape[basedim + 2:]
            elif layout is torch.sparse_bsc:
                shape = shape[:basedim] + (shape[basedim + 1] * blocksize[1], shape[basedim] * blocksize[0]) + shape[basedim + 2:]
            elif layout is torch.sparse_bsr:
                shape = shape[:basedim] + (shape[basedim] * blocksize[0], shape[basedim + 1] * blocksize[1]) + shape[basedim + 2:]
            return shape

        def values(lst, device=device):
            if layout in {torch.sparse_bsr, torch.sparse_bsc}:
                lst = [[[item]] for item in lst]
            return torch.tensor(lst, device=device)

        tensor = partial(torch.tensor, device=device)
        values = partial(values, device=device)

        yield ('incontiguous compressed_indices',
               tensor([0, -1, 2, -1, 4, -1])[::2],
               tensor([0, 1, 0, 2]),
               values([1, 2, 3, 4]),
               shape((2, 3)),
               'expected compressed_indices to be a contiguous tensor per batch')

        yield ('incontiguous plain_indices',
               tensor([0, 2, 4]),
               tensor([0, -1, 1, -1, 0, -1, 2, -1])[::2],
               values([1, 2, 3, 4]),
               shape((2, 3)),
               'expected plain_indices to be a contiguous tensor per batch')

        yield ('0-D compressed_indices',
               tensor(0),
               tensor([0, 1, 0, 2]),
               values([1, 2, 3, 4]),
               shape((2, 3)),
               'compressed_indices must have dimensionality >= 1 but got 0')

        yield ('compressed/plain_indices mismatch of dimensionalities',
               tensor([[0, 2, 4]]),
               tensor([0, 1, 0, 2]),
               values([1, 2, 3, 4]),
               shape((2, 3)),
               'compressed_indices and plain_indices dimensionalities must be equal but got 2 and 1, respectively')

        if layout in {torch.sparse_csr, torch.sparse_csc}:
            yield ('indices and values mismatch of dimensionalities',
                   tensor([[0, 2, 4]]),
                   tensor([[0, 1, 0, 2]]),
                   values([1, 2, 3, 4]),
                   shape((2, 3)),
                   r'values must have dimensionality > sum of batch and block dimensionalities \(=1 \+ 0\) but got 1')
        else:
            yield ('indices and values mismatch of dimensionalities',
                   tensor([[0, 2, 4]]),
                   tensor([[0, 1, 0, 2]]),
                   values([1, 2, 3, 4]),
                   shape((2, 3)),
                   r'values must have dimensionality > sum of batch and block dimensionalities \(=1 \+ 2\) but got 3')

        yield ('invalid size',
               tensor([0, 2, 4]),
               tensor([0, 1, 0, 2]),
               values([1, 2, 3, 4]),
               (2,),
               r'tensor dimensionality must be sum of batch, base, and dense dimensionalities \(=0 \+ 2 \+ 0\) but got 1')

        yield ('invalid batchsize',
               tensor([[0, 2, 4]]),
               tensor([[0, 1, 0, 2]]),
               values([[1, 2, 3, 4]]),
               shape((2, 2, 3), 1),
               r'all batch dimensions of compressed_indices \(=\[1\]\), plain_indices \(=\[1\]\), '
               r'and values \(=\[1\]\) must be equal to tensor batch dimensions \(=\[2\]\)')

        if layout is torch.sparse_bsr:
            yield ('invalid blocksize',
                   tensor([0, 2, 4]),
                   tensor([0, 1, 0, 2]),
                   tensor([[[1, 11]], [[2, 22]], [[3, 33]], [[4, 33]]]),
                   shape((2, 3)),
                   r'tensor shape\[1\] \(=3\) must be divisible with blocksize\[1\] \(=2\) as defined by values shape')

        if layout is torch.sparse_bsc:
            yield ('invalid blocksize',
                   tensor([0, 2, 4]),
                   tensor([0, 1, 0, 2]),
                   tensor([[[1, 11]], [[2, 22]], [[3, 33]], [[4, 33]]]),
                   shape((3, 2)),
                   r'tensor shape\[1\] \(=3\) must be divisible with blocksize\[1\] \(=2\) as defined by values shape')

        yield ('invalid compressed_indices shape',
               tensor([0, 2, 3, 4]),
               tensor([0, 1, 0, 2]),
               values([1, 2, 3, 4]),
               shape((2, 3)),
               r'compressed_indices.shape\[-1\] must be equal to the number of compressed_indices_names \+ 1 \(=3\), but got 4')

        yield ('invalid compressed_indices shape',
               tensor([0, 2, 4]),
               tensor([0, 1, 0, 1, 2]),
               values([1, 2, 3, 4]),
               shape((2, 3)),
               r'plain_indices.shape\[-1\] must be equal to nnz \(=4\) as defined by values.shape\[0\], but got 5')

        yield ('compressed/plain_indices mismatch of dtype',
               tensor([0, 2, 4], dtype=torch.int32),
               tensor([0, 1, 0, 2], dtype=torch.int64),
               values([1, 2, 3, 4]),
               shape((2, 3)),
               r'compressed_indices and plain_indices must have the same dtype, bot got Int and Long, respectively')

        yield ('invalid compressed/plain_indices dtype',
               tensor([0, 2, 4], dtype=torch.int16),
               tensor([0, 1, 0, 2], dtype=torch.int16),
               values([1, 2, 3, 4]),
               shape((2, 3)),
               r'compressed_indices and plain_indices dtype must be Int or Long, but got Short')

        # CUDA kernel asserts are not recoverable, so we skip these for now
        if torch.device(device).type == 'cpu':
            yield ('invalid compressed_indices[0]',
                   tensor([1, 2, 4]),
                   tensor([0, 1, 0, 2]),
                   values([1, 2, 3, 4]),
                   shape((2, 3)),
                   r'`compressed_indices\[..., 0\] == 0` is not satisfied.')

            yield ('invalid compressed_indices[0] when nnz == 0',
                   tensor([1, 0], dtype=torch.int64),
                   tensor([], dtype=torch.int64),
                   values([1])[:0],
                   shape((1, 1)),
                   r'`compressed_indices\[..., 0\] == 0` is not satisfied.')

            yield ('invalid compressed_indices[-1]',
                   tensor([0, 2, 5]),
                   tensor([0, 1, 0, 2]),
                   values([1, 2, 3, 4]),
                   shape((2, 3)),
                   r'`compressed_indices\[..., -1\] == nnz` is not satisfied.')

            yield ('invalid compressed_indices[-1] when nnz == 0',
                   tensor([0, 1], dtype=torch.int64),
                   tensor([], dtype=torch.int64),
                   values([1])[:0],
                   shape((1, 1)),
                   r'`compressed_indices\[..., -1\] == nnz` is not satisfied.')

            yield ('invalid compressed_indices.diff(dim=-1)',
                   tensor([0, 0, 4]),
                   tensor([0, 1, 0, 2]),
                   values([1, 2, 3, 4]),
                   shape((2, 3)),
                   r'0 <= compressed_indices\[..., 1:\] - compressed_indices\[..., :\-1\] <= plain_dim` is not satisfied.')

            yield ('invalid compressed_indices.diff(dim=-1)',
                   tensor([0, 5, 4]),
                   tensor([0, 1, 0, 2]),
                   values([1, 2, 3, 4]),
                   shape((2, 3)),
                   r'0 <= compressed_indices\[..., 1:\] - compressed_indices\[..., :\-1\] <= plain_dim` is not satisfied.')

            yield ('invalid min(plain_indices)',
                   tensor([0, 2, 4]),
                   tensor([0, -1, 0, 3]),
                   values([1, 2, 3, 4]),
                   shape((2, 3)),
                   r'`0 <= plain_indices < plain_dim` is not satisfied.')

            yield ('invalid max(plain_indices)',
                   tensor([0, 2, 4]),
                   tensor([0, 1, 0, 3]),
                   values([1, 2, 3, 4]),
                   shape((2, 3)),
                   r'`0 <= plain_indices < plain_dim` is not satisfied.')

            yield ('non-coalesced',
                   tensor([0, 2, 4]),
                   tensor([1, 0, 0, 2]),
                   values([1, 2, 3, 4]),
                   shape((2, 3)),
                   r'`plain_indices\[..., compressed_indices\[..., i - 1\]:compressed_indices\[..., i\]\] '
                   'for all i = 1, ..., compressed_dim '
                   'are sorted and distinct along the last dimension values` is not satisfied.')

        if TEST_CUDA and torch.device(device).type == 'cpu':
            yield ('indices and values mismatch of device',
                   torch.tensor([0, 2, 4]),
                   torch.tensor([0, 1, 0, 1]),
                   values([1, 2, 3, 4], device='cuda'),
                   shape((2, 3)),
                   r'device of compressed_indices \(=cpu\) must match device of values \(=cuda:0\)')
            yield ('compressed_indices and values mismatch of device',
                   torch.tensor([0, 2, 4], device='cuda'),
                   torch.tensor([0, 1, 0, 1]),
                   values([1, 2, 3, 4]),
                   shape((2, 3)),
                   r'Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!')
            yield ('compressed/plain_indices mismatch of device',
                   torch.tensor([0, 2, 4], device='cuda'),
                   torch.tensor([0, 1, 0, 1]),
                   values([1, 2, 3, 4], device='cuda'),
                   shape((2, 3)),
                   r'Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!')

        if TEST_CUDA and torch.device(device).type == 'cuda' and torch.cuda.device_count() >= 2:
            yield ('indices and values mismatch of device index',
                   torch.tensor([0, 2, 4], device='cuda:0'),
                   torch.tensor([0, 1, 0, 1], device='cuda:0'),
                   values([1, 2, 3, 4], device='cuda:1'),
                   shape((2, 3)),
                   r'device of compressed_indices \(=cuda:0\) must match device of values \(=cuda:1\)')
            yield ('compressed_indices and values mismatch of device index',
                   torch.tensor([0, 2, 4], device='cuda:0'),
                   torch.tensor([0, 1, 0, 1], device='cuda:1'),
                   values([1, 2, 3, 4], device='cuda:0'),
                   shape((2, 3)),
                   r'Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1!')

    @skipMeta
    @all_sparse_compressed_layouts()
    @parametrize('target', [subtest('validate_sparse_compressed_tensor_args'),
                            subtest('sparse_compressed_tensor'),
                            subtest('sparse_compressed_tensor_no_size')])
    def test_invalid_input(self, layout, device, target):
        for label, compressed_indices, plain_indices, values, size, errmsg in self._generate_invalid_input(layout, device):
            if layout is torch.sparse_bsr:
                errmsg = errmsg.replace('compressed_indices_name', 'row block').replace('plain_indices_name', 'column block')
            elif layout is torch.sparse_bsc:
                errmsg = errmsg.replace('compressed_indices_name', 'column block').replace('plain_indices_name', 'row block')
            elif layout is torch.sparse_csr:
                errmsg = errmsg.replace('compressed_indices_name', 'row').replace('plain_indices_name', 'column')
            elif layout is torch.sparse_csc:
                errmsg = errmsg.replace('compressed_indices_name', 'column').replace('plain_indices_name', 'row')
            if layout in {torch.sparse_csr, torch.sparse_bsr}:
                errmsg = errmsg.replace('compressed_indices', 'crow_indices') \
                               .replace('plain_indices', 'col_indices') \
                               .replace('plain_dim', 'ncols') \
                               .replace('compressed_dim', 'nrows')
            else:
                errmsg = errmsg.replace('compressed_indices', 'ccol_indices') \
                               .replace('plain_indices', 'row_indices') \
                               .replace('plain_dim', 'nrows') \
                               .replace('compressed_dim', 'ncols')

            if target == 'sparse_compressed_tensor_no_size' and label in {
                    'invalid size', 'invalid batchsize', 'invalid compressed_indices shape', 'invalid max(plain_indices)',
                    'invalid blocksize'}:
                # Skip invalid size input as a valid size is estimated for other inputs
                continue

            with self.assertRaisesRegex(RuntimeError, errmsg):
                if target == 'validate_sparse_compressed_tensor_args':
                    torch._validate_sparse_compressed_tensor_args(compressed_indices, plain_indices, values, size, layout)
                elif target == 'sparse_compressed_tensor':
                    torch.sparse_compressed_tensor(compressed_indices, plain_indices, values, size, layout=layout)
                elif target == 'sparse_compressed_tensor_no_size':
                    torch.sparse_compressed_tensor(compressed_indices, plain_indices, values, layout=layout)
                else:
                    raise NotImplementedError(target)

    @skipMeta
    @onlyCPU
    @largeTensorTest("30GB", "cpu")
    def test_invalid_input_csr_large(self):
        rows = 2 ** 31
        with self.assertRaisesRegex(RuntimeError, '32-bit integer overflow in row dimension'):
            torch.sparse_csr_tensor(torch.arange(rows + 1, dtype=torch.int32) // rows,
                                    torch.tensor([0], dtype=torch.int32),
                                    torch.tensor([1]), (rows, 1))
        torch.sparse_csr_tensor(torch.arange(rows + 1, dtype=torch.int64) // rows,
                                torch.tensor([0], dtype=torch.int64),
                                torch.tensor([1]), (rows, 1))

        cols = 2 ** 31
        with self.assertRaisesRegex(RuntimeError, '32-bit integer overflow in column dimension'):
            torch.sparse_csr_tensor(torch.arange(2, dtype=torch.int32),
                                    torch.tensor([0], dtype=torch.int32),
                                    torch.tensor([1]), (1, cols))
        torch.sparse_csr_tensor(torch.arange(2, dtype=torch.int64),
                                torch.tensor([0], dtype=torch.int64),
                                torch.tensor([1]), (1, cols))

        nnz = 2 ** 31
        with self.assertRaisesRegex(RuntimeError, '32-bit integer overflow in nnz'):
            # nnz cannot be stored in int32 crow_indices
            # but the `crow_indices[..., -1] == nnz`` check happens after the overflow validation
            # So we can use `nnz - 1` here to avoid `value cannot be converted to type int32 without overflow`
            # during construction of crow_indices
            torch.sparse_csr_tensor(torch.tensor([0, nnz // 2, nnz - 1], dtype=torch.int32),
                                    torch.arange(nnz // 2, dtype=torch.int32).repeat(2),
                                    torch.ones(nnz, dtype=torch.int8), (2, nnz // 2))
        torch.sparse_csr_tensor(torch.tensor([0, nnz // 2, nnz], dtype=torch.int64),
                                torch.arange(nnz // 2, dtype=torch.int64).repeat(2),
                                torch.ones(nnz, dtype=torch.int8), (2, nnz // 2))

    @skipMeta
    @onlyCPU
    @all_sparse_compressed_layouts()
    def test_dim(self, layout):
        for (compressed_indices, plain_indices, values), kwargs in self.generate_simple_inputs(layout, output_tensor=False):
            size = kwargs['size']
            batch_dim = compressed_indices.dim() - 1
            sparse_dim = 2
            block_dim = 2 if layout in {torch.sparse_bsr, torch.sparse_bsc} else 0
            dense_dim = values.dim() - batch_dim - block_dim - 1
            sparse = torch.sparse_compressed_tensor(compressed_indices, plain_indices, values, size, layout=layout)
            self.assertEqual(sparse.sparse_dim(), sparse_dim)
            self.assertEqual(sparse.dense_dim(), dense_dim)


    @skipMeta
    @all_sparse_compressed_layouts()
    @dtypes(*all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16))
    def test_to_dtype(self, layout, device, dtype):
        # to_dense does not support hybrid inputs
        for sparse in self.generate_simple_inputs(layout, dtype=dtype, device=device, enable_hybrid=False):
            for to_dtype in all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16):
                sparse_to_dtype = sparse.to(to_dtype)
                dense_to_dtype = sparse.to_dense().to(to_dtype)
                self.assertEqual(sparse_to_dtype.to_dense(), dense_to_dtype)

    @skipMeta
    @all_sparse_compressed_layouts()
    @dtypes(torch.double)
    def test_pickle(self, layout, dtype, device):
        import pickle

        for sparse in self.generate_simple_inputs(layout, device=device, dtype=dtype):
            serialized = pickle.dumps(sparse)
            sparse_loaded = pickle.loads(serialized)

            self.assertEqual(sparse, sparse_loaded)

    @all_sparse_compressed_layouts()
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_select_copy(self, device, dtype, index_dtype, layout):

        def is_view_of(base, other):
            # a shameless copy of TestViewOps.is_view_of
            if (
                not other._is_view() or
                other is base or
                other._base is not base or
                base.device != other.device
            ):
                return False
            if base.device.type in ('cpu', 'cuda'):
                if base.untyped_storage().data_ptr() != other.untyped_storage().data_ptr():
                    return False
            return True

        kwargs = dict(device=devi
```



## High-Level Overview


This Python file contains 6 class(es) and 188 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestSparseCSRSampler`, `TestSparseCompressed`, `TestSparseCSR`, `FakeBscMatrix`, `skipped_cls`, `TestSparseCompressedTritonKernels`

**Functions defined**: `_check_cusparse_spgemm_available`, `_test_addmm_addmv`, `convert_layout`, `test_make_crow_indices`, `all_sparse_compressed_layouts`, `sparse_compressed_nonblock_layouts`, `batched_nonbatched`, `hybrid_nonhybrid`, `genTensor`, `test_layout`, `test_sparse_compressed_constructor`, `test_empty`, `test_empty_errors`, `test_sparse_compressed_tensor_with_dims`, `get_sparse_compressed_tensor_properties`, `test_clone`, `test_print`, `test_copy`, `run_test`, `test_copy_errors`

**Key imports**: torch, random, io, itertools, unittest, functools, redirect_stderr, make_tensor, FileCheck, SM53OrLater, SM80OrLater, TEST_CUSPARSE_GENERIC, TEST_CUDA


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `random`
- `io`
- `itertools`
- `unittest`
- `functools`
- `contextlib`: redirect_stderr
- `torch.testing`: make_tensor, FileCheck
- `torch.testing._internal.common_cuda`: SM53OrLater, SM80OrLater, TEST_CUSPARSE_GENERIC
- `torch.testing._internal.opinfo.definitions.linalg`: sample_inputs_linalg_solve
- `torch.testing._internal.opinfo.definitions.sparse`: validate_sample_input_sparse
- `test_sparse`: CUSPARSE_SPMM_COMPLEX128_SUPPORTED, HIPSPARSE_SPMM_COMPLEX128_SUPPORTED
- `operator`
- `scipy.sparse as sp`
- `numpy as np`
- `from test_linalg instead of code duplication`
- `pickle`
- `re`
- `torch.testing._internal.common_methods_invocations`: sample_inputs_sparse_sampled_addmm
- `warnings`
- `torch.utils._triton`: has_triton
- `torch.sparse._triton_ops`: tile_to_blocksize


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/test_sparse_csr.py
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

- **File Documentation**: `test_sparse_csr.py_docs.md`
- **Keyword Index**: `test_sparse_csr.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
