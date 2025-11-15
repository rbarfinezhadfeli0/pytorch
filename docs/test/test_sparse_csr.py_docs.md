# Documentation: test_sparse_csr.py

## File Metadata
- **Path**: `test/test_sparse_csr.py`
- **Size**: 213333 bytes
- **Lines**: 4295
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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

        kwargs = dict(device=device, dtype=dtype, index_dtype=index_dtype)
        for sparse, dense in zip(self.generate_simple_inputs(layout, **kwargs),
                                 self.generate_simple_inputs(torch.strided, **kwargs)):
            if layout in {torch.sparse_csr, torch.sparse_bsr}:
                n_batchdim = sparse.crow_indices().ndim - 1
            elif layout in {torch.sparse_csc, torch.sparse_bsc}:
                n_batchdim = sparse.ccol_indices().ndim - 1
            else:
                assert 0  # unreachable
            self.assertEqual(sparse, dense)
            for dim in range(sparse.ndim):
                if sparse.shape[dim] == 0:
                    with self.assertRaisesRegex(IndexError, "index 0 out of range for tensor of size"):
                        torch.select_copy(sparse, dim, 0)
                    with self.assertRaisesRegex(IndexError, "index 0 out of range for tensor of size"):
                        torch.select_copy(dense, dim, 0)
                elif n_batchdim and dim >= n_batchdim and dim < n_batchdim + 2:
                    with self.assertRaisesRegex(
                            RuntimeError,
                            "selecting sparse dimensions is not supported for batched sparse compressed tensors"):
                        torch.select_copy(sparse, dim, 0)
                else:
                    for index in {0, sparse.shape[dim] // 2, sparse.shape[dim] - 1}:
                        dense_select = torch.select_copy(dense, dim, index)
                        sparse_select = torch.select_copy(sparse, dim, index)
                        self.assertEqual(sparse_select, dense_select)
                        self.assertFalse(is_view_of(sparse_select.values(), sparse.values()))


def _npref_block_addmm_addmv(c, a, b, alpha, beta):
    return alpha * (a @ b) + beta * c


class TestSparseCSR(TestCase):

    def test_csr_stride(self):
        a = self.genSparseCSRTensor((3, 3), 3, dtype=torch.float, device=self.device_type, index_dtype=torch.int64)

        with self.assertRaisesRegex(RuntimeError, "Sparse CSR tensors do not have strides"):
            a.stride()

        with self.assertRaisesRegex(RuntimeError, "Sparse CSR tensors do not have strides"):
            a.stride(-1)

    def test_csr_storage(self):
        a = self.genSparseCSRTensor((3, 3), 3, dtype=torch.float, device=self.device_type, index_dtype=torch.int64)

        with self.assertRaisesRegex(RuntimeError, "Cannot access storage of SparseCsrTensorImpl"):
            a.storage()

    def test_csr_is_contiguous(self):
        a = self.genSparseCSRTensor((3, 3), 3, dtype=torch.float, device=self.device_type, index_dtype=torch.int64)

        with self.assertRaisesRegex(RuntimeError, "Sparse CSR tensors do not have is_contiguous"):
            a.is_contiguous()

    @onlyCPU
    @largeTensorTest("20GB", "cpu")
    def test_csr_nnz(self):
        # Tests the limits of the number of specified elements in CSR tensors, see gh-102520.
        for nnz in [0, 2**31]:
            rows, cols = 1, max(nnz, 1)
            crow_indices = torch.tensor([0, nnz], dtype=torch.int64)
            col_indices = torch.arange(nnz, dtype=torch.int64)
            values = torch.ones(nnz, dtype=torch.int8)
            a = torch.sparse_csr_tensor(crow_indices, col_indices, values, (rows, cols))
            self.assertEqual(a._nnz(), nnz)

    def test_csr_double_to_sparse_csr(self):
        a = self.genSparseCSRTensor((3, 3), 3, dtype=torch.float, device=self.device_type, index_dtype=torch.int64)
        a.to_sparse_csr().to_sparse_csr()

    @all_sparse_compressed_layouts()
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_select(self, device, dtype, index_dtype, layout):
        compressed_indices_mth = {
            torch.sparse_csr: torch.Tensor.crow_indices,
            torch.sparse_bsr: torch.Tensor.crow_indices,
            torch.sparse_csc: torch.Tensor.ccol_indices,
            torch.sparse_bsc: torch.Tensor.ccol_indices,
        }[layout]

        plain_indices_mth = {
            torch.sparse_csr: torch.Tensor.col_indices,
            torch.sparse_bsr: torch.Tensor.col_indices,
            torch.sparse_csc: torch.Tensor.row_indices,
            torch.sparse_bsc: torch.Tensor.row_indices,
        }[layout]
        create_tensor_mth = {
            torch.sparse_csr: torch.sparse_csr_tensor,
            torch.sparse_bsr: torch.sparse_bsr_tensor,
            torch.sparse_csc: torch.sparse_csc_tensor,
            torch.sparse_bsc: torch.sparse_bsc_tensor,
        }[layout]

        shape = (2, 3, 6, 10)
        nnz = 6
        blocksize = (2, 2) if layout in {torch.sparse_bsr, torch.sparse_bsc} else ()
        sparse = self.genSparseCompressedTensor(
            shape, nnz, device=device, layout=layout, dtype=dtype, index_dtype=index_dtype, blocksize=blocksize)
        comp_indices = compressed_indices_mth(sparse)
        plain_indices = plain_indices_mth(sparse)
        values = sparse.values()

        # select from batch dimensions
        sparse_selected12 = sparse.select(1, 2)
        expected_sparse_selected12 = create_tensor_mth(comp_indices.select(1, 2).contiguous(),
                                                       plain_indices.select(1, 2).contiguous(),
                                                       values.select(1, 2).contiguous(),
                                                       size=(2, 6, 10),
                                                       dtype=dtype,
                                                       device=device)
        self.assertEqual(expected_sparse_selected12, sparse_selected12)

        # selecting rows/col with batch dims not allowed
        sparse_non_batched = sparse[0, 0]
        # select from sparse dimensions
        for select_args in [(0, 0), (1, 1)]:
            sparse_selected = sparse_non_batched.select(*select_args)
            dense_selected = sparse_non_batched.to_dense().select(*select_args)
            self.assertEqual(dense_selected, sparse_selected)

        self.assertEqual(sparse[0, 0, 0, 0], sparse.to_dense()[0, 0, 0, 0])
        # assigning to sparse through indexing is disabled
        with self.assertRaisesRegex(TypeError, "Cannot assign to a sparse tensor"):
            sparse[0, 0, 0, 0] = 99.0

        # select from sparse dimensions without removing batch dims
        msg = "selecting sparse dimensions is not supported for batched sparse compressed tensors."
        with self.assertRaisesRegex(RuntimeError, msg):
            sparse.select(-2, 0)

        with self.assertRaisesRegex(RuntimeError, msg):
            sparse.select(-1, 0)

    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_resize(self, device, dtype):

        def numel(tensor):
            r = 1
            for s in tensor.shape:
                r *= s
            return r

        batch_shapes = [(), (2,), (2, 3)]
        for index_dtype, b in zip([torch.int32, torch.int64], batch_shapes):
            shape = (*b, 2, 3)
            nnz = 6
            a = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=index_dtype)
            self.assertEqual(a.numel(), numel(a))

            new_shape = (*b, 4, 5)
            a.resize_(new_shape)

            self.assertEqual(a.shape, new_shape)
            # resize to larger shape doesn't add specified elements
            self.assertEqual(a._nnz(), nnz)
            self.assertEqual(a.numel(), numel(a))

            new_shape = (*b, 1, 5)
            a.resize_(new_shape)

            self.assertEqual(a.shape, new_shape)
            # resize to smaller shape trims specified elements
            self.assertEqual(a._nnz(), 5)
            self.assertEqual(a.numel(), numel(a))

            # trim batched dimensions
            a.resize_(new_shape[-2], new_shape[-1])
            self.assertEqual(a.shape, (new_shape[-2], new_shape[-1]))
            self.assertEqual(a._nnz(), 5)
            self.assertEqual(a.numel(), numel(a))

    @skipMeta
    @dtypes(torch.float, torch.bool)
    @all_sparse_compressed_layouts()
    def test_resize_as_sparse_compressed(self, device, dtype, layout):

        def _check_resize_b_as_a(b, a):
            br = b.clone()
            br.resize_as_sparse_(a)

            # shape is inherited from a
            self.assertEqual(a.shape, br.shape)
            # other metadata is not affected
            self.assertEqual(b.layout, br.layout)
            self.assertEqual(b.device, br.device)
            self.assertEqual(b.dtype, br.dtype)

            def _get_compressed_plain_inds(t):
                compressed_indices_mth, plain_indices_mth = sparse_compressed_indices_methods[t.layout]
                return compressed_indices_mth(t), plain_indices_mth(t)

            br_compressed_indices, br_plain_indices = _get_compressed_plain_inds(br)
            br_values = br.values()

            b_compressed_indices, b_plain_indices = _get_compressed_plain_inds(b)
            a_compressed_indices, a_plain_indices = _get_compressed_plain_inds(a)
            self.assertEqual(a_plain_indices.shape, br_plain_indices.shape)
            self.assertEqual(a_compressed_indices.shape, br_compressed_indices.shape)
            # We don't check the content of br_plain_indices and br_compressed_indices
            # because it is not well-defined (the content depends on the original
            # shape of `b` that `resize_as` ought to discard) nor needed (the
            # subsequent operation likely updates the indices and values of `b` anyway).
            # the device/dtype of indices should always be unaffected
            self.assertEqual(b_plain_indices.dtype, br_plain_indices.dtype)
            self.assertEqual(b_plain_indices.device, br_plain_indices.device)
            self.assertEqual(b_compressed_indices.dtype, br_compressed_indices.dtype)
            self.assertEqual(b_compressed_indices.device, br_compressed_indices.device)
            # values are generated empty, shape is updated
            self.assertEqual(a.values().shape, br_values.shape)
            # the device/dtype of indices should always be unaffected
            b_values = b.values()
            self.assertEqual(b_values.dtype, br_values.dtype)
            self.assertEqual(b_values.device, br_values.device)
            # nnz will be picked up from a via new shape of values
            self.assertEqual(a._nnz(), br._nnz())

            # post resize the invariants of the layout are respected
            torch._validate_sparse_compressed_tensor_args(br_compressed_indices, br_plain_indices, br_values, br.shape,
                                                          br.layout)

        block_sparse = layout in (torch.sparse_bsr, torch.sparse_bsc)
        shape = (2, 1, 6, 4)
        nnz = 4
        blocksize = (2, 1) if block_sparse else ()
        for index_dtype in [torch.int32, torch.int64]:
            a = self.genSparseCompressedTensor(shape,
                                               layout=layout,
                                               device=device,
                                               index_dtype=index_dtype,
                                               dtype=dtype,
                                               nnz=nnz,
                                               blocksize=blocksize)

            # same size, resize should not trigger
            b = self.genSparseCompressedTensor(shape,
                                               layout=layout,
                                               device=device,
                                               index_dtype=index_dtype,
                                               dtype=dtype,
                                               nnz=nnz,
                                               blocksize=blocksize)

            # This test will not always trigger a resize, if the layouts are the same nothing should happen to b.
            # The invariants of the function as checked should still hold
            _check_resize_b_as_a(b, a)

            # same ndim, but bigger, more nnz, different dtype, different blocksize if blocked
            b = self.genSparseCompressedTensor(tuple(s * 2 for s in shape),
                                               layout=layout,
                                               device=device,
                                               dtype=torch.chalf,
                                               index_dtype=torch.int64 if index_dtype == torch.int32 else torch.int32,
                                               nnz=nnz * 2,
                                               blocksize=tuple(2 * bi for bi in blocksize))
            _check_resize_b_as_a(b, a)

            # different device, only check on cuda pass as we know we are testing in an environment
            # that has multiple devices

            # TODO: .cpu() does not seem to work correctly for sparse. Causes a call to `copy_` which
            # complains about incompatible nnz between src and self?
            if torch.device(device).type == 'cuda' and (layout not in (torch.sparse_bsc, torch.sparse_bsr)):
                a_cpu = self.genSparseCompressedTensor(shape,
                                                       layout=layout,
                                                       device='cpu',
                                                       index_dtype=index_dtype,
                                                       dtype=dtype,
                                                       nnz=nnz,
                                                       blocksize=blocksize)
                _check_resize_b_as_a(b, a)

            # error on a strided
            a_strided = a.to_dense()
            with self.assertRaisesRegex(
                    RuntimeError, r'resize_as_sparse_compressed_: src  expected sparse compressed tensor layout'):
                b.resize_as_sparse_(a_strided)

            # error on b strided
            b_strided = b.to_dense()
            with self.assertRaisesRegex(
                    RuntimeError, r'resize_as_sparse_compressed_: self  expected sparse compressed tensor layout'):
                b_strided.resize_as_sparse_(a)

            # error if layout does not match, transpose induces layout flip
            with self.assertRaisesRegex(RuntimeError,
                                        r"resize_as_sparse_compressed_tensor_: self and src must have the same layout"):
                b.transpose(-2, -1).resize_as_sparse_(a)
            with self.assertRaisesRegex(RuntimeError,
                                        r"resize_as_sparse_compressed_tensor_: self and src must have the same layout"):
                b.resize_as_sparse_(a.transpose(-2, -1))

    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_resize_errors(self, device, dtype):
        for index_dtype in [torch.int32, torch.int64]:
            shape = (2, 3)
            nnz = 6
            a = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=index_dtype)

            with self.assertRaisesRegex(RuntimeError, "torch.resize_: Only batched sparse CSR matrices are supported"):
                new_shape = (4,)
                a.resize_(new_shape)

            # resizing of columns to smaller size is not implemented
            with self.assertRaisesRegex(
                RuntimeError,
                "torch.resize_: Resizing columns of sparse CSR tensors to a smaller value is not supported.",
            ):
                new_shape = (2, 2)
                a.resize_(new_shape)

    @skipIfTorchDynamo()
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_sparse_csr_from_dense(self, device, dtype):
        dense = torch.tensor([[4, 5, 0], [0, 0, 0], [1, 0, 0]], dtype=dtype, device=device)
        sparse = dense.to_sparse_csr()
        self.assertEqual(torch.tensor([0, 2, 2, 3], dtype=torch.int64), sparse.crow_indices())
        self.assertEqual(torch.tensor([0, 1, 0], dtype=torch.int64), sparse.col_indices())
        self.assertEqual(torch.tensor([4, 5, 1], dtype=dtype), sparse.values())

        dense = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=dtype, device=device)
        sparse = dense.to_sparse_csr()
        self.assertEqual(torch.tensor([0, 0, 1, 2], dtype=torch.int64), sparse.crow_indices())
        self.assertEqual(torch.tensor([2, 0], dtype=torch.int64), sparse.col_indices())
        self.assertEqual(torch.tensor([1, 1], dtype=dtype), sparse.values())

        dense = torch.tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]], dtype=dtype, device=device)
        sparse = dense.to_sparse_csr()
        self.assertEqual(torch.tensor([0, 3, 6, 9], dtype=torch.int64), sparse.crow_indices())
        self.assertEqual(torch.tensor([0, 1, 2] * 3, dtype=torch.int64), sparse.col_indices())
        self.assertEqual(torch.tensor([2] * 9, dtype=dtype), sparse.values())

    def _test_sparse_compressed_to_dense(self, device, dtype, layout):
        compressed_format_str = str(layout)[-3:]

        def to_compressed(t):
            return getattr(t, f"to_sparse_{compressed_format_str}")()

        def compressed_constructor(*input, **kwargs):
            constructor = getattr(torch, f"sparse_{compressed_format_str}_tensor")
            return constructor(*input, **kwargs)

        def get_dense_shape(shape, batch_ndim):
            if layout is torch.sparse_csc:
                compressed_dims_slice = slice(batch_ndim + 1, batch_ndim - 1, -1)
            else:
                compressed_dims_slice = slice(batch_ndim, batch_ndim + 2)
            return shape[:batch_ndim] + shape[compressed_dims_slice] + shape[batch_ndim + 2:]

        def transpose(t, batch_ndim):
            if layout is torch.sparse_csc:
                return t.transpose(batch_ndim, batch_ndim + 1)
            return t

        mn = [5, 2, 0]
        for (m, n) in itertools.product(mn, mn):
            size = (m, n)
            dense = make_tensor(size, dtype=dtype, device=device)
            sparse = to_compressed(dense)
            self.assertEqual(sparse.to_dense(), dense)

        batch_shape = (2, 3)
        compressed_indices = torch.tensor([0, 3, 5], device=device).repeat(6, 1).reshape(*batch_shape, -1)
        plain_indices = torch.tensor([0, 1, 2, 0, 1], device=device).repeat(6, 1).reshape(*batch_shape, -1)
        values = torch.tensor([1, 2, 1, 3, 4], device=device, dtype=dtype).repeat(6, 1).reshape(*batch_shape, -1)
        sparse = compressed_constructor(compressed_indices, plain_indices, values, dtype=dtype, device=device)
        dense_shape = get_dense_shape(sparse.shape, len(batch_shape))
        dense = torch.tensor([[1, 2, 1], [3, 4, 0]], dtype=dtype, device=device).repeat(6, 1).reshape(dense_shape)
        self.assertEqual(sparse.to_dense(), transpose(dense, len(batch_shape)))

    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_sparse_csr_to_dense(self, device, dtype):
        self._test_sparse_compressed_to_dense(device, dtype, torch.sparse_csr)

    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_sparse_csc_to_dense(self, device, dtype):
        self._test_sparse_compressed_to_dense(device, dtype, torch.sparse_csc)

    @skipMeta
    @skipCPUIfNoMklSparse
    @coalescedonoff
    @dtypes(torch.double)
    def test_coo_to_csr_convert(self, device, dtype, coalesced):
        with self.assertRaisesRegex(RuntimeError, "Input is supposed to be a vector"):
            torch._convert_indices_from_coo_to_csr(
                torch.randint(100, (5, 5), device=device),
                size=100)

        size = (5, 5)
        sparse_dim = 2
        nnz = 10
        sparse_coo, _, _ = self.genSparseTensor(size, sparse_dim, nnz, coalesced, device, dtype)
        sparse_csr = sparse_coo.to_sparse_csr()

        self.assertTrue(sparse_csr.is_sparse_csr)
        self.assertEqual(sparse_csr.to_dense(), sparse_coo.to_dense())

        vec = torch.randn((5, 1), dtype=dtype, device=device)
        coo_product = sparse_coo.matmul(vec)
        csr_product = sparse_csr.matmul(vec)

        self.assertEqual(coo_product, csr_product)

        vec = torch.randn((100, 1), dtype=dtype, device=device)
        index = torch.tensor([
            [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ], dtype=torch.int32)
        values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dtype, device=device)
        coo = torch.sparse_coo_tensor(index, values, torch.Size([100, 100]), dtype=dtype, device=device)
        csr = coo.to_sparse_csr()

        self.assertEqual(coo.matmul(vec), csr.matmul(vec))

        col_indices = torch.tensor([
            31, 92, 65, 50, 34, 62, 22, 56, 74, 89
        ], dtype=torch.int64, device=device)
        self.assertEqual(csr.col_indices(), col_indices)

        values = torch.tensor([2, 1, 6, 4, 10, 3, 5, 9, 8, 7], dtype=dtype, device=device)
        self.assertEqual(csr.values(), values)

    @parametrize("blocksize", [2, 4])
    @dtypes((torch.double, torch.int32), (torch.double, torch.int64))
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @skipMeta
    def test_csr_to_block_csr(self, device, dtypes, blocksize):
        for shape in [(24, 24), (12, 24)]:
            dtype, index_dtype = dtypes
            m, k = shape
            nnz = random.randint(0, m * k)
            t = self.genSparseCSRTensor((m * blocksize, k * blocksize), nnz, dtype=dtype,
                                        device=device, index_dtype=index_dtype)
            st = sp.csr_matrix((t.values().cpu(), t.col_indices().cpu(), t.crow_indices().cpu()), shape=tuple(t.size()))
            block_t = t.to_sparse_bsr((blocksize, blocksize))
            self.assertEqual(block_t.values().dim(), 3)
            self.assertTrue(block_t.layout == torch.sparse_bsr)
            block_st = st.tobsr(blocksize=(blocksize, blocksize))
            block_st.sort_indices()
            self.assertEqual(block_t.values().cpu(), block_st.data)
            self.assertEqual(block_t.col_indices().cpu(), torch.tensor(block_st.indices).to(index_dtype))
            self.assertEqual(block_t.crow_indices().cpu(), torch.tensor(block_st.indptr).to(index_dtype))

    @dtypes(torch.double)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_csr_to_block_csr_errors(self, device, dtype):
        for index_dtype in [torch.int32, torch.int64]:
            nnz = 15
            t = self.genSparseCSRTensor((16, 16), nnz, dtype=dtype,
                                        device=device, index_dtype=index_dtype)

            with self.assertRaisesRegex(RuntimeError,
                                        r"tensor sparse size \(.*,.*\) must be divisible by given blocksize \(.*,.*\)"):
                block_t = t.to_sparse_bsr((5, 5))

    # TODO: Support auto generation of device check for sparse tensors
    # See: https://github.com/pytorch/pytorch/issues/59058
    @onlyCUDA
    @dtypes(torch.double)
    def test_matmul_device_mismatch(self, device, dtype):
        cpu = torch.rand((10, 10))
        cuda = cpu.cuda()
        for s, m1, m2 in itertools.product((cpu, cuda), repeat=3):
            csr = m1.to_sparse()
            if s.device == csr.device == m2.device:
                torch.addmm(s, csr, m2)
            else:
                with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                    torch.addmm(s, csr, m2)

    @skipCPUIfNoMklSparse
    @skipCUDAIfNoSparseGeneric
    @dtypes(*floating_and_complex_types())
    @dtypesIfCUDA(*floating_and_complex_types_and(
                  *[torch.half] if SM53OrLater else [],
                  *[torch.bfloat16] if SM80OrLater else []))
    def test_csr_matvec(self, device, dtype):

        if TEST_WITH_ROCM and (dtype == torch.half or dtype == torch.bfloat16):
            self.skipTest("ROCm doesn't work with half dtypes correctly.")

        side = 100
        for index_dtype in [torch.int32, torch.int64]:
            csr = self.genSparseCSRTensor((side, side), 1000, device=device, dtype=dtype, index_dtype=index_dtype)
            vec = torch.randn(side, dtype=dtype, device=device)

            res = csr.matmul(vec)
            expected = csr.to_dense().matmul(vec)

            atol, rtol = (2e-3, 1e-3) if dtype == torch.half else (None, None)
            self.assertEqual(res, expected, atol=atol, rtol=rtol)

            bad_vec = torch.randn(side + 10, dtype=dtype, device=device)
            err_msg = "size mismatch, got"
            with self.assertRaisesRegex(RuntimeError, err_msg):
                csr.matmul(bad_vec)

    @onlyCUDA
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_baddbmm(self, device, dtype):

        # TODO: disable the invariant checks within torch.baddbmm that
        # constructs unconventional csr tensors leading to
        # RuntimeError: tensor dimensionality must be sum of batch,
        #     base, and dense dimensionalities (=0 + 2 + 0) but got 3
        # when invariant checking is enabled. When done, undecorate run_test.
        @torch.sparse.check_sparse_tensor_invariants(enable=False)
        def run_test(c, a, a_batched, b, op_b=False, op_out=False, *, dtype=None, device=None):
            alpha = complex(random.random(), random.random()) if dtype.is_complex else random.random()
            beta = complex(random.random(), random.random()) if dtype.is_complex else random.random()
            b = b.mH if (op_b and a.shape == b.shape) else b

            actual = torch.baddbmm(c, a_batched, b, alpha=alpha, beta=beta)

            out = torch.empty_like(c.mH if op_out and a.shape == b.shape else c)
            torch.baddbmm(c, a_batched, b, alpha=alpha, beta=beta, out=out)

            expected = [torch.addmm(c[i], a, b[i], alpha=alpha, beta=beta) for i in range(c.shape[0])]
            expected = torch.stack(expected, 0)

            self.assertEqual(actual, out)
            self.assertEqual(actual, expected)

        for index_dtype in [torch.int32, torch.int64]:
            for (m, n, k), batch_size, noncontiguous in zip(itertools.product([2, 5], repeat=3), [1, 3], [True, False]):
                nnz = random.randint(0, m * k)
                a = self.genSparseCSRTensor((m, k), nnz, dtype=dtype, device=device, index_dtype=index_dtype)

                # a_batched is a regular CSR tensor but with a batch dimension in the shape
                a_batched = torch.sparse_csr_tensor(
                    a.crow_indices(), a.col_indices(), a.values(), (batch_size, m, k), check_invariants=False)

                b = make_tensor((batch_size, k, n), dtype=dtype, device=device, noncontiguous=noncontiguous)
                c = make_tensor((batch_size, m, n), dtype=dtype, device=device, noncontiguous=noncontiguous)
                for op_b, op_out in itertools.product([True, False], repeat=2):
                    run_test(c, a, a_batched, b, op_b, op_out, dtype=dtype, device=device)

    @onlyCUDA
    @skipCUDAIfNoSparseGeneric
    @skipIfRocmVersionLessThan((6, 3))
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_bmm(self, device, dtype):
        def run_test(a, a_batched, b, op_b=False, op_out=False, *, dtype=None, device=None):
            b = b.mH if (op_b and a.shape == b.shape) else b

            actual = torch.bmm(a_batched, b)

            out = torch.empty_like(actual.mH if op_out and a.shape == b.shape else actual)
            torch.bmm(a_batched, b, out=out)

            expected = [torch.mm(a, b[i]) for i in range(b.shape[0])]
            expected = torch.stack(expected, 0)

            self.assertEqual(actual, out)
            self.assertEqual(actual, expected)

        for index_dtype in [torch.int32, torch.int64]:
            for (m, n, k), batch_size, noncontiguous in zip(itertools.product([2, 5], repeat=3), [1, 3], [True, False]):
                nnz = random.randint(0, m * k)
                a = self.genSparseCSRTensor((m, k), nnz, dtype=dtype, device=device, index_dtype=index_dtype)

                # a_batched is a regular CSR tensor but with a batch
                # dimension in the shape. It is unorthodox in PyTorch
                # to represent a batch sparse tensor in this way,
                # hence checking the tensor invariants is locally
                # turned off.
                a_batched = torch.sparse_csr_tensor(
                    a.crow_indices(), a.col_indices(), a.values(), (batch_size, m, k), check_invariants=False)

                b = make_tensor((batch_size, k, n), dtype=dtype, device=device, noncontiguous=noncontiguous)
                for op_b, op_out in itertools.product([True, False], repeat=2):
                    run_test(a, a_batched, b, op_b, op_out, dtype=dtype, device=device)

    def run_test_block_addmm_addmv(self,
                                   addmv_addmm,
                                   c,
                                   a,
                                   b,
                                   op_b=False,
                                   op_out=False,
                                   *,
                                   dtype=None,
                                   device=None,
                                   ref=_npref_block_addmm_addmv):
        alpha = complex(random.random(), random.random()) if dtype.is_complex else random.random()
        beta = complex(random.random(), random.random()) if dtype.is_complex else random.random()
        b = b.mH if (op_b and a.shape == b.shape) else b

        actual = addmv_addmm(c, a, b, alpha=alpha, beta=beta)

        out = torch.empty_like(c.mH if op_out and a.shape == b.shape else c)
        addmv_addmm(c, a, b, alpha=alpha, beta=beta, out=out)
        expected = ref(c, a, b, alpha, beta)

        self.assertEqual(actual, out)
        self.assertEqual(actual, expected, lambda msg: f"{msg}\na={a}\nc={c}\nb={b}\nalpha={alpha} beta={beta}")

    # TODO: block_size 1 is broken
    @parametrize("block_size", [2, 3])
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("noncontiguous", [True, False])
    @skipCPUIfNoMklSparse
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @skipIfTorchDynamo("raises 'sparse matrix length is ambiguous; use getnnz()'")
    @dtypes(*floating_and_complex_types())
    @dtypesIfCUDA(*floating_and_complex_types_and(
                  *[torch.half] if SM53OrLater else [],
                  *[torch.bfloat16] if SM80OrLater else []))
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-5, torch.complex128: 1e-5,
                        torch.float16: 1e-3, torch.bfloat16: 1e-3})
    def test_block_addmm(self, device, dtype, index_dtype, block_size, noncontiguous):

        def make_transposed_addmm_op(f):

            def tt(t):
                if isinstance(t, torch.Tensor):
                    return t.transpose(-2, -1)
                else:
                    # assume numpy/scipy spmatrix
                    return t.transpose()

            @functools.wraps(f)
            def wrapper(c, a, b, alpha=None, beta=None, out=None):
                if out is not None:
                    # the ref takes no out kwarg
                    assert isinstance(out, torch.Tensor)
                    # transpose inplace to propagate out to checking context
                    out.transpose_(-2, -1)
                    return f(tt(c), tt(b), tt(a), alpha=alpha, beta=beta, out=out)
                else:
                    return f(tt(c), tt(b), tt(a), alpha=alpha, beta=beta)

            return wrapper

        def ref_sp_numpy(c, a, b, alpha=None, beta=None, out=None):

            def prep_input(t):

                def to_sp_block_compressed(t):

                    if t.layout is torch.sparse_bsc:
                        tt = t.transpose(-1, -2)
                    else:
                        tt = t

                    t_sp_bsr = sp.bsr_matrix(
                        (
                            tt.values().cpu().numpy(),
                            tt.col_indices().cpu().numpy(),
                            tt.crow_indices().cpu().numpy(),
                        ),
                        shape=tt.shape,
                    )

                    if t.layout is torch.sparse_bsc:
                        return t_sp_bsr.transpose()
                    else:
                        return t_sp_bsr

                if t.layout is not torch.strided:
                    return to_sp_block_compressed(t)
                else:
                    return t.cpu().resolve_conj().numpy()

            res = _npref_block_addmm_addmv(
                *(prep_input(t) for t in (c, a, b)),
                alpha,
                beta
            )

            if out is not None:
                out.copy_(res)
                return out
            else:
                return res

        def ref_half_bfloat16(c, a, b, alpha=None, beta=None, out=None):
            res = alpha * (a.to_dense().to(torch.float32) @ b.to_dense().to(torch.float32)).to(a.dtype) + beta * c
            if out is not None:
                out.copy_(res)
                return out
            else:
                return res

        if dtype in (torch.half, torch.bfloat16):
            ref = ref_half_bfloat16
        else:
            ref = ref_sp_numpy

        for (m, n, k) in itertools.product([2, 5], repeat=3):
            nnz = random.randint(0, m * k)
            a = self.genSparseCSRTensor((m, k), nnz, dtype=dtype, device=device, index_dtype=index_dtype)
            a_data = make_tensor((nnz, block_size, block_size), dtype=dtype, device=device)
            a_data = a_data.mT if noncontiguous else a_data
            a = torch.sparse_bsr_tensor(a.crow_indices(), a.col_indices(),
                                        a_data, (m * block_size, k * block_size), check_invariants=False)
            b = make_tensor((k * block_size, n * block_size), dtype=dtype, device=device, noncontiguous=noncontiguous)
            c = make_tensor((m * block_size, n * block_size), dtype=dtype, device=device, noncontiguous=noncontiguous)
            for op_b, op_out in itertools.product([True, False], repeat=2):
                self.run_test_block_addmm_addmv(torch.addmm, c, a, b, op_b, op_out, dtype=dtype, device=device, ref=ref)
                self.run_test_block_addmm_addmv(make_transposed_addmm_op(torch.addmm),
                                                c,
                                                a,
                                                b,
                                                op_b,
                                                op_out,
                                                dtype=dtype,
                                                device=device,
                                                ref=make_transposed_addmm_op(ref))

    @parametrize("block_size", [2, 3])
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("noncontiguous", [True, False])
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_block_addmv(self, device, dtype, index_dtype, block_size, noncontiguous):
        # TODO: Explicitly disable block size 1 support
        # if (TEST_WITH_ROCM or not TEST_CUSPARSE_GENERIC) and block_size == 1:
        #     return
        def ref_block_addmv(c, a, b, alpha, beta):
            return _npref_block_addmm_addmv(c, a.to_dense(), b, alpha, beta)

        for (m, k) in itertools.product([2, 5], repeat=2):
            nnz = random.randint(0, m * k)
            if not noncontiguous:
                a = self.genSparseCSRTensor((m * block_size, k * block_size), nnz,
                                            dtype=dtype, device=device, index_dtype=index_dtype)
                a = a.to_sparse_bsr((block_size, block_size))
            else:
                a = self.genSparseCSRTensor((m, k), nnz, dtype=dtype, device=device, index_dtype=index_dtype)
                a_data = make_tensor((nnz, block_size, block_size), dtype=dtype, device=device)
                a_data = a_data.mT if noncontiguous else a_data   # Test column-major blocks
                a = torch.sparse_bsr_tensor(a.crow_indices(), a.col_indices(),
                                            a_data, (m * block_size, k * block_size), check_invariants=False)
            b = make_tensor((k * block_size,), dtype=dtype, device=device, noncontiguous=noncontiguous)
            c = make_tensor((m * block_size,), dtype=dtype, device=device, noncontiguous=noncontiguous)
            self.run_test_block_addmm_addmv(torch.addmv, c, a, b, dtype=dtype, device=device, ref=ref_block_addmv)

    @parametrize("matrix_shape", [(3, 3), (5, 7), (11, 9)], name_fn=lambda x: "shape_{}x{}".format(*x))
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @onlyCPU
    def test_addmv(self, device, dtype, matrix_shape):
        mat = torch.randn(matrix_shape, dtype=dtype, device=device)
        mat[mat.real < 0] = 0
        sparse_mat = mat.to_sparse_csr()
        mvec = torch.randn((mat.size(1),), dtype=dtype, device=device)
        avec = torch.randn((mat.size(0),), dtype=torch.float64, device=device)
        ref_output = torch.addmv(avec, mat, mvec)
        output = torch.addmv(avec, sparse_mat, mvec)
        self.assertEqual(ref_output, output)

    @parametrize("block_size", [2, 3])
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("noncontiguous", [True, False])
    @skipCPUIfNoMklSparse
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_block_triangular_solve(self, device, dtype, index_dtype, block_size, noncontiguous):
        def run_test(a, b, upper, transpose, unitriangular, op_out):
            if unitriangular and self.device_type == 'cpu':
                # TODO: When unitriangular=True results are not correct on CPU
                return

            if not upper and self.device_type == 'cpu':
                # TODO: When upper=False some generated inputs might crash on CPU
                return

            actual = torch.triangular_solve(b, a, upper=upper, unitriangular=unitriangular, transpose=transpose)
            actual_X = actual.solution
            actual_A_clone = actual.cloned_coefficient
            self.assertTrue(actual_A_clone.numel() == 0)
            if a._nnz() == 0:
                self.assertTrue(actual_X.isnan().all())
                return

            # TODO: replace with torch method when implemented to_dense() on block sparse tensor
            a_bsr = sp.bsr_matrix(
                (
                    a.values().cpu().numpy(),
                    a.col_indices().cpu().numpy(),
                    a.crow_indices().cpu().numpy(),
                ),
                shape=a.shape,
            )
            expected_X, _ = torch.triangular_solve(
                b,
                torch.tensor(a_bsr.todense(), device=device),
                transpose=transpose,
                upper=upper,
                unitriangular=unitriangular)

            if expected_X.isnan().any():
                # TODO: zeros on the diagonal are not handled for CPU path
                # there's no way to query this info from MKL
                if self.device_type == 'cuda' and not TEST_WITH_ROCM:
                    self.assertTrue(actual_X.isnan().any() or actual_X.isinf().any())
                return

            self.assertEqual(actual_X, expected_X)

            out = torch.empty_like(b.mH if op_out and a.shape == b.shape else b)
            torch.triangular_solve(
                b, a,
                upper=upper, unitriangular=unitriangular, transpose=transpose, out=(out, actual_A_clone)
            )
            self.assertEqual(out, actual_X)
            self.assertEqual(out, expected_X)

        for (m, k) in itertools.product([2, 3], [1, 3]):
            nnz = random.randint(0, m * m)
            if not noncontiguous:
                a = self.genSparseCSRTensor((m * block_size, m * block_size), nnz,
                                            dtype=dtype, device=device, index_dtype=index_dtype)
                a = a.to_sparse_bsr((block_size, block_size))
            else:
                a = self.genSparseCSRTensor((m, m), nnz, dtype=dtype, device=device, index_dtype=index_dtype)
                a_data = make_tensor((nnz, block_size, block_size), dtype=dtype, device=device)
                a_data = a_data.mT if noncontiguous else a_data  # Test column-major blocks
                a = torch.sparse_bsr_tensor(a.crow_indices(), a.col_indices(),
                                            a_data, (m * block_size, m * block_size), check_invariants=False)
            b = make_tensor((m * block_size, k), dtype=dtype, device=device, noncontiguous=noncontiguous)

            for (upper, unitriangular, transpose, op_out) in itertools.product([True, False], repeat=4):
                run_test(a, b, upper, unitriangular, transpose, op_out)

    @skipCPUIfNoMklSparse
    @skipIfRocmVersionLessThan((6, 3))
    @dtypes(torch.double)
    def test_mm(self, device, dtype):
        def test_shape(di, dj, dk, nnz0=None, nnz1=None):
            for index_dtype in [torch.int32, torch.int64]:
                alpha = random.random()
                beta = random.random()

                def _test_addmm(t, x, y):
                    # TODO: addmm doesn't support strided result for sparse inputs.
                    # res = beta * t  + alpha * (x @ y)
                    res = torch.addmm(t, x, y, beta=beta, alpha=alpha)
                    expected = torch.addmm(t, x.to_dense(), y.to_dense(), beta=beta, alpha=alpha)
                    self.assertEqual(res, expected)

                    res = torch.addmm(t, x, y)
                    expected = torch.addmm(t, x.to_dense(), y.to_dense())
                    self.assertEqual(res, expected)

                def _test_mm(x, y):
                    res = torch.mm(x, y)
                    expected = torch.mm(x.to_dense(), y.to_dense())
                    if x.layout is torch.strided or y.layout is torch.strided:
                        self.assertEqual(res.layout, torch.strided)
                    else:
                        self.assertEqual(res.layout, torch.sparse_csr)
                    self.assertEqual(res.to_dense(), expected)

                def _test(t, x, y):
                    _test_addmm(t, x, y)
                    _test_mm(x, y)

                if nnz0 is None:
                    nnz0 = random.randint(di * dk // 2, di * dk)
                t = torch.randn(di, dj, dtype=dtype, device=device)
                x = self.genSparseCSRTensor((di, dk), nnz0, device=device, dtype=dtype, index_dtype=index_dtype)
                y = torch.randn(dk, dj, dtype=dtype, device=device)
                _test(t, x, y)

                t = torch.randn(di, dj, dtype=dtype, device=device)
                x = self.genSparseCSCTensor((di, dk), nnz0, device=device, dtype=dtype, index_dtype=index_dtype)
                y = torch.randn(dk, dj, dtype=dtype, device=device)
                _test(t, x, y)

                if nnz1 is None:
                    nnz1 = random.randint(dk * dj // 2, dk * dj)
                t = torch.randn(di, dj, dtype=dtype, device=device)
                x = torch.randn(di, dk, dtype=dtype, device=device)
                y = self.genSparseCSRTensor((dk, dj), nnz1, device=device, dtype=dtype, index_dtype=index_dtype)
                _test(t, x, y)

                t = torch.randn(di, dj, dtype=dtype, device=device)
                x = torch.randn(di, dk, dtype=dtype, device=device)
                y = self.genSparseCSCTensor((dk, dj), nnz1, device=device, dtype=dtype, index_dtype=index_dtype)
                _test(t, x, y)

                x_shape, y_shape = x.shape, y.shape

                gen_csr_csc = [self.genSparseCSRTensor, self.genSparseCSCTensor]

                # Test mm({CSR, CSC}, {CSR, CSC})
                for gen_x, gen_y in itertools.product(gen_csr_csc, gen_csr_csc):
                    x = gen_x(x_shape, nnz0, device=device, dtype=dtype, index_dtype=index_dtype)
                    y = gen_y(y_shape, nnz1, device=device, dtype=dtype, index_dtype=index_dtype)
                    _test_mm(x, y)

        def test_empty_inputs(lhs_layout, rhs_layout):
            xd = torch.rand(10, 0, device=device, dtype=dtype)
            yd = xd.transpose(-2, -1)
            zd = torch.rand(0, 0, device=device, dtype=dtype)

            xls, yls, zls = (t.to_sparse(layout=lhs_layout) for t in (xd, yd, zd))
            xrs, yrs, zrs = (t.to_sparse(layout=rhs_layout) for t in (xd, yd, zd))

            for ls, rs, ld, rd in [(xls, yrs, xd, yd), (xls, zrs, xd, zd), (zls, yrs, zd, yd), (zls, zrs, zd, zd)]:
                res_sparse = ls @ rs
                res_dense = ld @ rd
                self.assertEqual(res_sparse.to_dense(), res_dense)

        def test_orthogonal_inputs(lhs_layout, rhs_layout):
            ones = torch.ones(2, 2, device=device, dtype=dtype)
            zeros = torch.zeros(2, 2, device=device, dtype=dtype)
            x = torch.cat((ones, zeros), -1).to_sparse(layout=lhs_layout)
            y = torch.cat((zeros, ones), -2).to_sparse(layout=rhs_layout)
            res = x @ y
            res_expected = torch.zeros(*res.shape, device=device, dtype=dtype, layout=res.layout)
            self.assertEqual(res, res_expected)

        for lhs_layout, rhs_layout in itertools.product([torch.sparse_csr, torch.sparse_csc], repeat=2):
            test_empty_inputs(lhs_layout, rhs_layout)
            test_orthogonal_inputs(lhs_layout, rhs_layout)

        for i in [2, 4]:
            for j in [2, 4, 7]:
                for k in [2, 3, 7]:
                    test_shape(i, j, k)
        test_shape(4, 4, 4, 0, 0)

    @skipCPUIfNoMklSparse
    @dtypes(*floating_and_complex_types())
    @dtypesIfCUDA(*floating_and_complex_types_and(
                  *[torch.half] if SM53OrLater and TEST_CUSPARSE_GENERIC else [],
                  *[torch.bfloat16] if SM80OrLater and TEST_CUSPARSE_GENERIC else []))
    @precisionOverride({torch.bfloat16: 1e-2, torch.float16: 1e-2})
    def test_sparse_mm(self, device, dtype):
        def test_shape(d1, d2, d3, nnz, transposed, index_dtype):
            if transposed:
                D = torch.randn(d3, d2, dtype=dtype, device=device).t_()
            else:
                D = torch.randn(d2, d3, dtype=dtype, device=device)
            S = self.genSparseCSRTensor((d1, d2), nnz, device=device, dtype=dtype, index_dtype=index_dtype)
            S_dense = S.to_dense()
            self.assertEqual(torch.sparse.mm(S, D), torch.mm(S_dense, D))

        for index_dtype in [torch.int32, torch.int64]:
            test_shape(7, 8, 9, 20, False, index_dtype)
            test_shape(7, 8, 9, 20, True, index_dtype)

    @dtypes(*floating_and_complex_types())
    @dtypesIfCUDA(*floating_and_complex_types_and(
                  *[torch.half] if SM53OrLater and TEST_CUSPARSE_GENERIC else [],
                  *[torch.bfloat16] if SM80OrLater and TEST_CUSPARSE_GENERIC else []))
    @precisionOverride({torch.bfloat16: 3.5e-2, torch.float16: 1e-2})
    def test_sparse_addmm(self, device, dtype):
        def test_shape(m, n, p, nnz, broadcast, index_dtype, alpha_beta=None):
            if alpha_beta is None:
                alpha = random.random()
                beta = random.random()
            else:
                alpha, beta = alpha_beta
            if broadcast:
                D1 = make_tensor((), dtype=dtype, device=device)
            else:
                D1 = make_tensor([n, p], dtype=dtype, device=device)
            D2 = make_tensor([m, p], dtype=dtype, device=device)
            S = self.genSparseCSRTensor([n, m], nnz, dtype=dtype, device=device, index_dtype=index_dtype)
            S_dense = S.to_dense()
            Y = torch.sparse.addmm(D1, S, D2, beta=beta, alpha=alpha)
            Y_dense = torch.addmm(D1, S_dense, D2, beta=beta, alpha=alpha)
            self.assertEqual(Y, Y_dense)

        for index_dtype in [torch.int32, torch.int64]:
            test_shape(7, 8, 9, 20, False, index_dtype, None)
            test_shape(7, 8, 9, 20, True, index_dtype, None)
            test_shape(7, 8, 9, 20, False, index_dtype, (1, 0))
            test_shape(7, 8, 9, 20, True, index_dtype, (1, 0))
            test_shape(7, 8, 9, 20, False, index_dtype, (1, 1))
            test_shape(7, 8, 9, 20, True, index_dtype, (1, 1))

    @skipCPUIfNoMklSparse
    @skipIfRocmVersionLessThan((6, 3))
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.double: 1e-8, torch.float: 1e-4, torch.bfloat16: 0.6,
                        torch.half: 1e-1, torch.cfloat: 1e-4, torch.cdouble: 1e-8})
    @dtypesIfCUDA(*floating_types_and(torch.complex64,
                                      *[torch.bfloat16] if (SM80OrLater and not TEST_WITH_ROCM) else [],
                                      *[torch.half] if (SM53OrLater and not TEST_WITH_ROCM) else [],
                                      *[torch.complex128]
                                      if CUSPARSE_SPMM_COMPLEX128_SUPPORTED or HIPSPARSE_SPMM_COMPLEX128_SUPPORTED
                                      else []))
    @sparse_compressed_nonblock_layouts()
    def test_addmm_all_sparse_csr(self, device, dtype, layout):
        M = torch.randn(10, 25, device=device).to(dtype)
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 6 class(es): TestSparseCSRSampler, TestSparseCompressed, TestSparseCSR, FakeBscMatrix, skipped_cls, TestSparseCompressedTritonKernels

### Functions
This file defines 188 function(s): _check_cusparse_spgemm_available, _test_addmm_addmv, convert_layout, test_make_crow_indices, all_sparse_compressed_layouts, sparse_compressed_nonblock_layouts, batched_nonbatched, hybrid_nonhybrid, genTensor, test_layout, test_sparse_compressed_constructor, test_empty, test_empty_errors, test_sparse_compressed_tensor_with_dims, get_sparse_compressed_tensor_properties, test_clone, test_print, test_copy, run_test, test_copy_errors, _smallest_divisor, test_consistency, _to_sparse, test_empty_like, test_validate, make_zero_batched, _generate_invalid_input, shape, values, test_invalid_input


## Key Components

The file contains 16857 words across 4295 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 213333 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
