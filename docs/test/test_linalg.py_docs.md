# Documentation: test_linalg.py

## File Metadata
- **Path**: `test/test_linalg.py`
- **Size**: 465650 bytes
- **Lines**: 10078
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
# Owner(s): ["module: linear algebra"]
# ruff: noqa: F841

import torch
import numpy as np

import unittest
import itertools
import warnings
import math
from math import inf, nan, isnan
import re
import random
from random import randrange
from itertools import product
from functools import reduce, partial
from typing import Union, Optional
from torch._prims_common import DimsType
from packaging import version

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, TEST_SCIPY, IS_MACOS, IS_WINDOWS, slowTest,
     TEST_WITH_ROCM, IS_FBCODE, IS_REMOTE_GPU, iter_indices,
     make_fullrank_matrices_with_distinct_singular_values,
     freeze_rng_state, IS_ARM64, IS_SANDCASTLE, TEST_OPT_EINSUM, parametrize, skipIfTorchDynamo,
     skipIfRocmArch, setBlasBackendsToDefaultFinally, setLinalgBackendsToDefaultFinally, serialTest,
     runOnRocmArch, MI300_ARCH, NAVI_ARCH, TEST_CUDA)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, has_cusolver, has_hipsolver,
     onlyCPU, skipCUDAIfNoMagma, skipCPUIfNoLapack, precisionOverride,
     skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, onlyNativeDeviceTypes, dtypesIfCUDA,
     onlyCUDA, skipMeta, skipCUDAIfNoCusolver, skipCUDAIfNotRocm, dtypesIfMPS, largeTensorTest)
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
    all_types, all_types_and_complex_and, floating_and_complex_types, integral_types,
    floating_and_complex_types_and, floating_types_and, complex_types,
)
from torch.testing._internal.common_cuda import SM53OrLater, SM80OrLater, SM90OrLater, tf32_on_and_off, _get_magma_version, \
    _get_torch_cuda_version, CDNA2OrLater, TEST_MULTIGPU
from torch.testing._internal.common_quantization import _group_quantize_tensor, _dynamically_quantize_per_channel, \
    _group_quantize_tensor_symmetric
from torch.testing._internal.common_mkldnn import reduced_f32_on_and_off
from torch.distributions.binomial import Binomial
import torch.backends.opt_einsum as opt_einsum
import operator
import contextlib

# Protects against includes accidentally setting the default dtype
assert torch.get_default_dtype() is torch.float32

if TEST_SCIPY:
    import scipy

def blaslt_supported_device():
    if torch.cuda.is_available():
        if torch.version.hip:
            ROCM_VERSION = tuple(int(v) for v in torch.version.hip.split('.')[:2])
            archs = ['gfx90a', 'gfx94']
            if ROCM_VERSION >= (6, 3):
                archs.extend(['gfx110', 'gfx120'])
            if ROCM_VERSION >= (6, 5):
                archs.append('gfx95')
            for arch in archs:
                if arch in torch.cuda.get_device_properties(0).gcnArchName:
                    return True
        else:
            return True
    return False

def tunableop_matmul(device, dtype, result_filename=None, offline=False):
    # Helper function to test TunableOp in a subprocess
    # requires helper function since lambda function
    # not supported by multiprocessing module
    import os
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"

    if offline:
        torch.cuda.tunable.tuning_enable(False)
        torch.cuda.tunable.record_untuned_enable(True)
    else:
        if result_filename is not None:
            torch.cuda.tunable.set_filename(result_filename)

    torch.cuda.tunable.set_max_tuning_duration(1)
    A = torch.randn((17, 17), device=device, dtype=dtype)
    B = torch.randn((17, 17), device=device, dtype=dtype)
    C = torch.matmul(A, B)
    del os.environ["PYTORCH_TUNABLEOP_ENABLED"]

def get_tunableop_validators():
    assert len(torch.cuda.tunable.get_validators()) > 0
    validators = dict(torch.cuda.tunable.get_validators())
    return validators

def find_tunableop_result(results, OpSig, ParamSig):
    assert isinstance(results, tuple)
    for inner_tuple in results:
        if OpSig in inner_tuple and ParamSig in inner_tuple:
            return inner_tuple
    return None

def get_tunableop_untuned_filename():
    import os
    ordinal = torch.cuda.current_device()
    untuned_filename_env = os.getenv("PYTORCH_TUNABLEOP_UNTUNED_FILENAME")
    untuned_filename_base, _, _ = untuned_filename_env.rpartition('.')
    untuned_filename = f"{untuned_filename_base}{ordinal}.csv"
    return untuned_filename

class TestLinalg(TestCase):
    def setUp(self):
        super().setUp()
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        super().tearDown()

    @contextlib.contextmanager
    def _tunableop_ctx(self):
        # Initialize and then tear down TunableOp
        import glob
        import os
        self._set_tunableop_defaults()
        torch.cuda.tunable.enable(True)

        try:
            yield
        finally:
            # disables TunableOp
            torch.cuda.tunable.enable(False)

            # clean up, remove any files that were generated
            results_filename = torch.cuda.tunable.get_filename()
            results_filename_pattern, _, _ = results_filename.rpartition('.')
            untuned_filename = get_tunableop_untuned_filename()
            untuned_filename_pattern, _, _ = untuned_filename.rpartition('.')
            patterns = [f"{results_filename_pattern[:-1]}*.csv", f"{untuned_filename_pattern[:-1]}*.csv"]
            files = [f for pattern in patterns for f in glob.glob(pattern)]
            for file in files:
                try:
                    os.remove(file)
                # NB: The file is locked on Windows
                except (FileNotFoundError, PermissionError):
                    pass

            # undo all the environment variables set
            # loop through a list of potentially used
            # environment variables.
            env_list = ["PYTORCH_TUNABLEOP_BLAS_LOG",
                        "PYTORCH_TUNABLEOP_UNTUNED_FILENAME"]
            for env in env_list:
                try:
                    del os.environ[env]
                except KeyError:
                    pass

    def _set_tunableop_defaults(self):
        if not torch.cuda.is_available():
            # TunableOp not supported on CPU at this time.
            return

        # disable TunableOp and restore to default values
        torch.cuda.tunable.enable(False)
        torch.cuda.tunable.record_untuned_enable(False)
        torch.cuda.tunable.tuning_enable(True)
        torch.cuda.tunable.set_max_tuning_duration(30)
        torch.cuda.tunable.set_max_tuning_iterations(100)
        torch.cuda.tunable.set_rotating_buffer_size(-1)
        torch.cuda.tunable.set_numerical_check_tolerances(False)
        ordinal = torch.cuda.current_device()

        # Set filenames to be unique on a per test basis
        import os
        unique_id = self.id().split(".")[-1]
        torch.cuda.tunable.set_filename(f"tunableop_results_{unique_id}_{ordinal}.csv")
        # ordinal gets automatically appended
        os.environ["PYTORCH_TUNABLEOP_UNTUNED_FILENAME"] = f"tunableop_untuned_{unique_id}_.csv"

    def _compare_untuned_tuned_entries(self, untuned_filename=None, tuned_filename=None):
        # Compare the entries of untuned and tuned Tunableop results
        # file. Verify that for each Op+Param Signature in the untuned file
        # there is a matching one in the tuned results file.
        import csv
        ok = False
        ordinal = torch.cuda.current_device()
        if untuned_filename is None:
            untuned_filename = get_tunableop_untuned_filename()
        if tuned_filename is None:
            tuned_filename = torch.cuda.tunable.get_filename()

        with open(untuned_filename) as file1:
            with open(tuned_filename) as file2:
                untuned_reader = csv.reader(file1)
                untuned_csv_entries = {(row[0], row[1]) for row in untuned_reader}

                tuned_reader = csv.reader(file2)
                for _ in range(5):  # Skip the first 5 lines for the validator
                    next(tuned_reader, None)

                result_csv_entries = {(row[0], row[1]) for row in tuned_reader}

                missing = untuned_csv_entries - result_csv_entries

                if missing:
                    ok = False
                else:
                    ok = True

        return ok

    exact_dtype = True

    @dtypes(torch.float, torch.cfloat)
    @precisionOverride({torch.float: 1e-06, torch.cfloat: 1e-06})
    @tf32_on_and_off(5e-3)
    @reduced_f32_on_and_off(5e-3)
    def test_inner(self, device, dtype):
        def check(a_sizes_, b_sizes_):
            for a_sizes, b_sizes in ((a_sizes_, b_sizes_), (b_sizes_, a_sizes_)):
                a = torch.randn(a_sizes, dtype=dtype, device=device)
                b = torch.randn(b_sizes, dtype=dtype, device=device)
                res = torch.inner(a, b)
                ref = np.inner(a.cpu().numpy(), b.cpu().numpy())
                self.assertEqual(res.cpu(), torch.from_numpy(np.array(ref)))
                out = torch.zeros_like(res)
                torch.inner(a, b, out=out)
                self.assertEqual(res, out)

        check([], [])                       # scalar x scalar
        check([], [0])                      # scalar x empty
        check([], [3])                      # scalar x 1D
        check([], [2, 3, 4])                # scalar x 3D

        check([0], [0])                     # empty x empty
        check([0], [2, 0])                  # empty x 2D

        check([2], [2])                     # 1D x 1D
        check([2], [3, 1, 2])               # 1D x 3D
        check([2], [3, 0, 2])               # 1D x 3D empty

        check([1, 2], [3, 2])               # 2D x 2D
        check([1, 2], [3, 4, 2])            # 2D x 3D
        check([2, 1, 3, 2], [1, 3, 2, 2])   # 4D x 4D

        # Test error message
        with self.assertRaisesRegex(RuntimeError,
                                    r"inner\(\) the last dimension must match on both "
                                    r"input tensors but got shapes \[2, 3\] and \[2, 2\]"):
            torch.randn(2, 3, device=device, dtype=dtype).inner(torch.randn(2, 2, device=device, dtype=dtype))

    # Tests torch.outer, and its alias, torch.ger, vs. NumPy
    @precisionOverride({torch.bfloat16: 1e-1})
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_outer(self, device, dtype):
        def run_test_case(a, b):
            if dtype == torch.bfloat16:
                a_np = a.to(torch.double).cpu().numpy()
                b_np = b.to(torch.double).cpu().numpy()
                exact_dtype = False
            else:
                a_np = a.cpu().numpy()
                b_np = b.cpu().numpy()
                exact_dtype = True
            expected = np.outer(a_np, b_np)

            self.assertEqual(torch.outer(a, b), expected, exact_dtype=False)
            self.assertEqual(torch.Tensor.outer(a, b), expected, exact_dtype=False)

            self.assertEqual(torch.ger(a, b), expected, exact_dtype=False)
            self.assertEqual(torch.Tensor.ger(a, b), expected, exact_dtype=False)

            # test out variant
            out = torch.empty(a.size(0), b.size(0), device=device, dtype=dtype)
            torch.outer(a, b, out=out)
            self.assertEqual(out, expected, exact_dtype=False)

            out = torch.empty(a.size(0), b.size(0), device=device, dtype=dtype)
            torch.ger(a, b, out=out)
            self.assertEqual(out, expected, exact_dtype=False)

        a = torch.randn(50).to(device=device, dtype=dtype)
        b = torch.randn(50).to(device=device, dtype=dtype)
        run_test_case(a, b)

        # test 0 strided tensor
        zero_strided = torch.randn(1).to(device=device, dtype=dtype).expand(50)
        run_test_case(zero_strided, b)
        run_test_case(a, zero_strided)

    def test_matrix_rank_removed_error(self, device):
        a = make_tensor(5, 5, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            torch.matrix_rank(a)

    def test_solve_removed_error(self, device):
        a = make_tensor(5, 5, device=device, dtype=torch.float32)
        b = make_tensor(5, 1, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            torch.solve(b, a)
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            b.solve(a)

    def test_eig_removed_error(self, device):
        a = make_tensor(5, 5, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            torch.eig(a)
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            a.eig()

    def test_symeig_removed_error(self, device):
        a = make_tensor(5, 5, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            torch.symeig(a)
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            a.symeig()

    def test_lstsq_removed_error(self, device):
        a = make_tensor(5, 5, device=device, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            torch.lstsq(a, a)
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            a.lstsq(a)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @skipIfTorchDynamo("flaky, needs investigation")
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_linalg_lstsq(self, device, dtype):
        from torch.testing._internal.common_utils import random_well_conditioned_matrix
        if self.device_type == 'cpu':
            drivers = ('gels', 'gelsy', 'gelsd', 'gelss', None)
        else:
            drivers = ('gels', None)

        def check_solution_correctness(a, b, sol):
            sol2 = a.pinverse() @ b
            self.assertEqual(sol, sol2, atol=1e-5, rtol=1e-5)

        def check_correctness_ref(a, b, res, ref, driver="default"):
            def apply_if_not_empty(t, f):
                if t.numel():
                    return f(t)
                else:
                    return t

            def select_if_not_empty(t, i):
                selected = apply_if_not_empty(t, lambda x: x.select(0, i))
                return selected

            m = a.size(-2)
            n = a.size(-1)
            nrhs = b.size(-1)
            batch_size = int(np.prod(a.shape[:-2]))
            if batch_size == 0:
                batch_size = 1
            a_3d = a.view(batch_size, m, n)
            b_3d = b.view(batch_size, m, nrhs)

            solution_3d = res.solution.view(batch_size, n, nrhs)
            residuals_2d = apply_if_not_empty(res.residuals, lambda t: t.view(-1, nrhs))
            rank_1d = apply_if_not_empty(res.rank, lambda t: t.view(-1))
            singular_values_2d = res.singular_values.view(batch_size, res.singular_values.shape[-1])

            if a.numel() > 0:
                for i in range(batch_size):
                    sol, residuals, rank, singular_values = ref(
                        a_3d.select(0, i).numpy(),
                        b_3d.select(0, i).numpy()
                    )
                    # Singular values are None when lapack_driver='gelsy' in SciPy
                    if singular_values is None:
                        singular_values = []
                    self.assertEqual(sol, solution_3d.select(0, i), atol=1e-5, rtol=1e-5)
                    self.assertEqual(rank, select_if_not_empty(rank_1d, i), atol=1e-5, rtol=1e-5)
                    self.assertEqual(singular_values, singular_values_2d.select(0, i), atol=1e-5, rtol=1e-5)

                    # SciPy and NumPy operate only on non-batched input and
                    # return an empty array with shape (0,) if rank(a) != n
                    # in PyTorch the batched inputs are supported and
                    # matrices in the batched input can have different ranks
                    # we compute residuals only if all matrices have rank == n
                    # see https://github.com/pytorch/pytorch/issues/56483
                    if m > n:
                        if torch.all(rank_1d == n):
                            self.assertEqual(
                                residuals, select_if_not_empty(residuals_2d, i), atol=1e-5, rtol=1e-5, exact_dtype=False
                            )
                        else:
                            self.assertTrue(residuals_2d.numel() == 0)

            else:
                self.assertEqual(res.solution.shape, (*a.shape[:-2], n, nrhs))
                self.assertEqual(res.rank.shape, a.shape[:-2])

                # residuals are not always computed (and have non-zero shape)
                if m > n and driver != "gelsy":
                    self.assertEqual(res.residuals.shape, (*a.shape[:-2], 0))
                else:
                    self.assertEqual(res.residuals.shape, (0, ))

                # singular_values are not always computed (and have non-zero shape)
                if driver == "default" or driver == "gelsd" or driver == "gelss":
                    self.assertEqual(res.singular_values.shape, (*a.shape[:-2], min(m, n)))
                else:
                    self.assertEqual(res.singular_values.shape, (0, ))

        def check_correctness_scipy(a, b, res, driver, cond):
            # SciPy provides 3 driver options: gelsd, gelss, gelsy
            if TEST_SCIPY and driver in ('gelsd', 'gelss', 'gelsy'):
                import scipy.linalg

                def scipy_ref(a, b):
                    return scipy.linalg.lstsq(a, b, lapack_driver=driver, cond=cond)
                check_correctness_ref(a, b, res, scipy_ref, driver=driver)

        def check_correctness_numpy(a, b, res, driver, rcond):
            # NumPy uses only gelsd routine
            if driver == 'gelsd':

                def numpy_ref(a, b):
                    return np.linalg.lstsq(a, b, rcond=rcond)
                check_correctness_ref(a, b, res, numpy_ref)

        ms = [2 ** i for i in range(5)]
        m_ge_n_sizes = [(m, m // 2) for m in ms] + [(m, m) for m in ms]
        # cases m < n are only supported on CPU and for cuSOLVER path on CUDA
        m_l_n_sizes = [(m // 2, m) for m in ms]
        include_m_l_n_case = (has_cusolver() or device == 'cpu')
        matrix_sizes = m_ge_n_sizes + (m_l_n_sizes if include_m_l_n_case else [])
        batches = [(), (2,), (2, 2), (2, 2, 2)]
        # we generate matrices with singular values sampled from a normal distribution,
        # that is why we use `cond=1.0`, the mean to cut roughly half of all
        # the singular values and compare whether torch.linalg.lstsq agrees with
        # SciPy and NumPy.
        # if rcond is True then set value for it based on the used algorithm
        # rcond == -1 or any other negative value forces LAPACK to use machine precision tolerance
        rconds = (None, True, -1)

        for batch, matrix_size, driver, rcond in itertools.product(batches, matrix_sizes, drivers, rconds):
            # keep the rcond value if it is None or -1, set the driver specific value if it is True
            if rcond and rcond != -1:
                if driver in ('gelss', 'gelsd'):
                    # SVD based algorithm; set to zero roughly half of all the singular values
                    rcond = 1.0
                else:
                    # driver == 'gelsy'
                    # QR based algorithm; setting the value too high might lead to non-unique solutions and flaky tests
                    # so we skip this case
                    continue

            # specifying rcond value has no effect for gels driver so no need to run the tests again
            if driver == 'gels' and rcond is not None:
                continue

            shape = batch + matrix_size
            a = random_well_conditioned_matrix(*shape, dtype=dtype, device=device)
            b = torch.rand(*shape, dtype=dtype, device=device)

            m = a.size(-2)
            n = a.size(-1)
            res = torch.linalg.lstsq(a, b, rcond=rcond, driver=driver)
            sol = res.solution

            # Only checks gelsd, gelss, gelsy drivers
            check_correctness_scipy(a, b, res, driver, rcond)

            # Only checks gelsd driver
            check_correctness_numpy(a, b, res, driver, rcond)

            # gels driver is not checked by comparing to NumPy or SciPy implementation
            # because NumPy and SciPy do not implement this driver
            if driver == 'gels' and rcond is None:
                check_solution_correctness(a, b, sol)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_linalg_lstsq_batch_broadcasting(self, device, dtype):
        from torch.testing._internal.common_utils import random_well_conditioned_matrix

        def check_correctness(a, b):
            sol = torch.linalg.lstsq(a, b).solution
            sol2 = a.pinverse() @ b
            self.assertEqual(sol, sol2, rtol=1e-5, atol=1e-5)

        ms = [2 ** i for i in range(5)]
        batches = [(), (0,), (2,), (2, 2), (2, 2, 2)]
        # the case when a single matrix is batch-broadcasted over the rhs
        for m, batch in itertools.product(ms, batches):
            a = random_well_conditioned_matrix(m, m, dtype=dtype, device=device).view(*([1] * len(batch)), m, m)
            b = torch.rand(*(batch + (m, m)), dtype=dtype, device=device)
            check_correctness(a, b)

        # cases with broadcastable shapes
        for m in ms:
            a = random_well_conditioned_matrix(1, 3, 1, 3, m, m, dtype=dtype, device=device)
            b = torch.rand(3, 1, 3, 1, m, m // 2, dtype=dtype, device=device)
            check_correctness(a, b)

            # rhs are vectors, not matrices in this test
            b = torch.rand(3, 1, 3, 1, m, dtype=dtype, device=device)
            # unsqueeze for b because `check_correctness` checks against
            # a.pinverse() @ b, which requires b to be a matrix
            check_correctness(a, b.unsqueeze(-1))

            a = random_well_conditioned_matrix(3, 1, 3, 1, m, m, dtype=dtype, device=device)
            b = torch.rand(1, 3, 1, 3, m, m // 2, dtype=dtype, device=device)
            check_correctness(a, b)

            # rhs are vectors, not matrices in this test
            b = torch.rand(1, 3, 1, 3, m, dtype=dtype, device=device)
            check_correctness(a, b.unsqueeze(-1))

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_linalg_lstsq_input_checks(self, device, dtype):
        # check empty inputs
        # empty batches
        a = torch.rand(0, 0, 3, 3, dtype=dtype, device=device)
        b = torch.rand(0, 0, 3, 2, dtype=dtype, device=device)
        self.assertEqual(
            torch.linalg.lstsq(a, b)[0],
            torch.zeros(0, 0, 3, 2, dtype=dtype, device=device)
        )
        # empty a and b
        a = torch.rand(2, 2, 0, 0, dtype=dtype, device=device)
        b = torch.rand(2, 2, 0, 0, dtype=dtype, device=device)
        self.assertEqual(
            torch.linalg.lstsq(a, b)[0],
            torch.zeros(2, 2, 0, 0, dtype=dtype, device=device)
        )
        # empty a and b
        a = torch.rand(2, 2, 3, 0, dtype=dtype, device=device)
        b = torch.rand(2, 2, 3, 0, dtype=dtype, device=device)
        self.assertEqual(
            torch.linalg.lstsq(a, b)[0],
            torch.zeros(2, 2, 0, 0, dtype=dtype, device=device)
        )
        # empty a but not b
        a = torch.rand(2, 2, 3, 0, dtype=dtype, device=device)
        b = torch.rand(2, 2, 3, 2, dtype=dtype, device=device)
        self.assertEqual(
            torch.linalg.lstsq(a, b)[0],
            torch.zeros(2, 2, 0, 2, dtype=dtype, device=device)
        )

        # empty a and b
        if torch.device(device).type == 'cpu':
            # only CPU since CUDA does not support overdetermined systems
            a = torch.rand(2, 2, 0, 3, dtype=dtype, device=device)
            b = torch.rand(2, 2, 0, 3, dtype=dtype, device=device)
            self.assertEqual(
                torch.linalg.lstsq(a, b)[0],
                torch.zeros(2, 2, 3, 3, dtype=dtype, device=device)
            )

        a = torch.rand(2, 3, dtype=dtype, device=device)
        b = torch.rand(3, dtype=dtype, device=device)

        with self.assertRaisesRegex(RuntimeError, 'input must have at least 2 dimensions'):
            torch.linalg.lstsq(b, b)

        with self.assertRaisesRegex(RuntimeError, 'other must have at least 1 dimension'):
            torch.linalg.lstsq(a, torch.tensor(1, dtype=dtype, device=device))

        with self.assertRaisesRegex(RuntimeError, r'input.size\(-2\) should match other.size\(-1\)'):
            torch.linalg.lstsq(a, b)

        with self.assertRaisesRegex(RuntimeError, r'input.size\(-2\) should match other.size\(-2\)'):
            torch.linalg.lstsq(a, b.unsqueeze(-1))

        a = torch.randn(1, 1, 1, dtype=dtype, device=device)
        b = torch.randn(3, 1, dtype=dtype, device=device)

        with self.assertRaisesRegex(RuntimeError, r'input.size\(-2\) should match other.size\(-2\)'):
            torch.linalg.lstsq(a, b)

        def complement_device(device):
            if device == 'cpu' and torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'

        a = torch.rand(2, 2, 2, 2, dtype=dtype, device=device)
        b = torch.rand(2, 2, 2, dtype=dtype, device=complement_device(device))
        if a.device != b.device:
            with self.assertRaisesRegex(RuntimeError, 'be on the same device'):
                torch.linalg.lstsq(a, b)

        b = (torch.rand(2, 2, 2, dtype=dtype, device=device) * 100).long()
        with self.assertRaisesRegex(RuntimeError, 'the same dtype'):
            torch.linalg.lstsq(a, b)

        a = torch.rand(2, 2, 2, 2, dtype=dtype, device=device)
        b = torch.rand(2, 2, 2, dtype=dtype, device=device)

        if device != 'cpu':
            with self.assertRaisesRegex(RuntimeError, '`driver` other than `gels` is not supported on CUDA'):
                torch.linalg.lstsq(a, b, driver='fictitious_driver')
        # if on cpu
        else:
            with self.assertRaisesRegex(RuntimeError, r'parameter `driver` should be one of \(gels, gelsy, gelsd, gelss\)'):
                torch.linalg.lstsq(a, b, driver='fictitious_driver')


    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_cholesky(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        def run_test(shape, batch, contiguous):
            A = random_hermitian_pd_matrix(shape, *batch, dtype=dtype, device=device)
            if A.numel() > 0 and not contiguous:
                A = A.mT
                self.assertFalse(A.is_contiguous())
            expected_L = np.linalg.cholesky(A.cpu().numpy())
            actual_L = torch.linalg.cholesky(A)

            # For fp32 individual entries in matrices can differ between PyTorch and NumPy
            # Let's compare the norms of matrices instead
            if A.numel() > 0 and dtype in [torch.float32, torch.complex64]:
                # axis is specified to calculate matrix norm for batched input
                expected_norm = np.linalg.norm(expected_L, ord=1, axis=(-2, -1))
                actual_norm = torch.linalg.norm(actual_L, ord=1, axis=(-2, -1))
                # Compare the norms with standard tolerances
                self.assertEqual(actual_norm, expected_norm)
                # and individual values with a higher tolerance
                self.assertEqual(actual_L, expected_L, atol=1e-2, rtol=1e-5)
            else:
                self.assertEqual(actual_L, expected_L)

        shapes = (0, 3, 5)
        batches = ((), (3, ), (2, 2))
        larger_input_case = [(100, (5, ), True)]
        for shape, batch, contiguous in list(itertools.product(shapes, batches, (True, False))) + larger_input_case:
            run_test(shape, batch, contiguous)

        # check the out= variant
        A = random_hermitian_pd_matrix(3, 3, dtype=dtype, device=device)
        out = torch.empty_like(A)
        ans = torch.linalg.cholesky(A, out=out)
        self.assertEqual(ans, out)
        expected = torch.linalg.cholesky(A)
        self.assertEqual(expected, out)

        # check the upper= variant
        expected = torch.linalg.cholesky(A).mH
        actual = torch.linalg.cholesky(A, upper=True)
        self.assertEqual(expected, actual)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_cholesky_errors_and_warnings(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        # cholesky requires the input to be a square matrix or batch of square matrices
        A = torch.randn(2, 3, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, r'must be batches of square matrices'):
            torch.linalg.cholesky(A)
        A = torch.randn(2, 2, 3, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, r'must be batches of square matrices'):
            torch.linalg.cholesky(A)
        with self.assertRaisesRegex(np.linalg.LinAlgError, r'Last 2 dimensions of the array must be square'):
            np.linalg.cholesky(A.cpu().numpy())

        # cholesky requires the input to be at least 2 dimensional tensor
        A = torch.randn(2, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, r'must have at least 2 dimensions'):
            torch.linalg.cholesky(A)
        with self.assertRaisesRegex(np.linalg.LinAlgError,
                                    r'1-dimensional array given\. Array must be at least two-dimensional'):
            np.linalg.cholesky(A.cpu().numpy())

        # if the input matrix is not positive definite, an error should be raised
        A = torch.eye(3, 3, dtype=dtype, device=device)
        A[-1, -1] = 0  # Now A is not positive definite
        with self.assertRaisesRegex(torch.linalg.LinAlgError, r'minor of order 3 is not positive-definite'):
            torch.linalg.cholesky(A)
        with self.assertRaisesRegex(np.linalg.LinAlgError, r'Matrix is not positive definite'):
            np.linalg.cholesky(A.cpu().numpy())

        # if at least one matrix in the batch is singular, an error should be raised
        A = torch.eye(3, 3, dtype=dtype, device=device)
        A = A.reshape((1, 3, 3))
        A = A.repeat(5, 1, 1)
        A[4, -1, -1] = 0  # Now A[4] is not positive definite
        with self.assertRaisesRegex(torch.linalg.LinAlgError, r'\(Batch element 4\): The factorization could not be completed'):
            torch.linalg.cholesky(A)

        # if out tensor with wrong shape is passed a warning is given
        A = random_hermitian_pd_matrix(3, dtype=dtype, device=device)
        out = torch.empty(2, 3, dtype=dtype, device=device)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.cholesky(A, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtypes should be safely castable
        out = torch.empty(*A.shape, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got int instead"):
            torch.linalg.cholesky(A, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty(0, device=wrong_device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                torch.linalg.cholesky(A, out=out)

    # NOTE: old_cholesky* tests were moved here from test_torch.py and test_autograd.py
    @slowTest
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double)
    def test_old_cholesky_batched_many_batches(self, device, dtype):
        from torch.testing._internal.common_utils import random_symmetric_pd_matrix

        def cholesky_test_helper(n, batchsize, device, upper):
            A = random_symmetric_pd_matrix(n, batchsize, dtype=dtype, device=device)
            chol_fact = torch.cholesky(A, upper=upper)
            if upper:
                # Correctness check
                self.assertEqual(A, chol_fact.mT.matmul(chol_fact))
                # Upper triangular check
                self.assertEqual(chol_fact, chol_fact.triu())
            else:
                # Correctness check
                self.assertEqual(A, chol_fact.matmul(chol_fact.mT))
                # Lower triangular check
                self.assertEqual(chol_fact, chol_fact.tril())

        for upper, batchsize in itertools.product([True, False], [262144, 524288]):
            cholesky_test_helper(2, batchsize, device, upper)

    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_old_cholesky_batched(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        def cholesky_test_helper(n, batch_dims, upper):
            A = random_hermitian_pd_matrix(n, *batch_dims, dtype=dtype, device=device)
            cholesky_exp = torch.stack([m.cholesky(upper=upper) for m in A.reshape(-1, n, n)])
            cholesky_exp = cholesky_exp.reshape_as(A)
            self.assertEqual(cholesky_exp, torch.cholesky(A, upper=upper))

        for upper, batchsize in itertools.product([True, False], [(3,), (3, 4), (2, 3, 4)]):
            cholesky_test_helper(3, batchsize, upper)

    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    @skipIfRocmArch(MI300_ARCH)
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @tf32_on_and_off(0.01)
    @reduced_f32_on_and_off(0.01)
    def test_old_cholesky(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        A = random_hermitian_pd_matrix(10, dtype=dtype, device=device)

        # default Case
        C = torch.cholesky(A)
        B = torch.mm(C, C.t().conj())
        self.assertEqual(A, B, atol=1e-14, rtol=0)

        # test Upper Triangular
        U = torch.cholesky(A, True)
        B = torch.mm(U.t().conj(), U)
        self.assertEqual(A, B, atol=1e-14, rtol=0, msg='cholesky (upper) did not allow rebuilding the original matrix')

        # test Lower Triangular
        L = torch.cholesky(A, False)
        B = torch.mm(L, L.t().conj())
        self.assertEqual(A, B, atol=1e-14, rtol=0, msg='cholesky (lower) did not allow rebuilding the original matrix')

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_old_cholesky_empty(self, device, dtype):
        def run_test(upper):
            A = torch.empty(0, 0, dtype=dtype, device=device)
            chol = torch.cholesky(A, upper)
            chol_A = torch.matmul(chol, chol.t().conj())
            self.assertEqual(A, chol_A)
        for upper in [True, False]:
            run_test(upper)

    # Test for issue
    # https://github.com/pytorch/pytorch/issues/57032
    # torch.cholesky with upper=True for batched CUDA inputs was wrong
    # it was using the lower triangular part instead of the upper one
    @onlyCUDA
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    def test_old_cholesky_batched_upper(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        batchsize = 2
        A = random_hermitian_pd_matrix(3, batchsize, dtype=dtype, device=device)
        A_triu = A.triu()  # fill the lower triangular part with zero

        U = torch.cholesky(A_triu, upper=True)

        reconstruct_A = U.mH @ U
        self.assertEqual(A, reconstruct_A)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_cholesky_ex(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        def run_test(n, batch):
            A = random_hermitian_pd_matrix(n, *batch, dtype=dtype, device=device)
            expected_L = np.linalg.cholesky(A.cpu().numpy())
            expected_info = torch.zeros(A.shape[:-2], dtype=torch.int32, device=device)
            actual_L, actual_info = torch.linalg.cholesky_ex(A)

            # For fp32 individual entries in matrices can differ between PyTorch and NumPy
            # Let's compare the norms of matrices instead
            if A.numel() > 0 and dtype in [torch.float32, torch.complex64]:
                # axis is specified to calculate matrix norm for batched input
                expected_norm = np.linalg.norm(expected_L, ord=1, axis=(-2, -1))
                actual_norm = torch.linalg.norm(actual_L, ord=1, axis=(-2, -1))
                # Compare the norms with standard tolerances
                self.assertEqual(actual_norm, expected_norm)
                # and individual values with a higher tolerance
                self.assertEqual(actual_L, expected_L, atol=1e-2, rtol=1e-5)
            else:
                self.assertEqual(actual_L, expected_L)
            self.assertEqual(actual_info, expected_info)

        ns = (0, 3, 5)
        batches = ((), (2, ), (2, 1))
        for n, batch in itertools.product(ns, batches):
            run_test(n, batch)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_cholesky_ex_non_pd(self, device, dtype):
        # if the input matrix is not positive definite, info with positive integer is returned
        A = torch.eye(3, 3, dtype=dtype, device=device)
        A[-1, -1] = 0  # Now A is singular
        _, info = torch.linalg.cholesky_ex(A)
        self.assertEqual(info, 3)
        with self.assertRaisesRegex(torch.linalg.LinAlgError, r'minor of order 3 is not positive-definite'):
            torch.linalg.cholesky_ex(A, check_errors=True)

        # if at least one matrix in the batch is not positive definite,
        # batched info with positive integer for the corresponding matrix is returned
        A = torch.eye(3, 3, dtype=dtype, device=device)
        A = A.reshape((1, 3, 3))
        A = A.repeat(5, 1, 1)
        A[3, -2, -2] = 0  # Now A[3] is singular
        _, info = torch.linalg.cholesky_ex(A)

        expected_info = torch.zeros(A.shape[:-2], dtype=torch.int32, device=device)
        expected_info[3] = 2
        self.assertEqual(info, expected_info)
        with self.assertRaisesRegex(torch.linalg.LinAlgError, r'\(Batch element 3\): The factorization could not be completed'):
            torch.linalg.cholesky_ex(A, check_errors=True)

    def _test_addr_vs_numpy(self, device, dtype, beta=1, alpha=1):
        def check(m, a, b, beta, alpha):
            if dtype == torch.bfloat16:
                a_np = a.to(torch.double).cpu().numpy()
                b_np = b.to(torch.double).cpu().numpy()
                m_np = m.to(torch.double).cpu().numpy()
                exact_dtype = False
            else:
                a_np = a.cpu().numpy()
                b_np = b.cpu().numpy()
                m_np = m.cpu().numpy()
                exact_dtype = True
            if beta == 0:
                expected = alpha * np.outer(a_np, b_np)
            else:
                expected = beta * m_np + alpha * np.outer(a_np, b_np)

            res = torch.addr(m, a, b, beta=beta, alpha=alpha)
            self.assertEqual(res, expected, exact_dtype=exact_dtype)

            # Test out variant
            out = torch.empty_like(res)
            torch.addr(m, a, b, beta=beta, alpha=alpha, out=out)
            self.assertEqual(out, expected, exact_dtype=exact_dtype)

        m = make_tensor((50, 50), device=device, dtype=dtype, low=-2, high=2)
        a = make_tensor((50,), device=device, dtype=dtype, low=-2, high=2)
        b = make_tensor((50,), device=device, dtype=dtype, low=-2, high=2)

        check(m, a, b, beta, alpha)

        # test transpose
        m_transpose = torch.transpose(m, 0, 1)
        check(m_transpose, a, b, beta, alpha)

        # test 0 strided tensor
        zero_strided = make_tensor((1,), device=device, dtype=dtype, low=-2, high=2).expand(50)
        check(m, zero_strided, b, beta, alpha)

        # test scalar
        m_scalar = torch.tensor(1, device=device, dtype=dtype)
        check(m_scalar, a, b, beta, alpha)

        # test nans and infs are not propagated to the output when beta == 0
        float_and_complex_dtypes = floating_and_complex_types_and(torch.half, torch.bfloat16)
        if beta == 0 and dtype in float_and_complex_dtypes:
            m[0][10] = m[10][10] = m[20][20] = float('inf')
            m[1][10] = m[11][10] = m[21][20] = float('nan')
        check(m, a, b, 0, alpha)

    @dtypes(torch.bool)
    def test_addr_bool(self, device, dtype):
        self._test_addr_vs_numpy(device, dtype, beta=True, alpha=False)
        self._test_addr_vs_numpy(device, dtype, beta=False, alpha=True)
        self._test_addr_vs_numpy(device, dtype, beta=False, alpha=False)
        self._test_addr_vs_numpy(device, dtype, beta=True, alpha=True)

    @dtypes(*integral_types())
    def test_addr_integral(self, device, dtype):
        with self.assertRaisesRegex(RuntimeError,
                                    'argument beta must not be a floating point number.'):
            self._test_addr_vs_numpy(device, dtype, beta=2., alpha=1)
        with self.assertRaisesRegex(RuntimeError,
                                    'argument alpha must not be a floating point number.'):
            self._test_addr_vs_numpy(device, dtype, beta=2, alpha=1.)
        with self.assertRaisesRegex(RuntimeError,
                                    'Boolean beta only supported for Boolean results.'):
            self._test_addr_vs_numpy(device, dtype, beta=True, alpha=1)
        with self.assertRaisesRegex(RuntimeError,
                                    'Boolean alpha only supported for Boolean results.'):
            self._test_addr_vs_numpy(device, dtype, beta=2, alpha=True)

        # when beta is zero
        self._test_addr_vs_numpy(device, dtype, beta=0, alpha=2)
        # when beta is not zero
        self._test_addr_vs_numpy(device, dtype, beta=2, alpha=2)

    @precisionOverride({torch.bfloat16: 1e-1})
    @dtypes(*floating_and_complex_types_and(torch.half, torch.bfloat16))
    def test_addr_float_and_complex(self, device, dtype):
        with self.assertRaisesRegex(RuntimeError,
                                    'Boolean beta only supported for Boolean results.'):
            self._test_addr_vs_numpy(device, dtype, beta=True, alpha=1)
        with self.assertRaisesRegex(RuntimeError,
                                    'Boolean alpha only supported for Boolean results.'):
            self._test_addr_vs_numpy(device, dtype, beta=2, alpha=True)

        # when beta is zero
        self._test_addr_vs_numpy(device, dtype, beta=0., alpha=2)
        # when beta is not zero
        self._test_addr_vs_numpy(device, dtype, beta=0.5, alpha=2)
        if dtype in complex_types():
            self._test_addr_vs_numpy(device, dtype, beta=(0 + 0.1j), alpha=(0.2 - 0.2j))

    @dtypes(*itertools.product(all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
                               all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool)))
    def test_outer_type_promotion(self, device, dtypes):
        a = torch.randn(5).to(device=device, dtype=dtypes[0])
        b = torch.randn(5).to(device=device, dtype=dtypes[1])
        for op in (torch.outer, torch.Tensor.outer, torch.ger, torch.Tensor.ger):
            result = op(a, b)
            self.assertEqual(result.dtype, torch.result_type(a, b))

    # don't use @dtypes decorator to avoid generating ~1700 tests per device
    def test_addr_type_promotion(self, device):
        for dtypes0, dtypes1, dtypes2 in product(all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool), repeat=3):
            a = make_tensor((5,), device=device, dtype=dtypes0, low=-2, high=2)
            b = make_tensor((5,), device=device, dtype=dtypes1, low=-2, high=2)
            m = make_tensor((5, 5), device=device, dtype=dtypes2, low=-2, high=2)

            desired_dtype = torch.promote_types(torch.promote_types(dtypes0, dtypes1),
                                                dtypes2)
            for op in (torch.addr, torch.Tensor.addr):
                result = op(m, a, b)
                self.assertEqual(result.dtype, desired_dtype)

    # Tests migrated from test_torch.py
    # 1) test the shape of the result tensor when there is empty input tensor
    # 2) test the Runtime Exception when there is scalar input tensor
    def test_outer_ger_addr_legacy_tests(self, device):
        for size in ((0, 0), (0, 5), (5, 0)):
            a = torch.rand(size[0], device=device)
            b = torch.rand(size[1], device=device)

            self.assertEqual(torch.outer(a, b).shape, size)
            self.assertEqual(torch.ger(a, b).shape, size)

            m = torch.empty(size, device=device)
            self.assertEqual(torch.addr(m, a, b).shape, size)

        m = torch.randn(5, 6, device=device)
        a = torch.randn(5, device=device)
        b = torch.tensor(6, device=device)
        self.assertRaises(RuntimeError, lambda: torch.outer(a, b))
        self.assertRaises(RuntimeError, lambda: torch.outer(b, a))
        self.assertRaises(RuntimeError, lambda: torch.ger(a, b))
        self.assertRaises(RuntimeError, lambda: torch.ger(b, a))
        self.assertRaises(RuntimeError, lambda: torch.addr(m, a, b))
        self.assertRaises(RuntimeError, lambda: torch.addr(m, b, a))

    # Tests torch.det and its alias, torch.linalg.det, vs. NumPy
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double, torch.cdouble)
    def test_det(self, device, dtype):
        tensors = (
            torch.randn((2, 2), device=device, dtype=dtype),
            torch.randn((129, 129), device=device, dtype=dtype),
            torch.randn((3, 52, 52), device=device, dtype=dtype),
            torch.randn((4, 2, 26, 26), device=device, dtype=dtype))

        ops = (torch.det, torch.Tensor.det,
               torch.linalg.det)
        for t in tensors:
            expected = np.linalg.det(t.cpu().numpy())
            for op in ops:
                actual = op(t)
                self.assertEqual(actual, expected)
                self.compare_with_numpy(op, np.linalg.det, t)

        # NOTE: det requires a 2D+ tensor
        t = torch.randn(1, device=device, dtype=dtype)
        with self.assertRaises(RuntimeError):
            op(t)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    def test_eigh(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_matrix

        def run_test(shape, batch, uplo):
            matrix = random_hermitian_matrix(shape, *batch, dtype=dtype, device=device)
            expected_w, expected_v = np.linalg.eigh(matrix.cpu().numpy(), UPLO=uplo)
            actual_w, actual_v = torch.linalg.eigh(matrix, UPLO=uplo)
            self.assertEqual(actual_w, expected_w)
            # sign of eigenvectors is not unique and therefore absolute values are compared
            self.assertEqual(abs(actual_v), abs(expected_v))
            # additionally we can multiply the eigenvector with a phase factor e^{i\phi} and then compare the values
            # let's choose the convention that the first element of the eigenvectors from torch and numpy be the same
            # for real inputs, this phase factor is plus or minus one
            if matrix.numel() > 0:
                phase = torch.from_numpy(expected_v[..., 0, :]).to(device=device).div(actual_v[..., 0, :])
                actual_v_rotated = actual_v * phase.unsqueeze(-2).expand_as(actual_v)
                self.assertEqual(actual_v_rotated, expected_v)

            # check the out= variant
            out_w = torch.empty_like(actual_w)
            out_v = torch.empty_like(actual_v)
            ans_w, ans_v = torch.linalg.eigh(matrix, UPLO=uplo, out=(out_w, out_v))
            self.assertEqual(ans_w, out_w)
            self.assertEqual(ans_v, out_v)
            self.assertEqual(ans_w, actual_w)
            self.assertEqual(abs(ans_v), abs(actual_v))

        shapes = (0, 3, 5)
        batches = ((), (3, ), (2, 2))
        uplos = ["U", "L"]
        for shape, batch, uplo in itertools.product(shapes, batches, uplos):
            run_test(shape, batch, uplo)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    def test_eigh_lower_uplo(self, device, dtype):
        def run_test(shape, batch, uplo):
            # check lower case uplo
            # use non-symmetric input to check whether uplo argument is working as intended
            matrix = torch.randn(shape, shape, *batch, dtype=dtype, device=device)
            expected_w, expected_v = np.linalg.eigh(matrix.cpu().numpy(), UPLO=uplo)
            actual_w, actual_v = torch.linalg.eigh(matrix, UPLO=uplo)
            self.assertEqual(actual_w, expected_w)
            self.assertEqual(abs(actual_v), abs(expected_v))

        uplos = ["u", "l"]
        for uplo in uplos:
            run_test(3, (2, 2), uplo)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_eigh_errors_and_warnings(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_matrix

        # eigh requires a square matrix
        t = torch.randn(2, 3, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.linalg.eigh(t)

        # eigh requires 'uplo' parameter to be 'U' or 'L'
        t = torch.randn(3, 3, device=device, dtype=dtype)
        for uplo in ["a", "wrong"]:
            with self.assertRaisesRegex(RuntimeError, "be 'L' or 'U'"):
                torch.linalg.eigh(t, UPLO=uplo)
            with self.assertRaisesRegex(ValueError, "be 'L' or 'U'"):
                np.linalg.eigh(t.cpu().numpy(), UPLO=uplo)

        # if non-empty out tensor with wrong shape is passed a warning is given
        a = random_hermitian_matrix(3, dtype=dtype, device=device)
        real_dtype = a.real.dtype if dtype.is_complex else dtype
        out_w = torch.empty(7, 7, dtype=real_dtype, device=device)
        out_v = torch.empty(7, 7, dtype=dtype, device=device)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.eigh(a, out=(out_w, out_v))
            # Check warning occurs
            self.assertEqual(len(w), 2)
            self.assertTrue("An output with one or more elements was resized" in str(w[-2].message))
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtypes should be safely castable
        out_w = torch.empty(0, dtype=real_dtype, device=device)
        out_v = torch.empty(0, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got int instead"):
            torch.linalg.eigh(a, out=(out_w, out_v))

        out_w = torch.empty(0, dtype=torch.int, device=device)
        out_v = torch.empty(0, dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got int instead"):
            torch.linalg.eigh(a, out=(out_w, out_v))

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out_w = torch.empty(0, device=wrong_device, dtype=dtype)
            out_v = torch.empty(0, device=device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.eigh(a, out=(out_w, out_v))
            out_w = torch.empty(0, device=device, dtype=dtype)
            out_v = torch.empty(0, device=wrong_device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.eigh(a, out=(out_w, out_v))

    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double)
    @unittest.skipIf((not TEST_WITH_ROCM) and _get_torch_cuda_version() < (12, 1), "Test is fixed on cuda 12.1 update 1.")
    def test_eigh_svd_illcondition_matrix_input_should_not_crash(self, device, dtype):
        # See https://github.com/pytorch/pytorch/issues/94772, https://github.com/pytorch/pytorch/issues/105359
        # This test crashes with `cusolver error: CUSOLVER_STATUS_EXECUTION_FAILED` on cuda 11.8,
        # but passes on cuda 12.1 update 1 or later.
        a = torch.ones(512, 512, dtype=dtype, device=device)
        a[0, 0] = 1.0e-5
        a[-1, -1] = 1.0e5

        eigh_out = torch.linalg.eigh(a)
        svd_out = torch.linalg.svd(a)

        # Matrix input a is too ill-conditioned.
        # We'll just compare the first two singular values/eigenvalues. They are 1.0e5 and 511.0
        # The precision override with tolerance of 1.0 makes sense since ill-conditioned inputs are hard to converge
        # to exact values.
        self.assertEqual(eigh_out.eigenvalues.sort(descending=True).values[:2], [1.0e5, 511.0], atol=1.0, rtol=1.0e-2)
        self.assertEqual(svd_out.S[:2], [1.0e5, 511.0], atol=1.0, rtol=1.0e-2)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    def test_eigvalsh(self, device, dtype):
        from torch.testing._internal.common_utils import random_hermitian_matrix

        def run_test(shape, batch, uplo):
            matrix = random_hermitian_matrix(shape, *batch, dtype=dtype, device=device)
            expected_w = np.linalg.eigvalsh(matrix.cpu().numpy(), UPLO=uplo)
            actual_w = torch.linalg.eigvalsh(matrix, UPLO=uplo)
            self.assertEqual(actual_w, expected_w)

            # check the out= variant
            out = torch.empty_like(actual_w)
            ans = torch.linalg.eigvalsh(matrix, UPLO=uplo, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, actual_w)

        shapes = (0, 3, 5)
        batches = ((), (3, ), (2, 2))
        uplos = ["U", "L"]
        for shape, batch, uplo in itertools.product(shapes, batches, uplos):
            run_test(shape, batch, uplo)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_eigvalsh_errors_and_warnings(self, device, dtype):
        # eigvalsh requires a square matrix
        t = torch.randn(2, 3, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.linalg.eigvalsh(t)

        # eigvalsh requires 'uplo' parameter to be 'U' or 'L'
        t = torch.randn(3, 3, device=device, dtype=dtype)
        for uplo in ["a", "wrong"]:
            with self.assertRaisesRegex(RuntimeError, "be 'L' or 'U'"):
                torch.linalg.eigvalsh(t, UPLO=uplo)
            with self.assertRaisesRegex(ValueError, "be 'L' or 'U'"):
                np.linalg.eigvalsh(t.cpu().numpy(), UPLO=uplo)

        # if non-empty out tensor with wrong shape is passed a warning is given
        real_dtype = t.real.dtype if dtype.is_complex else dtype
        out = torch.empty_like(t).to(real_dtype)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.linalg.eigvalsh(t, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtypes should be safely castable
        out = torch.empty(0, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got int instead"):
            torch.linalg.eigvalsh(t, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty(0, device=wrong_device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.eigvalsh(t, out=out)

    @onlyCPU
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_eigh_lwork_lapack(self, device, dtype):
        # test that the calculated lwork does not cause a crash, see https://github.com/pytorch/pytorch/issues/145801
        t = torch.rand(3000, 3000, device=device, dtype=dtype)
        y = torch.linalg.eigh(t)
        self.assertEqual(y.eigenvalues.shape, (3000,))

    @dtypes(*floating_and_complex_types())
    def test_kron(self, device, dtype):

        def run_test_case(a_shape, b_shape):
            a = torch.rand(a_shape, dtype=dtype, device=device)
            b = torch.rand(b_shape, dtype=dtype, device=device)

            expected = np.kron(a.cpu().numpy(), b.cpu().numpy())
            result = torch.kron(a, b)
            self.assertEqual(result, expected)

            # check the out= variant
            out = torch.empty_like(result)
            ans = torch.kron(a, b, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, result)

        shapes = [(4,), (2, 2), (1, 2, 3), (1, 2, 3, 3)]
        for a_shape, b_shape in itertools.product(shapes, reversed(shapes)):
            run_test_case(a_shape, b_shape)

    @dtypes(*floating_and_complex_types())
    def test_kron_empty(self, device, dtype):

        def run_test_case(empty_shape):
            a = torch.eye(3, dtype=dtype, device=device)
            b = torch.empty(empty_shape, dtype=dtype, device=device)
            result = torch.kron(a, b)
            expected = np.kron(a.cpu().numpy(), b.cpu().numpy())
            self.assertEqual(result, expected)

            # NumPy doesn't work if the first argument is empty
            result = torch.kron(b, a)
            self.assertEqual(result.shape, expected.shape)

        empty_shapes = [(0,), (2, 0), (1, 0, 3)]
        for empty_shape in empty_shapes:
            run_test_case(empty_shape)

    @dtypes(*floating_and_complex_types())
    def test_kron_errors_and_warnings(self, device, dtype):
        # if non-empty out tensor with wrong shape is passed a warning is given
        a = torch.eye(3, dtype=dtype, device=device)
        b = torch.ones((2, 2), dtype=dtype, device=device)
        out = torch.empty_like(a)
        with warnings.catch_warnings(record=True) as w:
            # Trigger warning
            torch.kron(a, b, out=out)
            # Check warning occurs
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtypes should match
        out = torch.empty_like(a).to(torch.int)
        with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
            torch.kron(a, b, out=out)

    # This test confirms that torch.linalg.norm's dtype argument works
    # as expected, according to the function's documentation
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble, torch.bfloat16, torch.float16)
    def test_norm_dtype(self, device, dtype):
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        def run_test_case(input_size, ord, keepdim, to_dtype):
            msg = (
                f'input_size={input_size}, ord={ord}, keepdim={keepdim}, '
                f'dtype={dtype}, to_dtype={to_dtype}')
            input = make_arg(input_size)
            result = torch.linalg.norm(input, ord, keepdim=keepdim)
            self.assertEqual(result.dtype, input.real.dtype, msg=msg)

            result_out = torch.empty((0), dtype=result.dtype, device=device)
            torch.linalg.norm(input, ord, keepdim=keepdim, out=result_out)
            self.assertEqual(result, result_out, msg=msg)

            result = torch.linalg.norm(input.to(to_dtype), ord, keepdim=keepdim)
            result_with_dtype = torch.linalg.norm(input, ord, keepdim=keepdim, dtype=to_dtype)
            self.assertEqual(result, result_with_dtype, msg=msg)

            result_out_with_dtype = torch.empty_like(result_with_dtype)
            torch.linalg.norm(input, ord, keepdim=keepdim, dtype=to_dtype, out=result_out_with_dtype)
            self.assertEqual(result_with_dtype, result_out_with_dtype, msg=msg)

        ord_vector = [0, 1, -1, 2, -2, 3, -3, 4.5, -4.5, inf, -inf, None]

        # In these orders we are computing the 10-th power and 10-th root of numbers.
        # We avoid them for half-precision types as it makes the tests above too badly conditioned
        if dtype != torch.float16 and dtype != torch.bfloat16:
            ord_vector.extend([0.1, -0.1])
        ord_matrix = ['fro', 'nuc', 1, -1, 2, -2, inf, -inf, None]
        S = 10

        if dtype == torch.cfloat:
            norm_dtypes = (torch.cfloat, torch.cdouble)
        elif dtype == torch.cdouble:
            norm_dtypes = (torch.cdouble,)
        elif dtype in (torch.float16, torch.bfloat16, torch.float):
            norm_dtypes = (torch.float, torch.double)
        elif dtype == torch.double:
            norm_dtypes = (torch.double,)
        else:
            raise RuntimeError("Unsupported dtype")

        for ord, keepdim, norm_dtype in product(ord_vector, (True, False), norm_dtypes):
            run_test_case((S,) , ord, keepdim, norm_dtype)

        for ord, keepdim, norm_dtype in product(ord_matrix, (True, False), norm_dtypes):
            if ord in [2, -2, 'nuc']:
                # We need torch.svdvals
                if dtype == torch.float16 or dtype == torch.bfloat16:
                    continue

                # We need LAPACK or equivalent
                if ((torch.device(device).type == 'cuda' and not torch.cuda.has_magma and not has_cusolver()) or
                   (torch.device(device).type == 'cpu' and not torch._C.has_lapack)):
                    continue
            run_test_case((S, S) , ord, keepdim, norm_dtype)

    # This test confirms torch.linalg.norm bfloat16 and half get right result.
    @dtypes(torch.bfloat16, torch.float16)
    def test_norm_bfloat16_and_half(self, device, dtype):
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        def run_test_case(input_size, ord, keepdim):
            msg = (
                f'input_size={input_size}, ord={ord}, keepdim={keepdim}, '
                f'dtype={dtype}')
            input = make_arg(input_size).fill_(1)
            result_ref = torch.linalg.norm(input.float(), ord, keepdim=keepdim).to(dtype=dtype)
            result = torch.linalg.norm(input, ord, keepdim=keepdim)
            self.assertEqual(result_ref, result, msg=msg)

        ord_vector = [0, 1, -1, 2, -2, 3, -3, 4.5, -4.5, inf, -inf, None]
        for S, ord, keepdim in product((10, 2049), ord_vector, (True, False)):
            run_test_case((S,) , ord, keepdim, )

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble, torch.bfloat16, torch.float16)
    def test_vector_norm(self, device, dtype):
        if IS_ARM64 and device == 'cpu' and dtype in [torch.float16, torch.bfloat16, torch.float32]:
            raise unittest.SkipTest("Fails on ARM, see https://github.com/pytorch/pytorch/issues/125438")
        # have to use torch.randn(...).to(bfloat16) instead of
        # This test compares torch.linalg.vector_norm's output with
        # torch.linalg.norm given a flattened tensor
        ord_vector = [0, 0.9, 1, 2, 3, inf, -0.5, -1, -2, -3, -inf, 1 + 2j]
        input_sizes = [
            (1, ),
            (10, ),
            (4, 5),
            (3, 4, 5),
            (0, ),
            (0, 10),
            (0, 0),
            (10, 0, 10),
        ]

        def vector_norm_reference(input, ord, dim=None, keepdim=False, dtype=None):
            if dim is None:
                input_maybe_flat = input.flatten(0, -1)
            else:
                input_maybe_flat = input

            result = torch.linalg.norm(input_maybe_flat, ord, dim=dim, keepdim=keepdim, dtype=dtype)
            if keepdim and dim is None:
                result = result.reshape([1] * input.dim())
            return result

        def run_test_case(input, ord, dim, keepdim, norm_dtype):
            if isinstance(ord, complex):
                error_msg = "Expected a non-complex scalar"
                with self.assertRaisesRegex(RuntimeError, error_msg):
                    torch.linalg.vector_norm(input, ord, dim=dim, keepdim=keepdim, dtype=norm_dtype)
            elif (input.numel() == 0 and
                  (ord < 0. or ord == inf) and
                  (dim is None or input.shape[dim] == 0)):
                # The operation does not have an identity.
                error_msg = "linalg.vector_norm cannot compute"
                with self.assertRaisesRegex(RuntimeError, error_msg):
                    torch.linalg.vector_norm(input, ord, dim=dim, keepdim=keepdim)
            else:
                msg = (f'input.size()={input.size()}, ord={ord}, dim={dim}, '
                       f'keepdim={keepdim}, dtype={dtype}, norm_dtype={norm_dtype}')
                result_dtype_reference = vector_norm_reference(input, ord, dim=dim, keepdim=keepdim, dtype=norm_dtype)
                result_dtype = torch.linalg.vector_norm(input, ord, dim=dim, keepdim=keepdim, dtype=norm_dtype)
                if dtype.is_complex:
                    result_dtype_reference = result_dtype_reference.real
                self.assertEqual(result_dtype, result_dtype_reference, msg=msg)

                if norm_dtype is not None:
                    ref = torch.linalg.vector_norm(input.to(norm_dtype), ord, dim=dim, keepdim=keepdim)
                    actual = torch.linalg.vector_norm(input, ord, dim=dim, keepdim=keepdim, dtype=norm_dtype)
                    self.assertEqual(ref, actual, msg=msg)

        if dtype == torch.cfloat:
            norm_dtypes = (None, torch.cfloat, torch.cdouble)
        elif dtype == torch.cdouble:
            norm_dtypes = (None, torch.cdouble)
        elif dtype in (torch.float16, torch.bfloat16, torch.float):
            norm_dtypes = (None, torch.float, torch.double)
        elif dtype == torch.double:
            norm_dtypes = (None, torch.double)
        else:
            raise RuntimeError("Unsupported dtype")

        for amp in [False, True]:
            with torch.autocast(device_type=device, enabled=amp):
                for input_size, ord, keepdim, norm_dtype in product(input_sizes, ord_vector, [True, False], norm_dtypes):
                    input = make_tensor(input_size, dtype=dtype, device=device, low=-9, high=9)
                    for dim in [None, random.randint(0, len(input_size) - 1)]:
                        run_test_case(
                            input,
                            ord,
                            dim,
                            keepdim,
                            norm_dtype)


    def test_vector_norm_decom_unbacked_checks(self):
        from torch._refs.linalg import _check_vector_norm_args

        class Mod(torch.nn.Module):
            def __init__(self, ord, dim):
                super().__init__()
                self.ord = ord
                self.dim = dim

            def forward(self, a):
                x = a.item()
                tensor_unbacked_size = torch.ones(x, x + 1, x + 2)
                _check_vector_norm_args(tensor_unbacked_size, self.ord, self.dim)
                return tensor_unbacked_size

        def test(
            ord: Union[float, int],
            dim: Optional[DimsType],
            expect_numel_runtime_check: bool,
            expect_index_0_check: bool = False,
        ) -> None:
            m = Mod(ord, dim)
            exported_program: torch.export.ExportedProgram = torch.export.export(
                m, args=tuple(torch.tensor([1]))
            )
            self.assertEqual(
                "Runtime assertion failed for expression Ne(u0*(u0 + 1)*(u0 + 2), 0)"
                in exported_program.graph_module.code,
                expect_numel_runtime_check,
            )
            self.assertEqual(
                "Runtime assertion failed for expression Ne(u0, 0) | Ne(u0*(u0 + 1)*(u0 + 2), 0)"
                in exported_program.graph_module.code,
                expect_index_0_check,
            )

        # dim is int
        test(-1, 1, True)

        # dim is None
        test(-1, None, True)

        # len(dim) == 0
        test(-1, [], True)

        # shape[d] == 0
        test(-1, [0], False, True)

        # u0 + 1 == 0 is False we do not see a runtime assert in the generated graph.
        test(-1, [1], False, False)

        test(-1, [0, 1], False, True)
        test(-1, [0, 0], False, True)

    def test_vector_norm_dim_tuple_arg(self, device):
        test_cases = [
            # input size, dim, error, error message
            ((4, ), (0, ), None, None),
            ((4, ), (1, ), IndexError, r'Dimension out of range'),
            ((4, ), (-2, ), IndexError, r'Dimension out of range'),
            ((4, 3), (0, -1), None, None),
            ((4, 3), (0, 0), RuntimeError, r'dim 0 appears multiple times in the list of dims'),
            ((4, 3), (0, -2), RuntimeError, r'dim 0 appears multiple times in the list of dims'),
            ((4, 3), (0, 1.0), TypeError, r"argument 'dim' must be tuple of ints"),
            ((4, 3), (None, ), TypeError, r"argument 'dim' must be tuple of ints"),
        ]
        for input_size, dim_tuple, error, error_msg in test_cases:
            input = torch.randn(input_size, device=device)
            # vector_norm should accept a tuple or a list for dim arg
            for dim in [dim_tuple, list(dim_tuple)]:
                if error is None:
                    torch.linalg.vector_norm(input, dim=dim)
                else:
                    with self.assertRaises(error, msg=error_msg):
                        torch.linalg.vector_norm(input, dim=dim)

    # This test compares torch.linalg.norm and numpy.linalg.norm to ensure that
    # their vector norm results match
    @dtypes(torch.float, torch.double)
    def test_norm_vector(self, device, dtype):
        def run_test_case(input, p, dim, keepdim):
            result = torch.linalg.norm(input, ord, dim, keepdim)
            input_numpy = input.cpu().numpy()
            result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)

            msg = f'input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}'
            self.assertEqual(result, result_numpy, msg=msg)

            result_out = torch.empty_like(result)
            torch.linalg.norm(input, ord, dim, keepdim, out=result_out)
            self.assertEqual(result, result_out, msg=msg)

        ord_vector = [0, 1, -1, 2, -2, 3, -3, 4.5, -4.5, inf, -inf]
        S = 10
        test_cases = [
            # input size, p settings, dim
            ((S, ), ord_vector, None),
            ((S, ), ord_vector, 0),
            ((S, S, S), ord_vector, 0),
            ((S, S, S), ord_vector, 1),
            ((S, S, S), ord_vector, 2),
            ((S, S, S), ord_vector, -1),
            ((S, S, S), ord_vector, -2),
        ]
        L = 1_000_000
        if dtype == torch.double:
            test_cases.append(((L, ), ord_vector, None))
        for keepdim in [True, False]:
            for input_size, ord_settings, dim in test_cases:
                input = torch.randn(*input_size, dtype=dtype, device=device)
                for ord in ord_settings:
                    run_test_case(input, ord, dim, keepdim)

    # This test compares torch.linalg.norm, torch.linalg.matrix_norm and numpy.linalg.norm to
    # ensure that their matrix norm results match.
    @skipMeta  # https://github.com/pytorch/pytorch/issues/54082
    @skipCUDAIfNoMagma
    @dtypes(torch.float, torch.double)
    @precisionOverride({torch.float32: 2e-4})
    def test_norm_matrix(self, device, dtype):
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        def run_test_case(input, ord, dim, keepdim):
            msg = f'input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}'
            result = torch.linalg.norm(input, ord, dim, keepdim)
            input_numpy = input.cpu().numpy()
            result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)

            result = torch.linalg.norm(input, ord, dim, keepdim)
            self.assertEqual(result, result_numpy, msg=msg)
            if ord is not None and dim is not None:
                result = torch.linalg.matrix_norm(input, ord, dim, keepdim)
                self.assertEqual(result, result_numpy, msg=msg)

        ord_matrix = [1, -1, 2, -2, inf, -inf, 'nuc', 'fro']
        S = 10
        test_cases = [
            # input size, dim
            ((S, S), None),
            ((S, S), (0, 1)),
            ((S, S), (1, 0)),
            ((S, S, S, S), (2, 0)),
            ((S, S, S, S), (-1, -2)),
            ((S, S, S, S), (-1, -3)),
            ((S, S, S, S), (-3, 2)),
        ]

        for (shape, dim), keepdim, ord in product(test_cases, [True, False], ord_matrix):
            if ord in [2, -2, 'nuc']:
                # We need torch.svdvals
                if dtype == torch.float16 or dtype == torch.bfloat16:
                    continue
                # We need LAPACK or equivalent
                if ((torch.device(device).type == 'cuda' and not torch.cuda.has_magma and not has_cusolver()) or
                   (torch.device(device).type == 'cpu' and not torch._C.has_lapack)):
                    continue
            run_test_case(make_arg(shape), ord, dim, keepdim)

    @onlyCUDA
    @dtypes(torch.bfloat16, torch.float16)
    def test_norm_fused_type_promotion(self, device, dtype):
        x = torch.randn(10, device=device, dtype=dtype)

        def profile_and_check(fn, x, kwargs):
            with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU,)) as p:
                fn(x, **kwargs, dtype=torch.float)
            # smoke check that profiler returned some events
            self.assertTrue("aten::linalg_vector_norm" in (e.name for e in p.events()))
            # test that there was no explicit copy
            self.assertFalse("aten::to" in (e.name for e in p.events()))

        for f, kwargs, in zip((torch.linalg.vector_norm, torch.norm), ({}, {"p" : 2})):
            profile_and_check(f, x, kwargs)

    @skipMeta  # https://github.com/pytorch/pytorch/issues/53739
    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-3})
    def test_cond(self, device, dtype):
        def run_test_case(input, p):
            result = torch.linalg.cond(input, p)
            result_numpy = np.linalg.cond(input.cpu().numpy(), p)
            self.assertEqual(result, result_numpy, rtol=1e-2, atol=self.precision, exact_dtype=False)
            self.assertEqual(result.shape, result_numpy.shape)

            # test out= variant
            out = torch.empty_like(result)
            ans = torch.linalg.cond(input, p, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, result)

        norm_types = [1, -1, 2, -2, inf, -inf, 'fro', 'nuc', None]
        input_sizes = [(32, 32), (2, 3, 3, 3)]
        for input_size in input_sizes:
            input = torch.randn(*input_size, dtype=dtype, device=device)
            for p in norm_types:
                run_test_case(input, p)

        # test empty batch sizes
        input_sizes = [(0, 3, 3), (0, 2, 5, 5)]
        for input_size in input_sizes:
            input = torch.randn(*input_size, dtype=dtype, device=device)
            for p in norm_types:
                run_test_case(input, p)

        # test non-square input
        input_sizes = [(16, 32), (32, 16), (2, 3, 5, 3), (2, 3, 3, 5)]
        for input_size in input_sizes:
            input = torch.randn(*input_size, dtype=dtype, device=device)
            for p in [2, -2, None]:
                run_test_case(input, p)

        # test for singular input
        a = torch.eye(3, dtype=dtype, device=device)
        a[-1, -1] = 0  # make 'a' singular
        for p in norm_types:
            try:
                run_test_case(a, p)
            except np.linalg.LinAlgError:
                # Numpy may fail to converge for some BLAS backends (although this is very rare)
                # See the discussion in https://github.com/pytorch/pytorch/issues/67675
                pass

        # test for 0x0 matrices. NumPy doesn't work for such input, we return 0
        input_sizes = [(0, 0), (2, 5, 0, 0)]
        for input_size in input_sizes:
            input = torch.randn(*input_size, dtype=dtype, device=device)
            for p in ['fro', 2]:
                expected_dtype = a.real.dtype if dtype.is_complex else dtype
                expected = torch.zeros(input_size[:-2], dtype=expected_dtype, device=device)
                actual = torch.linalg.cond(input, p)
                self.assertEqual(actual, expected)

    @skipMeta  # https://github.com/pytorch/pytorch/issues/53739
    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-3})
    def test_cond_errors_and_warnings(self, device, dtype):
        norm_types = [1, -1, 2, -2, inf, -inf, 'fro', 'nuc', None]

        # cond expects the input to be at least 2-dimensional
        a = torch.ones(3, dtype=dtype, device=device)
        for p in norm_types:
            with self.assertRaisesRegex(RuntimeError, r'at least 2 dimensions'):
                torch.linalg.cond(a, p)

        # for some norm types cond expects the input to be square
        a = torch.ones(3, 2, dtype=dtype, device=device)
        norm_types = [1, -1, inf, -inf, 'fro', 'nuc']
        for p in norm_types:
            with self.assertRaisesRegex(RuntimeError, r'must be batches of square matrices'):
                torch.linalg.cond(a, p)

        # if non-empty out tensor with wrong shape is passed a warning is given
        a = torch.ones((2, 2), dtype=dtype, device=device)
        for p in ['fro', 2]:
            real_dtype = a.real.dtype if dtype.is_complex else dtype
            out = torch.empty(a.shape, dtype=real_dtype, device=device)
            with warnings.catch_warnings(record=True) as w:
                # Trigger warning
                torch.linalg.cond(a, p, out=out)
                # Check warning occurs
                self.assertEqual(len(w), 1)
                self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtypes should be safely castable
        out = torch.empty(0, dtype=torch.int, device=device)
        for p in ['fro', 2]:
            with self.assertRaisesRegex(RuntimeError, "but got result with dtype Int"):
                torch.linalg.cond(a, p, out=out)

        # device should match
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty(0, dtype=dtype, device=wrong_device)
            for p in ['fro', 2]:
                with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                    torch.linalg.cond(a, p, out=out)

        # for batched input if at least one matrix in the batch is not invertible,
        # we can't get the result for all other (possibly) invertible matrices in the batch without an explicit for loop.
        # this should change when at::inverse works with silent errors
        # NumPy works fine in this case because it's possible to silence the error and get the inverse matrix results
        # possibly filled with NANs
        batch_dim = 3
        a = torch.eye(3, 3, dtype=dtype, device=device)
        a = a.reshape((1, 3, 3))
        a = a.repeat(batch_dim, 1, 1)
        a[1, -1, -1] = 0  # now a[1] is singular
        for p in [1, -1, inf, -inf, 'fro', 'nuc']:
            result = torch.linalg.cond(a, p)
            self.assertEqual(result[1], float('inf'))

        # check invalid norm type
        a = torch.ones(3, 3, dtype=dtype, device=device)
        for p in ['wrong_norm', 5]:
            with self.assertRaisesRegex(RuntimeError, f"linalg.cond got an invalid norm type: {p}"):
                torch.linalg.cond(a, p)

    # This test calls torch.linalg.norm and numpy.linalg.norm with illegal arguments
    # to ensure that they both throw errors
    @dtypes(torch.float, torch.double)
    def test_norm_errors(self, device, dtype):
        def run_error_test_case(input, ord, dim, keepdim, error_type, error_regex):
            test_case_info = (
                f'test case input.size()={input.size()}, ord={ord}, dim={dim}, '
                f'keepdim={keepdim}, dtype={dtype}')

            with self.assertRaisesRegex(error_type, error_regex, msg=test_case_info):
                torch.linalg.norm(input, ord, dim, keepdim)

            input_numpy = input.cpu().numpy()

            msg = f'numpy does not raise error but pytorch does, for case "{test_case_info}"'
            with self.assertRaises(Exception, msg=test_case_info):
                np.linalg.norm(input_numpy, ord, dim, keepdim)

        S = 10
        error_test_cases = [
            # input size, p settings, dim, error type, error regex
            ((S, ), ['fro', 'nuc'], None, RuntimeError, r'A must have at least 2 dimensions'),
            ((S, S), [3.5], None, RuntimeError, r'matrix_norm: Order 3.5 not supported'),
            ((S, S), [0], None, RuntimeError, r'matrix_norm: Order 0 not supported'),
            ((S, S), ['fail'], None, RuntimeError, r'matrix_norm: Order fail not supported'),
            ((S, S), ['fro', 'nuc'], 0, RuntimeError, r'matrix_norm: dim must be a 2-tuple'),
            ((S, S), ['fro', 'nuc', 2], (0, 0), RuntimeError, r'dims must be different'),
            ((S, S), ['fro', 'nuc', 2], (-1, 1), RuntimeError, r'dims must be different'),
            ((S, S), ['fro', 'nuc', 2], (0, 4), IndexError, r'Dimension out of range'),
            ((S, ), [0], (4, ), IndexError, r'Dimension out of range'),
            ((S, ), [None], (0, 0), RuntimeError, r'dim 0 appears multiple times'),
            ((S, S, S), [1], (0, 1, 2), RuntimeError, r"If dim is specified, it must be of length 1 or 2."),
            ((S, S, S), [1], None, RuntimeError, r"If dim is not specified but ord is, the input must be 1D or 2D"),
        ]
        for keepdim in [True, False]:
            for input_size, ord_settings, dim, error_type, error_regex in error_test_cases:
                input = torch.randn(*input_size, dtype=dtype, device=device)
                for ord in ord_settings:
                    run_error_test_case(input, ord, dim, keepdim, error_type, error_regex)

    # Test complex number inputs for linalg.norm
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.cfloat, torch.cdouble)
    @precisionOverride({torch.cfloat: 5e-4})
    def test_norm_complex(self, device, dtype):
        def gen_error_message(input_size, ord, keepdim, dim=None):
            return f"complex norm failed for input size {input_size}, ord={ord}, keepdim={keepdim}, dim={dim}"

        vector_ords = [None, 0, 1, 2, 3, inf, -1, -2, -3, -inf]
        matrix_ords = [None, 'fro', 'nuc', 1, 2, inf, -1, -2, -inf]

        # Test supported ords
        for keepdim in [False, True]:
            # vector norm
            x = torch.randn(25, device=device, dtype=dtype)
            xn = x.cpu().numpy()
            for ord in vector_ords:
                res = torch.linalg.norm(x, ord, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, ord, keepdims=keepdim)
                msg = gen_error_message(x.size(), ord, keepdim)
                self.assertEqual(res.shape, expected.shape, msg=msg)
                self.assertEqual(res, expected, msg=msg, exact_dtype=False)

                res_out = torch.tensor([], device=device, dtype=res.dtype)
                torch.linalg.norm(x, ord, keepdim=keepdim, out=res_out)
                self.assertEqual(res_out.shape, expected.shape, msg=msg)
                self.assertEqual(res_out, expected, msg=msg)

            # matrix norm
            x = torch.randn(25, 25, device=device, dtype=dtype)
            xn = x.cpu().numpy()
            for ord in matrix_ords:
                res = torch.linalg.norm(x, ord, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, ord, keepdims=keepdim)
                msg = gen_error_message(x.size(), ord, keepdim)
                self.assertEqual(res.shape, expected.shape, msg=msg)
                self.assertEqual(res, expected, msg=msg, exact_dtype=False)

                res_out = torch.tensor([], device=device, dtype=res.dtype)
                torch.linalg.norm(x, ord, keepdim=keepdim, out=res_out)
                self.assertEqual(res_out.shape, expected.shape, msg=msg)
                self.assertEqual(res_out, expected, msg=msg)

    @onlyCPU
    def test_norm_complexhalf(self, device):
        def gen_error_message(input_size, ord, keepdim, dim=None):
            return f"complex norm failed for input size {input_size}, ord={ord}, keepdim={keepdim}, dim={dim}"

        vector_ords = [None, 0, 1, 2, 3, inf, -1, -2, -3, -inf]

        # Test supported ords
        for keepdim in [False, True]:
            # vector norm
            x = torch.randn(25, device=device, dtype=torch.chalf)
            x_cfloat = x.to(torch.cfloat)
            for ord in vector_ords:
                res = torch.linalg.norm(x, ord, keepdim=keepdim)
                res_float = torch.linalg.norm(x_cfloat, ord, keepdim=keepdim)
                msg = gen_error_message(x.size(), ord, keepdim)
                self.assertEqual(res.shape, res_float.shape, msg=msg)
                self.assertEqual(res.dtype, torch.half, msg=msg)
                self.assertEqual(res, res_float, msg=msg, exact_dtype=False)

                res_out = torch.tensor([], device=device, dtype=res.dtype)
                torch.linalg.norm(x, ord, keepdim=keepdim, out=res_out)
                self.assertEqual(res_out.shape, res_float.shape, msg=msg)
                self.assertEqual(res_out.dtype, torch.half, msg=msg)
                self.assertEqual(res_out, res_float, msg=msg, exact_dtype=False)

    # Test that linal.vector_norm gives the same result as numpy when inputs
    # contain extreme values (inf, -inf, nan)
    def test_vector_norm_extreme_values(self, device):
        vector_ords = [0, 1, 2, 3, inf, -1, -2, -3, -inf]
        vectors = []
        for pair in itertools.product([inf, -inf, 0.0, nan, 1.0], repeat=2):
            vectors.append(list(pair))
        for vector in vectors:
            x = torch.tensor(vector, device=device)
            x_n = x.cpu().numpy()
            for ord in vector_ords:
                msg = f'ord={ord}, vector={vector}'
                result = torch.linalg.vector_norm(x, ord=ord)
                result_n = np.linalg.norm(x_n, ord=ord)
                self.assertEqual(result, result_n, msg=msg)

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_vector_norm_reduce_over_1D_vector(self, device, dtype):
        input_sizes_and_dims = [
            ((6, 1), -1),
            ((3, 1, 2, 1), (1, 3)),
            ((1,), None),
        ]
        orders = [float('inf'), -float('inf'), 0, 1, -1, 2, -2]
        keepdims = [True, False]

        for input_size_and_dim, ord, keepdim in product(input_sizes_and_dims, orders, keepdims):
            input_size = input_size_and_dim[0]
            dim = input_size_and_dim[1]
            if type(dim) is tuple and ord == 0:
                # skip because np.linalg.norm raises 'ValueError: Invalid norm order for matrices.'
                continue
            input = make_tensor(input_size, dtype=dtype, device=device, low=-9, high=9)
            result = torch.linalg.vector_norm(input, ord, dim, keepdim)
            result_numpy = np.linalg.norm(input.cpu().numpy(), ord, dim, keepdim)

            msg = f'input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}'
            self.assertEqual(result, result_numpy, msg=msg)

    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double)
    @precisionOverride({torch.float32: 2e-5})
    def test_matrix_norm(self, device, dtype):
        # Test only inputs for which torch.linalg.matrix_norm diverges from torch.linalg.norm
        A = make_tensor((2, 2, 2), dtype=dtype, device=device)

        with self.assertRaisesRegex(RuntimeError, r'linalg.matrix_norm:.*must have at least 2 dimensions.*'):
            torch.linalg.matrix_norm(make_tensor((2,), dtype=dtype, device=device))
        with self.assertRaisesRegex(RuntimeError, r'linalg.matrix_norm:.*must be a 2-tuple.*'):
            torch.linalg.matrix_norm(A, dim=(0,))
        with self.assertRaisesRegex(RuntimeError, r'.*not supported.*'):
            torch.linalg.matrix_norm(A, ord=0)
        with self.assertRaisesRegex(RuntimeError, r'.*not supported.*'):
            torch.linalg.matrix_norm(A, ord=3.0)
        with self.assertRaisesRegex(RuntimeError, "Expected a non-complex scalar"):
            torch.linalg.matrix_norm(A, ord=1 + 2j)

        # Test dim=None behavior
        ref = torch.linalg.norm(A, dim=(-2, -1))
        res = torch.linalg.matrix_norm(A)
        self.assertEqual(ref, res)

    # Test that linal.norm gives the same result as numpy when inputs
    # contain extreme values (inf, -inf, nan)
    @unittest.skipIf(IS_WINDOWS, "Skipped on Windows!")
    @unittest.skipIf(IS_MACOS, "Skipped on MacOS!")
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    def test_norm_extreme_values(self, device):
        vector_ords = [0, 1, 2, 3, inf, -1, -2, -3, -inf]
        # matrix_ords 'nuc', 2, -2 are skipped currently
        # See issue https://github.com/pytorch/pytorch/issues/71911
        matrix_ords = ['fro', 1, inf, -1, -inf]
        vectors = []
        matrices = []
        for pair in itertools.product([inf, -inf, 0.0, nan, 1.0], repeat=2):
            vectors.append(list(pair))
            matrices.append([[pair[0], pair[1]]])
            matrices.append([[pair[0]], [pair[1]]])
        for vector in vectors:
            x = torch.tensor(vector).to(device)
            x_n = x.cpu().numpy()
            for ord in vector_ords:
                msg = f'ord={ord}, vector={vector}'
                result = torch.linalg.norm(x, ord=ord)
                result_n = np.linalg.norm(x_n, ord=ord)
                self.assertEqual(result, result_n, msg=msg)

        # TODO: Remove this function once the broken cases are fixed
        def is_broken_matrix_norm_case(ord, x):
            if self.device_type == 'cuda':
                if x.size() == torch.Size([1, 2]):
                    if ord in ['nuc', 2, -2] and isnan(x[0][0]) and x[0][1] == 1:
                        # These cases are broken because of an issue with svd
                        # https://github.com/pytorch/pytorch/issues/43567
                        return True
                if ord in ['nuc', 2, -2]:
                    # These cases are broken because of another issue with svd
                    # https://github.com/pytorch/pytorch/issues/52633
                    return True
            return False

        for matrix in matrices:
            x = torch.tensor(matrix).to(device)
            x_n = x.cpu().numpy()
            for ord in matrix_ords:
                msg = f'ord={ord}, matrix={matrix}'
                if is_broken_matrix_norm_case(ord, x):
                    continue
                else:
                    result_n = np.linalg.norm(x_n, ord=ord)
                    result = torch.linalg.norm(x, ord=ord)
                    self.assertEqual(result, result_n, msg=msg)

    # Test degenerate shape results match numpy for linalg.norm vector norms
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_norm_vector_degenerate_shapes(self, device, dtype):
        def run_test_case(input, ord, dim, keepdim):
            msg = f'input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}'
            if (input.numel() == 0 and
                (ord < 0. or ord == inf) and
               (dim is None or input.shape[dim] == 0)):
                with self.assertRaises(RuntimeError):
                    torch.linalg.norm(input, ord, dim, keepdim)
            else:
                input_numpy = input.cpu().numpy()
                result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)
                result = torch.linalg.norm(input, ord, dim, keepdim)
                self.assertEqual(result, result_numpy, msg=msg)

        ord_vector = [0, 0.5, 1, 2, 3, inf, -0.5, -1, -2, -3, -inf]
        S = 10
        test_cases = [
            # input size, dim
            ((0, ), None),
            ((0, S), 0),
            ((0, S), 1),
            ((S, 0), 0),
            ((S, 0), 1),
        ]
        for keepdim in [True, False]:
            for input_size, dim in test_cases:
                input = torch.randn(*input_size, dtype=dtype, device=device)
                for ord in ord_vector:
                    run_test_case(input, ord, dim, keepdim)

    # Test degenerate shape results match numpy for linalg.norm matrix norms
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_norm_matrix_degenerate_shapes(self, device, dtype):
        def run_test_case(input, ord, dim, keepdim, should_error):
            msg = f'input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}'
            input_numpy = input.cpu().numpy()
            ops = [torch.linalg.norm]

            if ord is not None and dim is not None:
                ops.append(torch.linalg.matrix_norm)

            if should_error:
                with self.assertRaises(ValueError):
                    np.linalg.norm(input_numpy, ord, dim, keepdim)
                for op in ops:
                    with self.assertRaises(IndexError):
                        op(input, ord, dim, keepdim)
            else:
                result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)
                for op in ops:
                    result = op(input, ord, dim, keepdim)
                    self.assertEqual(result, result_numpy, msg=msg)

        ord_matrix = ['fro', 'nuc', 1, 2, inf, -1, -2, -inf, None]
        S = 10
        test_cases = [
            # input size, p settings that cause error, dim
            ((0, 0), [1, 2, inf, -1, -2, -inf], None),
            ((0, S), [2, inf, -2, -inf], None),
            ((S, 0), [1, 2, -1, -2], None),
            ((S, S, 0), [], (0, 1)),
            ((1, S, 0), [], (0, 1)),
            ((0, 0, S), [1, 2, inf, -1, -2, -inf], (0, 1)),
            ((0, 0, S), [1, 2, inf, -1, -2, -inf], (1, 0)),
        ]

        for keepdim in [True, False]:
            for input_size, error_ords, dim in test_cases:
                input = torch.randn(*input_size, dtype=dtype, device=device)
                for ord in ord_matrix:
                    run_test_case(input, ord, dim, keepdim, ord in error_ords)

    def test_norm_fastpaths(self, device):
        x = torch.randn(3, 5, device=device)

        # slow path
        result = torch.linalg.norm(x, 4.5, 1)
        expected = torch.pow(x.abs().pow(4.5).sum(1), 1.0 / 4.5)
        self.assertEqual(result, expected)

        # fast 0-norm
        result = torch.linalg.norm(x, 0, 1)
        expected = (x != 0).type_as(x).sum(1)
        self.assertEqual(result, expected)

        # fast 1-norm
        result = torch.linalg.norm(x, 1, 1)
        expected = x.abs().sum(1)
        self.assertEqual(result, expected)

        # fast 2-norm
        result = torch.linalg.norm(x, 2, 1)
        expected = torch.sqrt(x.pow(2).sum(1))
        self.assertEqual(result, expected)

        # fast 3-norm
        result = torch.linalg.norm(x, 3, 1)
        expected = torch.pow(x.pow(3).abs().sum(1), 1.0 / 3.0)
        self.assertEqual(result, expected)

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    # NumPy computes only in float64 and complex128 precisions
    # for float32 or complex64 results might be very different from float64 or complex128
    @dtypes(torch.float64, torch.complex128)
    def test_eig_numpy(self, device, dtype):
        def run_test(shape, *, symmetric=False):
            from torch.testing._internal.common_utils import random_symmetric_matrix

            if not dtype.is_complex and symmetric:
                # for symmetric real-valued inputs eigenvalues and eigenvectors have imaginary part equal to zero
                # unlike NumPy the result is not cast to float32 or float64 dtype in this case
                a = random_symmetric_matrix(shape[-1], *shape[:-2], dtype=dtype, device=device)
            else:
                a = make_tensor(shape, dtype=dtype, device=device)

            actual = torch.linalg.eig(a)

            # compare with NumPy
            # the eigenvalues are not necessarily ordered
            # so order of NumPy and PyTorch can be different
            expected = np.linalg.eig(a.cpu().numpy())

            # sort NumPy output
            ind = np.argsort(expected[0], axis=-1)[::-1]
            expected = (np.take_along_axis(expected[0], ind, axis=-1), np.take_along_axis(expected[1], ind[:, None], axis=-1))

            # sort PyTorch output
            # torch.argsort doesn't work with complex inputs, NumPy sorting on CPU is used instead
            # RuntimeError: _th_sort not supported on CUDAType for ComplexDouble
            # RuntimeError: "sorting_kernel_method_name" not implemented for 'ComplexDouble'
            ind = np.argsort(actual[0].cpu().numpy(), axis=-1)[::-1]
            actual_np = [x.cpu().numpy() for x in actual]
            sorted_actual = (
                np.take_along_axis(actual_np[0], ind, axis=-1),
                np.take_along_axis(actual_np[1], ind[:, None], axis=-1))

            self.assertEqual(expected[0], sorted_actual[0], exact_dtype=False)
            self.assertEqual(abs(expected[1]), abs(sorted_actual[1]), exact_dtype=False)

        shapes = [(0, 0),  # Empty matrix
                  (5, 5),  # Single matrix
                  (0, 0, 0), (0, 5, 5),  # Zero batch dimension tensors
                  (2, 5, 5),  # 3-dim tensors
                  (2, 1, 5, 5)]  # 4-dim tensors
        for shape in shapes:
            run_test(shape)
            run_test(shape, symmetric=True)

    @onlyCUDA
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    def test_eig_compare_backends(self, device, dtype):



... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 3 class(es): TestLinalg, Mod, PrecisionContext

### Functions
This file defines 465 function(s): blaslt_supported_device, tunableop_matmul, get_tunableop_validators, find_tunableop_result, get_tunableop_untuned_filename, setUp, tearDown, _tunableop_ctx, _set_tunableop_defaults, _compare_untuned_tuned_entries, test_inner, check, test_outer, run_test_case, test_matrix_rank_removed_error, test_solve_removed_error, test_eig_removed_error, test_symeig_removed_error, test_lstsq_removed_error, test_linalg_lstsq, check_solution_correctness, check_correctness_ref, apply_if_not_empty, select_if_not_empty, check_correctness_scipy, scipy_ref, check_correctness_numpy, numpy_ref, test_linalg_lstsq_batch_broadcasting, check_correctness


## Key Components

The file contains 40495 words across 10078 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 465650 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
