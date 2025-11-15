# Documentation: `test/test_ops.py`

## File Metadata

- **Path**: `test/test_ops.py`
- **Size**: 124,812 bytes (121.89 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: unknown"]
import contextlib
import copy
import inspect
import itertools
import os
import re
import unittest
import warnings
from collections import defaultdict
from collections.abc import Sequence
from functools import partial
from importlib import import_module

import torch
import torch._prims as prims
import torch.utils._pytree as pytree
from torch._prims.context import TorchRefsMode
from torch._prims_common.wrappers import _maybe_remove_out_wrapper
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch._subclasses.fake_utils import outputs_alias_inputs
from torch.testing import make_tensor
from torch.testing._internal import composite_compliance, opinfo
from torch.testing._internal.common_cuda import with_tf32_off
from torch.testing._internal.common_device_type import (
    deviceCountAtLeast,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
    onlyNativeDeviceTypesAnd,
    onlyOn,
    OpDTypes,
    ops,
    skipMeta,
    skipXPU,
)
from torch.testing._internal.common_dtype import (
    all_types_and_complex_and,
    floating_and_complex_types_and,
    integral_types_and,
)
from torch.testing._internal.common_methods_invocations import (
    BinaryUfuncInfo,
    op_db,
    ops_and_refs,
    python_ref_db,
    ReductionOpInfo,
    ReductionPythonRefInfo,
    skip,
    skipOps,
    SpectralFuncInfo,
    UnaryUfuncInfo,
    xfail,
)
from torch.testing._internal.common_utils import (
    clone_input_helper,
    first_sample,
    IS_CI,
    IS_FBCODE,
    is_iterable_of_tensors,
    IS_SANDCASTLE,
    noncontiguous_like,
    parametrize,
    run_tests,
    set_default_dtype,
    skipIfTorchDynamo,
    skipIfTorchInductor,
    slowTest,
    suppress_warnings,
    TEST_WITH_ROCM,
    TEST_WITH_TORCHDYNAMO,
    TEST_WITH_TORCHINDUCTOR,
    TestCase,
    unMarkDynamoStrictTest,
)
from torch.testing._internal.inductor_utils import maybe_skip_size_asserts
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map


assert torch.get_default_dtype() == torch.float32

# variant testing is only done with torch.float and torch.cfloat to avoid
#   excessive test times and maximize signal to noise ratio
_variant_ops = partial(
    ops, dtypes=OpDTypes.supported, allowed_dtypes=(torch.float, torch.cfloat)
)

# Get names of all the operators which have ref in their entry in OpInfo (testing infra)
#   except for elementwise unary operators (separately implemented in test/test_unary_ufuncs.py),
#   elementwise binary operators (separately implemented in test_binary_ufuncs.py),
#   reduction operations (separately implemented in test_reductions.py),
#   and Spectral Functions (separately implemented for only 1D as of now, in test/test_spectral_ops.py)
_ref_test_ops = tuple(
    filter(
        lambda op: not isinstance(
            op, (UnaryUfuncInfo, ReductionOpInfo, SpectralFuncInfo, BinaryUfuncInfo)
        )
        and op.ref is not None,
        op_db,
    )
)


def reduction_dtype_filter(op):
    if (
        not isinstance(op, ReductionPythonRefInfo)
        or not op.supports_out
        or torch.int16 not in op.dtypes
    ):
        return False
    return "dtype" in inspect.getfullargspec(op.op).kwonlyargs


def has_reduction_tag(op):
    """Check if an op has the reduction tag."""
    if not hasattr(torch.ops.aten, op.name):
        return False
    aten_op = getattr(torch.ops.aten, op.name)
    if not hasattr(aten_op, "default"):
        return False
    return torch.Tag.reduction in aten_op.default.tags


# Create a list of operators that are a subset of _ref_test_ops but don't have a
# numpy ref to compare them too, If both CPU and CUDA are compared to numpy
# then they do not need to be compared to each other
_ops_and_refs_with_no_numpy_ref = [op for op in ops_and_refs if op.ref is None]

aten = torch.ops.aten

meta_consistency_out_dtype_mismatch_xfails = {
    xfail("all"),
    xfail("amax"),
    xfail("amin"),
    xfail("aminmax"),
    xfail("any"),
    xfail("bucketize"),
    xfail("conj_physical"),
    xfail("cross"),
    xfail("cummax"),
    xfail("cummin"),
    xfail("diag"),
    xfail("fft.ihfft2"),
    xfail("fft.ihfftn"),
    xfail("frexp"),
    xfail("geqrf"),
    xfail("heaviside"),
    xfail("histc"),
    xfail("index_add"),
    xfail("index_copy"),
    xfail("index_select"),
    xfail("isin"),
    xfail("kthvalue"),
    xfail("lerp"),
    xfail("linalg.cross"),
    xfail("linalg.eigh"),
    xfail("linalg.eigvalsh"),
    xfail("linalg.ldl_factor"),
    xfail("linalg.ldl_factor_ex"),
    xfail("linalg.ldl_solve"),
    xfail("linalg.lu"),
    xfail("linalg.lu_factor"),
    xfail("linalg.lu_factor_ex"),
    xfail("linalg.lu_solve"),
    xfail("linalg.qr"),
    xfail("linalg.slogdet"),
    xfail("linalg.solve"),
    xfail("linalg.solve_ex"),
    xfail("linalg.solve_triangular"),
    xfail("logcumsumexp"),
    xfail("lu_solve"),
    xfail("lu_unpack"),
    xfail("mode"),
    xfail("msort"),
    xfail("multinomial"),
    xfail("nan_to_num"),
    xfail("native_batch_norm"),
    xfail("neg"),
    xfail("nn.functional.avg_pool3d"),
    xfail("nn.functional.gelu"),
    xfail("nn.functional.hardshrink"),
    xfail("nn.functional.logsigmoid"),
    xfail("nn.functional.softplus"),
    xfail("nn.functional.softshrink"),
    xfail("ormqr"),
    xfail("qr"),
    xfail("renorm"),
    xfail("round"),
    xfail("round", "decimals_0"),
    xfail("scatter_reduce", "amax"),
    xfail("scatter_reduce", "amin"),
    xfail("scatter_reduce", "mean"),
    xfail("scatter_reduce", "prod"),
    xfail("scatter_reduce", "sum"),
    xfail("searchsorted"),
    xfail("slice_scatter"),
    xfail("softmax"),
    xfail("sort"),
    xfail("sparse.sampled_addmm"),
    xfail("take"),
    xfail("tril"),
    xfail("triu"),
    xfail("unfold_copy"),
    # Output has dynamic shape.
    # Does not have a meta kernel implementation.
    skip("linalg.lstsq"),
}


# Tests that apply to all operators and aren't related to any particular
#   system
@unMarkDynamoStrictTest
class TestCommon(TestCase):
    exact_dtype = True

    # Verifies, on teardown, that no OpInfo is still using dynamic dtypes in CI
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

        if IS_CI:
            err_msg = (
                "The operator(s) below is(are) using dynamic_dtypes in the OpInfo entries."
                "This is OK for testing, but be sure to set the dtypes manually before landing your PR!"
            )
            # Assure no opinfo entry has dynamic_dtypes
            filtered_ops = list(filter(opinfo.utils.is_dynamic_dtype_set, op_db))
            for op in filtered_ops:
                fmt_str = opinfo.utils.str_format_dynamic_dtype(op)
                err_msg += "\n" + fmt_str

            assert len(filtered_ops) == 0, err_msg

    # Validates that each OpInfo works correctly on different CUDA devices
    @onlyOn(["cuda", "xpu"])
    @deviceCountAtLeast(2)
    @ops(op_db, allowed_dtypes=(torch.float32, torch.long))
    def test_multiple_devices(self, devices, dtype, op):
        for cuda_device_str in devices:
            cuda_device = torch.device(cuda_device_str)
            # NOTE: only tests on first sample
            samples = op.sample_inputs(cuda_device, dtype)
            sample = first_sample(self, samples)
            result = op(sample.input, *sample.args, **sample.kwargs)

            if isinstance(result, torch.Tensor):
                self.assertTrue(result.device == cuda_device)
            elif is_iterable_of_tensors(result):
                self.assertTrue(all(t.device == cuda_device for t in result))
            else:
                self.skipTest(
                    "Skipped! Only supports single tensor or iterable of tensor outputs."
                )

    def test_pointwise_tag_coverage(self):
        pytorch_dir = os.path.abspath(__file__ + "/../../")
        files = [
            "aten/src/ATen/native/UnaryOps.cpp",
            "aten/src/ATen/native/BinaryOps.cpp",
            "aten/src/ATen/native/PointwiseOps.cpp",
            "aten/src/ATen/native/TensorCompare.cpp",
        ]

        allowed_functions = (
            # reduction version of these operators
            "aten.max.default",
            "aten.max.dim",
            "aten.max.dim_max",
            "aten.max.names_dim",
            "aten.max.names_dim_max",
            "aten.max.unary_out",
            "aten.min.default",
            "aten.min.dim",
            "aten.min.dim_min",
            "aten.min.names_dim",
            "aten.min.names_dim_min",
            "aten.min.unary_out",
            # not pointwise
            "aten.isin.Tensor_Tensor",
            "aten.isin.Tensor_Tensor_out",
            "aten.isin.Tensor_Scalar",
            "aten.isin.Tensor_Scalar_out",
            "aten.isin.Scalar_Tensor",
            "aten.isin.Scalar_Tensor_out",
            "aten.mode.default",
            "aten.mode.dimname",
            "aten.mode.dimname_out",
            "aten.mode.values",
        )

        regex = re.compile(r"DEFINE_DISPATCH\(.*_stub")

        def get_opoverloadpacket_from_dispatch(kernel):
            if hasattr(torch.ops.aten, kernel):
                return kernel
            if hasattr(torch.ops.aten, f"__{kernel}__"):
                return f"__{kernel}__"
            if hasattr(torch.ops.aten, f"special_{kernel}"):
                return f"special_{kernel}"
            if "_" in kernel:
                kernel_split = kernel.split("_")
                new_kernel = "_".join(kernel_split[:-1])
                if hasattr(torch.ops.aten, new_kernel):
                    return new_kernel

            # could not find op from kernel dispatch string
            self.assertTrue(False)

        for file_name in files:
            with open(os.path.join(pytorch_dir, file_name)) as f:
                lines = f.read()
                matches = regex.findall(lines)
                for match in matches:
                    kernel = match[len("DEFINE_DISPATCH(") : -len("_stub")]

                    # no op definition for it, but defined with DEFINE_DISPATCH ?
                    if kernel == "trigamma":
                        continue

                    kernel = get_opoverloadpacket_from_dispatch(kernel)
                    overloadpacket = getattr(torch.ops.aten, kernel)

                    for overload_name in overloadpacket.overloads():
                        overload = getattr(overloadpacket, overload_name)

                        if not torch._C._dispatch_has_kernel(overload.name()):
                            continue

                        # TODO: tags are not propagated to generated overload,
                        # and there's no way of specifying them
                        if torch.Tag.generated in overload.tags:
                            continue

                        if str(overload) in allowed_functions:
                            continue

                        self.assertTrue(torch.Tag.pointwise in overload.tags)

    def test_reduction_tag_coverage(self):
        """Test that operators with reduction tag are from reduction operator files."""
        pytorch_dir = os.path.abspath(__file__ + "/../../")
        files = [
            "aten/src/ATen/native/ReduceOps.cpp",
            "aten/src/ATen/native/ReduceAllOps.h",
        ]

        # Operators that are not pure reduction but have reduction overloads
        allowed_functions = (
            # min/max have both elementwise (binary) and reduction versions
            "aten.min.other",
            "aten.min.out",
            "aten.max.other",
            "aten.max.out",
        )

        regex = re.compile(r"DEFINE_DISPATCH\(.*_stub")

        def get_opoverloadpacket_from_dispatch(kernel):
            # Skip cumulative operations - they're in ReduceOps.cpp but aren't reductions
            if kernel in ("cumsum", "cumprod", "logcumsumexp", "xor_sum"):
                return None

            # Special mappings for ambiguous kernel names
            if kernel == "and":
                return "all"
            if kernel == "or":
                return "any"

            if hasattr(torch.ops.aten, kernel):
                return kernel
            if hasattr(torch.ops.aten, f"__{kernel}__"):
                return f"__{kernel}__"
            if hasattr(torch.ops.aten, f"special_{kernel}"):
                return f"special_{kernel}"
            if "_" in kernel:
                kernel_split = kernel.split("_")
                new_kernel = "_".join(kernel_split[:-1])
                if hasattr(torch.ops.aten, new_kernel):
                    return new_kernel

            # could not find op from kernel dispatch string
            return None

        for file_name in files:
            file_path = os.path.join(pytorch_dir, file_name)
            if not os.path.exists(file_path):
                continue

            with open(file_path) as f:
                lines = f.read()
                matches = regex.findall(lines)
                for match in matches:
                    kernel = match[len("DEFINE_DISPATCH(") : -len("_stub")]

                    kernel = get_opoverloadpacket_from_dispatch(kernel)
                    if kernel is None:
                        continue

                    overloadpacket = getattr(torch.ops.aten, kernel)

                    for overload_name in overloadpacket.overloads():
                        overload = getattr(overloadpacket, overload_name)

                        if not torch._C._dispatch_has_kernel(overload.name()):
                            continue

                        # TODO: tags are not propagated to generated overload,
                        # and there's no way of specifying them
                        if torch.Tag.generated in overload.tags:
                            continue

                        if str(overload) in allowed_functions:
                            continue

                        self.assertTrue(
                            torch.Tag.reduction in overload.tags,
                            f"{overload} should have reduction tag",
                        )

    @ops([op for op in op_db if has_reduction_tag(op)], dtypes=OpDTypes.none)
    def test_reduction_ops_reduce(self, device, op):
        """Test that operators with reduction tag actually reduce numel when dim is specified."""
        samples = op.sample_inputs(device, torch.float32)

        for sample in samples:
            if "dim" not in sample.kwargs:
                continue

            dim_val = sample.kwargs["dim"]

            # Call the operation
            result = op(sample.input, *sample.args, **sample.kwargs)

            if isinstance(result, torch.Tensor):
                if dim_val is None:
                    dim_val = list(range(sample.input.ndim))
                reduction_dims = [dim_val] if isinstance(dim_val, int) else dim_val

                # Skip 0 dim for now
                if any(abs(dim) >= sample.input.ndim for dim in reduction_dims):
                    continue

                reduction_factor = 1
                for dim in reduction_dims:
                    reduction_factor *= sample.input.shape[dim]

                expected_numel = sample.input.numel() // reduction_factor

                self.assertEqual(
                    result.numel(),
                    expected_numel,
                    f"{op.name} with dim={dim_val} should reduce numel by factor of {reduction_factor} "
                    f"(input: {sample.input.numel()}, expected: {expected_numel}, got: {result.numel()})",
                )

    # Tests that the function and its (ndarray-accepting) reference produce the same
    #   values on the tensors from sample_inputs func for the corresponding op.
    # This test runs in double and complex double precision because
    # NumPy does computation internally using double precision for many functions
    # resulting in possible equality check failures.
    # skip windows case on CPU due to https://github.com/pytorch/pytorch/issues/129947
    # XPU test will be enabled step by step. Skip the tests temporarily.
    @skipXPU
    @onlyNativeDeviceTypesAnd(["hpu"])
    @suppress_warnings
    @ops(_ref_test_ops, allowed_dtypes=(torch.float64, torch.long, torch.complex128))
    def test_numpy_ref(self, device, dtype, op):
        if (
            TEST_WITH_TORCHINDUCTOR
            and op.formatted_name
            in ("signal_windows_exponential", "signal_windows_bartlett")
            and dtype == torch.float64
            and ("cuda" in device or "xpu" in device)
            or "cpu" in device
        ):  # noqa: E121
            raise unittest.SkipTest("XXX: raises tensor-likes are not close.")

        # Sets the default dtype to NumPy's default dtype of double
        with set_default_dtype(torch.double):
            for sample_input in op.reference_inputs(device, dtype):
                self.compare_with_reference(
                    op, op.ref, sample_input, exact_dtype=(dtype is not torch.long)
                )

    # Tests that the cpu and gpu results are consistent
    @onlyOn(["cuda", "xpu"])
    @suppress_warnings
    @slowTest
    @ops(_ops_and_refs_with_no_numpy_ref, dtypes=OpDTypes.any_common_cpu_cuda_one)
    def test_compare_cpu(self, device, dtype, op):
        def to_cpu(arg):
            if isinstance(arg, torch.Tensor):
                return arg.to(device="cpu")
            return arg

        samples = op.reference_inputs(device, dtype)

        for sample in samples:
            cpu_sample = sample.transform(to_cpu)
            cuda_results = op(sample.input, *sample.args, **sample.kwargs)
            cpu_results = op(cpu_sample.input, *cpu_sample.args, **cpu_sample.kwargs)

            # output_process_fn_grad has a very unfortunate name
            # We use this function in linalg extensively to postprocess the inputs of functions
            # that are not completely well-defined. Think svd and multiplying the singular vectors by -1.
            # CPU and CUDA implementations of the SVD can return valid SVDs that are different.
            # We use this function to compare them.
            cuda_results = sample.output_process_fn_grad(cuda_results)
            cpu_results = cpu_sample.output_process_fn_grad(cpu_results)

            # Lower tolerance because we are running this as a `@slowTest`
            # Don't want the periodic tests to fail frequently
            self.assertEqual(cuda_results, cpu_results, atol=1e-3, rtol=1e-3)

    # Tests that experimental Python References can propagate shape, dtype,
    # and device metadata properly.
    # See https://github.com/pytorch/pytorch/issues/78050 for a discussion of stride propagation.
    @skipXPU
    @onlyNativeDeviceTypesAnd(["hpu"])
    @ops(python_ref_db)
    @skipIfTorchInductor("Takes too long for inductor")
    def test_python_ref_meta(self, device, dtype, op):
        CHECK_CONJ_SKIPS = {
            torch._refs.linalg.svd,
        }

        with FakeTensorMode() as mode:
            pass

        def _to_tensormeta(x):
            if isinstance(x, torch.Tensor):
                out = FakeTensor.from_tensor(x, mode)
                return out
            return x

        # TODO: iterate over requires_grad true/false
        for sample in op.reference_inputs(device, dtype, requires_grad=False):
            result = op(sample.input, *sample.args, **sample.kwargs)

            meta_sample = sample.transform(_to_tensormeta)
            try:
                with mode:
                    meta_result = op(
                        meta_sample.input, *meta_sample.args, **meta_sample.kwargs
                    )
            except torch._subclasses.fake_tensor.UnsupportedFakeTensorException:
                continue
            except torch._subclasses.fake_tensor.DataDependentOutputException:
                continue
            except torch._subclasses.fake_tensor.UnsupportedOperatorException:
                continue

            if isinstance(result, torch.Tensor):
                self.assertTrue(isinstance(meta_result, FakeTensor))
                prims.utils.compare_tensor_meta(
                    result, meta_result, check_conj=op.op not in CHECK_CONJ_SKIPS
                )
            elif isinstance(result, Sequence):
                for a, b in zip(result, meta_result):
                    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
                        self.assertTrue(isinstance(b, FakeTensor))
                        prims.utils.compare_tensor_meta(
                            a, b, check_conj=op.op not in CHECK_CONJ_SKIPS
                        )

    def _ref_test_helper(
        self,
        ctx,
        device,
        dtype,
        op,
        skip_zero_numel=False,
        skip_zero_dim=False,
        skip_bfloat=False,
        skip_view_consistency=False,
    ):
        # NOTE: this test works by comparing the reference
        for sample in op.reference_inputs(device, dtype, requires_grad=False):
            ex = None
            if (
                isinstance(sample.input, torch.Tensor)
                and sample.input.numel() == 0
                and skip_zero_numel
            ):
                continue
            if (
                isinstance(sample.input, torch.Tensor)
                and sample.input.ndim == 0
                and skip_zero_dim
            ):
                continue

            if skip_bfloat and (
                (
                    isinstance(sample.input, torch.Tensor)
                    and sample.input.dtype == torch.bfloat16
                )
                or any(
                    isinstance(arg, torch.Tensor) and arg.dtype == torch.bfloat16
                    for arg in sample.args
                )
            ):
                continue
            with ctx():
                ref_result = op(sample.input, *sample.args, **sample.kwargs)
            torch_result = op.torch_opinfo(sample.input, *sample.args, **sample.kwargs)

            for a, b in zip(
                pytree.tree_leaves(ref_result), pytree.tree_leaves(torch_result)
            ):
                if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
                    prims.utils.compare_tensor_meta(a, b)
                    if (
                        getattr(op, "validate_view_consistency", True)
                        and not skip_view_consistency
                    ):
                        msg = (
                            f"The torch implementation {'returns' if b._is_view() else 'does not return'} "
                            f"a view, while the reference {'does' if a._is_view() else 'does not'}"
                        )
                        self.assertEqual(a._is_view(), b._is_view(), msg)

            # Computes the dtype the more precise computatino would occur in
            precise_dtype = torch.bool
            if prims.utils.is_integer_dtype(dtype):
                # Note: bool and integer dtypes do not have more
                # precise dtypes -- they simply must be close
                precise_dtype = dtype
            if prims.utils.is_float_dtype(dtype):
                precise_dtype = torch.double
            if prims.utils.is_complex_dtype(dtype):
                precise_dtype = torch.cdouble

            # Checks if the results are close
            try:
                self.assertEqual(
                    ref_result,
                    torch_result,
                    exact_stride=False,
                    exact_device=True,
                    exact_layout=True,
                    exact_is_coalesced=True,
                )
            except AssertionError as e:
                # Raises the error if the precise dtype comparison wouldn't be
                # different
                if dtype is precise_dtype:
                    raise e

                ex = e

            # Goes to next sample if these results are close
            if not ex:
                continue

            # If the results are not close, checks that the
            # reference is more accurate than the torch op
            def _make_precise(x):
                if isinstance(x, torch.dtype):
                    return precise_dtype
                if isinstance(x, torch.Tensor) and x.dtype is dtype:
                    return x.to(precise_dtype)
                return x

            precise_sample = sample.transform(_make_precise)
            precise_result = op.torch_opinfo(
                precise_sample.input, *precise_sample.args, **precise_sample.kwargs
            )

            def _distance(a, b):
                # Special-cases boolean comparisons
                if prims.utils.is_boolean_dtype(a.dtype):
                    assert b.dtype is torch.bool
                    return (a ^ b).sum()

                same = a == b
                if prims.utils.is_float_dtype(a.dtype) or prims.utils.is_complex_dtype(
                    a.dtype
                ):
                    same = torch.logical_or(
                        same, torch.logical_and(torch.isnan(a), torch.isnan(b))
                    )

                actual_error = torch.where(same, 0, torch.abs(a - b)).sum()
                return actual_error

            ref_distance = 0
            for a, b in zip(
                pytree.tree_leaves(ref_result), pytree.tree_leaves(precise_result)
            ):
                ref_distance = ref_distance + _distance(a, b)

            torch_distance = 0
            for a, b in zip(
                pytree.tree_leaves(torch_result), pytree.tree_leaves(precise_result)
            ):
                torch_distance = torch_distance + _distance(a, b)

            # TODO: consider adding some tolerance to this comparison
            msg = (
                f"Reference result was farther ({ref_distance}) from the precise "
                f"computation than the torch result was ({torch_distance})!"
            )
            self.assertTrue(ref_distance <= torch_distance, msg=msg)

        # Reports numerical accuracy discrepancies
        if ex is not None:
            msg = "Test passed because the reference was more accurate than the torch operator."
            warnings.warn(msg)

    # Tests that experimental Python References perform the same computation
    # as the operators they reference, when operator calls in the torch
    # namespace are remapped to the refs namespace (torch.foo becomes refs.foo).
    @skipXPU
    @onlyNativeDeviceTypesAnd(["hpu"])
    @ops(python_ref_db)
    @skipIfTorchInductor("Takes too long for inductor")
    def test_python_ref(self, device, dtype, op):
        # In this test, primTorch refs call into the refs namespace
        # For example, a ref with torch.foo in it will calls refs.foo instead
        # Direct calls to refs and prims are not affected
        if (
            TEST_WITH_ROCM
            and (op.name == "_refs.fft.ihfftn" or op.name == "_refs.fft.ihfft2")
            and dtype == torch.float16
        ):
            self.skipTest("Skipped on ROCm")
        self._ref_test_helper(lambda: TorchRefsMode(strict=True), device, dtype, op)

    # Tests that experimental Python References perform the same computation
    # as the operators they reference, when operator calls in the torch
    # namespace are preserved (torch.foo remains torch.foo).
    @skipXPU
    @onlyNativeDeviceTypesAnd(["hpu"])
    @ops(python_ref_db)
    @skipIfTorchInductor("Takes too long for inductor")
    def test_python_ref_torch_fallback(self, device, dtype, op):
        # In this test, refs call into the torch namespace (after the initial invocation)
        # For example, a ref with torch.foo in it will call torch.foo instead of refs.foo
        # Direct calls to refs and prims are not translated
        if TEST_WITH_ROCM and op.name == "_refs.fft.ihfftn" and dtype == torch.float16:
            self.skipTest("Skipped on ROCm")
        if op.full_name == "_refs.div.floor_rounding" and dtype == torch.bfloat16:
            self.skipTest(
                "Skipped _refs.div.floor_rounding with bfloat16"
                "Divide by 0: _refs produces NaN, torch produces +/-inf"
            )
        self._ref_test_helper(contextlib.nullcontext, device, dtype, op)

    @onlyCUDA
    @ops(python_ref_db)
    @parametrize("executor", ["aten"])
    @skipIfTorchInductor("Takes too long for inductor")
    def test_python_ref_executor(self, device, dtype, op, executor):
        if (
            TEST_WITH_ROCM
            and (op.name == "_refs.fft.ihfftn" or op.name == "_refs.fft.ihfft2")
            and dtype == torch.float16
        ):
            self.skipTest("Skipped on ROCm")
        from copy import copy

        from torch._prims.executor import make_traced

        op = copy(op)
        op.op = partial(make_traced(op.op), executor=executor)
        self._ref_test_helper(contextlib.nullcontext, device, dtype, op)

    @skipXPU
    @skipMeta
    @onlyNativeDeviceTypesAnd(["hpu"])
    @ops([op for op in op_db if op.error_inputs_func is not None], dtypes=OpDTypes.none)
    def test_errors(self, device, op):
        error_inputs = op.error_inputs(device)
        for ei in error_inputs:
            si = ei.sample_input
            with self.assertRaisesRegex(ei.error_type, ei.error_regex):
                out = op(si.input, *si.args, **si.kwargs)
                self.assertFalse(isinstance(out, type(NotImplemented)))

    @skipXPU
    @skipMeta
    @onlyNativeDeviceTypesAnd(["hpu"])
    @ops(
        [op for op in op_db if op.error_inputs_sparse_func is not None],
        dtypes=OpDTypes.none,
    )
    @parametrize(
        "layout",
        (
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
            torch.sparse_coo,
        ),
    )
    def test_errors_sparse(self, device, op, layout):
        for ei in op.error_inputs_sparse(device, layout):
            si = ei.sample_input
            with self.assertRaisesRegex(ei.error_type, ei.error_regex):
                out = op(si.input, *si.args, **si.kwargs)
                self.assertFalse(isinstance(out, type(NotImplemented)))

    @skipXPU
    @skipMeta
    @onlyNativeDeviceTypesAnd(["hpu"])
    @ops(
        [op for op in python_ref_db if op.error_inputs_func is not None],
        dtypes=OpDTypes.none,
    )
    @skipIfTorchInductor("Takes too long for inductor")
    def test_python_ref_errors(self, device, op):
        mode = FakeTensorMode()
        with mode:
            pass

        def _to_tensormeta(x):
            if isinstance(x, torch.Tensor):
                return FakeTensor.from_tensor(x, mode)
            return x

        error_inputs = op.error_inputs(device)
        for ei in error_inputs:
            si = ei.sample_input
            meta_sample = si.transform(_to_tensormeta)
            with self.assertRaisesRegex(ei.error_type, ei.error_regex):
                op(meta_sample.input, *meta_sample.args, **meta_sample.kwargs)

    # Tests that the function produces the same result when called with
    #   noncontiguous tensors.
    @skipXPU
    @with_tf32_off
    @onlyNativeDeviceTypesAnd(["hpu"])
    @suppress_warnings
    @ops(op_db, allowed_dtypes=(torch.float32, torch.long, torch.complex64))
    def test_noncontiguous_samples(self, device, dtype, op):
        test_grad = dtype in op.supported_backward_dtypes(torch.device(device).type)
        sample_inputs = op.sample_inputs(device, dtype, requires_grad=test_grad)
        for sample_input in sample_inputs:
            t_inp, t_args, t_kwargs = (
                sample_input.input,
                sample_input.args,
                sample_input.kwargs,
            )
            noncontig_sample = sample_input.noncontiguous()
            n_inp, n_args, n_kwargs = (
                noncontig_sample.input,
                noncontig_sample.args,
                noncontig_sample.kwargs,
            )

            # validates forward
            expected = op(t_inp, *t_args, **t_kwargs)
            actual = op(n_inp, *n_args, **n_kwargs)

            self.assertEqual(actual, expected)

            # Validate backward
            # Short-circuits if the op doesn't support grad in this device x dtype
            if not test_grad:
                continue

            expected = sample_input.output_process_fn_grad(expected)
            actual = sample_input.output_process_fn_grad(actual)

            if isinstance(expected, torch.Tensor):
                grad_for_expected = torch.randn_like(expected)
                grad_for_actual = noncontiguous_like(grad_for_expected)
            elif isinstance(expected, Sequence):
                # Filter output elements that do not require grad
                expected = [
                    t
                    for t in expected
                    if isinstance(t, torch.Tensor) and t.requires_grad
                ]
                actual = [
                    n for n in actual if isinstance(n, torch.Tensor) and n.requires_grad
                ]
                grad_for_expected = [torch.randn_like(t) for t in expected]
                grad_for_actual = [noncontiguous_like(n) for n in grad_for_expected]
            else:
                # Nothing to do if it returns a scalar or things like that
                continue

            # Concatenate inputs into a tuple
            t_inputs = (
                (t_inp,) + t_args
                if isinstance(t_inp, torch.Tensor)
                else tuple(t_inp) + t_args
            )
            n_inputs = (
                (n_inp,) + n_args
                if isinstance(n_inp, torch.Tensor)
                else tuple(n_inp) + n_args
            )

            # Filter the elements that are tensors that require grad
            t_input_tensors = [
                t for t in t_inputs if isinstance(t, torch.Tensor) and t.requires_grad
            ]
            n_input_tensors = [
                n for n in n_inputs if isinstance(n, torch.Tensor) and n.requires_grad
            ]

            self.assertEqual(len(t_input_tensors), len(n_input_tensors))

            # Some functions may not use all the inputs to generate gradients. One of the
            # few examples of this "odd" behaviour is F.hinge_embedding_loss
            t_grads = torch.autograd.grad(
                expected, t_input_tensors, grad_for_expected, allow_unused=True
            )
            n_grads = torch.autograd.grad(
                actual, n_input_tensors, grad_for_actual, allow_unused=True
            )

            msg = "Got different gradients for contiguous / non-contiguous inputs wrt input {}."
            for i, (t, n) in enumerate(zip(t_grads, n_grads)):
                self.assertEqual(t, n, msg=msg.format(i))

    # Separates one case from the following test_out because many ops don't properly implement the
    #   incorrectly sized out parameter warning properly yet
    # Cases test here:
    #   - out= with the correct dtype and device, but the wrong shape
    @skipXPU
    @ops(ops_and_refs, dtypes=OpDTypes.none)
    def test_out_warning(self, device, op):
        if TEST_WITH_TORCHDYNAMO and op.name == "_refs.clamp":
            self.skipTest("flaky")
        # Prefers running in float32 but has a fallback for the first listed supported dtype
        supported_dtypes = op.supported_dtypes(self.device_type)
        if len(supported_dtypes) == 0:
            self.skipTest("Skipped! Op has not supported dtypes on this device.")
        dtype = (
            torch.float32
            if torch.float32 in supported_dtypes
            else next(iter(supported_dtypes))
        )

        # Ops from python_ref_db point to python decomps that are potentially
        # wrapped with `torch._prims_common.wrappers.out_wrapper`. Unwrap these
        # ops before testing to avoid clashing with OpInfo.supports_out
        if not op.supports_out:
            op = copy.copy(op)
            op.op = _maybe_remove_out_wrapper(op.op)

        samples = op.sample_inputs(device, dtype)
        for sample in samples:
            # calls it normally to get the expected result
            expected = op(sample.input, *sample.args, **sample.kwargs)
            op_out = partial(op, sample.input, *sample.args, **sample.kwargs)

            # Short-circuits if output is not a single tensor or an
            #   iterable of tensors
            if not isinstance(expected, torch.Tensor) and not is_iterable_of_tensors(
                expected, include_empty=True
            ):
                self.skipTest(
                    "Skipped! Only supports single tensor or iterable of tensor outputs."
                )

            # Validates the op doesn't support out if it claims not to
            if not op.supports_out:
                with self.assertRaises(Exception):
                    assert op_out(out=expected) != NotImplemented
                return

            # A wrapper around map that works with single tensors and always
            #   instantiates the map. Used below to apply transforms to
            #   single tensor and iterable tensor outputs.
            def _apply_out_transform(fn, out):
                if isinstance(out, torch.Tensor):
                    return fn(out)

                # assumes (see above) that out is an iterable of tensors
                return tuple(map(fn, out))

            # Extracts strides from a tensor or iterable of tensors into a tuple
            def _extract_strides(out):
                if isinstance(out, torch.Tensor):
                    return (out.stride(),)

                # assumes (see above) that out is an iterable of tensors
                return tuple(t.stride() for t in out)

            # Extracts data pointers from a tensor or iterable of tensors into a tuple
            # NOTE: only extracts on the CPU and CUDA device types since some
            #   device types don't have storage
            def _extract_data_ptrs(out):
                if self.device_type != "cpu" and self.device_type != "cuda":
                    return ()

                if isinstance(out, torch.Tensor):
                    return (out.data_ptr(),)

                # assumes (see above) that out is an iterable of tensors
                return tuple(t.data_ptr() for t in out)

            @suppress_warnings
            def _compare_out(transform, *, compare_strides_and_data_ptrs=True):
                out = _apply_out_transform(transform, expected)
                original_strides = _extract_strides(out)
                original_ptrs = _extract_data_ptrs(out)

                op_out(out=out)
                final_strides = _extract_strides(out)
                final_ptrs = _extract_data_ptrs(out)

                self.assertEqual(expected, out)

                if compare_strides_and_data_ptrs:
                    stride_msg = (
                        f"Strides are not the same! Original strides were {original_strides} "
                        f"and strides are now {final_strides}"
                    )
                    self.assertEqual(original_strides, final_strides, msg=stride_msg)
                    self.assertEqual(original_ptrs, final_ptrs)

            # Case Zero: out= with the correct dtype and device, but the wrong shape
            #   Expected behavior: if nonempty, resize with a warning.
            def _case_zero_transform(t):
                wrong_shape = list(t.shape)

                if len(wrong_shape) == 0:
                    # Handles scalar tensor case (empty list)
                    wrong_shape = [2]
                else:
                    wrong_shape[-1] = wrong_shape[-1] + 1
                return make_tensor(wrong_shape, dtype=t.dtype, device=t.device)

            # Verifies the out values are correct
            _compare_out(_case_zero_transform, compare_strides_and_data_ptrs=False)

            # Additionally validates that the appropriate warning is thrown if a nonempty
            #   tensor is resized.
            def _any_nonempty(out):
                if isinstance(out, torch.Tensor):
                    return out.numel() > 0

                return any(x.numel() > 0 for x in out)

            out = _apply_out_transform(_case_zero_transform, expected)
            msg_fail = "Resized a non-empty tensor but did not warn about it."
            if _any_nonempty(out):
                with self.assertWarnsRegex(
                    UserWarning, "An output with one or more elements", msg=msg_fail
                ):
                    op_out(out=out)

    # Validates ops implement the correct out= behavior
    # See https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-does-out-work-in-pytorch
    #   for a description of the correct behavior
    # Validates the following cases:
    #   - Case 0: out has the correct shape, dtype, and device but is full of extremal values
    #   - Case 1: out has the correct shape, dtype, and device but is noncontiguous
    #   - Case 2: out has the correct dtype and device, but is zero elements
    #   - Case 3: out has the correct shape and dtype, but is on a different device type
    #   - Case 4: out has the correct shape and device, but a dtype that cannot
    #       "safely" cast to
    #
    # Case 3 and 4 are slightly different when the op is a factory function:
    #   - if device, dtype are NOT passed, any combination of dtype/device should be OK for out
    #   - if device, dtype are passed, device and dtype should match
    @skipXPU
    @ops(ops_and_refs, dtypes=OpDTypes.any_one)
    def test_out(self, device, dtype, op):
        # Prefers running in float32 but has a fallback for the first listed supported dtype
        samples = op.sample_inputs(device, dtype)

        # Ops from python_ref_db point to python decomps that are potentially
        # wrapped with `torch._prims_common.wrappers.out_wrapper`. Unwrap these
        # ops before testing to avoid clashing with OpInfo.supports_out
        if not op.supports_out:
            op = copy.copy(op)
            op.op = _maybe_remove_out_wrapper(op.op)

        for sample in samples:
            # calls it normally to get the expected result
            expected = op(sample.input, *sample.args, **sample.kwargs)
            op_out = partial(op, sample.input, *sample.args, **sample.kwargs)

            # Short-circuits if output is not a single tensor or an
            #   iterable of tensors
            if not isinstance(expected, torch.Tensor) and not is_iterable_of_tensors(
                expected, include_empty=True
            ):
                self.skipTest(
                    "Skipped! Only supports single tensor or iterable of tensor outputs."
                )

            # Validates the op doesn't support out if it claims not to
            if not op.supports_out:
                with self.assertRaises(Exception):
                    assert op_out(out=expected) != NotImplemented
                return

            # A wrapper around map that works with single tensors and always
            #   instantiates the map. Used below to apply transforms to
            #   single tensor and iterable tensor outputs.
            def _apply_out_transform(fn, out):
                if isinstance(out, torch.Tensor):
                    return fn(out)

                # assumes (see above) that out is an iterable of tensors
                return tuple(map(fn, out))

            # Extracts strides from a tensor or iterable of tensors into a tuple
            def _extract_strides(out):
                if isinstance(out, torch.Tensor):
                    return (out.stride(),)

                # assumes (see above) that out is an iterable of tensors
                return tuple(t.stride() for t in out)

            # Extracts data pointers from a tensor or iterable of tensors into a tuple
            # NOTE: only extracts on the CPU and CUDA device types since some
            #   device types don't have storage
            def _extract_data_ptrs(out):
                if self.device_type != "cpu" and self.device_type != "cuda":
                    return ()

                if isinstance(out, torch.Tensor):
                    return (out.data_ptr(),)

                # assumes (see above) that out is an iterable of tensors
                return tuple(t.data_ptr() for t in out)

            def _compare_out(transform, *, compare_strides_and_data_ptrs=True):
                out = _apply_out_transform(transform, expected)
                original_strides = _extract_strides(out)
                original_ptrs = _extract_data_ptrs(out)

                op_out(out=out)
                final_strides = _extract_strides(out)
                final_ptrs = _extract_data_ptrs(out)
                self.assertEqual(expected, out)

                if compare_strides_and_data_ptrs:
                    stride_msg = (
                        "Strides are not the same! "
                        f"Original strides were {original_strides} and strides are now {final_strides}"
                    )
                    self.assertEqual(original_strides, final_strides, msg=stride_msg)
                    self.assertEqual(original_ptrs, final_ptrs)

            # Case 0: out= with the correct shape, dtype, and device
            #   but NaN values for floating point and complex tensors, and
            #   maximum values for integer tensors.
            #   Expected behavior: out= values have no effect on the computation.
            def _case_zero_transform(t):
                try:
                    info = torch.iinfo(t.dtype)
                    return torch.full_like(t, info.max)
                except TypeError:
                    # for non-integer types fills with NaN
                    return torch.full_like(t, float("nan"))

            _compare_out(_case_zero_transform)

            # Case 1: out= with the correct shape, dtype, and device,
            #   but noncontiguous.
            #   Expected behavior: strides are respected and `out` storage is not changed.
            def _case_one_transform(t):
                return make_tensor(
                    t.shape, dtype=t.dtype, device=t.device, noncontiguous=True
                )

            _compare_out(_case_one_transform)

            # Case 2: out= with the correct dtype and device, but has no elements.
            #   Expected behavior: resize without warning.
            def _case_two_transform(t):
                return make_tensor((0,), dtype=t.dtype, device=t.device)

            _compare_out(_case_two_transform, compare_strides_and_data_ptrs=False)

            # Also validates that no warning is thrown when this out is resized
            out = _apply_out_transform(_case_two_transform, expected)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                op_out(out=out)

            # Verifies no warning is a resize warning
            for w in caught:
                if "An output with one or more elements" in str(w.message):
                    self.fail(
                        "Resizing an out= argument with no elements threw a resize warning!"
                    )

            # Case 3: out= with correct shape and dtype, but wrong device.
            #   Expected behavior: throws an error.
            #   This case is ignored on CPU to allow some scalar operations to succeed.
            factory_fn_msg = (
                "\n\nNOTE: If your op is a factory function (i.e., it accepts TensorOptions) you should mark its "
                "OpInfo with `is_factory_function=True`."
            )

            if torch.device(device).type != "cpu":
                wrong_device = "cpu"

                def _case_three_transform(t):
                    return make_tensor(t.shape, dtype=t.dtype, device=wrong_device)

                out = _apply_out_transform(_case_three_transform, expected)

                if op.is_factory_function and sample.kwargs.get("device", None) is None:
                    op_out(out=out)
                else:
                    msg_fail = (
                        f"Expected RuntimeError when calling with input.device={device} and out.device={wrong_device}."
                    ) + factory_fn_msg
                    with self.assertRaises(RuntimeError, msg=msg_fail):
                        op_out(out=out)

            # Case 4: out= with correct shape and device, but a dtype
            #   that output cannot be "safely" cast to (long).
            #   Expected behavior: error.
            # NOTE: this case is filtered by dtype since some ops produce
            #   bool tensors, for example, which can be safely cast to any
            #   dtype. It is applied when single tensors are floating point or complex
            #   dtypes, or if an op returns multiple tensors when at least one such
            #   tensor is a floating point or complex dtype.
            _dtypes = floating_and_complex_types_and(torch.float16, torch.bfloat16)
            if (
                isinstance(expected, torch.Tensor)
                and expected.dtype in _dtypes
                or (
                    not isinstance(expected, torch.Tensor)
                    and any(t.dtype in _dtypes for t in expected)
                )
            ):

                def _case_four_transform(t):
                    return make_tensor(t.shape, dtype=torch.long, device=t.device)

                out = _apply_out_transform(_case_four_transform, expected)
                msg_fail = "Expected RuntimeError when doing an unsafe cast!"
                msg_fail = (
                    msg_fail
                    if not isinstance(expected, torch.Tensor)
                    else (
                        "Expected RuntimeError when doing an unsafe cast from a result of dtype "
             
```



## High-Level Overview


This Python file contains 10 class(es) and 96 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestCommon`, `TestCompositeCompliance`, `TestMathBits`, `_TestTagsMode`, `TestTags`, `TestSelfKwarg`, `TestRefsOpsInfo`, `TestFakeTensor`, `TestPointwiseMode`, `TestForwardADWithScalars`

**Functions defined**: `reduction_dtype_filter`, `has_reduction_tag`, `tearDownClass`, `test_multiple_devices`, `test_pointwise_tag_coverage`, `get_opoverloadpacket_from_dispatch`, `test_reduction_tag_coverage`, `get_opoverloadpacket_from_dispatch`, `test_reduction_ops_reduce`, `test_numpy_ref`, `test_compare_cpu`, `to_cpu`, `test_python_ref_meta`, `_to_tensormeta`, `_ref_test_helper`, `_make_precise`, `_distance`, `test_python_ref`, `test_python_ref_torch_fallback`, `test_python_ref_executor`

**Key imports**: contextlib, copy, inspect, itertools, os, re, unittest, warnings, defaultdict, Sequence


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `copy`
- `inspect`
- `itertools`
- `os`
- `re`
- `unittest`
- `warnings`
- `collections`: defaultdict
- `collections.abc`: Sequence
- `functools`: partial
- `importlib`: import_module
- `torch`
- `torch._prims as prims`
- `torch.utils._pytree as pytree`
- `torch._prims.context`: TorchRefsMode
- `torch._prims_common.wrappers`: _maybe_remove_out_wrapper
- `torch._subclasses.fake_tensor`: FakeTensor, FakeTensorMode
- `torch._subclasses.fake_utils`: outputs_alias_inputs
- `torch.testing`: make_tensor
- `torch.testing._internal`: composite_compliance, opinfo
- `torch.testing._internal.common_cuda`: with_tf32_off
- `torch.testing._internal.inductor_utils`: maybe_skip_size_asserts
- `torch.utils._python_dispatch`: TorchDispatchMode
- `torch.utils._pytree`: tree_map
- `torch._prims.executor`: make_traced
- `torch.fx.experimental.symbolic_shapes`: ShapeEnv


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/test_ops.py
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

- **File Documentation**: `test_ops.py_docs.md`
- **Keyword Index**: `test_ops.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
