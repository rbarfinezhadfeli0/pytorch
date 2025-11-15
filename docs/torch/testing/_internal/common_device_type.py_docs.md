# Documentation: `torch/testing/_internal/common_device_type.py`

## File Metadata

- **Path**: `torch/testing/_internal/common_device_type.py`
- **Size**: 74,778 bytes (73.03 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# mypy: ignore-errors

import copy
import gc
import inspect
import os
import runpy
import sys
import threading
import unittest
from collections import namedtuple
from collections.abc import Callable, Iterable, Sequence
from enum import Enum
from functools import partial, wraps
from typing import Any, ClassVar, Optional, TypeVar, Union
from typing_extensions import ParamSpec

import torch
from torch._inductor.utils import GPU_TYPES
from torch.testing._internal.common_cuda import (
    _get_torch_cuda_version,
    _get_torch_rocm_version,
    TEST_CUSPARSE_GENERIC,
    TEST_HIPSPARSE_GENERIC,
)
from torch.testing._internal.common_dtype import get_all_dtypes
from torch.testing._internal.common_utils import (
    _TestParametrizer,
    clear_tracked_input,
    compose_parametrize_fns,
    dtype_name,
    get_tracked_input,
    IS_FBCODE,
    IS_MACOS,
    is_privateuse1_backend_available,
    IS_REMOTE_GPU,
    IS_S390X,
    IS_SANDCASTLE,
    IS_WINDOWS,
    NATIVE_DEVICES,
    PRINT_REPRO_ON_FAILURE,
    skipCUDANonDefaultStreamIf,
    skipIfTorchDynamo,
    TEST_HPU,
    TEST_MKL,
    TEST_MPS,
    TEST_WITH_ASAN,
    TEST_WITH_MIOPEN_SUGGEST_NHWC,
    TEST_WITH_MTIA,
    TEST_WITH_ROCM,
    TEST_WITH_TORCHINDUCTOR,
    TEST_WITH_TSAN,
    TEST_WITH_UBSAN,
    TEST_XPU,
    TestCase,
)


_T = TypeVar("_T")
_P = ParamSpec("_P")

try:
    import psutil  # type: ignore[import]

    HAS_PSUTIL = True
except ModuleNotFoundError:
    HAS_PSUTIL = False
    psutil = None

# Note [Writing Test Templates]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This note was written shortly after the PyTorch 1.9 release.
# If you notice it's out-of-date or think it could be improved then please
# file an issue.
#
# PyTorch has its own framework for instantiating test templates. That is, for
#   taking test classes that look similar to unittest or pytest
#   compatible test classes and optionally doing the following:
#
#     - instantiating a version of the test class for each available device type
#         (often the CPU, CUDA, and META device types)
#     - further instantiating a version of each test that's always specialized
#         on the test class's device type, and optionally specialized further
#         on datatypes or operators
#
# This functionality is similar to pytest's parametrize functionality
#   (see https://docs.pytest.org/en/6.2.x/parametrize.html), but with considerable
#   additional logic that specializes the instantiated test classes for their
#   device types (see CPUTestBase and CUDATestBase below), supports a variety
#   of composable decorators that allow for test filtering and setting
#   tolerances, and allows tests parametrized by operators to instantiate
#   only the subset of device type x dtype that operator supports.
#
# This framework was built to make it easier to write tests that run on
#   multiple device types, multiple datatypes (dtypes), and for multiple
#   operators. It's also useful for controlling which tests are run. For example,
#   only tests that use a CUDA device can be run on platforms with CUDA.
#   Let's dive in with an example to get an idea for how it works:
#
# --------------------------------------------------------
# A template class (looks like a regular unittest TestCase)
# class TestClassFoo(TestCase):
#
#   # A template test that can be specialized with a device
#   # NOTE: this test case is not runnable by unittest or pytest because it
#   #   accepts an extra positional argument, "device", that they do not understand
#   def test_bar(self, device):
#     pass
#
# # Function that instantiates a template class and its tests
# instantiate_device_type_tests(TestCommon, globals())
# --------------------------------------------------------
#
# In the above code example we see a template class and a single test template
#   that can be instantiated with a device. The function
#   instantiate_device_type_tests(), called at file scope, instantiates
#   new test classes, one per available device type, and new tests in those
#   classes from these templates. It actually does this by removing
#   the class TestClassFoo and replacing it with classes like TestClassFooCPU
#   and TestClassFooCUDA, instantiated test classes that inherit from CPUTestBase
#   and CUDATestBase respectively. Additional device types, like XLA,
#   (see https://github.com/pytorch/xla) can further extend the set of
#   instantiated test classes to create classes like TestClassFooXLA.
#
# The test template, test_bar(), is also instantiated. In this case the template
#   is only specialized on a device, so (depending on the available device
#   types) it might become test_bar_cpu() in TestClassFooCPU and test_bar_cuda()
#   in TestClassFooCUDA. We can think of the instantiated test classes as
#   looking like this:
#
# --------------------------------------------------------
# # An instantiated test class for the CPU device type
# class TestClassFooCPU(CPUTestBase):
#
#   # An instantiated test that calls the template with the string representation
#   #   of a device from the test class's device type
#   def test_bar_cpu(self):
#     test_bar(self, 'cpu')
#
# # An instantiated test class for the CUDA device type
# class TestClassFooCUDA(CUDATestBase):
#
#   # An instantiated test that calls the template with the string representation
#   #   of a device from the test class's device type
#   def test_bar_cuda(self):
#     test_bar(self, 'cuda:0')
# --------------------------------------------------------
#
# These instantiated test classes ARE discoverable and runnable by both
#   unittest and pytest. One thing that may be confusing, however, is that
#   attempting to run "test_bar" will not work, despite it appearing in the
#   original template code. This is because "test_bar" is no longer discoverable
#   after instantiate_device_type_tests() runs, as the above snippet shows.
#   Instead "test_bar_cpu" and "test_bar_cuda" may be run directly, or both
#   can be run with the option "-k test_bar".
#
# Removing the template class and adding the instantiated classes requires
#   passing "globals()" to instantiate_device_type_tests(), because it
#   edits the file's Python objects.
#
# As mentioned, tests can be additionally parametrized on dtypes or
#   operators. Datatype parametrization uses the @dtypes decorator and
#   require a test template like this:
#
# --------------------------------------------------------
# # A template test that can be specialized with a device and a datatype (dtype)
# @dtypes(torch.float32, torch.int64)
# def test_car(self, device, dtype)
#   pass
# --------------------------------------------------------
#
# If the CPU and CUDA device types are available this test would be
#   instantiated as 4 tests that cover the cross-product of the two dtypes
#   and two device types:
#
#     - test_car_cpu_float32
#     - test_car_cpu_int64
#     - test_car_cuda_float32
#     - test_car_cuda_int64
#
# The dtype is passed as a torch.dtype object.
#
# Tests parametrized on operators (actually on OpInfos, more on that in a
#   moment...) use the @ops decorator and require a test template like this:
# --------------------------------------------------------
# # A template test that can be specialized with a device, dtype, and OpInfo
# @ops(op_db)
# def test_car(self, device, dtype, op)
#   pass
# --------------------------------------------------------
#
# See the documentation for the @ops decorator below for additional details
#   on how to use it and see the note [OpInfos] in
#   common_methods_invocations.py for more details on OpInfos.
#
# A test parametrized over the entire "op_db", which contains hundreds of
#   OpInfos, will likely have hundreds or thousands of instantiations. The
#   test will be instantiated on the cross-product of device types, operators,
#   and the dtypes the operator supports on that device type. The instantiated
#   tests will have names like:
#
#     - test_car_add_cpu_float32
#     - test_car_sub_cuda_int64
#
# The first instantiated test calls the original test_car() with the OpInfo
#   for torch.add as its "op" argument, the string 'cpu' for its "device" argument,
#   and the dtype torch.float32 for is "dtype" argument. The second instantiated
#   test calls the test_car() with the OpInfo for torch.sub, a CUDA device string
#   like 'cuda:0' or 'cuda:1' for its "device" argument, and the dtype
#   torch.int64 for its "dtype argument."
#
# In addition to parametrizing over device, dtype, and ops via OpInfos, the
#   @parametrize decorator is supported for arbitrary parametrizations:
# --------------------------------------------------------
# # A template test that can be specialized with a device, dtype, and value for x
# @parametrize("x", range(5))
# def test_car(self, device, dtype, x)
#   pass
# --------------------------------------------------------
#
# See the documentation for @parametrize in common_utils.py for additional details
#   on this. Note that the instantiate_device_type_tests() function will handle
#   such parametrizations; there is no need to additionally call
#   instantiate_parametrized_tests().
#
# Clever test filtering can be very useful when working with parametrized
#   tests. "-k test_car" would run every instantiated variant of the test_car()
#   test template, and "-k test_car_add" runs every variant instantiated with
#   torch.add.
#
# It is important to use the passed device and dtype as appropriate. Use
#   helper functions like make_tensor() that require explicitly specifying
#   the device and dtype so they're not forgotten.
#
# Test templates can use a variety of composable decorators to specify
#   additional options and requirements, some are listed here:
#
#     - @deviceCountAtLeast(<minimum number of devices to run test with>)
#         Passes a list of strings representing all available devices of
#         the test class's device type as the test template's "device" argument.
#         If there are fewer devices than the value passed to the decorator
#         the test is skipped.
#     - @dtypes(<list of tuples of dtypes>)
#         In addition to accepting multiple dtypes, the @dtypes decorator
#         can accept a sequence of tuple pairs of dtypes. The test template
#         will be called with each tuple for its "dtype" argument.
#     - @onlyNativeDeviceTypes
#         Skips the test if the device is not a native device type (currently CPU, CUDA, Meta)
#     - @onlyCPU
#         Skips the test if the device is not a CPU device
#     - @onlyCUDA
#         Skips the test if the device is not a CUDA device
#     - @onlyMPS
#         Skips the test if the device is not a MPS device
#     - @skipCPUIfNoLapack
#         Skips the test if the device is a CPU device and LAPACK is not installed
#     - @skipCPUIfNoMkl
#         Skips the test if the device is a CPU device and MKL is not installed
#     - @skipCUDAIfNoMagma
#         Skips the test if the device is a CUDA device and MAGMA is not installed
#     - @skipCUDAIfRocm
#         Skips the test if the device is a CUDA device and ROCm is being used


# Note [Adding a Device Type]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To add a device type:
#
#   (1) Create a new "TestBase" extending DeviceTypeTestBase.
#       See CPUTestBase and CUDATestBase below.
#   (2) Define the "device_type" attribute of the base to be the
#       appropriate string.
#   (3) Add logic to this file that appends your base class to
#       device_type_test_bases when your device type is available.
#   (4) (Optional) Write setUpClass/tearDownClass class methods that
#       instantiate dependencies (see MAGMA in CUDATestBase).
#   (5) (Optional) Override the "instantiate_test" method for total
#       control over how your class creates tests.
#
# setUpClass is called AFTER tests have been created and BEFORE and ONLY IF
# they are run. This makes it useful for initializing devices and dependencies.


def _dtype_test_suffix(dtypes):
    """Returns the test suffix for a dtype, sequence of dtypes, or None."""
    if isinstance(dtypes, (list, tuple)):
        if len(dtypes) == 0:
            return ""
        return "_" + "_".join(dtype_name(d) for d in dtypes)
    elif dtypes:
        return f"_{dtype_name(dtypes)}"
    else:
        return ""


def _update_param_kwargs(param_kwargs, name, value):
    """Adds a kwarg with the specified name and value to the param_kwargs dict."""
    # Make name plural (e.g. devices / dtypes) if the value is composite.
    plural_name = f"{name}s"

    # Clear out old entries of the arg if any.
    if name in param_kwargs:
        del param_kwargs[name]
    if plural_name in param_kwargs:
        del param_kwargs[plural_name]

    if isinstance(value, (list, tuple)):
        param_kwargs[plural_name] = value
    elif value is not None:
        param_kwargs[name] = value

    # Leave param_kwargs as-is when value is None.


class DeviceTypeTestBase(TestCase):
    device_type: str = "generic_device_type"

    # Flag to disable test suite early due to unrecoverable error such as CUDA error.
    _stop_test_suite = False

    # Precision is a thread-local setting since it may be overridden per test
    _tls = threading.local()
    _tls.precision = TestCase._precision
    _tls.rel_tol = TestCase._rel_tol

    @property
    def precision(self):
        return self._tls.precision

    @precision.setter
    def precision(self, prec):
        self._tls.precision = prec

    @property
    def rel_tol(self):
        return self._tls.rel_tol

    @rel_tol.setter
    def rel_tol(self, prec):
        self._tls.rel_tol = prec

    # Returns a string representing the device that single device tests should use.
    # Note: single device tests use this device exclusively.
    @classmethod
    def get_primary_device(cls):
        return cls.device_type

    @classmethod
    def _init_and_get_primary_device(cls):
        try:
            return cls.get_primary_device()
        except Exception:
            # For CUDATestBase, XPUTestBase, XLATestBase, and possibly others, the primary device won't be available
            # until setUpClass() sets it. Call that manually here if needed.
            if hasattr(cls, "setUpClass"):
                cls.setUpClass()
            return cls.get_primary_device()

    # Returns a list of strings representing all available devices of this
    # device type. The primary device must be the first string in the list
    # and the list must contain no duplicates.
    # Note: UNSTABLE API. Will be replaced once PyTorch has a device generic
    #   mechanism of acquiring all available devices.
    @classmethod
    def get_all_devices(cls):
        return [cls.get_primary_device()]

    # Returns the dtypes the test has requested.
    # Prefers device-specific dtype specifications over generic ones.
    @classmethod
    def _get_dtypes(cls, test):
        if not hasattr(test, "dtypes"):
            return None

        default_dtypes = test.dtypes.get("all")
        msg = f"@dtypes is mandatory when using @dtypesIf however '{test.__name__}' didn't specify it"
        assert default_dtypes is not None, msg

        return test.dtypes.get(cls.device_type, default_dtypes)

    def _get_precision_override(self, test, dtype):
        if not hasattr(test, "precision_overrides"):
            return self.precision
        return test.precision_overrides.get(dtype, self.precision)

    def _get_tolerance_override(self, test, dtype):
        if not hasattr(test, "tolerance_overrides"):
            return self.precision, self.rel_tol
        return test.tolerance_overrides.get(dtype, tol(self.precision, self.rel_tol))

    def _apply_precision_override_for_test(self, test, param_kwargs):
        dtype = param_kwargs.get("dtype")
        dtype = param_kwargs.get("dtypes", dtype)
        if dtype:
            self.precision = self._get_precision_override(test, dtype)
            self.precision, self.rel_tol = self._get_tolerance_override(test, dtype)

    # Creates device-specific tests.
    @classmethod
    def instantiate_test(cls, name, test, *, generic_cls=None):
        def instantiate_test_helper(
            cls, name, *, test, param_kwargs=None, decorator_fn=lambda _: []
        ):
            # Add the device param kwarg if the test needs device or devices.
            param_kwargs = {} if param_kwargs is None else param_kwargs
            test_sig_params = inspect.signature(test).parameters
            if "device" in test_sig_params or "devices" in test_sig_params:
                device_arg: str = cls._init_and_get_primary_device()
                if hasattr(test, "num_required_devices"):
                    device_arg = cls.get_all_devices()
                _update_param_kwargs(param_kwargs, "device", device_arg)

            # Apply decorators based on param kwargs.
            for decorator in decorator_fn(param_kwargs):
                test = decorator(test)

            # Constructs the test
            @wraps(test)
            def instantiated_test(self, param_kwargs=param_kwargs):
                # Sets precision and runs test
                # Note: precision is reset after the test is run
                guard_precision = self.precision
                guard_rel_tol = self.rel_tol
                try:
                    self._apply_precision_override_for_test(test, param_kwargs)
                    result = test(self, **param_kwargs)
                except RuntimeError as rte:
                    # check if rte should stop entire test suite.
                    self._stop_test_suite = self._should_stop_test_suite()
                    # Check if test has been decorated with `@expectedFailure`
                    # Using `__unittest_expecting_failure__` attribute, see
                    # https://github.com/python/cpython/blob/ffa505b580464/Lib/unittest/case.py#L164
                    # In that case, make it fail with "unexpected success" by suppressing exception
                    if (
                        getattr(test, "__unittest_expecting_failure__", False)
                        and self._stop_test_suite
                    ):
                        import sys

                        print(
                            "Suppressing fatal exception to trigger unexpected success",
                            file=sys.stderr,
                        )
                        return
                    # raise the runtime error as is for the test suite to record.
                    raise rte
                finally:
                    self.precision = guard_precision
                    self.rel_tol = guard_rel_tol

                return result

            assert not hasattr(cls, name), f"Redefinition of test {name}"
            setattr(cls, name, instantiated_test)

        def default_parametrize_fn(test, generic_cls, device_cls):
            # By default, no parametrization is needed.
            yield (test, "", {}, lambda _: [])

        # Parametrization decorators set the parametrize_fn attribute on the test.
        parametrize_fn = getattr(test, "parametrize_fn", default_parametrize_fn)

        # If one of the @dtypes* decorators is present, also parametrize over the dtypes set by it.
        dtypes = cls._get_dtypes(test)
        if dtypes is not None:

            def dtype_parametrize_fn(test, generic_cls, device_cls, dtypes=dtypes):
                for dtype in dtypes:
                    param_kwargs: dict[str, Any] = {}
                    _update_param_kwargs(param_kwargs, "dtype", dtype)

                    # Note that an empty test suffix is set here so that the dtype can be appended
                    # later after the device.
                    yield (test, "", param_kwargs, lambda _: [])

            parametrize_fn = compose_parametrize_fns(
                dtype_parametrize_fn, parametrize_fn
            )

        # Instantiate the parametrized tests.
        for (
            test,  # noqa: B020
            test_suffix,
            param_kwargs,
            decorator_fn,
        ) in parametrize_fn(test, generic_cls, cls):
            test_suffix = "" if test_suffix == "" else "_" + test_suffix
            cls_device_type = (
                cls.device_type
                if cls.device_type != "privateuse1"
                else torch._C._get_privateuse1_backend_name()
            )
            device_suffix = "_" + cls_device_type

            # Note: device and dtype suffix placement
            # Special handling here to place dtype(s) after device according to test name convention.
            dtype_kwarg = None
            if "dtype" in param_kwargs or "dtypes" in param_kwargs:
                dtype_kwarg = (
                    param_kwargs["dtypes"]
                    if "dtypes" in param_kwargs
                    else param_kwargs["dtype"]
                )
            test_name = (
                f"{name}{test_suffix}{device_suffix}{_dtype_test_suffix(dtype_kwarg)}"
            )

            instantiate_test_helper(
                cls=cls,
                name=test_name,
                test=test,
                param_kwargs=param_kwargs,
                decorator_fn=decorator_fn,
            )

    def run(self, result=None):
        super().run(result=result)
        # Early terminate test if _stop_test_suite is set.
        if self._stop_test_suite:
            result.stop()


class CPUTestBase(DeviceTypeTestBase):
    device_type = "cpu"

    # No critical error should stop CPU test suite
    def _should_stop_test_suite(self):
        return False


class CUDATestBase(DeviceTypeTestBase):
    device_type = "cuda"
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True
    primary_device: ClassVar[str]
    cudnn_version: ClassVar[Any]
    no_magma: ClassVar[bool]
    no_cudnn: ClassVar[bool]

    def has_cudnn(self):
        return not self.no_cudnn

    @classmethod
    def get_primary_device(cls):
        return cls.primary_device

    @classmethod
    def get_all_devices(cls):
        primary_device_idx = int(cls.get_primary_device().split(":")[1])
        num_devices = torch.cuda.device_count()

        prim_device = cls.get_primary_device()
        cuda_str = "cuda:{0}"
        non_primary_devices = [
            cuda_str.format(idx)
            for idx in range(num_devices)
            if idx != primary_device_idx
        ]
        return [prim_device] + non_primary_devices

    @classmethod
    def setUpClass(cls):
        # has_magma shows up after cuda is initialized
        t = torch.ones(1).cuda()
        cls.no_magma = not torch.cuda.has_magma

        # Determines if cuDNN is available and its version
        cls.no_cudnn = not torch.backends.cudnn.is_acceptable(t)
        cls.cudnn_version = None if cls.no_cudnn else torch.backends.cudnn.version()

        # Acquires the current device as the primary (test) device
        cls.primary_device = f"cuda:{torch.cuda.current_device()}"


# See Note [Lazy Tensor tests in device agnostic testing]
lazy_ts_backend_init = False


class LazyTestBase(DeviceTypeTestBase):
    device_type = "lazy"

    def _should_stop_test_suite(self):
        return False

    @classmethod
    def setUpClass(cls):
        import torch._lazy
        import torch._lazy.metrics
        import torch._lazy.ts_backend

        global lazy_ts_backend_init
        if not lazy_ts_backend_init:
            # Need to connect the TS backend to lazy key before running tests
            torch._lazy.ts_backend.init()
            lazy_ts_backend_init = True


class MPSTestBase(DeviceTypeTestBase):
    device_type = "mps"
    primary_device: ClassVar[str]

    @classmethod
    def get_primary_device(cls):
        return cls.primary_device

    @classmethod
    def get_all_devices(cls):
        # currently only one device is supported on MPS backend
        prim_device = cls.get_primary_device()
        return [prim_device]

    @classmethod
    def setUpClass(cls):
        cls.primary_device = "mps:0"

    def _should_stop_test_suite(self):
        return False


class XPUTestBase(DeviceTypeTestBase):
    device_type = "xpu"
    primary_device: ClassVar[str]

    @classmethod
    def get_primary_device(cls):
        return cls.primary_device

    @classmethod
    def get_all_devices(cls):
        # currently only one device is supported on MPS backend
        primary_device_idx = int(cls.get_primary_device().split(":")[1])
        num_devices = torch.xpu.device_count()

        prim_device = cls.get_primary_device()
        xpu_str = "xpu:{0}"
        non_primary_devices = [
            xpu_str.format(idx)
            for idx in range(num_devices)
            if idx != primary_device_idx
        ]
        return [prim_device] + non_primary_devices

    @classmethod
    def setUpClass(cls):
        cls.primary_device = f"xpu:{torch.xpu.current_device()}"

    def _should_stop_test_suite(self):
        return False


class HPUTestBase(DeviceTypeTestBase):
    device_type = "hpu"
    primary_device: ClassVar[str]

    @classmethod
    def get_primary_device(cls):
        return cls.primary_device

    @classmethod
    def setUpClass(cls):
        cls.primary_device = "hpu:0"


class PrivateUse1TestBase(DeviceTypeTestBase):
    primary_device: ClassVar[str]
    device_mod = None
    device_type = "privateuse1"

    @classmethod
    def get_primary_device(cls):
        return cls.primary_device

    @classmethod
    def get_all_devices(cls):
        primary_device_idx = int(cls.get_primary_device().split(":")[1])
        num_devices = cls.device_mod.device_count()
        prim_device = cls.get_primary_device()
        device_str = f"{cls.device_type}:{{0}}"
        non_primary_devices = [
            device_str.format(idx)
            for idx in range(num_devices)
            if idx != primary_device_idx
        ]
        return [prim_device] + non_primary_devices

    @classmethod
    def setUpClass(cls):
        cls.device_type = torch._C._get_privateuse1_backend_name()
        cls.device_mod = getattr(torch, cls.device_type, None)
        assert (
            cls.device_mod is not None
        ), f"""torch has no module of `{cls.device_type}`, you should register
                                            a module by `torch._register_device_module`."""
        cls.primary_device = f"{cls.device_type}:{cls.device_mod.current_device()}"


# Adds available device-type-specific test base classes
def get_device_type_test_bases():
    # set type to List[Any] due to mypy list-of-union issue:
    # https://github.com/python/mypy/issues/3351
    test_bases: list[Any] = []

    if IS_SANDCASTLE or IS_FBCODE:
        if IS_REMOTE_GPU:
            # Skip if sanitizer is enabled or we're on MTIA machines
            if (
                not TEST_WITH_ASAN
                and not TEST_WITH_TSAN
                and not TEST_WITH_UBSAN
                and not TEST_WITH_MTIA
            ):
                test_bases.append(CUDATestBase)
        else:
            test_bases.append(CPUTestBase)
    else:
        test_bases.append(CPUTestBase)
        if torch.cuda.is_available():
            test_bases.append(CUDATestBase)

        if is_privateuse1_backend_available():
            test_bases.append(PrivateUse1TestBase)
        # Disable MPS testing in generic device testing temporarily while we're
        # ramping up support.
        # elif torch.backends.mps.is_available():
        #   test_bases.append(MPSTestBase)

    return test_bases


device_type_test_bases = get_device_type_test_bases()


def filter_desired_device_types(device_type_test_bases, except_for=None, only_for=None):
    # device type cannot appear in both except_for and only_for
    intersect = set(except_for if except_for else []) & set(
        only_for if only_for else []
    )
    assert not intersect, (
        f"device ({intersect}) appeared in both except_for and only_for"
    )

    # Replace your privateuse1 backend name with 'privateuse1'
    if is_privateuse1_backend_available():
        privateuse1_backend_name = torch._C._get_privateuse1_backend_name()

        def func_replace(x: str):
            return x.replace(privateuse1_backend_name, "privateuse1")

        except_for = (
            ([func_replace(x) for x in except_for] if except_for is not None else None)
            if not isinstance(except_for, str)
            else func_replace(except_for)
        )
        only_for = (
            ([func_replace(x) for x in only_for] if only_for is not None else None)
            if not isinstance(only_for, str)
            else func_replace(only_for)
        )

    if except_for:
        device_type_test_bases = filter(
            lambda x: x.device_type not in except_for, device_type_test_bases
        )
    if only_for:
        device_type_test_bases = filter(
            lambda x: x.device_type in only_for, device_type_test_bases
        )

    return list(device_type_test_bases)


# Note [How to extend DeviceTypeTestBase to add new test device]
# The following logic optionally allows downstream projects like pytorch/xla to
# add more test devices.
# Instructions:
#  - Add a python file (e.g. pytorch/xla/test/pytorch_test_base.py) in downstream project.
#    - Inside the file, one should inherit from `DeviceTypeTestBase` class and define
#      a new DeviceTypeTest class (e.g. `XLATestBase`) with proper implementation of
#      `instantiate_test` method.
#    - DO NOT import common_device_type inside the file.
#      `runpy.run_path` with `globals()` already properly setup the context so that
#      `DeviceTypeTestBase` is already available.
#    - Set a top-level variable `TEST_CLASS` equal to your new class.
#      E.g. TEST_CLASS = XLATensorBase
#  - To run tests with new device type, set `TORCH_TEST_DEVICE` env variable to path
#    to this file. Multiple paths can be separated by `:`.
# See pytorch/xla/test/pytorch_test_base.py for a more detailed example.
_TORCH_TEST_DEVICES = os.environ.get("TORCH_TEST_DEVICES", None)
if _TORCH_TEST_DEVICES:
    for path in _TORCH_TEST_DEVICES.split(":"):
        # runpy (a stdlib module) lacks annotations
        mod = runpy.run_path(path, init_globals=globals())  # type: ignore[func-returns-value]
        device_type_test_bases.append(mod["TEST_CLASS"])


PYTORCH_CUDA_MEMCHECK = os.getenv("PYTORCH_CUDA_MEMCHECK", "0") == "1"

PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY = "PYTORCH_TESTING_DEVICE_ONLY_FOR"
PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY = "PYTORCH_TESTING_DEVICE_EXCEPT_FOR"
PYTORCH_TESTING_DEVICE_FOR_CUSTOM_KEY = "PYTORCH_TESTING_DEVICE_FOR_CUSTOM"


def get_desired_device_type_test_bases(
    except_for=None, only_for=None, include_lazy=False, allow_mps=False, allow_xpu=False
):
    # allow callers to specifically opt tests into being tested on MPS, similar to `include_lazy`
    test_bases = device_type_test_bases.copy()
    if allow_mps and TEST_MPS and MPSTestBase not in test_bases:
        test_bases.append(MPSTestBase)
    if allow_xpu and TEST_XPU and XPUTestBase not in test_bases:
        test_bases.append(XPUTestBase)
    if TEST_HPU and HPUTestBase not in test_bases:
        test_bases.append(HPUTestBase)
    # Filter out the device types based on user inputs
    desired_device_type_test_bases = filter_desired_device_types(
        test_bases, except_for, only_for
    )
    if include_lazy:
        # Note [Lazy Tensor tests in device agnostic testing]
        # Right now, test_view_ops.py runs with LazyTensor.
        # We don't want to opt every device-agnostic test into using the lazy device,
        # because many of them will fail.
        # So instead, the only way to opt a specific device-agnostic test file into
        # lazy tensor testing is with include_lazy=True
        if IS_FBCODE:
            print(
                "TorchScript backend not yet supported in FBCODE/OVRSOURCE builds",
                file=sys.stderr,
            )
        else:
            desired_device_type_test_bases.append(LazyTestBase)

    def split_if_not_empty(x: str):
        return x.split(",") if x else []

    # run some cuda testcases on other devices if available
    # Usage:
    # export PYTORCH_TESTING_DEVICE_FOR_CUSTOM=privateuse1
    env_custom_only_for = split_if_not_empty(
        os.getenv(PYTORCH_TESTING_DEVICE_FOR_CUSTOM_KEY, "")
    )
    if env_custom_only_for:
        desired_device_type_test_bases += filter(
            lambda x: x.device_type in env_custom_only_for, test_bases
        )
        desired_device_type_test_bases = list(set(desired_device_type_test_bases))

    # Filter out the device types based on environment variables if available
    # Usage:
    # export PYTORCH_TESTING_DEVICE_ONLY_FOR=cuda,cpu
    # export PYTORCH_TESTING_DEVICE_EXCEPT_FOR=xla
    env_only_for = split_if_not_empty(
        os.getenv(PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY, "")
    )
    env_except_for = split_if_not_empty(
        os.getenv(PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY, "")
    )

    return filter_desired_device_types(
        desired_device_type_test_bases, env_except_for, env_only_for
    )


# Adds 'instantiated' device-specific test cases to the given scope.
# The tests in these test cases are derived from the generic tests in
# generic_test_class. This function should be used instead of
# instantiate_parametrized_tests() if the test class contains
# device-specific tests (NB: this supports additional @parametrize usage).
#
# See note "Writing Test Templates"
# TODO: remove "allow_xpu" option after Interl GPU support all test case instantiate by this function.
def instantiate_device_type_tests(
    generic_test_class,
    scope,
    except_for=None,
    only_for=None,
    include_lazy=False,
    allow_mps=False,
    allow_xpu=False,
):
    # Removes the generic test class from its enclosing scope so its tests
    # are not discoverable.
    del scope[generic_test_class.__name__]

    generic_members = set(generic_test_class.__dict__.keys())
    generic_tests = [x for x in generic_members if x.startswith("test")]

    # Creates device-specific test cases
    for base in get_desired_device_type_test_bases(
        except_for, only_for, include_lazy, allow_mps, allow_xpu
    ):
        class_name = generic_test_class.__name__ + base.device_type.upper()

        # type set to Any and suppressed due to unsupported runtime class:
        # https://github.com/python/mypy/wiki/Unsupported-Python-Features
        device_type_test_class: Any = type(class_name, (base, generic_test_class), {})

        # Arrange for setUpClass and tearDownClass methods defined both in the test template
        # class and in the generic base to be called. This allows device-parameterized test
        # classes to support setup and teardown.
        # NB: This should be done before instantiate_test() is called as that invokes setup.
        @classmethod
        def _setUpClass(cls):
            # This should always be called, whether or not the test class invokes
            # super().setUpClass(), to set the primary device.
            base.setUpClass()
            # We want to call the @classmethod defined in the generic base, but pass
            # it the device-specific class object (cls), hence the __func__ call.
            generic_test_class.setUpClass.__func__(cls)

        @classmethod
        def _tearDownClass(cls):
            # We want to call the @classmethod defined in the generic base, but pass
            # it the device-specific class object (cls), hence the __func__ call.
            generic_test_class.tearDownClass.__func__(cls)
            base.tearDownClass()

        device_type_test_class.setUpClass = _setUpClass
        device_type_test_class.tearDownClass = _tearDownClass

        for name in generic_members:
            if name in generic_tests:  # Instantiates test member
                test = getattr(generic_test_class, name)
                # XLA-compat shim (XLA's instantiate_test takes doesn't take generic_cls)
                sig = inspect.signature(device_type_test_class.instantiate_test)
                if len(sig.parameters) == 3:
                    # Instantiates the device-specific tests
                    device_type_test_class.instantiate_test(
                        name, copy.deepcopy(test), generic_cls=generic_test_class
                    )
                else:
                    device_type_test_class.instantiate_test(name, copy.deepcopy(test))
            # Ports non-test member. Setup / teardown have already been handled above
            elif name not in device_type_test_class.__dict__:
                nontest = getattr(generic_test_class, name)
                setattr(device_type_test_class, name, nontest)

        # Mimics defining the instantiated class in the caller's file
        # by setting its module to the given class's and adding
        # the module to the given scope.
        # This lets the instantiated class be discovered by unittest.
        device_type_test_class.__module__ = generic_test_class.__module__
        scope[class_name] = device_type_test_class

    # Delete the generic form of the test functions (e.g. TestFoo.test_bar()) so they're
    # not discoverable. This mutates the original class (TestFoo), which was removed from
    # scope above. At this point, device-specific tests (e.g. TestFooCUDA.test_bar_cuda)
    # have already been created and the generic forms are no longer needed.
    for name in generic_tests:
        delattr(generic_test_class, name)


# Category of dtypes to run an OpInfo-based test for
# Example use: @ops(dtype=OpDTypes.supported)
#
# There are 7 categories:
# - supported: Every dtype supported by the operator. Use for exhaustive
#              testing of all dtypes.
# - unsupported: Run tests on dtypes not supported by the operator. e.g. for
#                testing the operator raises an error and doesn't crash.
# - supported_backward: Every dtype supported by the operator's backward pass.
# - unsupported_backward: Run tests on dtypes not supported by the operator's backward pass.
# - any_one: Runs a test for one dtype the operator supports. Prioritizes dtypes the
#     operator supports in both forward and backward.
# - none: Useful for tests that are not dtype-specific. No dtype will be passed to the test
#         when this is selected.
# - any_common_cpu_cuda_one: Pick a dtype that supports both CPU and CUDA.
class OpDTypes(Enum):
    supported = 0  # Test all supported dtypes (default)
    unsupported = 1  # Test only unsupported dtypes
    supported_backward = 2  # Test all supported backward dtypes
    unsupported_backward = 3  # Test only unsupported backward dtypes
    any_one = 4  # Test precisely one supported dtype
    none = 5  # Instantiate no dtype variants (no dtype kwarg needed)
    any_common_cpu_cuda_one = (
        6  # Test precisely one supported dtype that is common to both cuda and cpu
    )


# Arbitrary order
ANY_DTYPE_ORDER = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.float16,
    torch.bfloat16,
    torch.long,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)


def _serialize_sample(sample_input):
    # NB: For OpInfos, SampleInput.summary() prints in a cleaner way.
    if getattr(sample_input, "summary", None) is not None:
        return sample_input.summary()
    return str(sample_input)


# Decorator that defines the OpInfos a test template should be instantiated for.
#
# Example usage:
#
# @ops(unary_ufuncs)
# def test_numerics(self, device, dtype, op):
#   <test_code>
#
# This will instantiate variants of test_numerics for each given OpInfo,
# on each device the OpInfo's operator supports, and for every dtype supported by
# that operator. There are a few caveats to the dtype rule, explained below.
#
# The @ops decorator can accept two
# additional arguments, "dtypes" and "allowed_dtypes". If "dtypes" is specified
# then the test variants are instantiated for those dtypes, regardless of
# what the operator supports. If given "allowed_dtypes" then test variants
# are instantiated only for the intersection of allowed_dtypes and the dtypes
# they would otherwise be instantiated with. That is, allowed_dtypes composes
# with the options listed above and below.
#
# The "dtypes" argument can also accept additional values (see OpDTypes above):
#   OpDTypes.supported - the test is instantiated for all dtypes the operator
#     supports
#   OpDTypes.unsupported - the test is instantiated for all dtypes the operator
#     doesn't support
#   OpDTypes.supported_backward - the test is instantiated for all dtypes the
#     operator's gradient formula supports
#   OpDTypes.unsupported_backward - the test is instantiated for all dtypes the
#     operator's gradient formula doesn't support
#   OpDTypes.any_one - the test is instantiated for one dtype the
#     operator supports. The dtype supports forward and backward if possible.
#   OpDTypes.none - the test is instantiated without any dtype. The test signature
#     should not include a dtype kwarg in this case.
#   OpDTypes.any_common_cpu_cuda_one - the test is instantiated for a dtype
#     that supports both CPU and CUDA.
#
# These options allow tests to have considerable control over the dtypes
#   they're instantiated for.


class ops(_TestParametrizer):
    def __init__(
        self,
        op_list,
        *,
        dtypes: Union[OpDTypes, Sequence[torch.dtype]] = OpDTypes.supported,
        allowed_dtypes: Optional[Sequence[torch.dtype]] = None,
        skip_if_dynamo=True,
    ):
        self.op_list = list(op_list)
        self.opinfo_dtypes = dtypes
        self.allowed_dtypes = (
            set(allowed_dtypes) if allowed_dtypes is not None else None
        )
        self.skip_if_dynamo = skip_if_dynamo

    def _parametrize_test(self, test, generic_cls, device_cls):
        """Parameterizes the given test function across each op and its associated dtypes."""
        if device_cls is None:
            raise RuntimeError(
                "The @ops decorator is only intended to be used in a device-specific "
                "context; use it with instantiate_device_type_tests() instead of "
                "instantiate_parametrized_tests()"
            )

        op = check_exhausted_iterator = object()
        for op in self.op_list:
            # Determine the set of dtypes to use.
            dtypes: Union[set[torch.dtype], set[None]]
            if isinstance(self.opinfo_dtypes, Sequence):
                dtypes = set(self.opinfo_dtypes)
            elif self.opinfo_dtypes == OpDTypes.unsupported_backward:
                dtypes = set(get_all_dtypes()).difference(
                    op.supported_backward_dtypes(device_cls.device_type)
                )
            elif self.opinfo_dtypes == OpDTypes.supported_backward:
                dtypes = op.supported_backward_dtypes(device_cls.device_type)
            elif self.opinfo_dtypes == OpDTypes.unsupported:
                dtypes = set(get_all_dtypes()).difference(
                    op.supported_dtypes(device_cls.device_type)
                )
            elif self.opinfo_dtypes == OpDTypes.supported:
                dtypes = set(op.supported_dtypes(device_cls.device_type))
            elif self.opinfo_dtypes == OpDTypes.any_one:
                # Tries to pick a dtype that supports both forward or backward
                supported = op.supported_dtypes(device_cls.device_type)
                supported_backward = op.supported_backward_dtypes(
                    device_cls.device_type
                )
                supported_both = supported.intersection(supported_backward)
                dtype_set = supported_both if len(supported_both) > 0 else supported
                for dtype in ANY_DTYPE_ORDER:
                    if dtype in dtype_set:
                        dtypes = {dtype}
                        break
                else:
                    dtypes = {}
            elif self.opinfo_dtypes == OpDTypes.any_common_cpu_cuda_one:
                # Tries to pick a dtype that supports both CPU and CUDA
                supported = set(op.dtypes).intersection(op.dtypesIfCUDA)
                if supported:
                    dtypes = {
                        next(dtype for dtype in ANY_DTYPE_ORDER if dtype in supported)
                    }
                else:
                    dtypes = {}

            elif self.opinfo_dtypes == OpDTypes.none:
                dtypes = {None}
            else:
                raise RuntimeError(f"Unknown OpDType: {self.opinfo_dtypes}")

            if self.allowed_dtypes is not None:
                dtypes = dtypes.intersection(self.allowed_dtypes)

            # Construct the test name; device / dtype parts are handled outside.
            # See [Note: device and dtype suffix placement]
            test_name = op.formatted_name

            # Filter sample skips / xfails to only those that apply to the OpInfo.
            # These are defined on the test function via decorators.
            sample_skips_and_xfails = getattr(test, "sample_skips_and_xfails", None)
            if sample_skips_and_xfails is not None:
                sample_skips_and_xfails = [
                    rule
                    for rule in sample_skips_and_xfails
                    if rule.op_match_fn(device_cls.device_type, op)
                ]

            for dtype in dtypes:
                # Construct parameter kwargs to pass to the test.
                param_kwargs = {"op": op}
                _update_param_kwargs(param_kwargs, "dtype", dtype)

                # NOTE: test_wrapper exists because we don't want to apply
                #   op-specific decorators to the original test.
                #   Test-specific decorators are applied to the original test,
                #   however.
                try:

                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        try:
                            return test(*args, **kwargs)
                        except unittest.SkipTest as e:
                            raise e
                        except Exception as e:
                            tracked_input = get_tracked_input()
                            if PRINT_REPRO_ON_FAILURE and tracked_input is not None:
                                e_tracked = Exception(  # noqa: TRY002
                                    f"{str(e)}\n\nCaused by {tracked_input.type_desc} "
                                    f"at index {tracked_input.index}: "
                                    f"{_serialize_sample(tracked_input.val)}"
                                )
                                e_tracked._tracked_input = tracked_input  # type: ignore[attr]
                                raise e_tracked from e
                            raise e
                        finally:
                            clear_tracked_input()

                    if self.skip_if_dynamo and not TEST_WITH_TORCHINDUCTOR:
                        test_wrapper = skipIfTorchDynamo(
                            "Policy: we don't run OpInfo tests w/ Dynamo"
                        )(test_wrapper)

                    # Initialize info for the last input seen. This is useful for tracking
                    # down which inputs caused a test failure. Note that TrackedInputIter is
                    # responsible for managing this.
                    test.tracked_input = None

                    decorator_fn = partial(
                        op.get_decorators,
                        generic_cls.__name__,
                        test.__name__,
                        device_cls.device_type,
                        dtype,
                    )

                    if sample_skips_and_xfails is not None:
                        test_wrapper.sample_skips_and_xfails = sample_skips_and_xfails

                    yield (test_wrapper, test_name, param_kwargs, decorator_fn)
                except Exception as ex:
                    # Provides an error message for debugging before rethrowing the exception
                    print(f"Failed to instantiate {test_name} for op {op.name}!")
                    raise ex
        if op is check_exhausted_iterator:
            raise ValueError(
                "An empty op_list was passed to @ops. "
                "Note that this may result from reuse of a generator."
            )


# Decorator that skips a test if the given condition is true.
# Notes:
#   (1) Skip conditions stack.
#   (2) Skip conditions can be bools or strings. If a string the
#       test base must have defined the corresponding attribute to be False
#       for the test to run. If you want to use a string argument you should
#       probably define a new decorator instead (see below).
#   (3) Prefer the existing decorators to defining the 'device_type' kwarg.
class skipIf:
    def __init__(self, dep, reason, device_type=None):
        self.dep = dep
        self.reason = reason
        self.device_type = device_type

    def __call__(self, fn):
        @wraps(fn)
        def dep_fn(slf, *args, **kwargs):
            if (
                self.device_type is None
                or self.device_type == slf.device_type
                or (
                    isinstance(self.device_type, Iterable)
                    and slf.device_type in self.device_type
                )
            ):
                if (isinstance(self.dep, str) and getattr(slf, self.dep, True)) or (
                    isinstance(self.dep, bool) and self.dep
                ):
                    raise unittest.SkipTest(self.reason)

            return fn(slf, *args, **kwargs)

        return dep_fn


# Skips a test on CPU if the condition is true.
class skipCPUIf(skipIf):
    def __init__(self, dep, reason):
        super().__init__(dep, reason, device_type="cpu")


# Skips a test on CUDA if the condition is
```



## High-Level Overview


This Python file contains 55 class(es) and 166 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestClassFoo`, `TestClassFooCPU`, `TestClassFooCUDA`, `DeviceTypeTestBase`, `CPUTestBase`, `CUDATestBase`, `LazyTestBase`, `MPSTestBase`, `XPUTestBase`, `HPUTestBase`, `PrivateUse1TestBase`, `OpDTypes`, `ops`, `skipIf`, `skipCPUIf`, `skipCUDAIf`, `skipXPUIf`, `skipGPUIf`, `skipLazyIf`, `skipMetaIf`

**Functions defined**: `test_bar`, `test_bar_cpu`, `test_bar_cuda`, `test_car`, `test_car`, `test_car`, `_dtype_test_suffix`, `_update_param_kwargs`, `precision`, `precision`, `rel_tol`, `rel_tol`, `get_primary_device`, `_init_and_get_primary_device`, `get_all_devices`, `_get_dtypes`, `_get_precision_override`, `_get_tolerance_override`, `_apply_precision_override_for_test`, `instantiate_test`

**Key imports**: copy, gc, inspect, os, runpy, sys, threading, unittest, namedtuple, Callable, Iterable, Sequence


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/testing/_internal`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `gc`
- `inspect`
- `os`
- `runpy`
- `sys`
- `threading`
- `unittest`
- `collections`: namedtuple
- `collections.abc`: Callable, Iterable, Sequence
- `enum`: Enum
- `functools`: partial, wraps
- `typing`: Any, ClassVar, Optional, TypeVar, Union
- `typing_extensions`: ParamSpec
- `torch`
- `torch._inductor.utils`: GPU_TYPES
- `torch.testing._internal.common_dtype`: get_all_dtypes
- `psutil  `
- `torch._lazy`
- `torch._lazy.metrics`
- `torch._lazy.ts_backend`
- `common_device_type inside the file.`
- `platform`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling


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
python torch/testing/_internal/common_device_type.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/testing/_internal`):

- [`common_jit.py_docs.md`](./common_jit.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`autograd_function_db.py_docs.md`](./autograd_function_db.py_docs.md)
- [`custom_op_db.py_docs.md`](./custom_op_db.py_docs.md)
- [`subclasses.py_docs.md`](./subclasses.py_docs.md)
- [`logging_utils.py_docs.md`](./logging_utils.py_docs.md)
- [`two_tensor.py_docs.md`](./two_tensor.py_docs.md)
- [`autocast_test_lists.py_docs.md`](./autocast_test_lists.py_docs.md)
- [`hypothesis_utils.py_docs.md`](./hypothesis_utils.py_docs.md)
- [`common_mkldnn.py_docs.md`](./common_mkldnn.py_docs.md)


## Cross-References

- **File Documentation**: `common_device_type.py_docs.md`
- **Keyword Index**: `common_device_type.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
