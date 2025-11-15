# Documentation: `docs/torch/testing/_internal/common_utils.py_docs.md`

## File Metadata

- **Path**: `docs/torch/testing/_internal/common_utils.py_docs.md`
- **Size**: 54,370 bytes (53.10 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/testing/_internal/common_utils.py`

## File Metadata

- **Path**: `torch/testing/_internal/common_utils.py`
- **Size**: 243,371 bytes (237.67 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# mypy: allow-untyped-defs

r"""Importing this file must **not** initialize CUDA context. test_distributed
relies on this assumption to properly run. This means that when this is imported
no CUDA calls shall be made, including torch.cuda.device_count(), etc.

torch.testing._internal.common_cuda.py can freely initialize CUDA context when imported.
"""

import argparse
import contextlib
import copy
import ctypes
import errno
import functools
import gc
import hashlib
import inspect
import io
import json
import logging
import math
import operator
import os
import pathlib
import platform
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import warnings
from collections.abc import Mapping, Sequence
from contextlib import closing, contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from itertools import product, chain
from pathlib import Path
from statistics import mean
from typing import (
    Any,
    Optional,
    TypeVar,
    Union,
)
from collections.abc import Callable
from collections.abc import Iterable, Iterator
from unittest.mock import MagicMock

import expecttest
import numpy as np

import __main__  # type: ignore[import]
import torch
import torch.backends.cudnn
import torch.backends.mkl
import torch.backends.mps
import torch.backends.xnnpack
import torch.cuda
from torch import Tensor
from torch._C import ScriptDict, ScriptList  # type: ignore[attr-defined]
from torch._utils_internal import get_writable_path
from torch._logging.scribe import open_source_signpost
from torch.nn import (
    ModuleDict,
    ModuleList,
    ParameterDict,
    ParameterList,
    Sequential,
)
from torch.onnx import (
    register_custom_op_symbolic,
    unregister_custom_op_symbolic,
)
from torch.testing import make_tensor
from torch.testing._comparison import (
    BooleanPair,
    NonePair,
    NumberPair,
    Pair,
    TensorLikePair,
)
from torch.testing._comparison import not_close_error_metas
from torch.testing._internal.common_dtype import get_all_dtypes
from torch.utils._import_utils import _check_module_exists
import torch.utils._pytree as pytree
from torch.utils import cpp_extension
try:
    import pytest  # type: ignore[import-not-found]
    has_pytest = True
except ImportError:
    has_pytest = False

SEED = 1234
MI350_ARCH = ("gfx950",)
MI300_ARCH = ("gfx942",)
MI200_ARCH = ("gfx90a")
NAVI_ARCH = ("gfx1030", "gfx1100", "gfx1101", "gfx1200", "gfx1201")
NAVI3_ARCH = ("gfx1100", "gfx1101")
NAVI4_ARCH = ("gfx1200", "gfx1201")

class ProfilingMode(Enum):
    LEGACY = 1
    SIMPLE = 2
    PROFILING = 3

# Set by parse_cmd_line_args() if called
CI_TEST_PREFIX = ""
DISABLED_TESTS_FILE = ""
GRAPH_EXECUTOR : Optional[ProfilingMode] = None
LOG_SUFFIX = ""
PYTEST_SINGLE_TEST = ""
REPEAT_COUNT = 0
RERUN_DISABLED_TESTS = False
RUN_PARALLEL = 0
SHOWLOCALS = False
SLOW_TESTS_FILE = ""
TEST_BAILOUTS = False
TEST_DISCOVER = False
TEST_IN_SUBPROCESS = False
TEST_SAVE_XML = ""
UNITTEST_ARGS : list[str] = []
USE_PYTEST = False

def is_navi3_arch():
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        gfx_arch = prop.gcnArchName.split(":")[0]
        if gfx_arch in NAVI3_ARCH:
            return True
    return False

def freeze_rng_state(*args, **kwargs):
    return torch.testing._utils.freeze_rng_state(*args, **kwargs)


# Class to keep track of test flags configurable by environment variables.
# Flags set here are intended to be read-only and should not be modified after
# definition.
# TODO: Expand this class to handle arbitrary settings in addition to boolean flags?
class TestEnvironment:
    # Set of env vars to set for the repro command that is output on test failure.
    # Specifically, this includes env vars that are set to non-default values and
    # are not implied. Maps from env var name -> value (int)
    repro_env_vars: dict = {}

    # Defines a flag usable throughout the test suite, determining its value by querying
    # the specified environment variable.
    #
    # Args:
    #     name (str): The name of the flag. A global variable with this name will be set
    #         for convenient access throughout the test suite.
    #     env_var (str): The name of the primary environment variable from which to
    #         determine the value of this flag. If this is None or the environment variable
    #         is unset, the default value will be used unless otherwise implied (see
    #         implied_by_fn). Default: None
    #     default (bool): The default value to use for the flag if unset by the environment
    #         variable and unimplied. Default: False
    #     include_in_repro (bool): Indicates whether this flag should be included in the
    #         repro command that is output on test failure (i.e. whether it is possibly
    #         relevant to reproducing the test failure). Default: True
    #     enabled_fn (Callable): Callable returning whether the flag should be enabled
    #         given the environment variable value and the default value. Default: Lambda
    #         requiring "0" to disable if on by default OR "1" to enable if off by default.
    #     implied_by_fn (Callable): Thunk returning a bool to imply this flag as enabled
    #         by something outside of its primary environment variable setting. For example,
    #         this can be useful if the value of another environment variable implies the flag
    #         as enabled. Default: Lambda returning False to indicate no implications.
    @staticmethod
    def def_flag(
        name,
        env_var=None,
        default=False,
        include_in_repro=True,
        enabled_fn=lambda env_var_val, default: (
            (env_var_val != "0") if default else (env_var_val == "1")),
        implied_by_fn=lambda: False,
    ):
        enabled = default
        env_var_val = None
        if env_var is not None:
            env_var_val = os.getenv(env_var)
            enabled = enabled_fn(env_var_val, default)
        implied = implied_by_fn()
        enabled = enabled or implied
        if include_in_repro and (env_var is not None) and (enabled != default) and not implied:
            TestEnvironment.repro_env_vars[env_var] = env_var_val

        # export flag globally for convenience
        assert name not in globals(), f"duplicate definition of flag '{name}'"
        globals()[name] = enabled
        return enabled

    # Defines a setting usable throughout the test suite, determining its value by querying
    # the specified environment variable. This differs from a flag in that it's not restricted
    # to a boolean value.
    #
    # Args:
    #     name (str): The name of the setting. A global variable with this name will be set
    #         for convenient access throughout the test suite.
    #     env_var (str): The name of the primary environment variable from which to
    #         determine the value of this setting. If this is None or the environment variable
    #         is unset, the default value will be used. Default: None
    #     default (Any): The default value to use for the setting if unset by the environment
    #         variable. Default: None
    #     include_in_repro (bool): Indicates whether this setting should be included in the
    #         repro command that is output on test failure (i.e. whether it is possibly
    #         relevant to reproducing the test failure). Default: True
    #     parse_fn (Callable): Callable parsing the env var string. Default value just uses
    #         the string itself.
    @staticmethod
    def def_setting(
        name,
        env_var=None,
        default=None,
        include_in_repro=True,
        parse_fn=lambda maybe_val_str: maybe_val_str,
    ):
        value = default if env_var is None else os.getenv(env_var)
        value = parse_fn(value)
        if include_in_repro and (value != default):
            TestEnvironment.repro_env_vars[env_var] = value

        # export setting globally for convenience
        assert name not in globals(), f"duplicate definition of setting '{name}'"
        globals()[name] = value
        return value

    # Returns a string prefix usable to set environment variables for any test
    # settings that should be explicitly set to match this instantiation of the
    # test suite.
    # Example: "PYTORCH_TEST_WITH_ASAN=1 PYTORCH_TEST_WITH_ROCM=1"
    @staticmethod
    def repro_env_var_prefix() -> str:
        return " ".join([f"{env_var}={value}"
                         for env_var, value in TestEnvironment.repro_env_vars.items()])


log = logging.getLogger(__name__)
torch.backends.disable_global_flags()

FILE_SCHEMA = "file://"
if sys.platform == 'win32':
    FILE_SCHEMA = "file:///"

# NB: This flag differs semantically from others in that setting the env var to any
# non-empty value will cause it to be true:
#   CI=1, CI="true", CI=0, etc. all set the flag to be true.
#   CI= and an unset CI set the flag to be false.
# GitHub sets the value to CI="true" to enable it.
IS_CI: bool = TestEnvironment.def_flag(
    "IS_CI",
    env_var="CI",
    include_in_repro=False,
    enabled_fn=lambda env_var_value, _: bool(env_var_value),
)
IS_SANDCASTLE: bool = TestEnvironment.def_flag(
    "IS_SANDCASTLE",
    env_var="SANDCASTLE",
    implied_by_fn=lambda: os.getenv("TW_JOB_USER") == "sandcastle",
    include_in_repro=False,
)
IN_RE_WORKER: bool = os.environ.get("INSIDE_RE_WORKER") is not None

_is_fbcode_default = (
    hasattr(torch._utils_internal, "IS_FBSOURCE") and
    torch._utils_internal.IS_FBSOURCE
)

IS_FBCODE: bool = TestEnvironment.def_flag(
    "IS_FBCODE",
    env_var="PYTORCH_TEST_FBCODE",
    default=_is_fbcode_default,
    include_in_repro=False,
)
IS_REMOTE_GPU: bool = TestEnvironment.def_flag(
    "IS_REMOTE_GPU",
    env_var="PYTORCH_TEST_REMOTE_GPU",
    include_in_repro=False,
)

DISABLE_RUNNING_SCRIPT_CHK: bool = TestEnvironment.def_flag(
    "DISABLE_RUNNING_SCRIPT_CHK",
    env_var="PYTORCH_DISABLE_RUNNING_SCRIPT_CHK",
    include_in_repro=False,
)
# NB: enabled by default unless in an fbcode context.
PRINT_REPRO_ON_FAILURE: bool = TestEnvironment.def_flag(
    "PRINT_REPRO_ON_FAILURE",
    env_var="PYTORCH_PRINT_REPRO_ON_FAILURE",
    default=(not IS_FBCODE),
    include_in_repro=False,
)

# possibly restrict OpInfo tests to a single sample input
OPINFO_SAMPLE_INPUT_INDEX: Optional[int] = TestEnvironment.def_setting(
    "OPINFO_SAMPLE_INPUT_INDEX",
    env_var="PYTORCH_OPINFO_SAMPLE_INPUT_INDEX",
    default=None,
    # Don't include the env var value in the repro command because the info will
    # be queried from the tracked sample input instead
    include_in_repro=False,
    parse_fn=lambda val: None if val is None else int(val),
)

DEFAULT_DISABLED_TESTS_FILE = '.pytorch-disabled-tests.json'
DEFAULT_SLOW_TESTS_FILE = 'slow_tests.json'

disabled_tests_dict = {}
slow_tests_dict = {}

def maybe_load_json(filename):
    if os.path.isfile(filename):
        with open(filename) as fp:
            return json.load(fp)
    log.warning("Attempted to load json file '%s' but it does not exist.", filename)
    return {}

# set them here in case the tests are running in a subprocess that doesn't call run_tests
if os.getenv("SLOW_TESTS_FILE", ""):
    slow_tests_dict = maybe_load_json(os.getenv("SLOW_TESTS_FILE", ""))
if os.getenv("DISABLED_TESTS_FILE", ""):
    disabled_tests_dict = maybe_load_json(os.getenv("DISABLED_TESTS_FILE", ""))

NATIVE_DEVICES = ('cpu', 'cuda', 'xpu', 'meta', 'mps', 'mtia', torch._C._get_privateuse1_backend_name())

# used for managing devices testing for torch profiler UTs
# for now cpu, cuda and xpu are added for testing torch profiler UTs
DEVICE_LIST_SUPPORT_PROFILING_TEST = ('cpu', 'cuda', 'xpu')
ALLOW_XPU_PROFILING_TEST = True

check_names = ['orin', 'concord', 'galen', 'xavier', 'nano', 'jetson', 'tegra', 'thor']
IS_JETSON = any(name in platform.platform() for name in check_names)

def gcIfJetson(fn):
    # Irregular Jetson host/device memory setup requires cleanup to avoid tests being killed
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if IS_JETSON:
            gc.collect()
            torch.cuda.empty_cache()
        fn(*args, **kwargs)
    return wrapper

# Tries to extract the current test function by crawling the stack.
# If unsuccessful, return None.
def extract_test_fn() -> Optional[Callable]:
    try:
        stack = inspect.stack()
        for frame_info in stack:
            frame = frame_info.frame
            if "self" not in frame.f_locals:
                continue
            self_val = frame.f_locals["self"]
            if isinstance(self_val, unittest.TestCase):
                test_id = self_val.id()
                *_, cls_name, test_name = test_id.rsplit('.', 2)
                if cls_name == type(self_val).__name__ and test_name.startswith("test"):
                    test_fn = getattr(self_val, test_name).__func__
                    return test_fn
    except Exception:
        pass
    return None

# Contains tracked input data useful for debugging purposes
@dataclass
class TrackedInput:
    index: int
    val: Any
    type_desc: str

# Attempt to pull out tracked input information from the test function.
# A TrackedInputIter is used to insert this information.
def get_tracked_input() -> Optional[TrackedInput]:
    test_fn = extract_test_fn()
    if test_fn is None:
        return None
    return getattr(test_fn, "tracked_input", None)

def clear_tracked_input() -> None:
    test_fn = extract_test_fn()
    if test_fn is None:
        return
    if not hasattr(test_fn, "tracked_input"):
        return
    test_fn.tracked_input = None  # type: ignore[attr-defined]

# Wraps an iterator and tracks the most recent value the iterator produces
# for debugging purposes. Tracked values are stored on the test function.
class TrackedInputIter:
    def __init__(
        self,
        child_iter,
        input_type_desc,
        item_callback=None,
        track_callback=None,
        set_seed=True,
        restrict_to_index=None
    ):
        self.child_iter = enumerate(child_iter)
        # Input type describes the things we're tracking (e.g. "sample input", "error input").
        self.input_type_desc = input_type_desc
        # NB: The two types of callbacks below exist because the thing we want to track isn't
        # always the same as the thing we want returned from the iterator. An example of this
        # is ErrorInput, which we want returned from the iterator, but which contains a
        # SampleInput that we want to track.
        # Item callback is run on each (iterated thing, index) to get the thing to return.
        self.item_callback = item_callback
        if self.item_callback is None:
            self.item_callback = lambda x, i: x
        # Track callback is run on each iterated thing to get the thing to track.
        self.track_callback = track_callback
        if self.track_callback is None:
            self.track_callback = lambda x: x
        self.test_fn = extract_test_fn()
        # Indicates whether the random seed should be set before each call to the iterator
        self.set_seed = set_seed
        # Indicates that iteration should be restricted to only the provided index.
        # If None, no restriction is done
        self.restrict_to_index = restrict_to_index

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self.set_seed:
                # use a test-name-specific hash for the seed if possible
                seed = (
                    int.from_bytes(hashlib.sha256(
                        self.test_fn.__qualname__.encode("utf-8")).digest()[:4], 'little')
                    if self.test_fn is not None else SEED
                )
                set_rng_seed(seed)

            # allow StopIteration to bubble up
            input_idx, input_val = next(self.child_iter)
            if (self.restrict_to_index is None) or (input_idx == self.restrict_to_index):
                break

        self._set_tracked_input(
            TrackedInput(
                index=input_idx, val=self.track_callback(input_val), type_desc=self.input_type_desc
            )
        )
        return self.item_callback(input_val, input_idx)

    def _set_tracked_input(self, tracked_input: TrackedInput):
        if self.test_fn is None:
            return
        if not hasattr(self.test_fn, "tracked_input"):
            return
        self.test_fn.tracked_input = tracked_input  # type: ignore[attr-defined]

class _TestParametrizer:
    """
    Decorator class for parametrizing a test function, yielding a set of new tests spawned
    from the original generic test, each specialized for a specific set of test inputs. For
    example, parametrizing a test across the set of ops will result in a test function per op.

    The decision of how to parametrize / what to parametrize over is intended to be implemented
    by each derived class.

    In the details, the decorator adds a 'parametrize_fn' property to the test function. This function
    is intended to be called later by one of:
      * Device-specific test instantiation via instantiate_device_type_tests(). Note that for this
        case there is no need to explicitly parametrize over device type, as that is handled separately.
      * Device-agnostic parametrized test instantiation via instantiate_parametrized_tests().

    If the decorator is applied to a test function that already has a 'parametrize_fn' property, a new
    composite 'parametrize_fn' will be created that generates tests with the product of the parameters
    generated by the old and new parametrize_fns. This allows for convenient composability of decorators.
    """
    def _parametrize_test(self, test, generic_cls, device_cls):
        """
        Parametrizes the given test function across whatever dimension is specified by the derived class.
        Tests can be parametrized over any arbitrary dimension or combination of dimensions, such as all
        ops, all modules, or all ops + their associated dtypes.

        Args:
            test (fn): Test function to parametrize over
            generic_cls (class): Generic test class object containing tests (e.g. TestFoo)
            device_cls (class): Device-specialized test class object (e.g. TestFooCPU); set to None
                if the tests are not part of a device-specific set

        Returns:
            Generator object returning 4-tuples of:
                test (fn): Parametrized test function; must support a device arg and args for any params
                test_name (str): Parametrized suffix for the test (e.g. opname_int64); will be appended to
                    the base name of the test
                param_kwargs (dict): Param kwargs to pass to the test (e.g. {'op': 'add', 'dtype': torch.int64})
                decorator_fn (callable): Callable[[Dict], List] for list of decorators to apply given param_kwargs
        """
        raise NotImplementedError

    def __call__(self, fn):
        if hasattr(fn, 'parametrize_fn'):
            # Do composition with the product of args.
            old_parametrize_fn = fn.parametrize_fn
            new_parametrize_fn = self._parametrize_test
            fn.parametrize_fn = compose_parametrize_fns(old_parametrize_fn, new_parametrize_fn)
        else:
            fn.parametrize_fn = self._parametrize_test
        return fn


def compose_parametrize_fns(old_parametrize_fn, new_parametrize_fn):
    """
    Returns a parametrize_fn that parametrizes over the product of the parameters handled
    by the given parametrize_fns. Each given parametrize_fn should each have the signature
    f(test, generic_cls, device_cls).

    The test names will be a combination of the names produced by the parametrize_fns in
    "<new_name>_<old_name>" order. This order is done to match intuition for constructed names
    when composing multiple decorators; the names will be built in top to bottom order when stacking
    parametrization decorators.

    Args:
        old_parametrize_fn (callable) - First parametrize_fn to compose.
        new_parametrize_fn (callable) - Second parametrize_fn to compose.
    """

    def composite_fn(test, generic_cls, device_cls,
                     old_parametrize_fn=old_parametrize_fn,
                     new_parametrize_fn=new_parametrize_fn):
        old_tests = list(old_parametrize_fn(test, generic_cls, device_cls))
        for (old_test, old_test_name, old_param_kwargs, old_dec_fn) in old_tests:
            for (new_test, new_test_name, new_param_kwargs, new_dec_fn) in \
                    new_parametrize_fn(old_test, generic_cls, device_cls):
                redundant_params = set(old_param_kwargs.keys()).intersection(new_param_kwargs.keys())
                if redundant_params:
                    raise RuntimeError('Parametrization over the same parameter by multiple parametrization '
                                       f'decorators is not supported. For test "{test.__name__}", the following parameters '
                                       f'are handled multiple times: {redundant_params}')
                full_param_kwargs = {**old_param_kwargs, **new_param_kwargs}
                merged_test_name = '{}{}{}'.format(new_test_name,
                                                   '_' if old_test_name != '' and new_test_name != '' else '',
                                                   old_test_name)

                def merged_decorator_fn(param_kwargs, old_dec_fn=old_dec_fn, new_dec_fn=new_dec_fn):
                    return list(old_dec_fn(param_kwargs)) + list(new_dec_fn(param_kwargs))

                yield (new_test, merged_test_name, full_param_kwargs, merged_decorator_fn)

    return composite_fn


def instantiate_parametrized_tests(generic_cls):
    """
    Instantiates tests that have been decorated with a parametrize_fn. This is generally performed by a
    decorator subclass of _TestParametrizer. The generic test will be replaced on the test class by
    parametrized tests with specialized names. This should be used instead of
    instantiate_device_type_tests() if the test class contains device-agnostic tests.

    You can also use it as a class decorator. E.g.

    ```
    @instantiate_parametrized_tests
    class TestFoo(TestCase):
        ...
    ```

    Args:
        generic_cls (class): Generic test class object containing tests (e.g. TestFoo)
    """
    for attr_name in tuple(dir(generic_cls)):
        class_attr = getattr(generic_cls, attr_name)
        if not hasattr(class_attr, 'parametrize_fn'):
            continue

        # Remove the generic test from the test class.
        delattr(generic_cls, attr_name)

        # Add parametrized tests to the test class.
        def instantiate_test_helper(cls, name, test, param_kwargs):
            @wraps(test)
            def instantiated_test(self, param_kwargs=param_kwargs):
                test(self, **param_kwargs)

            assert not hasattr(generic_cls, name), f"Redefinition of test {name}"
            setattr(generic_cls, name, instantiated_test)

        for (test, test_suffix, param_kwargs, decorator_fn) in class_attr.parametrize_fn(
                class_attr, generic_cls=generic_cls, device_cls=None):
            full_name = f'{test.__name__}_{test_suffix}'

            # Apply decorators based on full param kwargs.
            for decorator in decorator_fn(param_kwargs):
                test = decorator(test)

            instantiate_test_helper(cls=generic_cls, name=full_name, test=test, param_kwargs=param_kwargs)
    return generic_cls


class subtest:
    """
    Explicit subtest case for use with test parametrization.
    Allows for explicit naming of individual subtest cases as well as applying
    decorators to the parametrized test.

    Args:
        arg_values (iterable): Iterable of arg values (e.g. range(10)) or
            tuples of arg values (e.g. [(1, 2), (3, 4)]).
        name (str): Optional name to use for the test.
        decorators (iterable): Iterable of decorators to apply to the generated test.
    """
    __slots__ = ['arg_values', 'name', 'decorators']

    def __init__(self, arg_values, name=None, decorators=None):
        self.arg_values = arg_values
        self.name = name
        self.decorators = decorators if decorators else []


class parametrize(_TestParametrizer):
    """
    Decorator for applying generic test parametrizations.

    The interface for this decorator is modeled after `@pytest.mark.parametrize`.
    Basic usage between this decorator and pytest's is identical. The first argument
    should be a string containing comma-separated names of parameters for the test, and
    the second argument should be an iterable returning values or tuples of values for
    the case of multiple parameters.

    Beyond this basic usage, the decorator provides some additional functionality that
    pytest does not.

    1. Parametrized tests end up as generated test functions on unittest test classes.
    Since this differs from how pytest works, this decorator takes on the additional
    responsibility of naming these test functions. The default test names consists of
    the test's base name followed by each parameter name + value (e.g. "test_bar_x_1_y_foo"),
    but custom names can be defined using `name_fn` or the `subtest` structure (see below).

    2. The decorator specially handles parameter values of type `subtest`, which allows for
    more fine-grained control over both test naming and test execution. In particular, it can
    be used to tag subtests with explicit test names or apply arbitrary decorators (see examples
    below).

    Examples::

        @parametrize("x", range(5))
        def test_foo(self, x):
            ...

        @parametrize("x,y", [(1, 'foo'), (2, 'bar'), (3, 'baz')])
        def test_bar(self, x, y):
            ...

        @parametrize("x,y", [(1, 'foo'), (2, 'bar'), (3, 'baz')],
                     name_fn=lambda x, y: '{}_{}'.format(x, y))
        def test_bar_custom_names(self, x, y):
            ...

        @parametrize("x, y", [subtest((1, 2), name='double'),
                              subtest((1, 3), name='triple', decorators=[unittest.expectedFailure]),
                              subtest((1, 4), name='quadruple')])
        def test_baz(self, x, y):
            ...

    To actually instantiate the parametrized tests, one of instantiate_parametrized_tests() or
    instantiate_device_type_tests() should be called. The former is intended for test classes
    that contain device-agnostic tests, while the latter should be used for test classes that
    contain device-specific tests. Both support arbitrary parametrizations using the decorator.

    Args:
        arg_str (str): String of arg names separate by commas (e.g. "x,y").
        arg_values (iterable): Iterable of arg values (e.g. range(10)) or
            tuples of arg values (e.g. [(1, 2), (3, 4)]).
        name_fn (Callable): Optional function that takes in parameters and returns subtest name.
    """
    def __init__(self, arg_str, arg_values, name_fn=None):
        self.arg_names: list[str] = [s.strip() for s in arg_str.split(',') if s != '']
        self.arg_values = arg_values
        self.name_fn = name_fn

    def _formatted_str_repr(self, idx, name, value):
        """ Returns a string representation for the given arg that is suitable for use in test function names. """
        if isinstance(value, torch.dtype):
            return dtype_name(value)
        elif isinstance(value, torch.device):
            return str(value)
        # Can't use isinstance as it would cause a circular import
        elif type(value).__name__ in {'OpInfo', 'ModuleInfo'}:
            return value.formatted_name
        elif isinstance(value, (int, float, str)):
            return f"{name}_{str(value).replace('.', '_')}"
        else:
            return f"{name}{idx}"

    def _default_subtest_name(self, idx, values):
        return '_'.join([self._formatted_str_repr(idx, a, v) for a, v in zip(self.arg_names, values, strict=True)])

    def _get_subtest_name(self, idx, values, explicit_name=None):
        if explicit_name:
            subtest_name = explicit_name
        elif self.name_fn:
            subtest_name = self.name_fn(*values)
        else:
            subtest_name = self._default_subtest_name(idx, values)
        return subtest_name

    def _parametrize_test(self, test, generic_cls, device_cls):
        if len(self.arg_names) == 0:
            # No additional parameters needed for the test.
            test_name = ''
            yield (test, test_name, {}, lambda _: [])
        else:
            # Each "values" item is expected to be either:
            # * A tuple of values with one for each arg. For a single arg, a single item is expected.
            # * A subtest instance with arg_values matching the previous.
            values = check_exhausted_iterator = object()
            for idx, values in enumerate(self.arg_values):
                maybe_name = None

                decorators: list[Any] = []
                if isinstance(values, subtest):
                    sub = values
                    values = sub.arg_values
                    maybe_name = sub.name

                    @wraps(test)
                    def test_wrapper(*args, **kwargs):
                        return test(*args, **kwargs)

                    decorators = sub.decorators
                    gen_test = test_wrapper
                else:
                    gen_test = test

                values = list(values) if len(self.arg_names) > 1 else [values]  # type: ignore[call-overload]
                if len(values) != len(self.arg_names):
                    raise RuntimeError(f'Expected # values == # arg names, but got: {len(values)} '
                                       f'values and {len(self.arg_names)} names for test "{test.__name__}"')

                param_kwargs = dict(zip(self.arg_names, values, strict=True))

                test_name = self._get_subtest_name(idx, values, explicit_name=maybe_name)

                def decorator_fn(_, decorators=decorators):
                    return decorators

                yield (gen_test, test_name, param_kwargs, decorator_fn)

            if values is check_exhausted_iterator:
                raise ValueError(f'{test}: An empty arg_values was passed to @parametrize. '
                                 'Note that this may result from reuse of a generator.')


class reparametrize(_TestParametrizer):
    """
    Decorator for adjusting the way an existing parametrizer operates. This class runs
    the given adapter_fn on each parametrization produced by the given parametrizer,
    allowing for on-the-fly parametrization more flexible than the default,
    product-based composition that occurs when stacking parametrization decorators.

    If the adapter_fn returns None for a given test parametrization, that parametrization
    will be excluded. Otherwise, it's expected that the adapter_fn returns an iterable of
    modified parametrizations, with tweaked test names and parameter kwargs.

    Examples::

        def include_is_even_arg(test_name, param_kwargs):
            x = param_kwargs["x"]
            is_even = x % 2 == 0
            new_param_kwargs = dict(param_kwargs)
            new_param_kwargs["is_even"] = is_even
            is_even_suffix = "_even" if is_even else "_odd"
            new_test_name = f"{test_name}{is_even_suffix}"
            yield (new_test_name, new_param_kwargs)

        ...

        @reparametrize(parametrize("x", range(5)), include_is_even_arg)
        def test_foo(self, x, is_even):
            ...

        def exclude_odds(test_name, param_kwargs):
            x = param_kwargs["x"]
            is_even = x % 2 == 0
            yield None if not is_even else (test_name, param_kwargs)

        ...

        @reparametrize(parametrize("x", range(5)), exclude_odds)
        def test_bar(self, x):
            ...

    """
    def __init__(self, parametrizer, adapter_fn):
        self.parametrizer = parametrizer
        self.adapter_fn = adapter_fn

    def _parametrize_test(self, test, generic_cls, device_cls):
        for (gen_test, test_name, param_kwargs, decorator_fn) in \
                self.parametrizer._parametrize_test(test, generic_cls, device_cls):
            adapted = self.adapter_fn(test_name, param_kwargs)
            if adapted is not None:
                for adapted_item in adapted:
                    if adapted_item is not None:
                        new_test_name, new_param_kwargs = adapted_item
                        yield (gen_test, new_test_name, new_param_kwargs, decorator_fn)


class decorateIf(_TestParametrizer):
    """
    Decorator for applying parameter-specific conditional decoration.
    Composes with other test parametrizers (e.g. @modules, @ops, @parametrize, etc.).

    Examples::

        @decorateIf(unittest.skip, lambda params: params["x"] == 2)
        @parametrize("x", range(5))
        def test_foo(self, x):
            ...

        @parametrize("x,y", [(1, 'foo'), (2, 'bar'), (3, 'baz')])
        @decorateIf(
            unittest.expectedFailure,
            lambda params: params["x"] == 3 and params["y"] == "baz"
        )
        def test_bar(self, x, y):
            ...

        @decorateIf(
            unittest.expectedFailure,
            lambda params: params["op"].name == "add" and params["dtype"] == torch.float16
        )
        @ops(op_db)
        def test_op_foo(self, device, dtype, op):
            ...

        @decorateIf(
            unittest.skip,
            lambda params: params["module_info"].module_cls is torch.nn.Linear and \
                params["device"] == "cpu"
        )
        @modules(module_db)
        def test_module_foo(self, device, dtype, module_info):
            ...

    Args:
        decorator: Test decorator to apply if the predicate is satisfied.
        predicate_fn (Callable): Function taking in a dict of params and returning a boolean
            indicating whether the decorator should be applied or not.
    """
    def __init__(self, decorator, predicate_fn):
        self.decorator = decorator
        self.predicate_fn = predicate_fn

    def _parametrize_test(self, test, generic_cls, device_cls):

        # Leave test as-is and return the appropriate decorator_fn.
        def decorator_fn(params, decorator=self.decorator, predicate_fn=self.predicate_fn):
            if predicate_fn(params):
                return [decorator]
            else:
                return []

        @wraps(test)
        def test_wrapper(*args, **kwargs):
            return test(*args, **kwargs)

        test_name = ''
        yield (test_wrapper, test_name, {}, decorator_fn)


def cppProfilingFlagsToProfilingMode():
    old_prof_exec_state = torch._C._jit_set_profiling_executor(True)
    old_prof_mode_state = torch._C._get_graph_executor_optimize(True)
    torch._C._jit_set_profiling_executor(old_prof_exec_state)
    torch._C._get_graph_executor_optimize(old_prof_mode_state)

    if old_prof_exec_state:
        if old_prof_mode_state:
            return ProfilingMode.PROFILING
        else:
            return ProfilingMode.SIMPLE
    else:
        return ProfilingMode.LEGACY

@contextmanager
def enable_profiling_mode_for_profiling_tests():
    old_prof_exec_state = False
    old_prof_mode_state = False
    assert GRAPH_EXECUTOR
    if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
        old_prof_exec_state = torch._C._jit_set_profiling_executor(True)
        old_prof_mode_state = torch._C._get_graph_executor_optimize(True)
    try:
        yield
    finally:
        if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
            torch._C._jit_set_profiling_executor(old_prof_exec_state)
            torch._C._get_graph_executor_optimize(old_prof_mode_state)

@contextmanager
def enable_profiling_mode():
    old_prof_exec_state = torch._C._jit_set_profiling_executor(True)
    old_prof_mode_state = torch._C._get_graph_executor_optimize(True)
    try:
        yield
    finally:
        torch._C._jit_set_profiling_executor(old_prof_exec_state)
        torch._C._get_graph_executor_optimize(old_prof_mode_state)

@contextmanager
def num_profiled_runs(num_runs):
    old_num_runs = torch._C._jit_set_num_profiled_runs(num_runs)
    try:
        yield
    finally:
        torch._C._jit_set_num_profiled_runs(old_num_runs)

func_call = torch._C.ScriptFunction.__call__
meth_call = torch._C.ScriptMethod.__call__

def prof_callable(callable, *args, **kwargs):
    if 'profile_and_replay' in kwargs:
        del kwargs['profile_and_replay']
        assert GRAPH_EXECUTOR
        if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
            with enable_profiling_mode_for_profiling_tests():
                callable(*args, **kwargs)
                return callable(*args, **kwargs)

    return callable(*args, **kwargs)

def raise_on_run_directly(file_to_call):
    raise RuntimeError("This test file is not meant to be run directly, "
                       f"use:\n\n\tpython {file_to_call} TESTNAME\n\n"
                       "instead.")

def prof_func_call(*args, **kwargs):
    return prof_callable(func_call, *args, **kwargs)

def prof_meth_call(*args, **kwargs):
    return prof_callable(meth_call, *args, **kwargs)

torch._C.ScriptFunction.__call__ = prof_func_call  # type: ignore[method-assign]
torch._C.ScriptMethod.__call__ = prof_meth_call  # type: ignore[method-assign]

def _get_test_report_path():
    # allow users to override the test file location. We need this
    # because the distributed tests run the same test file multiple
    # times with different configurations.
    override = os.environ.get('TEST_REPORT_SOURCE_OVERRIDE')
    test_source = override if override is not None else 'python-unittest'
    return os.path.join('test-reports', test_source)

def parse_cmd_line_args():
    global CI_TEST_PREFIX
    global DISABLED_TESTS_FILE
    global GRAPH_EXECUTOR
    global LOG_SUFFIX
    global PYTEST_SINGLE_TEST
    global REPEAT_COUNT
    global RERUN_DISABLED_TESTS
    global RUN_PARALLEL
    global SHOWLOCALS
    global SLOW_TESTS_FILE
    global TEST_BAILOUTS
    global TEST_DISCOVER
    global TEST_IN_SUBPROCESS
    global TEST_SAVE_XML
    global UNITTEST_ARGS
    global USE_PYTEST

    is_running_via_run_test = "run_test.py" in getattr(__main__, "__file__", "")
    parser = argparse.ArgumentParser(add_help=not is_running_via_run_test, allow_abbrev=False)
    parser.add_argument('--subprocess', action='store_true',
                        help='whether to run each test in a subprocess')
    parser.add_argument('--accept', action='store_true')
    parser.add_argument('--jit-executor', '--jit_executor', type=str)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--test-bailouts', '--test_bailouts', action='store_true')
    parser.add_argument('--use-pytest', action='store_true')
    parser.add_argument('--save-xml', nargs='?', type=str,
                        const=_get_test_report_path(),
                        default=_get_test_report_path() if IS_CI else None)
    parser.add_argument('--discover-tests', action='store_true')
    parser.add_argument('--log-suffix', type=str, default="")
    parser.add_argument('--run-parallel', type=int, default=1)
    parser.add_argument('--import-slow-tests', type=str, nargs='?', const=DEFAULT_SLOW_TESTS_FILE)
    parser.add_argument('--import-disabled-tests', type=str, nargs='?', const=DEFAULT_DISABLED_TESTS_FILE)
    parser.add_argument('--rerun-disabled-tests', action='store_true')
    parser.add_argument('--pytest-single-test', type=str, nargs=1)
    parser.add_argument('--showlocals', action=argparse.BooleanOptionalAction, default=False)

# Only run when -h or --help flag is active to display both unittest and parser help messages.
    def run_unittest_help(argv):
        unittest.main(argv=argv)

    if '-h' in sys.argv or '--help' in sys.argv:
        help_thread = threading.Thread(target=run_unittest_help, args=(sys.argv,))
        help_thread.start()
        help_thread.join()

    args, remaining = parser.parse_known_args()
    if args.jit_executor == 'legacy':
        GRAPH_EXECUTOR = ProfilingMode.LEGACY
    elif args.jit_executor == 'profiling':
        GRAPH_EXECUTOR = ProfilingMode.PROFILING
    elif args.jit_executor == 'simple':
        GRAPH_EXECUTOR = ProfilingMode.SIMPLE
    else:
        # infer flags based on the default settings
        GRAPH_EXECUTOR = cppProfilingFlagsToProfilingMode()

    RERUN_DISABLED_TESTS = args.rerun_disabled_tests

    SLOW_TESTS_FILE = args.import_slow_tests
    DISABLED_TESTS_FILE = args.import_disabled_tests
    LOG_SUFFIX = args.log_suffix
    RUN_PARALLEL = args.run_parallel
    TEST_BAILOUTS = args.test_bailouts
    USE_PYTEST = args.use_pytest
    PYTEST_SINGLE_TEST = args.pytest_single_test
    TEST_DISCOVER = args.discover_tests
    TEST_IN_SUBPROCESS = args.subprocess
    TEST_SAVE_XML = args.save_xml
    REPEAT_COUNT = args.repeat
    SHOWLOCALS = args.showlocals
    if not getattr(expecttest, "ACCEPT", False):
        expecttest.ACCEPT = args.accept
    UNITTEST_ARGS = [sys.argv[0]] + remaining

    set_rng_seed()

    # CI Prefix path used only on CI environment
    CI_TEST_PREFIX = str(Path(os.getcwd()))

def wait_for_process(p, timeout=None):
    try:
        return p.wait(timeout=timeout)
    except KeyboardInterrupt:
        # Give `p` a chance to handle KeyboardInterrupt. Without this,
        # `pytest` can't print errors it collected so far upon KeyboardInterrupt.
        exit_status = p.wait(timeout=5)
        if exit_status is not None:
            return exit_status
        else:
            p.kill()
            raise
    except subprocess.TimeoutExpired:
        # send SIGINT to give pytest a chance to make xml
        p.send_signal(signal.SIGINT)
        exit_status = None
        try:
            exit_status = p.wait(timeout=5)
        # try to handle the case where p.wait(timeout=5) times out as well as
        # otherwise the wait() call in the finally block can potentially hang
        except subprocess.TimeoutExpired:
            pass
        if exit_status is not None:
            return exit_status
        else:
            p.kill()
        raise
    except:  # noqa: B001,E722, copied from python core library
        p.kill()
        raise
    finally:
        # Always call p.wait() to ensure exit
        p.wait()

def shell(command, cwd=None, env=None, stdout=None, stderr=None, timeout=None):
    sys.stdout.flush()
    sys.stderr.flush()
    # The following cool snippet is copied from Py3 core library subprocess.call
    # only the with
    #   1. `except KeyboardInterrupt` block added for SIGINT handling.
    #   2. In Py2, subprocess.Popen doesn't return a context manager, so we do
    #      `p.wait()` in a `final` block for the code to be portable.
    #
    # https://github.com/python/cpython/blob/71b6c1af727fbe13525fb734568057d78cea33f3/Lib/subprocess.py#L309-L323
    assert not isinstance(command, str), "Command to shell should be a list or tuple of tokens"
    p = subprocess.Popen(command, universal_newlines=True, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
    return wait_for_process(p, timeout=timeout)


def retry_shell(
    command,
    cwd=None,
    env=None,
    stdout=None,
    stderr=None,
    timeout=None,
    retries=1,
    was_rerun=False,
) -> tuple[int, bool]:
    # Returns exicode + whether it was rerun
    assert (
        retries >= 0
    ), f"Expecting non negative number for number of retries, got {retries}"
    try:
        exit_code = shell(
            command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, timeout=timeout
        )
        if exit_code == 0 or retries == 0:
            return exit_code, was_rerun
        print(
            f"Got exit code {exit_code}, retrying (retries left={retries})",
            file=stdout,
            flush=True,
        )
    except subprocess.TimeoutExpired:
        if retries == 0:
            print(
                f"Command took >{timeout // 60}min, returning 124",
                file=stdout,
                flush=True,
            )
            return 124, was_rerun
        print(
            f"Command took >{timeout // 60}min, retrying (retries left={retries})",
            file=stdout,
            flush=True,
        )
    return retry_shell(
        command,
        cwd=cwd,
        env=env,
        stdout=stdout,
        stderr=stderr,
        timeout=timeout,
        retries=retries - 1,
        was_rerun=True,
    )


def discover_test_cases_recursively(suite_or_case):
    if isinstance(suite_or_case, unittest.TestCase):
        return [suite_or_case]
    rc = []
    for element in suite_or_case:
        print(element)
        rc.extend(discover_test_cases_recursively(element))
    return rc

def get_test_names(test_cases):
    return ['.'.join(case.id().split('.')[-2:]) for case in test_cases]

def _print_test_names():
    suite = unittest.TestLoader().loadTestsFromModule(__main__)
    test_cases = discover_test_cases_recursively(suite)
    for name in get_test_names(test_cases):
        print(name)

def chunk_list(lst, nchunks):
    return [lst[i::nchunks] for i in range(nchunks)]

# sanitize filename e.g., distributed/pipeline/sync/skip/test_api.py -> distributed.pipeline.sync.skip.test_api
def sanitize_test_filename(filename):
    # inspect.getfile returns absolute path in some CI jobs, converting it to relative path if needed
    if filename.startswith(CI_TEST_PREFIX):
        filename = filename[len(CI_TEST_PREFIX) + 1:]
    strip_py = re.sub(r'.py$', '', filename)
    return re.sub('/', r'.', strip_py)

def lint_test_case_extension(suite):
    succeed = True
    for test_case_or_suite in suite:
        test_case = test_case_or_suite
        if isinstance(test_case_or_suite, unittest.TestSuite):
            first_test = test_case_or_suite._tests[0] if len(test_case_or_suite._tests) > 0 else None
            if first_test is not None and isinstance(first_test, unittest.TestSuite):
                return succeed and lint_test_case_extension(test_case_or_suite)
            test_case = first_test

        if test_case is not None:
            if not isinstance(test_case, TestCase):
                test_class = test_case.id().split('.', 1)[1].split('.')[0]
                err = "This test class should extend from torch.testing._internal.common_utils.TestCase but it doesn't."
                print(f"{test_class} - failed. {err}")
                succeed = False
    return succeed


def get_report_path(argv=None, pytest=False):
    if argv is None:
        argv = UNITTEST_ARGS
    test_filename = sanitize_test_filename(argv[0])
    test_report_path = TEST_SAVE_XML + LOG_SUFFIX
    test_report_path = os.path.join(test_report_path, test_filename)
    if pytest:
        test_report_path = test_report_path.replace('python-unittest', 'python-pytest')
        os.makedirs(test_report_path, exist_ok=True)
        test_report_path = os.path.join(test_report_path, f"{test_filename}-{os.urandom(8).hex()}.xml")
        return test_report_path
    os.makedirs(test_report_path, exist_ok=True)
    return test_report_path


def sanitize_pytest_xml(xml_file: str):
    # pytext xml is different from unittext xml, this function makes pytest xml more similar to unittest xml
    # consider somehow modifying the XML logger in conftest to do this instead
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_file)
    for testcase in tree.iter('testcase'):
        full_classname = testcase.attrib.get("classname")
        if full_classname is None:
            continue
        # The test prefix is optional
        regex_result = re.search(r"^(test\.)?(?P<file>.*)\.(?P<classname>[^\.]*)$", full_classname)
        if regex_result is None:
            continue
        classname = regex_result.group("classname")
        file = regex_result.group("file").replace(".", "/")
        testcase.set("classname", classname)
        testcase.set("file", f"{file}.py")
    tree.write(xml_file)


def get_pytest_test_cases(argv: list[str]) -> list[str]:
    class TestCollectorPlugin:
        def __init__(self) -> None:
            self.tests: list[Any] = []

        def pytest_collection_finish(self, session):
            for item in session.items:
                self.tests.append(session.config.cwd_relative_nodeid(item.nodeid))

    test_collector_plugin = TestCollectorPlugin()
    import pytest
    pytest.main(
        [arg for arg in argv if arg != '-vv'] + ['--collect-only', '-qq', '--use-main-module'],
        plugins=[test_collector_plugin]
    )
    return test_collector_plugin.tests


def run_tests(argv=None):
    parse_cmd_line_args()
    if argv is None:
        argv = UNITTEST_ARGS

    # import test files.
    if SLOW_TESTS_FILE:
        if os.path.exists(SLOW_TESTS_FILE):
            with open(SLOW_TESTS_FILE) as fp:
                global slow_tests_dict
                slow_tests_dict = json.load(fp)
                # use env vars so pytest-xdist subprocesses can still access them
                os.environ['SLOW_TESTS_FILE'] = SLOW_TESTS_FILE
        else:
            warnings.warn(f'slow test file provided but not found: {SLOW_TESTS_FILE}', stacklevel=2)
    if DISABLED_TESTS_FILE:
        if 
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/testing/_internal`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/testing/_internal`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/torch/testing/_internal/common_utils.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/testing/_internal`):

- [`hypothesis_utils.py_kw.md_docs.md`](./hypothesis_utils.py_kw.md_docs.md)
- [`fake_config_module2.py_kw.md_docs.md`](./fake_config_module2.py_kw.md_docs.md)
- [`static_module.py_kw.md_docs.md`](./static_module.py_kw.md_docs.md)
- [`common_pruning.py_kw.md_docs.md`](./common_pruning.py_kw.md_docs.md)
- [`composite_compliance.py_kw.md_docs.md`](./composite_compliance.py_kw.md_docs.md)
- [`common_mkldnn.py_docs.md_docs.md`](./common_mkldnn.py_docs.md_docs.md)
- [`triton_utils.py_docs.md_docs.md`](./triton_utils.py_docs.md_docs.md)
- [`common_dtype.py_docs.md_docs.md`](./common_dtype.py_docs.md_docs.md)
- [`common_methods_invocations.py_docs.md_docs.md`](./common_methods_invocations.py_docs.md_docs.md)
- [`hypothesis_utils.py_docs.md_docs.md`](./hypothesis_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `common_utils.py_docs.md_docs.md`
- **Keyword Index**: `common_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
