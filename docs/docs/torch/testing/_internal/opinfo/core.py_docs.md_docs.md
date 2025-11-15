# Documentation: `docs/torch/testing/_internal/opinfo/core.py_docs.md`

## File Metadata

- **Path**: `docs/torch/testing/_internal/opinfo/core.py_docs.md`
- **Size**: 53,355 bytes (52.10 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/testing/_internal/opinfo/core.py`

## File Metadata

- **Path**: `torch/testing/_internal/opinfo/core.py`
- **Size**: 124,158 bytes (121.25 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# mypy: ignore-errors

import collections
import collections.abc
import contextlib
import logging
import math
import operator
import unittest
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import partial
from itertools import product
from typing import Any, Optional, TypeVar, Union

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    skipCPUIfNoFFT,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_dtype import (
    _dispatch_dtypes,
    floating_and_complex_types,
    floating_and_complex_types_and,
    floating_types,
    get_all_dtypes,
)
from torch.testing._internal.common_utils import (
    extract_test_fn,
    IS_FBCODE,
    is_iterable_of_tensors,
    noncontiguous_like,
    OPINFO_SAMPLE_INPUT_INDEX,
    TEST_WITH_ROCM,
    torch_to_numpy_dtype_dict,
    TrackedInputIter,
    USE_PYTEST,
)
from torch.testing._internal.opinfo import utils
from torchgen.utils import dataclass_repr


# setup logging
log = logging.getLogger(__name__)

# Reasonable testing sizes for dimensions
L = 20
M = 10
S = 5
XS = 3

# Unique value to distinguish default from anything else
_NOTHING = object()


# Extension of getattr to support qualified names
# e.g. _getattr_qual(torch, 'linalg.norm') -> torch.linalg.norm
def _getattr_qual(obj, name, default=_NOTHING):
    try:
        for path in name.split("."):
            obj = getattr(obj, path)
        return obj
    except AttributeError:
        if default is not _NOTHING:
            return default
        else:
            raise


class DecorateInfo:
    """Describes which test, or type of tests, should be wrapped in the given
    decorators when testing an operator. Any test that matches all provided
    arguments will be decorated. The decorators will only be applied if the
    active_if argument is True."""

    __slots__ = [
        "decorators",
        "cls_name",
        "test_name",
        "device_type",
        "dtypes",
        "active_if",
    ]

    def __init__(
        self,
        decorators,
        cls_name=None,
        test_name=None,
        *,
        device_type=None,
        dtypes=None,
        active_if=True,
    ):
        self.decorators = (
            list(decorators)
            if isinstance(decorators, collections.abc.Sequence)
            else [decorators]
        )
        self.cls_name = cls_name
        self.test_name = test_name
        self.device_type = device_type
        self.dtypes = dtypes
        self.active_if = active_if

        # Validate dtypes
        if self.dtypes is not None:
            for dtype in self.dtypes:
                assert isinstance(dtype, torch.dtype)

    def is_active(self, cls_name, test_name, device_type, dtype, param_kwargs):
        return (
            self.active_if
            and (self.cls_name is None or self.cls_name == cls_name)
            and (self.test_name is None or self.test_name == test_name)
            and (self.device_type is None or self.device_type == device_type)
            and (self.dtypes is None or dtype in self.dtypes)
            # Support callables over kwargs to determine if the decorator is active.
            and (
                self.active_if(param_kwargs)
                if isinstance(self.active_if, Callable)
                else self.active_if
            )
        )


# FIXME
# Note: historically the 'input' kwarg had to be a Tensor or TensorList, but we are trying
#   to support scalar inputs, too. Some tests still depend on 'input' being a Tensor
#   or TensorList, however.
class SampleInput:
    """Represents sample inputs to a function."""

    __slots__ = [
        "input",
        "args",
        "kwargs",
        "output_process_fn_grad",
        "broadcasts_input",
        "name",
    ]

    def __init__(
        self,
        input,
        *var_args,
        args=None,
        kwargs=None,
        output_process_fn_grad=None,
        broadcasts_input=None,
        name=None,
        **var_kwargs,
    ):
        # input is the first input to the op and is typically either a Tensor or TensorList (Sequence[Tensor]).
        # This follows the typical pattern where for Tensor inputs op(t, ...) = t.op(...).
        self.input = input

        # Allow calling either as SampleInput(input, args=args, kwargs=kwargs), or as
        # SampleInput(input, *args, **kwargs) but not to mix the two forms
        if args is not None or kwargs is not None:
            assert not var_args and not var_kwargs, """
A SampleInput can be constructed "naturally" with *args and **kwargs or by
explicitly setting the "args" and "kwargs" parameters, but the two
methods of construction cannot be mixed!"""
        elif var_args or var_kwargs:
            assert (
                output_process_fn_grad is None
                and broadcasts_input is None
                and name is None
            ), """
A SampleInput constructed "naturally" with *args and **kwargs
cannot specify additional metadata in keyword arguments"""

        self.args = args if args is not None else var_args
        assert isinstance(self.args, tuple)
        self.kwargs = kwargs if kwargs is not None else var_kwargs
        assert isinstance(self.kwargs, dict)

        self.output_process_fn_grad = (
            output_process_fn_grad
            if output_process_fn_grad is not None
            else lambda x: x
        )
        self.name = name if name is not None else ""

        # Specifies if `self.input` is broadcasted or not,
        # given that the operator supports broadcasting.
        # This field is used to verify the behavior for inplace variant.
        #
        # If a SampleInput is marked with `broadcasts_input=True`,
        # it is verified that we get a `RuntimeError` with this sample,
        # and inplace variant. Also inplace grad{grad} tests are skipped,
        # for such inputs (as they will error out otherwise).
        self.broadcasts_input = (
            broadcasts_input if broadcasts_input is not None else False
        )

    def with_metadata(
        self, *, output_process_fn_grad=None, broadcasts_input=None, name=None
    ):
        if output_process_fn_grad is not None:
            self.output_process_fn_grad = output_process_fn_grad
        if broadcasts_input is not None:
            self.broadcasts_input = broadcasts_input
        if name is not None:
            self.name = name
        return self

    def _repr_helper(self, formatter):
        # Helper function to return the details of the SampleInput as `str`
        # It consolidates all the fields of SampleInput and allows,
        # formatting the fields like `input`, `args`, etc with `formatter`
        # callable to customize the representation.
        # Look at `summary` method for example.
        arguments = [
            f"input={formatter(self.input)}",
            f"args={formatter(self.args)}",
            f"kwargs={formatter(self.kwargs)}",
            f"broadcasts_input={self.broadcasts_input}",
            f"name={repr(self.name)}",
        ]

        return f"SampleInput({', '.join(a for a in arguments if a is not None)})"

    def __repr__(self):
        return self._repr_helper(lambda x: x)

    def summary(self):
        # Returns the SampleInput details in a more
        # friendly format.
        # It formats `Tensor` and `TensorList`
        # in a more condensed representation.
        def formatter(arg):
            # Format any instance of `Tensor` (standalone, in list, or in dict)
            # by Tensor[TensorShape]
            # Eg. Tensor with shape (3, 4) is formatted as Tensor[3, 4]
            if isinstance(arg, torch.Tensor):
                shape = str(tuple(arg.shape))
                dtype = str(arg.dtype)
                device = str(arg.device)
                contiguity_suffix = ""
                # NB: sparse CSR tensors annoyingly return is_sparse=False
                is_sparse = arg.is_sparse or arg.layout == torch.sparse_csr
                if not is_sparse and not arg.is_contiguous():
                    contiguity_suffix = ", contiguous=False"
                return f'Tensor[size={shape}, device="{device}", dtype={dtype}{contiguity_suffix}]'
            elif isinstance(arg, dict):
                return {k: formatter(v) for k, v in arg.items()}
            elif is_iterable_of_tensors(arg):
                return "TensorList[" + ", ".join(map(formatter, arg)) + "]"
            elif isinstance(arg, (list, tuple)):  # Handle list, tuple
                return "(" + ",".join(map(formatter, arg)) + ")"

            return repr(arg)

        return self._repr_helper(formatter)

    # Applies the transform f(t) -> t to each tensor and dtype in the SampleInput
    def transform(self, f):
        def tt(t):
            def _tt(t):
                with torch.no_grad():
                    return f(t)

            if isinstance(t, torch.Tensor):
                return _tt(t)
            elif isinstance(t, torch.dtype):
                return _tt(t)
            elif isinstance(t, list):
                return list(map(tt, t))
            elif isinstance(t, tuple):
                return tuple(map(tt, t))
            elif isinstance(t, dict):
                return {k: tt(v) for k, v in t.items()}
            else:
                return t

        sample_tt_input, tt_args, tt_kwargs = (
            tt(self.input),
            tt(self.args),
            tt(self.kwargs),
        )

        # Note the transformed SampleInput assumes metadata like output_process_fn_grad is still valid!
        return SampleInput(
            sample_tt_input,
            args=tt_args,
            kwargs=tt_kwargs,
            output_process_fn_grad=self.output_process_fn_grad,
            broadcasts_input=self.broadcasts_input,
            name=self.name + "_transformed",
        )

    # Returns the NumPy version of the sample input object in the form of a tuple: (input, args, kwargs)
    # Converts tensors to ndarrays by calling .detach().cpu().numpy() on them
    # Converts dtypes by remapping them using torch_to_numpy_dtype_dict
    def numpy(self):
        def to_numpy(t):
            if isinstance(t, torch.Tensor):
                if t.dtype is torch.bfloat16:
                    return t.detach().cpu().to(torch.float32).numpy()
                if t.dtype is torch.chalf:
                    return t.detach().cpu().to(torch.cfloat).numpy()
                return t.detach().cpu().numpy()
            elif isinstance(t, torch.dtype):
                return torch_to_numpy_dtype_dict[t]

            return t

        return self.transform(to_numpy)

    def noncontiguous(self):
        def to_noncontiguous(t):
            if isinstance(t, torch.Tensor):
                return noncontiguous_like(t)
            elif isinstance(t, torch.dtype):
                return t

            return t

        return self.transform(to_noncontiguous)


NumericsFilter = collections.namedtuple("NumericsFilter", ["condition", "safe_val"])


class ErrorInput:
    """
    A SampleInput that will cause the operation to throw an error plus information
    about the resulting error.
    """

    __slots__ = ["sample_input", "error_type", "error_regex"]

    def __init__(self, sample_input, *, error_type=RuntimeError, error_regex):
        self.sample_input = sample_input
        self.error_type = error_type
        self.error_regex = error_regex


class AliasInfo:
    """Class holds alias information. For example, torch.abs ->
    torch.absolute, torch.Tensor.absolute, torch.Tensor.absolute_
    """

    def __init__(self, alias_name):
        self.name = alias_name
        self.op = _getattr_qual(torch, alias_name)
        self.method_variant = getattr(torch.Tensor, alias_name, None)
        self.inplace_variant = getattr(torch.Tensor, alias_name + "_", None)

    def __call__(self, *args, **kwargs):
        return self.op(*args, **kwargs)


# Note [OpInfos]
# ~~~~~~~~~~~~~~
#
# The majority of this note was written shortly after the PyTorch 1.9 release.
# If you notice it's out-of-date or think it could be improved then please
# file an issue.
#
# See also: the OpInfo tracker (https://github.com/pytorch/pytorch/issues/54261)
# See also: "Writing Test Templates" in common_device_type.py to learn how to
#   parametrize a test template using OpInfos.
# See also: PyTorch's GitHub wiki on running and writing tests
#   https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests
# See also: ModuleInfos, OpInfo's sister class, defined in common_modules.py
#
# An OpInfo is a collection of metadata related to a PyTorch operator. This
#   metadata is used to generate tests that validate properties of the operator,
#   like if it implements the correct gradient formula.
#
# WHY OPINFOS?
# ~~~~~~~~~~~~
#
# OpInfos are principally intended to do three things:
#
#   1) to allow systematic testing over all PyTorch's operators
#   2) to simplify operating testing by autogenerating many tests
#   3) to allow systems (like autograd, torchscript, fx, nnc...) to test
#        against every PyTorch operator
#
# All these goals are still a work in progress. Not every operator has an
#   OpInfo, and some operator tests that could be automatically generated
#   still have to be written manually.
#
# It's helpful to understand that OpInfos are both about test simplification and
#   modularity. PyTorch is a complicated framework with many interrelated systems,
#   too many for any one person to keep track of. An OpInfo can be thought of as the
#   interface between an operator implementer and those other systems. Instead of
#   requiring the implementer of torch.foo understand how to test its forward
#   mode AD or NNC support that's typically handled automatically just by
#   defining an OpInfo.
#
# It's often surprising to OpInfo writers that just implementing an OpInfo
#   typically can't verify an operator is actually implemented correctly:
#
# "If an OpInfo doesn't validate my op works as expected, what's the point
#     of it?"
#
# But the point of is the above. OpInfos are intended to let you focus on testing
#   the operator logic you're familiar with instead of having to write tests for
#   how the operator interacts with each of PyTorch's many systems.
#
# And, OK, it turns out that SOMETIMES just writing an OpInfo DOES
#   validate your op works as expected, but that's only in special
#   cases. See below for details.
#
# WHAT'S AN OPINFO?
# ~~~~~~~~~~~~~~~~~
#
# So what is an OpInfo? It's a Python class that describes an operator's properties,
#   like which dtypes it supports on the CPU and whether it has any aliases.
#   These properties can be divided into three categories:
#
#   1) Metadata describing the operator, like the operator's name and if it
#     "supports" the out kwarg.
#   2) Test directives, like "skips" that tell the test suite to skip some
#     tests.
#   3) A "sample inputs" function that generates valid inputs for the operator.
#
# OpInfo attributes are described in more detail below.
#
# THE SAMPLE INPUTS FUNCTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The "sample inputs" function merits special elaboration. This function is
#   crucial to testing with OpInfos. A typical OpInfo test has to treat the operator
#   as a black box. There's no structure for the test to understand or exploit.
#   Without "sample inputs" it wouldn't even know how to call the OpInfo's
#   operator. The sample input function saves the day by providing different
#   "SampleInputs" that can be used to call the operator. A sample input
#   function should have the following signature:
#
#   def sample_inputs_foo(op_info, device, dtype, requires_grad, **kwargs):
#
#   And should return an iterable of SampleInputs (see the class description
#   above). Each SampleInput defines an "input", "args", "kwargs", an
#   "output_process_fn_grad" function, the "broadcasts_input" bool and a
#   "name".
#
#   All the "sample_inputs" functions are invoked within a `torch.no_grad()`
#   environment for efficiency and correctness. As such remember to set the
#   "requires_grad" flag on the inputs **after** performing any transformations
#   on them.
#
# The "input" is the first argument to the operator, or the tensor that
#   the method or inplace variants of the operator should be called on, and
#   should be on the requested device, of the requested dtype, and its
#   requires_grad attribute should be set to the requires_grad argument.
#
# "args" should contain positional arguments, and "kwargs" keyword arguments.
#
# "output_process_fn_grad" has an interesting name. It's a function that maps
#   the operator's output (when given the input, args, and kwargs) to the
#   portion of the output to gradcheck. For example, consider an operator
#   like torch.linalg.slogdet
#   (https://pytorch.org/docs/main/generated/torch.linalg.slogdet.html).
#   This operator returns a tuple of two tensors, but the first tensor
#   cannot be backwarded through. Its "output_process_fn_grad" filters
#   this output tuple to just the second argument, which we can call backward
#   on. Functions that produce a single tensor can ignore this argument.
#
# "broadcasts_input" is a bool indicated if the SampleInput causes the operator
#   to broadcast the "input" argument. This is important for tests to understand
#   because inplace variants of operations throw a runtime error if they
#   would broadcast their input arguments, so tests that work with inplace
#   variants filter SampleInputs that broadcast their input.
#
# "name" is a string that's just used for debugging. It appears when printing
#   the SampleInput.
#
# Sample inputs are designed to be used with many tests, some
#   that are very time consuming, so they should be a small
#   set with small tensors. An elaborated set of sample inputs
#   can be specified using the "reference_inputs_func" attribute.
#   The "reference inputs" for an operation are an extended
#   set of sample inputs that can more exhaustively test an
#   operator. They are used by only a few tests that are careful
#   not to take too long to run. Adding reference inputs
#   is highly encouraged!
#
# THE (OPTIONAL) ERROR INPUTS FUNCTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# OpInfos may optionally specify "error inputs" through an error function. If
#   specified test_errors in test_ops.py will call the op with these inputs
#   and validate that the desired error is thrown.
#
# Error inputs automate a common testing pattern where multiple inputs are
#   passed to an operation and the errors they thrown are reviewed. Tests
#   written in this style should be ported to the new OpInfo pattern.
#
# Error inputs are specified using the ErrorInputs class, which contains
#   a SampleInput (see above) and data about the expected error.
#
# OPINFO FILE ORGANIZATION
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# All OpInfos are currently defined in this file. Most OpInfo tests are defined
#   in test_ops.py, but some system-specific tests are defined in those
#   systems' test files, and subclass-specific tests are defined in the test
#   file that corresponds to that subclass (see the below).
#   Expect a reorganization in the future.
#
# WHAT'S TESTED?
# ~~~~~~~~~~~~~~
#
# Every OpInfo in the op_db sequence has the following properties validated in
# test_ops.py:
#
#   - that its supported dtypes are specified correctly
#   - that the operation produces the same results when called with noncontiguous inputs
#   - that it supports the out= argument properly (if it allows out=),
#       see https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-does-out-work-in-pytorch
#   - that it works with the conjugate view bit properly
#   - that its function, method, and inplace variants perform the same operation
#       (that is, that torch.add, torch.Tensor.add, and torch.Tensor.add_ all
#       do the same thing).
#   - that its inplace variant preserves the input's storage
#   - that its gradient formula is implemented correctly, and that it supports
#       gradgrad and complex grad and gradgrad and forward mode AD properly for
#       the op's function and inplace variants (method variants are skipped
#       to reduce test time).
#   - that the operation performs the same operation when traced or scripted
#       using the jit
#   - that the operation is autodifferentiated by the jit as expected
#   - that the operator's aliases, if any, perform the same operation and that
#       the jit understands the alias
#   - that the operator throws the correct errors (if error_inputs is defined)
#   - that the operator produces the same results as a NumPy reference (if ref is defined)
#   - that the operator produces the same results as a NumPy reference on an extended
#       set of "reference inputs" (if both ref and reference_inputs_func are defined)
#       (NOTE: elementwise unary and elementwise binary OpInfos do this even if only
#         ref is defined, because they effectively autogenerate reference inputs)
#   - that the operator works on different CUDA devices
#
# Additional OpInfo tests are in test_jit_fuser_te.py, test_fx_experimental.py,
#   and test_fx.py. These tests validate that operators work with NNC and FX
#   as expected.
#
# For performance, some of the above tests may only run on the first
#   SampleInput returned by an OpInfo's sample input function.
#
# In addition to these tests, some subclasses (discussed in the next section)
#   define additional tests.
#
# Critically, as mentioned above, what's not necessarily tested is that the operator
#   works as expected. When implementing an OpInfo an engineer must still
#   typically write one or more tests validating the operator's behavior.
#   The exception to this is if reference testing is sufficient, or if
#   the operation belongs to an OpInfo subclass that has more exhaustive
#   operator testing. Elementwise unary and elementwise binary operators,
#   in particular, usually don't require additional testing beyond
#   writing an Opinfo.
#
#
# OPINFO (SUB)CLASSES
# ~~~~~~~~~~~~~~~~~~~
#
# In addition to the OpInfo base class there are several specialized OpInfo
#   subclasses. For example, the UnaryUfuncInfo subclass is used for
#   unary elementwise operations. These operations have a common structure
#   that test_unary_ufuncs.py exploits with additional automated testing.
#   The automated testing in test_unary_ufuncs.py is so thorough, comparing
#   the operator to a NumPy reference function on a plethora of values, that
#   just implementing an OpInfo for a unary elementwise operation is often
#   sufficient testing.
#
# The ForeachFuncInfo is another OpInfo subclass that is hyper-specialized to a
#   very unique class of operations. These OpInfos aren't included in the
#   op_db sequence and have their own tests.
#
# Other OpInfo subclasses, like SpectralFuncInfo, are just for convenience
# when writing OpInfos.
#
# TESTING A NEW OPERATOR
# ~~~~~~~~~~~~~~~~~~~~~~
#
# If you're adding a new operator to any of the following namespaces:
#   - torch
#   - torch.fft
#   - torch.linalg,
#   - torch.special
#   - torch.nn.functional
# then you should typically add an OpInfo for it.
#
# As mentioned a couple times above, implementing an OpInfo is not
#   usually sufficient testing (unless the operator is a unary or binary elementwise
#   operator). The OpInfo will only test the properties described in the
#   "WHAT'S TESTED" section. It DOES NOT necessarily verify that the operator is
#   implemented correctly.
#
# TIPS FOR WRITING AN OPINFO AND OPINFO TESTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Writing an OpInfo can be a little daunting. Since the point of an OpInfo is to
#   be consumed by a variety of systems it can be hard to understand how to
#   deal with test failures or how to set the OpInfo metadata properly.
#
# Before adding an OpInfo it helps to look at other OpInfos. A sample inputs
#   function must be defined, and the operator's dtypes must be specified.
#   Once that's done you should run the operator's tests in test_ops.py
#   (these can be filtered using the "-k" argument in pytest). Tests that
#   fail should provide an error message that describes what to change about
#   your OpInfo. You don't need to worry about changing an OpInfo's default
#   values unless a test yells at you.
#
# Similarly, if you're writing a test that consumes OpInfos then it's critical
#   your test provides a clear error message describing what to do when it
#   fails. You should not assume the OpInfo implementer is familiar with your
#   system.
#
# If you see a confusing error message while developing an OpInfo then please
#   file an issue describing what happened.
#
# This trial-and-error approach to writing an OpInfo can be frustrating,
#   but it's probably necessary as long as OpInfos don't require
#   learning about all the systems that consume them. One thing that can help
#   is the get_supported_dtypes() function defined in utils.py. This
#   function can be used to programmatically specify the dtypes an operator
#   supports, and is especially useful if writing an OpInfo on a machine
#   without a CUDA device. See its documentation for more details.
#
# THE FUTURE OF OPINFOS AND OPINFO TESTING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the future we expect OpInfo coverage to improve and cover
#   the great majority of PyTorch's (public) operators.
#


# Classes and methods for the operator database
@dataclass
class OpInfo:
    """Operator information and helper functions for acquiring it."""

    # the string name of the function
    name: str

    # An optional reference function that accepts ndarrays (AKA "NumPy arrays").
    # If given, the op will be compared with its reference on each of its sample inputs.
    ref: Optional[Callable] = None

    # the following metadata describes the operator, its variants, and its aliases, if any

    # iterable of aliases, e.g. ("absolute",) for torch.abs
    aliases: Iterable = None

    # additional string to include in the test name
    # this is useful when an op needs multiple OpInfos,
    # like divide does, often because it's really several
    # different ops behind the scenes
    variant_test_name: str = ""

    # the function variant of the operation, populated as torch.<name> if None
    op: Callable = None

    # allows the method variant of this operation to be specified as follows:
    # - if _NOTHING (default), then the OpInfo attempts to discover the variant using its name
    # - if None, then the OpInfo explicitly specifies is has no associated method
    # - if a Callable, then that callable should be the method associated with this operation
    method_variant: Callable = _NOTHING

    # allows the inplace variant of this operation to be specified as follows:
    # - if _NOTHING (default), then the OpInfo attempts to discover the variant using its name
    # - if None, then the OpInfo explicitly specifies is has no associated inplace variant
    # - if a Callable, then that callable should be the inplace variant associated with this operation
    inplace_variant: Callable = _NOTHING

    # allows the operator variant of this operation to be specified as follows:
    # - if _NOTHING (default), then the OpInfo attempts to discover the variant using its name
    # - if None, then the OpInfo explicitly specifies is has no associated operator
    # - if a Callable, then that callable should be the operator associated with this operation
    operator_variant: Callable = _NOTHING

    # allows the inplace operator variant of this operation to be specified as follows:
    # - if _NOTHING (default), then the OpInfo attempts to discover the variant using its name
    # - if None, then the OpInfo explicitly specifies is has no associated inplace operator
    # - if a Callable, then that callable should be the inplace operator associated with this operation
    inplace_operator_variant: Callable = _NOTHING

    # the following metadata are test directives for skipping or modifying tests

    # information about which tests to skip
    skips: tuple = ()

    # decorators to apply to generated tests
    decorators: tuple = ()

    # the following are pointers to functions to generate certain classes of inputs

    # function to generate sample inputs with strided layouts
    sample_inputs_func: Callable = None

    # function to generate a more thorough set of samples inputs with strided layouts
    reference_inputs_func: Callable = None

    # function to generate inputs that will throw errors
    error_inputs_func: Callable = None

    # function to generate sparse (coo, csr, csc, bsr, bsc) inputs that will throw errors
    error_inputs_sparse_func: Callable = None

    # function to generate sample inputs with sparse coo layouts
    sample_inputs_sparse_coo_func: Callable = None

    # function to generate sample inputs with sparse csr layouts
    sample_inputs_sparse_csr_func: Callable = None

    # function to generate sample inputs with sparse csc layouts
    sample_inputs_sparse_csc_func: Callable = None

    # function to generate sample inputs with sparse bsr layouts
    sample_inputs_sparse_bsr_func: Callable = None

    # function to generate sample inputs with sparse bsc layouts
    sample_inputs_sparse_bsc_func: Callable = None

    # the following metadata relates to dtype support and is tested for correctness in test_ops.py

    # dtypes this function works with on the CPU,
    # inherited by other device types that don't specify their own dtypes
    dtypes: _dispatch_dtypes = None

    # the following dtypesIf... options override the dtypes value on their respective device types
    # I.e. instead of writing multiple `dtypesIfCUDA`, `dtypesIfROCM`, etc one can simply define a dict
    # dtypesIf = { 'cuda': (torch.float, torch.double), 'rocm': (torch.half, torch.bfloat16) }
    dtypesIf: dict[str, _dispatch_dtypes] = field(default_factory=dict)

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("dtypesIf") and name != "dtypesIf":
            # TODO: Warn if used
            dev_name = name.removeprefix("dtypesIf").lower()
            return self.dtypesIf.get(dev_name)
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        # TODO: After migration, start adding warnings here
        if name.startswith("dtypesIf") and name != "dtypesIf":
            assert isinstance(value, (_dispatch_dtypes, type(None)))
            dev_name = name.removeprefix("dtypesIf").lower()
            self.dtypesIf[dev_name] = value
            return
        super().__setattr__(name, value)

    # dtypes this function is expected to work with on CUDA
    dtypesIfCUDA: _dispatch_dtypes = None

    # dtypes this function is expected to work with on ROCM
    dtypesIfROCM: _dispatch_dtypes = None

    dtypesIfHpu: _dispatch_dtypes = None

    # dtypes this function is expected to work with on XPU
    dtypesIfXPU: _dispatch_dtypes = None

    # backward dtypes this function is expected to work with
    backward_dtypes: _dispatch_dtypes = None

    # backward dtypes this function is expected to work with on CUDA
    backward_dtypesIfCUDA: _dispatch_dtypes = None

    # backward dtypes this function is expected to work with on ROCM
    backward_dtypesIfROCM: _dispatch_dtypes = None

    backward_dtypesIfHpu: _dispatch_dtypes = None

    # the following metadata describes the operators out= support

    # whether the op supports the out kwarg
    # defaults to True, if the op does not allow the out kwarg or
    # supports it incorrectly then test_out in test_ops.py should fail
    supports_out: bool = True

    # the following metadata relates to autograd support
    # whether the operation supports backward mode AD
    # if true, gradient correctness is tested in test_ops.py
    # using the op's sample inputs
    supports_autograd: bool = True

    # whether the op supports second order gradients
    # if true, gradgrad correctness is tested in test_ops.py
    # defaults to support_autograd's value
    # TODO: rename this to supports_bwgrad_bwgrad to be consistent with below
    supports_gradgrad: bool = None

    # whether the ops supports second order gradients via
    # forward-over-reverse. If True, forward-over-reverse gradgrad correctness
    # is tested. If False, test that forward grad is not implemented.
    # Defaults to False.
    supports_fwgrad_bwgrad: bool = False

    # whether the operation supports inplace autograd
    # if true, tested in test_ops.py
    # defaults to supports_autograd's value
    supports_inplace_autograd: bool = None

    # Whether the operation support forward mode AD
    # If the value is True, we check that the gradients are correct
    # If the value is False, we test that forward grad is not implemented
    supports_forward_ad: bool = False

    # Whether the operation has a varargs variant
    # (e.g. functions like ones, zeros, methods like view, permute)
    supports_varargs: bool = False

    # Whether the forward operation avoids materializing COW tensor inputs
    supports_cow_input_no_materialize_forward: bool = True

    # Whether the backward operation avoids materializing COW tensor inputs
    supports_cow_input_no_materialize_backward: bool = True

    # Whether to skip the backward part of the COW tensor input test
    skip_cow_input_backward: bool = False

    # If `supports_cow_input_no_materialize_forward == True`, this list contains
    # the arg indices or kwarg names of inputs that are expected to materialize
    allow_cow_input_materialize_forward: list[Union[int, str]] = None

    # If `supports_cow_input_no_materialize_backward == True`, this list contains
    # the arg indices or kwarg names of inputs that are expected to materialize
    allow_cow_input_materialize_backward: list[Union[int, str]] = None

    # wrapper function for gradcheck
    gradcheck_wrapper: Callable = lambda op, *args, **kwargs: op(*args, **kwargs)

    # whether to check batched grad when doing gradcheck
    # defaults to support_autograd's value
    check_batched_grad: bool = None

    # whether to check batched grad grad when doing gradgradcheck
    # default's to support_gradgrad's value
    check_batched_gradgrad: bool = None

    # whether to check batched forward grad when doing gradcheck
    # defaults to the value of `supports_forward_ad`
    check_batched_forward_grad: bool = None

    # whether to check batched forward grad when doing gradcheck
    # defaults to the value of `check_batched_forward_grad`
    check_inplace_batched_forward_grad: bool = None

    # tolerance for nondeterminism while performing gradcheck
    gradcheck_nondet_tol: float = 0.0

    # Whether to use the fast implementation for gradcheck/gradgradcheck.
    # When set to None, defers to the default value provided by the wrapper
    # function around gradcheck (testing._internal.common_utils.gradcheck)
    gradcheck_fast_mode: bool = None

    # the following metadata relates to JIT support and is tested for correctness in test_ops.py

    # name of the corresponding aten:: operator
    aten_name: str = None

    # if this is a composite implicit autograd op, the decomposed op
    decomp_aten_name: Optional[str] = None

    # name of the corresponding aten:: operator for backwards
    aten_backward_name: Optional[str] = None

    # if a op's aten::node is expected to be symbolically autodiffed
    assert_autodiffed: bool = False

    # a list of strings with node names that are expected to be in a
    # DifferentiableGraph when autodiffed. Ex: ['aten::add', 'aten::mm'],
    # default is populated to be ['aten::(name of Python operator)']
    autodiff_nonfusible_nodes: list[str] = None

    # a list of strings with node names that are expected to be in FusionGroups
    # inside of DifferentiableGraphs when this operation is autodiffed.
    # Ex: ['aten::add', 'aten::mm'], defaults to an empty list
    # Note: currently no ops use fusible nodes
    autodiff_fusible_nodes: list[str] = None

    # the following metadata relates to sparse support and is used in test_sparse.py

    # whether the op supports sparse coo inputs, defaults to False
    # TODO: rename supports_sparse to supports_sparse_coo
    supports_sparse: bool = None

    # only run tracing tests
    supports_scripting: bool = True

    # if the operator can be traced
    supports_tracing: bool = True

    # the following metadata relates to sparse compressed support and
    # is used in test_sparse_csr.py and test_sparse.py

    # whether the op supports sparse csr inputs, defaults to False
    supports_sparse_csr: bool = None
    # whether the op supports sparse csc inputs, defaults to False
    supports_sparse_csc: bool = None
    # whether the op supports sparse bsr inputs, defaults to False
    supports_sparse_bsr: bool = None
    # whether the op supports sparse bsc inputs, defaults to False
    supports_sparse_bsc: bool = None
    # whether the op supports nested jagged inputs, defaults to False
    supports_njt: bool = None

    # whether the op promotes integer inputs to float
    promotes_int_to_float: bool = False

    # the following metadata relates to complex support and is checked in test_ops.py

    test_conjugated_samples: bool = True

    test_neg_view: bool = True

    # assert that jit shape analysis fully propagates shape
    assert_jit_shape_analysis: bool = False

    # the following metadata relates to ExpandedWeights support and is checked in test_expanded_weights.py

    supports_expanded_weight: bool = False

    is_factory_function: bool = False

    skip_correctness_check_compile_vs_eager: bool = False

    def __post_init__(self):
        self._original_opinfo_args = asdict(self).copy()

        assert self.dtypes is not None, f"OpInfo for {self.name} has no dtypes!"

        # Validates the dtypes are generated from the dispatch-related functions
        for name, val in self.dtypesIf.items():
            if val is not None:
                assert isinstance(val, _dispatch_dtypes)
                self.dtypesIf[name] = set(val)

        if self.aten_name is None:
            self.aten_name = self.name

        # Attribute to verify dynamic_dtypes are used.
        self.dynamic_dtypes = any(
            isinstance(dtypes, utils._dynamic_dispatch_dtypes)
            for dtypes in self.dtypesIf.values()
        )

        if self.dynamic_dtypes:
            # Make sure `dtyesIfCUDA` is dynamic, if dynamic dispatch is used for CPU
            # This is because, below we set dtypesIfCUDA to dtypes if they are None.
            assert isinstance(self.dtypesIfCUDA, utils._dynamic_dispatch_dtypes), (
                f"To use dynamic dtypes for operator {self.name}, "
                "acquire the dtypes dynamically for argument `dtypesIfCUDA`."
                "This is to ensure that CUDA dtypes are acquired correctly as they"
                "differ from CPU dtypes occasionally"
            )

        self.dtypes = set(self.dtypes)

        # NOTE: backward dtypes must be acquired before forward dtypes
        #   since they fallback to explicit (not implicit!) specifications of
        #   forward dtypes
        self.backward_dtypesIfROCM = (
            set(self.backward_dtypesIfROCM)
            if self.backward_dtypesIfROCM is not None
            else (
                self.backward_dtypesIfCUDA
                if self.backward_dtypesIfCUDA is not None
                else self.backward_dtypes
                if self.backward_dtypes is not None
                else self.dtypesIfROCM
                if self.dtypesIfROCM is not None
                else self.dtypesIfCUDA
                if self.dtypesIfCUDA is not None
                else self.dtypes
            )
        )
        self.backward_dtypesIfCUDA = (
            set(self.backward_dtypesIfCUDA)
            if self.backward_dtypesIfCUDA is not None
            else (
                self.backward_dtypes
                if self.backward_dtypes is not None
                else self.dtypesIfCUDA
                if self.dtypesIfCUDA is not None
                else self.dtypes
            )
        )
        self.backward_dtypesIfHpu = (
            set(self.backward_dtypesIfHpu)
            if self.backward_dtypesIfHpu is not None
            else (
                self.backward_dtypes
                if self.backward_dtypes is not None
                else self.dtypes
            )
        )

        self.backward_dtypes = (
            set(self.backward_dtypes)
            if self.backward_dtypes is not None
            else self.dtypes
        )

        # Inherit from cpu
        for dev_type in ["cuda", "hpu"]:
            if self.dtypesIf.get(dev_type) is None:
                self.dtypesIf[dev_type] = self.dtypes

        # Inherit from CUDA
        for dev_type in ["rocm", "xpu"]:
            if self.dtypesIf.get(dev_type) is None:
                self.dtypesIf[dev_type] = self.dtypesIf["cuda"]

        # NOTE: if the op is unspecified it is assumed to be under the torch namespace
        if not self.op:
            self.op = _getattr_qual(torch, self.name)

        if self.method_variant is _NOTHING:
            self.method_variant = getattr(torch.Tensor, self.name, None)

        # attributes like real, imag are not callable
        if not callable(self.method_variant):
            self.method_variant = None

        if self.inplace_variant is _NOTHING:
            inplace_name = self.name + "_"
            self.inplace_variant = getattr(torch.Tensor, inplace_name, None)

        if self.operator_variant is _NOTHING:
            self.operator_variant = getattr(operator, self.name, None)

        if self.inplace_operator_variant is _NOTHING:
            # Note: operator.i<op> will use operator.<op> and assign the result to the lhs when no
            # __i<op>__ method is found. This results in the appearance of an inplace operator variant which
            # does not have the correct inplace behavior. To avoid this, we guard automatic detection of the inplace
            # operator with a check that an inplace variant exists.
            if self.inplace_variant is not None:
                inplace_operator_name = "i" + self.name
                self.inplace_operator_variant = getattr(
                    operator, inplace_operator_name, None
                )
            else:
                self.inplace_operator_variant = None

        self.decorators = (*self.decorators, *self.skips)

        # Specifying sample inputs function without specifying the
        # corresponding layout support implies the layout support:
        if self.supports_sparse is None:
            self.supports_sparse = self.sample_inputs_sparse_coo_func is not None
        if self.sample_inputs_sparse_coo_func is None:
            self.sample_inputs_sparse_coo_func = self._sample_inputs_unspecified

        if self.supports_sparse_csr is None:
            self.supports_sparse_csr = self.sample_inputs_sparse_csr_func is not None
        if self.sample_inputs_sparse_csr_func is None:
            self.sample_inputs_sparse_csr_func = self._sample_inputs_unspecified

        if self.supports_sparse_csc is None:
            self.supports_sparse_csc = self.sample_inputs_sparse_csc_func is not None
        if self.sample_inputs_sparse_csc_func is None:
            self.sample_inputs_sparse_csc_func = self._sample_inputs_unspecified

        if self.supports_sparse_bsr is None:
            self.supports_sparse_bsr = self.sample_inputs_sparse_bsr_func is not None
        if self.sample_inputs_sparse_bsr_func is None:
            self.sample_inputs_sparse_bsr_func = self._sample_inputs_unspecified

        if self.supports_sparse_bsc is None:
            self.supports_sparse_bsc = self.sample_inputs_sparse_bsc_func is not None
        if self.sample_inputs_sparse_bsc_func is None:
            self.sample_inputs_sparse_bsc_func = self._sample_inputs_unspecified

        if self.supports_njt is None:
            self.supports_njt = False

        # We run the sampling functions without tracking the gradiends of the creation of inputs
        self.sample_inputs_func = torch.no_grad()(self.sample_inputs_func)
        self.sample_inputs_sparse_coo_func = torch.no_grad()(
            self.sample_inputs_sparse_coo_func
        )
        self.sample_inputs_sparse_csr_func = torch.no_grad()(
            self.sample_inputs_sparse_csr_func
        )
        self.sample_inputs_sparse_csc_func = torch.no_grad()(
            self.sample_inputs_sparse_csc_func
        )
        self.sample_inputs_sparse_bsr_func = torch.no_grad()(
            self.sample_inputs_sparse_bsr_func
        )
        self.sample_inputs_sparse_bsc_func = torch.no_grad()(
            self.sample_inputs_sparse_bsc_func
        )
        if self.reference_inputs_func is not None:
            self.reference_inputs_func = torch.no_grad()(self.reference_inputs_func)

        if not self.autodiff_fusible_nodes:
            self.autodiff_fusible_nodes = []

        if self.autodiff_nonfusible_nodes is None:
            self.autodiff_nonfusible_nodes = ["aten::" + self.name]

        # Autograd support

        # Autograd flags that depend on backward AD only
        # - If setting has been explicitly set, raise error if inconsistent
        if self.supports_gradgrad is None:
            self.supports_gradgrad = self.supports_autograd
        else:
            assert not (self.supports_gradgrad and not self.supports_autograd), (
                "supports_gradgrad refines the part of autograd is supported, so it should "
                "not be set if supports_autograd is False"
            )
        if self.check_batched_grad is None:
            self.check_batched_grad = self.supports_autograd or self.supports_forward_ad
        else:
            assert not (
                self.check_batched_grad
                and not (self.supports_autograd or self.supports_forward_ad)
            ), (
                "check_batched_grad refines the part of autograd that will be checked (by gradcheck), so "
                "it should not be set if supports_autograd is False"
            )
        if self.check_batched_gradgrad is None:
            self.check_batched_gradgrad = self.supports_gradgrad
        else:
            assert not (self.check_batched_gradgrad and not self.supports_gradgrad), (
                "check_batched_gradgrad refines the part of autograd that will be checked (by "
                "gradgradcheck), so it should not be set if either supports_gradgrad or supports_autograd "
                "is False."
            )
        if self.check_batched_forward_grad is None:
            self.check_batched_forward_grad = self.supports_forward_ad
        else:
            assert not (
                self.check_batched_forward_grad and not self.supports_forward_ad
            ), (
                "check_batched_forward_grad should only be used when supports_forward_ad "
                "is True. It is used to disable the test in the specific cases "
                "where the op supports forward ad but fails to compute "
                "batched forward grad."
            )

        if self.check_inplace_batched_forward_grad is None:
            self.check_inplace_batched_forward_grad = self.check_batched_forward_grad
        else:
            assert not (
                self.check_inplace_batched_forward_grad
                and not self.check_batched_forward_grad
            ), (
                "check_batched_forward_grad should only be used when check_batched_forward_grad "
                "is True. It is used to disable the test in the specific cases "
                "where the op supports batched forward grad but fails to compute batched forward "
                "grad for the inplace variant of the op."
            )

        assert not (self.supports_fwgrad_bwgrad and not self.supports_autograd), (
            "supports_fwgrad_bwgrad enables forward-over-backward gradgrad checks and should only be "
            "True if backward ad is also checked, i.e., supports_forward_ad should be True.",
            self.name,
        )

        # Autograd flags that depend on both forward AD and backward AD
        if self.supports_inplace_autograd is None:
            self.supports_inplace_autograd = (
                self.supports_autograd or self.supports_forward_ad
            )
        else:
            assert not (
                self.supports_inplace_autograd
                and not self.supports_autograd
                and not self.supports_forward_ad
            ), (
                "supports_inplace_autograd refines the part of autograd that is supported, so "
                "it should not be set if both supports_autograd and supports_forward_ad are False"
            )

        if self.aliases is not None:
            self.aliases = tuple(AliasInfo(a) for a in self.aliases)  # type: ignore[assignment]
        else:
            self.aliases = ()

    def __call__(self, *args, **kwargs):
        """Calls the function variant of the operator."""
        return self.op(*args, **kwargs)

    def __str__(self):
        return dataclass_repr(self)

    def get_op(self):
        """Returns the function variant of the operator, torch.<op_name>."""
        return self.op

    def get_method(self):
        """Returns the m
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/testing/_internal/opinfo`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/testing/_internal/opinfo`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/torch/testing/_internal/opinfo/core.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/testing/_internal/opinfo`):

- [`refs.py_kw.md_docs.md`](./refs.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`core.py_kw.md_docs.md`](./core.py_kw.md_docs.md)
- [`refs.py_docs.md_docs.md`](./refs.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`utils.py_kw.md_docs.md`](./utils.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `core.py_docs.md_docs.md`
- **Keyword Index**: `core.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
