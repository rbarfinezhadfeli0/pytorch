# Documentation: `test/test_overrides.py`

## File Metadata

- **Path**: `test/test_overrides.py`
- **Size**: 68,260 bytes (66.66 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# Owner(s): ["module: __torch_function__"]

import sys
import torch
import numpy as np
import inspect
import functools
import pprint
import pickle
import collections
import unittest
import os

from torch.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_CROSSREF
from torch.overrides import (
    handle_torch_function,
    has_torch_function,
    get_ignored_functions,
    get_overridable_functions,
    get_testing_overrides,
    resolve_name,
    is_tensor_method_or_property,
    TorchFunctionMode,
    _get_current_function_mode,
    _get_current_function_mode_stack,
    BaseTorchFunctionMode
)
from torch.utils._mode_utils import all_same_mode
from torch.utils._pytree import tree_map

Tensor = torch.Tensor

if os.getenv("ATEN_CPU_CAPABILITY") in ("default", "avx2"):
    # This test is not supported on ARM
    print(
        "Skipping due to failing when cuda build runs on non cuda machine, "
        + "see https://github.com/pytorch/pytorch/pull/150059 for example"
    )
    sys.exit()

# The functions below simulate the pure-python torch functions in the
# torch.functional namespace. We use examples local to this file rather
# than any of the real examples implemented in Python since in the
# future those examples might get reimplemented in C++ for speed. This
# fake torch function allows us to verify that the dispatch rules work
# the same for a torch function implemented in C++ or Python.

def foo(a, b, c=None):
    """A function multiple arguments and an optional argument"""
    if has_torch_function((a, b, c)):
        return handle_torch_function(foo, (a, b, c), a, b, c=c)
    if c:
        return a + b + c
    return a + b

def bar(a):
    """A function with one argument"""
    if has_torch_function((a,)):
        return handle_torch_function(bar, (a,), a)
    return a

def baz(a, b):
    """A function with multiple arguments"""
    if has_torch_function((a, b)):
        return handle_torch_function(baz, (a, b), a, b)
    return a + b

def quux(a):
    """Used to test that errors raised in user implementations get propagated"""
    if has_torch_function((a,)):
        return handle_torch_function(quux, (a,), a)
    return a

# HANDLED_FUNCTIONS_DIAGONAL is a dispatch table that
# DiagonalTensor.__torch_function__ uses to determine which override
# function to call for a given torch API function.  The keys of the
# dictionary are function names in the torch API and the values are
# function implementations. Implementations are added to
# HANDLED_FUNCTION_DIAGONAL by decorating a python function with
# implements_diagonal. See the overrides immediately below the definition
# of DiagonalTensor for usage examples.
HANDLED_FUNCTIONS_DIAGONAL = {}

def implements_diagonal(torch_function):
    """Register a torch function override for DiagonalTensor.

    This decorator takes a function in the torch API as a
    parameter. Applying this decorator to a function adds that function
    as the registered override for the torch function passed as a
    parameter to the decorator. See DiagonalTensor.__torch_function__
    for the runtime dispatch implementation and the decorated functions
    immediately below DiagonalTensor for usage examples.
    """
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS_DIAGONAL[torch_function] = func
        return func
    return decorator

class DiagonalTensor:
    """A class with __torch_function__ and a specific diagonal representation

    This class has limited utility and is mostly useful for verifying that the
    dispatch mechanism works as expected. It is based on the `DiagonalArray
    example`_ in the NumPy documentation.

    Note that this class does *not* inherit from ``torch.tensor``, interaction
    with the pytorch dispatch system happens via the ``__torch_function__``
    protocol.

    ``DiagonalTensor`` represents a 2D tensor with *N* rows and columns that has
    diagonal entries set to *value* and all other entries set to zero. The
    main functionality of ``DiagonalTensor`` is to provide a more compact
    string representation of a diagonal tensor than in the base tensor class:

    >>> d = DiagonalTensor(5, 2)
    >>> d
    DiagonalTensor(N=5, value=2)
    >>> d.tensor()
    tensor([[2., 0., 0., 0., 0.],
            [0., 2., 0., 0., 0.],
            [0., 0., 2., 0., 0.],
            [0., 0., 0., 2., 0.],
            [0., 0., 0., 0., 2.]])

    Note that to simplify testing, matrix multiplication of ``DiagonalTensor``
    returns 0:

    >>> torch.mm(d, d)
    0

    .. _DiagonalArray example:
        https://numpy.org/devdocs/user/basics.dispatch.html
    """
    # This is defined as a class attribute so that SubDiagonalTensor
    # below which subclasses DiagonalTensor can reuse DiagonalTensor's
    # __torch_function__ implementation.
    handled_functions = HANDLED_FUNCTIONS_DIAGONAL

    def __init__(self, N, value):
        self._N = N
        self._i = value

    def __repr__(self):
        return f"DiagonalTensor(N={self._N}, value={self._i})"

    def __array__(self):
        return self._i * np.eye(self._N)

    def tensor(self):
        return self._i * torch.eye(self._N)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func not in cls.handled_functions:
            return NotImplemented
        return cls.handled_functions[func](*args, **kwargs)

    def __eq__(self, other):
        return type(other) is type(self) and self._N == other._N and self._i == other._i

@implements_diagonal(torch.mean)
def mean(mat):
    return float(mat._i) / mat._N

@implements_diagonal(torch.mm)
def diagonal_mm(mat1, mat2):
    return 0

@implements_diagonal(torch.div)
def diagonal_div(input, other, out=None):
    return -1

@implements_diagonal(torch.add)
def add(mat1, mat2):
    raise ValueError

@implements_diagonal(foo)
def diagonal_foo(a, b, c=None):
    return -1

@implements_diagonal(bar)
def diagonal_bar(a):
    return -1

@implements_diagonal(quux)
def diagonal_quux(a):
    raise ValueError

# The dispatch table for SubTensor's __torch_function__ implementation.
HANDLED_FUNCTIONS_SUB = {}

def implements_sub(torch_function):
    "Register a torch function override for SubTensor"
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS_SUB[torch_function] = func
        return func
    return decorator

class SubTensor(torch.Tensor):
    """A subclass of torch.Tensor use for testing __torch_function__ dispatch

    This class has the property that matrix multiplication returns zero:

    >>> s = SubTensor([[1, 1], [1, 1]])
    >>> torch.mm(s, s)
    0
    >>> t = torch.tensor([[1, 1], [1, 1]])
    >>> torch.mm(s, t)
    0
    >>> torch.mm(t, s)
    0
    >>> torch.mm(t, t)
    tensor([[2, 2],
            [2, 2]])

    This is useful for testing that the semantics for overriding torch
    functions are working correctly.
    """
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func not in HANDLED_FUNCTIONS_SUB:
            return NotImplemented
        return HANDLED_FUNCTIONS_SUB[func](*args, **kwargs)

class SubTensor2(torch.Tensor):
    pass

class SubSubTensor2(SubTensor2):
    pass

class SubTensor3(torch.Tensor):
    pass

@implements_sub(torch.mean)
def sub_mean(mat):
    return 0

@implements_sub(torch.mm)
def sub_mm(mat1, mat2):
    return -1

@implements_sub(bar)
def sub_bar(mat):
    return 1

@implements_sub(torch.div)
def sub_div(input, other, out=None):
    return NotImplemented

# The dispatch table for SubDiagonalTensor's __torch_function__ implementation.
HANDLED_FUNCTIONS_SUB_DIAGONAL = {}

def implements_sub_diagonal(torch_function):
    "Register a torch function override for SubDiagonalTensor"
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS_SUB_DIAGONAL[torch_function] = func
        return func
    return decorator

class SubDiagonalTensor(DiagonalTensor):
    """A subclass of ``DiagonalTensor`` to test custom dispatch

    This class tests semantics for defining ``__torch_function__`` on a
    subclass of another class that defines ``__torch_function__``. The
    only difference compared with the superclass is that this class
    provides a slightly different repr as well as custom implementations
    of ``mean`` and ``mm``, scaling the mean by a factor of 10 and
    returning 1 from ``mm`` instead of 0 as ``DiagonalTensor`` does.
    """
    handled_functions = HANDLED_FUNCTIONS_SUB_DIAGONAL

    def __repr__(self):
        return f"SubDiagonalTensor(N={self._N}, value={self._i})"


@implements_sub_diagonal(torch.mean)
def sub_diagonal_mean(mat):
    return 10 * float(mat._i) / mat._N

@implements_sub_diagonal(bar)
def sub_diagonal_bar(mat):
    return 0

@implements_sub_diagonal(torch.mm)
def sub_diagonal_mm(mat1, mat2):
    return 1

@implements_sub_diagonal(torch.div)
def sub_diagonal_div(input, other, out=None):
    return NotImplemented

@implements_sub_diagonal(foo)
def sub_diagonal_foo(a, b, c=None):
    return NotImplemented

# The dispatch table for SubDiagonalTensor's __torch_function__ implementation.
HANDLED_FUNCTIONS_TENSOR_LIKE = {}


# Note: _triggered wrapper
# Dict that wraps the implementations from get_testing_overrides into another
# function with a _triggered slot/flag. The triggered flag is set when the
# implementation is called.
WRAPPED_TRIGGERED_IMPLS = {}


def triggered_wrapper(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        wrapped._triggered = True
        return f(*args, **kwargs)

    wrapped._triggered = False
    return wrapped

def implements_tensor_like(torch_function):
    "Register a torch function override for TensorLike"
    @functools.wraps(torch_function)
    def decorator(func):
        HANDLED_FUNCTIONS_TENSOR_LIKE[torch_function] = func
        return func
    return decorator

def generate_tensor_like_torch_implementations():
    untested_funcs = []
    testing_overrides = get_testing_overrides()
    # test/test_cpp_api_parity.py monkeypatches torch.nn to have a new
    # function sample_functional.  Depending on what order you run pytest
    # collection, this may trigger the error here.  This is a hack to fix
    # the problem.  A more proper fix is to make the "not tested" check
    # a test on its own, and to make sure the monkeypatch is only installed
    # for the span of the relevant test (and deleted afterwards)
    testing_ignore = {"sample_functional", "autocast"}
    for namespace, funcs in get_overridable_functions().items():
        for func in funcs:
            if func not in testing_overrides and func.__name__ not in testing_ignore:
                untested_funcs.append(f"{namespace}.{func.__name__}")
    msg = (
        "The following functions are not tested for __torch_function__ "
        "support, please ensure there is an entry in the dict returned by "
        "torch.overrides.get_testing_overrides for this function or if a "
        "__torch_function__ override does not make sense, add an entry to "
        "the tuple returned by torch._overrides.get_ignored_functions.\n\n{}"
    )
    assert len(untested_funcs) == 0, msg.format(pprint.pformat(untested_funcs))
    for func, override in testing_overrides.items():
        # decorate the overrides with implements_tensor_like if it's not a
        # torch.Tensor method
        wrapped = triggered_wrapper(override)
        # See note: "_triggered wrapper"
        WRAPPED_TRIGGERED_IMPLS[func] = wrapped
        if is_tensor_method_or_property(func):
            implements_sub(func)(wrapped)
        else:
            implements_tensor_like(func)(wrapped)

generate_tensor_like_torch_implementations()

class TensorLike:
    """A class that overrides the full torch API

    This class is used to explicitly test that the full torch.tensor API
    can be overridden with a class that defines __torch_function__.
    """
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func not in HANDLED_FUNCTIONS_TENSOR_LIKE:
            return NotImplemented
        # In this case _torch_function_ should override TensorLike objects
        return HANDLED_FUNCTIONS_TENSOR_LIKE[func](*args, **kwargs)

class TestTorchFunctionOverride(TestCase):
    def test_dtype_override(self):
        class MyDtype:
            def __torch_function__(self, *args, **kwargs):
                return 4

        self.assertEqual(torch.empty(4).view(MyDtype()), 4)

    def test_mean_semantics(self):
        """Test that a function with one argument can be overridden"""
        t1 = DiagonalTensor(5, 2)
        t2 = SubTensor([[1, 2], [1, 2]])
        t3 = SubDiagonalTensor(5, 2)
        self.assertEqual(torch.mean(t1), 0.4)
        self.assertEqual(bar(t1), -1)
        self.assertEqual(torch.mean(t2), 0)
        self.assertEqual(bar(t2), 1)
        self.assertEqual(torch.mean(t3), 4.0)
        self.assertEqual(bar(t3), 0)

    def test_has_torch_function_non_sequence(self):
        with self.assertRaisesRegex(TypeError, "expected a sequence"):
            has_torch_function(object())

    def test_mm_semantics(self):
        """Test that a function with multiple arguments can be overridden"""
        t1 = DiagonalTensor(5, 2)
        t2 = torch.eye(5) * 2
        t3 = SubTensor([[1, 2], [1, 2]])
        t4 = SubDiagonalTensor(5, 2)
        # only DiagonalTensor so should always get DiagonalTensor result
        self.assertEqual(torch.mm(t1, t1), 0)
        # tensor and DiagonalTensor, always return DiagonalTensor result
        self.assertEqual(torch.mm(t1, t2), 0)
        self.assertEqual(torch.mm(t2, t1), 0)
        # only SubTensor so should always get SubTensor result
        self.assertEqual(torch.mm(t3, t3), -1)
        # tensor and SubTensor so should always get SubTensor result
        self.assertEqual(torch.mm(t3, t2), -1)
        self.assertEqual(torch.mm(t2, t3), -1)
        # DiagonalTensor and SubTensor are unrelated classes so the result
        # depends on which argument appears first
        self.assertEqual(torch.mm(t3, t1), -1)
        self.assertEqual(torch.mm(t1, t3), 0)
        # SubDiagonalTensor should take precedence over DiagonalTensor
        # but should behave otherwise the same as DiagonalTensor
        self.assertEqual(torch.mm(t4, t4), 1)
        self.assertEqual(torch.mm(t4, t1), 1)
        self.assertEqual(torch.mm(t1, t4), 1)
        self.assertEqual(torch.mm(t4, t2), 1)
        self.assertEqual(torch.mm(t2, t4), 1)
        self.assertEqual(torch.mm(t3, t4), -1)
        self.assertEqual(torch.mm(t4, t3), 1)

    def test_precedence_semantics(self):
        """Test semantics for __torch_function__ for functions that take
        multiple arguments

        For functions that take multiple arguments, the appropriate
        __torch_function__ implementation to call is determined by
        examining the types of the arguments. The precedence order is
        left-to-right in the argument list, except subclasses are always
        checked before superclasses. The first result of calling the
        implementations in precedence order that is not NotImplemented
        is returned to the user. If all implementations return
        NotImplemented, a TypeError is raised.

        All cases are tested with functions implemented in C++ and
        either foo or baz, which are python functions defined above that
        are instrumented to obey the same dispatch rules as the
        functions in torch.functional.
        """
        # DiagonalTensor has a valid override and SubDiagonal has an
        # override that returns NotImplemented so we should call the
        # DiagonalTensor implementation, returning -1
        t1 = DiagonalTensor(5, 2)
        t2 = SubDiagonalTensor(5, 2)
        self.assertEqual(torch.div(t1, t2), -1)
        self.assertEqual(torch.div(t2, t1), -1)
        self.assertEqual(foo(t1, t2), -1)
        self.assertEqual(foo(t2, t1), -1)

        # SubTensor has an implementation that returns NotImplemented as
        # well so it should behave exactly like SubDiagonalTensor in the
        # test above
        t3 = SubTensor([[1, 2], [1, 2]])
        self.assertEqual(torch.div(t1, t3), -1)
        self.assertEqual(torch.div(t3, t1), -1)
        self.assertEqual(foo(t1, t3), -1)
        self.assertEqual(foo(t3, t1), -1)

        # div between SubTensor and SubDiagonalTensor should raise
        # TypeError since both have an implementation that
        # explicitly returns NotImplemented
        with self.assertRaises(TypeError):
            torch.div(t2, t3)
        with self.assertRaises(TypeError):
            torch.div(t3, t2)
        with self.assertRaises(TypeError):
            foo(t2, t3)
        with self.assertRaises(TypeError):
            foo(t3, t2)

        # none of DiagonalTensor, SubdiagonalTensor, or SubTensor have a
        # mul or a baz implementation so all ops should raise TypeError
        with self.assertRaises(TypeError):
            torch.mul(t1, t1)
        with self.assertRaises(TypeError):
            torch.mul(t1, t2)
        with self.assertRaises(TypeError):
            torch.mul(t1, t3)
        with self.assertRaises(TypeError):
            torch.mul(t2, t1)
        with self.assertRaises(TypeError):
            torch.mul(t2, t2)
        with self.assertRaises(TypeError):
            torch.mul(t2, t3)
        with self.assertRaises(TypeError):
            torch.mul(t3, t1)
        with self.assertRaises(TypeError):
            torch.mul(t3, t2)
        with self.assertRaises(TypeError):
            torch.mul(t3, t3)
        with self.assertRaises(TypeError):
            baz(t1, t1)
        with self.assertRaises(TypeError):
            baz(t1, t2)
        with self.assertRaises(TypeError):
            baz(t1, t3)
        with self.assertRaises(TypeError):
            baz(t2, t1)
        with self.assertRaises(TypeError):
            baz(t2, t2)
        with self.assertRaises(TypeError):
            baz(t2, t3)
        with self.assertRaises(TypeError):
            baz(t3, t1)
        with self.assertRaises(TypeError):
            baz(t3, t2)
        with self.assertRaises(TypeError):
            baz(t3, t3)

    def test_user_implementation_raises(self):
        """Test that errors raised in user implementations propagate correctly"""
        t1 = DiagonalTensor(5, 2)
        t2 = DiagonalTensor(5, 2)
        with self.assertRaises(ValueError):
            torch.add(t1, t2)
        with self.assertRaises(ValueError):
            quux(t1)

    def test_tensor_subclass_propagation(self):
        """this test exercises the functionality described in
        docs/source/notes/extending.rst#subclassing-torchtensor"""
        t1 = torch.tensor([5])
        t2 = torch.tensor([6])

        s1 = SubTensor2([5])
        s2 = SubTensor2([6])

        ss1 = SubSubTensor2([5])
        ss2 = SubSubTensor2([6])

        sn1 = SubTensor3([5])
        sn2 = SubTensor3([6])

        # Check that leaf subclass is kept regardless of order
        self.assertTrue(isinstance(s1 + t2, SubTensor2))
        self.assertTrue(isinstance(t1 + s2, SubTensor2))
        self.assertTrue(isinstance(s1 + s2, SubTensor2))

        # Check indexing subclass is kept
        self.assertTrue(isinstance(s1[0], SubTensor2))

        # Check case for subclass of subclass.
        self.assertTrue(isinstance(ss1 + ss2, SubSubTensor2))
        self.assertTrue(isinstance(ss1 + s2, SubSubTensor2))
        self.assertTrue(isinstance(s1 + ss2, SubSubTensor2))
        self.assertTrue(isinstance(ss1 + ss2, SubSubTensor2))
        self.assertTrue(isinstance(ss1 + t2, SubSubTensor2))
        self.assertTrue(isinstance(t1 + ss2, SubSubTensor2))
        self.assertTrue(isinstance(ss1[0], SubSubTensor2))

        # Make sure unrelated class trees are not merged.
        with self.assertRaises(TypeError):
            s1 + sn2
        with self.assertRaises(TypeError):
            sn1 + s2

    def test_base(self):
        # https://github.com/szagoruyko/pytorchviz/issues/65
        class DummyTensor(torch.Tensor):
            pass

        a = torch.ones(1)
        c = DummyTensor(a)
        self.assertTrue(c._is_view())
        self.assertTrue(c._base is a)

    def test_grad(self):
        # Previously, Tensor-like objects that did not subclass from Tensor
        # did not get wrapped into unary tuples before being passed into
        # handle_torch_function, in contradiction with how Tensor-likes
        # were handled
        #
        # NB: this asserts that the arguments get normalized into a tuple
        # before entering the torch function handler; it could go the
        # other way but beware https://github.com/pytorch/pytorch/issues/76037

        class Dummy:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                inputs, outputs = args
                self.assertEqual(inputs, (x,))
                self.assertEqual(outputs, (x,))
                return -1

        x = Dummy()
        self.assertEqual(torch.autograd.grad(x, x), -1)

    def test_pow_rpow(self):
        class NothingImplemented(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        class RPowOnly(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                if func is torch.Tensor.__rpow__:
                    return -1
                return NotImplemented

        self.assertEqual(NothingImplemented() ** RPowOnly(), -1)

    def test_torch_function_in_lists(self):
        """Test that __torch_function__ is called for objects inside lists"""

        class IntLike:
            """Object that can be used in int lists"""
            def __init__(self, value):
                self.value = value
                self.torch_function_called = False

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.torch_function_called = True
                # Return a result that makes the operation succeed
                if func.__name__ == 'pad':
                    # For pad, return the input with shape adjusted
                    return args[0]
                elif func.__name__ == 'layer_norm':
                    # For layer_norm, return normalized tensor
                    return torch.ones_like(args[0])
                elif func.__name__ == 'tensordot':
                    # For tensordot, return appropriate shape
                    return torch.tensor(42.0)
                # Fallback
                return torch.tensor(42.0)

        # Test with F.pad which takes int list
        import torch.nn.functional as F
        x = torch.randn(2, 3)
        obj = IntLike(1)

        # pad takes [left, right, top, bottom] as padding
        _ = F.pad(x, [1, obj, 0, 0])
        self.assertTrue(obj.torch_function_called,
                        "torch_function should be called for object in int list")

        # Test multiple objects in list
        obj1 = IntLike(1)
        obj2 = IntLike(2)
        _ = F.pad(x, [obj1, obj2, 0, 0])
        self.assertTrue(obj1.torch_function_called or obj2.torch_function_called,
                        "torch_function should be called for at least one object")

    def test_torch_function_in_float_lists(self):
        """Test that __torch_function__ is called for objects inside float lists"""

        class FloatLike:
            """Object that can be used in float lists"""
            def __init__(self, value):
                self.value = float(value)
                self.torch_function_called = False

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.torch_function_called = True
                # Return appropriate result
                if func.__name__ == 'layer_norm':
                    return torch.ones_like(args[0])
                return torch.tensor(42.0)

        import torch.nn.functional as F
        x = torch.randn(2, 3, 4)
        obj = FloatLike(4.0)

        # layer_norm takes normalized_shape as int/float list
        _ = F.layer_norm(x, [3, obj])
        self.assertTrue(obj.torch_function_called,
                        "torch_function should be called for object in float list")

    def test_torch_function_in_scalar_lists(self):
        """Test that __torch_function__ is called for scalar objects inside lists"""

        class ScalarLike:
            """Object that can be used as a scalar in lists"""
            def __init__(self, value):
                self.value = value
                self.torch_function_called = False

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.torch_function_called = True
                # Return a scalar tensor
                return torch.tensor(self.value)

            def __float__(self):
                return float(self.value)

            def __int__(self):
                return int(self.value)

        # Test with a function that takes scalar lists
        # Using torch.as_tensor which can take scalar lists
        obj1 = ScalarLike(1.0)
        obj2 = ScalarLike(2.0)

        # Create a tensor with scalar list containing torch function objects
        # Use a different operation that should trigger torch_function
        _ = torch.stack([obj1, obj2])
        self.assertTrue(obj1.torch_function_called or obj2.torch_function_called,
                        "torch_function should be called for scalar objects in list")

    def test_torch_function_precedence_in_lists(self):
        """Test precedence when multiple torch function objects are in a list"""

        call_order = []

        class HighPriority:
            def __torch_function__(self, func, types, args=(), kwargs=None):
                call_order.append('high')
                # Delegate to lower priority
                return NotImplemented

        class LowPriority:
            def __torch_function__(self, func, types, args=(), kwargs=None):
                call_order.append('low')
                # Return valid result
                if func.__name__ == 'pad':
                    return args[0]
                return torch.tensor(42.0)

        import torch.nn.functional as F
        x = torch.randn(2, 3)

        high = HighPriority()
        low = LowPriority()

        # Test with both objects in list
        call_order.clear()
        _ = F.pad(x, [1, high, low, 0])

        # High priority should be called first
        self.assertEqual(call_order[0], 'high',
                         "Higher priority torch_function should be called first")
        self.assertEqual(call_order[1], 'low',
                         "Lower priority torch_function should be called after NotImplemented")

    def test_torch_function_mixed_lists(self):
        """Test lists with mix of regular values and torch function objects"""

        class CountingInt:
            call_count = 0

            def __init__(self, value):
                self.value = value

            @classmethod
            def reset(cls):
                cls.call_count = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                CountingInt.call_count += 1
                # Return valid result
                if func.__name__ == 'pad':
                    return args[0]
                return torch.tensor(42.0)

            def __index__(self):
                return self.value

        import torch.nn.functional as F
        x = torch.randn(2, 3)

        obj = CountingInt(2)
        CountingInt.reset()

        # Mix regular ints with torch function object
        _ = F.pad(x, [1, obj, 0, 0])

        self.assertEqual(CountingInt.call_count, 1,
                         "torch_function should be called exactly once for mixed list")

    def test_torch_function_empty_lists(self):
        """Test that empty lists work correctly"""

        # This should work without calling any torch_function
        x = torch.randn(1)  # Single element tensor

        # Functions that accept empty lists should still work
        # torch.stack with empty list of tensors would fail,
        # but empty size lists should work
        result = x.view([])  # Empty list means scalar
        self.assertEqual(result.shape, torch.Size([]),
                         "Empty list should work for size arguments")

    def test_torch_function_not_first_in_list(self):
        """Test that torch_function is called even when object is not first in list"""

        class IntLikeNotFirst:
            """Object with torch_function that won't be first in list"""
            def __init__(self, value):
                self.value = value
                self.torch_function_called = False

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.torch_function_called = True
                # Return input tensor for pad
                return args[0]

            def __index__(self):
                return self.value

        import torch.nn.functional as F
        x = torch.randn(2, 3)

        # Test with torch_function object as second item
        obj_second = IntLikeNotFirst(2)
        _ = F.pad(x, [1, obj_second, 0, 0])
        self.assertTrue(obj_second.torch_function_called,
                        "torch_function should be called when object is second in list")

        # Test with torch_function object as third item
        obj_third = IntLikeNotFirst(1)
        _ = F.pad(x, [1, 1, obj_third, 0])
        self.assertTrue(obj_third.torch_function_called,
                        "torch_function should be called when object is third in list")

        # Test with torch_function object as last item
        obj_last = IntLikeNotFirst(1)
        _ = F.pad(x, [1, 1, 1, obj_last])
        self.assertTrue(obj_last.torch_function_called,
                        "torch_function should be called when object is last in list")

    def test_torch_function_nested_tuple_getitem(self):
        """Test that torch_function is called with getitem for TF objects inside nested tuples"""

        called_functions = []

        class TorchFunctionObj:
            """Object with torch_function that tracks which functions are called"""
            def __init__(self, value):
                self.value = value

            def __torch_function__(self, func, types, args=(), kwargs=None):
                called_functions.append(func.__name__)
                # For getitem, return the tensor unchanged
                if func.__name__ == '__getitem__':
                    return args[0]
                # Return a simple result for other functions
                return torch.tensor(42.0)

            def __index__(self):
                return self.value

        # Create a tensor to index
        x = torch.randn(5, 5, 5)

        # Create torch function objects - these will be INSIDE the nested structure
        tf_obj1 = TorchFunctionObj(0)
        tf_obj2 = TorchFunctionObj(1)

        # Clear the called functions list
        called_functions.clear()

        # Test with tuple of tuple where TF objects are only on the INSIDE
        # The outer structure is regular tuples, but inner elements have __torch_function__
        # This tests the recursive detection logic added in the recent commit
        x[(0, (tf_obj1, tf_obj2))]

        # Assert that torch_function was called
        self.assertTrue(len(called_functions) > 0,
                        "torch_function should be called for TF objects inside nested tuples")

        # Assert that getitem was called, not size
        self.assertIn('__getitem__', called_functions,
                      "getitem should be called for tuple indexing with torch function objects inside")

        self.assertNotIn('size', called_functions,
                         "size should not be called - we should use getitem, not convert to advanced indexing")


def generate_tensor_like_override_tests(cls):
    from torch.testing._internal.generated.annotated_fn_args import annotated_args

    def test_generator(func, override):
        # If func corresponds to a torch.Tensor method or property.
        if is_tensor_method_or_property(func):
            # Generate an instance by using SubTensor,
            def instance_gen():
                return SubTensor([5])
        else:
            # Otherwise, TensorLike.
            def instance_gen():
                return TensorLike()

        # FIXME The following code does not support kwonly args without defaults.
        # The fix is easy, as one just needs to save these args when generating the variable
        # annotated_args. The problem is that, if one does so, one finds a number
        # of functions that have problematic signatures in native_functions.yaml.
        # Fixing these would be BC breaking, so hence this terrible hack
        # https://github.com/pytorch/pytorch/issues/67008
        kwargs = {}
        if hasattr(func, "__name__") and "linalg_solve_triangular" in func.__name__:
            kwargs = {"upper": True}

        func_args = []
        is_method = is_tensor_method_or_property(func)

        def _simple_type_parser(func, arg_name, arg_type):
            # Guess valid input to aten function based on type of argument
            if arg_type == "Tensor":
                return instance_gen()
            elif arg_type == "TensorList" or arg_type == "ITensorListRef":
                return [instance_gen(), instance_gen()]
            elif arg_type == "c10::List<::std::optional<Tensor>>":
                return [instance_gen(), instance_gen()]
            elif arg_type == "IntArrayRef" or arg_type == "SymIntArrayRef":
                size = arg.get("size", 2)
                if size == 1:
                    return 1
                else:
                    return [1] * size
            elif arg_type == "Scalar":
                return 3.5
            elif arg_type == "bool":
                return False
            elif arg_type == "Dimname":
                return ""
            elif arg_type == "DimnameList":
                return [""]
            elif arg_type.startswith("int"):
                return 0
            elif arg_type in {"Stream"}:
                return torch.Stream()
            elif arg_type.startswith("float") or arg_type == "double":
                return 1.0
            elif arg_type in {"Generator", "MemoryFormat", "TensorOptions"}:
                return None
            elif arg_type == "ScalarType":
                return torch.float32
            elif arg_type == "c10::string_view":
                return ""
            elif arg_type in ("std::string_view", "::std::string_view"):
                return ""
            elif arg_type == "SymInt":
                # TODO: generate actual SymbolicInt
                return 1
            else:
                raise RuntimeError(
                    f"Unsupported argument type {arg_type} for {arg_name} of function {func}"
                )

        # Special case; this doesn't have a schema but takes a list
        if func is torch.sym_sum:
            func_args.append([TensorLike(), TensorLike()])
        elif func in annotated_args:
            for arg in annotated_args[func]:
                # Guess valid input to aten function based on type of argument
                t = arg["simple_type"]
                t = t.removesuffix("?")
                if t == "Tensor" and is_method and arg["name"] == "self":
                    # See "Note: properties and __get__"
                    func = func.__get__(instance_gen())
                    continue
                arg_to_add = _simple_type_parser(func, arg["name"], t)
                if "is_kwarg_only" in arg and arg["is_kwarg_only"] == str(True):
                    kwargs[arg["name"]] = arg_to_add
                else:
                    func_args.append(arg_to_add)
        else:
            args = inspect.getfullargspec(override)
            try:
                func_args = inspect.getfullargspec(func)
                # Remove annotations from argspec
                func_args = type(func_args)(**{**func_args, 'annotations': None})
                if func_args != args:
                    raise RuntimeError(f"Override for {func} doesn't match its argspec.\n"
                                       + f"Original: {inspect.signature(func)}\n"
                                       + f"Override: {inspect.signature(override)}")
            except TypeError:
                pass
            nargs = len(args.args)
            if args.defaults is not None:
                nargs -= len(args.defaults)
            func_args = [instance_gen() for _ in range(nargs)]
            if args.varargs is not None:
                func_args += [instance_gen(), instance_gen()]

        def test(self):
            ret = func(*func_args, **kwargs)
            # ret is None for certain protocols, e.g., `__weakref__` and `__setitem__`
            # This is currently the best check but doesn't work for, for example,
            # Tensor.__add__ because it redirects to Tensor.add.
            # See note "_triggered wrapper"
            if not is_method or ret is None:
                self.assertTrue(WRAPPED_TRIGGERED_IMPLS[func]._triggered)
                return

            self.assertEqual(ret, -1)

        return test

    for func, override in get_testing_overrides().items():
        test_method = test_generator(func, override)
        if func.__name__ == "__get__":
            # Note: properties and __get__
            # __get__ is part of the descriptor protocol.
            # https://docs.python.org/3/howto/descriptor.html
            # This is used for properties of the form
            # torch.Tensor.<property>, with the method __get__
            # In this case we get the property name in two ways:

            # This case for properties defined in C.
            module = getattr(
                func.__self__,
                "__qualname__",
                None
            )

            # This one for properties defined in Python.
            if module is None:
                module = "Tensor." + func.__self__.fget.__name__

            # Unfortunately I couldn't find a way to unify these two cases
            # and there is no way for general descriptors.
        elif is_tensor_method_or_property(func):
            module = "Tensor"
        else:
            module = func.__module__
        if module:
            name = 'test_{}_{}'.format(module.replace('.', '_'), func.__name__)
        else:
            name = f'test_{func.__name__}'
        test_method.__name__ = name
        setattr(cls, name, test_method)

generate_tensor_like_override_tests(TestTorchFunctionOverride)

class Wrapper:
    "Basic data container that knows how to unwrap itself"
    def __init__(self, data):
        self.__dict__["_data"] = data
        self.__dict__["used_attrs"] = set()
        self.__dict__["used_calls"] = set()

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        self.used_attrs.add(name)

        val = getattr(self._data, name)

        # If it's a method
        if not isinstance(val, torch.device) and callable(val):
            c = getattr(type(self._data), name)
            # Don't append self to args if classmethod/staticmethod
            if c is val:
                return lambda *a, **kw: wrap(self.__torch_function__(c, (Wrapper,), args=a, kwargs=kw))
            # Otherwise append self to args
            return lambda *a, **kw: wrap(self.__torch_function__(c, (Wrapper,), args=(self,) + a, kwargs=kw))

        return wrap(val)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value

        self.used_attrs.add(name)
        setattr(self._data, name, unwrap(value))

    def __setitem__(self, key, value):
        self._data[unwrap(key)] = unwrap(value)

    def __getitem__(self, key):
        return wrap(self._data[unwrap(key)])

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # Find an instance of this class in the arguments
        args_of_this_cls = []
        for a in args:
            if isinstance(a, cls):
                args_of_this_cls.append(a)
            elif isinstance(a, collections.abc.Sequence):
                args_of_this_cls.extend(el for el in a if isinstance(el, cls))
        assert len(args_of_this_cls) > 0
        for a in args_of_this_cls:
            a.used_calls.add(func)
        args = unwrap(tuple(args))
        kwargs = {k: unwrap(v) for k, v in kwargs.items()}

        return wrap(func(*args, **kwargs))

    def __add__(self, other):
        return self.__torch_function__(torch.add, (Wrapper,), (self, other))

    def __mul__(self, other):
        return self.__torch_function__(torch.mul, (Wrapper,), (self, other))

    def __sub__(self, other):
        return self.__torch_function__(torch.sub, (Wrapper,), (self, other))

    def __truediv__(self, other):
        return self.__torch_function__(torch.true_divide, (Wrapper,), (self, other))

    def __floordiv__(self, other):
        return self.__torch_function__(torch.floor_divide, (Wrapper,), (self, other))

    def __ge__(self, other):
        return self.__torch_function__(torch.ge, (Wrapper,), (self, other))

    def __gt__(self, other):
        return self.__torch_function__(torch.gt, (Wrapper,), (self, other))

    def __lt__(self, other):
        return self.__torch_function__(torch.lt, (Wrapper,), (self, other))

    def __le__(self, other):
        return self.__torch_function__(torch.le, (Wrapper,), (self, other))

    def __eq__(self, other):
        return self.__torch_function__(torch.eq, (Wrapper,), (self, other))

    def __ne__(self, other):
        return self.__torch_function__(torch.ne, (Wrapper,), (self, other))

    def __bool__(self):
        return self.__torch_function__(torch.Tensor.__bool__, (Wrapper,), (self,))

    def __int__(self):
        return self.__torch_function__(torch.Tensor.__int__, (Wrapper,), (self,))

    def __len__(self):
        return len(self._data)


# unwrap inputs if necessary
def unwrap(v):
    if type(v) in {tuple, list}:
        return type(v)(unwrap(vi) for vi in v)

    return v._data if isinstance(v, Wrapper) else v

# wrap inputs if necessary
def wrap(v):
    if type(v) in {tuple, list}:
        return type(v)(wrap(vi) for vi in v)

    return Wrapper(v) if isinstance(v, torch.Tensor) else v

class TestEinsumOverride(TestCase):
    "Regression test for gh-38479"
    def test_wrapper(self):
        x = Wrapper(torch.randn(5))
        y = Wrapper(torch.randn(4))
        self.assertEqual(torch.einsum('i,j->ij', x, y)._data,
                         torch.ger(x, y)._data)

        # in the old einsum interface, `operands` is a list
        a = Wrapper(torch.randn(2, 3))
        b = Wrapper(torch.randn(5, 3, 7))
        c = Wrapper(torch.randn(2, 7))
        self.assertEqual(torch.einsum('ik,jkl,il->ij', [a, b, c])._data,
                         torch.nn.functional.bilinear(a, c, b)._data)

class TestGradCheckOverride(TestCase):
    "Test that wrappers work with gradcheck."
    def test_gradcheck(self):
        from torch.testing._internal.common_utils import gradcheck, gradgradcheck

        def run_test(fast_mode):
            a = wrap(torch.tensor(5.0, dtype=torch.double))
            b = wrap(torch.tensor(6.0, dtype=torch.double))

            a.requires_grad = True
            b.requires_grad = True

            gradcheck(torch.add, (a, b), raise_exception=False, check_batched_grad=False, fast_mode=fast_mode)
            gradgradcheck(torch.add, (a, b), raise_exception=False, check_batched_grad=False, fast_mode=fast_mode)

            total_used_attrs = a.used_attrs.union(b.used_attrs)
            total_used_calls = a.used_calls.union(b.used_calls)

            # These attributes (and the functions below) may change
            # if the gradcheck implementation changes. It's best to
            # aim for attributes that may be commonly present on other
            # Tensor-likes.
            expected_used_attrs = {
                'data',
                'dtype',
                'is_floating_point',
                'is_sparse',
                'layout',
                'new_zeros',
                'numel',
                'requires_grad',
                'requires_grad_',
                'size',
                'stride',
            }
            if fast_mode:
                expected_used_attrs.add('is_complex')
                expected_used_attrs.add('device')
            self.assertEqual(expected_used_attrs, total_used_attrs)

            expected_used_calls = {
                torch.Tensor.new_zeros,
                torch.Tensor.size,
                torch.Tensor.is_floating_point,
                torch.Tensor.numel,
                torch.Tensor.stride,
                torch.Tensor.requires_grad_,
                torch.autograd.grad,
                torch.add,
            }
            if fast_mode:
                expected_used_calls.add(torch.Tensor.is_complex)
            self.assertEqual(expected_used_calls, total_used_calls)
        run_test(fast_mode=True)
        run_test(fast_mode=False)

class TestNamedTuple(TestCase):
    """ Regression test for gh-47090 """
    def test_max(self):
        x = torch.tensor([1, 2])
        xs = x.as_subclass(SubTensor2)
        r = torch.max(x, dim=0)
        rs = torch.max(xs, dim=0)
        self.assertEqual(type(r), type(rs))
        self.assertEqual(r, rs)

class TestGradNewOnesOverride(TestCase):
    """ Regression test for gh-47069 """
    def test_newones(self):
        t = torch.tensor([1, 2]).as_subclass(SubTensor2)
        n = t.new_ones((1, 2))
        self.assertEqual(type(n), SubTensor2)

class TestPickle(TestCase):
    "Regression test for gh-47051"
    def test_pickle(self):
        t = torch.tensor([1]).as_subclass(SubTensor2)
        t.abcd = "e"
        t2 = pickle.loads(pickle.dumps(t))
        self.assertIs(type(t2), SubTensor2)
        self.assertEqual(t2.abcd, "e")

class TestBroadcastAllOverride(TestCase):
    """ test for gh-37141 """
    def test_broadcast_all(self):
        from torch.distributions.utils import broadcast_all
        a = torch.tensor([1.2, 3.4, 5.6])
        a_w = Wrapper(a)
        b = torch.tensor(5.0)
        b_w = Wrapper(b)
        c = torch.tensor([5.0, 5.0, 5.0])

        o_1 = broadcast_all(a_w, b_w)
        self.assertTrue(isinstance(o_1[0], Wrapper))
        self.assertTrue(isinstance(o_1[1], Wrapper))
        self.assertEqual(o_1[0]._data, a)
        self.assertEqual(o_1[1]._data, c)

        o_2 = broadcast_all(a_w, b)
        self.assertTrue(isinstance(o_2[0], Wrapper))
        self.assertTrue(isinstance(o_2[1], Wrapper))
        self.assertEqual(o_2[0]._data, a)
        self.assertEqual(o_2[1]._data, c)

class TestWrapTorchFunction(TestCase):
    def test_wrap_torch_function(self):
        class A:
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs):
                return -1

        def dispatcher(a):
            return (a,)

        @torch.overrides.wrap_torch_function(dispatcher)
        def f(a):
            return a

        self.assertEqual(f(A()), -1)

class TestIndexing(TestCase):
    """ Regression tests for gh-46277 """
    def test_getitem(self):
        class A:
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                return -1

        t = torch.tensor([5])
        self.assertEqual(t[A()], -1)
        self.assertEqual(t, torch.tensor([5]))

    def test_getitem_subclass(self):
        class A(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                return -1

        t = torch.tensor([5])
        self.assertEqual(t[A()], -1)
        self.assertEqual(t[5, A()], -1)
        self.assertEqual(t, torch.tensor([5]))

    def test_setitem(self):
        triggered = set()

        class A:
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                triggered.add(func)
                return -1

        t = torch.tensor([5])
        t[A()] = 1
        t[5, A()] = 1
        self.assertIn(Tensor.__setitem__, triggered)
        self.assertEqual(t, torch.tensor([5]))

    def test_setitem_val(self):
        triggered = set()

        class A:
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                triggered.add(func)
                return -1

        t = torch.tensor([5])
        t[0] = A()
        self.assertIn(Tensor.__setitem__, triggered)
        self.assertEqual(t, torch.tensor([5]))

    def test_setitem_subclass(self):
        triggered = set()

        class A(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                triggered.add(func)
                return -1

        t = torch.tensor([5])
        t[A()] = 1
        t[5, A()] = 1
        self.assertIn(Tensor.__setitem__, triggered)
        self.assertEqual(t, torch.tensor([5]))


class TestIterator(TestCase):
    # Regression test for gh-54457
    def test_iterator(self):
        t = torch.tensor([5, 6, 7]).as_subclass(SubTensor2)
        it = iter(t)
        self.assertIs(type(next(it)), SubTensor2)
        self.assertIs(type(next(it)), SubTensor2)
        self.assertIs(type(next(it)), SubTensor2)


class TestR
```



## High-Level Overview

"""A function multiple arguments and an optional argument"""    if has_torch_function((a, b, c)):

This Python file contains 99 class(es) and 195 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DiagonalTensor`, `SubTensor`, `SubTensor2`, `SubSubTensor2`, `SubTensor3`, `SubDiagonalTensor`, `TensorLike`, `TestTorchFunctionOverride`, `MyDtype`, `DummyTensor`, `Dummy`, `NothingImplemented`, `RPowOnly`, `IntLike`, `FloatLike`, `ScalarLike`, `HighPriority`, `LowPriority`, `CountingInt`, `IntLikeNotFirst`

**Functions defined**: `foo`, `bar`, `baz`, `quux`, `implements_diagonal`, `decorator`, `__init__`, `__repr__`, `__array__`, `tensor`, `__torch_function__`, `__eq__`, `mean`, `diagonal_mm`, `diagonal_div`, `add`, `diagonal_foo`, `diagonal_bar`, `diagonal_quux`, `implements_sub`

**Key imports**: sys, torch, numpy as np, inspect, functools, pprint, pickle, collections, unittest, os


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `torch`
- `numpy as np`
- `inspect`
- `functools`
- `pprint`
- `pickle`
- `collections`
- `unittest`
- `os`
- `torch.testing._internal.common_utils`: TestCase, run_tests, TEST_WITH_CROSSREF
- `torch.utils._mode_utils`: all_same_mode
- `torch.utils._pytree`: tree_map
- `torch.nn.functional as F`
- `torch.testing._internal.generated.annotated_fn_args`: annotated_args
- `torch.distributions.utils`: broadcast_all
- `torch._C`: _is_torch_function_all_disabled
- `torch.utils._device`: DeviceContext


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/test_overrides.py
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
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `test_overrides.py_docs.md`
- **Keyword Index**: `test_overrides.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
