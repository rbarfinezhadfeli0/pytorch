# Documentation: `docs/test/dynamo/test_functions.py_docs.md`

## File Metadata

- **Path**: `docs/test/dynamo/test_functions.py_docs.md`
- **Size**: 54,548 bytes (53.27 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/dynamo/test_functions.py`

## File Metadata

- **Path**: `test/dynamo/test_functions.py`
- **Size**: 154,173 bytes (150.56 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]
# flake8: noqa: E731, C405, F811, C418, C417
import collections
import collections.abc
import contextlib
import functools
import inspect
import itertools
import keyword
import math
import operator
import random
import sys
import types
import typing
import unittest
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar
from typing_extensions import NamedTuple
from unittest.mock import patch

import numpy as np

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch import sub
from torch._dynamo.exc import Unsupported
from torch._dynamo.testing import (
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
    normalize_gm,
)
from torch._dynamo.utils import ifdynstaticdefault, range_iterator, same
from torch._dynamo.variables import ConstantVariable, SkipFunctionVariable
from torch._dynamo.variables.lists import RangeVariable
from torch.nn import functional as F
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_GPU

# Defines all the kernels for tests
from torch.testing._internal.triton_utils import *  # noqa: F403


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)

T = TypeVar("T")

d = torch.ones(10, 10)
e = torch.nn.Linear(10, 10)
flag = True


class CustomDictSubclass(collections.OrderedDict):
    pass


clip01 = functools.partial(torch.clip, min=0.0, max=1.0)


def constant3(a, b):
    return a - b + (1.0 + 2)


def call(f, *args, **kwargs):
    return f(*args, **kwargs)


_variable = 0


def update_global(x):
    global _variable
    _variable += 1
    # Check that updated global variable value is picked up
    return x * _variable


def pos_only_fn(*args, **kwargs):
    return _pos_only_fn(*args, **kwargs)


def _pos_only_fn(a, b=3, /, **kwargs):
    return (
        a * b + kwargs.get("a", -13) * kwargs.get("b", 42),
        "a" in kwargs,
        "b" in kwargs,
    )


@contextlib.contextmanager
def update_global_ctx(x):
    try:
        yield update_global(x)
    finally:
        pass


def func_with_default(a, b, some_default_arg=True):
    if some_default_arg:
        return a - b


def make_test(fn=None, expected_frame_count=1):
    if fn is None:
        return lambda fn: make_test(fn, expected_frame_count=expected_frame_count)

    nargs = len(inspect.signature(fn).parameters)

    def test_fn(self):
        return torch._dynamo.testing.standard_test(
            self,
            fn=fn,
            nargs=nargs,
            expected_frame_count=expected_frame_count,
        )

    return test_fn


class MyCls:
    a = 1


@torch.jit.script_if_tracing
def inline_script_if_tracing(x):
    return x + 1.2


@torch.jit.ignore
def inline_ignore(x):
    return x + 3.4


@torch.jit.unused
def inline_unused(x):
    return x + 5.6


@functools.lru_cache
def inline_lru_cache_fn_with_default_args(x, y, _=None):
    return torch.sin(x * y)


@torch.jit.script_if_tracing
def inline_script_if_tracing_fn_with_default_args(x, y, c=1.2):
    return torch.cos(x * y) + c


class FunctionTests(torch._dynamo.test_case.TestCase):
    @make_test
    def test_inline_jit_annotations(x):
        x = inline_script_if_tracing(x)
        x = inline_ignore(x)
        x = inline_unused(x)
        return

    @make_test
    def test_inline_script_if_tracing_fn_with_default_args(a, b):
        return inline_script_if_tracing_fn_with_default_args(a, b)

    @make_test
    def test_inline_lru_cache_fn_with_default_args(a, b):
        return inline_lru_cache_fn_with_default_args(a, 2, b)

    def test_lru_cache_warning_issued_during_tracing(self):
        import warnings
        from functools import lru_cache

        @lru_cache
        def foo(x):
            return x + 1

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch.compile(foo, backend="eager")(torch.randn(4))

        for warning in w:
            warning_message = str(warning.message)
            if (
                "Dynamo detected a call to a `functools.lru_cache`-wrapped"
                in warning_message
            ):
                break
        else:
            self.assertTrue(False, "Expected warning about lru_cache not found")

    @make_test
    def test_add(a, b):
        return a + b

    @make_test
    def test_add_(a, b):
        a_copy = torch.tensor(a)
        return a_copy.add_(b, alpha=5.0)

    @make_test
    def test_addcdiv(a, b, c):
        # dynamo decomposes this to avoid a graph break when
        # the value kwarg is populated
        return torch.addcdiv(a, b, c, value=5.0)

    @make_test
    def test_addcdiv_(a, b, c):
        a_copy = torch.tensor(a)
        return a_copy.addcdiv_(b, c, value=5.0)

    @make_test
    def test_is_not_null(a, b):
        if a is not None and b is not None:
            return a + b

    def test_foreach_lerp_(self):
        def fn(x, y, s):
            return torch._foreach_lerp_(x, y, s)

        cnt = torch._dynamo.testing.CompileCounter()

        fn_opt = torch.compile(backend=cnt, fullgraph=True)(fn)
        expected = fn(
            [torch.ones(2, 2) * 4.26, torch.ones(2, 2) * 3.14],
            [torch.ones(2, 2), torch.ones(2, 2)],
            torch.tensor(0.5),
        )

        actual = fn_opt(
            [torch.ones(2, 2) * 4.26, torch.ones(2, 2) * 3.14],
            [torch.ones(2, 2), torch.ones(2, 2)],
            torch.tensor(0.5),
        )
        self.assertTrue(same(expected, actual))

    def test_broadcast_foreach_pow(self):
        from torch._dynamo.utils import same

        def fn(x, y):
            return torch._foreach_pow(x, y)

        cnt = torch._dynamo.testing.CompileCounter()

        fn_opt = torch.compile(backend=cnt, fullgraph=True)(fn)
        inps = (torch.tensor(0.80), [torch.tensor(3.4), torch.tensor(7.8)])

        actual = fn_opt(*inps)
        expected = fn(*inps)
        self.assertTrue(same(actual, expected))
        self.assertTrue(cnt.frame_count, 1)

    def test_addcmul_(self):
        from copy import deepcopy

        from torch._dynamo.utils import same

        def fn(x, y, z, s):
            return x.addcmul_(y, z, value=s)

        cnt = torch._dynamo.testing.CompileCounter()
        fn_opt = torch.compile(backend=cnt, fullgraph=True)(fn)
        inps = (
            torch.ones(2, 2),
            torch.ones(2, 2) + 1,
            torch.rand(2, 2),
            torch.tensor(0.3),
        )
        inps_2 = deepcopy(inps)
        actual = fn_opt(*inps)
        expected = fn(*inps_2)
        self.assertTrue(same(actual, expected))
        self.assertEqual(cnt.frame_count, 1)

    @make_test
    def test_functools_partial(a, b):
        return clip01(a + b)

    @make_test
    def test_itertools_product(a, b):
        v = a
        for x, i in itertools.product([a, b], [1, 2]):
            v = v + x * i
        return v

    def test_itertools_product_args(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(*args, **kwargs):
            return torch.tensor(list(itertools.product(*args, **kwargs)))

        self.assertRaises(Unsupported, fn, [1, 2, 3], fake_arg=1)

    @make_test
    def test_itertools_product_various_iterators(a, b):
        itertools.product(
            [a, b],
            zip([1, 2], [3, 4]),
            map(lambda x: x, [1, 2]),
            filter(lambda x: True, [1, 2]),
        )
        return a

    def test_itertools_permutations_basic(self):
        def fn():
            return torch.tensor(list(itertools.permutations([1, 2, 3], 2)))

        actual = torch.compile(fn, backend="eager", fullgraph=True)()
        expected = fn()
        self.assertEqual(actual, expected)

    def test_itertools_permutations_args(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(*args, **kwargs):
            return torch.tensor(list(itertools.permutations(*args, **kwargs)))

        self.assertRaises(Unsupported, fn)
        self.assertRaises(Unsupported, fn, [1, 2, 3], 1, 2)
        self.assertRaises(Unsupported, fn, [1, 2, 3], fake_arg=1)

    @make_test
    def test_itertools_permutations_various_iterators(a, b):
        itertools.permutations([a, b])
        itertools.permutations(zip([1, 2], [3, 4]))
        itertools.permutations(map(lambda x: x, [1, 2]))
        itertools.permutations(filter(lambda x: True, [1, 2]))
        return a

    @make_test
    def test_itertools_filterfalse_basic(a, b):
        for x in itertools.filterfalse(lambda x: x > 0, [-0.5, 0, 0.5]):
            a += x
        return a

    @make_test
    def test_itertools_chain(a, b):
        v = a
        for x in itertools.chain([a, b], [1, 2]):
            v = v + x
        return v

    @make_test
    def test_itertools_chain_from_iterable(a, b):
        v = a
        for x in itertools.chain.from_iterable([[a, b], [1, 2]]):
            v = v + x
        return v

    def test_itertools_reconstruct(self):
        def fn(a):
            it1 = itertools.repeat(1)
            it2 = itertools.count(2)
            for _ in range(3):
                a += next(it1)
                a += next(it2)
            return it1, it2, a

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        i1, i2, a = fn(torch.ones(3, 3))
        it1, it2, b = opt_fn(torch.ones(3, 3))
        self.assertEqual(next(i1), next(it1))
        self.assertEqual(next(i2), next(it2))
        self.assertEqual(a, b)

    @make_test
    def test_obj_eq(a, b):
        v = a + b
        if MyCls() == None:  # noqa: E711
            return -1
        if MyCls() != None:  # noqa: E711
            v = v.sin()
        if MyCls() == MyCls():
            return -2
        if MyCls() != MyCls():
            return v + 1
        return -3

    @make_test
    def test_cls_eq(a, b):
        v = a + b
        if MyCls == None:  # noqa: E711
            return -1
        if MyCls != None:  # noqa: E711
            v = v.sin()
        if MyCls != MyCls:
            return -2
        if MyCls == MyCls:
            return v + 1
        return -3

    @make_test
    def test_obj_is(a, b):
        v = a + b
        if MyCls() is None:  # noqa: E711
            return -1
        if MyCls() is not None:  # noqa: E711
            v = v.sin()
        if MyCls() is MyCls():
            return -2
        if MyCls() is not MyCls():
            return v + 1
        return -3

    @make_test
    def test_cls_is(a, b):
        v = a + b
        if MyCls is None:  # noqa: E711
            return -1
        if MyCls is not None:  # noqa: E711
            v = v.sin()
        if MyCls is not MyCls:
            return -2
        if MyCls is MyCls:
            return v + 1
        return -3

    @make_test
    def test_itertools_combinations(a, b):
        combs = []
        for size in itertools.combinations((1, 2, 3, 4), 2):
            combs.append(torch.ones(size))
        return combs

    @make_test
    def test_itertools_pairwise(a):
        pairs = []
        for size in itertools.pairwise((1, 2, 3, 4)):
            pairs.append(torch.ones(size))
        return pairs

    def test_itertools_compress(self):
        def fn():
            return itertools.compress("ABCDEF", [1, 0, 1, 0, 1, 1])

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertListEqual(list(opt_fn()), list(fn()))

    def test_itertools_compress_tensors(self):
        def fn():
            return itertools.compress(
                [torch.tensor([0]), torch.tensor([1]), torch.tensor([2])], [1, 0, 1]
            )

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertListEqual(list(opt_fn()), list(fn()))

    @make_test
    def test_np_iinfo(a):
        max_dim = np.iinfo(np.int16).max
        return a + max_dim

    @make_test
    def test_np_finfo(a):
        min_dim = np.finfo(np.float32).min
        return a + min_dim

    @make_test
    def test_constant1(a, b, c):
        return a - b * c + 1.0

    @make_test
    def test_constant2(a, b, c):
        return a - b * c + 1

    @make_test
    def test_constant3(a):
        b = 1
        c = 2
        d = 3
        return b + c - d + a

    @make_test
    def test_constant4(a, b):
        c = 2
        d = 3
        if c > d:
            return a - b
        return b - a

    @make_test
    def test_cls_hasattr(self, x):
        if hasattr(MyCls, "a"):
            x = x + 1
        if hasattr(MyCls, "b"):
            x = x + 2
        return x

    @make_test
    def test_finfo(a, b):
        if torch.iinfo(torch.int32).bits == 32:
            return torch.finfo(a.dtype).min * b

    @make_test
    def test_globalfn(a, b):
        return sub(a, b)

    @make_test
    def test_viatorch(a, b):
        return torch.sub(a, b)

    @make_test
    def test_viamethod(a, b):
        return a.sub(b)

    @make_test
    def test_indirect1(a, b):
        t = a.sub
        return t(b)

    @make_test
    def test_indirect2(a, b):
        t = a.sub
        args = (b,)
        return t(*args)

    @make_test
    def test_indirect3(a, b):
        t = a.sub
        args = (b,)
        kwargs = {}
        return t(*args, **kwargs)

    @make_test
    def test_methodcall1(a, b, c):
        return constant3(a, b) * c

    @make_test
    def test_methodcall2(a, b):
        return constant3(a=b, b=a) + 1

    @make_test
    def test_methodcall3(a, b):
        return constant3(a, b=1.0) + b

    def test_is_integer(self):
        @torch.compile(backend="eager", fullgraph=True)
        def forward(t, m):
            return 2 * t if m.is_integer() else t

        t = torch.tensor([1])
        self.assertEqual(forward(t, 1.0).item(), 2)
        self.assertEqual(forward(t, 1.5).item(), 1)

    @parametrize(
        "method, num_type",
        (
            ("as_integer_ratio", int),
            ("bit_length", int),
            ("conjugate", int),
            ("as_integer_ratio", float),
            ("conjugate", float),
            ("hex", float),
            ("is_integer", float),
        ),
    )
    def test_number_method(self, method, num_type):
        def forward(t, m):
            return 2 * t if getattr(m, method)() else t

        wrapped = torch.compile(backend="eager", fullgraph=True)(forward)

        for i in (0, 1, 2.5):
            m = num_type(i)
            t = torch.tensor([1])
            actual = wrapped(t, m)
            expected = forward(t, m)
            self.assertEqual(actual, expected)

    @make_test
    def test_device_constant(a):
        return a + torch.ones(1, device=torch.device("cpu"))

    @make_test
    def test_tuple1(a, b):
        args = (a, b)
        return sub(*args)

    @make_test
    def test_tuple2(a, b):
        args = [a, b]
        return sub(*args)

    @make_test
    def test_tuple_map(a, b):
        t = tuple(map(torch.sin, [a, b]))
        return t[0] + t[1]

    def test_size_tuple_add(self):
        def fn():
            size = torch.Size([])
            assert isinstance(size + size, torch.Size)
            assert isinstance(size + (), tuple)
            assert isinstance(size + (), torch.Size)

        fn()
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled_fn()

    @make_test
    def test_is_in_onnx_export(x, y):
        if torch.onnx.is_in_onnx_export():
            return x - 1
        else:
            return y + 1

    @make_test
    def test_is_fx_tracing(x, y):
        if torch.fx._symbolic_trace.is_fx_tracing():
            return x - 1
        else:
            return y + 1

    @make_test
    def test_listarg1(a, b):
        return torch.cat([a, b])

    @make_test
    def test_listarg2(a, b):
        return torch.cat((a, b), dim=0)

    @make_test
    def test_listarg3(a, b):
        kwargs = {"tensors": (a, b), "dim": 0}
        return torch.cat(**kwargs)

    @make_test
    def test_listarg4(a, b):
        return torch.cat(tensors=[a, b], dim=0)

    @make_test
    def test_listarg5(a, b):
        args = [(a, b)]
        kwargs = {"dim": 0}
        return torch.cat(*args, **kwargs)

    def test_list_slice(self):
        class Mock:
            def __init__(self):
                self.ets = []
                self.counter = 0

            @torch.compile(backend="eager")
            def run(self, x):
                self.ets = self.ets[-3:]
                self.ets.append(x)
                return torch.sin(x)

        mock = Mock()
        mock.run(torch.randn(4))
        self.assertEqual(len(mock.ets), 1)

    @make_test
    def test_deque(a, b):
        d = collections.deque([a, b])
        d.append(a + 1)
        d.extend([a, b])
        d.insert(0, "foo")
        tmp = d.pop()

        another_deque = collections.deque([tmp])
        d.extendleft(another_deque)
        another_deque.clear()
        d.extend(another_deque)

        d[2] = "setitem"
        d = d.copy()
        d.append(d.popleft())

        empty = collections.deque()
        d.extend(empty)

        return d

    @make_test
    def test_slice1(a):
        return a[5]

    @make_test
    def test_slice2(a):
        return a[:5]

    @make_test
    def test_slice3(a):
        return a[5:]

    @make_test
    def test_slice4(a):
        return a[2:5]

    @make_test
    def test_slice5(a):
        return a[::2]

    @make_test
    def test_slice6(a):
        return torch.unsqueeze(a, 0)[:, 2:]

    @make_test
    def test_range1(a):
        return torch.tensor(range(a.size(0)))

    @make_test
    def test_range2(x, y):
        r = x + y
        for _ in range(x.size(0) + 2):
            r = r / y
        return r

    @make_test
    def test_unpack1(a):
        a, b = a[:5], a[5:]
        return a - b

    @make_test
    def test_unpack2(a):
        packed = [a[:5], a[5:]]
        a, b = packed
        return a - b

    @make_test
    def test_unpack3(a):
        packed = (a[:5], a[5:])
        a, b = packed
        return a - b

    @make_test
    def test_fn_with_self_set(a, b):
        # avg_pool2d is an odd one with __self__ set
        return F.avg_pool2d(
            torch.unsqueeze(a, 0) * torch.unsqueeze(b, 1), kernel_size=2, padding=1
        )

    @make_test
    def test_return_tuple1(a, b):
        return (a - b, b - a, a, b)

    @make_test
    def test_globalvar(a, b):
        return a - b + d

    @make_test
    def test_globalmodule(x):
        return e(x)

    @make_test
    def test_inline_with_default(a, b, c):
        return func_with_default(a, b) * c

    @make_test
    def test_inner_function(x):
        def fn(x):
            return torch.add(x, x)

        return fn(x)

    @make_test
    def test_transpose_for_scores(x):
        new_x_shape = x.size()[:-1] + (2, 5)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1)

    @make_test
    def test_return_tuple2(x):
        return (torch.add(x, x), x)

    @make_test
    def test_load_global_bool(x):
        if flag:
            return torch.add(x, x)
        else:
            return x

    @make_test
    def test_len_tensor(x):
        z = len(x)
        return torch.add(x, z)

    @make_test
    def test_len_constant_list(x):
        z = len([1, 2, 3])
        return torch.add(x, z)

    @make_test
    def test_len_constant_dict(x):
        z = len({"foo": "bar"})
        return torch.add(x, z)

    @make_test
    def test_dict_copy(x):
        z = dict({"foo": x + 1})
        return z

    @make_test
    def test_dict_keys(x):
        d = {3: x}
        keys = d.keys()
        d[4] = x + 1
        d2 = {3: 2, 4: "aa"}
        return 3 in keys, 4 in keys, 5 in keys, d2.keys() == keys

    @make_test
    def test_dict_values(x):
        d = {3: x}
        values = d.values()
        d[3] = x + 1
        d[4] = x + 2
        return len(values)

    @make_test
    def test_dict_setdefault1(x):
        d = {"a": 1, "b": 2}
        d.setdefault("a", 10)
        if d["a"] == 1:
            return x + 1
        else:
            return x - 1

    @make_test
    def test_dict_setdefault2(x):
        d = {"a": 1, "b": 2}
        d.setdefault("c", 10)
        if d["c"] == 10:
            return x + 1
        else:
            return x - 1

    @make_test
    def test_dict_setdefault3(x):
        d = {"a": 1, "b": 2}
        d.setdefault("c")
        if d["c"] is None:
            return x + 1
        else:
            return x - 1

    @make_test
    def test_dict_update_kwargs(x):
        d = {"a": 2}
        d.update(b=4)
        return x * d["a"] * d["b"]

    @make_test
    def test_defaultdict_setdefault1(x):
        d = collections.defaultdict.fromkeys("a", "b")
        d["a"] = 1
        d["b"] = 2
        d.setdefault("a", 10)
        if d["a"] == 1:
            return x + 1
        else:
            return x - 1

    @make_test
    def test_defaultdict_setdefault2(x):
        d = collections.defaultdict.fromkeys("a", "b")
        d["a"] = 1
        d["b"] = 2
        d.setdefault("c", 10)
        if d["c"] == 10:
            return x + 1
        else:
            return x - 1

    @make_test
    def test_defaultdict_setdefault3(x):
        d = collections.defaultdict.fromkeys("a", "b")
        d["a"] = 1
        d["b"] = 2
        d.setdefault("c")
        if d["c"] is None:
            return x + 1
        else:
            return x - 1

    def test_dict_id_guard(self):
        d1 = collections.OrderedDict({"a": 2})
        d2 = d1

        def fn(x):
            # Iteration forces DictGuardManager
            for k in d1:
                x = x * d1[k] * d2[k]
            return x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    @make_test
    def test_callable_lambda(x):
        if callable(lambda x: True):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_callable_torch(x):
        if callable(torch.abs):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_callable_builtin(x):
        if callable(sum):
            return x + 1
        else:
            return x - 1

    def test_callable_class(self):
        class CallableClass:
            def __call__():
                pass

        class NotCallableClass:
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn1(x, arg):
            if callable(arg):
                return x
            return x + 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn2(x, arg):
            if callable(arg):
                return x * 2
            return x + 1

        input = torch.randn(4)

        for f in [fn1, fn2]:
            self.assertEqual(f(input, NotCallableClass()), input + 1)
            self.assertEqual(
                f(input, CallableClass()), input if f is fn1 else input * 2
            )

            # passing tensor and scalars
            self.assertEqual(f(input, 1), input + 1)
            self.assertEqual(f(input, 1.1), input + 1)
            self.assertEqual(f(input, True), input + 1)
            self.assertEqual(f(input, input), input + 1)

    def test_callable_list(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, arg):
            if callable(arg):
                return x
            return x + 1

        input = torch.randn(4)
        self.assertEqual(fn(input, [1, 2, 3]), input + 1)
        self.assertEqual(fn(input, (1, 2, 3)), input + 1)

    def test_pos_only_args_with_same_name_in_star_kwargs(self):
        opt_fn = torch.compile(pos_only_fn, backend="eager", fullgraph=True)
        a = torch.randn(4)
        b = torch.randn(4)
        x = torch.randn(4)
        y = torch.randn(4)
        self.assertEqual(pos_only_fn(a), opt_fn(a))
        self.assertEqual(pos_only_fn(a, a=x), opt_fn(a, a=x))
        self.assertEqual(pos_only_fn(a, b=y), opt_fn(a, b=y))
        self.assertEqual(pos_only_fn(a, b=b, a=x), opt_fn(a, b=b, a=x))
        self.assertEqual(pos_only_fn(a, a=x, b=y), opt_fn(a, a=x, b=y))
        self.assertEqual(pos_only_fn(a, b, a=x, b=y), opt_fn(a, b, a=x, b=y))

    @make_test
    def test_len_constant_misc_iterables(x):
        a = len((1, 2, 3))
        b = len("test str")
        c = a + b
        return torch.add(x, c)

    @make_test
    def test_dict_kwargs(x):
        z = dict(text_embed=x + 1, other=x + 2)
        return z

    @make_test
    def test_ordered_dict_kwargs(x):
        z = collections.OrderedDict(sample=torch.ones(10))
        return z

    @make_test
    def test_custom_dict_kwargs(x):
        z = CustomDictSubclass(sample=torch.ones(10))
        return z

    @make_test
    def test_float(x):
        y = float(1.2)  # noqa: UP018
        y += float("1.2")
        return torch.add(x, y)

    @make_test
    def test_is_floating_point(x):
        y = x + 1
        return torch.is_floating_point(y), torch.is_floating_point(input=y)

    @make_test
    def test_dtype(x):
        if x.dtype == torch.float32:
            return x + 1

    @make_test
    def test_get_default_dtype(x):
        if x.dtype == torch.get_default_dtype():
            return x + 1
        else:
            return x - 1

    @make_test
    def test_get_autocast_gpu_dtype(x):
        dtype = torch.get_autocast_gpu_dtype()
        return x.type(dtype)

    @make_test
    def test_is_any_autocast_enabled(x):
        if torch._C._is_any_autocast_enabled():
            return x + 1
        else:
            return x - 1

    @make_test
    def test_is_checkpoint_valid(x):
        if torch.autograd._is_checkpoint_valid():
            return x + 1
        else:
            return x - 1

    @make_test
    def test_list_compare_polyfill(x):
        for a, b, c in [
            [(1, 2, 3), (1, 2, 3), 7.77],
            [(1, 4, 3), (1, 2, 3), 3.33],
            [(1, 2), (1, 2, 3), 5.55],
            [(1, 2, 3), (1, 2), 11.11],
            [(1, -1, 3), (1, 2, 3), 13.33],
        ]:
            if a != b:
                x = x + 1 * c
            if a == b:
                x = x + 2 * c
            if a < b:
                x = x + 4 * c
            if a > b:
                x = x + 8 * c
            if a <= b:
                x = x + 16 * c
            if a >= b:
                x = x + 32 * c
        return x

    @make_test
    def test_list_compare_polyfill_non_lists(x):
        conds = []

        # Non-list instances only work for eq and ne
        for a, b, c in [
            [(1, 2, 3), "(1, 2, 3)", 7.77],
            [143, (143,), 3.33],
        ]:
            conds.append(a != b)
            if conds[-1]:
                x = x + 1 * c

            conds.append(a == b)
            if conds[-1]:
                x = x + 2 * c

        return x, conds

    @make_test
    def test_promote_types(x):
        if x.dtype == torch.promote_types(torch.int32, torch.float32):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_cublas_allow_tf32(x):
        if torch.backends.cuda.matmul.allow_tf32:
            return x.sin() + 1

        return x.cos() - 1

    @make_test
    def test_get_calculate_correct_fan(x):
        fan_in = torch.nn.init._calculate_correct_fan(x, "fan_in")
        return x + fan_in

    @make_test
    def test_is_complex(x):
        if torch.is_complex(x):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_tensor_is_complex(x):
        if x.is_complex():
            return x + 1
        else:
            return x - 1

    @make_test
    def test_tensor_size(x):
        fn = torch.Tensor.size
        return fn(x + 1)

    @make_test
    def test_tensor_dim(x):
        fn = torch.Tensor.dim
        return fn(x + 1)

    def test_is_inference_recompilation(self):
        def fn(x):
            if x.is_inference():
                return x + 1
            else:
                return x - 1

        with torch.inference_mode():
            x_inference = torch.randn(2, 2)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=False)
        x = torch.randn(2, 2)

        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(cnts.frame_count, 1)

        self.assertEqual(fn(x_inference), opt_fn(x_inference))
        self.assertEqual(cnts.frame_count, 2)  # Recompiles

    def test_is_inference_mode_global_recompilation(self):
        def fn(x):
            if torch.is_inference_mode_enabled():
                return x + 1
            else:
                return x - 1

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=False)

        x = torch.randn(2, 2)

        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(cnts.frame_count, 1)

    @make_test
    def test_get_privateuse1_name(x):
        if torch._C._get_privateuse1_backend_name() == "privateuseone":
            return x + 1
        else:
            return x - 1

    @make_test
    def test_device(x):
        if not x.is_cuda:
            return x + 1

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @make_test
    def test_get_device_properties_tensor_device(a):
        x = a.to("cuda")
        prop = torch.cuda.get_device_properties(x.device)
        if prop.major == 8:
            return x + prop.multi_processor_count
        return x + prop.max_threads_per_multi_processor

    @make_test
    def test_tensor_type(a, b):
        m = a.to(torch.float16)
        return b.type(m.type())

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    @make_test
    def test_tensor_type2(a, b):
        m = a.to(device_type)
        return m + b.type(m.type())

    @make_test
    def test_tensor_type3(a, b):
        m = a.type(torch.HalfTensor)
        return b.type(m.type())

    @make_test
    def test_tensor_type4(a, b):
        m = a.type("torch.HalfTensor")
        return b.type(m.type())

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @make_test
    def test_tensor_type5(a, b):
        m = a.type(torch.cuda.HalfTensor)
        return b.type(m.type())

    @make_test
    def test_tensor_element_size(a):
        if a.element_size() > 1:
            return (a + a.element_size(), a - a.element_size())
        return (a - a.element_size(), a + a.element_size())

    @make_test
    def test_ndim(x):
        if x.ndim == 2 and x.ndimension() == 2 and x.dim() == 2:
            return x + 1

    @make_test
    def test_T(x):
        return torch.ones_like(x.T)

    @make_test
    def test_mT(x):
        return torch.ones_like(x.mT)

    @make_test
    def test_is_sparse(x):
        if not x.is_sparse:
            return x + 1

    @make_test
    def test_shape1(x):
        if x.shape[0] == 10:
            return x + 1

    @make_test
    def test_shape2(x):
        if x.size(1) == 10:
            return x + 1

    @make_test
    def test_del(a, b):
        c = a + 1
        d = c + 2
        del c, a
        return b + d

    @make_test
    def test_chunks1(x):
        chunk_size = 5
        assert x.shape[0] % chunk_size == 0
        assert x.shape[0] // chunk_size == 2
        return x[:chunk_size] - x[chunk_size:]

    @make_test
    def test_import1(x, y):
        import torch
        from torch import sub

        return sub(torch.add(x, y), y)

    @make_test
    def test_isinstance(x):
        results = []
        if isinstance([x], list):
            results.append(x.sin())
        else:
            results.append(x.cos())
        if isinstance([x], tuple):
            results.append(x.sin())
        else:
            results.append(x.cos())
        if isinstance([x], collections.abc.Sequence):
            results.append(x.sin())
        else:
            results.append(x.cos())
        if isinstance([x], typing.Sequence):
            results.append(x.sin())
        else:
            results.append(x.cos())
        if isinstance([x], (tuple, list, typing.Sequence)):
            results.append(x.sin())
        else:
            results.append(x.cos())
        # TODO: add sourceless builder for types.UnionType
        # if sys.version_info >= (3, 10):
        #     if isinstance([x], list | tuple):
        #         results.append(x.sin())
        #     else:
        #         results.append(x.cos())
        return results

    @make_test
    def test_return_dict(x, y):
        z = [x + y, y, False]
        return {"x": x, "z": z, "a": x, "b": z, "c": x}

    @make_test
    def test_return_dict2(x, y):
        tmp = {"x": x}
        tmp["z"] = [x + y, y]
        tmp["y"] = y
        tmp["z"].append(False)
        return tmp

    @make_test
    def test_funcdef_closure(x, y):
        x = x + y + 1.0

        def inner(z):
            nonlocal x, y
            y = x + z + 20.0
            x = y + z + 10.0

        inner(2.0)
        inner(3.0)

        return x, y

    @make_test
    def test_module_constant(x, y):
        r = x + y
        for _ in range(torch._dynamo.testing.three):
            r = r / y
        return r

    @make_test
    def test_inline_softmax(x, y):
        # This is common in some huggingface models
        return torch.nn.Softmax(dim=-1)(x + y * 2)

    @make_test
    def test_dtype_compare(a, b):
        if a.dtype == torch.float16:
            return a + 10
        if a.dtype == torch.float32:
            return a - b * 32

    @make_test
    def test_build_list_unpack(a, b):
        it1 = (x + 1 for x in (a, b))
        it2 = (x - 1 for x in (a, b))
        return torch.cat([*it1, *it2], dim=-1)

    @make_test
    def test_tensor_len(a, b):
        return a + b + len(a) + b.__len__()

    @make_test
    def test_pop(a, b):
        ll = [a, b]
        ll.append(a + 1)
        ll.extend(
            [
                b + 2,
                a + b,
            ]
        )
        ll.pop(-1)
        ll.pop(0)
        ll.pop()
        v1, v2 = ll
        return v1 - v2

    @make_test
    def test_list_convert(a, b):
        ll = [a + 2, b]
        ll = tuple(ll)
        tmp = b + 3
        ll = list(ll)
        v1, v2 = ll
        return v1 - v2 + tmp

    @make_test
    def test_list_add(a, b):
        l1 = (a, b)
        l2 = ()  # being a LOAD_CONST in the bytecode
        l3 = l1 + l2
        return l3[0] + l3[1]

    @make_test
    def test_list_index_with_constant_tensor(a, b):
        l1 = [a, b, a + 1, b + 1]
        return l1[torch.as_tensor(2)]

    @make_test
    def test_startswith(a, b):
        x = a + b
        if "foobar".startswith("foo") and "test" in constant3.__module__:
            x = x + 1
        return x

    @make_test
    def test_dict_ops(a, b):
        tmp = {"a": a + 1, "b": b + 2}
        assert tmp.get("zzz") is None
        v = tmp.pop("b") + tmp.get("a") + tmp.get("missing", 3) + tmp.pop("missing", 4)
        tmp.update({"d": 3})
        tmp["c"] = v + tmp["d"]
        if "c" in tmp and "missing" not in tmp:
            return tmp["c"] - tmp["a"] + len(tmp)

    @make_test
    def test_inline_jit__unwrap_optional(x):
        if torch.jit._unwrap_optional(x) is None:
            return torch.ones(2, 2)
        return x.sin()

    @make_test
    def test_zip_longest(x):
        list1 = [1, 2, 3]
        list2 = ["a", "b"]
        list3 = [True, False, True, False]
        return torch.sin(x + 1), list(
            itertools.zip_longest(list1, list2, list3, fillvalue=None)
        )

    def test_torch_size_as_dict_key(self):
        def fn(x, cached):
            if x.shape not in cached:
                cached[x.shape] = x
            return x + cached[x.shape]

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x1 = torch.randn(2, 3)
        x2 = torch.randn(2, 3)
        cached = {}
        ref1 = fn(x1, cached)
        ref2 = fn(x2, cached)
        cached = {}
        res1 = opt_fn(x1, cached)
        res2 = opt_fn(x2, cached)
        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)

    def test_dict_param_keys(self):
        a_param = torch.nn.Parameter(torch.ones([4, 4]))

        def fn(a):
            tmp = {"a": a, a_param: 3}
            return tmp["a"] + tmp[a_param]

        test = make_test(fn)
        test(self)

    def test_dict_mutable_map(self):
        from collections.abc import MutableMapping

        class TensorDict(MutableMapping):
            def __init__(self) -> None:
                self._dict = {}

            def add(self, key, value):
                self._dict[key] = value

            def items(self):
                return self._dict.items()

            def __delitem__(self, key):
                del self._dict[key]

            def __getitem__(self, key):
                return self._dict[key]

            def __iter__(self):
                return iter(self._dict)

            def __len__(self):
                return len(self._dict)

            def __setitem__(self, key, value):
                self._dict[key] = value

        tensor_dict = TensorDict()
        tensor_dict.add("a", torch.ones(4) * 2)

        def fn(x):
            copy_tensordict = dict(tensor_dict)
            return x * copy_tensordict["a"]

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_unpack_mutable_map(self):
        from collections.abc import MutableMapping

        class TensorDict(MutableMapping):
            def __init__(self) -> None:
                self._dict = {}

            def add(self, key, value):
                self._dict[key] = value

            def items(self):
                return self._dict.items()

            def __delitem__(self, key):
                del self._dict[key]

            def __getitem__(self, key):
                return self._dict[key]

            def __iter__(self):
                return iter(self._dict)

            def __len__(self):
                return len(self._dict)

            def __setitem__(self, key, value):
                self._dict[key] = value

        tensor_dict = TensorDict()
        tensor_dict.add("a", torch.ones(4) * 2)

        def gn(x, a=1):
            return x * a

        def fn(x):
            return gn(x, **tensor_dict)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.randn(4)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def _test_default_dict_helper(self, factory):
        dd = collections.defaultdict(factory)
        param = torch.nn.Parameter(torch.ones([2, 2]))

        def fn(x):
            dd["a"] = x + 1
            dd[param] = 123
            dd["c"] = x * 2
            return dd["b"], dd

        x = torch.randn(10, 10)
        ref = fn(x)
        opt_fn = torch._dynamo.optimize_assert("eager")(fn)
        res = opt_fn(x)

        self.assertTrue(same(ref[0], res[0]))
        self.assertTrue(same(ref[1]["a"], res[1]["a"]))
        self.assertTrue(same(ref[1]["c"], res[1]["c"]))
        self.assertTrue(same(ref[1][param], res[1][param]))

    def test_default_dict_dict(self):
        self._test_default_dict_helper(dict)

    def test_default_dict_list(self):
        self._test_default_dict_helper(list)

    def test_default_dict_tuple(self):
        self._test_default_dict_helper(tuple)

    def test_default_dict_set(self):
        self._test_default_dict_helper(set)

    def test_default_dict_lambda(self):
        self._test_default_dict_helper(lambda: dict())  # noqa: C408

    def test_default_dict_closure(self):
        def factory():
            return dict()  # noqa: C408

        self._test_default_dict_helper(factory)

    def test_class_dict(self):
        class A:
            x = 4
            y = 5

            def __init__(self) -> None:
                self.a = 6

        a = A()

        def fn(x):
            if "x" in type(a).__dict__:
                return x + 1
            return x + 2

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_default_dict_constr(self):
        param = torch.nn.Parameter(torch.ones([2, 2]))

        def fn(x):
            dd = collections.defaultdict(lambda: dict())  # noqa: C408
            dd["a"] = x + 1
            dd[param] = 123
            dd["c"] = x * 2
            dd.update({"b": x * 3})
            dd.update([["d", x - 2], ("e", x + 2)])
            dd.update(zip("ab", [x + 3, x + 4]))
            return dd["b"], dd

        x = torch.randn(10, 10)
        ref = fn(x)
        opt_fn = torch._dynamo.optimize_assert("eager")(fn)
        res = opt_fn(x)

        self.assertTrue(same(ref[0], res[0]))
        self.assertTrue(same(ref[1]["a"], res[1]["a"]))
        self.assertTrue(same(ref[1]["b"], res[1]["b"]))
        self.assertTrue(same(ref[1]["c"], res[1]["c"]))
        self.assertTrue(same(ref[1]["d"], res[1]["d"]))
        self.assertTrue(same(ref[1]["e"], res[1]["e"]))
        self.assertTrue(same(ref[1][param], res[1][param]))

    def test_dict_tuple_lazy_guard(self):
        @torch.compile(backend="eager")
        def fn(x, y):
            return torch.sin(x) * y[1]

        fn(torch.randn(3), {1: 1, 2: 2})
        # Changing the value of other key should not causing recompilation
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn(torch.randn(3), {1: 1, 2: 3})

        fn(torch.randn(3), (1, 2, 3))
        # Changing the value of index 0, 2 (not 1) should not cause recompilation
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn(torch.randn(3), (11, 2, 13))

    @make_test
    def test_call_dict1(x):
        d1 = dict()  # noqa: C408
        d1["x"] = x + 1
        d2 = collections.OrderedDict()
        d2["x"] = x + 2
        return d1["x"] + d2["x"] + 1

    @make_test
    def test_call_dict2(x):
        d1 = dict()  # noqa: C408
        d1["x"] = x
        d2 = collections.OrderedDict(d1)
        if isinstance(d2, collections.OrderedDict):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_call_dict3(x):
        my_list = [("a", x), ("b", x + 1), ("c", x + 2)]
        d1 = dict(my_list)
        d1["a"] = x + 10
        d2 = collections.OrderedDict(my_list)
        d2["c"] = x + 20
        return d1["a"] + d2["c"] + 1

    @make_test
    def test_call_dict4(x):
        my_list = (("a", x), ("b", x + 1), ("c", x + 2))
        d1 = dict(my_list)
        d1["a"] = x + 10
        d2 = collections.OrderedDict(my_list)
        d2["c"] = x + 20
        return d1["a"] + d2["c"] + 1

    @make_test
    def test_call_dict5(x):
        my_list = iter([("a", x), ("b", x + 1), ("c", x + 2)])
        d1 = dict(my_list)
        d1["a"] = x + 10
        d2 = collections.OrderedDict(my_list)
        d2["c"] = x + 20
        return d1["a"] + d2["c"] + 1

    @make_test
    def test_dict_fromkeys(x, y):
        lst = ["a", "b"]
        d = dict.fromkeys(lst)
        d1 = dict.fromkeys(d, x + 1)
        d2 = collections.defaultdict.fromkeys(iter(d1), x - 2)
        d3 = collections.OrderedDict.fromkeys(tuple(lst), value=y)
        return d1["a"] * d2["b"] + d2["a"] + d1["b"] + d3["a"] + d3["b"] + 1

    @make_test
    def test_dict_copy(x):
        my_list = [("a", x), ("b", x + 1), ("c", x + 2)]
        d1 = dict(my_list)
        d1["a"] = x + 10
        d2 = d1.copy()
        d2["a"] = x - 5
        d2["b"] = x + 3
        d3 = collections.OrderedDict(my_list)
        d3["c"] = x + 20
        d4 = d3.copy()
        d4["c"] = x - 10
        return d1["a"] * d2["a"] + d2["b"] + d3["c"] * d4["c"] + 1

    @make_test
    def test_dict_update(x, y, z):
        d = {"a": x, "b": y}
        d.update({"a": y - 1})
        d.update([("b", z + 1), ["c", z]])
        d.update(zip("ab", [z + 3, y + 2]))

        od = collections.OrderedDict(a=x * 3, b=y + 2)
        od.update({"a": y + 5})
        od.update([["b", z + 6], ("c", z - 7)])
        od.update(zip("ab", [z - 3, x + 2]))
        return d["a"] * od["a"] + od["c"] + d["b"] + od["b"] * d["c"]

    @make_test
    def test_min_max(a, b):
        c = a + b
        a = a.sum()
        b = b.sum()
        a = min(max(a, 0), 1)
        b = max(0, min(1, b))
        return max(a, b) - min(a, b) + c

    @make_test
    def test_symbool_to_int(x):
        # this is roughly the pattern found in einops.unpack()
        if sum(s == -1 for s in x.size()) == 0:
            return x + 1
        else:
            return x - 1

    @make_test
    def test_map_sum(a, b, c, d):
        return sum(map(lambda x: x + 1, [a, b, c, d]))

    @make_test
    def test_sum(a, b, c, d):
        return sum([a, b, c, d])

    @make_test
    def test_sum_with_start_arg(a, b, c, d):
        return sum([b, c, d], a)

    @make_test
    def test_sum_with_start_kwarg(a, b, c, d):
        return sum([b, c, d], start=a)

    @make_test(expected_frame_count=0)
    def test_sum_shortcut():
        return sum([0, 1.0, 2, 3.0])

    @make_test(expected_frame_count=0)
    def test_sum_shortcut_with_start_arg():
        return sum([0, 1.0, 2, 3.0], -10)

    @make_test(expected_frame_count=0)
    def test_sum_shortcut_with_start_kwarg():
        return sum([0, 1.0, 2, 3.0], start=-10)

    @make_test
    def test_reduce(a, b, c, d):
        return functools.reduce(operator.add, [a, b, c, d])

    @make_test
    def test_reduce_with_initial(a, b, c, d):
        return functools.reduce(operator.add, [b, c, d], a)

    @make_test
    def test_reduce_with_single(x):
        return functools.reduce(lambda a, b: (a, b), [x])

    @make_test(expected_frame_count=0)
    def test_reduce_with_single_with_initial(x, y):
        return functools.reduce(lambda a, b: (a, b), [y], x)

    @make_test(expected_frame_count=0)
    def test_reduce_with_none_initial(x):
        return functools.reduce(lambda a, b: (a, b), [x], None)

    @make_test
    def test_tuple_contains(a, b):
        v1 = "a"
        v2 = "b"
        v3 = "c"
        vals1 = (v1, v2, v3)
        vals2 = ("d", "e", "f")
        if "a" in vals1 and "b" not in vals2:
            return a + b
        return a - b

    @make_test
    def test_set_in_frozenset(x):
        var = set("abc")
        other = set([frozenset("abc")])
        if var in other:
            return x + 1
        else:
            return x - 1

    @make_test
    def test_set_update_bytecode(x):
        # This produces bytecode SET_UPDATE since python 3.9
        var = {"apple", "banana", "cherry"}
        if isinstance(var, set):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_set_update_list_with_duplicated_items(x):
        list1 = ["apple", "banana", "apple"]
        list2 = ["orange", "banana"]
        if len({*list1, *list2}) == 3:
            return x + 1
        else:
            return x - 1

    def test_set_keys_view(self):
        from collections.abc import KeysView

        class StringKeys(KeysView):
            def __init__(self, keys):
                self.keys = keys

            def __getitem__(self, key):
                return self.keys.__getitem__(key)

            def __iter__(self):
                yield from self.keys

            def __repr__(self):
                return f"{type(self).__name__}({self.keys})"

            def __len__(self):
                return len(self.keys)

            def __contains__(self, item):
                return self.keys.__contains__(item)

        a = StringKeys([1, 2, 3, 3])

        def fn(x):
            set_a = set(a)
            return len(set_a) * x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.rand(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_constant_set(self):
        s = set([1, 2])

        def fn(x):
            return torch.cos(x) * len(s)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.rand(4)
        self.assertEqual(fn(x), opt_fn(x))

        # This should cause recompilation
        s.add(3)
        self.assertEqual(fn(x), opt_fn(x))

    def test_set_add(self):
        s = set([1, 2])

        def fn(x):
            s.add(3)
            return torch.cos(x) * len(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.rand(4)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(len(s), 3)

    @make_test
    def test_tuple_iadd(a, b):
        output = (a, b)
        output += (a + b, a - b)
        return output

    @make_test
    def test_unpack_ex1(x):
        output = (x, x + 1, x + 2, x + 3)
        a, b, *cd = output
        return a - b / cd[0]

    @make_test
    def test_unpack_ex2(x):
        output = (x, x + 1, x + 2, x + 3)
        *ab, c, d = output
        return c - d / ab[0]

    @make_test
    def test_unpack_ex3(x):
        output = (x, x + 1, x + 2, x + 3)
        a, *bc, d = output
        return a - d / bc[0]

    @make_test
    def test_const_tuple_add1(x):
        output = (x, x + 1, x + 2, x + 3)
        output = () + output + ()
        return output[2] + output[3]

    @make_test
    def test_const_tuple_add2(x):
        output = (x, x + 1, x + 2, x + 3)
        output = (None,) + output + (None,)
        return 
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/dynamo`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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
python docs/test/dynamo/test_functions.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/dynamo`):

- [`test_error_messages.py_docs.md_docs.md`](./test_error_messages.py_docs.md_docs.md)
- [`test_hooks.py_kw.md_docs.md`](./test_hooks.py_kw.md_docs.md)
- [`test_unittest.py_docs.md_docs.md`](./test_unittest.py_docs.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)
- [`test_aot_autograd.py_kw.md_docs.md`](./test_aot_autograd.py_kw.md_docs.md)
- [`test_einops.py_docs.md_docs.md`](./test_einops.py_docs.md_docs.md)
- [`test_compile.py_kw.md_docs.md`](./test_compile.py_kw.md_docs.md)
- [`test_misc.py_docs.md_docs.md`](./test_misc.py_docs.md_docs.md)
- [`test_buffers_override.py_kw.md_docs.md`](./test_buffers_override.py_kw.md_docs.md)
- [`test_frame_init.py_docs.md_docs.md`](./test_frame_init.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_functions.py_docs.md_docs.md`
- **Keyword Index**: `test_functions.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
