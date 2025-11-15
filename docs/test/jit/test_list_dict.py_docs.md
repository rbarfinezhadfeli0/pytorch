# Documentation: `test/jit/test_list_dict.py`

## File Metadata

- **Path**: `test/jit/test_list_dict.py`
- **Size**: 93,597 bytes (91.40 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]
# ruff: noqa: F841

import inspect
import os
import sys
import types
import unittest
from collections import defaultdict, OrderedDict
from textwrap import dedent
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.testing import FileCheck


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import (
    raise_on_run_directly,
    skipIfTorchDynamo,
    TEST_CUDA,
)
from torch.testing._internal.jit_utils import JitTestCase, make_global


class TestList(JitTestCase):
    def test_list_bool_conversion(self):
        def if_predicate(l: List[int]):
            if l:
                s = 0
                for n in l:
                    s += n

                return s
            else:
                return -1

        self.checkScript(if_predicate, ([1, 2, 3],))
        self.checkScript(if_predicate, ([],))

        def while_predicate(l: List[int]):
            s = 0

            while l:
                s += l.pop()

        self.checkScript(while_predicate, ([1, 2, 3],))
        self.checkScript(while_predicate, ([],))

        def ternary_predicate(l: List[int]):
            return "non-empty" if l else "empty"

        self.checkScript(ternary_predicate, ([1, 2, 3],))
        self.checkScript(ternary_predicate, ([],))

    def test_in_check(self):
        def int_in(x: List[int]) -> bool:
            return 2 in x

        self.checkScript(int_in, ([1, 2, 3],))
        self.checkScript(int_in, ([1, 3, 3],))

        def float_in(x: List[float]) -> bool:
            return 2.0 in x

        self.checkScript(float_in, ([1.0, 2.0, 3.0],))
        self.checkScript(float_in, ([1.0, 3.0, 3.0],))

        def str_in(x: List[str]) -> bool:
            return "hi" in x

        self.checkScript(str_in, (["not", "here"],))
        self.checkScript(str_in, (["hi", "bye"],))
        self.checkScript(str_in, ([],))

    def test_list_literal(self):
        def reassign():
            x = [1]
            if 1 == 1:
                x = [2, 3]
            return

        self.checkScript(reassign, (), optimize=False)

        def reassign_arity_change():
            x = [1]
            if 1 == 1:
                x = [1, 2, 3]
            return

        self.checkScript(reassign_arity_change, (), optimize=False)

        def reassign_from_empty_literal():
            x = []
            if 1 == 1:
                x = [1, 2, 3]
            return

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"previously had type List\[Tensor\]", "x"
        ):
            self.checkScript(reassign_from_empty_literal, (), optimize=False)

        def reassign_from_empty_builtin():
            x = torch.jit.annotate(List[int], [])
            if 1 == 1:
                x = [1, 2, 3]
            y = torch.jit.annotate(List[float], [])
            if 1 == 1:
                y = [1.0, 2.0, 3.0]
            z = []
            if 1 == 1:
                z = [torch.randn([1])]
            return

        self.checkScript(reassign_from_empty_builtin, (), optimize=False)

        def reassign_bad_type():
            x = [1]
            if 1 == 1:
                x = [1.0]
            return

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "previously had type", "x"
        ):
            self.checkScript(reassign_bad_type, (), optimize=False)

        def reassign_nested():
            x = torch.jit.annotate(List[int], [])
            if 1 == 1:
                x = [1, 2, 3]
                if 1 == 1:
                    x = [1.0]
            return

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "previously had type", "x"
        ):
            self.checkScript(reassign_nested, (), optimize=False)

    def test_list_variance(self):
        """
        `List[T1]` is not a subtype of `List[T2]`, even if `T1` is a
        subtype of `T2`. However, if we have a temporary list object
        (that is, a list comprehension or a list literal) on the rhs of
        an assignment statement, we want to ignore the inferred type of
        the rhs if we can prove that: 1) both the lhs and the rhs are
        lists, and 2) the inner type of the lhs list is a subtype of the
        inner type of the rhs list.

        # This should pass
        x: List[Optional[int]] = [None, None, None]

        # This should fail
        y: List[None] = [None, None, None]
        x: List[Optional[int]] = y
        """

        def test_listliteral_is_typed_from_annotation():
            x: List[Optional[int]] = [None, None, None]
            return x

        self.checkScript(test_listliteral_is_typed_from_annotation, ())

        def test_listcomprehension_is_typed_from_annotation():
            x: List[Optional[int]] = [None for _ in range(3)]
            return x

        self.checkScript(test_listcomprehension_is_typed_from_annotation, ())

        def test_lists_with_different_internal_types_are_invariant(self):
            x: List[int] = [1, 2, 3]
            y: List[Optional[int]] = x
            return x

        with self.assertRaisesRegex(
            RuntimeError,
            "Variable 'y' is "
            "annotated with type "
            r"List\[Optional\[int\]\] but is "
            "being assigned to a value of type "
            r"List\[int\]",
        ):
            torch.jit.script(test_lists_with_different_internal_types_are_invariant)

        def test_lists_with_different_internal_types_are_invariant_recursive(self):
            x: List[List[int]] = [[1, 2], [3]]
            y: List[List[Optional[int]]] = x
            return x

        with self.assertRaisesRegex(
            RuntimeError,
            "Variable 'y' is "
            "annotated with type "
            r"List\[List\[Optional\[int\]\]\] "
            "but is being assigned to a value "
            r"of type List\[List\[int\]\]",
        ):
            torch.jit.script(
                test_lists_with_different_internal_types_are_invariant_recursive
            )

    def test_del(self):
        def inputs():
            return [1, 2, 3, 4]

        def fn(x: List[int]) -> List[int]:
            del x[1]
            return x

        python_out = fn(inputs())
        # checkScript reuses the same object, but here it's being mutated so do
        # it manually
        cu = torch.jit.CompilationUnit()
        cu.define(dedent(inspect.getsource(fn)))
        self.assertEqual(cu.fn(inputs()), python_out)
        self.assertEqual(torch.jit.script(fn)(inputs()), python_out)

        @torch.jit.script
        def fn2(x: List[int]) -> List[int]:
            del x[100]
            return x

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "out of range", "x[100]"
        ):
            fn2([])

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "deletion at a single index", "x[1:3]"
        ):

            @torch.jit.script
            def fn(x: List[int]) -> List[int]:
                del x[1:3]
                return x

    def test_list_keyword(self):
        def foo():
            return (
                list([1, 2, 3]),  # noqa: C410
                list(("a", "b")),  # noqa: C410
                list(range(5)),
                list("abcdefg"),
            )

        self.checkScript(foo, ())

        def foo2():
            x: List[int] = list()  # noqa: C408
            x.append(1)
            return (x,)

        self.checkScript(foo2, ())

        def foo3():
            return list(list("abc"))  # noqa: C414

        self.checkScript(foo3, ())
        FileCheck().check_count("aten::list", 2, exactly=True).run(
            torch.jit.script(foo3).graph
        )

    def test_dict_keyword_with_kwargs(self):
        def fn():
            return dict(foo=1, bar=2, baz=3)

        self.checkScript(fn, ())

    def test_dict_keyword_with_kwargs_using_container_values(self):
        def fn():
            return dict(foo=[1, 2, 3], bar=[4, 5, 6], baz=[7, 8, 9])

        self.checkScript(fn, ())

    def test_dict_keyword_with_iterable(self):
        def fn():
            return dict([("foo", 1), ("bar", 2), ("baz", 3)])  # noqa: C406

        self.checkScript(fn, ())

    def test_dict_keyword_with_empty_iterable(self):
        def fn():
            return dict([])  # noqa: C406

        self.checkScript(fn, ())

    def test_dict_keyword_with_internal_aggregate_function(self):
        def fn():
            return dict(zip(["foo", "baz", "bar"], [1, 2, 3]))

        self.checkScript(fn, ())

    def test_dict_keyword_with_mapping(self):
        def fn():
            return {"foo": 1, "bar": 2, "baz": 3}

        self.checkScript(fn, ())

    def test_dict_keyword_with_mapping_and_kwargs(self):
        def fn():
            return dict({"foo": 1, "bar": 2}, baz=3)

        self.checkScript(fn, ())

    def test_dict_keyword_with_dict_comprehension(self):
        def fn():
            return {i: chr(i + 65) for i in range(4)}

        self.checkScript(fn, ())

    def test_dict_keyword_with_dict_comprehension_and_kwargs(self):
        def fn():
            return dict({chr(65 + i): i for i in range(4)}, foo=2)

        self.checkScript(fn, ())

    def test_dict_keyword_with_empty_dict_comprehension(self):
        def fn():
            return {}

        self.checkScript(fn, ())

    def test_dict_keyword_is_correctly_typed(self):
        def fn():
            x: Dict[str, int] = dict()  # noqa: C408
            x["foo"] = 1
            return x

        self.checkScript(fn, ())

    def test_dict_keyword_with_mismatched_annotations(self):
        err_msg = (
            r"Dict type annotation `Dict\[int, str\]` did not "
            "match the type of an actual key type `str`"
        )
        with self.assertRaisesRegex(RuntimeError, err_msg):

            @torch.jit.script
            def fn():
                x: Dict[int, str] = dict(  # noqa: C406
                    [("foo", 1), ("bar", 2), ("baz", 3)]
                )
                return x

    def test_dict_keyword_with_nested_call(self):
        def fn():
            return dict(dict(foo=1, bar=2, baz=3))

        self.checkScript(fn, ())

    def test_dict_keyword_with_previously_declared_variable(self):
        def fn():
            d = {"foo": 1, "bar": 2}
            return dict(d)

        self.checkScript(fn, ())

    def test_dict_keyword_with_previously_declared_variable_and_kwargs(self):
        def fn():
            d = {"foo": 1, "bar": 2}
            return dict(d, baz=3)

        self.checkScript(fn, ())

    def test_min_bool_list(self):
        def jit_min_list(a: List[bool], b: List[bool]) -> List[bool]:
            return min(a, b)

        self.checkScript(jit_min_list, ([True, False], [False, True]))

    def test_min_max_list(self):
        def jit_min_list(a: List[int], b: List[int]) -> List[int]:
            return min(a, b)

        def jit_min_list_float(a: List[float], b: List[float]) -> List[float]:
            return min(a, b)

        def jit_min_list_bool(a: List[bool], b: List[bool]) -> List[bool]:
            return min(a, b)

        def run_tests(func, a, b):
            for t in zip(a, b):
                self.checkScript(func, t)

        args_left_int = [[1, 8, 8], [2, 1, 1], [], [2], [1], [1, 2, 3]]
        args_right_int = [[2, 1, 1], [1, 8, 8], [], [1], [], [1, 2]]
        run_tests(jit_min_list, args_left_int, args_right_int)

        args_left_float = [
            [1.0, 8.0, 8.0],
            [2.0, 1.0, 1.0],
            [],
            [2.0],
            [1.0],
            [1.0, 2.0, 3.0],
        ]
        args_right_float = [[2.0, 1.0, 1.0], [1.0, 8.0, 8.0], [], [1.0], [], [1.0, 2.0]]
        run_tests(jit_min_list_float, args_left_float, args_right_float)

        args_left_bool = [
            [],
            [],
            [],
            [False],
            [True],
            [False, True],
            [True, True],
            [False, False, False],
            [False, False, True],
        ]
        args_right_bool = [
            [],
            [False],
            [True],
            [True],
            [False],
            [True, True],
            [False, True],
            [False, False, True],
            [False, False, False],
        ]
        run_tests(jit_min_list_bool, args_left_bool, args_right_bool)

        def jit_max_list(a: List[int], b: List[int]) -> List[int]:
            return max(a, b)

        def jit_max_list_float(a: List[float], b: List[float]) -> List[float]:
            return max(a, b)

        def jit_max_list_bool(a: List[bool], b: List[bool]) -> List[bool]:
            return max(a, b)

        args_left_int = [[1, 8, 8], [8, 1, 1], [], [1], [], [1, 2]]
        args_right_int = [[8, 1, 1], [1, 8, 8], [], [2], [1], [1, 2, 3]]
        run_tests(jit_max_list, args_left_int, args_right_int)

        args_left_float = [[1.0, 8.0, 8.0], [8.0, 1.0, 1.0], [], [1.0], [], [1.0, 2.0]]
        args_right_float = [
            [8.0, 1.0, 1.0],
            [1.0, 8.0, 8.0],
            [],
            [2.0],
            [1.0],
            [1.0, 2.0, 3.0],
        ]
        run_tests(jit_max_list_float, args_left_float, args_right_float)

        run_tests(jit_max_list_bool, args_left_bool, args_right_bool)

    def test_list_gather(self):
        def index():
            a = [1, 2, 3]
            return a[1]

        self.checkScript(index, ())

        def negative_index():
            a = [1, 2, 3]
            return a[-1]

        self.checkScript(negative_index, ())

        def bad_index():
            a = [1, 2, 3]
            return a[4]

        self.checkScriptRaisesRegex(bad_index, (), Exception, "list index out of range")

        def bad_negative_index():
            a = [1, 2, 3]
            return a[-5]

        self.checkScriptRaisesRegex(
            bad_negative_index, (), Exception, "list index out of range"
        )

    def test_list_len(self):
        def func():
            a = [1, 2, 3]
            return len(a) == 3

        self.checkScript(func, ())

        def func2():
            a = []
            return len(a) == 0

        self.checkScript(func2, ())

    @skipIfTorchDynamo(
        "TorchDynamo fails to raise on this checkScriptRaisesRegex, because we trace it properly now"
    )
    def test_list_ops(self):
        def test_equality():
            a = [1, 2, 3]
            b = [1, 2, 3]
            return a == b

        self.checkScript(test_equality, (), optimize=True)

        def test_equality_str():
            a = ["foo", "bar"]
            b = ["foo", "bar"]
            return a == b

        self.checkScript(test_equality_str, (), optimize=True)

        def test_inequality():
            a = [1, 2, 3]
            b = [1, 2, 3]
            return a != b

        self.checkScript(test_inequality, (), optimize=True)

        def test_inequality_str():
            a = ["foo", "bar"]
            b = ["foo", "bar", "food"]
            return a != b

        self.checkScript(test_inequality_str, (), optimize=True)

        def test_non_equality():
            a = [1, 2, 3]
            b = [3]
            return a == b

        self.checkScript(test_non_equality, (), optimize=True)

        def test_non_inequality():
            a = [1, 2, 3]
            b = [3]
            return a != b

        self.checkScript(test_non_equality, (), optimize=True)

        def test_list_equality_as_cond():
            a = [1, 2, 3]
            b = [3]
            if a == b:
                c = 1
            else:
                c = 2
            return c

        self.checkScript(test_list_equality_as_cond, (), optimize=True)

        def test_list_add():
            a = [1, 2, 3]
            b = [2]
            c = a + b
            return c == [1, 2, 3, 2]

        self.checkScript(test_list_add, (), optimize=True)

        def test_list_add_empty():
            a = [1, 2, 3]
            b = torch.jit.annotate(List[int], [])
            c = a + b
            return c == [1, 2, 3]

        self.checkScript(test_list_add_empty, (), optimize=True)

        def test_tensor_list_equality():
            t1 = torch.ones([1, 1])
            t2 = torch.ones([1, 1])
            x = [t1, t2]
            y = [t2, t1]
            return x == y

        self.checkScript(test_tensor_list_equality, (), optimize=True)

        def test_invalid_list_equality():
            t1 = torch.ones([2, 2])
            t2 = torch.ones([2, 2])
            x = [t1, t2]
            y = [t2, t1]
            # will throw since the tensors have more than one element
            return x == y

        self.checkScriptRaisesRegex(
            test_invalid_list_equality, (), RuntimeError, "Boolean value of Tensor"
        )

    def test_list_sort(self):
        template = dedent(
            """
        def func():
            li_1 = {list_create}
            li_2 = {list_create}
            li_3 = {list_create}
            li_1.sort()
            li_2.sort(reverse=True)
            li_4 = sorted(li_3)
            return li_1, li_2, li_3, li_4
        """
        )

        lists = [
            "[]",
            "[1, 3, 2]",
            "[True, False, True]",
            "[1.2, .2, 3.2]",
            "[torch.tensor(1.0), torch.tensor(0.2), torch.tensor(0.5)]",
            "[torch.tensor(5), torch.tensor(-2), torch.tensor(4)]",
        ]
        for li in lists:
            code = template.format(list_create=li)
            scope = {}
            exec(code, globals(), scope)
            cu = torch.jit.CompilationUnit(code)
            t1 = cu.func()
            t2 = scope["func"]()
            self.assertEqual(t1, t2)

        def test_fail(x: List[Tensor]) -> List[Tensor]:
            x.sort()
            return x

        self.checkScriptRaisesRegex(
            test_fail,
            (([torch.zeros([2]), torch.zeros([2])],)),
            Exception,
            "Boolean value of Tensor with more than one value",
        )

        @torch.jit.script
        def test_mutation():
            a = [1, 2, 3]
            a.sort()
            return a

        test_mutation()
        FileCheck().check("aten::sort").run(test_mutation.graph_for())

        def test_sorted_copy():
            a = [torch.tensor(2), torch.tensor(0), torch.tensor(1)]
            b = sorted(a)
            a[0] = torch.tensor(10)
            return a, b

        self.checkScript(test_sorted_copy, ())

    def test_list_slice(self):
        def test_regular_slice():
            a = [0, 1, 2, 3, 4]
            return a[2:3] == [2]

        self.checkScript(test_regular_slice, ())

        def test_open_ended_slice():
            a = [0, 1, 2, 3, 4]
            return a[2:] == [2, 3, 4]

        self.checkScript(test_open_ended_slice, ())

        def test_open_ended_slice2():
            a = [0, 1, 2, 3, 4]
            return a[:2] == [0, 1]

        self.checkScript(test_open_ended_slice2, ())

        def test_negative_slice():
            a = [0, 1, 2, 3, 4]
            return a[:-1] == [0, 1, 2, 3]

        self.checkScript(test_negative_slice, ())

        def test_negative_slice2():
            a = [0, 1, 2, 3, 4]
            return a[-3:-1] == [2, 3]

        self.checkScript(test_negative_slice2, ())

        def test_backward_slice():
            a = [0, 1, 2, 3, 4]
            return a[3:2] == torch.jit.annotate(List[int], [])

        self.checkScript(test_backward_slice, ())

        def test_over_slice():
            a = [0, 1, 2, 3, 4]
            return a[3:10] == [3, 4]

        self.checkScript(test_backward_slice, ())

    def test_slice_index(self):
        a = torch.tensor(
            [
                [[1, 11], [2, 22]],
                [[3, 33], [4, 44]],
                [[5, 55], [6, 66]],
            ]
        )

        def test_index_slice1(x):
            x = x[:, :, [0, 1]]
            return x

        self.checkScript(test_index_slice1, (a,))

        def test_index_slice2(x):
            x = x[[2, 1, 0], :, :]
            return x

        self.checkScript(test_index_slice2, (a,))

        def test_index_slice3(x):
            x = x[[0, 1], :, [1]]
            return x

        self.checkScript(test_index_slice3, (a,))

        def test_index_slice_empty_list(x):
            empty_list: List[int] = []
            x = x[empty_list, :, :]
            return x

        self.checkScript(test_index_slice_empty_list, (a,))

        def test_index_slice_out_of_bounds_index(x):
            x = x[[4], :, :]
            return x

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "index 4 is out of bounds for dimension 0 with size 3",
            "x[[4], :, :]",
        ):
            self.checkScript(test_index_slice_out_of_bounds_index, (a,))

    def test_mutable_list_append(self):
        def test_append():
            a = [0, 1]
            a.append(2)
            a.append(3)
            return a == [0, 1, 2, 3]

        self.checkScript(test_append, ())

    def test_comprehensions_basic(self):
        def comp(l: List[int]) -> List[int]:
            n = [x * 3 for x in l]
            return n

        comp([1, 2, 3])
        self.checkScript(comp, ([1, 2, 3],))

    def test_comprehensions_basic_float(self):
        def comp(l: List[float]) -> List[float]:
            n = [x * 3 for x in l]
            return n

        self.checkScript(comp, ([1.0, 2.0, 3.0],))

    def test_comprehensions_two_comps(self):
        @torch.jit.script
        def comp(l1: List[int], l2: List[int]) -> List[int]:
            n = [x * 3 for x in l1]
            n2 = [x + 2 for x in l2]
            return n + n2

        self.assertEqual(comp([1, 2, 3], [4, 5]), [3, 6, 9, 6, 7])

    def test_comprehension_out_type_not_in_type(self):
        def list_cast() -> int:
            li = [int(i) for i in [torch.tensor(0), torch.tensor(1), torch.tensor(2)]]
            return li[0] + li[1] + li[2]

        self.checkScript(list_cast, ())

    def test_comprehension_iterable(self):
        def test_func(fn, inputs):
            self.assertEqual(fn(*inputs), torch.jit.script(fn)(*inputs))

        def foo(names: List[int], results: List[int]) -> List[Tuple[int, int]]:
            return [(k + 5, v - 2) for k, v in zip(names, results)]

        test_func(foo, ([1, 2, 4], [4, 7, 9]))
        test_func(foo, ([5], [4, 7, 9]))

        def fn(x: int) -> List[int]:
            return [i for i in range(x)]  # noqa: C416

        test_func(fn, (9,))
        test_func(fn, (0,))
        test_func(fn, (-1,))

        def changes_type():
            a = [float(i) for i in range(5)]
            b = [float(i) for i in [1, 2, 3, 4]]
            c = [(float(i), j) for i, j in enumerate([1, 2, 3, 8])]
            return a, b, c

        test_func(changes_type, ())

        def test_zero_iter():
            return [str(i) for i, j in zip("", "")]

        test_func(test_zero_iter, ())

    def test_mutable_list_append_2(self):
        def test_append_2():
            a = [0, 1]
            a.append(2)
            a = [1]
            a.append(4)
            return a == [1, 4]

        self.checkScript(test_append_2, ())

    def test_mutable_list_append_if(self):
        def test_append_if():
            a = [1]
            if 1 == 1:
                a.append(4)
            return a == [1, 4]

        self.checkScript(test_append_if, ())

    def test_mutable_list_append_if_else(self):
        def test_append_if_else():
            a = [1]
            if 1 == 2:
                a.append(4)
            else:
                a.append(10)
            return a == [1, 10]

        self.checkScript(test_append_if_else, ())

    def test_mutable_list_append_loop(self):
        def test_append_loop():
            a = torch.jit.annotate(List[int], [])
            for i in range(5):
                a.append(i)

            return a == [0, 1, 2, 3, 4]

        self.checkScript(test_append_loop, ())

    def test_mutable_list_append_loop_if(self):
        def test_append_loop_if():
            a = torch.jit.annotate(List[int], [])
            for i in range(5):
                if i > 3:
                    a.append(i)
                else:
                    a.append(0)

            return a == [0, 0, 0, 0, 4]

        self.checkScript(test_append_loop_if, ())

    def test_mutable_list_nested_loop(self):
        def test_nested_loop():
            a = torch.jit.annotate(List[int], [])
            for i in range(2):
                for j in range(2):
                    a.append(i + j)

            return a == [0, 1, 1, 2]

        self.checkScript(test_nested_loop, ())

    def test_mutable_list_function_inline(self):
        @torch.jit.script
        def bar(y: List[int]) -> None:
            y.append(4)

        @torch.jit.script
        def foo():
            x = [1, 2, 3]
            bar(x)
            return x

        self.assertEqual(foo(), [1, 2, 3, 4])

    def test_mutable_list_reverse_empty(self):
        def test_reverse_empty():
            a = []
            a.reverse()

            return a == []

        self.checkScript(test_reverse_empty, ())

    def test_mutable_list_reverse(self):
        def test_reverse():
            a = [1, 2, 3, 4]
            a.reverse()

            return a == [4, 3, 2, 1]

        self.checkScript(test_reverse, ())

    def test_mutable_tensor_list_reverse(self):
        def test_tensor_reverse():
            a = [torch.tensor(1), torch.tensor(2)]
            a.reverse()

            return a == [torch.tensor(2), torch.tensor(1)]

        self.checkScript(test_tensor_reverse, ())

    def test_mutable_list_pop_empty(self):
        @torch.jit.script
        def test_pop_empty():
            a = torch.jit.annotate(List[int], [])
            return a.pop()

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "pop from empty list", "a.pop"
        ):
            test_pop_empty()

    def test_mutable_list_pop(self):
        def test_pop():
            a = [1, 2, 3, 4]
            b = a.pop()

            return b == 4

        self.checkScript(test_pop, ())

    def test_mutable_list_pop2(self):
        def test_pop2():
            a = [1, 2, 3, 4]
            b = a.pop()

            return len(a) == 3

        self.checkScript(test_pop2, ())

    def test_mutable_list_pop_at(self):
        def test_pop_at():
            a = [1, 2, 3, 4]
            b = a.pop(1)

            return b == 2

        self.checkScript(test_pop_at, ())

    def test_mutable_list_pop_at2(self):
        def test_pop_at2():
            a = [1, 2, 3, 4]
            b = a.pop(1)

            return len(a) == 3

        self.checkScript(test_pop_at2, ())

    def test_mutable_list_pop_at_negative(self):
        def test_pop_at_negative():
            a = [1, 2, 3, 4]
            b = a.pop(-2)

            return b == 3

        self.checkScript(test_pop_at_negative, ())

    def test_mutable_list_pop_at_negative2(self):
        def test_pop_at_negative2():
            a = [1, 2, 3, 4]
            b = a.pop(-2)

            return len(a) == 3

        self.checkScript(test_pop_at_negative2, ())

    def test_mutable_list_pop_slice(self):
        def test_pop_slice():
            a = [1, 2, 3, 4]
            b = [1, 2, 3, 4]

            a.pop()
            b = b[:-1]

            return a == b

        self.checkScript(test_pop_slice, ())

    def test_mutable_list_clear_empty(self):
        def test_clear_empty():
            a = torch.jit.annotate(List[int], [])
            a.clear()

            return len(a) == 0

        self.checkScript(test_clear_empty, ())

    def test_mutable_list_clear(self):
        def test_clear():
            a = [1, 2, 3, 4]
            a.clear()

            return len(a) == 0

        self.checkScript(test_clear, ())

    def test_mutable_list_insert(self):
        def test_list_insert():
            a = [1, 2, 3, 4]
            a.insert(2, 5)

            return a == [1, 2, 5, 3, 4]

        self.checkScript(test_list_insert, ())

    def test_mutable_list_insert_negative(self):
        def test_list_insert_negative():
            a = [1, 2, 3, 4]
            a.insert(-1, 5)

            return a == [1, 2, 3, 5, 4]

        self.checkScript(test_list_insert_negative, ())

    def test_mutable_list_insert_neg_out_of_bounds(self):
        def test_list_insert_neg_out_of_bounds():
            a = [1, 2, 3, 4]
            a.insert(-10, 5)

            return a == [5, 1, 2, 3, 4]

        self.checkScript(test_list_insert_neg_out_of_bounds, ())

    def test_mutable_list_insert_out_of_bounds(self):
        def test_list_insert_out_of_bounds():
            a = [1, 2, 3, 4]
            a.insert(10, 5)

            return a == [1, 2, 3, 4, 5]

        self.checkScript(test_list_insert_out_of_bounds, ())

    def test_mutable_list_remove_not_existing(self):
        @torch.jit.script
        def test_list_remove_not_existing():
            a = [1, 2, 3, 4]
            a.remove(5)

            return a

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "x not in list", "a.remove"
        ):
            test_list_remove_not_existing()

    def test_mutable_list_remove(self):
        def test_list_remove():
            a = [1, 2, 3, 4]
            a.remove(3)

            return a == [1, 2, 4]

        self.checkScript(test_list_remove, ())

        def test_str_list_remove():
            a = ["foo", "bar"]
            a.remove("foo")

            return a == ["bar"]

        self.checkScript(test_str_list_remove, ())

    def test_list_index_not_existing(self):
        @torch.jit.script
        def list_index_not_existing():
            a = [4, 1, 3, 2]
            i = a.index(5)

            return i

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "'5' is not in list", "a.index"
        ):
            list_index_not_existing()

    def test_list_index(self):
        def list_index():
            a = [4, 1, 3, 2]
            i = a.index(3)

            return i == 2

        self.checkScript(list_index, ())

        def list_str_index():
            a = ["foo", "bar"]
            i = a.index("bar")

            return i == 1

        self.checkScript(list_str_index, ())

    def test_tensor_list_index(self):
        def tensor_list_index():
            a = [torch.tensor(4), torch.tensor(1), torch.tensor(3), torch.tensor(2)]
            i = a.index(torch.tensor(3))

            return i == 2

        self.checkScript(tensor_list_index, ())

    def test_tensor_list_index_not_existing(self):
        @torch.jit.script
        def tensor_list_index_not_existing():
            a = [torch.tensor(4), torch.tensor(1), torch.tensor(3), torch.tensor(2)]
            i = a.index(torch.tensor(5))

            return i

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "is not in list", "a.index"
        ):
            tensor_list_index_not_existing()

    def test_list_count(self):
        def list_count():
            a = [4, 1, 4, 2, 4]
            i = a.count(4)

            return i == 3

        self.checkScript(list_count, ())

        def list_str_count():
            a = ["foo", "bar", "foo"]
            i = a.count("foo")

            return i == 2

        self.checkScript(list_str_count, ())

    def test_list_count_not_existing(self):
        def list_count_not_existing():
            a = [4, 1, 4, 2, 4]
            i = a.count(5)

            return i == 0

        self.checkScript(list_count_not_existing, ())

    def test_tensor_list_count(self):
        def tensor_list_count():
            a = [torch.tensor(4), torch.tensor(1), torch.tensor(4), torch.tensor(4)]
            i = a.count(torch.tensor(4))

            return i == 3

        self.checkScript(tensor_list_count, ())

    def test_tensor_list_count_not_existing(self):
        def tensor_list_count_not_existing():
            a = [torch.tensor(4), torch.tensor(1), torch.tensor(4), torch.tensor(4)]
            i = a.count(torch.tensor(5))

            return i == 0

        self.checkScript(tensor_list_count_not_existing, ())

    def test_mutable_list_remove_tensor(self):
        def test_list_remove_tensor():
            a = [torch.ones(1), torch.zeros(1), torch.ones(2)]
            a.remove(torch.zeros(1))

            return len(a) == 2

        self.checkScript(test_list_remove_tensor, ())

    def test_mutable_list_remove2(self):
        def test_list_remove2():
            a = [1]
            a.remove(1)

            return len(a) == 0

        self.checkScript(test_list_remove2, ())

    def test_extend_list_mutable(self):
        @torch.jit.script
        def extend_list(a: List[Tensor], b: List[Tensor]) -> List[Tensor]:
            a.extend(b)
            return a

        for l in [[], [torch.rand(2)], [torch.rand(2), torch.rand(2), torch.rand(2)]]:
            for r in [
                [],
                [torch.rand(2)],
                [torch.rand(2), torch.rand(2), torch.rand(2)],
            ]:
                self.assertEqual(extend_list(l, r), l + r)

    def test_extend_list_immutable(self):
        @torch.jit.script
        def extend_list(a: List[int], b: List[int]) -> List[int]:
            a.extend(b)
            return a

        for l in [[], [1], [1, 2, 3]]:
            for r in [[], [1], [1, 2, 3]]:
                self.assertEqual(extend_list(l, r), l + r)

    def test_copy_list_mutable(self):
        @torch.jit.script
        def copy_list(a: List[Tensor]) -> List[Tensor]:
            return a.copy()

        for l in [[], [torch.rand(2)], [torch.rand(2), torch.rand(2), torch.rand(2)]]:
            self.assertEqual(copy_list(l), l)

    def test_copy_list_immutable(self):
        @torch.jit.script
        def copy_list(a: List[int]) -> List[int]:
            return a.copy()

        for l in [[], [1], [1, 2, 3]]:
            self.assertEqual(copy_list(l), l)

    def test_min_max_single_list(self):
        def min_intlist(li: List[int]) -> int:
            return min(li)

        def max_intlist(li: List[int]) -> int:
            return max(li)

        def min_boollist(li: List[bool]) -> bool:
            return min(li)

        def max_boollist(li: List[bool]) -> bool:
            return max(li)

        def min_floatlist(li: List[float]) -> float:
            return min(li)

        def max_floatlist(li: List[float]) -> float:
            return max(li)

        int_lists = [1], [2, 1, 2], [-3, 4, 2], [-2, -7, 1, 4], [2, 1, 0, 4], []

        def check_list(fn, li):
            if len(li) == 0:
                self.checkScriptRaisesRegex(fn, (li,), Exception, "empty")
            else:
                self.checkScript(fn, (li,))

        for int_list in int_lists:
            check_list(min_intlist, int_list)
            check_list(max_intlist, int_list)

            bool_li = [bool(x) for x in int_list]
            check_list(min_boollist, bool_li)
            check_list(max_boollist, bool_li)

            float_li = [float(x) for x in int_list]
            check_list(min_floatlist, float_li)
            check_list(max_floatlist, float_li)

    def test_to_list(self):
        """Unit tests for Tensor.tolist() function."""

        """
        Boolean dtype unit tests.
        """

        def to_list_bool_0D(x: torch.Tensor) -> bool:
            li = torch.jit.annotate(bool, x.tolist())
            return li

        def to_list_bool_1D(x: torch.Tensor) -> List[bool]:
            li = torch.jit.annotate(List[bool], x.tolist())
            return li

        def to_list_bool_2D(x: torch.Tensor) -> List[List[bool]]:
            li = torch.jit.annotate(List[List[bool]], x.tolist())
            return li

        def to_list_bool_3D(x: torch.Tensor) -> List[List[List[bool]]]:
            li = torch.jit.annotate(List[List[List[bool]]], x.tolist())
            return li

        self.checkScript(to_list_bool_0D, (torch.tensor(False, dtype=torch.bool),))
        bool_input_1D = torch.tensor([True, False, True, False], dtype=torch.bool)
        self.checkScript(to_list_bool_1D, (bool_input_1D,))
        bool_input_2D = torch.tensor(
            [[True, True, False], [False, True, False]], dtype=torch.bool
        )
        self.checkScript(to_list_bool_2D, (bool_input_2D,))
        bool_input_3D = torch.tensor(
            [[[True, False], [False, True]], [[True, False], [False, False]]],
            dtype=torch.bool,
        )
        self.checkScript(to_list_bool_3D, (bool_input_3D,))
        bool_input_noncontiguous = torch.tensor(
            [[[True, False], [False, True]], [[True, False], [False, False]]],
            dtype=torch.bool,
        ).transpose(0, 1)
        self.checkScript(to_list_bool_3D, (bool_input_noncontiguous,))

        """
        Int dtype unit tests.
        """

        def to_list_int_0D(x: torch.Tensor) -> int:
            li = torch.jit.annotate(int, x.tolist())
            return li

        def to_list_int_1D(x: torch.Tensor) -> List[int]:
            li = torch.jit.annotate(List[int], x.tolist())
            return li

        def to_list_int_2D(x: torch.Tensor) -> List[List[int]]:
            li = torch.jit.annotate(List[List[int]], x.tolist())
            return li

        def to_list_int_3D(x: torch.Tensor) -> List[List[List[int]]]:
            li = torch.jit.annotate(List[List[List[int]]], x.tolist())
            return li

        self.checkScript(to_list_int_0D, (torch.tensor(1, dtype=torch.long),))
        int_input_1D = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        self.checkScript(to_list_int_1D, (int_input_1D,))
        int_input_2D = torch.tensor([[1, 2, 3], [3, 4, 5]], dtype=torch.long)
        self.checkScript(to_list_int_2D, (int_input_2D,))
        int_input_3D = torch.tensor(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.long
        )
        self.checkScript(to_list_int_3D, (int_input_3D,))
        int_input_noncontiguous = torch.tensor(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.long
        ).transpose(0, 1)
        self.checkScript(to_list_int_3D, (int_input_noncontiguous,))

        """
        Float dtype unit tests.
        """

        def to_list_float_0D(x: torch.Tensor) -> float:
            li = torch.jit.annotate(float, x.tolist())
            return li

        def to_list_float_1D(x: torch.Tensor) -> List[float]:
            li = torch.jit.annotate(List[float], x.tolist())
            return li

        def to_list_float_2D(x: torch.Tensor) -> List[List[float]]:
            li = torch.jit.annotate(List[List[float]], x.tolist())
            return li

        def to_list_float_3D(x: torch.Tensor) -> List[List[List[float]]]:
            li = torch.jit.annotate(List[List[List[float]]], x.tolist())
            return li

        # Test with torch.float dtype Tensors to check that they are converted to double automatically.
        self.checkScript(to_list_float_0D, (torch.randn(5, dtype=torch.float)[0],))
        self.checkScript(to_list_float_1D, (torch.randn(5, dtype=torch.float),))
        self.checkScript(to_list_float_2D, (torch.randn(5, 6, dtype=torch.float),))
        self.checkScript(to_list_float_3D, (torch.randn(5, 6, 7, dtype=torch.float),))
        self.checkScript(
            to_list_float_3D, (torch.randn(5, 6, 7, dtype=torch.float).transpose(0, 1),)
        )

        self.checkScript(to_list_float_0D, (torch.randn(5, dtype=torch.double)[0],))
        self.checkScript(to_list_float_1D, (torch.randn(5, dtype=torch.double),))
        self.checkScript(to_list_float_2D, (torch.randn(5, 6, dtype=torch.double),))
        self.checkScript(to_list_float_3D, (torch.randn(5, 6, 7, dtype=torch.double),))
        self.checkScript(
            to_list_float_3D,
            (torch.randn(5, 6, 7, dtype=torch.double).transpose(0, 1),),
        )

        """
        Complex dtype unit tests.
        """

        def to_list_complex_0D(x: torch.Tensor) -> complex:
            li = torch.jit.annotate(complex, x.tolist())
            return li

        def to_list_complex_1D(x: torch.Tensor) -> List[complex]:
            li = torch.jit.annotate(List[complex], x.tolist())
            return li

        def to_list_complex_2D(x: torch.Tensor) -> List[List[complex]]:
            li = torch.jit.annotate(List[List[complex]], x.tolist())
            return li

        def to_list_complex_3D(x: torch.Tensor) -> List[List[List[complex]]]:
            li = torch.jit.annotate(List[List[List[complex]]], x.tolist())
            return li

        # Test with torch.complex dtype Tensors to check that they are converted to double automatically.
        self.checkScript(to_list_complex_0D, (torch.randn(5, dtype=torch.cfloat)[0],))
        self.checkScript(to_list_complex_1D, (torch.randn(5, dtype=torch.cfloat),))
        self.checkScript(to_list_complex_2D, (torch.randn(5, 6, dtype=torch.cfloat),))
        self.checkScript(
            to_list_complex_3D, (torch.randn(5, 6, 7, dtype=torch.cfloat),)
        )
        self.checkScript(
            to_list_complex_3D,
            (torch.randn(5, 6, 7, dtype=torch.cfloat).transpose(0, 1),),
        )

        self.checkScript(to_list_complex_0D, (torch.randn(5, dtype=torch.cdouble)[0],))
        self.checkScript(to_list_complex_1D, (torch.randn(5, dtype=torch.cdouble),))
        self.checkScript(to_list_complex_2D, (torch.randn(5, 6, dtype=torch.cdouble),))
        self.checkScript(
            to_list_complex_3D, (torch.randn(5, 6, 7, dtype=torch.cdouble),)
        )
        self.checkScript(
            to_list_complex_3D,
            (torch.randn(5, 6, 7, dtype=torch.cdouble).transpose(0, 1),),
        )

        """
        Non-happy path tests:
            - missing type annotation
            - mismatch between type annotation and input
            - type annotation with unsupported type
            - type annotation with the wrong dimension
            - type annotation with scalar type that doesn't match the input scalar type
        """

        def to_list_missing_type_annotation(x: torch.Tensor) -> List[float]:
            li = x.tolist()
            return li

        def to_list_incorrect_type_annotation(x: torch.Tensor) -> List[float]:
            li = torch.jit.annotate(float, x.tolist())
            return li

        def to_list_unsupported_type_annotation(x: torch.Tensor) -> List[float]:
            li = torch.jit.annotate(List[str], x.tolist())
            return li

        def to_list_type_annotation_wrong_dim(x: torch.Tensor) -> List[List[float]]:
            li = torch.jit.annotate(List[List[float]], x.tolist())
            return li

        def to_list_type_annotation_incorrect_scalar_type(
            x: torch.Tensor,
        ) -> List[float]:
            li = torch.jit.annotate(List[float], x.tolist())
            return li

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"Expected type hint for result of tolist()", "x.tolist("
        ):
            self.checkScript(to_list_missing_type_annotation, (torch.randn(5),))

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            r"Return value was annotated as having type List\[float\] but is actually of type float",
            "return li",
        ):
            self.checkScript(to_list_incorrect_type_annotation, (torch.randn(5),))

        with self.assertRaisesRegex(
            RuntimeError, r"str is not one of the supported element types for tolist"
        ):
            self.checkScript(to_list_unsupported_type_annotation, (torch.randn(5),))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Output annotation list dimension and runtime tensor dimension must match",
        ):
            self.checkScript(
                to_list_type_annotation_wrong_dim, (torch.randn(5, dtype=torch.double),)
            )

        with self.assertRaisesRegex(
            RuntimeError,
            r"Output annotation element type and runtime tensor element type must match",
        ):
            self.checkScript(
                to_list_type_annotation_incorrect_scalar_type,
                (torch.ones(5, dtype=torch.long),),
            )

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_to_list_gpu(self):
        """GPU tests for Tensor.tolist() function."""

        def to_list_bool_1D(x: torch.Tensor) -> List[bool]:
            li = torch.jit.annotate(List[bool], x.tolist())
            return li

        def to_list_int_1D(x: torch.Tensor) -> List[int]:
            li = torch.jit.annotate(List[int], x.tolist())
            return li

        def to_list_float_1D(x: torch.Tensor) -> List[float]:
            li = torch.jit.annotate(List[float], x.tolist())
            return li

        self.checkScript(
            to_list_bool_1D,
            (torch.tensor([True, False, True, False], dtype=torch.bool).cuda(),),
        )
        self.checkScript(
            to_list_int_1D, (torch.tensor([1, 2, 3, 4], dtype=torch.long).cuda(),)
        )
        self.checkScript(to_list_float_1D, (torch.randn(5, dtype=torch.double).cuda(),))

    def test_no_element_type_annotation(self):
        def fn_with_comment(x: torch.Tensor) -> List:
            a: List = x.tolist()
            return a

        def annotated_fn(x: torch.Tensor) -> List:
            a: List = x.tolist()
            return a

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use List without a contained type"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(fn_with_comment)))

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use List without a contained type"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(annotated_fn)))

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use List without a contained type"
        ):
            torch.jit.script(fn_with_comment)

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use List without a contained type"
        ):
            torch.jit.script(annotated_fn)

    def test_list_none(self):
        with self.assertRaisesRegex(
            RuntimeError, "Can not create ListType with None type"
        ):
            x = torch._C.ListType(None)

    def test_list_unification_hint(self):
        with self.assertRaisesRegex(
            RuntimeError, "Expected an annotation of type List"
        ):

            @torch.jit.script
            def x():
                b: int = [2, 3]
                return b


class TestDict(JitTestCase):
    def dict(self):
        return {"a": torch.ones(1), "b": torch.ones(1) + 1, "c": torch.ones(1) + 2}

    def dict2(self):
        return {
            "x": torch.ones(1) + 100,
            "y": torch.ones(1) + 101,
            "z": torch.ones(1) + 102,
        }

    def dict_bool(self):
        return {True: 1}

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_dict_bool_conversion(self):
        def if_predicate(d: Dict[int, int]):
            if d:
                s, t = 0, 0
                for k, v in d.items():
                    s += k
                    t += v

                return s, t
            else:
                return -1, -1

        self.checkScript(if_predicate, ({1: 2, 3: 5},))
        self.checkScript(if_predicate, ({},))

        def while_predicate(d: Dict[int, int]):
            while d:
                d.clear()

        self.checkScript(while_predicate, ({1: 2, 3: 5},))
        self.checkScript(while_predicate, ({},))

        def ternary_predicate(d: Dict[int, int]):
            return "non-empty" if d else "empty"

        self.checkScript(ternary_predicate, ({1: 2, 3: 5},))
        self.checkScript(ternary_predicate, ({},))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_del(self):
        def inputs():
            return {"hi": 2, "bye": 3}

        def fn(x: Dict[str, int]) -> Dict[str, int]:
            del x["hi"]
            return x

        python_out = fn(inputs())
        # checkScript reuses the same object, but here it's being mutated so do
        # it manually
        cu = torch.jit.CompilationUnit()
        cu.define(dedent(inspect.getsource(fn)))
        self.assertEqual(cu.fn(inputs()), python_out)
        self.assertEqual(torch.jit.script(fn)(inputs()), python_out)
        with self.assertRaisesRegexWithHighlight(RuntimeError, "KeyError", 'x["hi"]'):
            self.checkScript(fn, [{}])

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_dict_variance(self):
        """
        `Dict[T1, _]` is not a subtype of `Dict[T2, _]`, even if `T1` is
        a subtype of `T2`; similarly `Dict[_, T1]` would not be a
        subtype of `Dict[_, T2]`.

        However, if we have a temporary dict object (that is, a dict
        comprehension or a dict literal) on the rhs of an assignment
        statement, we want to ignore the inferred type of the rhs if we
        can prove that: 1) both the lhs and the rhs are dicts with the
        same key types (TorchScript has a restricted set of allowed key
        types, so we don't need to worry about subtyping relationships
        here), and 2) the value type of the dict is a subtype of the
        value type of the rhs dict.
        """

        def test_dictliteral_is_typed_from_annotation():
            x: Dict[str, Optional[int]] = {"foo": None, "bar": None, "baz": None}
            return x

        self.checkScript(test_dictliteral_is_typed_from_annotation, ())

        def test_dictcomprehension_is_typed_from_annotation():
            
```



## High-Level Overview


This Python file contains 28 class(es) and 429 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestList`, `TestDict`, `M`, `TestNamedTuple`, `FeatureVector`, `Tup`, `FeatureVector`, `Config`, `MyMod`, `TheType`, `MyModule`, `MyCoolNamedTuple`, `MyCoolNamedTuple`, `MyCoolNamedTuple`, `MyCoolNamedTuple`, `MyCoolNamedTuple`, `MyCoolNamedTuple`, `MyMod`, `FeatureVector`, `MyNamedTuple`

**Functions defined**: `test_list_bool_conversion`, `if_predicate`, `while_predicate`, `ternary_predicate`, `test_in_check`, `int_in`, `float_in`, `str_in`, `test_list_literal`, `reassign`, `reassign_arity_change`, `reassign_from_empty_literal`, `reassign_from_empty_builtin`, `reassign_bad_type`, `reassign_nested`, `test_list_variance`, `test_listliteral_is_typed_from_annotation`, `test_listcomprehension_is_typed_from_annotation`, `test_lists_with_different_internal_types_are_invariant`, `test_lists_with_different_internal_types_are_invariant_recursive`

**Key imports**: inspect, os, sys, types, unittest, defaultdict, OrderedDict, dedent, Any, Dict, List, NamedTuple, Optional, Tuple, torch, torch.nn as nn


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `inspect`
- `os`
- `sys`
- `types`
- `unittest`
- `collections`: defaultdict, OrderedDict
- `textwrap`: dedent
- `typing`: Any, Dict, List, NamedTuple, Optional, Tuple
- `torch`
- `torch.nn as nn`
- `torch.testing`: FileCheck
- `torch.testing._internal.jit_utils`: JitTestCase, make_global


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/jit/test_list_dict.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/jit`):

- [`test_dataclasses.py_docs.md`](./test_dataclasses.py_docs.md)
- [`test_recursive_script.py_docs.md`](./test_recursive_script.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_python_builtins.py_docs.md`](./test_python_builtins.py_docs.md)
- [`test_functional_blocks.py_docs.md`](./test_functional_blocks.py_docs.md)
- [`test_hooks_modules.py_docs.md`](./test_hooks_modules.py_docs.md)
- [`mydecorator.py_docs.md`](./mydecorator.py_docs.md)
- [`test_union.py_docs.md`](./test_union.py_docs.md)
- [`test_python_bindings.py_docs.md`](./test_python_bindings.py_docs.md)
- [`test_parametrization.py_docs.md`](./test_parametrization.py_docs.md)


## Cross-References

- **File Documentation**: `test_list_dict.py_docs.md`
- **Keyword Index**: `test_list_dict.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
