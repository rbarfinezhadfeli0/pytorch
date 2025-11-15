# Documentation: `docs/test/test_python_dispatch.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_python_dispatch.py_docs.md`
- **Size**: 54,559 bytes (53.28 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_python_dispatch.py`

## File Metadata

- **Path**: `test/test_python_dispatch.py`
- **Size**: 104,016 bytes (101.58 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: __torch_dispatch__"]
# ruff: noqa: F841

import pickle
import sys
import tempfile
import unittest
from copy import deepcopy

import torch
import torch._dynamo
from torch import SymInt
from torch._C import DispatchKey, DispatchKeySet
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.cuda.jiterator import _create_jit_fn
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.library import _scoped_library, fallthrough_kernel, impl, Library
from torch.multiprocessing.reductions import StorageWeakRef
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    first_sample,
    IS_WINDOWS,
    run_tests,
    TEST_WITH_ROCM,
    TestCase,
)
from torch.testing._internal.custom_op_db import custom_op_db
from torch.testing._internal.logging_tensor import (
    capture_logs,
    capture_logs_with_logging_tensor_mode,
    log_input,
    LoggingTensor,
    LoggingTensorMode,
    LoggingTensorReentrant,
)
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils import _pytree as pytree
from torch.utils._mode_utils import all_same_mode, no_dispatch
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _get_current_dispatch_mode_stack,
    is_in_torch_dispatch_mode,
    TorchDispatchMode,
)
from torch.utils._pytree import tree_map, tree_map_only


# used as DataLoader collate_fn below; named here to avoid trying to pickle a lambda
def _identity(x):
    return x


class TestDispatcherPythonBindings(TestCase):
    def test_call_boxed(self) -> None:
        sin = torch._C._dispatch_find_schema_or_throw("aten::sin", "")
        x = torch.randn(3)
        y = torch._C._dispatch_call_boxed(sin, x)
        self.assertEqual(y, x.sin())


class TestPythonRegistration(TestCase):
    test_ns = "_test_python_registration"

    def tearDown(self):
        if hasattr(torch.ops, self.test_ns):
            del torch.ops._test_python_registration

    def test_fallback(self) -> None:
        test_key = "TESTING_ONLY_GenericMode"
        test_keyset = torch._C.DispatchKeySet(test_key)
        include_to_set = torch._C._dispatch_tls_local_include_set() | test_keyset
        exclude_to_set = torch._C._dispatch_tls_local_exclude_set()

        with _scoped_library("_", "IMPL") as my_lib:
            expected_op = None
            expected_args = None
            expected_kwargs = None
            # Use this out shape to make sure the result from our fallback
            # is what is returned to the user
            out_shape = None

            def my_fallback(op, *args, **kwargs):
                # Disable our handler during checks and generating the output
                with torch._C._ForceDispatchKeyGuard(
                    include_to_set, exclude_to_set | test_keyset
                ):
                    self.assertIs(op, expected_op)
                    self.assertEqual(args, expected_args)
                    self.assertEqual(kwargs, expected_kwargs)
                    # Return something specific
                    return torch.empty(out_shape)

            my_lib.fallback(my_fallback, test_key)

            a, b = torch.rand(2), torch.rand(2)

            with torch._C._ForceDispatchKeyGuard(include_to_set, exclude_to_set):
                # Check a factory function
                expected_op = torch.ops.aten.empty.memory_format
                expected_args = ((2, 2),)
                # Extra kwargs to bypass issues with default args in factory functions
                expected_kwargs = {
                    "dtype": torch.float64,
                    "pin_memory": False,
                    "device": torch.device("cpu"),
                }
                out_shape = (3,)
                out = torch.empty(*expected_args, **expected_kwargs)
                self.assertEqual(out.size(), out_shape)

                # Check a regular function
                expected_op = torch.ops.aten.add.Tensor
                expected_args = (a, b)
                expected_kwargs = {}
                out_shape = (4,)
                out = a + b
                self.assertEqual(out.size(), out_shape)

    def test_fallback_keyset(self) -> None:
        test_key_first = "TESTING_ONLY_GenericMode"
        test_key_second = "TESTING_ONLY_GenericWrapper"
        test_keyset = torch._C.DispatchKeySet(test_key_first) | torch._C.DispatchKeySet(
            test_key_second
        )
        include_to_set = torch._C._dispatch_tls_local_include_set() | test_keyset
        exclude_to_set = torch._C._dispatch_tls_local_exclude_set()

        with _scoped_library("_", "IMPL") as my_lib:
            first_called = False
            second_called = False

            def first_fallback(keyset, op, *args, **kwargs):
                nonlocal first_called
                if second_called:
                    # Recursive call
                    first_called = True
                    with torch._C._ForceDispatchKeyGuard(
                        include_to_set, exclude_to_set | test_keyset
                    ):
                        return op(*args, **kwargs)
                else:
                    # Redispatch down
                    keyset = keyset.remove(test_key_first)
                    return op.redispatch(keyset, *args, **kwargs)

            def second_fallback(op, *args, **kwargs):
                nonlocal second_called
                # Set to avoid infinite recursion
                second_called = True
                # New dispatcher call should hit the first callback again
                self.assertFalse(first_called)
                a, b = args
                # Make a subtraction here instead of add !
                c = a - b
                self.assertTrue(first_called)
                return c

            my_lib.fallback(first_fallback, test_key_first, with_keyset=True)
            my_lib.fallback(second_fallback, test_key_second)

            a, b = torch.rand(2), torch.rand(2)
            with torch._C._ForceDispatchKeyGuard(include_to_set, exclude_to_set):
                c = a + b

            self.assertEqual(c, a - b)
            self.assertTrue(first_called)
            self.assertTrue(second_called)

    def test_fallback_fallthrough(self) -> None:
        test_key_first = "TESTING_ONLY_GenericMode"
        test_key_second = "TESTING_ONLY_GenericWrapper"
        test_keyset = torch._C.DispatchKeySet(test_key_first) | torch._C.DispatchKeySet(
            test_key_second
        )
        include_to_set = torch._C._dispatch_tls_local_include_set() | test_keyset
        exclude_to_set = torch._C._dispatch_tls_local_exclude_set()

        with _scoped_library("_", "IMPL") as my_lib:
            is_called = False

            def my_fallback(op, *args, **kwargs):
                nonlocal is_called
                is_called = True
                with torch._C._ForceDispatchKeyGuard(
                    include_to_set, exclude_to_set | test_keyset
                ):
                    return op(*args, **kwargs)

            my_lib.fallback(torch.library.fallthrough_kernel, test_key_first)
            my_lib.fallback(my_fallback, test_key_second)

            a, b = torch.rand(2), torch.rand(2)
            with torch._C._ForceDispatchKeyGuard(include_to_set, exclude_to_set):
                c = a + b

            self.assertEqual(c, a + b)
            self.assertTrue(is_called)

    @unittest.skip(
        "Causing flakiness, see https://github.com/pytorch/pytorch/issues/145108"
    )
    def test_fallthrough_for_dense_key_with_meta_in_tls(self) -> None:
        # This tests that if meta is included in TlS dispatch key set,
        # then a meta kernel should be called regardless if a dense
        # backend has a fallthrough kernel

        a = torch.randn((3, 3))
        with _scoped_library("custom", "DEF") as my_lib:
            my_lib.define("sum(Tensor self) -> Tensor")
            meta_is_called = False

            def sum_meta(*args, **kwargs):
                nonlocal meta_is_called
                meta_is_called = True
                return args[0]

            my_lib.impl("sum", fallthrough_kernel, "CPU")
            my_lib.impl("sum", sum_meta, "Meta")

            with torch._C._IncludeDispatchKeyGuard(torch.DispatchKey.Meta):
                torch.ops.custom.sum.default(a)
                self.assertTrue(meta_is_called)

    def test_dispatchkeyset_pickle(self) -> None:
        keyset = torch._C.DispatchKeySet(torch._C.DispatchKey.AutogradCPU)
        serialized = pickle.dumps(keyset)
        new_keyset = pickle.loads(serialized)
        self.assertEqual(new_keyset, keyset)

    def test_dispatchkeyset_eq(self) -> None:
        a = torch._C.DispatchKeySet(torch._C.DispatchKey.AutogradCPU)
        b = torch._C.DispatchKeySet(torch._C.DispatchKey.AutogradCPU)
        c = torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)
        self.assertTrue(a == b)
        self.assertFalse(a != b)
        self.assertTrue(a != c)

    def test_override_aten_ops_with_multiple_libraries(self) -> None:
        x = torch.tensor([1, 2])
        with _scoped_library("aten", "IMPL") as my_lib2:
            with _scoped_library("aten", "IMPL") as my_lib1:
                # Example 1
                def my_neg(*args, **kwargs):
                    return args[0]._neg_view()

                # Now we are secretly making the operator a view op so autograd needs to know how
                # to handle it
                my_lib1.impl("neg", my_neg, "AutogradCPU")

                self.assertTrue(torch.neg(x).is_neg())

                # RuntimeError: impl("aten::neg", ...):
                # Explicitly provided namespace (aten) in operator name does not match ...
                with self.assertRaisesRegex(
                    RuntimeError, "operator name does not match namespace"
                ):
                    with _scoped_library("foo", "DEF") as my_lib3:
                        my_lib3.define("neg(Tensor self) -> Tensor")
                        my_lib3.impl(torch.ops.aten.neg.default, my_neg, "AutogradCPU")

                # Example 2
                def my_mul(*args, **kwargs):
                    return torch.zeros_like(args[0])

                # torch.ops.aten.mul.Tensor
                my_lib2.impl("aten::mul.Tensor", my_mul, "ZeroTensor")

                y = torch._efficientzerotensor(2)
                self.assertFalse(torch.mul(x, y)._is_zerotensor())

                # Assert that a user can't override the behavior of a (ns, op, dispatch_key)
                # combination if someone overridden the behavior for the same before them
                with self.assertRaisesRegex(
                    RuntimeError, "already a kernel registered from python"
                ):
                    my_lib2.impl(torch.ops.aten.mul.Tensor, my_mul, "ZeroTensor")

            # Validate that lib2 is not affected by removing lib1
            self.assertFalse(torch.mul(x, y)._is_zerotensor())

        # Validate that the old behavior is restored for neg and mul
        self.assertFalse(torch.neg(x).is_neg())
        self.assertTrue(torch.mul(x, y)._is_zerotensor())

    def test_error_if_fn_not_callable(self):
        with self.assertRaisesRegex(
            TypeError, "Input function is required to be a callable"
        ):
            with _scoped_library("aten", "IMPL") as my_lib:
                my_lib.impl(torch.ops.aten.neg.default, [], "AutogradCPU")

    def test_finalizer(self):
        impls_refcnt = sys.getrefcount(torch.library._impls)
        lib = Library(self.test_ns, "FRAGMENT")  # noqa: TOR901
        lib.define("foo123(Tensor x) -> Tensor")

        # 1 for `lib`, 1 for sys.getrefcount
        self.assertEqual(sys.getrefcount(lib), 2)
        # We gained an additional reference that gets cleared when the finalizer runs
        self.assertEqual(sys.getrefcount(torch.library._impls), impls_refcnt + 1)
        # 1 for `lib`
        # 1 for the finalizer
        # 1 for sys.getrefcount
        self.assertEqual(sys.getrefcount(lib._op_impls), 3)

        def foo123(x):
            pass

        lib.impl(f"{self.test_ns}::foo123", foo123, "CPU")
        key = f"{self.test_ns}/foo123/CPU"
        self.assertTrue(key in torch.library._impls)

        saved_op_impls = lib._op_impls

        # del will definitely work if the following passes
        self.assertEqual(sys.getrefcount(lib), 2)
        del lib

        # 1 for saved_op_impls
        # 1 for sys.getrefcount
        # This function should be the last user of lib._op_impls:
        # - lib should not have a reference anymore (it was del'ed)
        # - lib's finalizer should not have a reference anymore
        self.assertEqual(sys.getrefcount(saved_op_impls), 2)

        self.assertTrue(key not in torch.library._impls)

        # lib's finalizer should not have a reference anymore
        self.assertEqual(sys.getrefcount(torch.library._impls), impls_refcnt)

    def test_override_cpu_sum(self) -> None:
        # Example 1
        run = [False]

        def my_sum(*args, **kwargs):
            run[0] = True
            return args[0].clone()

        with _scoped_library("aten", "IMPL") as my_lib1:
            my_lib1.impl("aten::sum", my_sum, "CPU")
            x = torch.tensor([1, 2])
            self.assertEqual(torch.sum(x), x)
            self.assertTrue(run[0])
        # Validate that the old behavior is restored for sum
        self.assertEqual(torch.sum(x), torch.tensor(3))

    def test_override_cuda_with_jiterator(self) -> None:
        def override_where_cuda() -> None:
            # Example 1: Invert the behavior of where's condition input
            not_where_code_string = """
            template <typename T> T inverted_where(bool cond, T a, T b){
                return !cond ? a : b;
            }
            """
            jitted_where = _create_jit_fn(not_where_code_string)

            CALLED = [False]

            def inverted_where(*args, **kwargs):
                CALLED[0] = True
                return jitted_where(*args, **kwargs)

            # overriding where's cuda kernel with Jiterator generated kernel
            with _scoped_library("aten", "IMPL") as my_lib:
                my_lib.impl("aten::where.self", inverted_where, "CUDA")

                device = "cuda"
                cond = torch.tensor(
                    [True, True, False], device=device, dtype=torch.bool
                )
                x = torch.tensor([1, 2, 3], device=device)
                y = torch.tensor([-1, -2, -3], device=device)

                self.assertEqual(torch.where(cond, x, y), torch.tensor([-1, -2, 3]))
                self.assertTrue(CALLED[0])

            # behavior restored after deregistration
            self.assertEqual(torch.where(cond, x, y), torch.tensor([1, 2, -3]))

        def override_gelu_cuda() -> None:
            # Example 2: Use relu to approximate gelu for faster compute
            fastest_gelu_code_string = """
            template <typename T> T fast_gelu(T a){
                return a > 0 ? a : 0;
            }
            """
            jitted_gelu = _create_jit_fn(fastest_gelu_code_string)

            CALLED = [False]

            def fast_gelu(*args, **kwargs):
                CALLED[0] = True
                return jitted_gelu(*args, **kwargs)

            # overriding gelu's cuda kernel with Jiterator generated relu kernel
            with _scoped_library("aten", "IMPL") as my_lib:
                my_lib.impl("aten::gelu", fast_gelu, "CUDA")

                x = torch.rand([3, 3], device="cuda", dtype=torch.float)
                self.assertEqual(
                    torch.nn.functional.gelu(x), torch.nn.functional.relu(x)
                )
                self.assertTrue(CALLED[0])

            # behavior restored after deregistration
            self.assertNotEqual(
                torch.nn.functional.gelu(x), torch.nn.functional.relu(x)
            )

        def override_exp_cuda() -> None:
            # Example 3: Preventing exp from exploding for float16
            clipped_exp_code_string = """
            template <typename T> T clipped_exp(T a){
                return a > T(10.0) ? T(22026.4657948) : exp(a);
            }
            """
            jitted_exp = _create_jit_fn(clipped_exp_code_string)

            CALLED = [False]

            def clipped_exp(*args, **kwargs):
                CALLED[0] = True
                return jitted_exp(*args, **kwargs)

            # overriding exp's cuda kernel with clipped_exp kernel
            with _scoped_library("aten", "IMPL") as my_lib:
                my_lib.impl("aten::exp", clipped_exp, "CUDA")

                x = torch.tensor([0.0, 100.0], device="cuda", dtype=torch.float16)
                self.assertEqual(
                    torch.exp(x),
                    torch.tensor([1.0, 22026.4657948], dtype=torch.float16),
                )
                self.assertTrue(CALLED[0])

            # behavior restored after deregistration
            self.assertEqual(
                torch.exp(x), torch.tensor([1.0, torch.inf], dtype=torch.float16)
            )

        def override_add_cuda() -> None:
            # Example 4: simulate a hardware bug, where the adder is always off by 1
            buggy_add_code_string = """
            template <typename T> T buggy_add(T a, T b){
                return a + b + T(1);
            }
            """
            jitted_add = _create_jit_fn(buggy_add_code_string)

            CALLED = [False]

            def buggy_add(*args, **kwargs):
                CALLED[0] = True
                return jitted_add(*args, **kwargs)

            with _scoped_library("aten", "IMPL") as my_lib:
                my_lib.impl("aten::add.Tensor", buggy_add, "CUDA")

                x_cpu = torch.rand([3, 3], device="cpu")
                y_cpu = torch.rand([3], device="cpu")

                x_cuda = x_cpu.cuda()
                y_cuda = y_cpu.cuda()

                self.assertEqual(x_cuda + y_cuda, x_cpu + y_cpu + 1)
                self.assertTrue(CALLED[0])

            # behavior restored after deregistration
            self.assertEqual(x_cuda + y_cuda, x_cpu + y_cpu)

        if torch.cuda.is_available() and not TEST_WITH_ROCM:
            override_where_cuda()
            override_gelu_cuda()
            override_exp_cuda()
            override_add_cuda()

    def test_extend_library_with_dispatch_key_arg(self):
        def my_sum(*args, **kwargs):
            return args[0].clone()

        with _scoped_library("aten", "IMPL", dispatch_key="CPU") as my_lib1:
            # RuntimeError: Explicitly provided dispatch key (Conjugate) is
            # inconsistent with the dispatch key of the enclosing TORCH_LIBRARY_IMPL block
            with self.assertRaisesRegex(
                RuntimeError, "inconsistent with the dispatch key"
            ):
                my_lib1.impl("sum", my_sum, "Conjugate")
            my_lib1.impl("aten::sum", my_sum)
            x = torch.tensor([1, 2])
            self.assertEqual(torch.sum(x), x)

    def test_create_new_library(self) -> None:
        with _scoped_library(self.test_ns, "DEF") as my_lib1:
            my_lib1.define("sum(Tensor self) -> Tensor")

            # Example 1
            @torch.library.impl(my_lib1, "sum", "CPU")
            def my_sum(*args, **kwargs):
                return args[0].clone()

            x = torch.tensor([1, 2])
            op = getattr(torch.ops, self.test_ns).sum
            self.assertEqual(op(x), x)

            with _scoped_library(self.test_ns, "IMPL") as my_lib2:
                # Example 2
                @torch.library.impl(my_lib2, op.default, "ZeroTensor")
                def my_sum_zt(*args, **kwargs):
                    if args[0]._is_zerotensor():
                        return torch._efficientzerotensor(args[0].shape)
                    else:
                        return args[0].clone()

                y = torch._efficientzerotensor(3)
                self.assertTrue(op(y)._is_zerotensor())
                self.assertEqual(op(x), x)

    def test_create_new_library_fragment_no_existing(self):
        with _scoped_library(self.test_ns, "FRAGMENT") as my_lib:
            my_lib.define("sum2(Tensor self) -> Tensor")

            @torch.library.impl(my_lib, "sum2", "CPU")
            def my_sum(*args, **kwargs):
                return args[0]

            x = torch.tensor([1, 2])
            self.assertEqual(getattr(torch.ops, self.test_ns).sum2(x), x)

    def test_create_new_library_fragment_with_existing(self):
        with _scoped_library(self.test_ns, "DEF") as my_lib1:
            # Create a fragment
            with _scoped_library(self.test_ns, "FRAGMENT") as my_lib2:
                my_lib2.define("sum4(Tensor self) -> Tensor")

                @torch.library.impl(my_lib2, "sum4", "CPU")
                def my_sum4(*args, **kwargs):
                    return args[0]

                x = torch.tensor([1, 2])
                self.assertEqual(getattr(torch.ops, self.test_ns).sum4(x), x)

                # Create another fragment
                with _scoped_library(self.test_ns, "FRAGMENT") as my_lib3:
                    my_lib3.define("sum3(Tensor self) -> Tensor")

                    @torch.library.impl(my_lib3, "sum3", "CPU")
                    def my_sum3(*args, **kwargs):
                        return args[0]

                    x = torch.tensor([1, 2])
                    self.assertEqual(getattr(torch.ops, self.test_ns).sum3(x), x)

    @unittest.skipIf(IS_WINDOWS, "Skipped under Windows")
    def test_alias_analysis(self):
        def test_helper(alias_analysis=""):
            my_lib1 = Library(self.test_ns, "DEF")  # noqa: TOR901

            called = [0]

            @torch.library.define(
                my_lib1, "_op() -> None", alias_analysis=alias_analysis
            )
            def _op(*args, **kwargs):
                called[0] += 1

            @torch.jit.script
            def _test():
                torch.ops._test_python_registration._op()

            assert "_test_python_registration::_op" in str(_test.graph)

        with self.assertRaises(AssertionError):
            test_helper("")  # alias_analysis="FROM_SCHEMA"

        test_helper("CONSERVATIVE")

    def test_error_for_unsupported_ns_or_kind(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported kind"):
            my_lib1 = Library("myns", "BLA")  # noqa: TOR901

        for kind in ("DEF", "FRAGMENT"):
            with self.assertRaisesRegex(ValueError, "reserved namespace"):
                my_lib1 = Library("prim", kind)  # noqa: TOR901

    def test_dispatcher_error_filenames(self) -> None:
        # Test that dispatcher errors report correct Python filenames and line numbers
        # when defining duplicate libraries (which triggers the filename tracking)
        import linecache
        import re

        # Create first library
        # NOTE: Using Library directly instead of _scoped_library because this test
        # specifically verifies filename tracking in error messages, and _scoped_library
        # would report library.py locations instead of the actual test file locations
        lib1 = Library(self.test_ns, "DEF")  # FIRST_LIB_MARKER  # noqa: TOR901
        try:
            lib1.define("duplicate_op(Tensor x) -> Tensor")

            # Try to create another library with same namespace - this should trigger error
            with self.assertRaises(RuntimeError) as cm:
                lib2 = Library(self.test_ns, "DEF")  # SECOND_LIB_MARKER  # noqa: TOR901
        finally:
            lib1._destroy()

        error_msg = str(cm.exception)

        # The error should NOT contain /dev/null (the old placeholder)
        self.assertNotIn("/dev/null", error_msg)
        # The error should contain the test file name for both registrations
        self.assertIn("test_python_dispatch.py", error_msg)
        # Extract line numbers from the error message and verify they point to the right lines
        line_matches = re.findall(r"test_python_dispatch\.py:(\d+)", error_msg)
        self.assertEqual(
            len(line_matches), 2, "Should have exactly 2 line number references"
        )

        # Get the actual source lines and verify they contain our markers
        first_line_num, second_line_num = sorted([int(x) for x in line_matches])
        first_line = linecache.getline(__file__, first_line_num).strip()
        second_line = linecache.getline(__file__, second_line_num).strip()

        # Verify the lines contain our expected markers
        self.assertIn("FIRST_LIB_MARKER", first_line)
        self.assertIn("SECOND_LIB_MARKER", second_line)

    def test_returning_symint(self) -> None:
        shape_env = ShapeEnv()
        fake_tensor_mode = FakeTensorMode(shape_env=shape_env)

        ft = fake_tensor_mode.from_tensor(torch.rand(2, 3))

        s0, s1 = ft.shape

        with _scoped_library(self.test_ns, "DEF") as tlib:
            tlib.define("sqsum(SymInt a, SymInt b) -> SymInt")

            @impl(tlib, "sqsum", "CompositeExplicitAutograd")
            def sqsum(a: SymInt, b: SymInt):
                return a * a + b * b

            out = getattr(torch.ops, self.test_ns).sqsum.default(s0, s1)
            out_val = shape_env.evaluate_expr(out.node.expr)
        self.assertEqual(out_val, 13)

    def test_register_fallthrough(self):
        with _scoped_library("aten", "IMPL") as my_lib:
            my_lib.impl("mm", fallthrough_kernel, "AutocastCPU")

            a = torch.randn(2, 3, device="cpu", dtype=torch.float32)
            b = torch.randn(3, 2, device="cpu", dtype=torch.float32)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                # dtype for mm should be float32 since we registered a fallthrough
                self.assertEqual(torch.mm(a, b).dtype, torch.float32)
                # ops that don't have a fallthrough registered should not be affected
                self.assertEqual(torch.matmul(a, b).dtype, torch.bfloat16)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            # default behavior should have been restored
            self.assertEqual(torch.mm(a, b).dtype, torch.bfloat16)


class TestPythonDispatch(TestCase):
    def test_basic(self) -> None:
        with capture_logs() as logs:
            x = LoggingTensor(torch.tensor([3.0]), requires_grad=True)
            log_input("x", x)
            y = x * x
            saved_x = y.grad_fn._saved_self
            grad_y = LoggingTensor(torch.tensor([1.0]))
            log_input("grad_y", grad_y)
            (g,) = torch.autograd.grad((y,), (x,), (grad_y,))

        self.assertEqual(g.elem, torch.tensor([6.0]))
        with torch.no_grad():
            self.assertEqual(saved_x, x)
            self.assertEqual(saved_x._version, x._version)
            x.add_(2)
            self.assertEqual(saved_x, x)
            # TODO: figure out why broken
            # self.assertEqual(saved_x._version, x._version)
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[1] = input('x')
$1: f32[1] = torch._ops.aten.mul.Tensor($0, $0)
$2: f32[1] = input('grad_y')
$3: f32[1] = torch._ops.aten.mul.Tensor($2, $0)
$4: f32[1] = torch._ops.aten.mul.Tensor($2, $0)
$5: f32[1] = torch._ops.aten.add.Tensor($4, $3)""",
        )

    def test_out(self) -> None:
        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(1))
            y = LoggingTensor(torch.zeros(1))
            log_input("x", x)
            log_input("y", y)
            torch.abs(x, out=y)

        self.assertEqual(y.elem, torch.ones(1))
        # TODO: arguably this shouldn't pass and we should complain
        # that out isn't a kwarg
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[1] = input('x')
$1: f32[1] = input('y')
$2: f32[1] = torch._ops.aten.abs.out($0, out=$1)""",
        )

    def test_kwarg_only(self) -> None:
        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(1))
            y = LoggingTensor(torch.ones(1, 1))
            z = LoggingTensor(torch.ones(1))
            log_input("x", x)
            log_input("y", y)
            log_input("z", z)
            torch.addmv(x, y, z)
            torch.addmv(x, y, z, beta=1)
            torch.addmv(x, y, z, beta=2)
            torch.addmv(x, y, z, alpha=2)
            torch.addmv(x, y, z, beta=2, alpha=2)

        # The expectation is that beta/alpha don't show up when they're
        # defaulted.  This is even if the user explicitly specified it.
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[1] = input('x')
$1: f32[1, 1] = input('y')
$2: f32[1] = input('z')
$3: f32[1] = torch._ops.aten.addmv.default($0, $1, $2)
$4: f32[1] = torch._ops.aten.addmv.default($0, $1, $2)
$5: f32[1] = torch._ops.aten.addmv.default($0, $1, $2, beta=2)
$6: f32[1] = torch._ops.aten.addmv.default($0, $1, $2, alpha=2)
$7: f32[1] = torch._ops.aten.addmv.default($0, $1, $2, beta=2, alpha=2)""",
        )

    def test_kwarg_only_and_positional_default(self) -> None:
        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(1))
            log_input("x", x)
            torch.ops.aten._foobar(x)
            torch.ops.aten._foobar(x, False)
            torch.ops.aten._foobar(x, arg3=False)
            torch.ops.aten._foobar(x, False, arg3=False)

        # What we are testing here is that we omit arg2
        # if it is defaulted, even if a kwarg is set
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[1] = input('x')
$1: f32[1] = torch._ops.aten._foobar.default($0)
$2: f32[1] = torch._ops.aten._foobar.default($0, False)
$3: f32[1] = torch._ops.aten._foobar.default($0, arg3=False)
$4: f32[1] = torch._ops.aten._foobar.default($0, False, arg3=False)""",
        )

    def test_produce_real_type(self) -> None:
        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(2, 2))
            log_input("x", x)
            x.to(dtype=torch.double)  # non-optional dtype
            torch.cumprod(x, 0, dtype=torch.double)  # optional dtype
            x[:, 1].contiguous(
                memory_format=torch.contiguous_format
            )  # optional memory format
            # There doesn't appear to be any layout signatures which are
            # triggerable using tensor subclasses (need to use a mode)

        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[2, 2] = input('x')
$1: f64[2, 2] = torch._ops.aten._to_copy.default($0, dtype=torch.float64)
$2: f64[2, 2] = torch._ops.aten.cumprod.default($0, 0, dtype=torch.float64)
$3: f32[2] = torch._ops.aten.select.int($0, 1, 1)
$4: f32[2] = torch._ops.aten.clone.default($3, memory_format=torch.contiguous_format)""",
        )

    def test_optional_tensor_list(self) -> None:
        def weird(xs):
            print("woof")
            return torch.empty(())

        with _scoped_library("my_lib", "DEF") as my_lib:
            my_lib.define("weird(Tensor?[] self) -> Tensor")
            my_lib.impl("weird", weird, "CPU")
            with capture_logs() as logs:
                x = LoggingTensor(torch.ones(2, 2))
                log_input("x", x)
                torch.ops.my_lib.weird.default([None, x])

        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[2, 2] = input('x')
$1: f32[] = torch._ops.my_lib.weird.default(['None', '$0'])""",
        )

    def test_list_ret(self) -> None:
        # test all sequence types are permissible returns
        for list_type in (list, tuple):

            class A(torch.Tensor):
                @staticmethod
                def __new__(cls, elem):
                    return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

                @classmethod
                def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                    if func.overloadpacket == torch.ops.aten.split:
                        with no_dispatch():
                            return list_type(torch.split(*args))
                    else:
                        raise AssertionError(f"unrecognized func: {func}")

            self.assertEqual(
                torch.split(A(torch.tensor([0, 1])), 2),
                torch.split(torch.tensor([0, 1]), 2),
            )

    def test_invalid_ret(self) -> None:
        # test invalid return gets reasonable error message
        class A(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                return "arf"

        # Wobbles depending on NDEBUG mode of pybind11
        self.assertRaisesRegex(
            RuntimeError,
            "Unable to cast",
            lambda: A(torch.zeros(1)).neg(),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Unable to cast",
            lambda: A(torch.zeros(1)).detach(),
        )

    def test_detach_appears_once_when_called_once(self) -> None:
        with capture_logs() as logs:
            x = LoggingTensor(torch.tensor([3.0]), requires_grad=True)
            log_input("x", x)
            x.detach()
        # FIXME: We actually want this to emit a single detach. However,
        # it currently emits two, for reasons unclear to us. Leaving
        # this test here to make sure we don't regress even further (it
        # would be bad if calling .detach() once emits 3+ detaches).
        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[1] = input('x')
$1: f32[1] = torch._ops.aten.detach.default($0)""",
        )

    def test_storage(self) -> None:
        # For now, just make sure it doesn't crash.  Ideally, we should
        # return some virtual storage that is safe to work with
        x = LoggingTensor(torch.ones(1))
        storage = x.untyped_storage()
        self.assertRaises(RuntimeError, lambda: storage.data_ptr())

    def test_make_wrapper_subclass_noalloc(self) -> None:
        # This is ludicrously big (8TB) and this should pass because wrapper
        # subclasses don't allocate
        torch.Tensor._make_wrapper_subclass(LoggingTensor, (1000000000000,))

    def test_version(self) -> None:
        x = LoggingTensor(torch.ones(1))
        prev_vc = x._version
        x.detach().add_(2)
        cur_vc = x._version
        self.assertNotEqual(prev_vc, cur_vc)
        x.data.add_(2)
        self.assertEqual(cur_vc, x._version)

    def test_subclass_priority(self) -> None:
        class ErrorA(RuntimeError):
            pass

        class ErrorB(RuntimeError):
            pass

        # The big tests for code coverage are test_precedence_semantics in
        # test_overrides.py; this is just to make sure it is wired up at all
        # correctly for __torch_dispatch__
        class A(torch.Tensor):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                raise ErrorA

        class B(A):
            @staticmethod
            def __new__(cls, elem):
                return torch.Tensor._make_subclass(cls, elem, elem.requires_grad)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                raise ErrorB

        self.assertRaises(
            ErrorA, lambda: torch.add(A(torch.empty(1)), A(torch.empty(1)))
        )
        self.assertRaises(
            ErrorB, lambda: torch.add(A(torch.empty(1)), B(torch.empty(1)))
        )
        self.assertRaises(
            ErrorB, lambda: torch.add(B(torch.empty(1)), A(torch.empty(1)))
        )
        self.assertRaises(
            ErrorB, lambda: torch.add(B(torch.empty(1)), B(torch.empty(1)))
        )

    def test_format(self) -> None:
        x = LoggingTensor(torch.ones(1))
        s1 = str(x)
        s2 = repr(x)
        s3 = f"{x}"
        self.assertExpectedInline(s1, """LoggingTensor(tensor([1.]))""")
        self.assertEqual(s1, s2)
        self.assertEqual(s1, s3)

    def test_custom_autograd(self) -> None:
        escape = [None]

        class Square(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = x**2
                ctx.save_for_backward(x)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                assert isinstance(grad_output, LoggingTensor)
                (x,) = ctx.saved_tensors
                assert isinstance(x, LoggingTensor)
                escape[0] = x
                return grad_output * 2 * x

        with capture_logs() as logs:
            x = LoggingTensor(torch.ones(1), requires_grad=True)
            log_input("x", x)
            x.grad = LoggingTensor(torch.zeros(1))
            log_input("x.grad", x.grad)
            y = Square.apply(x)
            grad_output = LoggingTensor(torch.ones(1))
            log_input("grad_output", grad_output)
            y.backward(grad_output)

        with torch.no_grad():
            self.assertEqual(escape[0], x)
            self.assertEqual(escape[0]._version, x._version)
            # TODO: figure out why x.requires_grad = False doesn't
            # trigger an error for LoggingTensor
            x.add_(2)
            self.assertEqual(escape[0], x)
            # TODO: figure out why this is broken
            # self.assertEqual(escape[0]._version, x._version)

        self.assertExpectedInline(
            "\n".join(logs),
            """\
$0: f32[1] = input('x')
$1: f32[1] = input('x.grad')
$2: f32[1] = torch._ops.aten.pow.Tensor_Scalar($0, 2)
$3: f32[1] = input('grad_output')
$4: f32[1] = torch._ops.aten.mul.Tensor($3, 2)
$5: f32[1] = torch._ops.aten.mul.Tensor($4, $0)
$6: f32[1] = torch._ops.aten.add_.Tensor($1, $5)""",
        )

    def test_subclass_creation(self):
        # Make sure these statements runs without error
        # In particular checking that when internal detach returns
        # subclasses, these are cleanly overwritten.
        class Foo(torch.Tensor):
            pass

        err_msg = "subclass Foo but.*already associated to a python object of type LoggingTensor"
        with self.assertRaisesRegex(RuntimeError, err_msg):
            a = torch.Tensor._make_subclass(Foo, LoggingTensor(torch.rand(2)))
        with self.assertRaisesRegex(RuntimeError, err_msg):
            b = LoggingTensor(torch.rand(2)).as_subclass(Foo)
        with self.assertRaisesRegex(RuntimeError, err_msg):
            Foo(LoggingTensor(torch.rand(2)))

        with self.assertRaisesRegex(TypeError, "Foo must define __torch_dispatch__"):
            torch.Tensor._make_wrapper_subclass(Foo, (2, 2))

    def test_new_ones(self) -> None:
        class MyTensor(torch.Tensor):
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                return MyTensor(3)

        self.assertEqual(type(MyTensor(2).new_ones(3)), MyTensor)

    def test_like(self) -> None:
        class MyTensor(torch.Tensor):
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                return MyTensor(3)

        for f in ["empty", "ones", "rand", "randn", "zeros"]:
            f_name = f + "_like"
            self.assertEqual(type(getattr(torch, f_name)(MyTensor(2))), MyTensor)

        self.assertEqual(type(torch.full_like(MyTensor(2), 1.0)), MyTensor)
        self.assertEqual(type(torch.randint_like(MyTensor(2), high=3)), MyTensor)

    def test_make_fx_with_subclass(self) -> None:
        def f(x, y):
            # Returns (TwoTensor, Tensor)
            return x * y, y + y

        x_a = torch.zeros(4)
        x_b = torch.zeros(4)
        y = torch.ones(4)

        # make_fx() is not responsible for unwrapping tensor subclass inputs,
        # so we do it manually here.
        # Why? In general, make_fx(f)(*args) promises that the graph returned has the same calling
        # convention as f(*args). Unwrapping tensor subclass inputs can potentially change
        # the number of input args to the graph, breaking that assumption
        def f_to_trace(x_a, x_b, y):
            x = TwoTensor(x_a, x_b)
            out1, out2 = f(x, y)
            out1_unwrapped_attrs, _ = out1.__tensor_flatten__()
            return (*[getattr(out1, attr) for attr in out1_unwrapped_attrs], out2)

        fx_g = make_fx(f_to_trace, tracing_mode="fake")(x_a, x_b, y)
        self.assertExpectedInline(
            fx_g.code,
            """\



def forward(self, x_a_1, x_b_1, y_1):
    mul = torch.ops.aten.mul.Tensor(x_a_1, y_1);  x_a_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(x_b_1, y_1);  x_b_1 = None
    add = torch.ops.aten.add.Tensor(y_1, y_1);  y_1 = None
    return (mul, mul_1, add)
    """,
        )

    # See https://github.com/pytorch/pytorch/issues/117794
    def test_return_and_correct_aliasing_gives_correct_stride(self):
        t = TwoTensor(torch.randn(2, 2), torch.randn(2, 2))
        x = torch.randn(2, 2)
        # slicing should result in the same stride for TwoTensor as a dense tensor would give
        self.assertEqual(t[:, 0].stride(), x[:, 0].stride())

    def test_make_wrapper_subclass_propagates_metadata(self) -> None:
        class WrapperTensor(torch.Tensor):
            elem: torch.Tensor

            __slots__ = ["elem"]

            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
                    cls,
                    elem.size(),
                    dtype=elem.dtype,
                    layout=elem.layout,
                    device=elem.device,
                    requires_grad=elem.requires_grad,
                    strides=elem.stride(),
                    storage_offset=elem.storage_offset(),
                )
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                raise RuntimeError("NYI")

        # non-contiguous strides, non-zero storage offset
        x = torch.randn(4, 6).t().diagonal(offset=2)
        y = WrapperTensor(x)
        self.assertEqual(y.size(), x.size())
        self.assertEqual(y.stride(), x.stride())
        self.assertEqual(y.storage_offset(), x.storage_offset())

    def test_wrapper_subclass_serializes(self) -> None:
        with tempfile.TemporaryFile() as f:
            # purposefully use int64 to test non-default dtype
            x = LoggingTensor(torch.randperm(3))
            torch.save(x, f)
            f.seek(0)
            with torch.serialization.safe_globals([LoggingTensor]):
                x_loaded = torch.load(f)
            self.assertTrue(type(x_loaded) is type(x))
            self.assertEqual(x, x_loaded)
            self.assertEqual(x.elem, x_loaded.elem)
            self.assertFalse(x is x_loaded)

    def test_deepcopy_wrapper_subclass(self) -> None:
        # purposefully use int64 to test non-default dtype
        x = LoggingTensor(torch.randperm(3))
        x_copy = deepcopy(x)
        self.assertTrue(type(x_copy) is type(x))
        self.assertEqual(x, x_copy)
        self.assertEqual(x.elem, x_copy.elem)
        self.assertFalse(x is x_copy)

    def test_deepcopy_wrapper_subclass_with_clone_returning_different_type(
        self,
    ) -> None:
        class MyWrapperTensor(torch.Tensor):
            elem: torch.Tensor

            __slots__ = ["elem"]

            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
                    cls,
                    elem.size(),
                    dtype=elem.dtype,
                    layout=elem.layout,
                    device=elem.device,
                    requires_grad=elem.requires_grad,
                    strides=elem.stride(),
                    storage_offset=elem.storage_offset(),
                )
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if func.overloadpacket.__name__ == "clone":
                    # Return a plain tensor from clone().
                    return args[0].elem.clone()
                raise RuntimeError("NYI")

            # NB: The default Tensor.__torch_function__ implementation called for deepcopy
            # disables __torch_function__ by the time we get to clone(), so there is no need to
            # explicitly disable __torch_function__ for this subclass.

        x = MyWrapperTensor(torch.randn(3))
        with self.assertRaisesRegex(
            RuntimeError,
            "for which cloning returns another instance of the same subclass",
        ):
            x_copy = deepcopy(x)

    def test_deepcopy_non_wrapper_subclass(self) -> None:
        # Ensure correct error is thrown for common error cases.
        class SubTensorError1(torch.Tensor):
            # Default implementation of new_empty() returns a plain tensor.
            pass

        class SubTensorError2(torch.Tensor):
            # new_empty() incorrectly returns a different type (i.e. a plain tensor).
            def new_empty(self, shape):
                return torch.Tensor(shape)

        for error_cls in [SubTensorError1, SubTensorError2]:
            x = error_cls(3)
            with self.assertRaisesRegex(
                RuntimeError,
                "for which that function returns another instance of the same subclass",
            ):
                x_copy = deepcopy(x)

        # Ensure a correctly implemented new_empty() causes deepcopy() to work.
        class SubTensorSuccess(torch.Tensor):
            def new_empty(self, shape):
                return type(self)(shape)

        x = SubTensorSuccess(3)
        x_copy = deepcopy(x)
        self.assertIs(type(x_copy), type(x))

    def test_wrapper_subclass_extra_dispatch_keys(self) -> None:
        class ExtraKeysTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                # NB: only the non-kwarg overload of _make_wrapper_subclass supports
                #     extra dispatch keys. We probably want to unify the two APIs
                #     in the future.
                r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
                    cls,
                    elem.size(),
                    elem.stride(),
                    elem.storage_offset(),
                    torch.contiguous_format,
                    elem.dtype,
                    elem.layout,
                    elem.device,
                    False,
                    False,
                    None,
                    False,
                    False,
                    DispatchKeySet(DispatchKey.NestedTensor),
                )
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                pass

        x = ExtraKeysTensor(torch.randn(3))
        self.assertTrue(torch._C._dispatch_keys(x).has(DispatchKey.NestedTensor))
        self.assertFalse(
            torch._C._dispatch_keys(x).has(DispatchKey.AutogradNestedTensor)
        )

    def test_wrapper_subclass_multiprocessing_preserves_dtype(self):
        # a and b have dtype of int64, which is purposefully different from the default
        # assumed by _make_wrapper_subclass().
        a = torch.randperm(5)
        b = torch.randperm(5)
        data = TwoTensor(a, b)
        expected_dtype = data.dtype

        loader = torch.utils.data.DataLoader(
            [data, data],
            batch_size=2,
            num_workers=2,
            collate_fn=_identity,
        )
        for batch in loader:
            self.assertEqual(batch[0].dtype, expected_dtype)

    def test_index_put_where_only_index_is_subclass(self) -> None:
        called_funcs = []

        class MyTensor(torch.Tensor):
            elem: torch.Tensor
            __slots__ = ["elem"]

            @staticmethod
            def __new__(cls, elem, *args, **kwargs):
                r = torch.Tensor._make_wrapper_subclass(
                    cls,
                    elem.size(),
                    dtype=elem.dtype,
                    layout=elem.layout,
                    device=elem.device,
                    requires_grad=elem.requires_grad,
                )
                r.elem = elem
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                called_funcs.append(func)
                return MyTensor(torch.tensor(3))

        x = torch.randn(3, 3)
        idxs = (MyTensor(torch.tensor(0)),)
        v = torch.randn(1)
        res = x.index_put_(idxs, v)
        self.assertEqual(called_funcs, [torch.ops.aten.index_put_.default])
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

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

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_python_dispatch.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_tensorexpr.py_docs.md_docs.md`](./test_tensorexpr.py_docs.md_docs.md)
- [`pytest_shard_custom.py_docs.md_docs.md`](./pytest_shard_custom.py_docs.md_docs.md)
- [`test_weak.py_kw.md_docs.md`](./test_weak.py_kw.md_docs.md)
- [`test_view_ops.py_kw.md_docs.md`](./test_view_ops.py_kw.md_docs.md)
- [`test_varlen_attention.py_kw.md_docs.md`](./test_varlen_attention.py_kw.md_docs.md)
- [`test_namedtensor.py_docs.md_docs.md`](./test_namedtensor.py_docs.md_docs.md)
- [`test_binary_ufuncs.py_docs.md_docs.md`](./test_binary_ufuncs.py_docs.md_docs.md)
- [`test_ops_gradients.py_kw.md_docs.md`](./test_ops_gradients.py_kw.md_docs.md)
- [`test_torchfuzz_repros.py_docs.md_docs.md`](./test_torchfuzz_repros.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_python_dispatch.py_docs.md_docs.md`
- **Keyword Index**: `test_python_dispatch.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
