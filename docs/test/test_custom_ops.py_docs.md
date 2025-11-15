# Documentation: `test/test_custom_ops.py`

## File Metadata

- **Path**: `test/test_custom_ops.py`
- **Size**: 176,476 bytes (172.34 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: custom-operators"]
# ruff: noqa: F841

import collections
import io
import itertools
import os
import re
import subprocess
import sys
import tempfile
import typing
import unittest
from functools import partial
from pathlib import Path
from typing import *  # noqa: F403

import numpy as np
import yaml

import torch._custom_ops as custom_ops
import torch.testing._internal.optests as optests
import torch.utils._pytree as pytree
import torch.utils.cpp_extension
from torch import Tensor
from torch._custom_op.impl import CustomOp, infer_schema
from torch._library.fake_profile import (
    generate_yaml_from_profiles,
    load_op_profiles,
    MissingOpProfile,
    OpProfile,
    read_profiles_from_yaml,
    save_op_profiles,
    TensorMetadata,
)
from torch._library.infer_schema import tuple_to_list
from torch._library.opaque_object import make_opaque, OpaqueType
from torch._utils_internal import get_file_path_2  # @manual
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal import custom_op_db
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    run_tests,
    scoped_load_inline,
    skipIfTorchDynamo,
    subtest,
    TemporaryFileName,
    TestCase,
)
from torch.testing._internal.custom_op_db import numpy_nonzero
from torch.testing._internal.two_tensor import TwoTensor


# Shadowed by `torch.testing._internal.common_utils.custom_op`
from torch._custom_op.impl import custom_op  # usort: skip

# Needed by TestTypeConversion.test_string_type:
MyList = list
MyTensor = torch.Tensor


def requires_compile(fun):
    fun = unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")(fun)
    return fun


class CustomOpTestCaseBase(TestCase):
    test_ns = "_test_custom_op"

    def setUp(self):
        super().setUp()
        self.libraries = []

    def tearDown(self):
        super().tearDown()
        import torch._custom_op

        keys = list(torch._custom_op.impl.global_registry.keys())
        for key in keys:
            if not key.startswith(f"{self.test_ns}::"):
                continue
            torch._custom_op.impl.global_registry[key]._destroy()
        if hasattr(torch.ops, self.test_ns):
            delattr(torch.ops, self.test_ns)
        for lib in self.libraries:
            lib._destroy()
        del self.libraries

    def ns(self):
        return getattr(torch.ops, self.test_ns)

    def lib(self):
        result = torch.library.Library(self.test_ns, "FRAGMENT")  # noqa: TOR901
        self.libraries.append(result)
        return result

    def get_op(self, qualname):
        return torch._custom_op.impl.get_op(qualname)


@requires_compile
class TestCustomOpTesting(CustomOpTestCaseBase):
    @parametrize("check_gradients", (False, "auto"))
    @parametrize("dynamic", (True, False))
    def test_aot_autograd_check_degenerate_cases(
        self, device, dynamic, check_gradients
    ):
        def simple(x):
            return x.clone()

        # Should not raise
        x = torch.randn(3, device=device)
        optests.aot_autograd_check(
            simple, (x,), {}, dynamic=dynamic, check_gradients=check_gradients
        )

        def outputs_dont_require_grad(x):
            return x.detach()

        # Should not raise
        y = torch.randn(3, device=device, requires_grad=True)
        optests.aot_autograd_check(
            simple, (y,), {}, dynamic=dynamic, check_gradients=check_gradients
        )

        def no_outputs(x):
            return x.detach()

        # Should not raise
        x = torch.randn(3, device=device, requires_grad=True)
        y = torch.randn(3, device=device, requires_grad=False)
        optests.aot_autograd_check(
            no_outputs, (x,), {}, dynamic=dynamic, check_gradients=check_gradients
        )
        optests.aot_autograd_check(
            no_outputs, (y,), {}, dynamic=dynamic, check_gradients=check_gradients
        )

    def test_incorrect_schema_mutation(self, device):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                guard = torch._C._AutoDispatchBelowAutograd()
                try:
                    return op(x)
                finally:
                    del guard

            @staticmethod
            def backward(ctx, gx):
                return gx

        def foo_impl(x):
            x.sin_()
            return x.clone()

        lib.impl("foo", Foo.apply, "Autograd")
        lib.impl("foo", foo_impl, "CPU")
        lib.impl("foo", foo_impl, "CUDA")
        lib.impl("foo", foo_impl, "XPU")

        x = torch.tensor(3.14159 / 3, requires_grad=True, device=device)
        with self.assertRaisesRegex(
            optests.OpCheckError, "Argument x is not defined as mutable but was mutated"
        ):
            torch.library.opcheck(op, (x,), {})

    def test_incorrect_schema_view(self, device):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # Emulate AutoDispatchBelowADInplaceOrView, which is not bound into python
                with torch._C._AutoDispatchBelowAutograd():
                    with torch._C._ExcludeDispatchKeyGuard(
                        torch._C.DispatchKeySet(torch._C.DispatchKey.ADInplaceOrView)
                    ):
                        return op(x)

            @staticmethod
            def backward(ctx, gx):
                return gx

        def foo_impl(x):
            return x.view_as(x)

        def foo_meta(x):
            return x.view_as(x)

        lib.impl("foo", Foo.apply, "Autograd")
        lib.impl("foo", foo_impl, "CPU")
        lib.impl("foo", foo_meta, "Meta")

        x = torch.tensor(3.14159 / 3, requires_grad=True)
        with self.assertRaisesRegex(
            optests.OpCheckError,
            "Argument x is not defined to alias output but was aliasing",
        ):
            torch.library.opcheck(op, (x,), {})

    # https://github.com/pytorch/pytorch/issues/142410
    def test_opcheck_unbacked_stride(self, device):
        @torch.library.custom_op("test::f", mutates_args=[])
        def f(x: torch.Tensor) -> torch.Tensor:
            return x.new_zeros((x.size(0), 18))

        @f.register_fake
        def _(x: torch.Tensor) -> torch.Tensor:
            ctx = torch.library.get_ctx()
            s = ctx.new_dynamic_size()
            return torch.empty(x.shape[0], s, device=x.device, dtype=x.dtype)

        example = torch.zeros([10, 20], device=device)
        torch.library.opcheck(f, args=[example])

    # https://github.com/pytorch/pytorch/issues/150472
    def test_single_element_tuple_output(self, device):
        # Helper function to register id_tuple custom and the fake tensor implementation
        # so that Dynamo has the fake tensor implementation
        def get_id_tuple():
            @torch.library.custom_op("test::id_tuple", mutates_args=[])
            def id_tuple(x: torch.Tensor) -> Tuple[torch.Tensor]:
                return (x.clone(),)

            @id_tuple.register_fake
            def _(
                x: torch.Tensor,
            ) -> Tuple[torch.Tensor]:
                return (x.clone(),)

            return id_tuple

        id_tuple = get_id_tuple()
        x = torch.randn(3, device=device)
        ret = id_tuple(x)
        # Check if ret is a tuple and has exactly one and the same element
        self.assertIsInstance(ret, tuple)
        self.assertEqual(len(ret), 1)
        self.assertEqual(x, ret[0])

    def test_missing_abstract_impl(self, device):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                with torch._C._AutoDispatchBelowAutograd():
                    return op(x)

            @staticmethod
            def backward(ctx, gx):
                return 2 * gx

        def foo_impl(x):
            return torch.tensor(x.cpu().numpy() ** 2, device=x.device)

        lib.impl("foo", Foo.apply, "Autograd")
        lib.impl("foo", foo_impl, "CPU")
        lib.impl("foo", foo_impl, "CUDA")
        lib.impl("foo", foo_impl, "XPU")

        x = torch.tensor([0, 1.0], requires_grad=True)
        with self.assertRaisesRegex(
            optests.OpCheckError,
            "_test_custom_op.foo.default",
        ):
            torch.library.opcheck(op, (x,), {})

    @skipIfTorchDynamo("Expected to fail due to no FakeTensor support; not a bug")
    def test_incorrect_abstract_impl(self, device):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                # Emulate AutoDispatchBelowADInplaceOrView, which is not bound into python
                guard = torch._C._AutoDispatchBelowAutograd()
                guard2 = torch._C.ExcludeDispatchKeyGuard(
                    torch._C.DispatchKeySet(torch._C.DispatchKey.ADInplaceOrView)
                )
                try:
                    return op(x)
                finally:
                    del guard
                    del guard2

            @staticmethod
            def backward(ctx, gx):
                return gx

        def foo_impl(x):
            return x**2

        def foo_meta(x):
            return x.unsqueeze(1) ** 2

        lib.impl("foo", Foo.apply, "Autograd")
        lib.impl("foo", foo_impl, "CPU")
        lib.impl("foo", foo_impl, "CUDA")
        lib.impl("foo", foo_impl, "XPU")
        lib.impl("foo", foo_meta, "Meta")

        x = torch.tensor([0, 1.0], requires_grad=True)
        with self.assertRaisesRegex(optests.OpCheckError, "Shapes .* are not equal"):
            torch.library.opcheck(op, (x,), {})

    def test_missing_functionalization(self, device):
        lib = self.lib()
        lib.define("foo(Tensor(a!) x) -> Tensor(a!)")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.mark_dirty(x)
                with torch._C._AutoDispatchBelowAutograd():
                    return op(x)

            @staticmethod
            def backward(ctx, gx):
                return gx

        def foo_impl(x):
            return x.sin_()

        def foo_meta(x):
            return x

        lib.impl("foo", Foo.apply, "Autograd")
        lib.impl("foo", foo_impl, "CPU")
        lib.impl("foo", foo_impl, "CUDA")
        lib.impl("foo", foo_impl, "XPU")
        lib.impl("foo", foo_meta, "Meta")

        x = torch.tensor([0, 1.0])
        y = x.clone()
        with self.assertRaisesRegex(
            optests.OpCheckError,
            "We only support functionalizing operators whose outputs do not have alias annotations",
        ):
            torch.library.opcheck(op, (y,), {})

    def test_autograd_registered_at_backend(self, device):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, gx):
                return gx * 0.5

        lib.impl("foo", Foo.apply, "CPU")
        lib.impl("foo", Foo.apply, "CUDA")
        lib.impl("foo", Foo.apply, "XPU")
        lib.impl("foo", lambda x: x.clone(), "Meta")

        x = torch.randn([], requires_grad=True)

        with self.assertRaisesRegex(
            torch.testing._internal.optests.OpCheckError,
            "does not have an autograd kernel",
        ):
            torch.library.opcheck(op, (x,), {})

        # I'm not sure why this is necessary
        del lib

    def test_global_state_mutation(self, device):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            invoked = 0

            @staticmethod
            def forward(ctx, x):
                Foo.invoked += 1
                return x.clone() * Foo.invoked

            @staticmethod
            def backward(ctx, gx):
                return gx

        lib.impl("foo", Foo.apply, "CompositeImplicitAutograd")

        x = torch.tensor(3.14159 / 3, requires_grad=True)
        with self.assertRaisesRegex(
            optests.OpCheckError, "eager-mode PyTorch vs AOTDispatcher"
        ):
            torch.library.opcheck(op, (x,), {})

        # Test that we can actually see the absolute difference numbers
        try:
            torch.library.opcheck(op, (x,), {})
        except optests.OpCheckError as err:
            orig = err.__context__.__context__
            self.assertIn("Absolute difference:", str(orig))

        # Test atol/rtol overrides
        torch.library.opcheck(op, (x,), {}, atol=3, rtol=0.01)

    @ops(custom_op_db.custom_op_db, dtypes=OpDTypes.any_one)
    def test_opcheck_opinfo(self, device, dtype, op):
        for sample_input in op.sample_inputs(
            device, dtype, requires_grad=op.supports_autograd
        ):
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs
            torch.library.opcheck(op.op, args, kwargs)

    def test_opcheck_fails_basic(self, device):
        @custom_op(f"{self.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor: ...

        @foo.impl(["cpu", "cuda"])
        def foo_impl(x):
            return x.sum()

        x = torch.randn(3, device=device, requires_grad=True)
        # Triggers the CustomOp autograd NYI error
        with self.assertRaisesRegex(
            optests.OpCheckError, "Autograd has not been implemented for operator"
        ):
            torch.library.opcheck(self.get_op(f"{self.test_ns}::foo"), (x,), {})

    def test_autograd_registration_check_autograd_kernel(self, device):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                with torch._C._AutoDispatchBelowAutograd():
                    return op(x)

            @staticmethod
            def backward(ctx, gx):
                return gx

        def foo_impl(x):
            return x.sin()

        lib.impl("foo", Foo.apply, "Autograd")
        lib.impl("foo", foo_impl, "CPU")
        lib.impl("foo", foo_impl, "CUDA")
        lib.impl("foo", foo_impl, "XPU")

        x = torch.randn(3, requires_grad=True, device=device)
        # Should not raise
        optests.autograd_registration_check(op, (x,), {})

    def test_autograd_registration_check_compositeimplicitautograd(self, device):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        def foo_impl(x):
            return x.sin().cos()

        lib.impl("foo", foo_impl, "CompositeImplicitAutograd")

        x = torch.randn(3, requires_grad=True, device=device)
        # Should not raise
        optests.autograd_registration_check(op, (x,), {})

    def test_autograd_registration_check_incorrect_composite(self, device):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        def foo_impl(x):
            return x.sin().cos()

        lib.impl("foo", foo_impl, "CompositeExplicitAutograd")

        x = torch.randn(3, requires_grad=True, device=device)
        with self.assertRaisesRegex(AssertionError, "incorrectly registered"):
            optests.autograd_registration_check(op, (x,), {})

    def test_autograd_registration_check_incorrect(self, device):
        lib = self.lib()
        lib.define("foo(Tensor x) -> Tensor")
        op = self.ns().foo.default

        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return torch.sin(x)

            @staticmethod
            def backward(ctx, gx):
                return gx

        lib.impl("foo", Foo.apply, "CPU")
        lib.impl("foo", Foo.apply, "CUDA")
        lib.impl("foo", Foo.apply, "XPU")

        x = torch.randn(3, requires_grad=True, device=device)
        with self.assertRaisesRegex(AssertionError, "incorrectly registered"):
            optests.autograd_registration_check(op, (x,), {})

    def test_assert_raises_regex(self, device):
        from torch.testing._internal.optests.aot_autograd import assert_raises_regex

        with assert_raises_regex(RuntimeError, "c"):
            raise RuntimeError("abcd")
        with assert_raises_regex(RuntimeError, "c.*"):
            raise RuntimeError("abcd")
        with self.assertRaisesRegex(AssertionError, "instead got"):
            with assert_raises_regex(RuntimeError, "c.*"):
                raise ValueError("abcd")
        with self.assertRaisesRegex(AssertionError, "Expected exception"):
            with assert_raises_regex(RuntimeError, "c.*"):
                pass
        with self.assertRaisesRegex(AssertionError, "to match regex"):
            with assert_raises_regex(RuntimeError, "f"):
                raise RuntimeError("abcd")


class TestCustomOp(CustomOpTestCaseBase):
    test_ns = "_test_custom_op"

    @requires_compile
    def test_functionalize_error(self):
        with torch.library._scoped_library(self.test_ns, "FRAGMENT") as lib:
            lib.define("foo(Tensor(a!) x) -> Tensor(a!)")

            def foo(x):
                return x.sin_()

            lib.impl("foo", foo, "CompositeExplicitAutograd")
            foo_op = self.get_op(f"{self.test_ns}::foo")

            lib.define("bar(Tensor(a) x) -> Tensor(a)")

            def bar(x):
                return x.view(-1)

            lib.impl("bar", bar, "CompositeExplicitAutograd")
            bar_op = self.get_op(f"{self.test_ns}::bar")

            msg = r".*We only support functionalizing operators whose outputs do not have alias annotations"

            x = torch.randn(3)

            @torch.compile(backend="aot_eager", fullgraph=True)
            def f(x):
                return foo_op(x)

            @torch.compile(backend="aot_eager", fullgraph=True)
            def g(x):
                return bar_op(x)

            with self.assertRaisesRegex(RuntimeError, msg):
                f(x)
            with self.assertRaisesRegex(RuntimeError, msg):
                g(x)

    def test_invalid_schemas(self):
        # function schema validation goes through torchgen, so this is just a
        # basic test.
        with self.assertRaisesRegex(AssertionError, "Invalid function schema: foo"):
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo", "(")

    def test_invalid_qualname(self):
        with self.assertRaisesRegex(ValueError, "overload"):
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo.Tensor", "() -> ()")

    def test_name_must_match(self):
        with self.assertRaisesRegex(ValueError, "to have name"):

            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def baz(x: Tensor) -> Tensor:
                raise NotImplementedError

    def test_unsupported_schemas(self):
        with self.assertRaisesRegex(ValueError, "only supports functional"):
            custom_ops.custom_op(
                f"{TestCustomOp.test_ns}::foo", "(Tensor(a!) x) -> Tensor(a)"
            )(foo)
        with self.assertRaisesRegex(ValueError, "only supports functional"):
            custom_ops.custom_op(
                f"{TestCustomOp.test_ns}::foo", "(Tensor(a) x) -> Tensor(a)"
            )(foo)
        with self.assertRaisesRegex(ValueError, "only supports functional"):
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo", "(Tensor x) -> ()")(
                foo
            )
        with self.assertRaisesRegex(ValueError, "self"):
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo", "(Tensor self) -> ()")(
                foo
            )

    # Tests for the older custom_op API
    def test_schema_matches_signature(self):
        with self.assertRaisesRegex(ValueError, "signature to match"):

            @custom_op(f"{TestCustomOp.test_ns}::blah", "(Tensor y) -> Tensor")
            def blah(x):
                pass

        with self.assertRaisesRegex(ValueError, "signature to match"):

            @custom_op(
                f"{TestCustomOp.test_ns}::blah2", "(Tensor x, *, Tensor y) -> Tensor"
            )
            def blah2(x, y):
                pass

        with self.assertRaisesRegex(ValueError, "signature to match"):

            @custom_op(
                f"{TestCustomOp.test_ns}::blah3",
                "(Tensor x, *, Tensor w, Tensor z) -> Tensor",
            )
            def blah3(x, *, y, z):
                pass

        with self.assertRaisesRegex(ValueError, "signature to match"):

            @custom_op(
                f"{TestCustomOp.test_ns}::blah4",
                "(Tensor x, *, Tensor z, Tensor y) -> Tensor",
            )
            def blah4(x, *, y, z):
                pass

        with self.assertRaisesRegex(ValueError, "not supported"):

            @custom_op(f"{TestCustomOp.test_ns}::blah5", "(Tensor x) -> Tensor")
            def blah5(*args):
                pass

        with self.assertRaisesRegex(ValueError, "not supported"):

            @custom_op(
                f"{TestCustomOp.test_ns}::blah6", "(*, Tensor z, Tensor y) -> Tensor"
            )
            def blah6(**kwargs):
                pass

        with self.assertRaisesRegex(ValueError, "default arguments"):

            @custom_op(
                f"{TestCustomOp.test_ns}::blah7", "(Tensor x, *, Tensor y) -> Tensor"
            )
            def blah7(x=1, *, y):
                pass

        with self.assertRaisesRegex(ValueError, "default arguments"):

            @custom_op(
                f"{TestCustomOp.test_ns}::blah8", "(Tensor x, *, Tensor y) -> Tensor"
            )
            def blah8(x, *, y=1):
                pass

        # kwonly-arg works
        @custom_op(
            f"{TestCustomOp.test_ns}::blah9", "(Tensor x, *, Tensor y) -> Tensor"
        )
        def blah9(x, *, y):
            pass

    def test_infer_schema_no_return(self):
        with self.assertRaisesRegex(
            ValueError, "No return type annotation was provided. Please add one."
        ):

            @torch.library.custom_op("mylib::foo", mutates_args={})
            def foo(x: torch.Tensor, y: int):
                return x * y

    def test_infer_schema_supported(self):
        def a(x: Tensor) -> Tensor:
            return torch.empty([])

        self.assertExpectedInline(
            infer_schema(a, mutates_args=()), """(Tensor x) -> Tensor"""
        )

        def kwonly1(x: Tensor, *, y: int, z: float) -> Tensor:
            return torch.empty([])

        self.assertExpectedInline(
            infer_schema(kwonly1, mutates_args=()),
            """(Tensor x, *, SymInt y, float z) -> Tensor""",
        )

        def kwonly2(*, y: Tensor) -> Tensor:
            return torch.empty([])

        self.assertExpectedInline(
            infer_schema(kwonly2, mutates_args=()), """(*, Tensor y) -> Tensor"""
        )

        def b(
            x: Tensor,
            y: int,
            z: bool,
            a: float,
            b: torch.dtype,
            c: torch.device,
            d: torch.types.Number,
        ) -> Tuple[Tensor, int, float, bool]:
            return torch.empty([]), 1, 0.1, True

        self.assertExpectedInline(
            infer_schema(b, mutates_args=()),
            """(Tensor x, SymInt y, bool z, float a, ScalarType b, Device c, Scalar d) -> (Tensor, SymInt, float, bool)""",
        )

        def c(
            x: Tensor,
            y: Sequence[Tensor],
            z: Optional[Tensor],
            w: Sequence[Optional[Tensor]],
        ) -> List[Tensor]:
            return [torch.empty([])]

        self.assertExpectedInline(
            infer_schema(c, mutates_args=()),
            """(Tensor x, Tensor[] y, Tensor? z, Tensor?[] w) -> Tensor[]""",
        )

        def d(x: Tensor) -> Tuple[List[Tensor], Tensor]:
            return [torch.empty([])], torch.empty([])

        self.assertExpectedInline(
            infer_schema(d, mutates_args=()), """(Tensor x) -> (Tensor[], Tensor)"""
        )

        def e() -> Tensor:
            return torch.empty([])

        self.assertExpectedInline(infer_schema(e, mutates_args=()), """() -> Tensor""")

        def f(x: Tensor) -> None:
            pass

        self.assertExpectedInline(
            infer_schema(f, mutates_args=()), """(Tensor x) -> ()"""
        )

        def g(
            x: Tensor, y: List[Tensor], z: List[Tensor], w: List[Optional[Tensor]]
        ) -> None:
            pass

        self.assertExpectedInline(
            infer_schema(g, mutates_args=()),
            """(Tensor x, Tensor[] y, Tensor[] z, Tensor?[] w) -> ()""",
        )

        self.assertExpectedInline(
            infer_schema(g, mutates_args={"x", "w", "z"}),
            """(Tensor(a0!) x, Tensor[] y, Tensor(a2!)[] z, Tensor(a3!)?[] w) -> ()""",
        )

        self.assertExpectedInline(
            infer_schema(g, mutates_args="unknown"),
            """(Tensor(a0!) x, Tensor(a1!)[] y, Tensor(a2!)[] z, Tensor(a3!)?[] w) -> ()""",
        )

        def h(
            x: Tensor,
            a: Optional[int] = None,
            b: float = 3.14,
            c: bool = True,
            d: int = 3,
            e: str = "foo",
            f: torch.dtype = torch.float,
            g: torch.dtype = torch.float32,
            h: torch.dtype = torch.int,
            i: torch.device = torch.device("cpu:0"),
            j: torch.device = "cpu",
        ) -> None:
            pass

        self.assertExpectedInline(
            infer_schema(h, mutates_args=()),
            (
                """(Tensor x, SymInt? a=None, float b=3.14, bool c=True, SymInt d=3, str e="foo", """
                """ScalarType f=float32, ScalarType g=float32, ScalarType h=int32, Device i="cpu:0", Device j="cpu") -> ()"""
            ),
        )

        def foo_impl(x: torch.Tensor) -> torch.Tensor:
            return x.sin()

        schema = torch.library.infer_schema(foo_impl, op_name="myop", mutates_args={})
        self.assertExpectedInline(schema, "myop(Tensor x) -> Tensor")

        # Ensure that a global in this file is properly found & evaluated.
        def stringy_fn(x: torch.Tensor) -> "MyList[torch.Tensor]":
            return [torch.randn_like(x)]

        schema = infer_schema(stringy_fn, mutates_args={})
        self.assertExpectedInline(schema, "(Tensor x) -> Tensor[]")

        # Make sure that substrings are evaluated properly.
        def substringy_fn(
            x: torch.Tensor,
        ) -> list["MyTensor"]:
            return [torch.randn_like(x)]

        schema = infer_schema(substringy_fn, mutates_args={})
        self.assertExpectedInline(schema, "(Tensor x) -> Tensor[]")

    def test_infer_schema_unsupported(self):
        with self.assertRaisesRegex(ValueError, "varargs"):

            def foo(*args):
                raise NotImplementedError

            infer_schema(foo, mutates_args=())

        with self.assertRaisesRegex(ValueError, "varkwargs"):

            def foo(**kwargs):
                raise NotImplementedError

            infer_schema(foo, mutates_args=())

        with self.assertRaisesRegex(ValueError, "must have a type annotation"):

            def foo(x):
                raise NotImplementedError

            infer_schema(foo, mutates_args=())

        with self.assertRaisesRegex(ValueError, "unsupported"):

            def foo(x: Tensor) -> Tuple[Tensor, ...]:
                raise NotImplementedError

            infer_schema(foo, mutates_args=())

        with self.assertRaisesRegex(ValueError, "can be mutated"):

            def foo(x: Tensor, y: int) -> Tensor:
                raise NotImplementedError

            infer_schema(foo, mutates_args={"y"})

        # Ensure that a global defined in infer_schema's file ISN'T found.
        with self.assertRaisesRegex(
            ValueError,
            r"Unsupported type annotation list\[_TestTensor\]\. It is not a type\.",
        ):

            def stringy_bad_type(
                x: torch.Tensor,
            ) -> "list[_TestTensor]":
                return [torch.randn_like(x)]

            self.assertTrue(hasattr(torch._library.infer_schema, "_TestTensor"))
            schema = infer_schema(stringy_bad_type, mutates_args={})

    def _generate_examples(self, typ):
        if typ is int:
            return [17]
        if typ is float:
            return [3.14]
        if typ is bool:
            return [True]
        if typ is str:
            return ["foo"]
        if typ is torch.dtype:
            return [torch.float32]
        if typ is torch.device:
            return [torch.device("cpu")]
        if typ == torch.types.Number:
            return [2.718]
        if typ is torch.Tensor:
            return [torch.tensor(3)]
        if typ == Optional[torch.types.Number]:
            return [None, 2.718]
        if typ == OpaqueType:
            return [make_opaque("moo")]
        origin = typing.get_origin(typ)
        if origin is Union:
            args = typing.get_args(typ)
            assert len(args) == 2 and (args[0] is type(None) or args[1] is type(None))
            elt = args[0] if args[1] is type(None) else args[1]
            return self._generate_examples(elt) + [None]
        if origin is list:
            args = typing.get_args(typ)
            assert len(args) == 1
            elt = args[0]
            return [
                self._generate_examples(elt),
                self._generate_examples(elt),
                self._generate_examples(elt),
            ]
        if origin is collections.abc.Sequence:
            args = typing.get_args(typ)
            assert len(args) == 1
            examples = self._generate_examples(args[0])
            return list(itertools.product(examples, examples)) + []
        raise NotImplementedError(
            f"testrunner cannot generate instanstance of type {typ}"
        )

    def test_supported_return_types_single_return(self):
        for typ in torch._library.infer_schema.SUPPORTED_RETURN_TYPES:
            for example in self._generate_examples(typ):
                try:

                    @custom_ops.custom_op(f"{self.test_ns}::foo")
                    def foo(x: Tensor) -> typ:
                        raise NotImplementedError

                    @custom_ops.impl(f"{self.test_ns}::foo")
                    def foo_impl(x: Tensor) -> typ:
                        return example

                    op = self.get_op(f"{self.test_ns}::foo")
                    result = op(torch.randn([]))
                    self.assertEqual(result, example, msg=f"{typ} {example}")
                finally:
                    custom_ops._destroy(f"{self.test_ns}::foo")

    def test_supported_return_types_multi_return(self):
        for typ in torch._library.infer_schema.SUPPORTED_RETURN_TYPES:
            for example in self._generate_examples(typ):
                try:

                    @custom_ops.custom_op(f"{self.test_ns}::foo")
                    def foo(x: Tensor) -> Tuple[typ, typ]:
                        raise NotImplementedError

                    @custom_ops.impl(f"{self.test_ns}::foo")
                    def foo_impl(x: Tensor) -> Tuple[typ, typ]:
                        return (example, example)

                    op = self.get_op(f"{self.test_ns}::foo")
                    result = op(torch.randn([]))
                    expected = (example, example)
                    self.assertEqual(result, expected, msg=f"{typ} {example}")
                finally:
                    custom_ops._destroy(f"{self.test_ns}::foo")

    def test_supported_param_types(self):
        for typ in torch._library.infer_schema.SUPPORTED_PARAM_TYPES:

            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: typ) -> Tensor:
                raise NotImplementedError

            yeet = None

            @custom_ops.impl(f"{TestCustomOp.test_ns}::foo", device_types=["cpu"])
            def foo_cpu(x, y):
                nonlocal yeet
                yeet = y
                return x.clone()

            try:
                for example in self._generate_examples(typ):
                    op = self.get_op(f"{self.test_ns}::foo")
                    op(torch.randn([]), example)
                    self.assertEqual(yeet, example, msg=f"{typ} {example}")
                    yeet = None
            finally:
                custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")

    def test_sequences(self):
        # Sequence[int] gets automagically turned into int[] in the schema.
        # This test checks that we actually do support arbitrary sequence types.
        class MySequence(collections.abc.Sequence):
            def __init__(self) -> None:
                self._container = [1, 2, 3]

            def __getitem__(self, idx):
                return self._container[idx]

            def __len__(self):
                return len(self._container)

        @custom_ops.custom_op(f"{self.test_ns}::foo")
        def foo(x: torch.Tensor, sizes: Sequence[int]) -> torch.Tensor:
            raise NotImplementedError

        called = 0

        @custom_ops.impl(f"{self.test_ns}::foo", device_types="cpu")
        def foo_cpu(x, sizes):
            nonlocal called
            called += 1
            # Dispatcher will normalize the sequence type into a List
            self.assertEqual(sizes, [1, 2, 3])
            return x.clone()

        x = torch.randn([])
        seq = MySequence()
        op = self.get_op(f"{self.test_ns}::foo")
        op(x, seq)
        self.assertEqual(called, 1)

    def test_unsupported_param_types(self):
        # Not comprehensive (it doesn't need to be), just a check that our mechanism works
        with self.assertRaisesRegex(ValueError, "unsupported type"):

            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: List[Optional[int]]) -> Tensor:
                raise NotImplementedError

            del foo

        with self.assertRaisesRegex(ValueError, "unsupported type"):
            # int[N] in Dispatcher is a bit wild, so we don't try to support it.
            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: Tuple[int, int]) -> Tensor:
                raise NotImplementedError

            del foo

        with self.assertRaisesRegex(ValueError, r"For example, list\[int\]"):
            # test that we propose a correct and supported type.
            @torch.library.custom_op(f"{TestCustomOp.test_ns}::foo", mutates_args={})
            def foo(x: Tensor, y: Tuple[int, int]) -> Tensor:
                raise NotImplementedError

            del foo

        with self.assertRaises(ValueError) as cm:

            @torch.library.custom_op(f"{TestCustomOp.test_ns}::foo", mutates_args={})
            def foo(x: Tensor, y: Tuple[int, float]) -> Tensor:
                raise NotImplementedError

            del foo

            self.assertNotIn("example", str(cm.exception), "")

        with self.assertRaisesRegex(ValueError, "unsupported type"):

            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: Callable) -> Tensor:
                raise NotImplementedError

            del foo

        # Define a named tuple for a Point with x and y coordinates
        Point = collections.namedtuple("Point", ["x", "y"])
        with self.assertRaisesRegex(ValueError, "unsupported type"):

            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: Tensor, y: Point) -> Tensor:
                raise NotImplementedError

            del foo

    def test_supported_schemas(self):
        # All of these should already be tested by PyTorch codegen
        # (we share the same mechanism), but here's a sanity check.
        schemas = [
            "(Tensor x) -> Tensor",
            "(Tensor x) -> Tensor y",
            "(Tensor[] x) -> Tensor y",
            "(Tensor x) -> (Tensor, Tensor)",
            "(Tensor x) -> (Tensor y, Tensor z)",
            "(Tensor x) -> (Tensor y, Tensor z)",
        ]
        other_schemas = [
            "(Tensor x, Tensor w) -> (Tensor y, Tensor z)",
            "(Tensor x, Tensor w) -> (Tensor, Tensor)",
            "(Tensor x, Tensor w) -> Tensor",
            "(Tensor? x, Tensor w) -> Tensor",
            "(Tensor? x, Tensor[] w) -> Tensor",
            "(Tensor x, int[] w) -> Tensor",
            "(Tensor x, SymInt[] w) -> Tensor",
            "(Tensor x, Scalar w) -> Tensor",
            "(Tensor x, float w) -> Tensor",
            "(Tensor x, float? w) -> Tensor",
            "(Tensor x, bool[] w) -> Tensor",
        ]

        for schema in schemas:
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo", schema)
            custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")
        for schema in other_schemas:
            custom_ops.custom_op(f"{TestCustomOp.test_ns}::bar", schema)
            custom_ops._destroy(f"{TestCustomOp.test_ns}::bar")

    def test_reserved_ns(self):
        from torch._custom_op.impl import RESERVED_NS

        for ns in RESERVED_NS:
            with self.assertRaisesRegex(ValueError, "is a reserved namespace"):
                custom_ops.custom_op(f"{ns}::foo", "(Tensor x) -> Tensor")

            with self.assertRaisesRegex(ValueError, "is a reserved namespace"):

                @custom_ops.custom_op(f"{ns}::foo2")
                def foo2(x: torch.Tensor) -> torch.Tensor:
                    raise NotImplementedError

    def test_private_ctor(self):
        with self.assertRaisesRegex(RuntimeError, "CustomOp constructor is private"):
            CustomOp(None, None, None, None, None)

    def test_lifetime(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        custom_op = torch._custom_op.impl.get_op(f"{TestCustomOp.test_ns}::foo")

        # We can't define an op multiple times,
        with self.assertRaisesRegex(RuntimeError, "multiple times"):

            @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
            def foo(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
                raise NotImplementedError

        # Unless we delete the original op.
        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")

        # Smoke test
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            raise NotImplementedError

        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")

    def test_autograd_notimplemented(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            raise NotImplementedError

        x = torch.randn(3, requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        with self.assertRaisesRegex(RuntimeError, "Autograd has not been implemented"):
            op(x)
        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")
        del foo

        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: Sequence[torch.Tensor]) -> torch.Tensor:
            raise NotImplementedError

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        with self.assertRaisesRegex(RuntimeError, "Autograd has not been implemented"):
            op([y, x])
        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")
        del foo

        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        with self.assertRaisesRegex(RuntimeError, "Autograd has not been implemented"):
            op(y, x)
        custom_ops._destroy(f"{TestCustomOp.test_ns}::foo")

    def test_autograd_notimplemented_gradmode(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x, y):
            return x * y

        x = torch.randn(3, requires_grad=True)
        y = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        with torch.no_grad():
            # Shouldn't raise, because we are in no_grad
            op(y, x)

    def test_impl_cpu(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo", device_types="cpu")
        def foo_cpu(x):
            return x.sin()

        x = torch.randn(3)
        op = self.get_op(f"{self.test_ns}::foo")
        result = op(x)
        self.assertEqual(result, foo_cpu(x))

    def test_impl_invalid_devices(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        def foo_impl(x):
            return x.sin()

        from torch._custom_op.impl import SUPPORTED_DEVICE_TYPE_TO_KEY

        for device_type in SUPPORTED_DEVICE_TYPE_TO_KEY:
            # Smoke test: should not raise error
            custom_ops.impl(f"{TestCustomOp.test_ns}::foo", device_types=device_type)(
                foo_impl
            )

        # Not supported by this API: we can either support them in the future
        # or provide some other CustomOp.def_* function. This depends on how
        # common the use cases are.
        for invalid_type in ["hip", "xla", "mkldnn", ["cpu", "hip"]]:
            with self.assertRaisesRegex(ValueError, "we only support device_type"):
                custom_ops.impl(
                    f"{TestCustomOp.test_ns}::foo", device_types=invalid_type
                )(foo_impl)

    def test_backward_partially_registered(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            return x.sin()

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return grad * saved.cos()

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        with self.assertRaisesRegex(
            RuntimeError, "unable to find a 'save_for_backward'"
        ):
            y = op(x)
            y.backward()

    def test_save_for_backward_inputs_are_namedtuple(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            return x.sin()

        hit = 0

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            nonlocal hit
            hit += 1
            self.assertTrue(isinstance(inputs, tuple))
            self.assertEqual(list(inputs._asdict().keys()), ["x"])
            return inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"x": grad * saved.cos()}

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(x)
        self.assertEqual(hit, 1)
        y.backward()
        self.assertEqual(hit, 1)

    def test_backward_returns_dict(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            return x.sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return grad * saved.cos()

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(x)
        with self.assertRaisesRegex(RuntimeError, "to be a dict"):
            y.backward()

    def test_backward_dict_invalid_keys(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            return x.sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"x": grad * saved.cos(), "y": None}

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(x)
        with self.assertRaisesRegex(RuntimeError, "to have keys {'x'}"):
            y.backward()

    def test_backward_dict_grad_for_nontensor(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, dim: int) -> torch.Tensor:
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x, dim):
            return x.sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"x": grad * saved.cos(), "dim": None}

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(x, 32)
        with self.assertRaisesRegex(RuntimeError, "non-Tensor-like types"):
            y.backward()

    def test_backward_dict_requires_keys_for_input_tensors(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x, y):
            return x.sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"x": grad * saved.cos()}

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(x, x)
        with self.assertRaisesRegex(RuntimeError, r"to have keys {.*'y'.*}"):
            y.backward()

    def test_backward_dict_requires_keys_for_input_optional_tensors(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x, y):
            return x.sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"x": grad * saved.cos()}

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(x, None)
        with self.assertRaisesRegex(RuntimeError, r"to have keys {.*'y'.*}"):
            y.backward()

    def test_backward_grads_are_tensor_or_none(self):
        @custom_ops.custom_op(f"{TestCustomOp.test_ns}::foo")
        def foo(x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError

        @custom_ops.impl(f"{TestCustomOp.test_ns}::foo")
        def foo_impl(x):
            return x.sin()

        @custom_ops.impl_save_for_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_save_for_backward(inputs, output):
            return inputs.x

        @custom_ops.impl_backward(f"{TestCustomOp.test_ns}::foo")
        def foo_backward(ctx, saved, grad):
            return {"x": (grad * saved.cos(),)}

        x = torch.randn([], requires_grad=True)
        op = self.get_op(f"{self.test_ns}::foo")
        y = op(x)
        with self.assertRaisesRegex(RuntimeError, "either None or a Tensor"):
            y.backward()

    def test_backward_tensorlist_input_requires_list_grads_
```



## High-Level Overview


This Python file contains 26 class(es) and 527 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `CustomOpTestCaseBase`, `TestCustomOpTesting`, `Foo`, `Foo`, `Foo`, `Foo`, `Foo`, `Foo`, `Foo`, `Foo`, `Foo`, `TestCustomOp`, `MySequence`, `MiniOpTest`, `Op`, `TestCustomOpAPI`, `MyTwoTensor`, `Stack`, `Foo`, `Bar`

**Functions defined**: `requires_compile`, `setUp`, `tearDown`, `ns`, `lib`, `get_op`, `test_aot_autograd_check_degenerate_cases`, `simple`, `outputs_dont_require_grad`, `no_outputs`, `test_incorrect_schema_mutation`, `forward`, `backward`, `foo_impl`, `test_incorrect_schema_view`, `forward`, `backward`, `foo_impl`, `foo_meta`, `test_opcheck_unbacked_stride`

**Key imports**: collections, io, itertools, os, re, subprocess, sys, tempfile, typing, unittest


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `collections`
- `io`
- `itertools`
- `os`
- `re`
- `subprocess`
- `sys`
- `tempfile`
- `typing`
- `unittest`
- `functools`: partial
- `pathlib`: Path
- `numpy as np`
- `yaml`
- `torch._custom_ops as custom_ops`
- `torch.testing._internal.optests as optests`
- `torch.utils._pytree as pytree`
- `torch.utils.cpp_extension`
- `torch`: Tensor
- `torch._custom_op.impl`: CustomOp, infer_schema
- `torch._library.infer_schema`: tuple_to_list
- `torch._library.opaque_object`: make_opaque, OpaqueType
- `torch._utils_internal`: get_file_path_2  
- `torch.fx.experimental.proxy_tensor`: make_fx
- `torch.fx.experimental.symbolic_shapes`: ShapeEnv
- `torch.testing._internal`: custom_op_db
- `torch.testing._internal.common_cuda`: TEST_CUDA
- `torch.testing._internal.custom_op_db`: numpy_nonzero
- `torch.testing._internal.two_tensor`: TwoTensor


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components
- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/test_custom_ops.py
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

- **File Documentation**: `test_custom_ops.py_docs.md`
- **Keyword Index**: `test_custom_ops.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
