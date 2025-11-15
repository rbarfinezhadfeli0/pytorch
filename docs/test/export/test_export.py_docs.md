# Documentation: `test/export/test_export.py`

## File Metadata

- **Path**: `test/export/test_export.py`
- **Size**: 674,732 bytes (658.92 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# Owner(s): ["oncall: export"]
# ruff: noqa: F841
# flake8: noqa
import contextlib
import copy
import dataclasses
import enum
import functools
import logging
import math
import operator
import os
import re
import sys
import traceback
import unittest
import warnings
import weakref
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from re import escape
from typing import Dict, List, Union
from unittest.mock import MagicMock, patch

import torch
import torch._dynamo as torchdynamo
import torch.fx.traceback as fx_traceback
import torch.nn.functional as F
import torch.utils._pytree as pytree
from functorch.experimental.control_flow import cond, map
from torch import Tensor
from torch._decomp import decomposition_table, get_decompositions
from torch._dynamo._trace_wrapped_higher_order_op import mod_index
from torch._dynamo.test_case import TestCase
from torch._dynamo.testing import normalize_gm
from torch._export import config
from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse
from torch._export.utils import (
    get_buffer,
    get_param,
    is_buffer,
    is_param,
    register_dataclass_as_pytree_node,
)
from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
from torch._higher_order_ops.associative_scan import associative_scan
from torch._higher_order_ops.hints_wrap import hints_wrapper
from torch._higher_order_ops.scan import scan
from torch._higher_order_ops.while_loop import while_loop
from torch._inductor.compile_fx import split_const_gm
from torch._subclasses import FakeTensorMode
from torch.export import default_decompositions, Dim, export, unflatten
from torch.export._trace import (
    _export,
    _export_to_torch_ir,
    DEFAULT_EXPORT_DYNAMO_CONFIG,
)
from torch.export.graph_signature import (
    ExportGraphSignature,
    InputKind,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
from torch.export.passes import move_to_device_pass
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    xfailIfDistributedNotSupported,
)
from torch.testing._internal.common_utils import (
    find_library_location,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    run_tests,
    skipIfCrossRef,
    skipIfXpu,
    TEST_TRANSFORMERS,
    TEST_WITH_CROSSREF,
    TestCase as TorchTestCase,
)
from torch.testing._internal.custom_tensor import (
    ConstantExtraMetadataTensor,
    CustomTensorPlainOut,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.testing._internal.torchbind_impls import load_torchbind_test_lib
from torch.testing._internal.triton_utils import requires_cuda_and_triton, requires_gpu
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils._pytree import (
    register_constant,
    tree_flatten,
    tree_map,
    tree_unflatten,
    TreeSpec,
    treespec_dumps,
    treespec_leaf,
    treespec_loads,
)


if HAS_GPU:
    import triton
    import triton.language as tl

    from torch._library import capture_triton

try:
    from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

    HAS_TORCHREC = True
except ImportError:
    HAS_TORCHREC = False

try:
    from . import testing
except ImportError:
    import testing  # @manual=fbcode//caffe2/test:test_export-library
# The following import pattern matters as `test_export.export` is patched
# in other files (like test_export_nonstrict.py). `torch.export.export`
# will invalidate the patch.
from torch.export import export


torch.library.define("testlib::returns_tensor_symint", "(Tensor x) -> (Tensor, SymInt)")
torch.library.define(
    "testlib::foo",
    "(Tensor(a!) x, Tensor(b!) z) -> (Tensor, Tensor, Tensor)",
    tags=torch.Tag.pt2_compliant_tag,
)
torch.library.define(
    "testlib::foo_mutated",
    "(Tensor(a!) x) -> (Tensor, Tensor)",
    tags=torch.Tag.pt2_compliant_tag,
)
torch.library.define(
    "testlib::foo_functional",
    "(Tensor x) -> (Tensor)",
    tags=torch.Tag.pt2_compliant_tag,
)
torch.library.define(
    "testlib::foo_unbacked",
    "(Scalar x) -> (Tensor)",
    tags=torch.Tag.pt2_compliant_tag,
)


@torch.library.impl("testlib::returns_tensor_symint", "cpu")
@torch.library.register_fake("testlib::returns_tensor_symint")
def returns_tensor_symint_impl(x):
    return x, x.shape[0]


@torch.library.impl("testlib::foo", "cpu")
@torch._dynamo.disable
def foo_impl(x, z):
    x.add_(5)
    z.add_(5)
    return x, z, x + z


@torch.library.register_fake("testlib::foo")
def foo_abstract(x, z):
    return x, z, x + z


@torch.library.impl("testlib::foo_mutated", "CompositeImplicitAutograd")
def foo_mutated(x):
    a, b, c = torch.ops.testlib.foo(x, x.cos())
    return a, a.cos()


@torch.library.impl("testlib::foo_functional", "CompositeImplicitAutograd")
def foo_functional(x):
    a, b, c = torch.ops.testlib.foo(x.cos(), x.cos())
    return a.cos()


@torch.library.impl("testlib::foo_unbacked", "CompositeImplicitAutograd")
def foo_unbacked(x):
    if x > 2:
        return torch.ones(4, 4)
    if x < 6:
        return torch.ones(4, 4)
    return torch.ones(4, 4)


@dataclass
class Inp1:
    x: Tensor
    y: List[Tensor]
    z: Dict[str, Tensor]


@dataclass
class Inp2:
    a: Tensor
    b: Tensor


@dataclass
class Inp3:
    f: torch.Tensor
    p: torch.Tensor


NON_STRICT_SUFFIX = "_nonstrict"
STRICT_SUFFIX = "_strict"
INLINE_AND_INSTALL_STRICT_SUFFIX = "_inline_and_install_strict"
RETRACEABILITY_STRICT_SUFFIX = "_retraceability_strict"
RETRACEABILITY_NON_STRICT_SUFFIX = "_retraceability_nonstrict"
SERDES_SUFFIX = "serdes"
SERDES_STRICT_SUFFIX = "_serdes_strict"
SERDES_NON_STRICT_SUFFIX = "_serdes_nonstrict"
PREDISPATCH_SUFFIX = "_pre_dispatch"
TRAINING_IR_DECOMP_STRICT_SUFFIX = "_training_ir_to_decomp_strict"
TRAINING_IR_DECOMP_NON_STRICT_SUFFIX = "_training_ir_to_decomp_nonstrict"
CPP_RUNTIME_STRICT_SUFFIX = "_cpp_runtime_strict"
CPP_RUNTIME_NONSTRICT_SUFFIX = "_cpp_runtime_nonstrict"
STRICT_EXPORT_V2_SUFFIX = "_strict_export_v2"


# Now default mode is non strict, so original unammended test names
# should be treated as non-strict
def is_non_strict_test(test_name):
    return not test_name.endswith(STRICT_SUFFIX) and not test_name.endswith(
        STRICT_EXPORT_V2_SUFFIX
    )


def is_strict_test(test_name):
    return test_name.endswith(STRICT_SUFFIX)


def is_strict_v2_test(test_name):
    return test_name.endswith(STRICT_EXPORT_V2_SUFFIX)


def is_inline_and_install_strict_test(test_name: str) -> bool:
    return test_name.endswith(INLINE_AND_INSTALL_STRICT_SUFFIX)


def is_retracebility_test(test_name):
    return test_name.endswith(RETRACEABILITY_STRICT_SUFFIX) or test_name.endswith(
        RETRACEABILITY_NON_STRICT_SUFFIX
    )


def is_serdes_test(test_name):
    return test_name.endswith(SERDES_STRICT_SUFFIX) or test_name.endswith(
        SERDES_NON_STRICT_SUFFIX
    )


def need_serdes_test(test_name):
    return SERDES_SUFFIX in test_name


def is_training_ir_test(test_name):
    return test_name.endswith(TRAINING_IR_DECOMP_STRICT_SUFFIX) or test_name.endswith(
        TRAINING_IR_DECOMP_NON_STRICT_SUFFIX
    )


def is_training_ir_strict_test(test_name):
    return test_name.endswith(TRAINING_IR_DECOMP_STRICT_SUFFIX)


def is_cpp_runtime_test(test_name):
    return test_name.endswith(CPP_RUNTIME_STRICT_SUFFIX) or test_name.endswith(
        CPP_RUNTIME_NONSTRICT_SUFFIX
    )


def get_hop_schema(ep: torch.export.ExportedProgram):
    hop_node = next(
        node
        for node in ep.graph.nodes
        if isinstance(node.target, torch._ops.HigherOrderOperator)
    )
    return torch._library.utils.hop_schema_from_fx_node(hop_node)


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestDynamismExpression(TestCase):
    def test_export_inline_constraints(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                b = x.item()
                return torch.full((b, 1), 1)

        f = Module()
        inp = (torch.tensor([3]),)
        ref = f(*inp)

        gm = export(f, inp)
        res = gm.module()(*inp)

        self.assertTrue(torchdynamo.utils.same(ref, res))

        gm = make_fx(f, tracing_mode="symbolic")(*inp)
        res = gm(*inp)
        self.assertTrue(torchdynamo.utils.same(ref, res))

    def test_export_constraints_error_not_in_range(self):
        class InvalidInputConflictWithInputConstraints(torch.nn.Module):
            def forward(self, x):
                return x + 1

        inp = torch.zeros([3])
        dim_x = torch.export.Dim("dim_x", min=6)

        if is_non_strict_test(self._testMethodName):
            error_type = torch.fx.experimental.symbolic_shapes.ConstraintViolationError
        else:
            error_type = torch._dynamo.exc.UserError

        with self.assertRaisesRegex(error_type, "not in range"):
            export(
                InvalidInputConflictWithInputConstraints(),
                (inp,),
                dynamic_shapes={"x": {0: dim_x}},
            )

    def test_export_slice_maxsize(self):
        class Slice(torch.nn.Module):
            def forward(self, *args):
                return torch.ops.aten.slice.Tensor(*args)

        inp = (torch.rand((10, 3, 224, 224)), 0, 0, 9223372036854775807)
        dynamic_shapes = (({0: Dim("dim")}, None, None, None),)
        torch.export.export(
            Slice(),
            inp,
            dynamic_shapes=dynamic_shapes,
        )

    def test_no_grad_param_inplace(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parameter = torch.nn.Parameter(torch.ones(4, 4))

            def forward(self, x):
                with torch.no_grad():
                    self.parameter.div_(2)
                return x + self.parameter

        foo_ep = Foo()
        foo_eager = Foo()
        ep = export(foo_ep, (torch.rand(4, 4),)).run_decompositions()
        val = ep.graph_signature.parameters_to_mutate
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %p_parameter : [num_users=1] = placeholder[target=p_parameter]
    %x : [num_users=1] = placeholder[target=x]
    %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%p_parameter, 2), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %div), kwargs = {})
    return (div, add)""",
        )

        self.assertTrue("div" in val.keys())
        self.assertTrue("parameter" in val.values())

        test_inp = torch.rand(4, 4)

        res = foo_eager(test_inp)

        # TODO We almost need to make the param mutation happen outside
        # of the graph. Or wrap the param mutation in a no_grad HOP. Simply
        # overriding gm.__call__ doesn't seem to work due to:
        #   1. graph module does something weird to __call__ so it is not easy to override
        #   2. We inspect module.forward to bind fake args when retracing
        with self.assertRaisesRegex(RuntimeError, "leaf"):
            res_export = ep.module()(torch.rand(4, 4))

        with torch.no_grad():
            res_export = ep.module()(test_inp)

        self.assertTrue(torch.allclose(res, res_export))

    def test_export_slice_unbacked_dim1(self):
        class MySlice(torch.nn.Module):
            def forward(self, x, seq_len):
                l = seq_len.item()
                x = x.narrow(1, 0, l)
                return x

        x = torch.randn(10, 7)
        seq_len = torch.tensor(5)
        torch.export.export(MySlice(), args=(x, seq_len))

    @torch.fx.experimental._config.patch(backed_size_oblivious=True)
    def test_reshape_view_backed_size_oblivious(self):
        N = 3

        class MyModel(torch.nn.Module):
            def forward(self, x):
                y = x[:-1, :]  # [s0 - 1, 32]
                stacked = torch.stack([y] * N, dim=0)  # [N * (s0 - 1), 32]
                reshaped = stacked.reshape(-1, N, 32)  # [(s0 - 1), N, 32]
                return reshaped

        inps = (torch.randn(10, 32),)
        spec = {
            "x": (Dim.AUTO, Dim.STATIC),
        }
        ep = export(MyModel(), inps, dynamic_shapes=spec)

    def test_export_constraints_error(self):
        class ConflictingConstraints(torch.nn.Module):
            def forward(self, x):
                b = x.item()
                torch._check(b >= 4)
                torch._check(b <= 5)
                torch._check(b <= 5)
                torch._check(True)
                return torch.full((b, 1), 1)

        inp = (torch.tensor([3]),)
        ep = export(ConflictingConstraints(), inp)

        with self.assertRaisesRegex(
            RuntimeError, r"Runtime assertion failed for expression u[\d+] \>\= 4"
        ):
            ep.module()(torch.tensor([3]))

    def test_export_assume_static_by_default(self):
        class Module(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                if x.shape[0] == 4:
                    return x + 1
                else:
                    return x

        branch_on_shape = Module()
        inp = (torch.rand(4, 5),)

        # Being able to export means shape is preserved as static
        export(branch_on_shape, inp)

    def test_export_strict_narrow_unbacked_expr(self):
        # Tests that we are able to handle 0/1 specialization on sizes represented
        # by unbacked int expressions by transforming them into an unbacked int.
        #
        # This test only works with strict=True, since it relies on dynamo tracing
        # for transforming the expression into an unbacked SymInt.

        def identity(x):
            return x

        class Module(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, x, p):
                u0 = p.item()
                torch._check(u0 + 5 <= x.shape[0])
                torch._check(u0 >= 0)
                # Create a tensor of size: (x.shape[0] - u0 - 5).
                return x.narrow(0, u0 + 5, self.fn(x.shape[0] - u0 - 5))

        inputs = (torch.arange(10), torch.tensor(2))

        # See https://github.com/pytorch/pytorch/issues/154574
        # # Without transforming the unbacked int expression, we can't export.
        # with self.assertRaisesRegex(
        #     RuntimeError, escape("Could not guard on data-dependent expression")
        # ):
        #     export(Module(identity), inputs, strict=True)

        # It works if we transform the whole unbacked int expression into
        # an unbacked int.
        export(Module(torch.sym_fresh_size), inputs, strict=True)


class InputModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x, y):
        return self.linear(x) * y


class InputModuleWithNestedSubclass(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = torch.nn.Parameter(torch.ones(2, 2))
        self.p2 = torch.nn.Parameter(
            CustomTensorPlainOut(
                CustomTensorPlainOut(
                    torch.Tensor([[0, 0], [0, 1]]),
                    torch.Tensor([[0, 0], [1, 0]]),
                ),
                CustomTensorPlainOut(
                    torch.Tensor([[1, 0], [0, 0]]),
                    torch.Tensor([[0, 1], [0, 0]]),
                ),
            )
        )

    def forward(self, x):
        a = (x + 2 * self.p1 + self.p2).sum().sum()
        return x + a


@unittest.skipIf(IS_WINDOWS, "Windows isn't supported for this case")
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestExport(TestCase):
    def _test_export_same_as_eager(self, f, args, kwargs=None):
        kwargs = kwargs or {}
        exported_program = export(f, args, kwargs)
        self.assertEqual(exported_program.module()(*args, **kwargs), f(*args, **kwargs))
        # this is not supported by .module()
        # reversed_kwargs = {key: kwargs[key] for key in reversed(kwargs)}
        # self.assertEqual(
        #     exported_program.module()(*args, **reversed_kwargs), f(*args, **reversed_kwargs)
        # )

    def _check_dynamic_shapes_specs_and_shapes(
        self,
        model,
        inputs,
        specs,
        passing_shapes,
        failing_shapes,
        test_serdes=False,
    ):
        from torch._export.serde.dynamic_shapes import (
            _dump_dynamic_shapes,
            _load_dynamic_shapes,
        )
        from torch.utils._pytree import tree_map

        def _construct_inputs(shapes):
            def _is_tensor_leaf(x):
                return isinstance(x, tuple) and all(isinstance(y, int) for y in x)

            return tree_map(
                lambda x: torch.randn(*x) if _is_tensor_leaf(x) else x,
                shapes,
                is_leaf=_is_tensor_leaf,
            )

        # exports with a list of equivalent dynamic shapes specs,
        # then tests for pass/fail on list of shapes
        for _specs in specs:
            ep = export(model, inputs, dynamic_shapes=_specs)
            eps = [ep]
            if test_serdes:
                # test dynamic shapes serialization
                # test that behavior remains the same when exporting with Ser/Des specs:
                # serialize + deserialize original specs, and export.
                ep_serdes = export(
                    model,
                    inputs,
                    dynamic_shapes=_load_dynamic_shapes(
                        _dump_dynamic_shapes(_specs, inputs)
                    ),
                )
                eps.append(ep_serdes)

            for ep in eps:
                for shapes in passing_shapes:
                    test_inputs = _construct_inputs(shapes)
                    ep.module()(*test_inputs)
                for shapes in failing_shapes:
                    test_inputs = _construct_inputs(shapes)
                    with self.assertRaisesRegex(AssertionError, "Guard failed"):
                        ep.module()(*test_inputs)

    def test_basic(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                return x[0] + y

        f = Module()
        inp = ([torch.ones(1, 3)], torch.ones(1, 3))
        self._test_export_same_as_eager(f, inp)

    @testing.expectedFailureStrictV2
    @skipIfCrossRef
    def test_custom_tag_metadata_re_export(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.rand(4, 2))
                self.b = torch.nn.Parameter(torch.rand(4))

            def forward(self, x):
                out = torch.nn.functional.linear(x, self.w, self.b)
                return out

        f = Foo()
        inputs = (torch.zeros(1, 2),)
        ep = export(f, inputs)

        new_gm = copy.deepcopy(ep.graph_module)
        new_gm.meta["custom"] = {}
        new_gm.meta["custom"]["f"] = "bar"

        for node in new_gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.linear.default
            ):
                node.meta["custom"] = {}
                node.meta["custom"]["quantization_tag"] = "foo"

        new_ep = ep._update(new_gm, ep.graph_signature)
        new_ep = export(new_ep.module(), inputs)
        self.assertEqual(new_ep.graph_module.meta["custom"]["f"], "bar")

        # the custom field should be preserved after re-export and
        # should not be copied to other nodes
        counter = 0
        for node in new_ep.graph.nodes:
            if "custom" in node.meta:
                counter += 1
                self.assertTrue(node.meta["custom"]["quantization_tag"] == "foo")
                self.assertTrue(node.target == torch.ops.aten.linear.default)

        self.assertEqual(counter, 1)

    @testing.expectedFailureSerDer  # can't serialize functorch ops
    @testing.expectedFailureSerDerNonStrict  # can't serialize functorch ops
    def test_vmap_to_assert(self):
        class VmapToAssert(torch.nn.Module):
            def forward(self, x, y):
                f = lambda x, y: (
                    (x * y).to("cpu", memory_format=torch.channels_last) + 1
                ).sum(dim=0)  # noqa: E731
                vmapped = torch.vmap(f)(x, y)
                return vmapped.sum(dim=0)

        ep = export(VmapToAssert(), (torch.zeros(4, 4, 4, 4), torch.zeros(4, 4, 4, 4)))
        exported = ep.module()(torch.ones(4, 4, 4, 4), torch.ones(4, 4, 4, 4))
        eager = VmapToAssert()(torch.ones(4, 4, 4, 4), torch.ones(4, 4, 4, 4))
        self.assertEqual(exported, eager)

    def test_from_node_metadata_export(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1d = torch.nn.Conv1d(3, 3, 3)
                self.conv2d = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                x = self.conv2d(x)
                x = x.squeeze(0)
                x = self.conv1d(x)
                return x

            def example_inputs(self):
                return

        f = Foo()
        inputs = (torch.randn(1, 3, 5, 5),)
        ep = export(f, inputs)
        graph_id = id(ep.graph)
        gm = ep.module()
        from torch.fx.traceback import NodeSourceAction

        for node in gm.graph.nodes:
            if node.op in ("placeholder", "output", "call_module"):
                continue
            if "weight" in node.name or "bias" in node.name:
                self.assertTrue(
                    node.meta["from_node"][-1].pass_name
                    == "ExportedProgram.module().unlift()"
                )
                self.assertTrue(
                    node.meta["from_node"][-1].action
                    == [NodeSourceAction.CREATE, NodeSourceAction.REPLACE]
                )
                self.assertEqual(
                    node.meta["from_node"][-1].from_node[-1].graph_id, graph_id
                )
            else:
                self.assertTrue(
                    node.meta["from_node"][-1].pass_name == "ExportedProgram.module()"
                )
                self.assertTrue(
                    node.meta["from_node"][-1].action == [NodeSourceAction.CREATE]
                )
                self.assertEqual(node.meta["from_node"][-1].graph_id, graph_id)

        ## re-export
        ep2 = export(gm, inputs)
        gm2 = ep2.module()
        graph_id = id(ep2.graph)

        for node in gm2.graph.nodes:
            if node.op in ("placeholder", "output", "call_module"):
                continue

            if "weight" in node.name or "bias" in node.name:
                self.assertTrue(
                    node.meta["from_node"][-1].pass_name
                    == "ExportedProgram.module().unlift()"
                )
                self.assertTrue(
                    node.meta["from_node"][-1].action
                    == [NodeSourceAction.CREATE, NodeSourceAction.REPLACE]
                )
                self.assertEqual(
                    node.meta["from_node"][-1].from_node[-1].graph_id, graph_id
                )
            else:
                self.assertTrue(
                    node.meta["from_node"][-1].pass_name == "ExportedProgram.module()"
                )
                self.assertTrue(
                    node.meta["from_node"][-1].action == [NodeSourceAction.CREATE]
                )
                self.assertEqual(node.meta["from_node"][-1].graph_id, graph_id)

    def test_annotate_on_assert(self):
        # nodes added in `apply_runtime_assertion_pass` will be annotated
        class M(torch.nn.Module):
            def forward(self, x, y):
                with torch.fx.traceback.annotate({"moo": 0}):
                    x = torch.cat([x, x])
                    b = y.item()
                    torch._check(b >= x.shape[0])
                    return x * b

        with torch.fx.traceback.preserve_node_meta():
            ep = torch.export.export(
                M(),
                (torch.randn(3), torch.tensor(6)),
                dynamic_shapes={"x": {0: Dim("b")}, "y": None},
            )

        # clean up _torchdynamo related meta data as it could vary depending on the caller
        # https://github.com/pytorch/pytorch/issues/167432
        for node in ep.graph.nodes:
            if "custom" in node.meta:
                node.meta["custom"] = {
                    k: v
                    for k, v in node.meta["custom"].items()
                    if "_torchdynamo_disable" not in k
                }

        custom_metadata = torch.fx.traceback._get_custom_metadata(ep.module())

        self.assertExpectedInline(
            str(custom_metadata),
            """\
('call_function', 'cat', {'moo': 0})
('call_function', 'item', {'moo': 0})
('call_function', 'ge_1', {'moo': 0})
('call_function', '_assert_scalar_default', {'moo': 0})
('call_function', 'mul', {'moo': 0})""",
        )

    @requires_gpu
    def test_flex_attention_export(self):
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        class MixedFakeModeModel(torch.nn.Module):
            def __init__(self, dim=64, use_inductor=True):
                super().__init__()
                self.dim = dim
                self.q_proj = torch.nn.Linear(64, 64)
                self.k_proj = torch.nn.Linear(64, 64)
                self.v_proj = torch.nn.Linear(64, 64)
                self.use_inductor = use_inductor

            def forward(self, x):
                batch_size, seq_len, _ = x.shape

                # Process input first - this creates fake tensors in export's fake mode
                processed = self.q_proj(x)

                # Create some computation that depends on processed tensor
                intermediate = processed.sum(dim=-1).detach()  # Shape: (batch, seq_len)

                # Now call create_block_mask which internally calls torch.compile
                # The mask function will capture 'intermediate' which is a fake tensor
                # from export's fake mode, but create_block_mask will create its own fake mode
                def dynamic_mask_function(batch_idx, head_idx, q_idx, kv_idx):
                    # This captures the intermediate tensor from the outer scope
                    # When torch.compile is called inside create_block_mask,
                    # this tensor will be from export's fake mode while new tensors
                    # created inside will be from the nested fake mode
                    threshold = intermediate[
                        batch_idx, q_idx % seq_len
                    ]  # Access the captured tensor
                    return (kv_idx <= q_idx) & (threshold > 0)  # Mix fake modes

                block_mask = create_block_mask(
                    mask_mod=dynamic_mask_function,
                    B=batch_size,
                    H=None,
                    Q_LEN=seq_len,
                    KV_LEN=seq_len,
                    device=x.device,
                )
                q = self.q_proj(processed).view(batch_size, 1, seq_len, self.dim)
                k = self.k_proj(processed).view(batch_size, 1, seq_len, self.dim)
                v = self.v_proj(processed).view(batch_size, 1, seq_len, self.dim)

                # Use flex_attention with the problematic block_mask
                backend = "inductor" if self.use_inductor else "eager"
                out = torch.compile(flex_attention, backend=backend)(
                    q, k, v, block_mask=block_mask
                )

                return out

        model = MixedFakeModeModel(use_inductor=False)
        x = torch.randn(2, 128, 64)
        # Inductor doesn't work in eager mode flex attention
        eager_out = model(x)
        model.use_inductor = True
        exported_mod = torch.export.export(model, (x,), strict=False).module()
        self.assertExpectedInline(
            str(exported_mod.code).strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    q_proj_weight = self.q_proj.weight
    q_proj_bias = self.q_proj.bias
    k_proj_weight = self.k_proj.weight
    k_proj_bias = self.k_proj.bias
    v_proj_weight = self.v_proj.weight
    v_proj_bias = self.v_proj.bias
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    linear = torch.ops.aten.linear.default(x, q_proj_weight, q_proj_bias);  x = None
    sum_1 = torch.ops.aten.sum.dim_IntList(linear, [-1])
    detach = torch.ops.aten.detach.default(sum_1);  sum_1 = None
    arange = torch.ops.aten.arange.start(0, 2, device = device(type='cpu'), pin_memory = False)
    arange_1 = torch.ops.aten.arange.start(0, 1, device = device(type='cpu'), pin_memory = False)
    arange_2 = torch.ops.aten.arange.start(0, 128, device = device(type='cpu'), pin_memory = False)
    arange_3 = torch.ops.aten.arange.start(0, 128, device = device(type='cpu'), pin_memory = False)
    lazy_load_decompositions = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions = None
    _vmap_increment_nesting = torch._functorch.predispatch._vmap_increment_nesting(2, 'error');  _vmap_increment_nesting = None
    _add_batch_dim = torch._functorch.predispatch._add_batch_dim(arange, 0, 1);  arange = None
    lazy_load_decompositions_1 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_1 = None
    _vmap_increment_nesting_1 = torch._functorch.predispatch._vmap_increment_nesting(1, 'error');  _vmap_increment_nesting_1 = None
    _add_batch_dim_1 = torch._functorch.predispatch._add_batch_dim(arange_1, 0, 2);  arange_1 = _add_batch_dim_1 = None
    lazy_load_decompositions_2 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_2 = None
    _vmap_increment_nesting_2 = torch._functorch.predispatch._vmap_increment_nesting(128, 'error');  _vmap_increment_nesting_2 = None
    _add_batch_dim_2 = torch._functorch.predispatch._add_batch_dim(arange_2, 0, 3);  arange_2 = None
    lazy_load_decompositions_3 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_3 = None
    _vmap_increment_nesting_3 = torch._functorch.predispatch._vmap_increment_nesting(128, 'error');  _vmap_increment_nesting_3 = None
    _add_batch_dim_3 = torch._functorch.predispatch._add_batch_dim(arange_3, 0, 4);  arange_3 = None
    remainder = torch.ops.aten.remainder.Scalar(_add_batch_dim_2, 128)
    torch__dynamo__trace_wrapped_higher_order_op_mod_index0 = self.torch__dynamo__trace_wrapped_higher_order_op_ModIndex0
    function_const_func_spec0 = self.function_const_func_spec0
    flat_apply = torch.ops.higher_order.flat_apply(function_const_func_spec0, torch__dynamo__trace_wrapped_higher_order_op_mod_index0, 'torch._dynamo._trace_wrapped_higher_order_op.ModIndex', detach, _add_batch_dim, remainder);  function_const_func_spec0 = torch__dynamo__trace_wrapped_higher_order_op_mod_index0 = _add_batch_dim = remainder = None
    le = torch.ops.aten.le.Tensor(_add_batch_dim_3, _add_batch_dim_2);  _add_batch_dim_3 = _add_batch_dim_2 = None
    gt = torch.ops.aten.gt.Scalar(flat_apply, 0);  flat_apply = None
    and_1 = torch.ops.aten.__and__.Tensor(le, gt);  le = gt = None
    _remove_batch_dim = torch._functorch.predispatch._remove_batch_dim(and_1, 4, 128, 0);  and_1 = None
    _vmap_decrement_nesting = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting = None
    _remove_batch_dim_1 = torch._functorch.predispatch._remove_batch_dim(_remove_batch_dim, 3, 128, 0);  _remove_batch_dim = None
    _vmap_decrement_nesting_1 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_1 = None
    _remove_batch_dim_2 = torch._functorch.predispatch._remove_batch_dim(_remove_batch_dim_1, 2, 1, 0)
    expand = torch.ops.aten.expand.default(_remove_batch_dim_1, [1, 128, 128]);  _remove_batch_dim_1 = expand = None
    _vmap_decrement_nesting_2 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_2 = None
    _remove_batch_dim_3 = torch._functorch.predispatch._remove_batch_dim(_remove_batch_dim_2, 1, 2, 0);  _remove_batch_dim_2 = None
    _vmap_decrement_nesting_3 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_3 = None
    pad = torch.ops.aten.pad.default(_remove_batch_dim_3, [0, 0, 0, 0]);  _remove_batch_dim_3 = None
    view = torch.ops.aten.view.default(pad, [2, 1, 1, 128, 1, 128]);  pad = None
    permute = torch.ops.aten.permute.default(view, [0, 1, 2, 4, 3, 5]);  view = None
    sum_2 = torch.ops.aten.sum.dim_IntList(permute, [-2, -1]);  permute = None
    eq = torch.ops.aten.eq.Scalar(sum_2, 16384)
    gt_1 = torch.ops.aten.gt.Scalar(sum_2, 0)
    lt = torch.ops.aten.lt.Scalar(sum_2, 16384);  sum_2 = None
    and_2 = torch.ops.aten.__and__.Tensor(gt_1, lt);  gt_1 = lt = None
    _assert_tensor_metadata_default = torch.ops.aten._assert_tensor_metadata.default(and_2, dtype = torch.bool, device = device(type='cpu'), layout = torch.strided);  _assert_tensor_metadata_default = None
    to = torch.ops.aten.to.dtype(and_2, torch.int8);  and_2 = None
    _assert_tensor_metadata_default_1 = torch.ops.aten._assert_tensor_metadata.default(eq, dtype = torch.bool, device = device(type='cpu'), layout = torch.strided);  _assert_tensor_metadata_default_1 = None
    to_1 = torch.ops.aten.to.dtype(eq, torch.int8);  eq = None
    _assert_tensor_metadata_default_2 = torch.ops.aten._assert_tensor_metadata.default(to, dtype = torch.int8, device = device(type='cpu'), layout = torch.strided);  _assert_tensor_metadata_default_2 = None
    to_2 = torch.ops.aten.to.dtype(to, torch.int32);  to = None
    sum_3 = torch.ops.aten.sum.dim_IntList(to_2, [-1])
    argsort = torch.ops.aten.argsort.stable(to_2, stable = True, descending = True);  to_2 = None
    _assert_tensor_metadata_default_3 = torch.ops.aten._assert_tensor_metadata.default(sum_3, dtype = torch.int64, device = device(type='cpu'), layout = torch.strided);  _assert_tensor_metadata_default_3 = None
    to_3 = torch.ops.aten.to.dtype(sum_3, torch.int32, False, False, torch.contiguous_format);  sum_3 = None
    _assert_tensor_metadata_default_4 = torch.ops.aten._assert_tensor_metadata.default(argsort, dtype = torch.int64, device = device(type='cpu'), layout = torch.strided);  _assert_tensor_metadata_default_4 = None
    to_4 = torch.ops.aten.to.dtype(argsort, torch.int32, False, False, torch.contiguous_format);  argsort = None
    _assert_tensor_metadata_default_5 = torch.ops.aten._assert_tensor_metadata.default(to_1, dtype = torch.int8, device = device(type='cpu'), layout = torch.strided);  _assert_tensor_metadata_default_5 = None
    to_5 = torch.ops.aten.to.dtype(to_1, torch.int32);  to_1 = None
    sum_4 = torch.ops.aten.sum.dim_IntList(to_5, [-1])
    argsort_1 = torch.ops.aten.argsort.stable(to_5, stable = True, descending = True);  to_5 = None
    _assert_tensor_metadata_default_6 = torch.ops.aten._assert_tensor_metadata.default(sum_4, dtype = torch.int64, device = device(type='cpu'), layout = torch.strided);  _assert_tensor_metadata_default_6 = None
    to_6 = torch.ops.aten.to.dtype(sum_4, torch.int32, False, False, torch.contiguous_format);  sum_4 = None
    _assert_tensor_metadata_default_7 = torch.ops.aten._assert_tensor_metadata.default(argsort_1, dtype = torch.int64, device = device(type='cpu'), layout = torch.strided);  _assert_tensor_metadata_default_7 = None
    to_7 = torch.ops.aten.to.dtype(argsort_1, torch.int32, False, False, torch.contiguous_format);  argsort_1 = None
    lazy_load_decompositions_4 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_4 = None
    _vmap_increment_nesting_4 = torch._functorch.predispatch._vmap_increment_nesting(2, 'error');  _vmap_increment_nesting_4 = None
    _add_batch_dim_4 = torch._functorch.predispatch._add_batch_dim(to_3, 0, 1)
    _add_batch_dim_5 = torch._functorch.predispatch._add_batch_dim(to_4, 0, 1)
    lazy_load_decompositions_5 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_5 = None
    _vmap_increment_nesting_5 = torch._functorch.predispatch._vmap_increment_nesting(1, 'error');  _vmap_increment_nesting_5 = None
    _add_batch_dim_6 = torch._functorch.predispatch._add_batch_dim(_add_batch_dim_4, 0, 2);  _add_batch_dim_4 = None
    _add_batch_dim_7 = torch._functorch.predispatch._add_batch_dim(_add_batch_dim_5, 0, 2);  _add_batch_dim_5 = None
    new_zeros = torch.ops.aten.new_zeros.default(_add_batch_dim_7, [1, 2], dtype = torch.int32, pin_memory = False)
    arange_4 = torch.ops.aten.arange.default(1, dtype = torch.int32, device = device(type='cpu'), pin_memory = False)
    unsqueeze = torch.ops.aten.unsqueeze.default(arange_4, -1);  arange_4 = None
    arange_5 = torch.ops.aten.arange.default(1, dtype = torch.int32, device = device(type='cpu'), pin_memory = False)
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(_add_batch_dim_6, -1);  _add_batch_dim_6 = None
    lt_1 = torch.ops.aten.lt.Tensor(arange_5, unsqueeze_1);  arange_5 = unsqueeze_1 = None
    where = torch.ops.aten.where.ScalarOther(lt_1, _add_batch_dim_7, 1);  lt_1 = _add_batch_dim_7 = None
    new_ones = torch.ops.aten.new_ones.default(new_zeros, [], pin_memory = False)
    index_put_ = torch.ops.aten.index_put_.default(new_zeros, [unsqueeze, where], new_ones);  new_zeros = unsqueeze = where = new_ones = None
    slice_1 = torch.ops.aten.slice.Tensor(index_put_, 1, 0, 1);  index_put_ = None
    _remove_batch_dim_4 = torch._functorch.predispatch._remove_batch_dim(slice_1, 2, 1, 0);  slice_1 = None
    _vmap_decrement_nesting_4 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_4 = None
    _remove_batch_dim_5 = torch._functorch.predispatch._remove_batch_dim(_remove_batch_dim_4, 1, 2, 0);  _remove_batch_dim_4 = None
    _vmap_decrement_nesting_5 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_5 = None
    transpose = torch.ops.aten.transpose.int(_remove_batch_dim_5, -2, -1);  _remove_batch_dim_5 = None
    _assert_tensor_metadata_default_8 = torch.ops.aten._assert_tensor_metadata.default(transpose, dtype = torch.int32, device = device(type='cpu'), layout = torch.strided);  _assert_tensor_metadata_default_8 = None
    to_8 = torch.ops.aten.to.dtype(transpose, torch.int32);  transpose = None
    sum_5 = torch.ops.aten.sum.dim_IntList(to_8, [-1])
    argsort_2 = torch.ops.aten.argsort.stable(to_8, stable = True, descending = True);  to_8 = None
    _assert_tensor_metadata_default_9 = torch.ops.aten._assert_tensor_metadata.default(sum_5, dtype = torch.int64, device = device(type='cpu'), layout = torch.strided);  _assert_tensor_metadata_default_9 = None
    to_9 = torch.ops.aten.to.dtype(sum_5, torch.int32, False, False, torch.contiguous_format);  sum_5 = None
    _assert_tensor_metadata_default_10 = torch.ops.aten._assert_tensor_metadata.default(argsort_2, dtype = torch.int64, device = device(type='cpu'), layout = torch.strided);  _assert_tensor_metadata_default_10 = None
    to_10 = torch.ops.aten.to.dtype(argsort_2, torch.int32, False, False, torch.contiguous_format);  argsort_2 = None
    lazy_load_decompositions_6 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_6 = None
    _vmap_increment_nesting_6 = torch._functorch.predispatch._vmap_increment_nesting(2, 'error');  _vmap_increment_nesting_6 = None
    _add_batch_dim_8 = torch._functorch.predispatch._add_batch_dim(to_6, 0, 1)
    _add_batch_dim_9 = torch._functorch.predispatch._add_batch_dim(to_7, 0, 1)
    lazy_load_decompositions_7 = torch._functorch.predispatch.lazy_load_decompositions();  lazy_load_decompositions_7 = None
    _vmap_increment_nesting_7 = torch._functorch.predispatch._vmap_increment_nesting(1, 'error');  _vmap_increment_nesting_7 = None
    _add_batch_dim_10 = torch._functorch.predispatch._add_batch_dim(_add_batch_dim_8, 0, 2);  _add_batch_dim_8 = None
    _add_batch_dim_11 = torch._functorch.predispatch._add_batch_dim(_add_batch_dim_9, 0, 2);  _add_batch_dim_9 = None
    new_zeros_1 = torch.ops.aten.new_zeros.default(_add_batch_dim_11, [1, 2], dtype = torch.int32, pin_memory = False)
    arange_6 = torch.ops.aten.arange.default(1, dtype = torch.int32, device = device(type='cpu'), pin_memory = False)
    unsqueeze_2 = torch.ops.aten.unsqueeze.default(arange_6, -1);  arange_6 = None
    arange_7 = torch.ops.aten.arange.default(1, dtype = torch.int32, device = device(type='cpu'), pin_memory = False)
    unsqueeze_3 = torch.ops.aten.unsqueeze.default(_add_batch_dim_10, -1);  _add_batch_dim_10 = None
    lt_2 = torch.ops.aten.lt.Tensor(arange_7, unsqueeze_3);  arange_7 = unsqueeze_3 = None
    where_1 = torch.ops.aten.where.ScalarOther(lt_2, _add_batch_dim_11, 1);  lt_2 = _add_batch_dim_11 = None
    new_ones_1 = torch.ops.aten.new_ones.default(new_zeros_1, [], pin_memory = False)
    index_put__1 = torch.ops.aten.index_put_.default(new_zeros_1, [unsqueeze_2, where_1], new_ones_1);  new_zeros_1 = unsqueeze_2 = where_1 = new_ones_1 = None
    slice_2 = torch.ops.aten.slice.Tensor(index_put__1, 1, 0, 1);  index_put__1 = None
    _remove_batch_dim_6 = torch._functorch.predispatch._remove_batch_dim(slice_2, 2, 1, 0);  slice_2 = None
    _vmap_decrement_nesting_6 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_6 = None
    _remove_batch_dim_7 = torch._functorch.predispatch._remove_batch_dim(_remove_batch_dim_6, 1, 2, 0);  _remove_batch_dim_6 = None
    _vmap_decrement_nesting_7 = torch._functorch.predispatch._vmap_decrement_nesting();  _vmap_decrement_nesting_7 = None
    transpose_1 = torch.ops.aten.transpose.int(_remove_batch_dim_7, -2, -1);  _remove_batch_dim_7 = None
    _assert_tensor_metadata_default_11 = torch.ops.aten._assert_tensor_metadata.default(transpose_1, dtype = torch.int32, device = device(type='cpu'), layout = torch.strided);  _assert_tensor_metadata_default_11 = None
    to_11 = torch.ops.aten.to.dtype(transpose_1, torch.int32);  transpose_1 = None
    sum_6 = torch.ops.aten.sum.dim_IntList(to_11, [-1])
    argsort_3 = torch.ops.aten.argsort.stable(to_11, stable = True, descending = True);  to_11 = None
    _assert_tensor_metadata_default_12 = torch.ops.aten._assert_tensor_metadata.default(sum_6, dtype = torch.int64, device = device(type='cpu'), layout = torch.strided);  _assert_tensor_metadata_default_12 = None
    to_12 = torch.ops.aten.to.dtype(sum_6, torch.int32, False, False, torch.contiguous_format);  sum_6 = None
    _assert_tensor_metadata_default_13 = torch.ops.aten._assert_tensor_metadata.default(argsort_3, dtype = torch.int64, device = device(type='cpu'), layout = torch.strided);  _assert_tensor_metadata_default_13 = None
    to_13 = torch.ops.aten.to.dtype(argsort_3, torch.int32, False, False, torch.contiguous_format);  argsort_3 = None
    linear_1 = torch.ops.aten.linear.default(linear, q_proj_weight, q_proj_bias);  q_proj_weight = q_proj_bias = None
    view_1 = torch.ops.aten.view.default(linear_1, [2, 1, 128, 64]);  linear_1 = None
    linear_2 = torch.ops.aten.linear.default(linear, k_proj_weight, k_proj_bias);  k_proj_weight = k_proj_bias = None
    view_2 = torch.ops.aten.view.default(linear_2, [2, 1, 128, 64]);  linear_2 = None
    linear_3 = torch.ops.aten.linear.default(linear, v_proj_weight, v_proj_bias);  linear = v_proj_weight = v_proj_bias = None
    view_3 = torch.ops.aten.view.default(linear_3, [2, 1, 128, 64]);  linear_3 = None
    sdpa_score0 = self.sdpa_score0
    sdpa_mask0 = self.sdpa_mask0
    flex_attention = torch.ops.higher_order.flex_attention(view_1, view_2, view_3, sdpa_score0, (128, 128, to_3, to_4, to_6, to_7, to_9, to_10, to_12, to_13, 128, 128, sdpa_mask0), 0.125, {'PRESCALE_QK': False, 'ROWS_GUARANTEED_SAFE': False, 'BLOCKS_ARE_CONTIGUOUS': False, 'WRITE_DQ': True, 'OUTPUT_LOGSUMEXP': False, 'OUTPUT_MAX': False}, (), (detach,));  view_1 = view_2 = view_3 = sdpa_score0 = to_3 = to_4 = to_6 = to_7 = to_9 = to_10 = to_12 = to_13 = sdpa_mask0 = detach = None
    getitem = flex_attention[0]
    getitem_1 = flex_attention[1];  getitem_1 = None
    getitem_2 = flex_attention[2];  flex_attention = getitem_2 = None
    return pytree.tree_unflatten((getitem,), self._out_spec)""",
        )
        exported_out = exported_mod(x)
        self.assertEqual(exported_out, eager_out)

    def test_inductor_backend_inside_nonstrict(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                def i_want_faster_code(inp1, inp2):
                    nonlocal x
                    return x + inp1 + inp2

                out = torch.compile(i_want_faster_code)(x, x)
                return x + out

        foo = Foo()
        with self.assertWarnsRegex(
            UserWarning, "You are calling torch.compile inside torch.export region"
        ):
            ep = export(foo, (torch.randn(4, 4),), strict=False).module()
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %x : [num_users=4] = placeholder[target=x]
    %_guards_fn : [num_users=0] = call_module[target=_guards_fn](args = (%x,), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %x), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %x), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %add_1), kwargs = {})
    return (add_2,)""",
        )

    def test_bincount(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                weights = torch.linspace(0, 1, steps=5)
                bc = x.bincount(weights)
                return bc

        model = M()
        ep = export(model, (torch.randint(0, 8, (5,), dtype=torch.int64),))

        inp = torch.randint(0, 8, (5,), dtype=torch.int64)
        self.assertTrue(torch.allclose(ep.module()(inp), M()(inp)))

    def test_symint_output(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                z, y = x.size()
                return z + y + x[0], z

        inputs = (torch.ones(2, 3),)
        dim0_x, dim1_x = torch.export.dims("dim0_x", "dim1_x")
        dynamic_shapes = {"x": (dim0_x, dim1_x)}
        export(Foo(), inputs, dynamic_shapes=dynamic_shapes)

    @testing.expectedFailureStrictV2
    def test_no_tensor_computation(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                return y

        f = Module()
        inp = ([torch.ones(1, 3)], 1)
        ep = export(f, inp)
        self.assertEqual(ep.module()(*inp), f(*inp))
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %x_0 : [num_users=0] = placeholder[target=x_0]
    %y : [num_users=0] = placeholder[target=y]
    return (1,)""",
        )

    def test_inline_script_function(self):
        @torch.jit.script
        def _forward(x: torch.Tensor):
            if torch.jit.is_scripting():
                return x.cos()
            return x.sin()

        class M(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                return _forward(x)

        x = torch.randn(3, 4)
        ep = torch.export.export(M(), (x,))
        FileCheck().check_count("torch.ops.aten.sin", 1, exactly=True).run(
            str(ep.graph)
        )
        FileCheck().check_count("torch.ops.aten.cos", 0, exactly=True).run(
            str(ep.graph)
        )
        res = ep.module()(x)
        # We're inlining the original _forward function
        # instead of the scripted function, so we get x.sin()
        self.assertEqual(res, x.sin())

    def test_nested_module_fake_tensor_leak(self):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._tensor_cache = None

            def forward(self, x):
                if self._tensor_cache is None:
                    self._tensor_cache = x + 2
                return self._tensor_cache.sum() + x.sum()

        class Foo(torch.nn.Module):
            def __init__(self, bar):
                super().__init__()
                self.bar = bar

            def forward(self, x):
                return self.bar(x)

        foo = Foo(Bar())
        _ = export(foo, (torch.ones(4, 4),), strict=False)
        self.assertTrue(foo.bar._tensor_cache is None)

    def test_export_leak_compile(self):
        class BaseModule(torch.nn.Module):
            def forward(self, *args, **kwargs):
                raise NotImplementedError

        class CacheModule(BaseModule):
            def __init__(self, cache: torch.Tensor):
                super().__init__()
                assert cache.ndim == 3
                self.cache = torch.nn.Parameter(cache, requires_grad=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                n_tokens = x.size(1)
                rolled_cache = torch.roll(self.cache.data, -n_tokens, dims=1)
                rolled_cache[:, -n_tokens:, :] = x
                self.cache.data = rolled_cache
                return self.cache

        class LinearBlock(torch.nn.Module):
            def __init__(self, in_features, out_features, activation=None):
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features)
                self.activation = activation

            def forward(self, x):
                x = self.linear(x)
                return self.activation(x) if self.activation else x

        class MyModel(BaseModule):
            def __init__(self):
                super().__init__()
                default_cache = torch.zeros(1, 10, 5)
                self.cache_layer = CacheModule(default_cache)
                self.fc1 = LinearBlock(5, 10, activation=torch.nn.ReLU())
                self.fc2 = LinearBlock(10, 5)

            def forward(self, x):
                cached = self.cache_layer(x)
                out = self.fc1(cached)
                out = self.fc2(out)
                return out

        with self.assertRaisesRegex(
            RuntimeError,
            "We found a fake tensor in the exported program constant's list. "
            "This typically means our tracing system encountered an op that we can't trace through. "
            "For the potential source, you can refer to following model attribute: cache_layer.lifted_tensor_0. "
            "Please file an issue on github.",
        ):
            _ = export(MyModel(), (torch.randn(1, 3, 5),), strict=False)

        with self.assertWarnsRegex(
            UserWarning,
            "We found a fake tensor in the exported program constant's list. "
            "This typically means our tracing system encountered an op that we can't trace through. "
            "For the potential source, you can refer to following model attribute: cache_layer.lifted_tensor_0. "
            "Please file an issue on github.",
        ):
            # can't trigger all variant of export because later on it will crash
            # and it is good because we warned :).
            with torch._export.config.patch(error_on_lifted_constant_tensors=False):
                _ = torch.export.export(
                    MyModel(), (torch.randn(1, 3, 5),), strict=False
                )

    def test_inline_script_class_method(self):
        class M(torch.nn.Module):
            @staticmethod
            @torch.jit.script
            def _forward(x: torch.Tensor):
                if torch.jit.is_scripting():
                    return x.cos()
                return x.sin()

            def forward(self, x: torch.Tensor):
                return M._forward(x)

        x = torch.randn(3, 4)
        ep = torch.export.export(M(), (x,))
        FileCheck().check_count("torch.ops.aten.sin", 1, exactly=True).run(
            str(ep.graph)
        )
        FileCheck().check_count("torch.ops.aten.cos", 0, exactly=True).run(
            str(ep.graph)
        )
        res = ep.module()(x)
        # We're inlining the original _forward function
        # instead of the scripted function, so we get x.sin()
        self.assertEqual(res, x.sin())

    def test_tag_ac_export(self):
        ops_to_save = [torch.ops.aten.addmm.default]

        def policy_fn(ctx, op, *args, **wargs):
            if op in ops_to_save:
                return torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE
            else:
                return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE

        context_fn = functools.partial(
            torch.utils.checkpoint.create_selective_checkpoint_contexts, policy_fn
        )

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(128, 128)
                self.relu = torch.nn.ReLU()
                self.linear2 = torch.nn.Linear(128, 128)

            def forward(self, x):
                return self.linear2(self.relu(self.linear1(x)))

        # Wrap the block with checkpointing
        class CheckpointedBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block = Block()

            def forward(self, x):
                return torch.utils.checkpoint.checkpoint(
                    self.block, x, context_fn=context_fn
                )

        model = CheckpointedBlock()
        x = torch.randn(16, 128, requires_grad=True)

        ep = torch.export.export(model, (x,), strict=True)
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %p_block_linear1_weight : [num_users=1] = placeholder[target=p_block_linear1_weight]
    %p_block_linear1_bias : [num_users=1] = placeholder[target=p_block_linear1_bias]
    %p_block_linear2_weight : [num_users=1] = placeholder[target=p_block_linear2_weight]
    %p_block_linear2_bias : [num_users=1] = placeholder[target=p_block_linear2_bias]
    %x : [num_users=1] = placeholder[target=x]
    %wrap_body0 : [num_users=1] = get_attr[target=wrap_body0]
    %tag_activation_checkpoint : [num_users=7] = call_function[target=torch.ops.higher_order.tag_activation_checkpoint](args = (%wrap_body0, %x, %p_block_linear1_weight, %p_block_linear1_bias, %p_block_linear2_weight, %p_block_linear2_bias), kwargs = {})
    %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%tag_activation_checkpoint, 0), kwargs = {})
    %getitem_1 : [num_users=0] = call_function[target=operator.getitem](args = (%tag_activation_checkpoint, 1), kwargs = {})
    %getitem_2 : [num_users=0] = call_function[target=operator.getitem](args = (%tag_activation_checkpoint, 2), kwargs = {})
    %getitem_3 : [num_users=0] = call_function[target=operator.getitem](args = (%tag_activation_checkpoint, 3), kwargs = {})
    %getitem_4 : [num_users=0] = call_function[target=operator.getitem](args = (%tag_activation_checkpoint, 4), kwargs = {})
    %getitem_5 : [num_users=0] = call_function[target=operator.getitem](args = (%tag_activation_checkpoint, 5), kwargs = {})
    %getitem_6 : [num_users=0] = call_function[target=operator.getitem](args = (%tag_activation_checkpoint, 6), kwargs = {})
    return (getitem,)""",
        )

        self.assertExpectedInline(
            str(ep.graph_module.wrap_body0.graph).strip(),
            """\
graph():
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=2] = placeholder[target=arg1_1]
    %arg2_1 : [num_users=2] = placeholder[target=arg2_1]
    %arg3_1 : [num_users=2] = placeholder[target=arg3_1]
    %arg4_1 : [num_users=2] = placeholder[target=arg4_1]
    %linear : [num_users=2] = call_function[target=torch.ops.aten.linear.default](args = (%arg0_1, %arg1_1, %arg2_1), kwargs = {})
    %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%linear,), kwargs = {})
    %linear_1 : [num_users=1] = call_function[target=torch.ops.aten.linear.default](args = (%relu, %arg3_1, %arg4_1), kwargs = {})
    return (linear_1, arg1_1, arg2_1, linear, relu, arg3_1, arg4_1)""",
        )

        stack = contextlib.ExitStack()

        with stack:
            jwd = aot_export_joint_with_descriptors(stack, ep.module(), (x,))
            for node in jwd.graph_module.graph.nodes:
                if "recompute" in node.meta:
                    actual = node.meta["recompute"]
                    expected = policy_fn(None, node.target, None, None)
                    self.assertEqual(actual, expected)
            self.assertExpectedInline(
                str(jwd.graph_module.code).strip(),
                """\
def forward(self, primals, tangents):
    primals_1, primals_2, primals_3, primals_4, primals_5, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    t = torch.ops.aten.t.default(primals_1);  primals_1 = None
    addmm = torch.ops.aten.addmm.default(primals_2, primals_5, t);  primals_2 = None
    relu = torch.ops.aten.relu.default(addmm);  addmm = None
    detach_3 = torch.ops.aten.detach.default(relu)
    t_1 = torch.ops.aten.t.default(primals_3);  primals_3 = None
    addmm_1 = torch.ops.aten.addmm.default(primals_4, relu, t_1);  primals_4 = None
    t_2 = torch.ops.aten.t.default(t_1);  t_1 = None
    mm = torch.ops.aten.mm.default(tangents_1, t_2);  t_2 = None
    t_3 = torch.ops.aten.t.default(tangents_1)
    mm_1 = torch.ops.aten.mm.default(t_3, relu);  t_3 = relu = None
    t_4 = torch.ops.aten.t.default(mm_1);  mm_1 = None
    sum_1 = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view = torch.ops.aten.view.default(sum_1, [128]);  sum_1 = None
    t_5 = torch.ops.aten.t.default(t_4);  t_4 = None
    detach_6 = torch.ops.aten.detach.default(detach_3);  detach_3 = None
    threshold_backward = torch.ops.aten.threshold_backward.default(mm, detach_6, 0);  mm = detach_6 = None
    t_6 = torch.ops.aten.t.default(t);  t = None
    mm_2 = torch.ops.aten.mm.default(threshold_backward, t_6);  t_6 = None
    t_7 = torch.ops.aten.t.default(threshold_backward)
    mm_3 = torch.ops.aten.mm.default(t_7, primals_5);  t_7 = primals_5 = None
    t_8 = torch.ops.aten.t.default(mm_3);  mm_3 = None
    sum_2 = torch.ops.aten.sum.dim_IntList(threshold_backward, [0], True);  threshold_backward = None
    view_1 = torch.ops.aten.view.default(sum_2, [128]);  sum_2 = None
    t_9 = torch.ops.aten.t.default(t_8);  t_8 = None
    return pytree.tree_unflatten([addmm_1, t_9, view_1, t_5, view, mm_2], self._out_spec)""",
            )

    def test_inline_script_class_method_recursive(self):
        f = 0.4
        i = 2
        s = "foo"

        @torch.jit.script
        def _inner(x: torch.Tensor, y: torch.Tensor, f: float, i: int, s_len: int):
            return x * y * f * i * s_len

        class M(torch.nn.Module):
            @staticmethod
            @torch.jit.script
            def _forward(x: torch.Tensor, y: torch.Tensor, f: float, i: int, s: str):
                if torch.jit.is_scripting():
                    return _inner(x.cos(), y.cos(), f, i, len(s))
                return _inner(x.sin(), y.sin(), f, i, len(s))

            def forward(self, x: torch.Tensor):
                return M._forward(x, y=x, f=f, i=i, s=s)

        x = torch.randn(3, 4)
        ep = torch.export.export(M(), (x,))
        FileCheck().check_count("torch.ops.aten.sin", 2, exactly=True).run(
            str(ep.graph)
        )
        FileCheck().check_count("torch.ops.aten.cos", 0, exactly=True).run(
            str(ep.graph)
        )
        res = ep.module()(x)
        # We're inlining the original _forward function
        # instead of the scripted function, so we get x.sin()
        self.assertEqual(res, _inner(x.sin(), x.sin(), f, i, len(s)))

    def test_inline_script_method(self):
        class M(torch.jit.ScriptModule):
            @torch.jit.script_method
            def _forward(self, x: torch.Tensor):
                if torch.jit.is_scripting():
                    return x.cos()
                return x.sin()

            def forward(self, x):
                return self._forward(x)

        class Wrapped(torch.nn.Module):
            def __init__(self, mod):
                super().__init__()
                self.mod = mod

            def forward(self, x):
                return self.mod(x)

        x = torch.randn(3, 4)
        ep = torch.export.export(Wrapped(M()), (x,))
        FileCheck().check_count("torch.ops.aten.sin", 1, exactly=True).run(
            str(ep.graph)
        )
        FileCheck().check_count("torch.ops.aten.cos", 0, exactly=True).run(
            str(ep.graph)
        )
        res = ep.module()(x)
        # We're inlining the original _forward function
        # instead of the scripted function, so we get x.sin()
        self.assertEqual(res, x.sin())

    @testing.expectedFailureStrictV2
    def test_no_tensor_computation_2(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                return x

        f = Module()
        inp = (torch.randn(3), 1)
        ep = export(f, inp)
        self.assertEqual(ep.module()(*inp), f(*inp))
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %x : [num_users=1] = placeholder[target=x]
    %y : [num_users=0] = placeholder[target=y]
    return (x,)""",
        )

    @testing.expectedFailureStrictV2
    def test_no_tensor_computation_3(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                return 5

        f = Module()
        inp = (2, 1)
        ep = export(f, inp)
        self.assertEqual(ep.module()(*inp), f(*inp))
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %x : [num_users=0] = placeholder[target=x]
    %y : [num_users=0] = placeholder[target=y]
    return (5,)""",
        )

    @testing.expectedFailureStrictV2
    def test_no_tensor_computation_4(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                return x

        f = Module()
        inp = ([torch.randn(3)], 1)
        ep = export(f, inp)
        self.assertEqual(ep.module()(*inp), f(*inp))
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %x_0 : [num_users=1] = placeholder[target=x_0]
    %y : [num_users=0] = placeholder[target=y]
    return (x_0,)""",
        )

    def test_not_registered_parameter(self):
        class Basic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.params = {"foo": torch.nn.Parameter(torch.ones(3, 3))}

            def forward(self, x):
                return x + self.params["foo"]

        f = Basic()
        args = (torch.randn(1, 3),)
        # strict-mode will error out because foo is registered as parameter
        # in dynamo (a behavior that's different from eager). We decided to
        # follow eager behavior.
        ep = export(f, args, strict=False)
        gm = ep.module()
        self.assertEqual(len(ep.graph_signature.lifted_tensor_constants), 1)
        self.assertEqual(len(ep.graph_signature.parameters), 0)
        # check foo is not a parameter in the final graph
        self.assertEqual(len(list(gm.named_parameters())), 0)
        self.assertEqual(gm(*args), f(*args))
        self.assertExpectedInline(
            str(gm.graph).strip(),
            """\
graph():
    %lifted_tensor_0 : [num_users=1] = get_attr[target=lifted_tensor_0]
    %x : [num_users=2] = placeholder[target=x]
    %_guards_fn : [num_users=0] = call_module[target=_guards_fn](args = (%x,), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %lifted_tensor_0), kwargs = {})
    return (add,)""",
        )

    def test_int_shape_specialization(self):
        class M(torch.nn.Module):
            def forward(self, x):
                ori_size = (
                    int(x.shape[-2] / 1),
                    int(x.shape[-1] / 1),
                )
                x = F.interpolate(x, size=ori_size, mode="bilinear")
                return x

        input1 = (torch.rand(1, 3, 28, 28),)
        input2 = (torch.rand(1, 3, 56, 56),)
        inputs = [input1, input2]
        model = M()
        dynamic_shapes = {
            "x": {2: torch.export.Dim.DYNAMIC, 3: torch.export.Dim.DYNAMIC},
        }
        with self.assertRaisesRegex(
            (
                torch.fx.experimental.symbolic_shapes.ConstraintViolationError,
                torch._dynamo.exc.UserError,
            ),
            (
                r"your code specialized it to be a constant \(28\)(.*\n)*.*"
                r"your code specialized it to be a constant \(28\)(.*\n)*.*"
            ),
        ):
            export(model, input1, dynamic_shapes=dynamic_shapes, strict=False)

    def test_external_call_non_strict_real_tensor(self):
        class ExternalMethod:
            def add(self, x):
                return x + x

        class Basic(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.external_add = ExternalMethod().add

            def forward(self, x):
                return self.external_add(x)

        f = Basic()
        args = (torch.randn(1, 3),)
        ep = export(f, args, strict=False)
        self.assertEqual(ep.module()(*args), f(*args))

    def test_export_statically_known_true(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                shape = y.shape[0] ** 2 - 3 * y.shape[0]
                end = shape
                return x[:, :end]

        dynamic_shapes = (
            (torch.export.Dim.DYNAMIC, torch.export.Dim.DYNAMIC),
            (torch.export.Dim.DYNAMIC, torch.export.Dim.DYNAMIC),
        )

        m = Foo()
        inp = (torch.randn(4, 4), torch.randn(4, 4))
        ep = export(
            m,
            inp,
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )

        self.assertTrue(torch.allclose(ep.module()(*inp), m(*inp)))

        FileCheck().check_count("torch.ops.aten.slice.Tensor", 1, exactly=True).run(
            str(ep.graph)
        )
        FileCheck().check_count("operator.sub", 1, exactly=True).run(str(ep.graph))

    def test_colon_parameter(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter("foo:bar", torch.nn.Parameter(torch.ones(3, 3)))

            def forward(self, x):
                return x + getattr(self, "foo:bar")

        ep = export(M(), (torch.randn(3, 3),))
        x = torch.randn(3, 3)
        self.assertEqual(ep.module()(x), M()(x))

    def test_conv_dynamic(self):
        # Simple module for demonstration
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    in_channels=3, out_channels=32, kernel_size=3, padding=1
                )
                self.relu = torch.nn.ReLU()
                self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                a = self.conv(x)
                a.add_(y)
                return self.maxpool(self.relu(a))

        example_args = (torch.randn(2, 3, 256, 256), torch.ones(2, 32, 256, 256))
        dynamic_shapes = {"x": {0: Dim("batch")}, "y": {0: Dim("batch")}}
        m = M()
        exported_program: torch.export.ExportedProgram = export(
            m, args=example_args, dynamic_shapes=dynamic_shapes
        )

        args = (torch.randn(17, 3, 256, 256), torch.ones(17, 32, 256, 256))
        self.assertEqual(exported_program.module()(*args), m(*args))
        args = (torch.randn(15, 3, 256, 256), torch.ones(15, 32, 256, 256))
        self.assertEqual(exported_program.module()(*args), m(*args))

        gm: torch.fx.GraphModule = torch.export.export(
            m, args=example_args, dynamic_shapes=dynamic_shapes
        ).module()

        args = (torch.randn(17, 3, 256, 256), torch.ones(17, 32, 256, 256))
        self.assertEqual(gm(*args), m(*args))
        args = (torch.randn(15, 3, 256, 256), torch.ones(15, 32, 256, 256))
        self.assertEqual(gm(*args), m(*args))

    # stride() is called for an undefined tensor
    @testing.expectedFailureCppRuntimeNonStrict
    def test_native_multi_attention_head(self):
        embed_dim = 64
        num_heads = 4
        bs = 16
        sl = 8
        device = "cpu"

        q = 6 * torch.rand(bs, sl, embed_dim, device=device, dtype=torch.float32) - 3
        k = q
        v = q

        qkv = torch.nn.Linear(
            embed_dim, 3 * embed_dim, device=device, dtype=torch.float32
        )
        proj = torch.nn.Linear(embed_dim, embed_dim, device=device, dtype=torch.float32)

        class NativeMHA(torch.nn.Module):
            def __init__(
                self,
                embed_dim,
                num_heads,
                qkv,
                proj,
                need_weights,
                average_attn_weights,
                mask_type,
            ):
                super().__init__()
                self.qkv = qkv
                self.proj = proj
                self.embed_dim = embed_dim
                self.num_heads = num_heads
                self.need_weights = need_weights
                self.average_attn_weights = average_attn_weights
                self.mask_type = mask_type

            def forward(self, q, k, v, key_padding_mask):
                return torch._native_multi_head_attention(
                    q,
                    k,
                    v,
                    self.embed_dim,
                    self.num_heads,
                    self.qkv.weight,
                    self.qkv.bias,
                    self.proj.weight,
                    self.proj.bias,
                    key_padding_mask,
                    need_weights=False,
                    average_attn_weights=False,
                    mask_type=1,  # mask_type = 1 => src_key_padding_mask, mask_type = 0 => src_mask
                )

        for mask_type in (0, 1):
            for need_weights in (True, False):
                for average_attn_weights in (True, False):
                    npt = NativeMHA(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        qkv=qkv,
                        proj=proj,
                        need_weights=need_weights,
                        average_attn_weights=average_attn_weights,
                        mask_type=mask_type,
                    )
                    sample_input = (q, k, v, None)

                    ep = export(
                        npt,
                        args=sample_input,
                        dynamic_shapes={
                            "q": {
                                0: Dim("dim0_q", max=1024),
                            },
                            "k": {
                                0: Dim("dim0_k", max=1024),
                            },
                            "v": {
                                0: Dim("dim0_v", max=1024),
                            },
                            "key_padding_mask": None,
                        },
                    )
                    self.assertEqual(ep.module()(*sample_input), npt(*sample_input))

    def test_unused_constant(self):
        class M(torch.nn.Module):
            def forward(self, x):
                y = torch.tensor(3)
                return x * x

        ep = export(M(), (torch.ones(3),))
        self.assertEqual(len(ep.constants), 0)

        class M(torch.nn.Module):
            def __init__(self, num_features: int = 1) -> None:
                super().__init__()
                self.num_features = num_features

            def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
                res = [torch.Tensor([])] * self.num_features
                for i in range(self.num_features):
                    res[i] = x * (i + 1)
                return res

        inp = torch.ones(3)
        ep = export(M(), (inp,))
        self.assertEqual(len(ep.constants), 0)

        unf = unflatten(ep)
        self.assertTrue(torch.allclose(M()(inp)[0], unf(inp)[0]))

    def test_unbacked_bincount(self):
        class Foo(torch.nn.Module):
            def forward(self, xs):
                u0, u1 = xs.tolist()
                x = torch.ones(u0, dtype=torch.int64)
                y = torch.bincount(x, minlength=u1)
                return y

        m = Foo()
        x = torch.tensor([20, 10])
        ep = export(m, (x,))
        self.assertTrue(torch.allclose(ep.module()(x), m(x)))
        y = torch.tensor([5, 10])
        self.assertTrue(torch.allclose(ep.module()(y), m(y)))

    @requires_gpu
    def test_export_custom_triton_kernel(self):
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        @torch.library.triton_op("mylib::add", mutates_args=())
        def custom_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            capture_triton(add_kernel)[grid](x, y, output, n_elements, 16)
            return output

        class M(torch.nn.Module):
            def forward(self, x, y):
                return custom_add(x, y)

        args = (
            torch.randn(3, device=GPU_TYPE),
            torch.randn(3, device=GPU_TYPE),
        )
        max_len = 128
        dynamic_shapes = {
            "x": {0: Dim("dim0_x", max=max_len)},
            "y": {0: Dim("dim0_y", max=max_len)},
        }
        m = M()
        ep = export(m, args, dynamic_shapes=dynamic_shapes)

        FileCheck().check_count("torch.ops.mylib.add", 1, exactly=True).run(
            ep.graph_module.code
        )
        ep_decomposed = ep.run_decompositions(decompose_custom_triton_ops=False)
        FileCheck().check_count("torch.ops.mylib.add", 1, exactly=True).run(
            ep.graph_module.code
        )
        ep_decomposed = ep.run_decompositions(decompose_custom_triton_ops=True)
        FileCheck().check_count(
            "torch.ops.higher_order.triton_kernel_wrapper_functional", 1, exactly=True
        ).run(ep_decomposed.graph_module.code)
        exp_out = m(*args)
        self.assertEqual(exp_out, ep.module()(*args))

    @requires_gpu
    def test_export_custom_triton_kernel_mutable(self):
        @triton.jit
        def add_kernel(
            in_ptr0,
            in_ptr1,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

        @torch.library.triton_op("mylib::add", mutates_args={"output"})
        def custom_add_out(
            x: torch.Tensor, y: torch.Tensor, output: torch.Tensor
        ) -> torch.Tensor:
            n_elements = output.numel()

            def grid(meta):
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            capture_triton(add_kernel)[grid](x, y, output, n_elements, 16)
            return output.clone()

        class M(torch.nn.Module):
            def forward(self, x, y, out):
                return custom_add_out(x, y, out)

        args = (
            torch.randn(3, device=GPU_TYPE),
            torch.randn(3, device=GPU_TYPE),
            torch.zeros(3, device=GPU_TYPE),
        )
        custom_add_out(*args)
        max_len = 128
        dynamic_shapes = {
            "x": {0: Dim("dim0_x", max=max_len)},
            "y": {0: Dim("dim0_y", max=max_len)},
            "out": {0: Dim("dim0_z", max=max_len)},
        }

        m = M()
        ep = export(m, args, dynamic_shapes=dynamic_shapes)

        FileCheck().check_count("torch.ops.mylib.add", 1, exactly=True).run(
            ep.graph_module.code
        )
        ep_decomposed = ep.run_decompositions(decompose_custom_triton_ops=False)
        FileCheck().check_count(
            "torch.ops.higher_order.auto_functionalized", 1, exactly=True
        ).run(ep_decomposed.graph_module.code)

        ep_decomposed = ep.run_decompositions(decompose_custom_triton_ops=True)
        if is_training_ir_test(self._testMethodName):
            # TODO: For training IR test, we functionalize the custom triton op with auto_functionalized.
            # The custom op's functional decomposition is not triggered as a result. It might be better to
            # decompose the custom triton ops. Users can workaround by unwrapping auto_functionalized
            # in order to get the functional triton hop if needed.
            FileCheck().check_count(
                "torch.ops.higher_order.auto_functionalized", 1, exactly=True
            ).run(ep_decomposed.graph_module.code)
        else:
            FileCheck().check_count(
                "torch.ops.higher_order.triton_kernel_wrapper_functional",
                1,
                exactly=True,
            ).run(ep_decomposed.graph_module.code)

        x, y, out = (
            torch.randn(3, device=GPU_TYPE),
            torch.randn(3, device=GPU_TYPE),
            torch.zeros(3, device=GPU_TYPE),
        )
        exp_out = m(x, y, out)
        out_copy = out.clone()
        out_copy2 = out.clone()
        out_copy3 = out.clone()
        self.assertEqual(exp_out, ep.module()(x, y, out_copy))
        # For non-functional graph module, out_copy is mutated
        self.assertEqual(out, out_copy)
        self.assertEqual(exp_out, ep_decomposed.module()(x, y, out_copy2))
        # For non-functional graph module, out_copy is not mutated
        self.assertEqual(out_copy2, out_copy3)

    def test_masked_select_dynamic(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                mask = x.ge(0.5)
                return torch.masked_select(x, mask)

        example_args = (torch.randn(3, 4, 5),)
        dim0_x_max, dim1_x_max = 100, 7
        dynamic_shapes = {
            "x": {
                0: Dim("dim0_x", max=dim0_x_max),
                1: Dim("dim1_x_max", max=dim1_x_max),
            }
        }
        m = M()
        exported_program: torch.export.ExportedProgram = export(
            m, args=example_args, dynamic_shapes=dynamic_shapes
        )

        # Test that the expected upper bound is among the range constraints.
        expected_upper_bound = dim0_x_max * dim1_x_max * 5
        vr_upper_bounds = [
            vr.upper for vr in exported_program.range_constraints.values()
        ]
        self.assertTrue(expected_upper_bound in set(vr_upper_bounds))
        # Test that none of the upper bounds are larger.
        for vr_upper in vr_upper_bounds:
            self.assertTrue(vr_upper <= expected_upper_bound)

    def test_nonzero_dynamic(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x: torch.Tensor, as_tuple: bool) -> torch.Tensor:
                return torch.nonzero(x, as_tuple=as_tuple)

        # Case 1 and 2: as_tuple is True and as_tuple is False.
        for as_tuple in [True, False]:
            example_args = (torch.randn(3, 4, 5), as_tuple)
            dim0_x_max, dim1_x_max = 100, 7
            dynamic_shapes = {
                "x": {
                    0: Dim("dim0_x", max=dim0_x_max),
                    1: Dim("dim1_x_max", max=dim1_x_max),
                },
                "as_tuple": None,
            }
            m = M()
            exported_program: torch.export.ExportedProgram = export(
                m, args=example_args, dynamic_shapes=dynamic_shapes
            )

            # Test that the expected upper bound is among the range constraints.
            expected_upper_bound = dim0_x_max * dim1_x_max * 5
            vr_upper_bounds = [
                vr.upper for vr in exported_program.range_constraints.values()
            ]
            self.assertTrue(expected_upper_bound in set(vr_upper_bounds))
            # Test that none of the upper bounds are larger.
            for vr_upper in vr_upper_bounds:
                self.assertTrue(vr_upper <= expected_upper_bound)

        # Case 3: Test special case when input has zero dimensions and a nonzero
        # scalar value.
        example_args = (torch.tensor(10), as_tuple)
        dim0_x_max = 100
        dynamic_shapes = {
            "x": None,
            "as_tuple": None,
        }
        m = M()
        exported_program: torch.export.ExportedProgram = export(
            m, args=example_args, dynamic_shapes=dynamic_shapes
        )

        # Test that the expected upper bound is equal to 1, since our output
        # for this edge case should always be a tensor of size 1.
        vr_upper_bounds = [
            vr.upper for vr in exported_program.range_constraints.values()
        ]
        for vr_upper in vr_upper_bounds:
            self.assertEqual(vr_upper, 1)

    @testing.expectedFailureStrictV2
    def test_detect_leak_strict(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x + y

        global_list = []

        class ReferenceControl:
            def __init__(self, mod):
                self.bank = []
                self.bank_dict = {}
                self.mod = mod

                def hacked_up_forward(self_, x, y):
                    self.bank.append(x.clone())
                    self.bank_dict["x"] = x.clone()
                    global_list.append(x.clone())
                    return x + y

                self.mod.forward = hacked_up_forward.__get__(self.mod, Foo)

            def __call__(self, x, y):
                ep = export(self.mod, (x, y), strict=True).module()
                out = ep(x, y)
                return out

            def update(self):
                print(self.bank)

        foo = Foo()
        ref = ReferenceControl(foo)
        # TODO (tmanlaibaatar) this kinda sucks but today there is no good way to get
        # good source name. We should have an util that post processes dynamo source names
        # to be more readable.
        with self.assertWarnsRegex(
            UserWarning,
            r"(L\['self']\._modules\['_export_root']\.forward\.__func__\.__closure__\[1\]\.cell_contents\.bank"
            r"|L\['self']\._modules\['_export_root']\.forward\.__func__\.__closure__\[1\]\.cell_contents\.bank_dict"
            r"|L\['self']\._modules\['_export_root']\.forward\.__func__\.__closure__\[0\]\.cell_contents)",
        ):
            ref(torch.randn(4, 4), torch.randn(4, 4))

    def test_mask_nonzero_static(self):
        class TestModule(torch.nn.Module):
            def forward(self, seq_embeddings, mask, exp):
                # Instead of `output = seq_embeddings[mask]`` which makes
                # output.shape have unbacked symint, encode side knowledge of
                # output.shape as exp.shape to force it to have backed symint
                index = torch.nonzero_static(mask, size=exp.shape[0])
                chunked_index = index.chunk(chunks=mask.dim(), dim=1)
                output = seq_embeddings[chunked_index].squeeze()
                final_output = output * 2
                return final_output

        m = TestModule()

        seq_embeddings = torch.randn(5, 5)
        mask = torch.ones(5, 5, dtype=torch.bool)
        exp = torch.randn(25)
        output = m(seq_embeddings, mask, exp)

        batch = torch.export.Dim("batch")
        exp_size = torch.export.Dim("exp_size", max=100)
        ep = export(
            m,
            (seq_embeddings, mask, exp),
            dynamic_shapes={
                "seq_embeddings": (batch, None),
                "mask": (batch, None),
                "exp": (exp_size,),
            },
        )
        ep_output = ep.module()(seq_embeddings, mask, exp)
        self.assertTrue(torch.allclose(output, ep_output))

        seq_embeddings = torch.randn(6, 5)
        mask = torch.ones(6, 5, dtype=torch.bool)
        exp = torch.randn(30)
        output = m(seq_embeddings, mask, exp)
        ep_output = ep.module()(seq_embeddings, mask, exp)
        self.assertTrue(torch.allclose(output, ep_output))

    def test_setgrad_lifted_tensor(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                with torch.enable_grad():
                    c = torch.tensor(4)
                    z = c + x + y

                return z * z

        m = M()
        x = torch.randn(4)
        y = torch.randn(4)
        # Need to surround export with no_grad to bypass AutogradStateOpsFailSafeguard.
        with torch.no_grad():
            ep = export(m, (x, y))
        self.assertEqual(ep.module()(x, y), m(x, y))

    def test_subclass_context(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x + 1

        input = TwoTensor(
            TwoTensor(torch.randn(4, 4), torch.rand(4, 4)),
            TwoTensor(torch.randn(4, 4), torch.rand(4, 4)),
        )

        input_test = TwoTensor(
            TwoTensor(torch.randn(6, 6), torch.rand(6, 6)),
            TwoTensor(torch.randn(6, 6), torch.rand(6, 6)),
        )

        for strict in [True, False]:
            dim = torch.export.ShapesCollection()
            dim[input] = [Dim.STATIC, Dim.AUTO]
            ep = torch.export.export(Foo(), (input,), strict=strict, dynamic_shapes=dim)
            self.assertExpectedInline(
                str(ep.graph).strip(),
                """\
graph():
    %x : [num_users=1] = placeholder[target=x]
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, 1), kwargs = {})
    return (add,)""",
            )

            with self.assertRaisesRegex(
                AssertionError, escape("Guard failed: x.size()[0] == 4")
            ):
                ep.module()(input_test)

    def test_basic_non_strict_real_tensor(self):
        class Basic(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(1, 3))

            def forward(self, x, y):
                return x[0] + y - self.param

        f = Basic()
        args = ([torch.randn(1, 3)], torch.randn(1, 3))
        ep = export(f, args, strict=False)
        self.assertEqual(ep.module()(*args), f(*args))

    def test_where_decomp(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.ops.aten.where.default(x > 0)

        test_module = TestModule()
        sample_input = (torch.randn(2, 3),)

        def auto_dynamic_shapes_from_args(args):  # pyre-ignore
            """
            This function creates dynamic shapes specification with Dim.AUTO
            in all dimensions of all tensors for given argument list.
            """
            if isinstance(args, list):
                return [auto_dynamic_shapes_from_args(arg) for arg in args]
            elif isinstance(args, tuple):
                return tuple(auto_dynamic_shapes_from_args(arg) for arg in args)
            elif isinstance(args, dict):
                return {k: auto_dynamic_shapes_from_args(v) for k, v in args.items()}
            elif isinstance(args, torch.Tensor):
                return {j: Dim.AUTO for j in range(args.dim())}
            else:
                print(f"args type: {type(args)}")
                return None

        ep = torch.export.export(
            test_module,
            sample_input,
            dynamic_shapes=auto_dynamic_shapes_from_args(sample_input),
        ).run_decompositions({})

    def test_basic_non_strict_fake_tensor(self):
        class Basic(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn(3, 2))

            def forward(self, x, y):
                return x[0] + y - self.param

        fake_mode = FakeTensorMode(shape_env=ShapeEnv(tracked_fakes=[]))
        f = Basic()
        with fake_mode:
            args = ([torch.empty(3, 2)], torch.empty(3, 2))
        ep = export(f, args, strict=False)
        inputs = ([torch.randn(3, 2)], torch.randn(3, 2))
        self.assertEqual(ep.module()(*inputs), f(*inputs))

    def test_non_strict_dynamic_shapes(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.u = torch.nn.Buffer(torch.ones(1))
                self.v = torch.nn.Buffer(torch.ones(1))

            def forward(self, x, ys, zs, c):
                y = ys[0] + ys[1] + zs["a"] + zs["b"]
                self.v.add_(3)
                w = self.u - self.v
                if x.shape[0] < 3 and c.shape[0] != 4:
                    return x + w, x + y
                else:
                    return x - w, x - y

        foo = Foo()

        inp = (
            torch.ones(5),
            [torch.zeros(5), torch.ones(5)],
            {"a": torch.zeros(5), "b": torch.ones(5)},
            torch.ones(4),
        )
        dim = torch.export.Dim("dim", min=3)
        dynamic_shapes = (
            {0: dim},
            [{0: dim}, {0: dim}],
            {"a": {0: dim}, "b": {0: dim}},
            None,
        )

        ep_ns = torch.export.export(
            foo, inp, dynamic_shapes=dynamic_shapes, strict=False
        )

        bad_runtime_inp1 = (
            torch.ones(6),
            [torch.zeros(5), torch.ones(5)],
            {"a": torch.zeros(5), "b": torch.ones(5)},
            torch.ones(4),
        )
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: ys[0].size()[0] == x.size()[0]"),
        ):
            # expected 6, but got 5
            ep_ns.module()(*bad_runtime_inp1)

        bad_runtime_inp2 = (
            torch.ones(5),
            [torch.zeros(5), torch.ones(5)],
            {"a": torch.zeros(5), "b": torch.ones(5)},
            torch.ones(6),
        )
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: c.size()[0] == 4"),
        ):
            # expected 4, but got 6
            ep_ns.module()(*bad_runtime_inp2)

        good_runtime_inp = (
            torch.ones(7),
            [torch.zeros(7), torch.ones(7)],
            {"a": torch.zeros(7), "b": torch.ones(7)},
            torch.ones(4),
        )
        ep_ns.module()(*good_runtime_inp)

        bad_example_inp = (
            torch.ones(2),
            [torch.zeros(2), torch.ones(2)],
            {"a": torch.zeros(2), "b": torch.ones(2)},
            torch.ones(4),
        )
        with self.assertRaisesRegex(
            torch.fx.experimental.symbolic_shapes.ConstraintViolationError,
            "2 not in range.*3,",
        ):
            ep_ns = torch.export.export(
                foo, bad_example_inp, dynamic_shapes=dynamic_shapes, strict=False
            )

    def test_non_strict_dynamic_shapes_suggested_fixes(self):
        class Foo(torch.nn.Module):
            def forward(self, x, c):
                if x.shape[0] <= 6:
                    return x + 1, c + 2
                else:
                    return x - 1, c - 2

        foo = Foo()

        bad_example_inp = (
            torch.ones(5),
            torch.ones(4),
        )
        dim = torch.export.Dim("dim", min=3)
        dynamic_shapes = (
            {0: dim},
            None,
        )

        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Constraints violated \\(dim\\)!(.*\n)*.*"
            "Not all values of dim.*satisfy the generated guard(.*\n)*.*"
            "Suggested fixes:(.*\n)*.*"
            "dim = Dim\\('dim', min=3, max=6\\)",
        ):
            torch.export.export(
                foo, bad_example_inp, dynamic_shapes=dynamic_shapes, strict=False
            )

    def test_symint_item(self):
        class M(torch.nn.Module):
            def forward(self, tensor):
                return tensor.item()

        input = (torch.tensor([1], dtype=torch.int),)

        orig_res = M()(*input)
        ep_res = torch.export.export(M(), input).module()(*input)
        self.assertEqual(orig_res, ep_res)

    def test_symbool_item(self):
        class M(torch.nn.Module):
            def forward(self, tensor):
                return tensor.item()

        input = (torch.tensor([1], dtype=torch.bool),)

        orig_res = M()(*input)
        ep_res = torch.export.export(M(), input).module()(*input)
        self.assertEqual(orig_res, ep_res)

    def test_symfloat_item(self):
        class M(torch.nn.Module):
            def forward(self, tensor):
                return tensor.item()

        input = (torch.tensor([3.14], dtype=torch.float),)

        orig_res = M()(*input)
        ep_res = torch.export.export(M(), input).module()(*input)
        self.assertEqual(orig_res, ep_res)

    def test_unbacked_to_cond(self):
        strict = True

        class M(torch.nn.Module):
            def forward(self, a):
                az = a.nonzero()

                def true_fn(x):
                    return (x + 1).sum()

                def false_fn(x):
                    return (x + 3).sum()

                r = torch.cond(az.size(0) > 3, true_fn, false_fn, (az,))
                return r * 2

        M()(torch.randn(7))
        torch.export.export(M(), (torch.randn(7),), strict=strict)

    def test_unbacked_to_cond_passthrough(self):
        strict = True

        class M(torch.nn.Module):
            def forward(self, a):
                az = a.nonzero()

                def true_fn(x):
                    return x + 1

                def false_fn(x):
                    return x + 3

                r = torch.cond(az.size(0) > 3, true_fn, false_fn, (az,))
                return r * 2

        M()(torch.randn(7))
        torch.export.export(M(), (torch.randn(7),), strict=strict)

    def test_cond_branches_return_constant_int(self):
        if "cpp_runtime_nonstrict" in self.id():
            self.skipTest("TODO Unexpected success in OSS but not in fbcode.")

        class M(torch.nn.Module):
            def forward(self, x):
                idx = torch.cond(x.sum() > 3, lambda: 0, lambda: 1, tuple())
                return x[idx]

        args = (torch.randn(3, 3),)
        m = M()
        ep = export(M(), args)
        if self._testMethodName == "test_cond_branches_return_constant_int":
            self.assertExpectedInline(
                normalize_gm(ep.module().print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, x):
        x: "f32[3, 3]";

        x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
        _guards_fn = self._guards_fn(x);  _guards_fn = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(x)
        gt: "b8[]" = torch.ops.aten.gt.Scalar(sum_1, 3);  sum_1 = None

        true_graph_0 = self.true_graph_0
        false_graph_0 = self.false_graph_0
        cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, ());  gt = true_graph_0 = false_graph_0 = None

        getitem_1: "Sym(u0)" = cond[0];  cond = None

        ge_1: "Sym(u0 >= 0)" = getitem_1 >= 0
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default = None
        le_1: "Sym(u0 <= 1)" = getitem_1 <= 1
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(le_1, "Runtime assertion failed for expression u0 <= 1 on node 'le_1'");  le_1 = _assert_scalar_default_1 = None

        select: "f32[3]" = torch.ops.aten.select.int(x, 0, getitem_1);  x = getitem_1 = None
        return pytree.tree_unflatten((select,), self._out_spec)

    class true_graph_0(torch.nn.Module):
        def forward(self):
            return (0,)

    class false_graph_0(torch.nn.Module):
        def forward(self):
            return (1,)
""",  # noqa: B950
            )
        self.assertEqual(m(*args), ep.module()(*args))

    @testing.expectedFailureCppRuntimeNonStrict
    def test_cond_access_identical_symint_closure(self):
        class Example2(torch.nn.Module):
            def forward(self, x, trigger, target):
                return torch.cond(
                    trigger == 1,
                    lambda: x + target,
                    lambda: x * target,
                    (),
                )

        m = Example2()
        x = torch.randn(2)
        trigger = 0
        target = 2
        args = (x, trigger, target)
        with config.patch(use_new_tracer_experimental=True):
            ep = export(m, args, dynamic_shapes=(None, Dim.DYNAMIC, Dim.DYNAMIC))
            self.assertExpectedInline(
                str(tuple(ep.range_constraints.values())),
                """(VR[0, int_oo], VR[0, int_oo])""",
            )
        self.assertEqual(m(*args), ep.module()(*args))

    def test_cond_branches_return_same_int(self):
        class M(torch.nn.Module):
            def forward(self, x):
                idx = torch.cond(x.sum() > 3, lambda: 0, lambda: 0, tuple())
                return x[idx]

        args = (torch.randn(3, 3),)
        m = M()
        ep = export(M(), args)
        # Ideally, we could remove the cond at the front end directly
        # since it's not used anyway. But we can only do this early
        # optimization if all the outputs are the same constants, which
        # will complicates the output check so just keep it in the graph.
        # let downstream to dce it.
        if self._testMethodName == "test_cond_branches_return_same_int":
            self.assertExpectedInline(
                normalize_gm(ep.module().print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, x):
        x: "f32[3, 3]";

        x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
        _guards_fn = self._guards_fn(x);  _guards_fn = None

        sum_1: "f32[]" = torch.ops.aten.sum.default(x)
        gt: "b8[]" = torch.ops.aten.gt.Scalar(sum_1, 3);  sum_1 = None

        true_graph_0 = self.true_graph_0
        false_graph_0 = self.false_graph_0
        cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, ());  gt = true_graph_0 = false_graph_0 = None
        getitem = cond[0];  cond = getitem = None

        select: "f32[3]" = torch.ops.aten.select.int(x, 0, 0);  x = None
        return pytree.tree_unflatten((select,), self._out_spec)

    class true_graph_0(torch.nn.Module):
        def forward(self):
            return (0,)

    class false_graph_0(torch.nn.Module):
        def forward(self):
            return (0,)
""",  # noqa: B950
            )

        self.assertEqual(m(*args), ep.module()(*args))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_cond_contains_unbacked_no_escape(self):
        class M(torch.nn.Module):
            def forward(self, a, b1, b2, c):
                def true_fn(x):
                    return x * b1.item()

                def false_fn(x):
                    return x * b2.item()

                r = torch.cond(a, true_fn, false_fn, (c,))
                return r * 2

        args = (
            torch.tensor(True),
            torch.tensor([4]),
            torch.tensor([4]),
            torch.randn(10, requires_grad=True),
        )
        torch.export.export(M(), args)

    def test_cond_int_closure(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num = 4

            def forward(self, a, x):
                def true_fn(x):
                    return x * self.num

                def false_fn(x):
                    return x + self.num

                r = torch.cond(a, true_fn, false_fn, (x,))
                return r * 2

        args = (torch.tensor(True), torch.randn(10))
        ep = torch.export.export(M(), args)
        self.assertEqual(ep.module()(*args), M()(*args))

    def test_state_tensors(self):
        class M(torch.nn.Module):  # simple with register buffer
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(2, 3), persistent=False)

            def forward(self, x):
                # x = 2
                y = self.buf
                # y = 1
                w1 = self.buf + 3
                w2 = self.buf + 4
                w3 = self.buf + 5
                self.buf = w1
                z = self.buf
                self.buf = w3
                # z = 4
                return x + y + z + w2

        ep = export(M(), (torch.randn(2, 3),), strict=False).run_decompositions({})
        self.assertEqual(list(ep.graph_signature.buffers_to_mutate.values()), ["buf"])
        self.assertTrue(
            torch.allclose(ep.module()(torch.ones(2, 3) + 1), torch.ones(2, 3) * 12)
        )

        class M(torch.nn.Module):  # simple without register buffer
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.ones(2, 3)

            def forward(self, x):
                # x = 2
                y = self.buf
                # y = 1
                self.buf = self.buf + 3
                z = self.buf
                # z = 3
                return x + y + z

        with self.assertWarnsRegex(
            UserWarning,
            "The tensor attribute self.buf was assigned during export",
        ):
            export(M(), (torch.randn(2, 3),), strict=False)

        class M(torch.nn.Module):  # complex with register buffer
            def __init__(self) -> None:
                super().__init__()
                tensors = [torch.ones(2, 3), torch.ones(2, 3)]
                for i, tensor in enumerate(tensors):
                    self.register_buffer(f"buf_{i}", tensor, persistent=False)

            def get_tensor(self, i):
                return getattr(self, f"buf_{i}")

            def set_tensor(self, i, val):
                setattr(self, f"buf_{i}", val)

            def forward(self, x):
                # x = 2
                y = self.get_tensor(0) + self.get_tensor(1)
                # y = 1 + 1
                self.set_tensor(0, torch.ones(2, 3) + 2)
                self.set_tensor(1, torch.ones(2, 3) + 2)
                z = self.get_tensor(0) + self.get_tensor(1)
                # z = 3 + 3
                return x + y + z

        ep = export(M(), (torch.randn(2, 3),), strict=False).run_decompositions({})
        self.assertEqual(
            list(ep.graph_signature.buffers_to_mutate.values()), ["buf_0", "buf_1"]
        )
        self.assertTrue(
            torch.allclose(ep.module()(torch.ones(2, 3) + 1), torch.ones(2, 3) * 10)
        )

        class M(torch.nn.Module):  # complex without register buffer
            def __init__(self) -> None:
                super().__init__()
                self.tensors = [torch.ones(2, 3), torch.ones(2, 3)]

            def get_tensor(self, i):
                return self.tensors[i]

            def set_tensor(self, i, val):
                self.tensors[i] = val

            def forward(self, x):
                # x = 2
                y = self.get_tensor(0) + self.get_tensor(1)
                # y = 1 + 1
                self.set_tensor(0, torch.ones(2, 3) + 2)
                self.set_tensor(1, torch.ones(2, 3) + 2)
                z = self.get_tensor(0) + self.get_tensor(1)
                # z = 3 + 3
                return x + y + z

        with self.assertWarnsRegex(
            UserWarning,
            "The tensor attributes self.tensors\\[0\\], self.tensors\\[1\\] were assigned during export",
        ):
            export(M(), (torch.randn(2, 3),), strict=False)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_while_loop_tensor_constant_idx(self):
        def while_loop_decomp(x, y0):
            out = torch.zeros_like(x)

            def cond_fn(idx, out, y0):
                return idx < out.size(0)

            def body_fn(idx, out, y0):
                i = idx.item()
                # TODO removing those causes PendingUnbackedSymbolNotFound.
                torch._check(i >= 0)
                torch._check(i < x.size(0))
                y0 = x[i] + y0
                out = out.clone()
                out[i] = y0
                return idx + 1, out, y0

            cnt = torch.tensor(0)
            _, out, _ = while_loop(cond_fn, body_fn, [cnt, out, y0])
            return out

        class TestModel(torch.nn.Module):
            def forward(self, x, y0):
                return while_loop_decomp(x, y0)

        x, y0 = torch.randn(16, 8), torch.randn(8)
        exp_out = TestModel()(x, y0)
        ep = export(TestModel(), (x, y0))
        out = ep.module()(x, y0)
        self.assertEqual(exp_out, out)

    def test_malformed_fqn_from_source_name(self):
        # See https://github.com/pytorch/pytorch/issues/141939
        from types import MethodType

        class Block(torch.nn.Module):
            def __init__(self, i, o):
                super().__init__()
                self.to_out = torch.nn.ModuleList([])
                self.to_out.append(torch.nn.Linear(i, o, bias=True))
                self.to_out.append(torch.nn.Dropout(0.5))

            def forward(self, x):
                for l in self.to_out:
                    x = l(x)
                return x

        class Problem1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleDict(
                    {f"{i}": Block(64, 64) for i in range(5)}
                )

            def forward(self, x):
                for k, m in self.blocks.items():
                    x = m(x)
                return x

        class Problem2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList([Block(64, 64) for i in range(5)])

            def forward(self, x):
                x = self.blocks[0](x)
                for m in self.blocks[1:4]:
                    x = m(x)
                return x

        def _split_after_forward(self, *args, **kwargs):
            return self._orig_forward(*args, **kwargs)

        def annotate_split_points(mod: torch.nn.Module, spec):
            for qualname, split_type in spec.items():
                atoms = qualname.split(".")
                predecessor_module = mod
                for i, atom in enumerate(atoms[:-1]):
                    try:
                        predecessor_module = getattr(predecessor_module, atom)
                    except AttributeError as e:
                        raise e
                mod_to_wrap = getattr(predecessor_module, atoms[-1])
                mod_to_wrap._orig_forward = mod_to_wrap.forward
                mod_to_wrap.forward = MethodType(_split_after_forward, mod_to_wrap)

        for problem in [Problem1, Problem2]:
            m = problem()
            m(torch.rand(64, 64))
            # simplified torch.distributed.pipeline code
            annotate_split_points(m, {"blocks.1": 1, "blocks.3": 1})
            gm = export(m, (torch.rand(64, 64),))
            torch.export.unflatten(gm)

    @testing.expectedFailureStrictV2
    def test_unflatten_closure(self):
        class Dummy(torch.nn.Module):
            def forward(self, fn, x):
                y = x + 2
                z = fn(y)
                return z + 4

        class N(torch.nn.Module):
            def forward(self, x):
                return x + 3

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = Dummy()
                self.n = N()

            def forward(self, x):
                y = x + 1
                z = self.dummy(lambda k: self.n(y + k) + y, y)
                return z + 5

        m = M()
        x = torch.randn(3)
        ep = export(m, (x,))

        ufm = torch.export.unflatten(ep)
        self.assertExpectedInline(
            str(ufm.graph_module.code).strip(),
            """\
def forward(self, x):
    add = torch.ops.aten.add.Tensor(x, 1);  x = None
    dummy = self.dummy(add);  add = None
    add_6 = torch.ops.aten.add.Tensor(dummy, 5);  dummy = None
    return (add_6,)""",
        )
        self.assertExpectedInline(
            str(ufm.dummy.graph_module.code).strip(),
            """\
def forward(self, add):
    add_1 = torch.ops.aten.add.Tensor(add, 2)
    add_2 = torch.ops.aten.add.Tensor(add, add_1);  add_1 = None
    add_3 = torch.ops.aten.add.Tensor(add_2, 3);  add_2 = None
    add_4 = torch.ops.aten.add.Tensor(add_3, add);  add_3 = add = None
    add_5 = torch.ops.aten.add.Tensor(add_4, 4);  add_4 = None
    return add_5""",
        )

    def test_state_primitives(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.x = 1
                self.y = {"k": 2}
                self.z = (3,)

            def forward(self, x):
                self.x = self.x + 4
                self.y["k"] = self.y["k"] + 5
                self.z = (self.z[0] + 6,)
                return x + self.x + self.y["k"] + self.z[0]

        ep = export(M(), (torch.randn(2, 3),))
        self.assertTrue(
            torch.allclose(ep.module()(torch.zeros(2, 3)), torch.ones(2, 3) * 21)
        )

    def test_state_shape_attribute_assignment(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.last_z_shape = self.linear.weight.shape

            def forward(self, x):
                self.last_z_shape = x.shape
                return self.linear(x)

        model = TestModule()
        x = torch.randn(20, 10)
        ep_model = export(model, (x,), strict=False).module()
        self.assertTrue(torch.allclose(model(x), ep_model(x)))

    def test_output_node_name(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        model = TestModule()
        x = torch.randn(20, 10)
        ep_model = export(model, (x,), strict=False).module()
        self.assertEqual(list(ep_model.graph.nodes)[-1].name, "output")
        self.assertTrue(torch.allclose(model(x), ep_model(x)))

    def test_real_tensor_size_mismatch(self):
        from torch._subclasses.fake_tensor import MetadataMismatchError

        class M(torch.nn.Module):
            def forward(self, a, b):
                return torch.ops.mylib.foo(a, b)

        @torch.library.custom_op("mylib::foo", mutates_args={})
        def foo(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a + b

        @foo.register_fake
        def foo_fake_impl(a, b):
            m, n = a.shape
            return torch.empty(n, m)  # incorrectly permute

        error_type = (
            MetadataMismatchError
            if is_non_strict_test(self._testMethodName)
            else torch._dynamo.exc.TorchRuntimeError
        )
        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            # won't catch anything if dims are equal
            export(
                M(),
                (torch.randn(4, 4), torch.randn(4, 4)),
            )
            # catch concrete inequality
            with self.assertRaisesRegex(
                error_type,
                r"Real tensor propagation found an output size mismatch between fake shape 8 and real shape 4, "
                r"at output\.size\(0\), for func: mylib.foo.default",
            ):
                export(
                    M(),
                    (torch.randn(4, 8), torch.randn(4, 8)),
                )
            # same test with dynamic shapes
            d0 = Dim("d0")
            d1 = Dim("d1")
            export(
                M(),
                (torch.randn(4, 4), torch.randn(4, 4)),
                dynamic_shapes={
                    "a": (d0, d1),
                    "b": (d0, d1),
                },
            )
            with self.assertRaisesRegex(
                error_type,
                r"Real tensor propagation found an output size mismatch between fake shape s\d+ and real shape 4, "
                r"at output\.size\(0\), for func: mylib.foo.default",
            ):
                export(
                    M(),
                    (torch.randn(4, 8), torch.randn(4, 8)),
                    dynamic_shapes={
                        "a": (d0, d1),
                        "b": (d0, d1),
                    },
                )

    def test_real_tensor_alias_dtype_mismatch(self):
        from torch._subclasses.fake_tensor import MetadataMismatchError

        error_type = (
            MetadataMismatchError
            if is_non_strict_test(self._testMethodName)
            else torch._dynamo.exc.TorchRuntimeError
        )

        # test alias case
        class M(torch.nn.Module):
            def forward(self, a):
                return torch.ops.mylib.foo_alias(a)

        @torch.library.custom_op("mylib::foo_alias", mutates_args={})
        def foo_alias(a: torch.Tensor) -> torch.Tensor:
            return a * 2

        @foo_alias.register_fake
        def foo_fake_impl(a):
            return a

        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            with self.assertRaisesRegex(
                error_type,
                r"Real tensor propagation found an aliasing mismatch between fake output (.*\n)*.* "
                r"and real output (.*\n)*.* for func: mylib.foo_alias.default",
            ):
                ep = export(M(), (torch.randn(4, 4),))

        # test dtype case
        class N(torch.nn.Module):
            def forward(self, a):
                return torch.ops.mylib.foo_dtype(a)

        @torch.library.custom_op("mylib::foo_dtype", mutates_args={})
        def foo_dtype(a: torch.Tensor) -> torch.Tensor:
            return a * 2

        @foo_dtype.register_fake
        def foo_fake_impl(a):
            m, n = a.shape
            return torch.empty([m, n], dtype=torch.int32)

        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            with self.assertRaisesRegex(
                error_type,
                r"Real tensor propagation found a metadata mismatch between fake tensor (.*\n)*.* "
                r"and real tensor (.*\n)*.* at output, for func: mylib.foo_dtype.default",
            ):
                ep = export(N(), (torch.randn(4, 4),))

    def test_real_tensor_for_max_op(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                x = x[x > 0]
                y = y[y > 0]
                return max(x.shape[0], y.shape[0])

        model = Foo()
        inputs = (torch.zeros(64), torch.ones(64))
        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            ep = export(model, inputs)

        self.assertEqual(ep.module()(*inputs), model(*inputs))
        x = torch.zeros(64)
        y = torch.ones(64)
        # This seems to be a bug with old export because when we pass in x, x
        # as input, runtime assertion should fail. This is because we would create
        # guard on y.shape[0] > x.shape[0] but somehow in old export, we dce this
        # assertion.
        self.assertEqual(ep.module()(x, x), model(x, x))
        self.assertEqual(ep.module()(x, y), model(x, y))

    def test_draft_export_checks_mutation_with_nan(self):
        @torch.library.custom_op("export::foo", mutates_args={})
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

        @foo.register_fake
        def _(x, y):
            return x + y

        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return foo(x, y)

        model = Foo()
        inputs = (torch.full((64,), torch.nan), torch.full((64,), torch.nan))
        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            ep = export(model, inputs)

    def test_draft_export_checks_mutation(self):
        @torch.library.custom_op("export::foo", mutates_args={})
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            y.add_(1)
            return x.clone()

        @foo.register_fake
        def _(x, y):
            return x.clone()

        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return foo(x, y)

        model = Foo()
        inputs = (torch.randn(64), torch.randn(64))
        with self.assertRaisesRegex(RuntimeError, "for argument 'y'"):
            with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
                ep = export(model, inputs)

        @torch.library.custom_op("export::foo", mutates_args={"y"})
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            y.add_(1)
            return x.clone()

        @foo.register_fake
        def _(x, y):
            return x.clone()

        # No errors
        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            ep = export(model, inputs)

    def test_draft_export_checks_mutation_list(self):
        @torch.library.custom_op("export::foo", mutates_args={})
        def foo(xs: List[torch.Tensor]) -> torch.Tensor:
            x, y = xs
            y.add_(1)
            return x.clone()

        @foo.register_fake
        def _(xs):
            x, y = xs
            return x.clone()

        class Foo(torch.nn.Module):
            def forward(self, xs):
                return foo(xs)

        model = Foo()
        inputs = ([torch.randn(64), torch.randn(64)],)
        with self.assertRaisesRegex(RuntimeError, "for argument 'xs'"):
            with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
                ep = export(model, inputs)

        @torch.library.custom_op("export::foo", mutates_args={"xs"})
        def foo(xs: List[torch.Tensor]) -> torch.Tensor:
            x, y = xs
            y.add_(1)
            return x.clone()

        @foo.register_fake
        def _(xs):
            x, y = xs
            return x.clone()

        # No errors
        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            ep = export(model, inputs)

    def test_draft_export_checks_aliasing(self):
        @torch.library.custom_op("export::foo", mutates_args={})
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x

        @foo.register_fake
        def _(x, y):
            return x.clone()

        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return foo(x, y)

        model = Foo()
        inputs = (torch.randn(64), torch.randn(64))
        with self.assertRaisesRegex(RuntimeError, "may not alias"):
            with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
                ep = export(model, inputs)

        @torch.library.custom_op("export::foo", mutates_args={})
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x.clone()

        @foo.register_fake
        def _(x, y):
            return x.clone()

        # No errors
        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            ep = export(model, inputs)

    def test_draft_export_infers_fake_kernel(self):
        strict = True
        with torch.library._scoped_library("export", "FRAGMENT") as lib:
            lib.define("bar(Tensor x) -> Tensor")
            lib.impl("bar", lambda x: x[0].clone(), "CPU")

            @torch.library.custom_op("export::foo", mutates_args={})
            def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x * y

            class Foo(torch.nn.Module):
                def forward(self, x, y):
                    return foo(x, y), torch.ops.export.bar(y)

            model = Foo()
            inputs = (torch.randn(1, 3), torch.randn(2, 1))
            with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
                ep = export(model, inputs, strict=strict)

        # expecttest only works for the base TestExport class.
        if self.__class__ != TestExport:
            return

        self.assertExpectedInline(
            str(ep.graph_module.code).strip(),
            """\
def forward(self, x, y):
    foo = torch.ops.export.foo.default(x, y);  x = None
    sym_size_int = torch.ops.aten.sym_size.int(foo, 0)
    sym_size_int_1 = torch.ops.aten.sym_size.int(foo, 1)
    sym_constrain_range_for_size_default = torch.ops.aten.sym_constrain_range_for_size.default(sym_size_int);  sym_constrain_range_for_size_default = None
    ge = sym_size_int >= 0;  sym_size_int = None
    _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
    sym_constrain_range_for_size_default_1 = torch.ops.aten.sym_constrain_range_for_size.default(sym_size_int_1);  sym_constrain_range_for_size_default_1 = None
    ge_1 = sym_size_int_1 >= 0;  sym_size_int_1 = None
    _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_1 = None
    bar = torch.ops.export.bar.default(y);  y = None
    sym_size_int_2 = torch.ops.aten.sym_size.int(bar, 0)
    sym_constrain_range_for_size_default_2 = torch.ops.aten.sym_constrain_range_for_size.default(sym_size_int_2);  sym_constrain_range_for_size_default_2 = None
    ge_2 = sym_size_int_2 >= 0;  sym_size_int_2 = None
    _assert_scalar_default_2 = torch.ops.aten._assert_scalar.default(ge_2, "Runtime assertion failed for expression u2 >= 0 on node 'ge_2'");  ge_2 = _assert_scalar_default_2 = None
    return (foo, bar)""",
        )

    def test_draft_export_fake_kernel_inference_errors(self):
        @torch.library.custom_op("export::foo", mutates_args={})
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x.expand(32, 3).contiguous()[4]

        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return foo(x, y)

        model = Foo()
        inputs = (torch.randn(1, 3), torch.randn(2, 1))

        with self.assertRaisesRegex(RuntimeError, "non-zero storage offset"):
            with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
                ep = export(model, inputs)

        @torch.library.custom_op("export::foo", mutates_args={})
        def foo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.randn(3, 3).diagonal()

        with self.assertRaisesRegex(RuntimeError, "not dense in memory"):
            with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
                ep = export(model, inputs)

    def test_subclasses_parameterization(self):
        if "cpp_runtime_nonstrict" in self.id():
            self.skipTest("TODO Unexpected success in OSS but not in fbcode.")

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4))
                self.p2 = torch.nn.Parameter(
                    CustomTensorPlainOut(torch.ones(3, 4), torch.ones(3, 4))
                )

            def forward(self, x):
                a = (2 * self.p1 + self.p2).sum()
                return x + a

        m = Foo()
        ref_x = torch.randn(3, 4)
        ref_out = m(ref_x)

        ep_training = torch.export.export(m, (ref_x,))
        self.assertExpectedInline(
            str(ep_training.graph).strip(),
            """\
graph():
    %p_p1 : [num_users=1] = placeholder[target=p_p1]
    %p_p2 : [num_users=1] = placeholder[target=p_p2]
    %x : [num_users=1] = placeholder[target=x]
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%p_p1, 2), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %p_p2), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%add,), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %sum_1), kwargs = {})
    return (add_1,)""",
        )

        ep = export(m, (ref_x,)).run_decompositions({})
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %p_p1 : [num_users=1] = placeholder[target=p_p1]
    %p_parametrizations_p2_original0 : [num_users=1] = placeholder[target=p_parametrizations_p2_original0]
    %p_parametrizations_p2_original1 : [num_users=1] = placeholder[target=p_parametrizations_p2_original1]
    %x : [num_users=1] = placeholder[target=x]
    %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%p_p1, 2), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %p_parametrizations_p2_original0), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %p_parametrizations_p2_original1), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %add_1), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%add_2,), kwargs = {})
    %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %sum_1), kwargs = {})
    return (add_3,)""",
        )
        res = ep.module()(ref_x)

        self.assertEqual(res, ref_out)

    @testing.expectedFailureCppRuntimeNonStrict
    def test_subclasses_parameterization_nested(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(2, 2))
                self.p2 = torch.nn.Parameter(
                    CustomTensorPlainOut(
                        CustomTensorPlainOut(
                            torch.Tensor([[0, 0], [0, 1]]),
                            torch.Tensor([[0, 0], [1, 0]]),
                        ),
                        CustomTensorPlainOut(
                            torch.Tensor([[1, 0], [0, 0]]),
                            torch.Tensor([[0, 1], [0, 0]]),
                        ),
                    )
                )

            def forward(self, x):
                a = (x + 2 * self.p1 + self.p2).sum().sum()
                return x + a

        m = Foo()
        ref_x = torch.randn(2, 2)
        ref_out = m(ref_x)

        ep_training = torch.export.export(m, (ref_x,), strict=False)
        self.assertExpectedInline(
            str(ep_training.graph).strip(),
            """\
graph():
    %p_p1 : [num_users=1] = placeholder[target=p_p1]
    %p_p2 : [num_users=1] = placeholder[target=p_p2]
    %x : [num_users=2] = placeholder[target=x]
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%p_p1, 2), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %mul), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %p_p2), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%add_1,), kwargs = {})
    %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%sum_1,), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %sum_2), kwargs = {})
    return (add_2,)""",
        )

        ep = export(m, (ref_x,))
        ep = ep.run_decompositions({})
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %p_p1 : [num_users=1] = placeholder[target=p_p1]
    %p_parametrizations_p2_original0 : [num_users=1] = placeholder[target=p_parametrizations_p2_original0]
    %p_parametrizations_p2_original1 : [num_users=1] = placeholder[target=p_parametrizations_p2_original1]
    %p_parametrizations_p2_original2 : [num_users=1] = placeholder[target=p_parametrizations_p2_original2]
    %p_parametrizations_p2_original3 : [num_users=1] = placeholder[target=p_parametrizations_p2_original3]
    %x : [num_users=2] = placeholder[target=x]
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%p_p1, 2), kwargs = {})
    %add : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %mul), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %p_parametrizations_p2_original0), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %p_parametrizations_p2_original1), kwargs = {})
    %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %add_2), kwargs = {})
    %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %p_parametrizations_p2_original2), kwargs = {})
    %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %p_parametrizations_p2_original3), kwargs = {})
    %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %add_5), kwargs = {})
    %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %add_6), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%add_7,), kwargs = {})
    %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%sum_1,), kwargs = {})
    %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %sum_2), kwargs = {})
    return (add_8,)""",
        )
        res = ep.module()(ref_x)
        self.assertEqual(res, ref_out)

    @testing.expectedFailureSerDer  # can't serialize functorch ops
    @testing.expectedFailureSerDerNonStrict  # can't serialize functorch ops
    @testing.expectedFailureCppRuntime
    def test_vmap(self):
        class Vmap(torch.nn.Module):
            def forward(self, x, y):
                f = lambda x, y: (x * y + 1).sum(dim=0)  # noqa: E731
                vmapped = torch.vmap(f)(x, y)
                return vmapped.sum(dim=0)

        DYN = torch.export.Dim.DYNAMIC
        inputs = (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.1, 0.2, 0.3]))
        dynamic = {"x": {0: DYN}, "y": {0: DYN}}
        ep = torch.export.export(Vmap(), inputs, {}, dynamic_shapes=dynamic)
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %x : [num_users=1] = placeholder[target=x]
    %y : [num_users=2] = placeholder[target=y]
    %sym_size_int_3 : [num_users=2] = call_function[target=torch.ops.aten.sym_size.int](args = (%y, 0), kwargs = {})
    %lazy_load_decompositions : [num_users=0] = call_function[target=torch._functorch.predispatch.lazy_load_decompositions](args = (), kwargs = {})
    %_vmap_increment_nesting : [num_users=0] = call_function[target=torch._functorch.predispatch._vmap_increment_nesting](args = (%sym_size_int_3, error), kwargs = {})
    %_add_batch_dim : [num_users=1] = call_function[target=torch._functorch.predispatch._add_batch_dim](args = (%x, 0, 1), kwargs = {})
    %_add_batch_dim_1 : [num_users=1] = call_function[target=torch._functorch.predispatch._add_batch_dim](args = (%y, 0, 1), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_add_batch_dim, %_add_batch_dim_1), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, 1), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add, [0]), kwargs = {})
    %_remove_batch_dim : [num_users=1] = call_function[target=torch._functorch.predispatch._remove_batch_dim](args = (%sum_1, 1, %sym_size_int_3, 0), kwargs = {})
    %_vmap_decrement_nesting : [num_users=0] = call_function[target=torch._functorch.predispatch._vmap_decrement_nesting](args = (), kwargs = {})
    %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%_remove_batch_dim, [0]), kwargs = {})
    return (sum_2,)""",
        )
        ep = torch.export.export(
            Vmap(), inputs, {}, dynamic_shapes=dynamic, strict=True
        )
        self.assertExpectedInline(
            str(ep.graph).strip(),
            """\
graph():
    %x : [num_users=1] = placeholder[target=x]
    %y : [num_users=2] = placeholder[target=y]
    %sym_size_int_2 : [num_users=2] = call_function[target=torch.ops.aten.sym_size.int](args = (%y, 0), kwargs = {})
    %lazy_load_decompositions : [num_users=0] = call_function[target=torch._functorch.predispatch.lazy_load_decompositions](args = (), kwargs = {})
    %_vmap_increment_nesting : [num_users=0] = call_function[target=torch._functorch.predispatch._vmap_increment_nesting](args = (%sym_size_int_2, error), kwargs = {})
    %_add_batch_dim : [num_users=1] = call_function[target=torch._functorch.predispatch._add_batch_dim](args = (%x, 0, 1), kwargs = {})
    %_add_batch_dim_1 : [num_users=1] = call_function[target=torch._functorch.predispatch._add_batch_dim](args = (%y, 0, 1), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_add_batch_dim, %_add_batch_dim_1), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, 1), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%add, [0]), kwargs = {})
    %_remove_batch_dim : [num_users=1] = call_function[target=torch._functorch.predispatch._remove_batch_dim](args = (%sum_1, 1, %sym_size_int_2, 0), kwargs = {})
    %_vmap_decrement_nesting : [num_users=0] = call_function[target=torch._functorch.predispatch._vmap_decrement_nesting](args = (), kwargs = {})
    %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%_remove_batch_dim, [0]), kwargs = {})
    return (sum_2,)""",
        )
        self.assertTrue(torch.allclose(ep.module()(*inputs), Vmap()(*inputs)))
        ep = export(Vmap(), inputs, {}, dynamic_shapes=dynamic).run_decompositions({})
        self.assertTrue(torch.allclose(ep.module()(*inputs), Vmap()(*inputs)))

    @testing.expectedFailureLegacyExportNonStrict  # Old export doesn't work with subclasses
    @testing.expectedFailureLegacyExportStrict  # Old export doesn't work with subclasses
    def test_subclass_nested_attr_access(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4))
                self.p2 = torch.nn.Parameter(
                    TwoTensor(
                        TwoTensor(torch.ones(3, 4), torch.ones(3, 4)),
                        TwoTensor(torch.ones(3, 4), torch.ones(3, 4)),
                    )
                )
                self.b1 = torch.nn.Buffer(
                    TwoTensor(
                        TwoTensor(torch.ones(3, 4), torch.ones(3, 4)),
                        TwoTensor(torch.ones(3, 4), torch.ones(3, 4)),
                    )
                )

            def forward(self, x):
                res = (2 * self.p1 + self.p2 + self.b1).sum()
                return x + res.get_elem_a().b

        m = Foo()
        ref_x = torch.randn(3, 4)
        ref_out = m(ref_x)
        ep_training = torch.export.export(m, (ref_x,), strict=False)
        self.assertTrue(torch.allclose(ep_training.module()(ref_x), ref_out))
        self.assertExpectedInline(
            str(ep_training.graph).strip(),
            """\
graph():
    %p_p1 : [num_users=1] = placeholder[target=p_p1]
    %p_p2 : [num_users=1] = placeholder[target=p_p2]
    %b_b1 : [num_users=1] = placeholder[target=b_b1]
    %x : [num_users=1] = placeholder[target=x]
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%p_p1, 2), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %p_p2), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %b_b1), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%add_1,), kwargs = {})
    %access_subclass_inner_tensor_default_64 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%sum_1, a), kwargs = {})
    %access_subclass_inner_tensor_default_69 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%access_subclass_inner_tensor_default_64, b), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %access_subclass_inner_tensor_default_69), kwargs = {})
    return (add_2,)""",
        )
        ep = export(m, (ref_x,))
        self.assertTrue(torch.allclose(ep.module()(ref_x), ref_out))

    def test_subclass_nested_attr_access_submodule(self):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4))
                self.p2 = torch.nn.Parameter(
                    TwoTensor(
                        TwoTensor(torch.ones(3, 4), torch.ones(3, 4)),
                        TwoTensor(torch.ones(3, 4), torch.ones(3, 4)),
                    )
                )
                self.b1 = torch.nn.Buffer(
                    TwoTensor(
                        TwoTensor(torch.ones(3, 4), torch.ones(3, 4)),
                        TwoTensor(torch.ones(3, 4), torch.ones(3, 4)),
                    )
                )

            def forward(self, x):
                return x

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bar = Bar()

            def forward(self, x):
                res = (2 * self.bar.p1 + self.bar.p2 + self.bar.b1).sum()
                return x + res.get_elem_a().b

        m = Foo()
        ref_x = torch.randn(3, 4)
        ref_out = m(ref_x)
        ep_training = torch.export.export(m, (ref_x,), strict=False)
        self.assertExpectedInline(
            str(ep_training.graph).strip(),
            """\
graph():
    %p_bar_p1 : [num_users=1] = placeholder[target=p_bar_p1]
    %p_bar_p2 : [num_users=1] = placeholder[target=p_bar_p2]
    %b_bar_b1 : [num_users=1] = placeholder[target=b_bar_b1]
    %x : [num_users=1] = placeholder[target=x]
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%p_bar_p1, 2), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %p_bar_p2), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %b_bar_b1), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%add_1,), kwargs = {})
    %access_subclass_inner_tensor_default_64 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%sum_1, a), kwargs = {})
    %access_subclass_inner_tensor_default_69 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%access_subclass_inner_tensor_default_64, b), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %access_subclass_inner_tensor_default_69), kwargs = {})
    return (add_2,)""",
        )
        ep = export(m, (ref_x,))
        self.assertTrue(torch.allclose(ep.module()(ref_x), ref_out))

    def test_subclass_nested_attr_access_const_metadata(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4))
                self.p2 = torch.nn.Parameter(
                    ConstantExtraMetadataTensor(
                        ConstantExtraMetadataTensor(torch.ones(3, 4)),
                    )
                )

            def forward(self, x):
                res = 2 * self.p1 + self.p2
                res2 = res + res.constant_attribute
                return x + res2.elem.elem

        m = Foo()
        ref_x = torch.randn(3, 4)
        ref_out = m(ref_x)
        ep_training = torch.export.export(m, (ref_x,), strict=False)
        self.assertExpectedInline(
            str(ep_training.graph).strip(),
            """\
graph():
    %p_p1 : [num_users=1] = placeholder[target=p_p1]
    %p_p2 : [num_users=1] = placeholder[target=p_p2]
    %x : [num_users=1] = placeholder[target=x]
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%p_p1, 2), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %p_p2), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, 4), kwargs = {})
    %access_subclass_inner_tensor_default_10 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%add_1, elem), kwargs = {})
    %access_subclass_inner_tensor_default_13 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%access_subclass_inner_tensor_default_10, elem), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %access_subclass_inner_tensor_default_13), kwargs = {})
    return (add_2,)""",
        )
        ep = export(m, (ref_x,))
        self.assertTrue(torch.allclose(ep.module()(ref_x), ref_out))

    def test_subclass_nested_attr_access_const_metadata_not_top_level(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4))
                self.p2 = torch.nn.Parameter(
                    ConstantExtraMetadataTensor(
                        ConstantExtraMetadataTensor(torch.ones(3, 4)),
                    )
                )

            def forward(self, x):
                res = 2 * self.p1 + self.p2
                res2 = res + res.constant_attribute
                return x + res2.elem.elem

        m = Foo()
        ref_x = torch.randn(3, 4)
        ref_out = m(ref_x)
        ep_training = torch.export.export(m, (ref_x,), strict=False)
        self.assertExpectedInline(
            str(ep_training.graph).strip(),
            """\
graph():
    %p_p1 : [num_users=1] = placeholder[target=p_p1]
    %p_p2 : [num_users=1] = placeholder[target=p_p2]
    %x : [num_users=1] = placeholder[target=x]
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%p_p1, 2), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %p_p2), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, 4), kwargs = {})
    %getattr_22 : [num_users=1] = call_function[target=builtins.getattr](args = (%add_1, elem), kwargs = {})
    %getattr_27 : [num_users=1] = call_function[target=builtins.getattr](args = (%getattr_22, elem), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %getattr_27), kwargs = {})
    return (add_2,)""",
        )
        ep = export(m, (ref_x,))
        self.assertTrue(torch.allclose(ep.module()(ref_x), ref_out))

    def test_subclass_nested_attr_access_const_metadata_not_top_level(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4))
                self.p2 = torch.nn.Parameter(
                    TwoTensor(
                        ConstantExtraMetadataTensor(torch.ones(3, 4)),
                        ConstantExtraMetadataTensor(torch.ones(3, 4)),
                    )
                )

            def forward(self, x):
                res = 2 * self.p1 + self.p2
                res2 = res + res.a.elem + res.b.constant_attribute
                return x + res2.a.elem

        m = Foo()
        ref_x = torch.randn(3, 4)
        ref_out = m(ref_x)
        ep_training = torch.export.export(m, (ref_x,), strict=False)
        self.assertExpectedInline(
            str(ep_training.graph).strip(),
            """\
graph():
    %p_p1 : [num_users=1] = placeholder[target=p_p1]
    %p_p2 : [num_users=1] = placeholder[target=p_p2]
    %x : [num_users=1] = placeholder[target=x]
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%p_p1, 2), kwargs = {})
    %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %p_p2), kwargs = {})
    %access_subclass_inner_tensor_default_18 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%add, a), kwargs = {})
    %access_subclass_inner_tensor_default_21 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%access_subclass_inner_tensor_default_18, elem), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %access_subclass_inner_tensor_default_21), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, 4), kwargs = {})
    %access_subclass_inner_tensor_default_25 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%add_2, a), kwargs = {})
    %access_subclass_inner_tensor_default_28 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%access_subclass_inner_tensor_default_25, elem), kwargs = {})
    %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %access_subclass_inner_tensor_default_28), kwargs = {})
    return (add_3,)""",
        )
        ep = export(m, (ref_x,))
        self.assertTrue(torch.allclose(ep.module()(ref_x), ref_out))

    def test_subclass_nested_attr_access_complicated_metadata(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4))
                self.p2 = torch.nn.Parameter(
                    ConstantExtraMetadataTensor(
                        ConstantExtraMetadataTensor(torch.ones(3, 4)),
                    )
                )

            def forward(self, x):
                res = x + 2 * self.p1 + self.p2
                return res.elem.elem + self.p2.get_complicated_metadata().foo

        m = Foo()
        ref_x = torch.randn(3, 4)
        ref_out = m(ref_x)
        ep_training = torch.export.export(m, (ref_x,), strict=False)
        self.assertExpectedInline(
            str(ep_training.graph).strip(),
            """\
graph():
    %p_p1 : [num_users=1] = placeholder[target=p_p1]
    %p_p2 : [num_users=1] = placeholder[target=p_p2]
    %x : [num_users=1] = placeholder[target=x]
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%p_p1, 2), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x, %mul), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %p_p2), kwargs = {})
    %access_subclass_inner_tensor_default_10 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%add_1, elem), kwargs = {})
    %access_subclass_inner_tensor_default_13 : [num_users=1] = call_function[target=torch.ops.export.access_subclass_inner_tensor.default](args = (%access_subclass_inner_tensor_default_10, elem), kwargs = {})
    %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%access_subclass_inner_tensor_default_13, 4), kwargs = {})
    return (add_2,)""",
        )
        ep = export(m, (ref_x,))
        self.assertTrue(torch.allclose(ep.module()(ref_x), ref_out))

    def test_real_tensor_errors_on_aliasing_custom_op(self):
        @torch.library.custom_op("export::foo_alias", mutates_args={})
        def foo(x: torch.Tensor) -> torch.Tensor:
            return x

        class Foo(torch.nn.Module):
            def forward(self, x):
                return torch.ops.export.foo_alias(x) * 2

        model = Foo()
        inputs = (torch.randn(4, 4),)
        error_type = (
            RuntimeError
            if is_non_strict_test(self._testMethodName)
            else torch._dynamo.exc.TorchRuntimeError
        )
        with self.assertRaisesRegex(
            error_type,
            (
                r"The output of this custom operator \(1\) must not also be an input "
                r"to this custom operator and \(2\) may not alias any inputs to this "
                r"custom operator or other returns"
            ),
        ):
            with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
                ep = export(model, inputs)

    def test_real_tensor_bool_cast(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return bool(x.eq(0.1).any())

        model = Foo()
        inputs = (torch.randn(64),)
        with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
            ep = export(model, inputs, strict=False)

    def test_is_nonzero(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return torch.is_nonzero(x)

        def _long_tensor(nz):
            return torch.full((), int(nz))

        def _float_tensor(nz):
            return torch.full((), int(nz), dtype=torch.float32)

        def _bool_tensor(nz):
            return torch.full((), int(nz)).bool()

        mod = Foo()
        for _tensor in [
            _long_tensor,
            _float_tensor,
            _bool_tensor,
            # local_scalar_dense on complex NYI for fake tensors
        ]:
            with torch._functorch.config.patch(fake_tensor_propagate_real_tensors=True):
                for nz in [True, False]:
                    sample_input = _tensor(nz=nz)
                    ep = export(mod, (sample_input,), strict=False)
                    self.assertEqual(ep.module()(sample_input), nz)

    def test_export_script_module(self):
        class Foo(torch.nn.Module):
            def forward(self, rv: torch.Tensor, t: torch.Tensor):
                i = t.item()
                return rv + i

        foo = Foo()
        foo_script = torch.jit.script(foo)
        inp = (torch.zeros(3, 4), torch.tensor(7))

        with self.assertRaisesRegex(
            ValueError, "Exporting a ScriptModule is not supported"
        ):
            export(foo_script, inp)

        from torch._export.converter import TS2EPConverter

        TS2EPConverter(foo_script, inp).convert()

    def test_dim_auto_and_dim(self):
        # test basic Dims
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x - y

        inputs = (torch.randn(4, 4), torch.randn(4, 4))
        shapes = {
            "x": (Dim.AUTO, Dim("d1", min=3)),
            "y": (Dim("d0", max=8), Dim.DYNAMIC),
        }
        ep = export(Foo(), inputs, dynamic_shapes=shapes)
        x, y = [node for node in ep.graph.nodes if node.op == "placeholder"]
        self.assertEqual((s0 := x.meta["val"].shape[0]), y.meta["val"].shape[0])
        self.assertEqual((s1 := x.meta["val"].shape[1]), y.meta["val"].shape[1])
        vr0 = ep.range_constraints[s0.node.expr]
        vr1 = ep.range_constraints[s1.node.expr]
        self.assertEqual([vr0.upper, vr1.lower], [8, 3])

        # test derived Dims
        class Bar(torch.nn.Module):
            def forward(self, x, y, z):
                return x + y[1::3] + z

        inputs = (torch.randn(4), torch.randn(13), torch.randn(4))
        dx = Dim("dx", min=2, max=10)
        shapes = {
            "x": (dx,),
            "y": (3 * dx + 1,),
            "z": (Dim.AUTO,),
        }
        ep = export(Bar(), inputs, dynamic_shapes=shapes)
        x, y, z = [node for node in ep.graph.nodes if node.op == "placeholder"]
        self.assertEqual((s0 := x.meta["val"].shape[0]), z.meta["val"].shape[0])
        expr = y.meta["val"].shape[0]
        free_symbols = expr.node.expr.free_symbols
        self.assertEqual(len(free_symbols), 1)
        self.assertEqual(next(iter(free_symbols)), s0.node.expr)

        # test specialization still complains
        inputs = (torch.randn(4), torch.randn(4))
        shapes = {
            "x": (Dim.STATIC,),
            "y": (Dim("dy"),),
        }
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r"You marked.*but your code specialized it to be a constant.*"
            r"If you're using Dim.DYNAMIC, replace it with either Dim.STATIC or Dim.AUTO",
        ):
            export(Foo(), inputs, dynamic_shapes=shapes)

    def test_issue_157289(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()

            def forward(self, causal_mask, fill_value):
                causal_mask = causal_mask.clone()
                mask_length = fill_value.shape[-1]
                causal_mask[:, :, :, :mask_length] = fill_value
                return causal_mask

        causal_mask = torch.randn(2, 2, 3, 4)
        fill_value = torch.randn(2, 2, 3, 3)
        dynamic_shapes = {
            "causal_mask": {3: Dim("M")},
            "fill_value": {3: Dim("N")},
        }
        ep = export(
            MyModule(), (causal_mask, fill_value), dynamic_shapes=dynamic_shapes
        )
        if not is_training_ir_test(self._testMethodName) and not is_retracebility_test(
            self._testMethodName
        ):
            self.assertExpectedInline(
                str(ep.graph_module.code).strip(),
                """\
def forward(self, causal_mask, fill_value):
    sym_size_int_4 = torch.ops.aten.sym_size.int(fill_value, 3)
    clone = torch.ops.aten.clone.default(causal_mask);  causal_mask = None
    slice_1 = torch.ops.aten.slice.Tensor(clone, 3, 0, sym_size_int_4);  sym_size_int_4 = None
    copy_ = torch.ops.aten.copy_.default(slice_1, fill_value);  slice_1 = fill_value = copy_ = None
    return (clone,)""",
            )
            decomposed_ep = ep.run_decompositions()
            self.assertExpectedInline(
                str(decomposed_ep.graph_module.code).strip(),
                """\
def forward(self, causal_mask, fill_value):
    sym_size_int_5 = torch.ops.aten.sym_size.int(fill_value, 3)
    clone = torch.ops.aten.clone.default(causal_mask);  causal_mask = None
    slice_1 = torch.ops.aten.slice.Tensor(clone, 3, 0, sym_size_int_5)
    copy = torch.ops.aten.copy.default(slice_1, fill_value);  slice_1 = fill_value = None
    slice_scatter = torch.ops.aten.slice_scatter.default(clone, copy, 3, 0, sym_size_int_5);  clone = copy = sym_size_int_5 = None
    return (slice_scatter,)""",
            )

    def test_dim_dynamic_specialization(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x + 2

        # 0/1 specialization
        with self.assertRaisesRegex(
            ValueError,
            r"Received user-specified dim hint Dim.DYNAMIC.*"
            r"but export 0/1 specialized due to hint of 0 for dimension "
            r"inputs\['x'\]\.shape\[0\](.*\n)*.*"
            r"Received user-specified dim hint Dim.DYNAMIC.*"
            r"but export 0/1 specialized due to hint of 1 for dimension "
            r"inputs\['x'\]\.shape\[1\].*",
        ):
            export(
                Foo(),
                (torch.randn(0, 1),),
                dynamic_shapes={
                    "x": {0: Dim.DYNAMIC, 1: Dim.DYNAMIC},
                },
            )

        class Bar(torch.nn.Module):
            def forward(self, x):
                assert x.shape[0] <= 32
                return x + 2

        # static specialization
        with self.assertRaisesRegex(
            ValueError,
            r"Received user-specified dim hint Dim.DYNAMIC.*"
            r"but tracing inferred a static shape of 32 for dimension "
            r"inputs\['x'\]\.shape\[0\](.*\n)*.*",
        ):
            export(
                Bar(),
                (torch.randn(32),),
                dynamic_shapes={
                    "x": {0: Dim.DYNAMIC(min=32)},
                },
            )

    def test_unbacked_slice_forward(self):
        class Foo(torch.nn.Module):
            def forward(self, x, xs):
                u0, u1 = xs.tolist()
                out = x[u0:u1]
                return out

        x = torch.randn(10)
        idxs = torch.tensor([3, 6])
        mod = Foo()
        ep = export(mod, (x, idxs))
        for xs in [
            idxs,
            torch.tensor([-9, -1]),
            torch.tensor([-10000, 10000]),
            torch.tensor([0, -10]),
        ]:
            self.assertTrue(torch.allclose(ep.module()(x, xs), mod(x, xs)))

        # check unbacked bindings
        # should be 4 symbols: u0, u1, output size, output storage offset
        bound_unbacked = set()
        for node in ep.graph.nodes:
            bound_unbacked |= node.meta.get("unbacked_bindings", {}).keys()
        self.assertEqual(len(bound_unbacked), 4)

    def test_dim_hint_ranges(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        inputs = (
            torch.randn(6, 4),
            torch.randn(6, 4),
        )
        shapes = {
            "x": (Dim.AUTO(min=4), Dim.AUTO),
            "y": (Dim.DYNAMIC(max=16), Dim.AUTO(max=32)),
        }
        ep = export(Foo(), inputs, dynamic_shapes=shapes)
        ep.module()(torch.randn(8, 5), torch.randn(8, 5))
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: x.size()[0] >= 4"),
        ):
            # expected >= 4, but got 3
            ep.module()(torch.randn(3, 5), torch.randn(3, 5))
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: x.size()[0] <= 16"),
        ):
            # expected <= 16, but got 17
            ep.module()(torch.randn(17, 5), torch.randn(17, 5))
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: x.size()[1] <= 32"),
        ):
            # expected <= 32, but got 33
            ep.module()(torch.randn(9, 33), torch.randn(9, 33))

    def test_dim_hint_range_violations(self):
        class Foo(torch.nn.Module):
            def forward(self, xs):
                x, y = xs["data"][0]
                assert y.shape[0] <= 32
                return x[6:], y + 2

        x, y = torch.randn(8), torch.randn(8)

        # conflict with lower bound
        shapes = torch.export.ShapesCollection()
        shapes[x] = [Dim.DYNAMIC(max=5)]
        with self.assertRaisesRegex(
            ValueError,
            r"Received user-specified .* \[None, 5\], conflicting with the inferred .*"
            r"\[8, int_oo\],.* for inputs\['xs'\]\['data'\]\[0\]\[0\]\.shape\[0\]",
        ):
            export(Foo(), ({"data": [[x, y]]},), dynamic_shapes=shapes)

        # conflict with upper bound
        shapes = torch.export.ShapesCollection()
        shapes[y] = [Dim.AUTO(min=48, max=62)]
        with self.assertRaisesRegex(
            ValueError,
            r"Received user-specified .* \[48, 62\], conflicting with the inferred .*"
            r"\[2, 32\],.* for inputs\['xs'\]\['data'\]\[0\]\[1\]\.shape\[0\]",
        ):
            export(Foo(), ({"data": [[x, y]]},), dynamic_shapes=shapes)

        class Bar(torch.nn.Module):
            def forward(self, x):
                return x + 2

        # conflict with static range
        shapes = {"x": [Dim.STATIC(min=6, max=8)]}
        with self.assertRaisesRegex(
            ValueError,
            r"Received user-specified .* \[6, 8\], conflicting with the inferred .*"
            r"\[4, 4\],.* for inputs\['x'\].shape\[0\]",
        ):
            export(Bar(), (torch.randn(4),), dynamic_shapes=shapes)

        # multiple conflicts
        class Moo(torch.nn.Module):
            def forward(self, x, y):
                assert x.shape[0] <= 32
                assert y.shape[0] >= 128
                return x + 2, y + 2

        inps = (torch.randn(16), torch.randn(256))
        shapes = {
            "x": (Dim.DYNAMIC(min=33),),
            "y": (Dim.DYNAMIC(max=127),),
        }
        with self.assertRaisesRegex(
            ValueError,
            r"Received user-specified .* \[33, None\], conflicting with the inferred .*"
            r"\[2, 32\],.* for inputs\['x'\].shape\[0\](.*\n)*.*"
            r"Received user-specified .* \[None, 127\], conflicting with the inferred .*"
            r"\[128, int_oo\],.* for inputs\['y'\].shape\[0\]",
        ):
            export(Moo(), inps, dynamic_shapes=shapes)

    def test_torch_fn(self):
        class M1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.linear(x)
                x = self.linear(x)
                x = self.relu(x)
                x = x + x
                return x

        ep1 = export(M1(), (torch.randn(3, 3),)).run_decompositions()
        expected_result = [
            ("linear_1", "builtin_function_or_method.linear"),
            ("linear_1", "builtin_function_or_method.linear"),
            ("linear_2", "builtin_function_or_method.linear"),
            ("linear_2", "builtin_function_or_method.linear"),
            ("relu_1", "function.relu"),
            ("add_1", "method_descriptor.add"),
        ]
        actual_result = []
        for i, node in enumerate(ep1.graph.nodes):
            if node.op == "call_function":
                actual_result.append(node.meta.get("torch_fn"))
        self.assertEqual(actual_result, expected_result)

        class M2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x, weight, bias):
                x = torch.nn.functional.linear(x, weight, bias)
                x = torch.nn.functional.relu(x)
                x = torch.add(x, x)
                return x

        ep2 = export(
            M2(), (torch.randn(3, 3), torch.randn(3, 3), torch.randn(3))
        ).run_decompositions()
        expected_result = [
            ("linear_1", "builtin_function_or_method.linear"),
            ("linear_1", "builtin_function_or_method.linear"),
            ("relu_1", "function.relu"),
            ("add_1", "builtin_function_or_method.add"),
        ]
        actual_result = []
        for i, node in enumerate(ep2.graph.nodes):
            if node.op == "call_function":
                actual_result.append(node.meta.get("torch_fn"))
        self.assertEqual(actual_result, expected_result)

    def test_hoo_inline_users_issue(self):
        # This came from an issue where replace_with_hop passes would inline subgraphs,
        # and mess up node.users for nodes present in multiple subgraphs (e.g. _x in SetGradCase
        # below, since it's used in both set_grad_enabled HOO modules).
        # This checks that node.users and node.args are in correspondence.
        def check_users_for_graph(graph):
            def _tuple_contains(_tuple, val):
                # check nested, since output node args have format ((x, y, ...),)
                return any(
                    _tuple_contains(x, val) if isinstance(x, tuple) else x == val
                    for x in _tuple
                )

            for node in graph.nodes:
                # check node.users
                for user in node.users.keys():
                    assert _tuple_contains(user.args, node)
                # check node.args
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node):
                        assert _tuple_contains(arg.users, node)

        # check set grad enabled
        class SetGradCase(torch.nn.Module):
            def forward(self, x):
                _x = x.shape[0] + 2
                _xx = _x + 2
                with torch.no_grad():
                    y = _x * 4
                return _xx, y

        ep = export(
            SetGradCase(),
            (torch.randn(6),),
            dynamic_shapes={"x": (Dim("dx"),)},
            strict=False,
        )
        check_users_for_graph(ep.graph)

    def test_export_custom_op_lib(self):
        ops_registered_before = set(torch.ops.mylib)

        # Assert warning for CompositeImplicitAutograd op
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("foo123(Tensor x) -> Tensor")
            lib.impl("foo123", lambda x: x.sin(), "CompositeImplicitAutograd")

        ops_registered_after = set(torch.ops.mylib)
        self.assertEqual(ops_registered_after, ops_registered_before)

    def test_export_preserve_linear_but_not_custom_op(self):
        table = torch.export.default_decompositions()
        del table[torch.ops.aten.linear.default]

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("foo123(Tensor x) -> Tensor")
            lib.impl("foo123", lambda x: x.sin(), "CompositeImplicitAutograd")

            class Bar(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(4, 4)

                def forward(self, x):
                    lin = self.linear(x)
                    return torch.ops.mylib.foo123(lin)

            x = torch.randn(4, 4)
            ep = export(Bar(), (x,)).run_decompositions(table)

            self.assertExpectedInline(
                str(ep.graph_module.code).strip(),
                """\
def forward(self, p_linear_weight, p_linear_bias, x):
    linear = torch.ops.aten.linear.default(x, p_linear_weight, p_linear_bias);  x = p_linear_weight = p_linear_bias = None
    sin = torch.ops.aten.sin.default(linear);  linear = None
    return (sin,)""",
            )

    def test_export_preserve_linear_at_aot_level(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                x = self.linear(x)
                return torch.ops.aten.chunk.default(x, 3, 0)

        ep = torch.export.export(Foo(), (torch.randn(3, 3),))
        decomp_table = default_decompositions()
        del decomp_table[torch.ops.aten.linear.default]
        ep = ep.run_decompositions(decomp_table)

        gm = ep.graph_module
        # linear is CompositeImplicitAutograd functional op so we should preserve it
        # chunk is CompositeImplicitAutograd non-functional op we decompose.
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, p_linear_weight, p_linear_bias, x):
    linear = torch.ops.aten.linear.default(x, p_linear_weight, p_linear_bias);  x = p_linear_weight = p_linear_bias = None
    split_with_sizes = torch.ops.aten.split_with_sizes.default(linear, [1, 1, 1]);  linear = None
    getitem = split_with_sizes[0]
    getitem_1 = split_with_sizes[1]
    getitem_2 = split_with_sizes[2];  split_with_sizes = None
    return (getitem, getitem_1, getitem_2)""",
        )

    def test_export_cond_preserve_torch_fn_for_subgraphs(self):
        class MySubModule(torch.nn.Module):
            def foo(self, x):
                return x.cos()

            def forward(self, x):
                return self.foo(x)

        class CondBranchClassMethod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.subm = MySubModule()

            def bar(self, x):
                return x.sin()

            def forward(self, x):
                return cond(x.sum() <= 2, self.subm.forward, self.bar, [x])

        example_inputs = (torch.randn(1, 3, 3, 3),)
        m = CondBranchClassMethod()
        m.eval()
        gm = export(m, example_inputs).module()

        actual_torch_fns = []
        for mod in gm.modules():
            if hasattr(mod, "graph"):
                for node in mod.graph.nodes:
                    if node.name in {"sin", "cos"}:
                        torch_fn = node.meta.get("torch_fn")
                        print(torch_fn)
                        actual_torch_fns.append(torch_fn)
        exp_torch_fns = [
            ("cos_1", "method_descriptor.cos"),
            ("sin_1", "method_descriptor.sin"),
        ]
        self.assertEqual(actual_torch_fns, exp_torch_fns)

    def test_is_exporting(self):
        class Mod(torch.nn.Module):
            def forward(self, pred, x):
                def f(x):
                    return x.sin() if torch.compiler.is_exporting() else x.cos()

                y = f(x)

                def true_fn(x):
                    return f(x) - 1 if torch.compiler.is_exporting() else f(x) + 1

                def false_fn(x):
                    return f(x) + 1 if torch.compiler.is_exporting() else f(x) - 1

                return torch.cond(pred, true_fn, false_fn, (x,)) * y

        ep = export(
            Mod(),
            (
                torch.tensor(False),
                torch.randn(3, 4),
            ),
        )
        FileCheck().check_count("torch.ops.aten.sin", 1, exactly=True).run(
            ep.graph_module.code
        )
        FileCheck().check_count("torch.ops.higher_order.cond", 1, exactly=True).run(
            ep.graph_module.code
        )

        # True graph should contain sin and sub
        FileCheck().check_count("torch.ops.aten.sub", 1, exactly=True).run(
            ep.graph_module.true_graph_0.code
        )
        FileCheck().check_count("torch.ops.aten.sin", 1, exactly=True).run(
            ep.graph_module.true_graph_0.code
        )

        # False graph should contain sin and add
        FileCheck().check_count("torch.ops.aten.add", 1, exactly=True).run(
            ep.graph_module.false_graph_0.code
        )
        FileCheck().check_count("torch.ops.aten.sin", 1, exactly=True).run(
            ep.graph_module.false_graph_0.code
        )

    def test_ends_of_bounds_oblivious(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.zeros(10))

            def forward(self, x, y):
                self.buf[0 : x.shape[0]] = x
                return x + 2, y[:, ::1]

        inps = (torch.randn(10), torch.randn(32, 36))
        dynamic_shapes = {
            "x": {0: Dim("dx", min=1, max=10)},
            "y": {0: Dim("dy0"), 1: Dim("dy1")},
        }
        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            ep = export(Foo(), inps, dynamic_shapes=dynamic_shapes)
        ep.module()(torch.randn(9), torch.randn(4, 4))
        ep.module()(torch.randn(1), torch.randn(1, 1))

    def test_colin_unbacked_backed_vr_sub(self):
        class Model(torch.nn.Module):
            def forward(self, a, b, c):
                nz = torch.nonzero(a)
                ones = a.new_ones([nz.size(0), b.size(0)])
                torch._check(ones.size(0) >= 1)
                equals = torch.add(ones, c)
                return equals

        model = Model()
        example_inputs = (
            torch.ones(64),
            torch.randn(32),
            torch.randn(64, 32),
        )
        dynamic_shapes = {"a": None, "b": None, "c": (Dim.DYNAMIC, Dim.STATIC)}
        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            ep = export(model, example_inputs, dynamic_shapes=dynamic_shapes)

        # check lower bound
        for sym, vr in ep.range_constraints.items():
            if str(sym) in ["u0", "s0"]:
                self.assertEqual(vr.lower, 1)

    @testing.expectedFailureStrictV2
    def test_duplicate_modules_with_non_persistent_buffers(self):
        class FooWithBuf(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("buf", torch.randn(4), persistent=False)

            def forward(self, x):
                return x + self.buf

        class BarWithFoo(torch.nn.Module):
            def __init__(self, foo):
                super().__init__()
                self.foo = foo

            def forward(self, x):
                return self.foo(x)

        class ModWith2Bars(torch.nn.Module):
            def __init__(self):
                super().__init__()
                foo = FooWithBuf()
                self.b1 = BarWithFoo(foo)
                self.b2 = BarWithFoo(foo)

            def forward(self, x):
                return self.b1(x) + self.b2(x)

        mod = ModWith2Bars()
        inputs = (torch.randn(4),)
        ep = export(mod, inputs)
        self.assertTrue(torch.allclose(ep.module()(*inputs), mod(*inputs)))

    def test_derived_dim_basic(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x + y[1:]

        foo = Foo()

        x, y = torch.randn(5), torch.randn(6)
        dimx = torch.export.Dim("dimx", min=3, max=6)

        dimy = torch.export.Dim("dimy", min=4, max=7)  # doesn't work
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Constraints violated \\(dimy\\)!(.*\n)*.*"
                "The values of dimy.*must always be related to the values of dimx.*by.*(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "dimy = dimx \\+ 1"
            ),
        ):
            export(
                foo,
                (x, y),
                dynamic_shapes=({0: dimx}, {0: dimy}),
            )

        dimy = dimx * 2  # doesn't work
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Expected input.*size.* to be equal to 2\\*dimx, where dimx = 5, but got 6",
        ):
            export(
                foo,
                (x, y),
                dynamic_shapes=({0: dimx}, {0: dimy}),
            )

        dimy = dimx + 1  # works
        ep = export(
            foo,
            (x, y),
            dynamic_shapes=({0: dimx}, {0: dimy}),
        )
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: x.size()[0] == -1 + y.size()[0]"),
        ):
            # expected 5, but got 6
            ep.module()(torch.randn(4), torch.randn(6))

        self.assertEqual(ep.module()(torch.randn(4), torch.randn(5)).size()[0], 4)

    def test_derived_dim_nested(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x + y[1::2]

        foo = Foo()

        x, y = torch.randn(5), torch.randn(11)
        dimx = torch.export.Dim("dimx", min=3, max=6)
        dimy = dimx * 2 + 1  # works
        ep = export(
            foo,
            (x, y),
            dynamic_shapes=({0: dimx}, {0: dimy}),
        )
        self.assertEqual(ep.module()(torch.randn(4), torch.randn(9)).size()[0], 4)

        class Foo(torch.nn.Module):
            def forward(self, z, y):
                return z[1:] + y[1::2]

        foo = Foo()

        z, y = torch.randn(6), torch.randn(11)

        dimz = dimx
        dimy = dimx * 2 - 1  # works
        ep = export(
            foo,
            (z, y),
            dynamic_shapes=({0: dimz}, {0: dimy}),
        )
        self.assertEqual(ep.module()(torch.randn(5), torch.randn(9)).size()[0], 4)

        dimz = dimx + 1
        dimy = dimx * 2 - 1  # doesn't work

        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Expected input.*size.*to be equal to 2\\*dimx - 1, where dimx = 5, but got 11",
        ):
            export(
                foo,
                (z, y),
                dynamic_shapes=({0: dimz}, {0: dimy}),
            )

        dimy = dimx * 2 + 1  # works
        ep = export(
            foo,
            (z, y),
            dynamic_shapes=({0: dimz}, {0: dimy}),
        )
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: z.size()[0] <= 7"),
        ):
            # expected <= 7, but got 8
            ep.module()(torch.randn(8), torch.randn(15))
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: -1 + 2 * z.size()[0] == y.size()[0]"),
        ):
            # expected 9, but got 8
            ep.module()(torch.randn(5), torch.randn(8))

        self.assertEqual(ep.module()(torch.randn(5), torch.randn(9)).size()[0], 4)

    def test_derived_dim_integer(self):
        class Foo(torch.nn.Module):
            def forward(self, w):
                if w.shape[0] % 2 == 0:
                    return w[::2]
                else:
                    return w[1:-1:2]

        foo = Foo()

        w = torch.randn(10)
        dimx = torch.export.Dim("dimx", min=3, max=6)
        dimw = dimx * 2 + 1  # doesn't work
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Expected shape.*= 10 of input Tensor to be "
            "of the form 2\\*dimx \\+ 1, where dimx is an integer",
        ):
            export(
                foo,
                (w,),
                dynamic_shapes=({0: dimw},),
            )

        dimw = dimx * 2  # works
        ep = export(
            foo,
            (w,),
            dynamic_shapes=({0: dimw},),
        )
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: w.size()[0] % 2 == 0"),
        ):
            # expected 2*..., got 9
            ep.module()(torch.randn(9))

        self.assertEqual(ep.module()(torch.randn(8)).size()[0], 4)
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: w.size()[0] <= 12"),
        ):
            # expected <= 12, but got 14
            ep.module()(torch.randn(14))

    def test_derived_dim_repeat_derived(self):
        class Foo(torch.nn.Module):
            def forward(self, u, v):
                return u[::2] + v[::2]

        foo = Foo()

        u, v = torch.randn(10), torch.randn(10)
        dimx = torch.export.Dim("dimx", min=3, max=6)
        dimw = dimx * 2  # works
        ep = export(
            foo,
            (u, v),
            dynamic_shapes=({0: dimw}, {0: dimw}),
        )
        self.assertEqual(ep.module()(torch.randn(8), torch.randn(8)).size()[0], 4)

    def test_derived_dim_out_of_order(self):
        dimy = torch.export.Dim("dimy", min=5, max=7)
        dimx = dimy - 1  # out of order, effectively dimy = dimx + 1
        dimz = dimy + 1  # out of order, effectively dimz = dimx + 2

        class Foo(torch.nn.Module):
            def forward(self, x, y, z):
                return x + y[1:] + z[2:]

        foo = Foo()

        u, v, w = torch.randn(5), torch.randn(6), torch.randn(7)
        ep = export(
            foo,
            (u, v, w),
            dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimz}),
        )
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: z.size()[0] >= 6"),
        ):
            # expected 8, but got 5
            ep.module()(torch.randn(6), torch.randn(7), torch.randn(5))

        self.assertEqual(
            ep.module()(torch.randn(6), torch.randn(7), torch.randn(8)).size()[0], 6
        )

    def test_derived_dim_out_of_order_repeat_derived(self):
        dimy = torch.export.Dim("dimy", min=5, max=7)
        dimx = dimy - 1  # out of order, effectively dimy = dimx + 1
        dimz = dimy + 1  # out of order, effectively dimz = dimx + 2
        dimx1 = dimx
        dimx2 = dimz - 2  # works, effectively = dimx

        class Foo(torch.nn.Module):
            def forward(self, x, y, z, x1, x2):
                return x + y[1:] + z[2:] + x1 + x2

        foo = Foo()

        u, v, w, u1, u2 = (
            torch.randn(5),
            torch.randn(6),
            torch.randn(7),
            torch.randn(5),
            torch.randn(5),
        )
        ep = export(
            foo,
            (u, v, w, u1, u2),
            dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimz}, {0: dimx1}, {0: dimx2}),
        )
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: x2.size()[0] == x.size()[0]"),
        ):
            # expected 6, but got 5
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(8),
                torch.randn(6),
                torch.randn(5),
            )

        self.assertEqual(
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(8),
                torch.randn(6),
                torch.randn(6),
            ).size()[0],
            6,
        )

        ep = export(
            foo,
            (u, v, w, u, u),  # reused inputs
            dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimz}, {0: dimx1}, {0: dimx2}),
        )
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: x2.size()[0] == x.size()[0]"),
        ):
            # expected 6, but got 5
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(8),
                torch.randn(6),
                torch.randn(5),
            )

        self.assertEqual(
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(8),
                torch.randn(6),
                torch.randn(6),
            ).size()[0],
            6,
        )

    def test_specialize_derived_dim_roots(self):
        # dim & derived dim both specialize
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x.reshape([-1]) + y

        dy = Dim("dy", min=6)
        x, y = torch.randn(6, 2), torch.randn(12)
        dynamic_shapes = {
            "x": (dy - 6, 2),
            "y": (dy,),
        }
        try:
            export(Foo(), (x, y), dynamic_shapes=dynamic_shapes)
            raise Exception(
                "export() call should have failed with dynamic shapes error."
            )
        except torch._dynamo.exc.UserError as exc:
            expected_error_msg = (
                "Specializations unexpectedly required \(dy\)!(.*\n)*.*"
                ".*solving the guards generated for dy - 6.*resulted in a specialized value of 6(.*\n)*.*"
                "Suggested fixes(.*\n)*.*"
                ".*dy = 12(.*\n)*.*"
            )
            self.assertTrue(re.search(expected_error_msg, exc.args[0]) is not None)
            self.assertTrue(
                "dy - 6 = 6" not in exc.args[0]
            )  # don't suggest fix for non-root dim

    @unittest.skip("See https://github.com/pytorch/pytorch/issues/135759")
    def test_keep_composite_ops_invalid(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                x = self.linear(x)
                return torch.ops.aten.chunk.default(x, 3, 0)

        def _(*args, **kwargs):
            return NotImplemented

        with self.assertWarnsRegex(UserWarning, "The op aten.chunk.default"):
            _ = torch.export.export(
                Foo(),
                (torch.randn(3, 3),),
            ).run_decompositions({torch.ops.aten.chunk.default: _})

        with self.assertWarnsRegex(UserWarning, "The op aten.sym_size.default"):
            _ = torch.export.export(
                Foo(),
                (torch.randn(3, 3),),
            ).run_decompositions({torch.ops.aten.sym_size.default: _})

        with self.assertWarnsRegex(
            UserWarning,
            "The op aten.native_batch_norm.default",
        ):
            _ = torch.export.export(
                Foo(),
                (torch.randn(3, 3),),
            ).run_decompositions({torch.ops.aten.native_batch_norm.default: _})

    def test_keep_composite_ops_linear_convd(self):
        class MyLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.randn(20, 98)
                self.bias = torch.randn(20)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.conv1d = torch.nn.Conv1d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x, y):
                x_conv = self.conv(x)
                y_conv_1d = self.conv1d(y)
                x_linear = self.linear(x_conv)
                return x_linear.cos() + y_conv_1d.sum()

        ep = torch.export.export(
            Foo(), (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50))
        )
        ep_has_linear_convd = ep.run_decompositions({})

        self.assertExpectedInline(
            str(ep_has_linear_convd.graph_module.code).strip(),
            """\
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, c_linear_weight, c_linear_bias, x, y):
    conv2d = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias);  x = p_conv_weight = p_conv_bias = None
    conv1d = torch.ops.aten.conv1d.default(y, p_conv1d_weight, p_conv1d_bias);  y = p_conv1d_weight = p_conv1d_bias = None
    linear = torch.ops.aten.linear.default(conv2d, c_linear_weight, c_linear_bias);  conv2d = c_linear_weight = c_linear_bias = None
    cos = torch.ops.aten.cos.default(linear);  linear = None
    sum_1 = torch.ops.aten.sum.default(conv1d);  conv1d = None
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    return (add,)""",
        )

        decomp_table = default_decompositions()
        del decomp_table[torch.ops.aten.conv2d.default]
        del decomp_table[torch.ops.aten.conv1d.default]

        ep_has_convd = ep.run_decompositions(decomp_table=decomp_table)
        self.assertExpectedInline(
            str(ep_has_convd.graph_module.code).strip(),
            """\
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, c_linear_weight, c_linear_bias, x, y):
    conv2d = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias);  x = p_conv_weight = p_conv_bias = None
    conv1d = torch.ops.aten.conv1d.default(y, p_conv1d_weight, p_conv1d_bias);  y = p_conv1d_weight = p_conv1d_bias = None
    view = torch.ops.aten.view.default(conv2d, [31680, 98]);  conv2d = None
    permute = torch.ops.aten.permute.default(c_linear_weight, [1, 0]);  c_linear_weight = None
    addmm = torch.ops.aten.addmm.default(c_linear_bias, view, permute);  c_linear_bias = view = permute = None
    view_1 = torch.ops.aten.view.default(addmm, [20, 33, 48, 20]);  addmm = None
    cos = torch.ops.aten.cos.default(view_1);  view_1 = None
    sum_1 = torch.ops.aten.sum.dim_IntList(conv1d, []);  conv1d = None
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    return (add,)""",
        )
        decomp_table = default_decompositions()
        del decomp_table[torch.ops.aten.conv2d.default]

        ep_has_convd = ep_has_convd.run_decompositions(decomp_table=decomp_table)
        self.assertExpectedInline(
            str(ep_has_convd.graph_module.code).strip(),
            """\
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, c_linear_weight, c_linear_bias, x, y):
    conv2d = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias);  x = p_conv_weight = p_conv_bias = None
    convolution = torch.ops.aten.convolution.default(y, p_conv1d_weight, p_conv1d_bias, [1], [0], [1], False, [0], 1);  y = p_conv1d_weight = p_conv1d_bias = None
    view = torch.ops.aten.view.default(conv2d, [31680, 98]);  conv2d = None
    permute = torch.ops.aten.permute.default(c_linear_weight, [1, 0]);  c_linear_weight = None
    addmm = torch.ops.aten.addmm.default(c_linear_bias, view, permute);  c_linear_bias = view = permute = None
    view_1 = torch.ops.aten.view.default(addmm, [20, 33, 48, 20]);  addmm = None
    cos = torch.ops.aten.cos.default(view_1);  view_1 = None
    sum_1 = torch.ops.aten.sum.dim_IntList(convolution, []);  convolution = None
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    return (add,)""",
        )

    def test_keep_composite_ops_linear_convd_for_training_ir(self):
        class MyLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Buffer(torch.randn(20, 98))
                self.bias = torch.nn.Buffer(torch.randn(20))

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.conv1d = torch.nn.Conv1d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x, y):
                x_conv = self.conv(x)
                y_conv_1d = self.conv1d(y)
                x_linear = self.linear(x_conv)
                return x_linear.cos() + y_conv_1d.sum()

        ep = torch.export.export(
            Foo(), (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50))
        )

        ep_has_linear_convd = ep.run_decompositions(
            decomp_table={},
        )

        self.assertExpectedInline(
            str(ep_has_linear_convd.graph_module.code).strip(),
            """\
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, b_linear_weight, b_linear_bias, x, y):
    conv2d = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias);  x = p_conv_weight = p_conv_bias = None
    conv1d = torch.ops.aten.conv1d.default(y, p_conv1d_weight, p_conv1d_bias);  y = p_conv1d_weight = p_conv1d_bias = None
    linear = torch.ops.aten.linear.default(conv2d, b_linear_weight, b_linear_bias);  conv2d = b_linear_weight = b_linear_bias = None
    cos = torch.ops.aten.cos.default(linear);  linear = None
    sum_1 = torch.ops.aten.sum.default(conv1d);  conv1d = None
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    return (add,)""",
        )

        decomp_table = default_decompositions()
        del decomp_table[torch.ops.aten.conv2d.default]
        del decomp_table[torch.ops.aten.conv1d.default]

        ep_has_convd = ep.run_decompositions(decomp_table=decomp_table)

        self.assertExpectedInline(
            str(ep_has_convd.graph_module.code).strip(),
            """\
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, b_linear_weight, b_linear_bias, x, y):
    conv2d = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias);  x = p_conv_weight = p_conv_bias = None
    conv1d = torch.ops.aten.conv1d.default(y, p_conv1d_weight, p_conv1d_bias);  y = p_conv1d_weight = p_conv1d_bias = None
    view = torch.ops.aten.view.default(conv2d, [31680, 98]);  conv2d = None
    permute = torch.ops.aten.permute.default(b_linear_weight, [1, 0]);  b_linear_weight = None
    addmm = torch.ops.aten.addmm.default(b_linear_bias, view, permute);  b_linear_bias = view = permute = None
    view_1 = torch.ops.aten.view.default(addmm, [20, 33, 48, 20]);  addmm = None
    cos = torch.ops.aten.cos.default(view_1);  view_1 = None
    sum_1 = torch.ops.aten.sum.dim_IntList(conv1d, []);  conv1d = None
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    return (add,)""",
        )

        decomp_table = default_decompositions()
        del decomp_table[torch.ops.aten.conv2d.default]
        ep_has_convd = ep_has_convd.run_decompositions(decomp_table=decomp_table)

        self.assertExpectedInline(
            str(ep_has_convd.graph_module.code).strip(),
            """\
def forward(self, p_conv_weight, p_conv_bias, p_conv1d_weight, p_conv1d_bias, b_linear_weight, b_linear_bias, x, y):
    conv2d = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias);  x = p_conv_weight = p_conv_bias = None
    convolution = torch.ops.aten.convolution.default(y, p_conv1d_weight, p_conv1d_bias, [1], [0], [1], False, [0], 1);  y = p_conv1d_weight = p_conv1d_bias = None
    view = torch.ops.aten.view.default(conv2d, [31680, 98]);  conv2d = None
    permute = torch.ops.aten.permute.default(b_linear_weight, [1, 0]);  b_linear_weight = None
    addmm = torch.ops.aten.addmm.default(b_linear_bias, view, permute);  b_linear_bias = view = permute = None
    view_1 = torch.ops.aten.view.default(addmm, [20, 33, 48, 20]);  addmm = None
    cos = torch.ops.aten.cos.default(view_1);  view_1 = None
    sum_1 = torch.ops.aten.sum.dim_IntList(convolution, []);  convolution = None
    add = torch.ops.aten.add.Tensor(cos, sum_1);  cos = sum_1 = None
    return (add,)""",
        )

    @unittest.skip("See https://github.com/pytorch/pytorch/issues/135759")
    def test_error_when_passing_mutating_primitive_op(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x.sin()

        ep = export(Foo(), (torch.ones(3, 3),))
        with self.assertWarnsRegex(
            UserWarning,
            "The op aten.index_put_.default",
        ):
            ep.run_decompositions({torch.ops.aten.index_put_.default: None})

    def test_export_cond_warns_constant_pred(self):
        class Mod(torch.nn.Module):
            def forward(self, pred, x):
                return torch.cond(pred, lambda x: x.sin(), lambda x: x.cos(), (x,))

        mod = Mod()
        with self.assertWarnsRegex(UserWarning, "Pred is a Python constant"):
            ep = export(mod, (True, torch.randn(3, 3)))

        nodes = ep.module().graph.find_nodes(
            op="call_function", target=torch.ops.aten.sin.default
        )
        self.assertEqual(len(nodes), 1)

    def test_export_custom_decomp_table_basic_pop(self):
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("foo123(Tensor x) -> Tensor")
            lib.impl("foo123", lambda x: x.sin(), "CompositeImplicitAutograd")

            lib.define("foo456(Tensor x) -> Tensor")
            lib.impl("foo456", lambda x: x.sin(), "CompositeImplicitAutograd")

            table = default_decompositions()
            # Since this table hasn't been materialized yet, we shouldn't error
            val = table.pop(torch.ops.mylib.foo123.default)
            self.assertIsNotNone(val)

            with self.assertRaisesRegex(KeyError, "mylib.foo123.default"):
                table.pop(torch.ops.mylib.foo123.default)

            val = table.pop(torch.ops.mylib.foo123.default, "HELLO")
            self.assertEqual(val, "HELLO")

            all_ops = set(k for k, v in table.items())
            self.assertTrue(table.has_materialized)
            # When we force materialize, torch.ops.mylib.foo123.default should have gone
            self.assertFalse(torch.ops.mylib.foo123.default in all_ops)
            self.assertTrue(torch.ops.mylib.foo456.default in all_ops)

    def test_export_custom_decomp_table_container_methods(self):
        # tests __len__
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            table = default_decompositions()
            length_before = len(table)
            lib.define("foo123(Tensor x) -> Tensor")
            lib.impl("foo123", lambda x: x.sin(), "CompositeImplicitAutograd")

            lib.define("foo456(Tensor x) -> Tensor")
            lib.impl("foo456", lambda x: x.sin(), "CompositeImplicitAutograd")

            table = default_decompositions()
            self.assertEqual(len(table) - length_before, 2)

        # tests __contains__
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("foo123(Tensor x) -> Tensor")
            lib.impl("foo123", lambda x: x.sin(), "CompositeImplicitAutograd")

            table = default_decompositions()
            self.assertTrue(torch.ops.mylib.foo123.default in table)
            del table[torch.ops.mylib.foo123.default]
            self.assertFalse(torch.ops.mylib.foo123.default in table)

        # Lot of ppl do
        # for op in all_ops:
        #     if op in table:
        #        del table[op]
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("foo123(Tensor x) -> Tensor")
            lib.impl("foo123", lambda x: x.sin(), "CompositeImplicitAutograd")

            table = default_decompositions()
            if torch.ops.mylib.foo123.default in table:
                del table[torch.ops.mylib.foo123.default]

            self.assertFalse(torch.ops.mylib.foo123.default in table)
            table.materialize()
            self.assertFalse(torch.ops.mylib.foo123.default in table)

    @testing.expectedFailureStrictV2
    def test_if_post_autograd_op_preserved(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x.sin() + x.sum()

        ep = export(Foo(), (torch.ones(3, 3),))
        decomp_table = default_decompositions()
        del decomp_table[torch.ops.aten.sum.default]
        ep_preserve_sum = ep.run_decompositions(decomp_table)

        # Even though we are decomposing to core aten which should make
        # sum into sum.dim_IntList, we explicitly marked it to not do that.
        self.assertExpectedInline(
            str(ep_preserve_sum.graph_module.code).strip(),
            """\
def forward(self, x):
    sin = torch.ops.aten.sin.default(x)
    sum_1 = torch.ops.aten.sum.default(x);  x = None
    add = torch.ops.aten.add.Tensor(sin, sum_1);  sin = sum_1 = None
    return (add,)""",
        )

        ep_no_preserve_sum = ep.run_decompositions()
        self.assertExpectedInline(
            str(ep_no_preserve_sum.graph_module.code).strip(),
            """\
def forward(self, x):
    sin = torch.ops.aten.sin.default(x)
    sum_1 = torch.ops.aten.sum.dim_IntList(x, []);  x = None
    add = torch.ops.aten.add.Tensor(sin, sum_1);  sin = sum_1 = None
    return (add,)""",
        )

    def test_set_grad_empty(self):
        class M(torch.nn.Module):
            def forward(self, x):
                with torch.no_grad():
                    x = x + 1
                    return x, None

        ep = export(M(), (torch.ones(3, 3),))
        inp = torch.randn(3, 3)
        self.assertTrue(torch.allclose(ep.module()(inp)[0], inp + 1))

    def test_set_grad_as_side_effect(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                torch._C._set_grad_enabled(False)
                return x.sum()

        before = torch.is_grad_enabled()
        ep = torch.export.export(Foo(), (torch.randn(4, 4),))
        after = torch.is_grad_enabled()
        self.assertEqual(before, after)

    def test_derived_dim_out_of_order_simplified(self):
        _dimz = torch.export.Dim("_dimz", min=6, max=8)
        dimy = _dimz - 1
        dimx = dimy - 1
        dimz = torch.export.Dim("dimz", min=6, max=8)  # doesn't work, should be = _dimz

        class Foo(torch.nn.Module):
            def forward(self, x, y, z):
                return x + y[1:] + z[2:]

        foo = Foo()
        u, v, w = torch.randn(5), torch.randn(6), torch.randn(7)
        try:
            export(
                foo,
                (u, v, w),
                dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimz}),
            )
        except torch._dynamo.exc.UserError as exc:
            expected_error_msg = (
                "Constraints violated \(dimz\)!(.*\n)*.*"
                "The values of dimz.*must always be related to the values of _dimz - 2.*by.*(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "dimz = _dimz"
            )
            self.assertTrue(re.search(expected_error_msg, exc.args[0]) is not None)
            # don't suggest fix for non-root dims, and no need to update root here
            self.assertTrue("_dimz - 2 = Dim(" not in exc.args[0])
            self.assertTrue("_dimz - 1 = _dimz - 1" not in exc.args[0])
            self.assertTrue("_dimz = Dim(" not in exc.args[0])

        dimz = dimx + 2  # works, effectively = _dimz
        ep = export(
            foo,
            (u, v, w),
            dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimz}),
        )
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: z.size()[0] >= 6"),
        ):
            # expected 8, but got 5
            ep.module()(torch.randn(6), torch.randn(7), torch.randn(5))

        self.assertEqual(
            ep.module()(torch.randn(6), torch.randn(7), torch.randn(8)).size()[0], 6
        )

    def test_simple_export_for_training(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        eager_model = Foo()
        ep_for_training = torch.export.export(eager_model, (torch.ones(2, 2),))
        self.assertExpectedInline(
            str(ep_for_training.graph_module.code).strip(),
            """\
def forward(self, p_linear_weight, p_linear_bias, x):
    linear = torch.ops.aten.linear.default(x, p_linear_weight, p_linear_bias);  x = p_linear_weight = p_linear_bias = None
    return (linear,)""",
        )
        gm = ep_for_training.module()
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    linear_weight = self.linear.weight
    linear_bias = self.linear.bias
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    linear = torch.ops.aten.linear.default(x, linear_weight, linear_bias);  x = linear_weight = linear_bias = None
    return pytree.tree_unflatten((linear,), self._out_spec)""",
        )

        self.assertTrue(
            torch.allclose(gm(torch.ones(2, 2)), eager_model(torch.ones(2, 2)))
        )

    def test_export_for_training_with_mutation(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.ones(4, 4))

            def forward(self, x):
                x.add_(5)
                self.buffer.add_(5)
                return x + self.buffer

        eager_model_for_export = Foo()
        eager_model_for_testing = Foo()
        ep_for_training = torch.export.export(
            eager_model_for_export, (torch.ones(4, 4),)
        )
        self.assertExpectedInline(
            str(ep_for_training.graph_module.code).strip(),
            """\
def forward(self, b_buffer, x):
    add_ = torch.ops.aten.add_.Tensor(x, 5);  x = None
    add__1 = torch.ops.aten.add_.Tensor(b_buffer, 5);  b_buffer = None
    add = torch.ops.aten.add.Tensor(add_, add__1);  add_ = add__1 = None
    return (add,)""",
        )
        gm = ep_for_training.module()
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    buffer = self.buffer
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    add_ = torch.ops.aten.add_.Tensor(x, 5);  x = None
    add__1 = torch.ops.aten.add_.Tensor(buffer, 5);  buffer = None
    add = torch.ops.aten.add.Tensor(add_, add__1);  add_ = add__1 = None
    return pytree.tree_unflatten((add,), self._out_spec)""",
        )

        self.assertTrue(
            torch.allclose(
                gm(torch.ones(4, 4)), eager_model_for_testing(torch.ones(4, 4))
            )
        )

    def test_export_for_training_with_dynamic_shapes(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.ones(4, 4))

            def forward(self, x):
                x.add_(5)
                self.buffer.add_(5)
                return x + self.buffer.sum()

        eager_model_for_export_training = Foo()
        eager_model_for_export_inference = Foo()
        eager_model_for_testing = Foo()
        ep_for_training = torch.export.export(
            eager_model_for_export_training,
            (torch.ones(4, 4),),
            dynamic_shapes=({0: Dim("x")},),
        )

        self.assertTrue(
            torch.allclose(
                ep_for_training.module()(torch.ones(2, 4)),
                eager_model_for_testing(torch.ones(2, 4)),
            )
        )

        ep_for_real = export(
            eager_model_for_export_inference,
            (torch.ones(4, 4),),
            dynamic_shapes=({0: Dim("x")},),
        )

        # Since symbol names are based on hash of source names, and these differ across inference and
        # training, we do range comparisons instead.
        self.assertEqual(
            str(ep_for_training.range_constraints.values()),
            str(ep_for_real.range_constraints.values()),
        )

    def test_unbacked_unsqueeze(self):
        class Unsqueeze(torch.nn.Module):
            def forward(self, xs):
                u0, u1 = xs.tolist()
                x = torch.zeros(1, u0 + u1).contiguous()
                return x.unsqueeze(-1)

        mod = Unsqueeze()
        x = torch.tensor([5, 6])
        ep = export(mod, (x,), strict=False)
        self.assertTrue(torch.allclose(mod(x), ep.module()(x)))
        x = torch.tensor([1, 2])
        self.assertTrue(torch.allclose(mod(x), ep.module()(x)))

    def test_export_for_training_with_container_type(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.ones(4, 4))

            def forward(self, container):
                x = container[0][0]
                y = container[0][1]
                x.add_(5)
                y.add_(5)
                return x + y + self.buffer.sum()

        eager_model = Foo()
        ep_for_training = torch.export.export(
            eager_model,
            ([torch.ones(4, 4), torch.ones(4, 4)],),
        )

        self.assertTrue(
            torch.allclose(
                ep_for_training.module()(
                    ([torch.ones(4, 4), torch.ones(4, 4)]),
                ),
                eager_model(([torch.ones(4, 4), torch.ones(4, 4)])),
            )
        )

    def test_function_holding_tensor(self):
        global_storage = []

        class FunctionClosureLeak(torch.nn.Module):
            def forward(self, x):
                fake_tensor = x + 1  # In real export, this would be a FakeTensor

                def closure():
                    return fake_tensor.shape  # Captures fake_tensor

                # Store closure globally - this creates the leak
                global_storage.append(closure)
                return x.sin()

        with (
            torch._export.config.patch(detect_non_strict_fake_tensor_leaks=True),
            self.assertWarnsRegex(
                UserWarning, "Detected 1 fake tensors that are still alive after export"
            ),
        ):
            export(FunctionClosureLeak(), (torch.randn(4, 4),), strict=False)

    def test_detect_leak_nonstrict(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x + y

        global_list = []

        class ReferenceControl:
            def __init__(self, mod):
                self.bank = []
                self.bank_dict = {}
                self.mod = mod

                def hacked_up_forward(self_, x, y):
                    self.bank.append(x.clone())
                    self.bank_dict["x"] = x.clone()
                    global_list.append(x.clone())
                    return x + y

                self.mod.forward = hacked_up_forward.__get__(self.mod, Foo)

            def __call__(self, x, y):
                ep = export(self.mod, (x, y), strict=False).module()
                out = ep(x, y)
                return out

            def update(self):
                return self.bank

        foo = Foo()
        ref = ReferenceControl(foo)
        ref(torch.randn(4, 4), torch.randn(4, 4))
        self.assertTrue(
            isinstance(ref.bank[0], torch._subclasses.fake_tensor.FakeTensor)
        )

        with (
            torch._export.config.patch(detect_non_strict_fake_tensor_leaks=True),
            self.assertWarnsRegex(
                UserWarning, "Detected 3 fake tensors that are still alive after export"
            ),
        ):
            ref(torch.randn(4, 4), torch.randn(4, 4))

    def test_detect_leak_nonstrict_with_stacktrace(self):
        global_list = []

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                nonlocal global_list
                global_list.append(x + y)
                return x + y

        foo = Foo()
        ep = export(foo, (torch.randn(4, 4), torch.randn(4, 4)), strict=False)
        self.assertTrue(
            isinstance(global_list[0], torch._subclasses.fake_tensor.FakeTensor)
        )

        with torch._export.config.patch(detect_non_strict_fake_tensor_leaks=True):
            warn_re = re.compile(
                r"Detected\s+\d+\s+fake\s+tensors?"
                r".*test_export\.py.*global_list\.append\(x \+ y\)",
                re.S,
            )
            with self.assertWarnsRegex(UserWarning, warn_re):
                ep = export(foo, (torch.randn(4, 4), torch.randn(4, 4)), strict=False)

    def test_export_cyclic_reference_leak(self):
        class Node:
            def __init__(self, tag):
                self.tag = tag
                self.ref = None
                self.tensor = None

        bank = []

        class LeakyCycle(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = x + y
                node1 = Node("A")
                node2 = Node("B")
                node1.ref = node2
                node2.ref = node1
                node1.tensor = z
                # Keep the cycle alive intentionally -> leak
                nonlocal bank
                bank.append(node1)
                return (z.sin()).cos()

        lc = LeakyCycle()
        ep = export(lc, (torch.randn(4, 4), torch.randn(4, 4)), strict=False)

        node1_ref = weakref.ref(bank[0])
        node2_ref = weakref.ref(bank[0].ref)

        bank.clear()
        del bank
        bank = []

        self.assertIsNotNone(node1_ref(), "node1 should still be alive due to cycle")
        self.assertIsNotNone(node2_ref(), "node2 should still be alive due to cycle")

        with torch._export.config.patch(detect_non_strict_fake_tensor_leaks=True):
            warn_re = re.compile(
                r"Detected\s+\d+\s+fake\s+tensors?"
                r'.*?[/\\]test_export\.py",\s+line\s+\d+,\s+in\s+forward'
                r"(?:\\n|\n)\s*z\s*=\s*x\s*\+\s*y",
                re.S,
            )
            with self.assertWarnsRegex(UserWarning, warn_re):
                ep = export(lc, (torch.randn(4, 4), torch.randn(4, 4)), strict=False)

    def test_export_for_training_run_decomp(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.ones(2, 2))
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                self.buffer.add_(5)
                return self.linear(x) + self.buffer.sum()

        eager_model = Foo()
        ep_for_training = torch.export.export(
            eager_model,
            (torch.ones(2, 2),),
        )
        ep_for_inference = ep_for_training.run_decompositions()
        self.assertExpectedInline(
            str(ep_for_inference.graph_module.code).strip(),
            """\
def forward(self, p_linear_weight, p_linear_bias, b_buffer, x):
    add = torch.ops.aten.add.Tensor(b_buffer, 5);  b_buffer = None
    permute = torch.ops.aten.permute.default(p_linear_weight, [1, 0]);  p_linear_weight = None
    addmm = torch.ops.aten.addmm.default(p_linear_bias, x, permute);  p_linear_bias = x = permute = None
    sum_1 = torch.ops.aten.sum.dim_IntList(add, [])
    add_1 = torch.ops.aten.add.Tensor(addmm, sum_1);  addmm = sum_1 = None
    return (add, add_1)""",
        )

    def test_derived_dim_out_of_order_simplified_repeat_non_derived(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y, y1, z):
                return x + y[1:] + y1[1:] + z[2:]

        foo = Foo()

        u, v, v1, w = torch.randn(5), torch.randn(6), torch.randn(6), torch.randn(7)
        _dimz = torch.export.Dim("_dimz", min=6, max=8)
        dimy = _dimz - 1
        dimx = dimy - 1
        dimz = dimx + 2  # works, effectively = _dimz
        ep = export(
            foo,
            (u, v, v1, w),
            dynamic_shapes=({0: dimx}, {0: dimy}, {0: dimy}, {0: dimz}),
        )
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: y1.size()[0] == y.size()[0]"),
        ):
            # expected 7, but got 5
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(5),
                torch.randn(8),
            )

        self.assertEqual(
            ep.module()(
                torch.randn(6),
                torch.randn(7),
                torch.randn(7),
                torch.randn(8),
            ).size()[0],
            6,
        )

    def test_static_dim_constraints(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.l = torch.nn.Linear(6, 4)

            def forward(self, x, y, z):
                x0 = self.l(x) + y[1:]
                return x0, z * 2.0

        foo = Foo()
        inputs = (torch.randn(4, 6), torch.randn(5, 4), torch.randn(3, 3))
        dx = Dim("dx", min=3, max=6)
        dy = dx + 1
        dz = Dim("dz", min=3, max=6)

        # test that tweaking shapes fails
        wrong_shape_inputs = [
            (torch.randn(4, 7), torch.randn(5, 4), torch.randn(3, 3)),
            (torch.randn(4, 6), torch.randn(5, 5), torch.randn(3, 3)),
            (torch.randn(4, 6), torch.randn(5, 4), torch.randn(3, 4)),
        ]

        # all of these should be fine
        for dynamic_shapes in [
            ({0: dx, 1: 6}, {0: dy, 1: 4}, {0: dz, 1: 3}),
            ((dx, None), (dy, 4), (dz, 3)),
            ((None, 6), (5, None), (None, None)),
            ((4, 6), {0: None, 1: 4}, {0: None, 1: 3}),
            (None, None, (Dim.STATIC, Dim.STATIC)),
        ]:
            ep = export(foo, inputs, dynamic_shapes=dynamic_shapes)
            self.assertEqual(foo(*inputs), ep.module()(*inputs))
            for wrong_inputs in wrong_shape_inputs:
                with self.assertRaisesRegex(AssertionError, "Guard failed"):
                    with self.assertRaises(RuntimeError):
                        ep.module()(*wrong_inputs)

        # check range_constraints - static dims shouldn't be present
        ep = export(foo, inputs, dynamic_shapes=((dx, None), (dy, 4), (dz, 3)))
        self.assertEqual(len(ep.range_constraints), 3)
        for vr in ep.range_constraints.values():
            self.assertTrue(vr.lower < vr.upper)

        # check raised errors
        with self.assertRaisesRegex(
            (
                torch.fx.experimental.symbolic_shapes.ConstraintViolationError,
                torch._dynamo.exc.UserError,
            ),
            "Static shape constraint of 5 does not match input size of 4, for .*",
        ):
            _ = export(foo, inputs, dynamic_shapes=((5, None), None, None))
        with self.assertRaisesRegex(
            (
                torch.fx.experimental.symbolic_shapes.ConstraintViolationError,
                torch._dynamo.exc.UserError,
            ),
            "Static shape constraint of 9 does not match input size of 6, for .*",
        ):
            _ = export(foo, inputs, dynamic_shapes=((dx, 9), (dy, 4), (3, 3)))

    def test_dim_1_2(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x * 2

        dx = Dim("dx", min=1, max=2)
        ep = export(Foo(), (torch.randn(2, 2),), dynamic_shapes=({0: dx, 1: None},))
        ep.module()(torch.randn(1, 2))
        ep.module()(torch.randn(2, 2))
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: x.size()[0] <= 2"),
        ):
            # expected <= 2, but got 3
            ep.module()(torch.randn(3, 2))
        vr = list(ep.range_constraints.values())[0]
        self.assertEqual(vr.lower, 1)
        self.assertEqual(vr.upper, 2)

    def test_derived_dim_1_2(self):
        class Bar(torch.nn.Module):
            def forward(self, x, y):
                return x + y[1:]

        dx = Dim("dx", min=1, max=2)
        ep = export(
            Bar(),
            (torch.randn(2, 2), torch.randn(3, 2)),
            dynamic_shapes=({0: dx, 1: None}, {0: dx + 1, 1: None}),
        )
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: -1 + y.size()[0] != 1"),
        ):
            # TODO: this should not error?
            ep.module()(torch.randn(1, 2), torch.randn(2, 2))
        range_lower_bounds = sorted(vr.lower for vr in ep.range_constraints.values())
        range_upper_bounds = sorted(vr.upper for vr in ep.range_constraints.values())
        self.assertEqual(range_lower_bounds, [1, 2])
        self.assertEqual(range_upper_bounds, [2, 3])

    def test_issue_161902(self):
        class Add(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        m = Add()
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)

        dx = Dim("dx", min=1, max=2)
        conflicting = {"x": (2 * dx, Dim.STATIC), "y": (dx + 1, Dim.STATIC)}

        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r"Constraints violated.*"
            r"\n.*You marked 2\*dx as dynamic but your code specialized it to be a constant \(2\).*"
            r"\n.*You marked dx \+ 1 as dynamic but your code specialized it to be a constant \(2\).*",
        ):
            export(m, (x, y), dynamic_shapes=conflicting)

    def test_range_constraints_with_replacement(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                return (x + y)[:3]

        m = M()
        inp = (torch.randn(4), torch.randn(4))
        dynamic_shapes = ((torch.export.Dim.DYNAMIC,), (torch.export.Dim.DYNAMIC,))
        ep = export(m, inp, dynamic_shapes=dynamic_shapes)
        assert len(ep.range_constraints) == 1
        vr = next(iter(ep.range_constraints.values()))
        self.assertEqual(vr.lower, 3)

    def test_unbacked_linear_layer_norm_input(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(387, 128, bias=True)
                self.layer_norm = torch.nn.LayerNorm(387)

            def forward(self, x, mask):
                masked_select = x.masked_select(mask)
                view = masked_select.view(-1, 387)

                linear = self.linear(view)
                layer_norm = self.layer_norm(view)
                return linear, layer_norm

        inputs = (
            torch.randn((256, 387), dtype=torch.float),
            torch.randint(low=0, high=1, size=(256, 1), dtype=torch.bool),
        )

        model = MyModel()
        ep = export(model, inputs)

        ref = model(*inputs)
        actual = ep.module()(*inputs)
        self.assertTrue(torch.allclose(ref[0], actual[0]))
        self.assertTrue(torch.allclose(ref[1], actual[1]))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_layer_norm_unbacked_normalized_shape(self):
        class MyModel(torch.nn.Module):
            def forward(self, scalar, weight, bias):
                u1 = scalar.item()
                y = torch.ones(2, u1)

                return torch.nn.functional.layer_norm(
                    input=y, normalized_shape=(u1,), weight=weight, bias=bias
                )

        model = MyModel()
        inputs = (
            torch.scalar_tensor(16, dtype=torch.int32),
            torch.randn(16),
            torch.randn(16),
        )
        ep = export(model, inputs)

        actual = ep.module()(*inputs)
        ref = model(*inputs)
        self.assertTrue(torch.allclose(ref[0], actual[0]))

    def test_unbacked_3d_matmul(self):
        class Model(torch.nn.Module):
            def forward(self, x, repeat):
                u0 = repeat.item()
                t1 = x.unsqueeze(1).expand(x.size(0), u0 // 2, x.size(-1))
                t2 = torch.ones(3)
                return torch.matmul(t1, t2)

        model = Model()
        inputs = (torch.randn(4, 3), torch.scalar_tensor(2, dtype=torch.int))

        exported = export(model, inputs).module()
        self.assertEqual(model(*inputs), exported(*inputs))

    def test_dynamic_shapes_wrapped_with_shape_guards(self):
        class Neuron(torch.nn.Module):
            def __init__(self, n_dims: int = 5, n_targets: int = 3):
                super().__init__()
                self.linear = torch.nn.Linear(n_dims, n_targets)

            def forward(self, x, y):
                return torch.sigmoid(self.linear(x + y))

        args = (torch.randn(2, 5), torch.randn(2, 5))
        batch = torch.export.Dim.DYNAMIC

        n = Neuron()
        compiled = export(n, args, dynamic_shapes=({0: batch}, {0: batch}))
        expected = n(*args)
        mod = compiled.module()
        got = mod(*args)
        self.assertTrue(torch.allclose(expected, got))

        class Wrapped(Neuron):
            def forward(self, *args):
                return super().forward(*args)

        w = Wrapped()

        if is_retracebility_test(self._testMethodName):
            with self.assertRaisesRegex(
                torch._dynamo.exc.UserError,
                "Detected mismatch between the structure of `inputs` and `dynamic_shapes`"
                ": `inputs` has 2 elements, but `dynamic_shapes` has 1 elements",
            ):
                export(w, args, dynamic_shapes={"args": ({0: batch}, {0: batch})})
        else:
            compiled = export(
                w, args, dynamic_shapes={"args": ({0: batch}, {0: batch})}
            )
            expected = w(*args)
            mod = compiled.module()
            got = mod(*args)
            self.assertTrue(torch.allclose(expected, got))

    def test_dynamic_shapes_builder_basic(self):
        class M(torch.nn.Module):
            def forward(self, x, y, z):
                return x + y[0] + z["k"]

        m = M()

        x = torch.randn(4)
        y = [torch.randn(4)]
        z = {"k": torch.randn(4)}
        args = (x, y, z)

        shapes_collection = torch.export.ShapesCollection()
        dim = torch.export.Dim("dim", max=10)
        # specify shape of tensor
        shapes_collection[x] = (dim,)
        # tensor can be arbitrarily deep
        shapes_collection[y[0]] = (dim,)
        # can also specify some dimension in shape of tensor
        shapes_collection[z["k"]][0] = dim

        ep = export(m, args, dynamic_shapes=shapes_collection)
        sym = next(iter(ep.range_constraints.keys()))
        for node in ep.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(str(tuple(node.meta["val"].shape)), f"({sym},)")

    def test_dynamic_shapes_builder_kwargs(self):
        class M(torch.nn.Module):
            def forward(self, x, y, z):
                return x + y[0] + z["k"]

        m = M()

        x = torch.randn(4)
        y = [torch.randn(4)]
        z = {"k": torch.randn(4)}
        args = (x,)
        kwargs = {"z": z, "y": y}

        shapes_collection = torch.export.ShapesCollection()
        dim = torch.export.Dim("dim", max=10)
        shapes_collection[x] = (dim,)
        shapes_collection[y[0]] = (dim,)
        shapes_collection[z["k"]] = (dim,)

        ep = export(m, args, kwargs=kwargs, dynamic_shapes=shapes_collection)
        sym = next(iter(ep.range_constraints.keys()))
        for node in ep.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(str(tuple(node.meta["val"].shape)), f"({sym},)")

    def test_dynamic_shapes_builder_pytree(self):
        torch.export.register_dataclass(
            Inp1,
            serialized_type_name="test_dynamic_shapes_builder_pytree.Inp1",
        )

        class M(torch.nn.Module):
            def forward(self, inp: Inp1):
                return inp.x + inp.y[0] + inp.z["k"]

        m = M()
        x = torch.randn(4)
        y = [torch.randn(4)]
        z = {"k": torch.randn(4)}
        args = (Inp1(x, y, z),)

        shapes_collection = torch.export.ShapesCollection()
        dim = torch.export.Dim("dim", max=10)
        shapes_collection[x] = (dim,)
        shapes_collection[y[0]] = (dim,)
        shapes_collection[z["k"]] = (dim,)

        ep = export(m, args, dynamic_shapes=shapes_collection.dynamic_shapes(m, args))
        sym = next(iter(ep.range_constraints.keys()))
        for node in ep.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(str(tuple(node.meta["val"].shape)), f"({sym},)")

    def test_dynamic_shapes_inferred_basic(self):
        class M(torch.nn.Module):
            def forward(self, x, y, z):
                # x and y[0] must have same dynamic shape (say `dim`) >= 3
                tmp = (x + y[0])[:3]
                # z["k"] must have static shape = 3
                return tmp * z["k"]

        m = M()
        args = (torch.randn(4), [torch.randn(4)], {"k": torch.randn(3)})

        additional_inputs = torch.export.AdditionalInputs()
        # 4->5, 4->5, 3->3
        good_args = (torch.randn(5), [torch.randn(5)], {"k": torch.randn(3)})
        additional_inputs.add(good_args)

        ep = export(m, args, dynamic_shapes=additional_inputs)
        got_shapes = [
            str(tuple(node.meta["val"].shape))
            for node in ep.graph.find_nodes(op="placeholder")
        ]
        dim = next(iter(ep.range_constraints.keys()))
        expected_shapes = [f"({dim},)", f"({dim},)", "(3,)"]
        self.assertEqual(got_shapes, expected_shapes)

        def expect_error(bad_args, run_time_msg, compile_time_msg):
            with self.assertRaisesRegex(AssertionError, run_time_msg):
                ep.module()(*bad_args)

            additional_inputs = torch.export.AdditionalInputs()
            additional_inputs.add(bad_args)

            with self.assertRaisesRegex(RuntimeError, compile_time_msg):
                export(m, args, dynamic_shapes=additional_inputs)

        expect_error(
            # 4->2, 4->2, 3->3
            bad_args=(torch.randn(2), [torch.randn(2)], {"k": torch.randn(3)}),
            run_time_msg=escape(
                "Guard failed: x.size()[0] >= 3"
            ),  # expected >= 3, but got 2
            compile_time_msg="Expected input.*to be >= 3, but got 2",
        )

        expect_error(
            # 4->6, 4->7, 3->3
            bad_args=(torch.randn(6), [torch.randn(7)], {"k": torch.randn(3)}),
            run_time_msg=escape(
                "Guard failed: y[0].size()[0] == x.size()[0]"
            ),  # expected 6, but got 7
            compile_time_msg="Expected input.*to be equal to 6, but got 7",
        )

        expect_error(
            # 4->5, 4->5, 3->4
            bad_args=(torch.randn(5), [torch.randn(5)], {"k": torch.randn(4)}),
            run_time_msg=escape(
                "Guard failed: z['k'].size()[0] == 3"
            ),  # expected 3, but got 4
            compile_time_msg=r"You marked.*but your code specialized it to be a constant.*If you're using Dim.DYNAMIC, replace it with either Dim.STATIC or Dim.AUTO",
        )

    def test_additional_inputs_constants(self):
        @dataclass
        class D:
            b: bool
            i: int
            f: float
            t: torch.Tensor

        pytree.register_dataclass(D)

        class M(torch.nn.Module):
            def forward(self, d: D):
                return d.i + d.f + d.t

        input1 = (D(True, 3, 3.0, torch.ones(3)),)

        # int and tensor change
        input2 = (D(True, 4, 3.0, torch.ones(4)),)
        ai = torch.export.AdditionalInputs()
        ai.add(input1)
        ai.add(input2)
        dynamic_shapes = ai.dynamic_shapes(M(), input1)
        self.assertEqual(
            dynamic_shapes, {"d": [None, Dim.DYNAMIC, None, (Dim.DYNAMIC,)]}
        )
        torch.export.export(M(), input1, dynamic_shapes=ai)

        # float changes, error
        input2 = (D(True, 3, 4.0, torch.ones(3)),)
        ai = torch.export.AdditionalInputs()
        ai.add(input1)
        ai.add(input2)
        with self.assertRaisesRegex(
            ValueError, r"they cannot be marked as dynamic: \(3\.0, 3\.0, 4\.0\)"
        ):
            ai.dynamic_shapes(M(), input1)
        with self.assertRaisesRegex(
            ValueError, r"they cannot be marked as dynamic: \(3\.0, 3\.0, 4\.0\)"
        ):
            torch.export.export(M(), input1, dynamic_shapes=ai)

        # bool changes, error
        input2 = (D(False, 3, 3.0, torch.ones(3)),)
        ai = torch.export.AdditionalInputs()
        ai.add(input1)
        ai.add(input2)
        with self.assertRaisesRegex(
            ValueError, r"they cannot be marked as dynamic: \(True, True, False\)"
        ):
            ai.dynamic_shapes(M(), input1)
        with self.assertRaisesRegex(
            ValueError, r"they cannot be marked as dynamic: \(True, True, False\)"
        ):
            torch.export.export(M(), input1, dynamic_shapes=ai)

        # Differing types
        input1 = (D(True, 0, 3.0, torch.ones(3)),)
        input2 = (D(True, False, 3.0, torch.ones(3)),)
        ai = torch.export.AdditionalInputs()
        ai.add(input1)
        ai.add(input2)
        with self.assertRaisesRegex(
            ValueError,
            r"differing types, so they cannot be marked as dynamic: \(0, 0, False\)",
        ):
            print(ai.dynamic_shapes(M(), input1))
        with self.assertRaisesRegex(
            ValueError,
            r"differing types, so they cannot be marked as dynamic: \(0, 0, False\)",
        ):
            torch.export.export(M(), input1, dynamic_shapes=ai)

    def test_mismatched_dynamic_shapes(self):
        AUTO, STATIC = Dim.AUTO, Dim.STATIC

        class M(torch.nn.Module):
            def forward(self, x):
                return x["k"]["k"][0] + x["k"]["k"][1]

        inputs = ({"k": {"k": [torch.rand(4), torch.rand(4)]}},)
        dim = torch.export.Dim("dim")

        dynamic_shapes = {
            "k": {"k": [dim, dim]}
        }  # ValueError: Node keys mismatch; missing key(s): {'x'}; extra key(s): {'k'}.
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            re.escape(
                "When `dynamic_shapes` is specified as a dict, its top-level keys "
                "must be the arg names ['x'] of `inputs`, but here they are ['k']. "
                "Since here `inputs` is a list/tuple enclosing a single dict, "
                "maybe you just forgot to enclose `dynamic_shapes` in a list/tuple?"
            ),
        ):
            export(M(), inputs, dynamic_shapes=dynamic_shapes)

        dynamic_shapes = (
            {"k": {"k": [dim, dim]}},
        )  # torch._dynamo.exc.UserError: Unexpected dynamic_shape .*dim.* of Tensor, try None instead
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Unexpected input tensor shape .*dim.* "
            + re.escape(
                "specified at `dynamic_shapes[0]['k']['k'][0]` "
                "(expected either a list/tuple of dimensions, or a dict mapping indices to dimensions,"
                " where each dimension is an int, a Dim, Dim.AUTO, Dim.STATIC, or Dim.DYNAMIC)"
            ),
        ):
            export(M(), inputs, dynamic_shapes=dynamic_shapes)

        dynamic_shapes = (
            {"k": {"k": (dim, dim)}},
        )  # ValueError: Node type mismatch; expected <class 'list'>, but got <class 'tuple'>.
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            re.escape(
                "Detected mismatch between the structure of `inputs` and `dynamic_shapes`: "
                "`inputs[0]['k']['k']` is a <class 'list'>, but `dynamic_shapes[0]['k']['k']` is a <class 'tuple'>"
            ),
        ):
            export(M(), inputs, dynamic_shapes=dynamic_shapes)

        dynamic_shapes = ({"k": {"k": [(dim,), (dim,)]}},)  # ok
        export(M(), inputs, dynamic_shapes=dynamic_shapes)

        dynamic_shapes = (
            {"k": {"k": dim}},
        )  # ValueError: Node type mismatch; expected <class 'list'>, but got .*Dim.*.
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            re.escape(
                "Detected mismatch between the structure of `inputs` and `dynamic_shapes`: "
                "`inputs[0]['k']['k']` is a <class 'list'>, but `dynamic_shapes[0]['k']['k']` is not"
            ),
        ):
            export(M(), inputs, dynamic_shapes=dynamic_shapes)

        dynamic_shapes = {
            "x": {"k": [(dim,), (dim,)]},
            "k": {"k": [(dim,), (dim,)]},
        }  # ValueError: Node arity mismatch; expected 1, but got 2.
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            re.escape(
                "When `dynamic_shapes` is specified as a dict, its top-level keys "
                "must be the arg names ['x'] of `inputs`, but here they are ['x', 'k']. "
                "Alternatively, you could also ignore arg names entirely "
                "and specify `dynamic_shapes` as a list/tuple matching `inputs`."
            ),
        ):
            export(M(), inputs, dynamic_shapes=dynamic_shapes)

        dynamic_shapes = (
            {"k": {"k": [(dim,), (dim,), (dim,)]}},
        )  # ValueError: Node arity mismatch; expected 2, but got 3.
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            re.escape(
                "Detected mismatch between the structure of `inputs` and `dynamic_shapes`: "
                "`inputs[0]['k']['k']` has 2 elements, but `dynamic_shapes[0]['k']['k']` has 3 elements"
            ),
        ):
            export(M(), inputs, dynamic_shapes=dynamic_shapes)

        dynamic_shapes = (
            {"k": {"K": [(dim,), (dim,), (dim,)]}},
        )  # ValueError: Node keys mismatch; missing key(s): {'k'}; extra key(s): {'K'}.
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            re.escape(
                "Detected mismatch between the structure of `inputs` and `dynamic_shapes`: "
                "`inputs[0]['k']` has keys ['k'], but `dynamic_shapes[0]['k']` has keys ['K']"
            ),
        ):
            export(M(), inputs, dynamic_shapes=dynamic_shapes)

        class N(torch.nn.Module):
            def forward(self, x):
                return x["k"]["k1"][0] + x["k"]["k2"][0]

        inputs = ({"k": {"k1": [torch.rand(4)], "k2": [torch.rand(4)]}},)
        dim = torch.export.Dim("dim")

        dynamic_shapes = ({"k": {"k2": [(dim,)], "k1": [(dim,)]}},)  # ok
        export(N(), inputs, dynamic_shapes=dynamic_shapes)

        class O(torch.nn.Module):
            def forward(self, x):
                return x + 2

        inputs = (torch.randn(4, 8, 6),)
        dynamic_shapes = {"x": (dim, None)}
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r"Expected dynamic shape spec .* at `dynamic_shapes\['x'\]` to have the same length "
            r"as the actual tensor shape torch\.Size\(\[4, 8, 6\]\) \(expected 3, but got 2 instead\)",
        ):
            export(O(), inputs, dynamic_shapes=dynamic_shapes)

    def test_unbacked_bindings_for_divisible_u_symint(self):
        from torch._export.utils import _get_shape_env_from_gm
        from torch.utils._sympy.symbol import prefix_str, symbol_is_type, SymT

        class M(torch.nn.Module):
            def forward(self, a, b):
                return torch.ops.mylib.foo_unbacked(a, b)

        @torch.library.custom_op("mylib::foo_unbacked", mutates_args={})
        def foo_unbacked(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a[b.item()]

        @foo_unbacked.register_fake
        def foo_unbacked_fake_impl(a, b):
            ctx = torch.library.get_ctx()
            u = ctx.new_dynamic_size(min=0, max=len(a) // 10) * 10
            return torch.empty(u, a.shape[1], dtype=a.dtype)

        # check binding path is correct
        ep = export(
            M(),
            (torch.randn(100, 4), torch.tensor(10)),
        )
        foo = [node for node in ep.graph.nodes if node.name == "foo_unbacked"][0]
        unbacked_bindings = foo.meta["unbacked_bindings"]
        self.assertEqual(len(unbacked_bindings), 1)  # check binding is {u: path}
        u = next(iter(unbacked_bindings.keys()))
        self.assertEqual(
            type(u).__name__, "Symbol"
        )  # check binding is symbol, not expr
        path = unbacked_bindings[u]
        self.assertEqual(len(path), 3)  # check path is [size, 0, DivideByKey(10)]
        self.assertEqual(type(path[2]).__name__, "DivideByKey")
        self.assertEqual(path[2].divisor, 10)

        # collect bound symbols
        bound = set()
        for node in ep.graph.nodes:
            bound.update(node.meta.get("unbacked_bindings", {}))

        # check ShapeEnv counters compared to binding indices
        shape_env = _get_shape_env_from_gm(ep.graph_module)
        next_index = shape_env.unbacked_symint_counter
        shape_env.unbacked_symint_counter += 1
        for symbol in bound:
            self.assertTrue(symbol_is_type(symbol, SymT.UNBACKED_INT))
            self.assertTrue(
                int(str(symbol)[len(prefix_str[SymT.UNBACKED_INT]) :]) < next_index
            )

    def test_torch_check_eq_commutativity(self):
        class M1(torch.nn.Module):
            def forward(self, x1, x2, x3, y):
                z1 = x1.item()
                z2 = x2.item()
                z3 = x3.item()
                # instead of: torch._check((z2 + z3) == z1)
                torch._check(z1 == (z2 + z3))
                if z2 + z3 == z1:
                    return y * 2
                else:
                    return y + 3

        export(
            M1(),
            (torch.tensor(6), torch.tensor(3), torch.tensor(3), torch.randn(1)),
        )

        class M2(torch.nn.Module):
            def forward(self, x1, x2, x3, y):
                z1 = x1.item()
                z2 = x2.item()
                z3 = x3.item()
                # instead of: torch._check((z2 + z3) != z1)
                torch._check(z1 != (z2 + z3))
                if z2 + z3 == z1:
                    return y * 2
                else:
                    return y + 3

        export(
            M2(),
            (torch.tensor(6), torch.tensor(6), torch.tensor(6), torch.randn(1)),
        )

    def test_replaced_unbacked_bindings(self):
        import sympy

        from torch.utils._sympy.symbol import prefix_str, symbol_is_type, SymT

        class Foo(torch.nn.Module):
            def forward(self, x, y, z):
                m, n = x.item(), y.item()
                torch._check(m == 4)
                torch._check(n == z.shape[0])
                return m + n + z

        inps = (
            torch.tensor(4),
            torch.tensor(5),
            torch.randn(5),
        )
        dynamic_shapes = {
            "x": None,
            "y": None,
            "z": (Dim("dx", max=16),),
        }
        ep = export(Foo(), inps, dynamic_shapes=dynamic_shapes)
        # values should have no unbacked symbols, bindings should be empty
        for node in ep.graph.nodes:
            val = node.meta.get("val")
            bindings = node.meta.get("unbacked_bindings")
            self.assertTrue(
                not (
                    isinstance(val, sympy.Symbol)
                    and symbol_is_type(val, SymT.UNBACKED_INT)
                )
            )
            self.assertTrue(bindings is None)

    def test_raise_user_error_when_guard_on_data_dependent_operation(self):
        class M(torch.nn.Module):
            def forward(self, x):
                y = x.nonzero()
                z = y.shape[0]
                if z > 2:
                    return x.cos()
                else:
                    return x.sin()

        with self.assertRaisesRegex(
            (
                torchdynamo.exc.UserError,
                torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode,
            ),
            "Could not guard on data-dependent expression",
        ):
            _ = export(M(), (torch.tensor([2, 3, 5]),))

    def test_unbacked_infer_size(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                u0 = x.item()
                t = torch.empty(u0 - 1)
                return t + t

        ep = torch.export.export(Foo(), (torch.tensor([5]),))
        ep.module()(torch.tensor([5]))
        ep.module()(torch.tensor([1]))

    def test_unbacked_pad(self):
        class Foo(torch.nn.Module):
            def forward(self, xs, pad):
                u0, u1, u2 = xs.tolist()
                x = torch.ones(u0, u1, u2)
                pl0, pr0, pl1, pr1 = pad.tolist()
                return torch.nn.functional.pad(x, (pl0, pr0, pl1, pr1))

        x = torch.tensor([64, 64, 64])
        pad = torch.tensor([8, -8, 4, 0])
        m = Foo()
        ep = export(m, (x, pad))
        self.assertEqual(ep.module()(x, pad).shape, m(x, pad).shape)

        # don't guard on negative/positive pad values
        pad2 = torch.tensor([-5, 9, 0, 8])
        self.assertEqual(ep.module()(x, pad2).shape, m(x, pad2).shape)

    def test_suggested_fixes_for_data_dependent_errors_basic(self):
        # suggested fixes for data-dependent errors only work in non-strict mode
        strict = False
        error_type = torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode

        # Just to introduce some indirection: N is a top-level module N that calls
        # module M, defined next.
        class N(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m = M()

            def forward(self, t):
                return self.m(t) + 1

        # example input
        t = torch.tensor([1, 4, 4], dtype=torch.int32)

        # We define a series of versions of M() below. Each version has
        # raises a data-dependent error that the next version fixes, by
        # copy-pasting a suggested fix in the error message. The fix is
        # always a torch.check() on an unresolved condition (or its negation)
        # on unbacked symints mentioned in the error message.
        # Note that the suggested fixes are in terms of local variables
        # near the location of error that "contain" the unbacked symints
        # in the unresolved condition (either directly or indirectly, e.g.,
        # inside a list or inside the shape of a tensor).

        class M_v0(torch.nn.Module):
            def forward(self, t):
                items = [t[i].item() for i in range(t.numel())]
                r = torch.randn([items[0], items[1]])
                return r.view(items[0], items[2])

        M = M_v0
        export(N(), (t,), strict=strict)

    def test_suggested_fixes_for_data_dependent_errors_puzzlers(self):
        # suggested fixes for data-dependent errors only work in non-strict mode
        strict = False
        error_type = torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode

        def retry_export(m, inp, fixes):
            # API that applies a series of fixes, retrying export after applying each fix,
            # and asserting the applied fix was suggested in the previous try.
            # Using this API avoids the need to define multiple versions of the same test
            # module, as in `test_suggested_fixes_for_data_dependent_errors_basic` above.
            def code(snippets):
                return f"[{', '.join(snippets)}]"

            for i in range(len(fixes)):
                with self.assertRaisesRegex(error_type, re.escape(fixes[i])):
                    export(m, (*inp, code(fixes[:i])), strict=strict)
            export(m, (*inp, code(fixes)), strict=strict)

        # The following examples are lifted from @ezyang's "Data-dependent shape puzzlers"
        # notebook at https://www.internalfb.com/intern/anp/view/?id=5330476

        # These test modules are written in a way that works well with retry_export above.
        # Specifically, they take an extra `fixes` argument and `eval` it at the location
        # that is expected to raise errors.

        class cf_implicitsize(torch.nn.Module):
            def forward(self, x, y, fixes):
                i = x.item()
                eval(fixes)
                # instead of y[i]
                return y.narrow(0, i, 1).squeeze()

        retry_export(
            cf_implicitsize(),
            (torch.tensor(2), torch.randn(10)),
            fixes=[],
        )

        class cf_stacklist(torch.nn.Module):
            def forward(self, xs, y, fixes):
                i = y.item()
                eval(fixes)
                return torch.stack(xs, 0).narrow(0, i, 1).squeeze()

        retry_export(
            cf_stacklist(),
            ([torch.ones(5) * i for i in range(10)], torch.tensor(2)),
            fixes=[],
        )

        class cf_tensorsplit(torch.nn.Module):
            def forward(self, x, offsets_t, fixes):
                lengths = torch.diff(offsets_t).tolist()
                rs = []
                start = 0
                for length in lengths:
                    eval(fixes)
                    rs.append(x.narrow(0, start, length))
                    start += length
                return rs

        retry_export(
            cf_tensorsplit(),
            (torch.arange(10), torch.tensor([0, 2, 5, 7, 10])),
            fixes=[],  # nothing to fix!
        )

    def test_simple_unbacked_view(self):
        if "cpp_runtime_nonstrict" in self.id():
            self.skipTest("TODO Unexpected success in OSS but not in fbcode.")

        class Foo(torch.nn.Module):
            def forward(self, x):
                u0 = x.item()
                y = torch.empty(5, u0)
                return y.view(u0, 5)  # [5, u0] -> [u0, 5]

        ep = export(Foo(), (torch.tensor([9]),))
        self.assertEqual(ep.module()(torch.tensor([8])).size(0), 8)
        self.assertEqual(ep.module()(torch.tensor([5])).size(0), 5)

        class Foov2(torch.nn.Module):
            def forward(self, xs):
                xsl = xs.tolist()
                a, b = xsl
                x = torch.zeros(a)
                return x.reshape(b)

        xs = torch.tensor([4, 4])
        ep = export(Foov2(), (xs,))
        self.assertEqual(ep.module()(xs).size(0), 4)
        self.assertEqual(ep.module()(torch.tensor([5, 5])).size(0), 5)

    def test_no_suggested_fixes_for_data_dependent_errors(self):
        # suggested fixes for data-dependent errors only work in non-strict mode
        strict = False
        error_type = torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode

        class cf_stacklist(torch.nn.Module):
            def forward(self, xs, y):
                # y.item() is not a local, so we can't suggest a fix
                if y.item() < 0:
                    return (
                        torch.stack(xs, 0).narrow(0, y.item() + xs.size(), 1).squeeze()
                    )
                else:
                    return torch.stack(xs, 0).narrow(0, y.item(), 1).squeeze()

        with self.assertRaisesRegex(
            error_type,
            "Could not guard on data-dependent expression u0 < 0",
        ):
            export(
                cf_stacklist(),
                ([torch.ones(5) * i for i in range(10)], torch.tensor(2)),
                strict=strict,
            )

        class Box:
            def __init__(self, content):
                self.content = content

        from torch.utils._pytree import register_pytree_node

        register_pytree_node(
            Box,
            lambda box: ([box.content], None),  # flatten_fn
            lambda contents, _context: Box(*contents),  # unflatten_fn
            flatten_with_keys_fn=None,  # unflatten_fn
            serialized_type_name="test_no_suggested_fixes_for_data_dependent_errors.Box",
        )

        class cf_stacklist_udd(torch.nn.Module):
            def forward(self, xs, y):
                box = Box(y.item())
                # box.content is not a local, so we can't suggest a fix
                if box.content < 0:
                    return (
                        torch.stack(xs, 0)
                        .narrow(0, box.content + xs.size(), 1)
                        .squeeze()
                    )
                else:
                    return (
                        torch.stack(xs, 0)
                        .narrow(0, box.content + xs.size(), 1)
                        .squeeze()
                    )

        with self.assertRaisesRegex(
            error_type,
            "Could not guard on data-dependent expression u0 < 0",
        ):
            export(
                cf_stacklist_udd(),
                ([torch.ones(5) * i for i in range(10)], torch.tensor(2)),
                strict=strict,
            )

    def test_tolist(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.tolist()

        ep = export(M(), (torch.ones(3, dtype=torch.int),))
        self.assertEqual(ep.module()(torch.tensor([1, 2, 3])), [1, 2, 3])

    def test_if_functional(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                z = x + 4
                z.add_(4)
                y = z.view(x.shape)
                return x.cos() + y.cos()

        foo = Module()
        gm = export(foo, (torch.tensor([2, 3, 5]),)).run_decompositions({})

        view_count = 0
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add_.Tensor:
                # No more inplace mutation
                self.assertNotEqual(
                    node.target,
                    torch.ops.aten.add_.Tensor,
                    "There shouldn't be any inplace mutation node in the graph.",
                )
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.view.default
            ):
                view_count += 1

        # There should be nonzero view nodes in the graph
        self.assertTrue(view_count > 0)

    def test_solver_unsupported_sympy_function(self):
        # repro of https://github.com/pytorch/pytorch/issues/131897

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = torch.nn.functional.interpolate(
                    x, scale_factor=0.5, mode="bilinear"
                )
                x = torch.nn.functional.interpolate(
                    x, scale_factor=2.0, mode="bilinear"
                )
                x = x + y
                return x

        model = MyModule().eval()

        inputs = (
            torch.rand((1, 1, 32, 32)),
            torch.rand((1, 1, 32, 32)),
        )

        dim = torch.export.Dim.AUTO
        dynamic_shapes = {"x": {2: dim, 3: dim}, "y": {2: dim, 3: dim}}

        exported_program = export(model, inputs, dynamic_shapes=dynamic_shapes)
        self.assertEqual(exported_program.module()(*inputs), model(*inputs))

    def test_export_max_onnx_reported(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                s1 = max(x.shape[0], y.shape[0])
                s2 = max(x.shape[1], y.shape[1])
                z = torch.zeros((s1, s2), dtype=x.dtype)
                z[: x.shape[0], : x.shape[1]] = x
                z[: y.shape[0], : y.shape[1]] += y
                return z

        model = Model()
        x = torch.arange(6).reshape((2, 3))
        y = torch.arange(6).reshape((3, 2)) * 10
        DYN = torch.export.Dim.DYNAMIC

        ep = export(
            model,
            (x, y),
            dynamic_shapes=({0: DYN, 1: DYN}, {0: DYN, 1: DYN}),
            strict=True,
        )
        self.assertTrue(torch.allclose(ep.module()(x, y), model(x, y)))
        x2 = torch.arange(4).reshape((2, 2))
        y2 = torch.arange(9).reshape((3, 3))
        with self.assertRaisesRegex(
            AssertionError,
            (
                escape("Guard failed: max(x.size()[1], y.size()[1]) == x.size()[1]")
                if is_retracebility_test(self._testMethodName)
                else escape(
                    "Guard failed: max(1, x.size()[1], y.size()[1]) == x.size()[1]"
                )
            ),
        ):
            # TODO: this should not error?
            self.assertTrue(torch.allclose(ep.module()(x2, y2), model(x2, y2)))

    def test_export_max_nonstrict(self):
        class FooMax(torch.nn.Module):
            def forward(self, x):
                return torch.ones(max(x.item(), 1024))

        ep_non_strict_foo_max_symint = export(
            FooMax(), (torch.tensor(4),), strict=False
        ).graph
        FileCheck().check_count("torch.sym_max", count=1, exactly=True).run(
            str(ep_non_strict_foo_max_symint)
        )

        class FooMaxTensors(torch.nn.Module):
            def forward(self, x):
                return torch.ones(max(x, x)) + torch.ones(min(x, x))

        ep_non_strict_foo_max_symint = export(
            FooMaxTensors(), (torch.tensor(4),), strict=False
        ).graph
        FileCheck().check_count(
            "torch.ops.aten.maximum.default", count=1, exactly=True
        ).run(str(ep_non_strict_foo_max_symint))
        FileCheck().check_count(
            "torch.ops.aten.minimum.default", count=1, exactly=True
        ).run(str(ep_non_strict_foo_max_symint))

        class FooMaxTensorsIter(torch.nn.Module):
            def forward(self, x):
                return max([x, x]) + min([x, x]) + max(x, 5) + min(x, 3)

        ep_non_strict_foo_max_symint = export(
            FooMaxTensorsIter(), (torch.tensor(4),), strict=False
        ).graph
        FileCheck().check_count(
            "torch.ops.aten.maximum.default", count=1, exactly=True
        ).run(str(ep_non_strict_foo_max_symint))
        FileCheck().check_count(
            "torch.ops.aten.minimum.default", count=1, exactly=True
        ).run(str(ep_non_strict_foo_max_symint))
        FileCheck().check_count(
            "torch.ops.aten.clamp.default", count=2, exactly=True
        ).run(str(ep_non_strict_foo_max_symint))

        class FooMaxTensorsSymInt(torch.nn.Module):
            def forward(self, x, y):
                return max([x.shape[0], y.shape[0], x.shape[0]]) + min(
                    [x.shape[0], y.shape[0], x.shape[0]]
                )

        dynamic_shapes = {
            "x": {0: torch.export.Dim.AUTO},
            "y": {0: torch.export.Dim.AUTO},
        }

        ep_non_strict_foo_max_symint = export(
            FooMaxTensorsSymInt(),
            (torch.randn(4, 4), torch.randn(4, 4)),
            dynamic_shapes=dynamic_shapes,
            strict=False,
        ).graph
        FileCheck().check_count("torch.sym_max", count=1, exactly=True).run(
            str(ep_non_strict_foo_max_symint)
        )
        FileCheck().check_count("torch.sym_min", count=1, exactly=True).run(
            str(ep_non_strict_foo_max_symint)
        )

        class FooMaxTensorsSymShape(torch.nn.Module):
            def forward(self, x):
                return max(x, x.shape[0])

        dynamic_shapes = {
            "x": {0: torch.export.Dim.AUTO},
        }

        with self.assertRaisesRegex(
            RuntimeError, "Dynamo failed to run FX node with fake tensors"
        ):
            _ = export(
                FooMaxTensorsSymShape(),
                (torch.randn(4, 4),),
                dynamic_shapes=dynamic_shapes,
                strict=True,
            ).graph

        with self.assertRaisesRegex(
            RuntimeError,
            "Boolean value of Tensor with more than one value is ambiguous",
        ):
            _t = export(
                FooMaxTensorsSymShape(),
                (torch.randn(4, 4),),
                dynamic_shapes=dynamic_shapes,
                strict=False,
            ).graph

    def test_math_pow(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                b = x.item()
                p = min(b, 10)
                p = math.pow(p, 10)
                return y * p

        ep = export(M(), (torch.tensor(5), torch.randn(5)), strict=False)
        FileCheck().check_count("torch.sym_min", count=1, exactly=True).run(
            str(ep.graph)
        )
        FileCheck().check_count("operator.pow", count=1, exactly=True).run(
            str(ep.graph)
        )

    def test_export_mod_constraints(self):
        class BasicDynamiShapeModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.view(x.shape[0] - 1, -1)

        m = BasicDynamiShapeModel()
        a = torch.randn(3, 4)
        dim0_x = torch.export.Dim("dim0_x", min=3)
        dim1_x = torch.export.Dim("dim1_x", max=8000)
        dynamic_shapes = {"x": (dim0_x, dim1_x)}
        em = torch.export.export(
            m,
            (a,),
            dynamic_shapes=dynamic_shapes,
            prefer_deferred_runtime_asserts_over_guards=True,
        )
        em.module()(torch.randn(4, 3))
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Eq\(Mod\(s27\*s77, s77 \- 1\), 0\)",
        ):
            em.module()(torch.randn(4, 5))

        dim0_x = None
        dim1_x = 2 * torch.export.Dim("_dim1_x", max=4000)
        dynamic_shapes = {"x": (dim0_x, dim1_x)}
        em = torch.export.export(m, (a,), dynamic_shapes=dynamic_shapes)
        x = torch.randn(3, 5)
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: 3 * x.size()[1] % 2 == 0"),
        ):
            # expected 2*..., but got 5
            em.module()(x)

    def test_dont_duck_size_for_auto_dynamic(self):
        AUTO, STATIC = Dim.AUTO, Dim.STATIC

        class Foo(torch.nn.Module):
            def forward(self, x, y):
                # x: [s0, s1], y: [s0 + 1, 4]
                assert y.shape[1] == 4
                assert x.shape[0] == y.shape[0] - 1
                return x * 2, y * 2

        # duck sizing would make all static based on these sample inputs
        inputs = (torch.randn(4, 4), torch.randn(5, 4))
        shapes = {
            "x": (AUTO, AUTO),
            "y": (AUTO, AUTO),
        }
        ep = export(Foo(), inputs, dynamic_shapes=shapes)
        ep.module()(torch.randn(6, 3), torch.randn(7, 4))

    def test_map(self):
        if "cpp_runtime_nonstrict" in self.id():
            self.skipTest("TODO Unexpected success in OSS but not in fbcode.")

        class Module(torch.nn.Module):
            def forward(self, xs, y, z):
                def body(x, y, z):
                    return x + y + z

                return map(body, xs, y, z)

        list_tensor_map = Module()
        inps = (torch.ones(6, 4), torch.tensor(5), torch.tensor(4))
        self._test_export_same_as_eager(list_tensor_map, inps)

    @unittest.expectedFailure
    def test_crop_like(self):
        # https://fb.workplace.com/groups/1405155842844877/posts/8195050017188725/

        # Minimal crop code copied from https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/functional
        class CropLike(torch.nn.Module):
            def forward(self, image, crop_height, crop_width):
                c, image_height, image_width = image.shape
                crop_top = int(round((image_height - crop_height) / 2.0))
                crop_left = int(round((image_width - crop_width) / 2.0))
                return image[
                    ...,
                    crop_top : crop_top + crop_height,
                    crop_left : crop_left + crop_width,
                ]

        crop = CropLike()
        imagew = Dim("width")
        imageh = Dim("height")
        dynamic_dims = {
            "image": {0: None, 1: imageh, 2: imagew},
            "crop_height": None,
            "crop_width": None,
        }
        args = (torch.rand(3, 512, 512), 150, 150)
        ecrop = export(crop, args=args, dynamic_shapes=dynamic_dims)

        args = (torch.rand(3, 700, 700), 150, 150)
        self.assertEqual(ecrop.module()(*args), ecrop(*args))

    def test_dim_dynamic_divisibility(self):
        class M(torch.nn.Module):
            def forward(self, x):
                if x.size(0) % 2 == 0:
                    return x.clone() * 2
                else:
                    return x.clone() * 0

        input1 = (torch.randn(4),)
        model = M()
        dynamic_shapes = {
            "x": {0: torch.export.Dim.DYNAMIC},
        }
        export(model, input1, dynamic_shapes=dynamic_shapes)

    def test_export_func_with_kwargs(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, kw1, kw2):
                return arg1 + arg2, kw1 + kw2

        kw_func = Module()
        args = (torch.ones(6, 4), torch.ones(1, 1))
        kwargs = {"kw1": torch.ones(1, 1), "kw2": torch.ones(6, 4)}
        self._test_export_same_as_eager(kw_func, args, kwargs)

    def test_export_func_with_pytree_kwargs(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, a, b):
                return arg1 + a["kw1"] + b[0], arg2 + a["kw2"] + b[1]

        kw_func = Module()
        args = (torch.ones(2, 3), torch.ones(3, 4))
        kwargs = {
            "a": {"kw1": torch.ones(2, 3), "kw2": torch.ones(3, 4)},
            "b": [torch.ones(2, 3), torch.ones(3, 4)],
        }
        self._test_export_same_as_eager(kw_func, args, kwargs)

    def test_export_func_with_default_kwargs(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, a, b=1):
                return arg1 + arg2, a["kw1"] + a["kw2"] + b

        kw_func = Module()

        class Module2(torch.nn.Module):
            def forward(self, arg1, arg2, a=1, b=2):
                return arg1 + a, arg2 + b

        kw_func2 = Module2()

        args = (torch.ones(6, 4), torch.ones(1, 1))
        kwargs1 = {"a": {"kw1": torch.ones(1, 1), "kw2": torch.ones(6, 4)}}
        kwargs2 = {"a": {"kw1": torch.ones(1, 1), "kw2": torch.ones(6, 4)}, "b": 2}
        self._test_export_same_as_eager(kw_func, args, kwargs1)
        self._test_export_same_as_eager(kw_func, args, kwargs2)
        kwargs3 = {"b": 1}
        self._test_export_same_as_eager(kw_func2, args, kwargs3)

    def test_kwargs_reorder(self):
        class M(torch.nn.Module):
            def forward(self, *, x, y, z):
                return x + y + z

        ep = export(
            M(), (), {"z": torch.ones(3), "y": torch.ones(3), "x": torch.ones(3)}
        )
        ep.module()(**{"z": torch.ones(3), "y": torch.ones(3), "x": torch.ones(3)})
        ep.module()(z=torch.ones(3), y=torch.ones(3), x=torch.ones(3))
        ep.module()(x=torch.ones(3), z=torch.ones(3), y=torch.ones(3))

    def test_set_example_inputs(self):
        class M(torch.nn.Module):
            def forward(self, a, *, x, y, z):
                return a, x + y + z

        inp = (
            (torch.ones(3),),
            {"z": torch.ones(3), "y": torch.ones(3), "x": torch.ones(3)},
        )
        ep = export(M(), inp[0], inp[1])
        ep.module()(*ep.example_inputs[0], **ep.example_inputs[1])

        ep.example_inputs = (
            (torch.ones(3),),
            {"x": torch.ones(3), "z": torch.ones(3), "y": torch.ones(3)},
        )
        ep.module()(*ep.example_inputs[0], **ep.example_inputs[1])

        with self.assertRaisesRegex(ValueError, "Example inputs should be a tuple"):
            ep.example_inputs = (torch.ones(3),)

        with self.assertRaisesRegex(ValueError, "Ran into a kwarg keyword mismatch"):
            ep.example_inputs = ((torch.ones(3),), {})

        with self.assertRaisesRegex(ValueError, "Trying to flatten user inputs"):
            ep.example_inputs = (
                (),
                {"x": torch.ones(3), "z": torch.ones(3), "y": torch.ones(3)},
            )

    def test_export_func_with_var_postional_args(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, *args):
                return arg1 + args[0], arg2 + args[1]

        kw_func = Module()
        args = (torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 3), torch.ones(3, 4))
        self._test_export_same_as_eager(kw_func, args)

    @testing.expectedFailureCppRuntime
    def test_export_module(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = torch.nn.Parameter(torch.ones(3, 4))
                self.p2 = torch.nn.Parameter(
                    CustomTensorPlainOut(
                        torch.ones(3, 4),
                        torch.ones(3, 4),
                    )
                )

            def forward(self, x):
                a = (2 * self.p1 + self.p2).sum()
                return x + a

        model = Foo()
        example_inputs = (torch.randn(3, 4),)
        ep = export(model, example_inputs, strict=False)
        before = list(ep.state_dict.keys())
        ep.run_decompositions()
        after = list(ep.state_dict.keys())
        self.assertEqual(before, after)

    def test_export_func_with_keyword_only_args(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, *args, kw1, kw2):
                return arg1 + args[0] + kw1, arg2 + args[1] + kw2

        kw_func = Module()
        args = (torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 3), torch.ones(3, 4))
        kwargs = {"kw1": torch.ones(2, 3), "kw2": torch.ones(3, 4)}
        self._test_export_same_as_eager(kw_func, args, kwargs)

    def test_export_func_with_var_keyword_args(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, *args, kw1, kw2, **kwargs):
                return (
                    arg1 + args[0] + kw1 + kwargs["kw3"],
                    arg2 + args[1] + kw2 + kwargs["kw4"],
                )

        kw_func = Module()
        args = (torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 3), torch.ones(3, 4))
        kwargs = {
            "kw1": torch.ones(2, 3),
            "kw2": torch.ones(3, 4),
            "kw3": torch.ones(2, 3),
            "kw4": torch.ones(3, 4),
        }
        self._test_export_same_as_eager(kw_func, args, kwargs)

    def test_unbacked_stack(self):
        class M(torch.nn.Module):
            def forward(self, x):
                nz = torch.nonzero(x)
                nz_size = nz.size(0)
                torch._check(nz_size % 4 == 0)

                # Create two tensors whose leading dimensions are equivalent at
                # runtime but expressed via different SymInt formulas.
                first = torch.zeros((nz_size // 2, 4))
                second = torch.zeros(((nz_size // 4) * 2, 4))
                return torch.stack([first, second], dim=0)

        inputs = (torch.ones((32,)),)

        ep = export(M(), inputs)
        orig_res = M()(*inputs)
        ep_res = ep.module()(*inputs)
        self.assertTrue(torch.allclose(orig_res, ep_res))

    def test_unbacked_slice_simple(self):
        class M(torch.nn.Module):
            def forward(self, scores, score_thr, topk: torch.Tensor, results=None):
                valid_mask = scores > score_thr
                scores = scores[valid_mask]
                valid_idxs = torch.nonzero(valid_mask).to(scores.device)

                num_topk = torch.minimum(topk, torch.tensor(valid_idxs.shape[0])).item()
                scores, idxs = scores.sort(descending=True)
                scores = scores[:num_topk]
                topk_idxs = valid_idxs[idxs[:num_topk]]
                keep_idxs, labels = topk_idxs.unbind(dim=1)

                return scores, labels, keep_idxs

        score = torch.tensor(
            [[0.1, 0.3, 0.2], [0.12, 0.7, 0.9], [0.02, 0.8, 0.08], [0.4, 0.1, 0.08]]
        )
        bbox_pred = torch.tensor([[0.2, 0.3], [0.4, 0.7], [0.1, 0.1], [0.5, 0.1]])
        score_thr = 0.15
        nms_pre = torch.tensor(4)
        inputs = (score, score_thr, nms_pre, dict(bbox_pred=bbox_pred))

        ep = export(M(), inputs)
        orig_res = M()(*inputs)
        ep_res = ep.module()(*inputs)
        self.assertTrue(torch.allclose(orig_res[0], ep_res[0]))
        self.assertTrue(torch.allclose(orig_res[1], ep_res[1]))
        self.assertTrue(torch.allclose(orig_res[2], ep_res[2]))

    def test_multidimensional_slicing(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                b = x.item()
                torch._check(b >= 0)
                torch._check(b < y.shape[0])

                return y[0, b]

        if is_non_strict_test(self._testMethodName):
            m = M()
            inp = (torch.tensor(4), torch.ones(10, 10))
            r = m(*inp)

            epm = export(m, inp).module()
            er = epm(*inp)

            self.assertTrue(torch.allclose(er, r))

    @testing.expectedFailureSerDerNonStrict
    @testing.expectedFailureCppRuntimeNonStrict
    def test_more_multidimensional_slicing(self):
        # Inputs: a 3d tensor t and a 1d tensor x of indices into t
        # Output: a 3-tuple of indices
        @torch.library.custom_op("demo::indices3d", mutates_args=())
        def indices3d(t: torch.Tensor, x: torch.Tensor) -> tuple[int, int, int]:
            assert t.ndim == 3
            assert x.ndim == 1 and x.shape[0] == 3
            return tuple(x[i].item() for i in range(3))

        # The meta-kernel for this op constrains the indices in x
        # to be within bounds of t via torch._checks.
        @torch.library.register_fake("demo::indices3d")
        def _(t, x):
            assert t.ndim == 3
            assert x.ndim == 1 and x.shape[0] == 3
            sizes = tuple(torch.library.get_ctx().new_dynamic_size() for i in range(3))
            for i, size in enumerate(sizes):
                torch._check(size >= 0)
                torch._check(size <= t.shape[i])
            return sizes

        # example inputs
        t = torch.randn([4, 5, 6])
        x = torch.tensor([2, 3, 4])

        def test(m, g, debug=False):
            # Dynamo does not yet support some cases of indexing tested here,
            # so don't export in strict mode.
            if is_non_strict_test(self._testMethodName):
                em = export(m, (t, x)).module()
                if debug:
                    print(em)
                self.assertTrue(torch.allclose(m(t, x), g(t, x)))
                self.assertTrue(torch.allclose(em(t, x), m(t, x)))

        # In the following series of test cases, M_* corresponds to indexing code
        # that a user might write, and G_* corresponds to equivalent code that
        # export might generate by rewriting the indexing in terms of a sequence
        # of lower-level ops.

        # indexing with ints
        class M_ints(torch.nn.Module):
            def forward(self, t, x):
                i, j, k = indices3d(t, x)
                return t[i, j, k]

        class G_ints(torch.nn.Module):
            def forward(self, t, x):
                i, j, k = indices3d(t, x)
                a = torch.select(t, 0, i)
                b = torch.select(a, 0, j)
                c = torch.select(b, 0, k)
                return c

        test(M_ints(), G_ints())

        # indexing with slices
        class M_slices(torch.nn.Module):
            def forward(self, t, x):
                i, j, k = indices3d(t, x)
                return t[:i, :j, :k]

        class G_slices(torch.nn.Module):
            def forward(self, t, x):
                i, j, k = indices3d(t, x)
                a = torch.narrow(t, 0, 0, i)
                b = torch.narrow(a, 1, 0, j)
                c = torch.narrow(b, 2, 0, k)
                return c

        test(M_slices(), G_slices())

        # indexing with ints and slices
        class M_ints_slices(torch.nn.Module):
            def forward(self, t, x):
                i, j, k = indices3d(t, x)
                return t[:i, j, :k]

        class G_ints_slices(torch.nn.Module):
            def forward(self, t, x):
                i, j, k = indices3d(t, x)
                a = torch.narrow(t, 0, 0, i)
                b = torch.select(a, 1, j)
                c = torch.narrow(b, 1, 0, k)
                return c

        test(M_ints_slices(), G_ints_slices())

        # indexing with ints and None
        class M_ints_None(torch.nn.Module):
            def forward(self, t, x):
                i, j, k = indices3d(t, x)
                return t[None, i, None]

        class G_ints_None(torch.nn.Module):
            def forward(self, t, x):
                i, j, k = indices3d(t, x)
                a = torch.unsqueeze(t, 0)
                b = torch.select(a, 1, i)
                c = torch.unsqueeze(b, 1)
                return c

        test(M_ints_None(), G_ints_None())

        # indexing with slices and None
        class M_slices_None(torch.nn.Module):
            def forward(self, t, x):
                i, j, k = indices3d(t, x)
                return t[:i, None, :j, None, None, :k]

        class G_slices_None(torch.nn.Module):
            def forward(self, t, x):
                i, j, k = indices3d(t, x)
                a = torch.narrow(t, 0, 0, i)
                b = torch.unsqueeze(a, 1)
                c = torch.narrow(b, 2, 0, j)
                d = torch.unsqueeze(c, 3)
                e = torch.unsqueeze(d, 4)
                f = torch.narrow(e, 5, 0, k)
                return f

        test(M_slices_None(), G_slices_None())

        # indexing with None, Ellipsis, and int
        class M_None_Ellipsis_int(torch.nn.Module):
            def forward(self, t, x):
                i, j, k = indices3d(t, x)
                return t[None, ..., None, j]

        class G_None_Ellipsis_int(torch.nn.Module):
            def forward(self, t, x):
                i, j, k = indices3d(t, x)
                a = torch.unsqueeze(t, 0)
                b = torch.unsqueeze(a, 3)
                c = torch.select(b, 4, j)
                return c

        test(M_None_Ellipsis_int(), G_None_Ellipsis_int())

        # indexing with slice, None, Ellipsis, and int
        class M_slice_None_Ellipsis_int(torch.nn.Module):
            def forward(self, t, x):
                i, j, k = indices3d(t, x)
                return t[:i, None, ..., None, j]

        class G_slice_None_Ellipsis_int(torch.nn.Module):
            def forward(self, t, x):
                i, j, k = indices3d(t, x)
                a = torch.narrow(t, 0, 0, i)
                b = torch.unsqueeze(a, 1)
                c = torch.unsqueeze(b, 3)
                d = torch.select(c, 4, j)
                return d

        test(M_slice_None_Ellipsis_int(), G_slice_None_Ellipsis_int())

    def test_sequential_slicing(self):
        # See https://github.com/pytorch/pytorch/issues/137455

        class TestModule1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.seq = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.Linear(4, 4),
                    torch.nn.Linear(4, 4),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # seq_last as local variable works
                seq_last = self.seq[1:]
                return seq_last(x)

        class TestModule2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.seq = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.Linear(4, 4),
                    torch.nn.Linear(4, 4),
                )
                # seq_last as initialized submodule works
                self.seq_last = self.seq[1:]

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.seq_last(x)

        inp = (torch.randn(4, 4),)
        for mod in [TestModule1(), TestModule2()]:
            epm = export(mod, inp).module()
            self.assertTrue(torch.allclose(epm(*inp), mod(*inp)))

    def test_unflatten_isinstance(self):
        class N(torch.nn.Module):
            def forward(self, x, b):
                if b:
                    return x + 1
                else:
                    return x + 2

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n = N()

            def forward(self, x):
                return self.n(x + 1, True) + self.n(x + 1, False)

        x = torch.zeros(4)
        types = {"n": N}
        ep = export(
            M(),
            (x,),
            preserve_module_call_signature=tuple(types.keys()),
        )
        ufm = torch.export.unflatten(ep)
        self.assertTrue(torch.allclose(ufm(x), x + 5))
        for fqn, mod in ufm.named_modules(remove_duplicate=False):
            if cls := types.get(fqn):
                ty = f"{cls.__module__}.{cls.__qualname__}"
                self.assertTrue(ty, mod.type_name())

    def test_unflatten_asserts(self):
        # TODO: strict-export fails
        class M1(torch.nn.Module):
            def forward(self, x, y):
                b = x.item()

                torch._check(b >= 0)
                torch._check(b < y.size(0))
                return y[:b]

        class M3(torch.nn.Module):
            def forward(self, x, y):
                b = x.item()

                torch._check(b >= 0)
                torch._check(b < y.size(0) * 2)
                return y[:b]

        class M2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.m1 = M1()
                self.m3 = M3()

            def forward(self, x, y):
                return self.m1(x, y) + self.m3(x, y)

        inputs = (torch.tensor(3), torch.randn(10))

        ep = torch.export.export(
            M2(), inputs, dynamic_shapes={"x": None, "y": (Dim("moo"),)}, strict=False
        )
        orig_res = M2()(*inputs)
        ep_res = ep.module()(*inputs)
        self.assertTrue(torch.allclose(orig_res[0], ep_res[0]))
        self.assertTrue(torch.allclose(orig_res[1], ep_res[1]))
        self.assertTrue(torch.allclose(orig_res[2], ep_res[2]))

        unflattened = torch.export.unflatten(ep)
        ep_res = unflattened(*inputs)
        self.assertTrue(torch.allclose(orig_res[0], ep_res[0]))
        self.assertTrue(torch.allclose(orig_res[1], ep_res[1]))
        self.assertTrue(torch.allclose(orig_res[2], ep_res[2]))

    def test_unflatten_placeholder_update_child2parent_swap(self):
        class Child(torch.nn.Module):
            def forward(self, x):
                torch.ops.aten.dropout_(x, 0.5, False)  # Applying dropout inplace
                return x - 2

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.child = Child()

            def forward(self, x):
                f1 = self.child(x)
                f2 = x * 4
                return f1 + f2

        m = Foo()
        inp = torch.ones(3, 10, dtype=torch.float32)
        orig_result = m(inp)

        if not is_retracebility_test(self._testMethodName):
            inp = torch.ones(3, 10, dtype=torch.float32)
            ep = export(m, (inp,), preserve_module_call_signature=("child",))
            unf = unflatten(ep)
            unf.print_readable()

            inp = torch.ones(3, 10, dtype=torch.float32)
            ep_result = ep.module()(inp)
            self.assertTrue(torch.allclose(ep_result, orig_result))

            unf.set_submodule("child", m.child)
            inp = torch.ones(3, 10, dtype=torch.float32)
            unf_result = unf(inp)
            self.assertTrue(torch.allclose(unf_result, orig_result))

    def test_unflatten_placeholder_update_grandchild2cousin_swap(self):
        class Grandchild(torch.nn.Module):
            def forward(self, x):
                a = x.to(torch.float32)  # .to is considered a mutation
                return x + 4, a

        class Child(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.grandchild = Grandchild()

            def forward(self, x):
                y, a = self.grandchild(x)
                return y + a

        class OtherGrandchild(torch.nn.Module):
            def forward(self, x):
                return x * 2

        class OtherChild(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.other_grandchild = OtherGrandchild()

            def forward(self, x):
                return x + self.other_grandchild(x)

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.child = Child()
                self.other_child = OtherChild()

            def forward(self, x):
                f1 = self.child(x)
                f2 = self.other_child(x)
                return f1 + f2

        inp = torch.ones(2, 3, dtype=torch.float32)
        orig_result = Foo()(inp)
        self.assertTrue(torch.allclose(orig_result, torch.ones(2, 3) * 9))

        if not is_retracebility_test(self._testMethodName):
            inp = torch.ones(2, 3, dtype=torch.float32)
            ep = export(Foo(), (inp,), preserve_module_call_signature=("child",))
            unf = unflatten(ep)

            inp = torch.ones(2, 3, dtype=torch.float32)
            ep_result = ep.module()(inp)
            self.assertTrue(torch.allclose(ep_result, orig_result))

            unf.set_submodule("child", Child())
            inp = torch.ones(2, 3, dtype=torch.float32)
            unf_result = unf(inp)
            self.assertTrue(torch.allclose(unf_result, orig_result))

    def test_unflatten_buffer_update_child2parent_swap(self):
        class Child(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.buf = torch.nn.Buffer(torch.tensor(10))

            def forward(self, x):
                self.buf.add_(1)
                return x + 2

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.child = Child()

            def forward(self, x):
                y = self.child(x)  # child.buf <- 10 + 1 = 11, x + 2 = 3
                x = y + self.child.buf  # 14
                y = self.child(x)  # child.buf <- 11 + 1 = 12, x + 2 = 16
                x = y + self.child.buf  # 28
                y = self.child(x)  # child.buf <- 12 + 1 = 13, x + 2 = 30
                x = y + self.child.buf  # 43
                return x

        inp = torch.ones(2, 3, dtype=torch.float32)
        orig_result = Foo()(inp)
        self.assertTrue(torch.allclose(orig_result, torch.ones(2, 3) * 43))

        if not is_retracebility_test(self._testMethodName):
            inp = torch.ones(2, 3, dtype=torch.float32)
            ep = export(Foo(), (inp,), preserve_module_call_signature=("child",))
            unf = unflatten(ep)

            inp = torch.ones(2, 3, dtype=torch.float32)
            ep_result = ep.module()(inp)
            self.assertTrue(torch.allclose(ep_result, orig_result))

            unf.set_submodule("child", Child())
            inp = torch.ones(2, 3, dtype=torch.float32)
            unf_result = unf(inp)
            self.assertTrue(torch.allclose(unf_result, orig_result))

    def test_export_func_with_var_keyword_pytree_args(self):
        class Module(torch.nn.Module):
            def forward(self, arg1, arg2, *args, kw1, kw2, **kwargs):
                return (
                    arg1 + arg2[0][0] + args[0] + kw1[0] + kwargs["kw3"][0],
                    arg2[1] + args[1] + kw2 + kwargs["kw4"],
                )

        kw_func = Module()
        args = (
            torch.ones(2, 3),
            [(torch.ones(2, 3),), torch.ones(3, 4)],
            torch.ones(2, 3),
            torch.ones(3, 4),
        )
        kwargs = {
            "kw1": (torch.ones(2, 3),),
            "kw2": torch.ones(3, 4),
            "kw3": (torch.ones(2, 3), torch.ones(3, 4)),
            "kw4": torch.ones(3, 4),
        }
        self._test_export_same_as_eager(kw_func, args, kwargs)

    @testing.expectedFailureSerDer  # we don't save placeholder metadata
    @testing.expectedFailureCppSerDes  # we don't save placeholder metadata
    @testing.expectedFailureSerDerNonStrict
    @testing.expectedFailureStrictV2
    def test_linear_conv(self):
        strict = True

        class MyLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.randn(20, 98)
                self.bias = torch.randn(20)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x):
                x_conv = self.conv(x)
                x_linear = self.linear(x_conv)
                return x_linear.cos()

        ep = export(Foo(), (torch.randn(20, 16, 50, 100),), strict=strict)
        for node in ep.graph.nodes:
            if (
                node.op == "placeholder"
                and node.name in ep.graph_signature.inputs_to_buffers
                or node.name in ep.graph_signature.inputs_to_parameters
            ):
                self.assertTrue("source_fn_stack" in node.meta)

    def test_dynamic_shapes_dataclass(self):
        torch.export.register_dataclass(
            Inp2,
            serialized_type_name="test_export_api_with_dynamic_shapes.Inp2",
        )

        class Foo(torch.nn.Module):
            def forward(self, inputs):
                return torch.matmul(inputs.a, inputs.b)

        foo = Foo()
        inputs = (Inp2(a=torch.randn(10, 2, 3), b=torch.randn(10, 3, 4)),)
        batch = Dim("batch")
        efoo = export(
            foo,
            inputs,
            dynamic_shapes={"inputs": [{0: batch}, {0: batch}]},
        )
        self.assertEqual(
            [
                # First dimension varies across strict and non-strict
                # since the source names are different, resulting in
                # different symbol names.
                str(node.meta["val"].shape[1:])
                for node in efoo.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([2, 3])", "torch.Size([3, 4])"],
        )

    @testing.expectedFailureCppSerDes
    def test_export_method(self):
        from torch._export.utils import sync_state, wrap_method

        strict = True

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.t = torch.nn.Buffer(torch.tensor(10))

            def forward(self, x):
                return self.foo(x) * self.bar(x)

            def foo(self, x):
                self.t.mul_(2)
                return x + self.t

            def bar(self, x):
                return x - self.t

        # exporting...
        em = M()
        ex = torch.randn(4)

        # ...foo
        epm_foo = export(
            wrap_method(em.foo),
            (ex,),
            dynamic_shapes={"x": (Dim.DYNAMIC,)},
            strict=strict,
        ).module()

        # ...bar
        epm_bar = export(
            wrap_method(em.bar),
            (ex,),
            dynamic_shapes=((Dim.DYNAMIC,),),
            strict=strict,
        ).module()

        if is_serdes_test(self._testMethodName):
            sync_state(epm_foo, epm_bar)

        # running...
        m = M()
        rx = torch.randn(5)

        self.assertTrue(torch.allclose(m.t, epm_foo.t))
        self.assertTrue(torch.allclose(m.t, epm_bar.t))

        # ...foo
        self.assertTrue(torch.allclose(epm_foo(rx), m.foo(rx)))
        self.assertTrue(torch.allclose(m.t, epm_foo.t))
        self.assertTrue(torch.allclose(m.t, epm_bar.t))

        # ...bar
        self.assertTrue(torch.allclose(epm_bar(rx), m.bar(rx)))
        self.assertTrue(torch.allclose(m.t, epm_foo.t))
        self.assertTrue(torch.allclose(m.t, epm_bar.t))

    def test_export_api_with_dynamic_shapes(self):
        from torch.export import Dim, dims

        # pass dynamic shapes of inputs [args]
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)

        foo = Foo()
        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch = Dim("batch")
        efoo = export(
            foo,
            inputs,
            dynamic_shapes={k: {0: batch} for k in ["x", "y"]},
        )
        self.assertEqual(efoo.module()(*inputs).shape, foo(*inputs).shape)

        foo = Foo()
        inputs = (torch.randn(10, 2, 3),)
        kwinputs = {"y": torch.randn(10, 3, 4)}
        batch = Dim("batch")
        efoo = export(
            foo, inputs, kwinputs, dynamic_shapes={k: {0: batch} for k in ["x", "y"]}
        )
        self.assertEqual(
            efoo.module()(*inputs, **kwinputs).shape, foo(*inputs, **kwinputs).shape
        )

        # pass dynamic shapes of inputs [partial, error]
        foo = Foo()
        inputs = (torch.randn(10, 2, 3),)
        kwinputs = {"y": torch.randn(10, 3, 4)}
        batch = Dim("batch")
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "You marked.*but your code specialized it to be a constant.*"
                "If you're using Dim.DYNAMIC, replace it with either Dim.STATIC or Dim.AUTO(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "batch = 10"
            ),
        ):
            export(
                foo,
                inputs,
                kwinputs,
                dynamic_shapes={"x": {0: batch}, "y": None},
            )

        # pass dynamic shapes of inputs [module]
        foo = Foo()
        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch = Dim("batch")
        efoo = export(
            foo,
            inputs,
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        self.assertEqual(efoo.module()(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [bounds, mostly shared]
        foo = Foo()
        inputs = (torch.randn(10, 3, 3), torch.randn(10, 3, 3))
        batch = Dim("batch", min=8, max=64)
        size = Dim("size")
        efoo = export(
            foo,
            inputs,
            dynamic_shapes={
                "x": (batch, size, size),
                "y": (batch, size, size),
            },
        )

        for node in efoo.graph_module.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(node.meta["val"].shape[1], node.meta["val"].shape[2])
        self.assertEqual(efoo.module()(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [multiple, mostly distinct]
        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch, M, K, N = dims("batch", "M", "K", "N")
        efoo = export(
            Foo(),
            inputs,
            dynamic_shapes={"x": (batch, M, K), "y": (batch, K, N)},
        )
        placeholders = [
            node.meta["val"].shape
            for node in efoo.graph_module.graph.nodes
            if node.op == "placeholder"
        ]
        self.assertEqual(
            placeholders[0][2],
            placeholders[1][1],
        )
        self.assertEqual(efoo.module()(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [dict]
        class Foo(torch.nn.Module):
            def forward(self, inputs):
                return torch.matmul(inputs["x"], inputs["y"])

        foo = Foo()
        inputs = ({"x": torch.randn(10, 2, 3), "y": torch.randn(10, 3, 4)},)
        batch = Dim("batch")
        efoo = export(
            foo, inputs, dynamic_shapes={"inputs": {k: {0: batch} for k in ["x", "y"]}}
        )
        self.assertEqual(
            [
                # First dimension varies across strict and non-strict
                # since the source names are different, resulting in
                # different symbol names.
                str(node.meta["val"].shape[1:])
                for node in efoo.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([2, 3])", "torch.Size([3, 4])"],
        )
        self.assertEqual(efoo.module()(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [list]
        class Foo(torch.nn.Module):
            def forward(self, inputs):
                return torch.matmul(inputs[0], inputs[1])

        foo = Foo()
        inputs = ([torch.randn(10, 2, 3), torch.randn(10, 3, 4)],)
        batch = Dim("batch")
        efoo = export(
            foo, inputs, dynamic_shapes={"inputs": [{0: batch} for _ in range(2)]}
        )
        self.assertEqual(
            [
                # First dimension varies across strict and non-strict
                # since the source names are different, resulting in
                # different symbol names.
                str(node.meta["val"].shape[1:])
                for node in efoo.graph_module.graph.nodes
                if node.op == "placeholder"
            ],
            ["torch.Size([2, 3])", "torch.Size([3, 4])"],
        )
        self.assertEqual(efoo.module()(*inputs).shape, foo(*inputs).shape)

        # pass dynamic shapes of inputs [pytree-registered classes]
        if HAS_TORCHREC:
            # skipping tests if torchrec not available
            class Foo(torch.nn.Module):
                def forward(self, kjt) -> torch.Tensor:
                    return kjt.values() + 0, kjt.offsets() + 0

            foo = Foo()
            kjt = KeyedJaggedTensor(
                values=torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                keys=["index_0", "index_1"],
                lengths=torch.IntTensor([0, 2, 0, 1, 1, 1, 0, 3]),
                offsets=torch.IntTensor([0, 0, 2, 2, 3, 4, 5, 5, 8]),
            )
            inputs = (kjt,)
            dim = Dim("dim")
            dim_plus_one = Dim("dim_plus_one")
            efoo = torch.export.export(
                foo,
                inputs,
                dynamic_shapes={
                    "kjt": [{0: dim}, None, {0: dim}, {0: dim_plus_one}, None, None]
                },
            )
            self.assertEqual(
                [out.shape for out in efoo.module()(*inputs)],
                [out.shape for out in foo(*inputs)],
            )

        # pass dynamic shapes of inputs [distinct, error]
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)

        foo = Foo()
        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch, M, K1, K2, N = dims("batch", "M", "K1", "K2", "N")
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Constraints violated \\(K2\\)!(.*\n)*.*"
                "K2.*and.*K1.*must always be equal(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "K2 = K1"
            ),
        ):
            export(
                foo,
                inputs,
                dynamic_shapes={"x": (batch, M, K1), "y": (batch, K2, N)},
            )

        # pass dynamic shapes of inputs [specialized, error]
        foo = Foo()
        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch, M, K1, N = dims("batch", "M", "K1", "N")
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "You marked.*but your code specialized it to be a constant.*"
                "If you're using Dim.DYNAMIC, replace it with either Dim.STATIC or Dim.AUTO(.*\n)*"
                "Suggested fixes:(.*\n)*.*"
                "K1 = 3"
            ),
        ):
            export(
                foo,
                inputs,
                dynamic_shapes={"x": (batch, M, K1), "y": (batch, None, N)},
            )

        # pass dynamic shapes of inputs [guards, error]
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                if x.shape[0] < 16 and y.shape[1] % 3 == 0:
                    return torch.matmul(x, y)
                else:
                    return x + y

        foo = Foo()
        inputs = (torch.randn(10, 2, 3), torch.randn(10, 3, 4))
        batch, M, K, N = dims("batch", "M", "K", "N")
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Constraints violated.*!(.*\n)*.*"
                "Not all values of K.*satisfy the generated guard(.*\n)*.*"
                "Not all values of batch.*satisfy the generated guard(.*\n)*.*"
                "Suggested fixes:(.*\n)*.*"
                "batch = Dim\\('batch', max=15\\)(.*\n)*.*"
                "K = 3\\*_K"
            ),
        ):
            export(
                foo,
                inputs,
                dynamic_shapes={"x": (batch, M, K), "y": (batch, K, N)},
            )

    def test_suggested_fixes_new_roots(self):
        from torch.export import dims

        # suggested fixes should introduce new root dim for modulo guard
        class Foo(torch.nn.Module):
            def forward(self, x, y, z):
                # dy = 3 * _dx
                # dx = 3 * _dx - 1
                # dz = 3 * _dx + 2
                # suggested fixes results will look something like
                # {"dx": {"eq": 3*_dx-1, "min": 5, "max": 36}, "dy": {"eq": dx+1}, ...}
                if x.shape[0] >= 5 and x.shape[0] <= 36 and y.shape[0] % 3 == 0:
                    return x + y[1:] + z[3:]

        foo = Foo()
        inputs = (
            torch.randn(
                11,
            ),
            torch.randn(
                12,
            ),
            torch.randn(
                14,
            ),
        )
        dx, dy, dz = dims("dx", "dy", "dz")
        dynamic_shapes = {
            "x": (dx,),
            "y": (dy,),
            "z": (dz,),
        }
        with self.assertRaisesRegex(  # figure out regex later
            torch._dynamo.exc.UserError,
            (
                "Constraints violated.*!(.*\n)*.*"
                "Suggested fixes(.*\n)*.*"
                "_dx = Dim\(\\'_dx\\', max=12\)(.*\n)*.*"
                "dx = 3\*_dx - 1(.*\n)*.*"
                "dy = 3\*_dx(.*\n)*.*"
                "dz = 3\*_dx \+ 2"
            ),
        ):
            export(Foo(), inputs, dynamic_shapes=dynamic_shapes)
        # retry export
        _dx = Dim("_dx", min=2, max=12)
        dynamic_shapes = {"x": (3 * _dx - 1,), "y": (3 * _dx,), "z": (3 * _dx + 2,)}
        export(Foo(), inputs, dynamic_shapes=dynamic_shapes)

    def test_refine_dynamic_shapes_from_suggested_fixes(self):
        from torch.export.dynamic_shapes import (
            refine_dynamic_shapes_from_suggested_fixes,
        )

        def helper(model, inputs, dynamic_shapes):
            # export, fail, parse & refine suggested fixes, re-export
            try:
                export(Foo(), inps, dynamic_shapes=dynamic_shapes)
                raise Exception("should have raised constraint violation error")
            except torch._dynamo.exc.UserError as exc:
                new_shapes = refine_dynamic_shapes_from_suggested_fixes(
                    exc.msg, dynamic_shapes
                )
                export(Foo(), inps, dynamic_shapes=new_shapes)
                return new_shapes

        # specialize dims + derived dims
        class Foo(torch.nn.Module):
            def forward(self, x, y, z):
                x0 = x + y[1:] + z[2:]
                x1 = x @ torch.randn(4, 4)
                return x0, x1

        inps = (
            torch.randn(
                4,
            ),
            torch.randn(
                5,
            ),
            torch.randn(
                6,
            ),
        )
        dx = Dim("dx", max=16)
        dynamic_shapes = {"x": (dx,), "y": (dx + 1,), "z": (dx + 2,)}
        new_shapes = helper(Foo(), inps, dynamic_shapes)
        self.assertEqual(new_shapes["x"][0], 4)
        self.assertEqual(new_shapes["z"][0], 6)

        # refine lower, upper bound
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                if x.shape[0] >= 6 and y.shape[0] <= 16:
                    return x * 2.0, y + 1

        inps = (torch.randn(16), torch.randn(12))
        dynamic_shapes = {"x": (Dim("dx"),), "y": (Dim("dy"),)}
        new_shapes = helper(Foo(), inps, dynamic_shapes)
        self.assertEqual(new_shapes["x"][0].min, 6)
        self.assertEqual(new_shapes["y"][0].max, 16)

        # divisiblity, will introduce new root
        class Foo(torch.nn.Module):
            def forward(self, x):
                if x.shape[0] >= 9:
                    return x.reshape([-1, 3])

        inps = (
            torch.randn(
                15,
            ),
        )
        dynamic_shapes = ((Dim("dx"),),)
        new_shapes = helper(Foo(), inps, dynamic_shapes)
        dim = new_shapes[0][0]
        root = dim.root
        self.assertEqual(dim.fn(2), 6)
        self.assertEqual(root.min, 3)

        # turn dim into derived dim/relation
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x + y[4:]

        inps = (torch.randn(6, 4), torch.randn(10, 4))
        dynamic_shapes = {
            "x": (Dim("dx0"), Dim("dx1")),
            "y": (Dim("dy0"), Dim("dy1")),
        }
        new_shapes = helper(Foo(), inps, dynamic_shapes)
        self.assertEqual(new_shapes["x"][0], new_shapes["y"][0].root)  # dy0 = dx0 + 4
        self.assertEqual(new_shapes["y"][0].fn(5), 9)
        self.assertEqual(new_shapes["x"][1], new_shapes["y"][1])  # dx1 = dy1

        # nested dynamic shapes spec
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                x0 = x[0]["data"] + x[1] + x[2][2:]
                x1 = y["a"] @ torch.randn(4, 4)
                x2 = y["b"] @ torch.randn(6, 6)
                return x0, x1, x2

        inps = (
            (
                {"data": torch.randn(4, 4)},
                torch.randn(4, 4),
                torch.randn(6, 4),
            ),
            {
                "a": torch.randn(8, 4),
                "b": torch.randn(9, 6),
            },
        )
        dynamic_shapes = {
            "x": (
                {"data": (Dim("dx00"), Dim("dx01"))},
                (Dim("dx10"), Dim("dx11")),
                (Dim("dx20"), Dim("dx21")),
            ),
            "y": {
                "a": (Dim("dya0"), Dim("dya1")),
                "b": (Dim("dyb0"), Dim("dyb1")),
            },
        }
        new_shapes = helper(Foo(), inps, dynamic_shapes)
        self.assertEqual(
            new_shapes["x"][0]["data"][0], new_shapes["x"][1][0]
        )  # dx10 = dx00
        self.assertEqual(
            new_shapes["x"][2][0].root, new_shapes["x"][0]["data"][0]
        )  # dx20 = dx00 + 2
        self.assertEqual(new_shapes["x"][2][0].fn(10), 12)
        self.assertEqual(
            new_shapes["x"][0]["data"][1], new_shapes["x"][1][1]
        )  # dx11 = dx01
        self.assertEqual(new_shapes["y"]["a"][1], 4)
        self.assertEqual(new_shapes["y"]["b"][1], 6)
        self.assertEqual(new_shapes["y"]["b"][0].__name__, "dyb0")  # unchanged

    def test_dynamic_shapes_spec_with_pytree(self):
        from torch.export import Dim, export
        from torch.utils._pytree import tree_map

        inputs = {
            "tensor": torch.randn(3),
            "dict_of_tensors": {k: torch.randn(3) for k in ["A", "B", "C", "D"]},
            "list_of_tensors": [torch.randn(3) for _ in range(4)],
        }

        batch = Dim("batch")
        # uniformly specify dynamic shapes for all inputs
        spec = tree_map(lambda x: {0: batch}, inputs)

        class Foo(torch.nn.Module):
            def forward(self, inputs):
                return (
                    inputs["tensor"]
                    + inputs["dict_of_tensors"]["A"]
                    + inputs["list_of_tensors"][0]
                )

        ep = export(Foo(), (inputs,), dynamic_shapes={"inputs": spec})
        input_shapes = [
            str(node.meta["val"].shape)
            for node in ep.graph_module.graph.nodes
            if node.op == "placeholder"
        ]
        self.assertEqual(len(input_shapes), 9)
        self.assertEqual(len(set(input_shapes)), 1)

    def test_error_does_not_reference_eager_fallback(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                y = x.nonzero()
                z = y.shape[0]
                if z > 2:
                    return x.cos()
                else:
                    return x.sin()

        fn_ddo = Module()
        if is_non_strict_test(self._testMethodName):
            error = torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode
            error_msg = r"Could not guard on data-dependent expression"
        else:
            error = torchdynamo.exc.UserError
            error_msg = r"^(?!.*fall back to eager).*"
        with self.assertRaisesRegex(error, error_msg):
            _ = export(fn_ddo, (torch.tensor([2, 3, 5]),))

    def test_pytree_register_data_class(self):
        @dataclass
        class MyDataClass:
            x: int
            y: int
            z: int = None

        dt = MyDataClass(x=3, y=4)
        flat, spec = tree_flatten(dt)
        self.assertTrue(spec, treespec_leaf())
        self.assertTrue(len(flat) == 1)

        torch.export.register_dataclass(
            MyDataClass,
            serialized_type_name="test_pytree_register_data_class.MyDataClass",
        )

        flat, spec = tree_flatten(dt)
        self.assertEqual(
            spec,
            TreeSpec(
                MyDataClass, [["x", "y"], ["z"]], [treespec_leaf(), treespec_leaf()]
            ),
        )
        self.assertEqual(flat, [3, 4])

        orig_dt = tree_unflatten(flat, spec)
        self.assertTrue(isinstance(orig_dt, MyDataClass))
        self.assertEqual(orig_dt.x, 3)
        self.assertEqual(orig_dt.y, 4)
        self.assertEqual(orig_dt.z, None)

        roundtrip_spec = treespec_loads(treespec_dumps(spec))
        self.assertEqual(roundtrip_spec, spec)

        @dataclass
        class MyOtherDataClass:  # the pytree registration don't allow registering the same class twice
            x: int
            y: int
            z: int = None

        # Override the registration with keep none fields
        register_dataclass_as_pytree_node(
            MyOtherDataClass,
            return_none_fields=True,
            serialized_type_name="test_pytree_regster_data_class.MyOtherDataClass",
        )

        dt = MyOtherDataClass(x=3, y=4)
        flat, spec = tree_flatten(dt)
        self.assertEqual(
            spec,
            TreeSpec(
                MyOtherDataClass,
                [["x", "y", "z"], []],
                [treespec_leaf(), treespec_leaf(), treespec_leaf()],
            ),
        )
        self.assertEqual(flat, [3, 4, None])

        orig_dt = tree_unflatten(flat, spec)
        self.assertTrue(isinstance(orig_dt, MyOtherDataClass))
        self.assertEqual(orig_dt.x, 3)
        self.assertEqual(orig_dt.y, 4)
        self.assertEqual(orig_dt.z, None)

        roundtrip_spec = treespec_loads(treespec_dumps(spec))
        self.assertEqual(roundtrip_spec, spec)

    def test_pytree_register_nested_data_class(self):
        @dataclass
        class Inner:
            x: int
            y: int

        @dataclass
        class Outer:
            xy: Inner
            ab: Inner

        xy = Inner(1, 2)
        ab = Inner(3, 4)
        dt = Outer(xy, ab)
        inp = {"dt1": (dt, ({},)), "dt2": ((torch.ones(1),), dt)}

        torch.export.register_dataclass(
            Inner, serialized_type_name="test_pytree_register_nested_data_class.Inner"
        )
        torch.export.register_dataclass(
            Outer, serialized_type_name="test_pytree_register_nested_data_class.Outer"
        )

        flat, spec = tree_flatten(inp)
        self.assertEqual(flat, [1, 2, 3, 4, torch.ones(1), 1, 2, 3, 4])

        unflat = tree_unflatten(flat, spec)
        self.assertEqual(unflat, inp)

        roundtrip_spec = treespec_loads(treespec_dumps(spec))
        self.assertEqual(roundtrip_spec, spec)

    def test_param_util(self):
        class Basic(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = torch.nn.Linear(10, 1)

            def forward(self, x):
                return self.lin(x)

        ep = export(Basic(), (torch.randn(5, 10),))
        num_params = 0
        params = []
        for node in ep.graph.nodes:
            if is_param(ep, node):
                num_params += 1
                params.append(get_param(ep, node))
        self.assertEqual(num_params, 2)
        self.assertEqual(params[0].shape, [1, 10])  # weight
        self.assertEqual(params[1].shape, [1])  # bias

    def test_buffer_util(self):
        ep = export(
            torch.nn.BatchNorm2d(100, affine=False), (torch.ones(20, 100, 35, 45),)
        )
        num_buffer = 0
        buffer = []

        for node in ep.graph.nodes:
            if is_buffer(ep, node):
                num_buffer += 1
                buffer.append(get_buffer(ep, node))
        self.assertEqual(num_buffer, 3)

        # The insertion order is not guaranteed to be same for strict vs
        # non-strict, so commenting this out.
        # self.assertEqual(buffer[0].shape, torch.Size([100]))  # running_mean
        # self.assertEqual(buffer[1].shape, torch.Size([100]))  # running_var
        # self.assertEqual(buffer[2].shape, torch.Size([]))  # num_batches_tracked

    def test_export_dynamo_config(self):
        class MyModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.LSTM(input_size=4, hidden_size=5, num_layers=1)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.lstm(inputs)

        config = DEFAULT_EXPORT_DYNAMO_CONFIG
        mod = MyModule()

        @contextmanager
        def _patch_config(kwargs):
            orig_config_dict = dataclasses.asdict(config)

            try:
                for k, v in kwargs.items():
                    setattr(config, k, v)
                yield
            finally:
                for k, v in orig_config_dict.items():
                    setattr(config, k, v)

        inp = (torch.rand(5, 4),)
        exported_program = export(mod, inp, strict=True)

        with _patch_config({"allow_rnn": False}):
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported,
                "Dynamo does not support RNN, GRU, or LSTM.",
            ):
                _ = export(mod, inp, strict=True)

    def test_device_to_static(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return x.to("cpu")

        ep = export(Module(), (torch.tensor(1, device="cpu"),))
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)

        if is_training_ir_test(self._testMethodName):
            # aten.to will just specialize by decomposing to a no-op
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                ],
            )
        else:
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten.to.dtype_layout,
                ],
            )

        ep = ep.run_decompositions({})
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        self.assertEqual(len(ops), 1)

    def test_device_to_dynamic(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return x.to("cpu")

        ep = export(
            Module(),
            (torch.tensor([1, 2], device="cpu"),),
            dynamic_shapes={"x": {0: Dim("i")}},
        )
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)

        if is_training_ir_test(self._testMethodName):
            # aten.to will just specialize by decomposing to a no-op
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                ],
            )
        else:
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten.to.dtype_layout,
                ],
            )

        ep = ep.run_decompositions({})
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        self.assertEqual(len(ops), 1)

    def test_device_to_mutation(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                y = x.to("cpu")
                y.add_(1)
                return y, x

        ep = export(Module(), (torch.tensor(1, device="cpu"),))
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        if is_training_ir_test(self._testMethodName):
            # aten.to decomposes to no-op, add_ decomposes to functional variant
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten.add.Tensor,
                ],
            )
        else:
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten.to.dtype_layout,
                    torch.ops.aten.add_.Tensor,
                ],
            )

        # test mutation
        x = torch.tensor(2, device="cpu")
        y, _ = ep.module()(x)
        self.assertEqual(x.item(), 3)
        self.assertEqual(id(y), id(x))

        # test decomp ep
        ep = ep.run_decompositions({})
        for node in ep.graph.nodes:
            if node.op == "call_function":
                self.assertNotEqual(node.target, torch.ops.aten.to.dtype_layout)

        # test mutation for decomposed program
        y, _ = ep.module()(x)
        self.assertEqual(x.item(), 4)
        self.assertEqual(id(y), id(x))

    @requires_gpu
    @testing.expectedFailureCppRuntime
    def test_device_to_gpu(self):
        class Foo(torch.nn.Module):
            def forward(self, x):
                return x.to("cpu")

        ep = export(Foo(), (torch.randn(64).to(GPU_TYPE),))
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        if is_training_ir_test(self._testMethodName):
            # aten.to decomposes to _to_copy
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten._to_copy.default,
                ],
            )
        else:
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten.to.dtype_layout,
                ],
            )

        # Check device assertion
        with self.assertRaisesRegex(RuntimeError, "Tensor device mismatch!"):
            ep.module()(torch.randn(64))

        ep = ep.run_decompositions()
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        self.assertEqual(len(ops), 2)
        self.assertEqual(
            ops,
            [
                torch.ops.aten._assert_tensor_metadata.default,
                torch.ops.aten._to_copy.default,
            ],
        )

        # Check device assertion again after decomp
        with self.assertRaisesRegex(RuntimeError, "Tensor device mismatch!"):
            ep.module()(torch.randn(64))

    def test_tensor_constant_aten_to(self):
        class Module(torch.nn.Module):
            def __init__(self):
                super(Module, self).__init__()
                self.t = torch.tensor([1.0])

            def forward(self, x):
                return x + self.t.to(torch.float64)

        inputs = (torch.randn(1, 10),)
        model = Module()
        ep = export(model, inputs).run_decompositions({})
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        self.assertGreater(len(ops), 0)
        self.assertIn(torch.ops.aten._to_copy.default, ops)

        self.assertEqual(ep.module()(*inputs), model(*inputs))

    def test_export_aten_to_unflatten(self):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.sum()

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bar = Bar()

            def forward(self, x):
                to = x.to(torch.float)
                return self.bar(to).sum()

        inp = torch.randn(4, 4)

        ep = export(
            Foo(), (inp,), strict=False, preserve_module_call_signature=("bar",)
        )
        mod = ep.module()
        self.assertTrue(torch.allclose(mod(inp), Foo()(inp)))

    @testing.expectedFailureLegacyExportNonStrict
    @testing.expectedFailureLegacyExportStrict
    @testing.expectedFailureRetraceabilityNonStrict  # when we retrace, ep.module() is hierarchical
    @testing.expectedFailureRetraceability  # when we retrace, ep.module() is hierarchical
    def test_export_aten_to_unflatten_subclass(self):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.sum()

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bar = Bar()
                self.param = torch.nn.Parameter(
                    TwoTensor(torch.ones(4, 4), torch.ones(4, 4))
                )

            def forward(self, x):
                to = self.param.to(torch.float)
                return (self.bar(to).sum() + x.sum()).get_elem_a()

        inp = torch.randn(4, 4)

        with self.assertRaisesRegex(
            ValueError, "It looks like p_param is a tensor subclass."
        ):
            export(
                Foo(), (inp,), strict=False, preserve_module_call_signature=("bar",)
            ).run_decompositions({})

    def test_export_aten_to_unflatten_subclass_pre_dispatch(self):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.sum()

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bar = Bar()
                self.param = torch.nn.Parameter(
                    TwoTensor(torch.ones(4, 4), torch.ones(4, 4))
                )

            def forward(self, x):
                to = self.param.to(torch.float)
                return (self.bar(to).sum() + x.sum()).get_elem_a()

        inp = torch.randn(4, 4)

        ep = torch.export.export(
            Foo(), (inp,), strict=False, preserve_module_call_signature=("bar",)
        )
        unflat = unflatten(ep).bar
        self.assertExpectedInline(
            str(unflat.graph).strip(),
            """\
graph():
    %_positional_arg_0 : [num_users=1] = placeholder[target=_positional_arg_0]
    %_spec_0 : [num_users=1] = get_attr[target=_spec_0]
    %tree_flatten_spec : [num_users=1] = call_function[target=torch.fx._pytree.tree_flatten_spec](args = (((%_positional_arg_0,), {}), %_spec_0), kwargs = {})
    %to : [num_users=1] = call_function[target=operator.getitem](args = (%tree_flatten_spec, 0), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%to,), kwargs = {})
    %_spec_1 : [num_users=1] = get_attr[target=_spec_1]
    %tree_unflatten : [num_users=1] = call_function[target=torch.utils._pytree.tree_unflatten](args = ((%sum_1,), %_spec_1), kwargs = {})
    return tree_unflatten""",
        )

        with self.assertRaisesRegex(
            ValueError, "It looks like p_param is a tensor subclass."
        ):
            ep.run_decompositions()

    def test_float_conversion(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return x.float()

        ep = export(Module(), (torch.tensor(1, dtype=torch.float),))
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        if is_training_ir_test(self._testMethodName):
            # .float() decomposes to no-op
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                ],
            )
        else:
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten.to.dtype,
                ],
            )

        ep = ep.run_decompositions({})
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        self.assertEqual(len(ops), 1)

        # test aliasing
        x = torch.tensor(1, dtype=torch.float)
        out = ep.module()(x)
        self.assertEqual(id(x), id(out))

    def test_float_conversion_from_int(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return x.float()

        ep = export(Module(), (torch.tensor(1, dtype=torch.int32),))
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        if is_training_ir_test(self._testMethodName):
            # .float() decomposes to _to_copy()
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten._to_copy.default,
                ],
            )
        else:
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten.to.dtype,
                ],
            )

        # Raises error because the input dtype is not the same as the input
        # tensor when exporting.
        with self.assertRaisesRegex(RuntimeError, "Tensor dtype mismatch!"):
            ep.module()(torch.tensor(1, dtype=torch.float32))

        ep = ep.run_decompositions({})
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        self.assertEqual(
            ops,
            [
                torch.ops.aten._assert_tensor_metadata.default,
                torch.ops.aten._to_copy.default,
            ],
        )

        # Check dtype assertion again after decomp
        with self.assertRaisesRegex(RuntimeError, "Tensor dtype mismatch!"):
            ep.module()(torch.tensor(1, dtype=torch.float32))

        self.assertEqual(ep.module()(torch.tensor(1, dtype=torch.int32)), 1)

    def test_device_to_mutation_float(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                y = x.float()
                y.add_(1)
                return y, x

        ep = export(Module(), (torch.tensor(1, dtype=torch.float),))
        ops = []
        for node in ep.graph.nodes:
            if node.op == "call_function":
                ops.append(node.target)
        if is_training_ir_test(self._testMethodName):
            # aten.to decomposes to no-op, add_ decomposes to functional variant
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten.add.Tensor,
                ],
            )
        else:
            self.assertEqual(
                ops,
                [
                    torch.ops.aten._assert_tensor_metadata.default,
                    torch.ops.aten.to.dtype,
                    torch.ops.aten.add_.Tensor,
                ],
            )

        # test mutation
        x = torch.tensor(2, dtype=torch.float)
        y, _ = ep.module()(x)
        self.assertEqual(x.item(), 3.0)
        self.assertEqual(id(y), id(x))

        # test decomp ep
        ep = ep.run_decompositions({})
        for node in ep.graph.nodes:
            if node.op == "call_function":
                self.assertNotEqual(node.target, torch.ops.aten.to.dtype)

        # test mutation for decomposed program
        y, _ = ep.module()(x)
        self.assertEqual(x.item(), 4.0)
        self.assertEqual(id(y), id(x))

    def test_module(self):
        class MyLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.randn(20, 98)
                self.bias = torch.randn(20)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x):
                a, b = x
                a_conv = self.conv(a)
                a_linear = self.linear(a_conv)
                b_conv = self.conv(b)
                b_linear = self.linear(b_conv)
                return (
                    a_linear.cos() + b_linear.sin(),
                    a_linear.sin() + b_linear.cos(),
                )

        inp_container = ((torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)),)

        ep = export(Foo(), inp_container)
        ep_rexported = export(ep.module(), inp_container)

        inp_test = ((torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)),)

        self.assertTrue(
            torch.allclose(
                ep.module()(*inp_test)[0], ep_rexported.module()(*inp_test)[0]
            )
        )
        self.assertTrue(
            torch.allclose(
                ep.module()(*inp_test)[1], ep_rexported.module()(*inp_test)[1]
            )
        )

    def test_use_embedding_twice(self):
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(4, 4)

            def forward(self, x):
                return self.embed(x) + self.embed.weight[x]

        inputs = (torch.tensor([0, 1, 2, 3]),)
        ep = export(Foo(), inputs)

    def test_module_with_dict_container_inp_out(self):
        class MyLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.randn(20, 98)
                self.bias = torch.randn(20)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x):
                a1, a2 = x["a"]
                b = x["b"]
                a1_conv = self.conv(a1)
                a1_linear = self.linear(a1_conv)
                a2_conv = self.conv(a2)
                a2_linear = self.linear(a2_conv)
                b_conv = self.conv(b)
                b_linear = self.linear(b_conv)
                return {
                    "a": a1_linear.cos() + b_linear.sin(),
                    "b": a2_linear.sin() + b_linear.cos(),
                }

        inp_container = (
            {
                "a": (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)),
                "b": torch.randn(20, 16, 50, 100),
            },
        )

        ep = export(Foo(), inp_container)
        ep_rexported = export(ep.module(), inp_container)

        inp_test = (
            {
                "a": (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)),
                "b": torch.randn(20, 16, 50, 100),
            },
        )

        self.assertTrue(
            torch.allclose(
                ep.module()(*inp_test)["a"], ep_rexported.module()(*inp_test)["a"]
            )
        )
        self.assertTrue(
            torch.allclose(
                ep.module()(*inp_test)["b"], ep_rexported.module()(*inp_test)["b"]
            )
        )

    def test_args_type_checked(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x + 1

        inp = torch.rand(2, 2)
        with self.assertRaisesRegex(torch._dynamo.exc.UserError, "to be a tuple"):
            # Intentionally not wrapping `inp` in a tuple to trigger the error
            _ = export(M(), inp)

    def test_decomp_item_in_prim_before_decomposition(self):
        class M(torch.nn.Module):
            def forward(self, x):
                torch.ops.aten._assert_async.msg(torch.tensor(True), "Fail")
                return x

        ep = export(M(), (torch.randn(2, 2),))
        FileCheck().check_count(
            "torch.ops.aten._assert_async.msg", 1, exactly=True
        ).run(ep.graph_module.code)

    def test_decomp_item_in_prim_after_decomposition(self):
        class M(torch.nn.Module):
            def forward(self, x):
                torch.ops.aten._assert_async.msg(torch.tensor(True), "Fail")
                return x

        decomp_table = {**default_decompositions(), **decomposition_table}

        ep = torch.export.export(M(), (torch.randn(2, 2),)).run_decompositions(
            decomp_table
        )

        self.assertExpectedInline(
            str(ep.graph_module.code).strip(),
            """\
def forward(self, c_lifted_tensor_0, x):
    clone = torch.ops.prims.clone.default(c_lifted_tensor_0, memory_format = torch.preserve_format);  c_lifted_tensor_0 = None
    _assert_async = torch.ops.aten._assert_async.msg(clone, 'Fail');  clone = _assert_async = None
    return (x,)""",
        )

    def test_decomp_batch_norm_functional_predispatch(self):
        class ConvBatchnorm(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 3, 1, 1)
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return (x,)

        mod = ConvBatchnorm()
        mod.eval()
        inp = torch.randn(1, 1, 3, 3)

        gm = torch.export.export(mod, (inp,)).module()
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    conv_weight = self.conv.weight
    conv_bias = self.conv.bias
    bn_weight = self.bn.weight
    bn_bias = self.bn.bias
    bn_running_mean = self.bn.running_mean
    bn_running_var = self.bn.running_var
    bn_num_batches_tracked = self.bn.num_batches_tracked;  bn_num_batches_tracked = None
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    conv2d = torch.ops.aten.conv2d.default(x, conv_weight, conv_bias);  x = conv_weight = conv_bias = None
    batch_norm = torch.ops.aten.batch_norm.default(conv2d, bn_weight, bn_bias, bn_running_mean, bn_running_var, False, 0.1, 1e-05, True);  conv2d = bn_weight = bn_bias = bn_running_mean = bn_running_var = None
    return pytree.tree_unflatten((batch_norm,), self._out_spec)""",
        )

        mod.train()
        gm_train = torch.export.export(mod, (inp,)).module()
        self.assertExpectedInline(
            str(gm_train.code).strip(),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    conv_weight = self.conv.weight
    conv_bias = self.conv.bias
    bn_weight = self.bn.weight
    bn_bias = self.bn.bias
    bn_running_mean = self.bn.running_mean
    bn_running_var = self.bn.running_var
    bn_num_batches_tracked = self.bn.num_batches_tracked
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    conv2d = torch.ops.aten.conv2d.default(x, conv_weight, conv_bias);  x = conv_weight = conv_bias = None
    add_ = torch.ops.aten.add_.Tensor(bn_num_batches_tracked, 1);  bn_num_batches_tracked = add_ = None
    batch_norm = torch.ops.aten.batch_norm.default(conv2d, bn_weight, bn_bias, bn_running_mean, bn_running_var, True, 0.1, 1e-05, True);  conv2d = bn_weight = bn_bias = bn_running_mean = bn_running_var = None
    return pytree.tree_unflatten((batch_norm,), self._out_spec)""",
        )

    def test_constrain_size_in_eager(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                n = x.max().item()
                torch._check(n >= 0)
                return y + n

        fn = Module()
        ep = export(
            fn,
            (torch.randint(1, 2, (2, 2)), torch.randint(3, 5, (2, 3))),
        )
        test_inp = (torch.randint(1, 2, (2, 2)), torch.randint(3, 5, (2, 3)))
        self.assertTrue(torch.allclose(ep.module()(*test_inp), fn(*test_inp)))

    def test_constrain_size_with_constrain_value(self):
        class Module(torch.nn.Module):
            def forward(self, x, y):
                n = x.max().item()
                torch._check(n >= 2)
                torch._check(n <= 10)
                return y + n

        fn = Module()
        with self.assertRaisesRegex(
            RuntimeError, r"Expected cond to be True, but got False"
        ):
            _ = fn(torch.randint(1, 2, (2, 2)), torch.randint(3, 5, (2, 3)))

        ep = export(
            fn,
            (torch.randint(3, 4, (2, 2)), torch.randint(3, 5, (2, 3))),
        )
        with self.assertRaisesRegex(
            RuntimeError, r"Runtime assertion failed for expression u[\d+] \>\= 2"
        ):
            test_inp = (torch.randint(1, 2, (2, 2)), torch.randint(3, 5, (2, 3)))
            _ = ep.module()(*test_inp)

    def test_while_loop_simple(self):
        class Simple(torch.nn.Module):
            def forward(self, ci, a, b):
                def cond_fn(i, x, y):
                    return i > 0

                def body_fn(i, x, y):
                    return i - 1, x + y, y - x

                return torch._higher_order_ops.while_loop(cond_fn, body_fn, [ci, a, b])

        example_inputs = (
            torch.tensor(1),
            torch.randn(10, 20),
            torch.randn(10, 20),
        )
        ep = export(Simple(), example_inputs)
        self.assertEqual(ep.module()(*example_inputs), Simple()(*example_inputs))

    def test_constrain_size_with_various_cases(self):
        class Module1(torch.nn.Module):
            def forward(self, x, y):
                n = x.item()
                torch._check(n >= 0)
                return y.sum() + torch.ones(n, 5).sum()

        case1 = Module1()

        class Module2(torch.nn.Module):
            def forward(self, x, y):
                n = x.item()
                torch._check(n >= 0)
                torch._check(n <= 6)
                return y.sum() + torch.ones(n, 5).sum()

        case2 = Module2()

        class Module3(torch.nn.Module):
            def forward(self, x, y):
                n = x.item()
                torch._check(n >= 0)
                torch._check(n <= 1)
                return y.sum() + torch.ones(n, 5).sum()

        case3 = Module3()

        class Module4(torch.nn.Module):
            def forward(self, x, y):
                n = x.item()
                torch._check(n >= 2)
                return y.sum() + torch.ones(n, 5).sum()

        case4 = Module4()

        class Module5(torch.nn.Module):
            def forward(self, x, y):
                n = x.item()
                torch._check(n >= 1)
                return y.sum() + torch.ones(n, 5).sum()

        case5 = Module5()

        ep = export(case1, (torch.tensor(1), torch.ones(4, 5)))

        with self.assertRaisesRegex(
            RuntimeError, r"Expected cond to be True, but got False"
        ):
            _ = case1(torch.tensor(-1), torch.randn(4, 5))

        self.assertTrue(
            torch.allclose(
                ep.module()(torch.tensor(1), torch.ones(4, 5)),
                case1(torch.tensor(1), torch.ones(4, 5)),
            )
        )

        ep = export(case2, (torch.tensor(5), torch.randn(4, 5)))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected cond to be True, but got False",
        ):
            _ = case2(torch.tensor(7), torch.randn(4, 5))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected cond to be True, but got False",
        ):
            _ = case2(torch.tensor(9), torch.randn(4, 5))

        self.assertTrue(
            torch.allclose(
                ep.module()(torch.tensor(5), torch.ones(4, 5)),
                case2(torch.tensor(5), torch.ones(4, 5)),
            )
        )

        _ = case3(torch.tensor(1), torch.randn(4, 5))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected cond to be True, but got False",
        ):
            _ = case4(torch.tensor(1), torch.randn(4, 5))

        ep = export(case4, (torch.tensor(5), torch.randn(4, 5)))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected cond to be True, but got False",
        ):
            _ = case4(torch.tensor(1), torch.randn(4, 5))

        self.assertTrue(
            torch.allclose(
                ep.module()(torch.tensor(5), torch.ones(4, 5)),
                case4(torch.tensor(5), torch.ones(4, 5)),
            )
        )

        ep = export(case5, (torch.tensor(5), torch.randn(4, 5)))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected cond to be True, but got False",
        ):
            _ = case5(torch.tensor(0), torch.randn(4, 5))

        self.assertTrue(
            torch.allclose(
                ep.module()(torch.tensor(5), torch.ones(4, 5)),
                case5(torch.tensor(5), torch.ones(4, 5)),
            )
        )

    @testing.expectedFailureStrictV2
    def test_automatic_constrain_size(self):
        class M(torch.nn.Module):
            def forward(self, x, y):
                n = x.item()
                return y.sum() + torch.ones(n, 5).sum()

        ep = export(M(), (torch.tensor(1), torch.ones(4, 5)))

        # This is because we insert sym_constrain_range in the graph now
        error_msg = r".* failed for expression u0 >= 0 on node .*"
        with self.assertRaisesRegex(RuntimeError, error_msg):
            _ = ep.module()(torch.tensor(-1), torch.randn(4, 5))

        self.assertTrue(
            torch.allclose(
                ep.module()(torch.tensor(1), torch.ones(4, 5)),
                M()(torch.tensor(1), torch.ones(4, 5)),
            )
        )

    def test_cleanup_dynamic_markers(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, inputs):
                x, y = inputs["x"], inputs["y"]
                return x + y

        inputs = (
            {
                "x": torch.randn(4, 8),
                "y": torch.randn(4, 8),
            },
        )
        shapes = {
            "inputs": {
                "x": (Dim.AUTO, Dim.STATIC),
                "y": (Dim.DYNAMIC, Dim.STATIC),
            },
        }
        ep = export(Foo(), inputs, dynamic_shapes=shapes)
        for tensor in inputs[0].values():
            for attr in [
                "_dynamo_weak_dynamic_indices",
                "_dynamo_dynamic_indices",
                "_dynamo_dynamic_range",
                "_dynamo_static_indices",
                "_dynamo_unbacked_indices",
            ]:
                self.assertFalse(hasattr(tensor, attr))

    @testing.expectedFailureCppRuntime
    def test_while_loop_index_assertions(self):
        from torch._higher_order_ops import while_loop

        class Foo(torch.nn.Module):
            def forward(self, x):
                def cond_fn(idx, acc):
                    i = idx.item()
                    return i < x.size(0)

                def body_fn(idx, acc):
                    # this check_is_size call needs to be traced by this subgraph for the select call,
                    # it can't be in the cond graph, as that fires & fails right before loop termination.
                    i = idx.item()
                    return idx + 1, acc + x[i]

                acc = torch.zeros(x.size(1))
                n = torch.full((), 0, dtype=torch.int64)
                _, out = while_loop(cond_fn, body_fn, [n, acc])
                return out

        x = torch.randn(8, 4)
        ep = export(Foo(), (x,), strict=False)
        self.assertTrue(torch.allclose(x.sum(dim=0), ep.module()(x)))

    @testing.expectedFailureCppRuntime
    def test_while_loop_assert_separation(self):
        from torch._higher_order_ops import while_loop

        class Bar(torch.nn.Module):
            def forward(self, idx, x):
                i = idx.item()

                def cond_fn(idx, x):
                    i = idx.item()
                    torch._check(i != 5)
                    return i <= 9

                def body_fn(idx, x):
                    i = idx.item()
                    torch._check(i % 2 == 0)
                    return idx + 2, x + i

                return while_loop(cond_fn, body_fn, [idx, x + i])

        inps = (torch.tensor([0]), torch.zeros(1))
        ep = export(Bar(), inps, strict=False)
        i, out = ep.module()(*inps)
        self.assertEqual(i, 10)
        self.assertEqual(out.item(), 20)

        # check assertions are separate for each subgraph
        with self.assertRaisesRegex(
            RuntimeError, r"Runtime assertion failed for expression Ne\(u[\d]+, 5\).*"
        ):
            ep.graph_module.while_loop_cond_graph_0(torch.tensor([5]), torch.zeros(1))
        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression Eq\(PythonMod\(u[\d]+, 2\), 0\).*",
        ):
            ep.graph_module.while_loop_body_graph_0(torch.tensor([5]), torch.zeros(1))

    @testing.expectedFailureStrictV2
    def test_constrain_decomp(self) -> None:
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.freq = torch.ones(5, 5)

            def forward(self, start_pos: torch.Tensor):
                pos = start_pos.item()
                torch._check(pos >= 0)
                torch._check(pos <= 4)
                return self.freq[pos] * self.freq[pos]

        ep = export(M(), (torch.tensor(1),))
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 2, exactly=True
        ).run(ep.graph_module.code)

        decompose_ep = ep.run_decompositions()
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 2, exactly=True
        ).run(ep.graph_module.code)

    def test_mixed_input(self):
        class Module(torch.nn.Module):
            def forward(self, a, b, alpha: int):
                return torch.add(a, b, alpha=alpha)

        func = Module()

        a = torch.rand(1, 2)
        b = torch.rand(1, 2)
        alpha = 10

        exported = export(func, (a, b, alpha))
        for node in exported.graph_module.graph.nodes:
            if node.op == "placeholder":
                self.assertTrue(isinstance(node.meta["val"], (Tensor, int)))

    @testing.expectedFailureRetraceability  # size gets unflattened into a tuple
    def test_size_input(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, theta, size):
                return torch.nn.functional.affine_grid(theta, size, align_corners=None)

        model = Model()
        theta = torch.ones((1, 2, 3))
        size = torch.Size((1, 3, 24, 24))
        inp = (theta, size)
        eager_result = model(*inp)

        ep = export(model, inp)

        epm = ep.module()
        ep_result = epm(*inp)
        self.assertTrue(torch.allclose(ep_result, eager_result))

        args, _kwargs = ep.example_inputs
        self.assertTrue(torch.allclose(arg, i) for arg, i in zip(args, inp))

    def test_tensor_constant_with_wrapped_method(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.constant = torch.ones(4, 4)

            def forward(self, x):
                return x + self.constant, self.constant

        class Wrapper(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, *arg, **kwargs):
                return self.fn(*arg, **kwargs)

        inp = (torch.zeros(4, 4),)

        def test(m):
            m_result = m(*inp)
            ep_result = export(m, inp).module()(*inp)
            for m_t, ep_t in zip(m_result, ep_result):
                self.assertTrue(torch.allclose(m_t, ep_t))

        test(M())
        test(Wrapper(M().forward))

    def test_export_with_inline_constraints(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                a = x.item()
                torch._check(a >= 4)
                torch._check(a <= 7)
                return torch.randn((a, 4))

        f = Module()
        ep = export(f, (torch.tensor([5]),))
        self.assertEqual(ep.module()(torch.tensor([6])).shape, (6, 4))

        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 2, exactly=True
        ).run(ep.graph_module.code)

        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression u[\d+] \<\= 7",
        ) as cm:
            ep.module()(torch.tensor([30]))

    def test_export_with_inline_constraints_complex(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                a = x.item()
                torch._check(a >= 4)
                torch._check(a <= 7)
                randn = torch.randn((a, 4))

                return torch.cat((randn.transpose(0, 1), torch.zeros(6, a)), 0)

        f = Module()
        ep = export(f, (torch.tensor([6]),))
        self.assertEqual(ep.module()(torch.tensor([5])).shape, (10, 5))
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 2, exactly=True
        ).run(ep.graph_module.code)

    def test_to_module_with_mutated_buffer(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.zeros(1))

            def forward(self, x):
                self.buf.add_(1)
                return x.sum() + self.buf.sum()

        exported = export(Foo(), (torch.ones(5, 5),))
        stateful_gm = exported.module()
        export_return_val = stateful_gm(torch.ones(5, 5))
        eager = Foo()
        eager_return_val = eager(torch.ones(5, 5))
        self.assertTrue(torch.allclose(eager_return_val, export_return_val))

        for name, buffer in stateful_gm.named_buffers():
            self.assertTrue(torch.allclose(torch.ones(1), buffer))

        changed = stateful_gm.graph.eliminate_dead_code()
        self.assertFalse(changed)
        self.assertTrue(
            torch.allclose(stateful_gm(torch.ones(5, 5)), eager(torch.ones(5, 5)))
        )

        for name, buffer in stateful_gm.named_buffers():
            self.assertTrue(torch.allclose(torch.tensor(2, dtype=torch.float), buffer))

    def test_to_module_with_mutated_buffer_multiple(self):
        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))

            def forward(self, x):
                self.buf.add_(1)
                return x.sum() + self.buf.sum()

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.zeros(1))
                self.bar = Bar()

            def forward(self, x):
                self.buf.add_(1)
                self.bar.buf.add_(2)
                bar = self.bar(x)
                return bar.sum() + self.buf.sum()

        exported = export(Foo(), (torch.ones(5, 5),))
        stateful_gm = exported.module()
        export_return_val = stateful_gm(torch.ones(5, 5))
        eager = Foo()
        eager_return_val = eager(torch.ones(5, 5))
        self.assertTrue(torch.allclose(eager_return_val, export_return_val))

        for name, buffer in stateful_gm.named_buffers():
            if name == "L__self___buf":
                self.assertTrue(torch.allclose(torch.ones(1), buffer))
            if name == "L__self___bar_buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(4, dtype=torch.float), buffer)
                )

        changed = stateful_gm.graph.eliminate_dead_code()
        self.assertFalse(changed)
        self.assertTrue(
            torch.allclose(stateful_gm(torch.ones(5, 5)), eager(torch.ones(5, 5)))
        )

        for name, buffer in stateful_gm.named_buffers():
            if name == "L__self___buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(2, dtype=torch.float), buffer)
                )
            if name == "L__self___bar_buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(7, dtype=torch.float), buffer)
                )

    def test_module_input(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y, m):
                return m(x, y) + x + y

        i = InputModule()
        f = Foo()
        ep = export(f, (torch.randn(3), torch.randn(3), i), strict=False)

        m = InputModule()
        inputs = (torch.randn(3), torch.randn(3), m)
        self.assertEqual(f(*inputs), ep.module()(*inputs))

    def test_module_input_subclasses_parameterization_nested(self):
        class Module(torch.nn.Module):
            def forward(self, x, m):
                return m(x) * 2

        mod = InputModuleWithNestedSubclass()
        f = Module()
        ref_x = torch.randn(2, 2)
        ref_out = f(ref_x, mod)

        ep = torch.export.export(f, (torch.randn(2, 2), mod), strict=False)
        self.assertEqual(ref_out, ep.module()(ref_x, mod))

    def test_unbacked_noncontig_lin(self):
        if "cpp_runtime_nonstrict" in self.id():
            self.skipTest("TODO Unexpected success in OSS but not in fbcode.")

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(32, 64)

            def forward(self, x):
                n = x.item()
                y = torch.empty(x).view(1, -1, 32)
                return self.lin(y)

        mod = Foo()
        x = torch.tensor([128])
        ep = export(mod, (x,))
        self.assertEqual(mod(x).shape, ep.module()(x).shape)
        x = torch.tensor([512])
        self.assertEqual(mod(x).shape, ep.module()(x).shape)

    def test_runtime_assert_for_prim(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        foo = Foo()
        tensor_inp = torch.ones(7, 5)
        dim0_x = torch.export.Dim("dim0_x", min=6)
        dynamic_shapes = {"x": {0: dim0_x}, "y": None}
        exported = torch.export.export(
            foo, (tensor_inp, 5), dynamic_shapes=dynamic_shapes
        )
        self.assertTrue(
            torch.allclose(
                exported.module()(torch.ones(8, 5), 5), foo(torch.ones(8, 5), 5)
            )
        )
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: y == 5"),
        ):
            # expected 5, but got 6
            _ = exported.module()(torch.ones(8, 5), 6)

        exported = torch.export.export(
            foo, (tensor_inp, 5.0), dynamic_shapes=dynamic_shapes
        )
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: y == 5.0"),
        ):
            # expected 5.0, but got 6.0
            _ = exported.module()(torch.ones(7, 5), 6.0)

    def test_runtime_assert_for_prm_str(self):
        class Foo(torch.nn.Module):
            def forward(self, a, b, mode):
                return torch.div(a, b, rounding_mode=mode)

        foo = Foo()
        inps = (torch.randn(4, 4), torch.randn(4), "trunc")
        exported = export(foo, inps)
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: mode == 'trunc'"),
        ):
            # expected 'trunc', but got 'floor'
            _ = exported.module()(torch.randn(4, 4), torch.randn(4), "floor")
        self.assertTrue(torch.allclose(exported.module()(*inps), foo(*inps)))

    def test_sym_or_sym_and(self):
        if "cpp_runtime_nonstrict" in self.id():
            self.skipTest("TODO Unexpected success in OSS but not in fbcode.")

        from torch.fx.experimental.symbolic_shapes import sym_and, sym_or

        class Foo(torch.nn.Module):
            def forward(self, xs):
                u0, u1, u2 = xs.tolist()
                torch._check(sym_or(u0 == 2, u0 == 4, u0 == 6))
                torch._check(sym_and(u1 >= 4, u1 <= 8, u2 == 5))
                return u0 + u1 + u2

        ep = export(Foo(), (torch.tensor([2, 6, 5]),), strict=False)
        ep.module()(torch.tensor([2, 6, 5]))
        ep.module()(torch.tensor([4, 7, 5]))
        ep.module()(torch.tensor([6, 5, 5]))
        with self.assertRaisesRegex(
            RuntimeError, r".* expression Eq\(u0, 2\) \| Eq\(u0, 4\) \| Eq\(u0, 6\) .*"
        ):
            ep.module()(torch.tensor([3, 6, 5]))
        with self.assertRaisesRegex(RuntimeError, r".* expression u[\d]+ <= 5 .*"):
            ep.module()(torch.tensor([6, 6, 6]))

    def test_redundant_assert_max_upper_bound(self):
        class M(torch.nn.Module):
            def forward(self, x):
                b = x.nonzero()
                torch._check(b.shape[0] >= 3)
                return b

        m = M()
        inp = (torch.tensor([1, 1, 1, 0, 1]),)
        dim = torch.export.Dim("dim")
        ep = export(m, inp, dynamic_shapes=((dim,),))
        FileCheck().check_count(
            "torch.ops.aten._assert_scalar.default", 1, exactly=True
        ).run(ep.graph_module.code)

    def test_to_module_with_mutated_buffer_multiple_update_sub_later(self):
        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))

            def forward(self, x):
                self.buf.add_(1)
                return x.sum() + self.buf.sum()

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.zeros(1))
                self.bar = Bar()

            def forward(self, x):
                self.buf.add_(1)
                bar = self.bar(x)
                self.bar.buf.add_(2)
                return bar.sum() + self.buf.sum()

        exported = export(Foo(), (torch.ones(5, 5),))
        stateful_gm = exported.module()
        export_return_val = stateful_gm(torch.ones(5, 5))
        eager = Foo()
        eager_return_val = eager(torch.ones(5, 5))
        self.assertTrue(torch.allclose(eager_return_val, export_return_val))

        for name, buffer in stateful_gm.named_buffers():
            if name == "L__self___buf":
                self.assertTrue(torch.allclose(torch.ones(1), buffer))
            if name == "L__self___bar_buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(4, dtype=torch.float), buffer)
                )

        changed = stateful_gm.graph.eliminate_dead_code()
        self.assertFalse(changed)
        self.assertTrue(
            torch.allclose(stateful_gm(torch.ones(5, 5)), eager(torch.ones(5, 5)))
        )

        for name, buffer in stateful_gm.named_buffers():
            if name == "L__self___buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(2, dtype=torch.float), buffer)
                )
            if name == "L__self___bar_buf":
                self.assertTrue(
                    torch.allclose(torch.tensor(7, dtype=torch.float), buffer)
                )

    def test_retracable_ep(self):
        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.ones(1))

            def forward(self, x):
                self.buf.add_(1)
                return x.sum() + self.buf.sum()

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buf = torch.nn.Buffer(torch.zeros(1))
                self.bar = Bar()

            def forward(self, x):
                self.buf.add_(1)
                bar = self.bar(x)
                self.bar.buf.add_(2)
                return bar.sum() + self.buf.sum()

        inp = torch.ones(5, 5)
        exported = torch.export.export(Foo(), (inp,))
        reexported = torch.export.export(exported.module(), (inp,))

        self.assertTrue(torch.allclose(Foo()(inp), reexported.module()(inp)))

        dim0_x = torch.export.Dim("dim0_x")
        exported = torch.export.export(Foo(), (inp,), dynamic_shapes=({0: dim0_x},))
        reexported = torch.export.export(exported.module(), (inp,))

        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: x.size()[0] == 5"),
        ):
            # expected 5, but got 7
            reexported.module()(torch.ones(7, 5))

        reexported = torch.export.export(
            exported.module(), (inp,), dynamic_shapes=({0: dim0_x},)
        )
        self.assertTrue(
            torch.allclose(
                Foo()(torch.ones(7, 5)), reexported.module()(torch.ones(7, 5))
            )
        )

        # can't retrace with invalid inputs with respect to the original ExportedProgram
        dim0_x_v2 = torch.export.Dim("dim0_x_v2", min=3)
        exported_v2 = torch.export.export(
            Foo(), (inp,), dynamic_shapes={"x": {0: dim0_x_v2}}
        )
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: x.size()[0] >= 3"),
        ):
            # expected >= 3, but got 2
            torch.export.export(exported_v2.module(), (torch.randn(2, 2),))

    def test_export_cond_symbool_pred(self):
        class A(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.ones(6, 4))

            def forward(self):
                return self.buffer.cos()

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = A()

            def forward(self, x):
                def true_fn(x):
                    return x.cos() + self.a().sum()

                def false_fn(x):
                    return x.sin()

                return cond(x.shape[0] > 4, true_fn, false_fn, [x])

        dim0 = torch.export.Dim("dim0", min=3)
        inp = torch.ones(6, 4)
        ep = export(Foo(), (inp,), dynamic_shapes={"x": {0: dim0}})
        schema = get_hop_schema(ep)
        self.assertExpectedInline(
            str(schema),
            """cond(SymBool pred, GraphModule true_fn, GraphModule false_fn, Tensor[2] operands) -> Tensor[1]""",
        )
        # serdes deserializes tuple as list
        if need_serdes_test(self._testMethodName):
            self.assertExpectedInline(
                ep.graph_module.code.strip(),
                """\
def forward(self, b_a_buffer, x):
    sym_size_int_1 = torch.ops.aten.sym_size.int(x, 0)
    gt = sym_size_int_1 > 4;  sym_size_int_1 = None
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, [x, b_a_buffer]);  gt = true_graph_0 = false_graph_0 = x = b_a_buffer = None
    getitem = cond[0];  cond = None
    return (getitem,)""",
            )

        else:
            self.assertExpectedInline(
                ep.graph_module.code.strip(),
                """\
def forward(self, b_a_buffer, x):
    sym_size_int_1 = torch.ops.aten.sym_size.int(x, 0)
    gt = sym_size_int_1 > 4;  sym_size_int_1 = None
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (x, b_a_buffer));  gt = true_graph_0 = false_graph_0 = x = b_a_buffer = None
    getitem = cond[0];  cond = None
    return (getitem,)""",
            )
        self.assertTrue(
            torch.allclose(ep.module()(torch.ones(6, 4)), Foo()(torch.ones(6, 4)))
        )

    def test_ccode_python_mod(self):
        import sympy

        from torch.utils._sympy.functions import PythonMod

        class Foo(torch.nn.Module):
            def forward(self, xs):
                u0, u1 = xs.tolist()
                return u0, u1

        ep = export(Foo(), (torch.tensor([2, 3]),), strict=False)
        u0_node, u1_node = list(ep.graph.nodes)[-1].args[0]
        u0 = u0_node.meta["val"]
        u1 = u1_node.meta["val"]
        self.assertExpectedInline(
            sympy.ccode(PythonMod(u0, 3)), """(u0 % 3) < 0 ? u0 % 3 + 3 : u0 % 3"""
        )
        self.assertExpectedInline(
            sympy.ccode(PythonMod(u0, u1)),
            """(u0 % u1) < 0 ? u0 % u1 + abs(u1) : u0 % u1""",
        )

    def test_aten_lift_fresh_copy(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.lift_fresh_copy(x)

        ep = export(M(), (torch.ones(6, 4),)).run_decompositions({})
        found = False

        op = "torch.ops.aten.clone.default"
        FileCheck().check_count(op, 1, exactly=True).run(ep.graph_module.code)

    def test_cond_buffers(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter(
                    "param", torch.nn.Parameter(torch.ones(2, 3), requires_grad=False)
                )
                self.buffer = torch.nn.Buffer(torch.ones(2, 3) + 1)

            def true_fn(self, x):
                return x + self.param

            def false_fn(self, x):
                return x + self.buffer

            def forward(self, x):
                return cond(x.shape[0] == 4, self.true_fn, self.false_fn, [x])

        inp = torch.ones(2, 3)
        ep = torch.export.export(M(), (inp,))
        inp = torch.randn(2, 3)
        epm = ep.module()
        self.assertTrue(torch.allclose(epm(inp), M()(inp)))

        for gm in epm.named_modules():
            if not isinstance(gm, torch.fx.GraphModule):
                continue
            self.assertEqual(
                len([node for node in gm.graph.nodes if node.op == "placeholder"]), 1
            )

    @requires_cuda_and_triton
    @testing.expectedFailureCppRuntime
    def test_export_associative_scan_symbol_dim(self):
        device = torch.device("cuda")
        combine_mode = "pointwise"

        dim1 = torch.export.Dim("dim0", min=5, max=15)
        xs = torch.ones(3, 10, 2, device=device)

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def combine_fn(self, x, y):
                return x + y

            def forward(self, x):
                return associative_scan(
                    self.combine_fn, x, 2, combine_mode=combine_mode
                )

        ep = export(Foo(), (xs,), dynamic_shapes={"x": {1: dim1}})
        module_out = Foo()(xs)
        self.assertTrue(torch.allclose(ep.module()(xs), module_out))

    @requires_cuda_and_triton
    @testing.expectedFailureCppRuntime
    def test_export_associative_scan_symbol_scandim(self):
        device = torch.device("cuda")
        combine_mode = "pointwise"

        dim1 = torch.export.Dim("dim0", min=5, max=15)
        xs = torch.ones(3, 10, 2, device=device)

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def combine_fn(self, x, y):
                return x + y

            def forward(self, x):
                return associative_scan(
                    self.combine_fn, x, 1, combine_mode=combine_mode
                )

        ep = export(Foo(), (xs,), dynamic_shapes={"x": {1: dim1}})
        module_out = Foo()(xs)
        self.assertTrue(torch.allclose(ep.module()(xs), module_out))

    @requires_cuda_and_triton
    @testing.expectedFailureStrictV2
    def test_export_associative_scan_lifted_buffers(self):
        if "cpp_runtime_nonstrict" in self.id():
            self.skipTest("TODO Unexpected success in OSS but not in fbcode.")

        device = torch.device("cuda")
        combine_mode = "pointwise"

        class A(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.buffer = torch.nn.Buffer(torch.ones(3, 2, device=device))

            def forward(self):
                return self.buffer.cos()

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = A()

            def combine_fn(self, x, y):
                return (x + y) * self.a()

            def forward(self, x):
                return associative_scan(
                    self.combine_fn, x, 1, combine_mode=combine_mode
                )

        inp = torch.ones(3, 10, 2, device=device)
        ep = export(M(), (inp,))
        epm = ep.module()

        self.assertTrue(torch.allclose(epm(inp), M()(inp)))

        for gm in epm.named_modules():
            if not isinstance(gm, torch.fx.GraphModule):
                continue
            self.assertEqual(
                len([node for node in gm.graph.nodes if node.op == "placeholder"]),
                1,
            )

    # scan is not supported in sigmoid yet
    @testing.expectedFailureCppRuntime
    def test_export_scan_pytree_output(self):
        def add(carry, accum):
            return carry + carry, (accum[0]["moo"] + 1, accum[0]["moo2"] + 1)

        class M(torch.nn.Module):
            def forward(self, init, accum):
                return scan(add, init, accum)

        inp = torch.randn(3)
        init, xs = torch.ones(3), ({"moo": torch.ones(3), "moo2": torch.ones(3)},)
        ep = export(M(), (init, xs))
        self.assertEqual(ep.module()(init, xs), M()(init, xs))

    def test_map_buffers(self):
        class M1(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter(
                    "param", torch.nn.Parameter(torch.tensor(5), requires_grad=False)
                )
                self.buffer = torch.nn.Buffer(torch.tensor(6) + 1)

        m1 = M1()

        def map_fn(x, y):
            z = x + y + m1.param + m1.buffer
            z.add_(4)
            return z

        class M(torch.nn.Module):
            def forward(self, xs, y):
                return map(map_fn, xs, y)

        example_inputs = (torch.ones(3, 2), torch.tensor(3))
        ep = torch.export.export(M(), example_inputs)
        example_inputs = (torch.randn(3, 2), torch.tensor(3))
        epm = ep.module()
        self.assertTrue(torch.allclose(epm(*example_inputs), M()(*example_inputs)))

        for gm in epm.named_modules():
            if not isinstance(gm, torch.fx.GraphModule):
                continue
            self.assertEqual(
                len([node for node in gm.graph.nodes if node.op == "placeholder"]), 2
            )

    @testing.expectedFailureStrictV2
    def test_no_check_is_size_error(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                a = x.item()
                return torch.randn(24).view(a, 4)

        f = Module()
        ep = export(f, (torch.tensor(6),))
        ep.module()(torch.tensor(6))
        with self.assertRaisesRegex(
            RuntimeError, r"Runtime assertion failed for .* u.* 6"
        ):
            ep.module()(torch.tensor(5))

    def test_suggest_torch_checks_with_non_negative_check(self):
        from unittest.mock import patch

        import sympy

        from torch.export.dynamic_shapes import defaultdict
        from torch.fx.experimental.symbolic_shapes import _suggest_torch_checks

        u = sympy.Symbol("u")
        cond = u >= 0
        mock_exception = MagicMock(
            spec=torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode
        )
        mock_exception.args = ["Test error message"]
        mock_exception.cond = cond

        mock_printer = MagicMock()
        mock_printer.doprint.side_effect = lambda expr: (
            str(cond) if expr == cond else "u < 0"  # Simulating the condition
        )
        with patch(
            "torch.fx.experimental.symbolic_shapes._PythonMsgPrinter",
            return_value=mock_printer,
        ):
            src_map = defaultdict(list)
            src_map["u"] = ["u"]
            _suggest_torch_checks(mock_exception, src_map)
            error_msg = mock_exception.args[0]
            self.assertIn("torch._check(u < 0)", error_msg)

    def test_suggest_torch_checks_with_regular_check(self):
        import sympy

        from torch.export.dynamic_shapes import defaultdict
        from torch.fx.experimental.symbolic_shapes import _suggest_torch_checks

        mock_exception = MagicMock(
            spec=torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode
        )
        mock_exception.args = ["Test error message"]

        mock_cond = MagicMock()
        mock_cond.free_symbols = {sympy.Symbol("u")}
        mock_exception.cond = mock_cond

        mock_printer = MagicMock()
        mock_printer.doprint.side_effect = lambda expr: (
            "u > 5" if expr == mock_cond else "u <= 5"
        )

        with patch(
            "torch.fx.experimental.symbolic_shapes._PythonMsgPrinter",
            return_value=mock_printer,
        ):
            src_map = defaultdict(list)
            src_map["u"] = ["u"]

            _suggest_torch_checks(mock_exception, src_map)

            error_msg = mock_exception.args[0]
            self.assertIn("torch._check(u > 5)", error_msg)
            self.assertIn("torch._check(u <= 5)", error_msg)

    def test_train_eval_on_exported_preautograd_module(self):
        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                if x.shape[0] > 4:
                    return x.cos()
                return x.sin()

        graph_module = _export(Foo(), (torch.ones(7, 5),), pre_dispatch=True).module()
        with self.assertRaisesRegex(
            NotImplementedError, r"Calling train\(\) is not supported yet."
        ):
            graph_module.train()

        with self.assertRaisesRegex(
            NotImplementedError, r"Calling eval\(\) is not supported yet."
        ):
            graph_module.eval()

    def test_lifted_constants(self) -> None:
        class Module(torch.nn.Module):
            def forward(self, x):
                return x + torch.tensor(3)

        f = Module()
        ep = export(f, (torch.tensor(1),))

        self.assertEqual(len(ep.graph_signature.input_specs), 2)
        self.assertEqual(len(ep.constants), 1)

        class Foo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor(3)

            def forward(self, x):
                list_tensor = [torch.tensor(3), torch.tensor(4)]
                return x + self.a + list_tensor[0] + list_tensor[1]

        ep = export(Foo(), (torch.tensor(1),))

        self.assertEqual(len(ep.graph_signature.input_specs), 4)
        self.assertEqual(len(ep.state_dict), 0)
        self.assertEqual(len(ep.constants), 3)

        inp = (torch.tensor(5),)
        self.assertTrue(torch.allclose(ep.module()(*inp), Foo()(*inp)))

        transform = ep.run_decompositions()
        self.assertEqual(len(ep.graph_signature.input_specs), 4)
        self.assertTrue(torch.allclose(ep.module()(*inp), transform.module()(*inp)))

        class Boo(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor(True)

            def forward(self, x):
                list_tensor = [torch.tensor(False), torch.tensor(True)]
                return x + self.a + list_tensor[0] + list_tensor[1]

        ep = export(Boo(), (torch.tensor(False),))

        self.assertEqual(len(ep.graph_signature.input_specs), 4)
        self.assertEqual(len(ep.state_dict), 0)
        self.assertEqual(len(ep.constants), 3)

        inp = (torch.tensor(True),)
        self.assertTrue(torch.allclose(ep.module()(*inp), Boo()(*inp)))

        transform = ep.run_decompositions()
        self.assertEqual(len(ep.graph_signature.input_specs), 4)
        self.assertTrue(torch.allclose(ep.module()(*inp), transform.module()(*inp)))

    @testing.expectedFailureStrictV2
    def test_tensor_attribute_zero_args(self):
        class Foo(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.x = torch.tensor(value)

            def forward(self):
                return self.x.clone()

        m = Foo([1, 2])
        ep = export(m, ())
        self.assertEqual(ep.graph_signature.lifted_tensor_constants, ["x"])

    @testing.expectedFailureStrictV2
    def test_preserve_shape_dynamism_for_unused_inputs(self):
        torch.export.register_dataclass(
            Inp3,
            serialized_type_name="test_preserve_shape_dynamism_for_unused_inputs.Inp3",
        )

        class Module(torch.nn.Module):
            def forward(self, x: Inp3):
                return x.f + 1

        mod = Module()
        example_inputs = (Inp3(f=torch.ones(10, 4), p=torch.zeros(10, 4)),)
        ep_static = export(mod, example_inputs)
        for node in ep_static.graph.nodes:
            if node.op == "placeholder":
                for s in node.meta["val"].shape:
                    self.assertIsInstance(s, int)

        dim0_x_f, dim0_x_p = torch.export.dims("dim0_x_f", "dim0_x_p")
        dynamic_shapes = {"x": [{0: dim0_x_f}, {0: dim0_x_p}]}
        ep_dynamic = export(mod, example_inputs, dynamic_shapes=dynamic_shapes)
        for node in ep_dynamic.graph.nodes:
            if node.op == "placeholder":
                for i, s in enumerate(node.meta["val"].shape):
                    if i == 0:
                        self.assertIsInstance(s, torch.SymInt)
                    else:
                        self.assertIsInstance(s, int)

    def test_multiple_definitions_same_name_dim(self):
        class Foo(torch.nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)

        A = torch.export.Dim("C", min=3)
        B = torch.export.Dim("C", max=12)
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Found different definitions Dim\\(.*min=3\\) and Dim\\(.*max=12\\) "
            "for the same symbolic dimension",
        ):
            torch.export.export(
                Foo(),
                (torch.randn(10, 10), torch.randn(10, 10)),
                dynamic_shapes={"x": (A, B), "y": (B, A)},
            )

    def test_multinomial_dynamic(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return torch.multinomial(x, y.shape[0])

        model = Model()
        DYNAMIC = torch.export.Dim.DYNAMIC

        def exported_module(inputs):
            dynamic_shapes = tuple(tuple(DYNAMIC for _ in inp.shape) for inp in inputs)
            ep = export(model, inputs, dynamic_shapes=dynamic_shapes)
            return ep.module()

        def check(inputs, epm):
            eager_result = model(*inputs)
            ep_result = epm(*inputs)
            self.assertEqual(ep_result.shape, eager_result.shape)

        inputs = (
            torch.tensor([0, 10, 3, 0], dtype=torch.float32),
            torch.ones(2, dtype=torch.int64),
        )
        epm = exported_module(inputs)
        # output shape is (2,), where n_sample 2 <= dist_size 4
        check(inputs, epm)

        inputs = (
            torch.tensor([0, 10, 3, 7, 6, 0], dtype=torch.float32),
            torch.ones(3, dtype=torch.int64),
        )
        # output shape is (3,), with n_sample 3 <= dist_size 6
        check(inputs, epm)

        inputs = (
            torch.tensor([0, 10, 3, 0], dtype=torch.float32),
            torch.ones(5, dtype=torch.int64),
        )
        with self.assertRaisesRegex(RuntimeError, "cannot sample"):
            # n_sample 5 > dist_size 4
            epm(*inputs)

        inputs = (
            torch.tensor([[4, 5], [6, 7], [8, 9]], dtype=torch.float32),
            torch.ones(2, dtype=torch.int64),
        )
        epm = exported_module(inputs)
        # output shape is (3, 2), with n_row 3 and n_sample 2 <= dist_size 2
        check(inputs, epm)

        inputs = (
            torch.tensor([[4, 5], [6, 7], [8, 9]], dtype=torch.float32),
            torch.ones(3, dtype=torch.int64),
        )
        epm = exported_module(inputs)
        with self.assertRaisesRegex(RuntimeError, "cannot sample"):
            # n_sample 3 > dist_size 2
            epm(*inputs)

    def test_export_with_wrong_inputs(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + x

        exported_program = export(MyModule(), (torch.rand(2, 3),), {})
        with self.assertRaisesRegex(ValueError, "Trying to flatten user inputs"):
            exported_program.module()(torch.rand(2, 3), torch.rand(2, 3))

    def test_export_decomps_simple(self):
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = torch.nn.Linear(10, 1)

            def forward(self, x):
                return self.lin(x)

        inp = (torch.randn(5, 10),)
        m = M()
        ep = export(m, inp)
        state_dict = ep.state_dict

        self.assertTrue(torch.allclose(ep.module()(*inp), m(*inp)))

        core_aten_ep = ep.run_decompositions()
        FileCheck().check_count("torch.ops.aten.permute.default", 1, exactly=True).run(
            core_aten_ep.graph_module.code
        )
        FileCheck().check_count("torch.ops.aten.t.default", 0, exactly=True).run(
            core_aten_ep.graph_module.code
        )
        self.assertTrue(torch.allclose(core_aten_ep.module()(*inp), m(*inp)))
        self.assertEqual(id(state_dict), id(ep.state_dict))

    @unittest.skipIf(IS_FBCODE, "We can't customize decomp in fbcode")
    def test_export_decomp_torture_case_1(self):


... [truncated: 7838 more lines]
```

*Note: Content truncated due to size*

## High-Level Overview


This Python file contains 376 class(es) and 918 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Inp1`, `Inp2`, `Inp3`, `TestDynamismExpression`, `Module`, `InvalidInputConflictWithInputConstraints`, `Slice`, `Foo`, `MySlice`, `MyModel`, `ConflictingConstraints`, `Module`, `Module`, `InputModule`, `InputModuleWithNestedSubclass`, `TestExport`, `Module`, `Foo`, `VmapToAssert`, `Foo`

**Functions defined**: `returns_tensor_symint_impl`, `foo_impl`, `foo_abstract`, `foo_mutated`, `foo_functional`, `foo_unbacked`, `is_non_strict_test`, `is_strict_test`, `is_strict_v2_test`, `is_inline_and_install_strict_test`, `is_retracebility_test`, `is_serdes_test`, `need_serdes_test`, `is_training_ir_test`, `is_training_ir_strict_test`, `is_cpp_runtime_test`, `get_hop_schema`, `test_export_inline_constraints`, `forward`, `test_export_constraints_error_not_in_range`

**Key imports**: contextlib, copy, dataclasses, enum, functools, logging, math, operator, os, re


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/export`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `copy`
- `dataclasses`
- `enum`
- `functools`
- `logging`
- `math`
- `operator`
- `os`
- `re`
- `sys`
- `traceback`
- `unittest`
- `warnings`
- `weakref`
- `typing`: Dict, List, Union
- `unittest.mock`: MagicMock, patch
- `torch`
- `torch._dynamo as torchdynamo`
- `torch.fx.traceback as fx_traceback`
- `torch.nn.functional as F`
- `torch.utils._pytree as pytree`
- `functorch.experimental.control_flow`: cond, map
- `torch._decomp`: decomposition_table, get_decompositions
- `torch._dynamo._trace_wrapped_higher_order_op`: mod_index
- `torch._dynamo.test_case`: TestCase


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

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/export/test_export.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/export`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_schema.py_docs.md`](./test_schema.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_cpp_serdes.py_docs.md`](./test_cpp_serdes.py_docs.md)
- [`test_export_opinfo.py_docs.md`](./test_export_opinfo.py_docs.md)
- [`test_lift_unlift.py_docs.md`](./test_lift_unlift.py_docs.md)
- [`test_retraceability.py_docs.md`](./test_retraceability.py_docs.md)
- [`test_converter.py_docs.md`](./test_converter.py_docs.md)
- [`test_nativert.py_docs.md`](./test_nativert.py_docs.md)


## Cross-References

- **File Documentation**: `test_export.py_docs.md`
- **Keyword Index**: `test_export.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
