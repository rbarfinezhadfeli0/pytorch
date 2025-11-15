# Documentation: test_export.py

## File Metadata
- **Path**: `test/export/test_export.py`
- **Size**: 674732 bytes
- **Lines**: 17838
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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
        false_grap

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 720 class(es): class, class, class, TestDynamismExpression, Module, InvalidInputConflictWithInputConstraints, Slice, Foo, MySlice, MyModel, ConflictingConstraints, Module, Module, InputModule, InputModuleWithNestedSubclass, TestExport, Module, Foo, VmapToAssert, Foo

### Functions
This file defines 1729 function(s): returns_tensor_symint_impl, foo_impl, foo_abstract, foo_mutated, foo_functional, foo_unbacked, is_non_strict_test, is_strict_test, is_strict_v2_test, is_inline_and_install_strict_test, is_retracebility_test, is_serdes_test, need_serdes_test, is_training_ir_test, is_training_ir_strict_test, is_cpp_runtime_test, get_hop_schema, test_export_inline_constraints, forward, test_export_constraints_error_not_in_range, forward, test_export_slice_maxsize, forward, test_no_grad_param_inplace, __init__, forward, test_export_slice_unbacked_dim1, forward, test_reshape_view_backed_size_oblivious, forward


## Key Components

The file contains 52992 words across 17838 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 674732 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
