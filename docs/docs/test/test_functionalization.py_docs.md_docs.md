# Documentation: `docs/test/test_functionalization.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_functionalization.py_docs.md`
- **Size**: 53,463 bytes (52.21 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_functionalization.py`

## File Metadata

- **Path**: `test/test_functionalization.py`
- **Size**: 96,959 bytes (94.69 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: codegen"]
# ruff: noqa: F841

import unittest
from contextlib import nullcontext

import torch
from torch._dispatch.python import (
    enable_crossref_functionalize,
    enable_python_dispatcher,
)
from torch._subclasses.functional_tensor import (
    dispatch_functionalize,
    FunctionalTensor,
    FunctionalTensorMode,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.reinplace import reinplace
from torch.multiprocessing.reductions import StorageWeakRef
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xfail_inherited_tests,
)
from torch.testing._internal.logging_tensor import capture_logs, LoggingTensor
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map_only


def are_aliased(x, y):
    x_storage = StorageWeakRef(x.storage())
    y_storage = StorageWeakRef(y.storage())
    return x_storage == y_storage


# We can unify testing and use functionalize() here instead
# if/when functorch moves into core.
# This is basically a crappy version of `functionalize()`.
def _functionalize(
    f, *, reapply_views: bool, crossref: bool, skip_input_mutations: bool = False
):
    def to_fun(t: torch.Tensor):
        func_t = torch._to_functional_tensor(t)
        func_t.requires_grad = t.requires_grad
        return func_t

    def wrapped(*inputs):
        ctx = nullcontext()
        if crossref:
            ctx = enable_crossref_functionalize()
        with ctx:
            inputs_functional = tree_map_only(torch.Tensor, to_fun, inputs)
            torch._enable_functionalization(reapply_views=reapply_views)
            try:
                out = f(*inputs_functional)
            finally:
                torch._disable_functionalization()
            flat_inputs = pytree.tree_leaves(inputs)
            flat_inputs_functional = pytree.tree_leaves(inputs_functional)

            for inpt, input_functional in zip(flat_inputs, flat_inputs_functional):
                torch._sync(input_functional)
                inpt_new = torch._from_functional_tensor(input_functional)
                if inpt_new is not inpt and not skip_input_mutations:
                    # Existing deficiency in functionalize():
                    # we don't correctly mutate input metadata (yet?)
                    if inpt_new.shape == inpt.shape:
                        inpt.copy_(inpt_new)
            tree_map_only(torch.Tensor, torch._sync, out)
            out_unwrapped = tree_map_only(
                torch.Tensor, torch._from_functional_tensor, out
            )
            return out_unwrapped

    return wrapped


@unittest.skipIf(
    TEST_WITH_TORCHDYNAMO, "https://github.com/pytorch/pytorch/issues/81457"
)
class TestFunctionalization(TestCase):
    crossref = False

    def get_logs(self, func, *inpts, reapply_views=False, run_reinplace=False):
        inpts_clone = tree_map_only(torch.Tensor, torch.clone, inpts)
        traced_f = make_fx(
            _functionalize(func, reapply_views=reapply_views, crossref=self.crossref)
        )(*inpts)
        if run_reinplace:
            traced_f = reinplace(traced_f, *inpts_clone)
        return traced_f.code

    def assert_functionalization(
        self, func, *inpts, reapply_views=False, mutated_input_metadata=False
    ):
        clones1 = tree_map_only(torch.Tensor, torch.clone, inpts)
        clones2 = tree_map_only(torch.Tensor, torch.clone, inpts)
        clones3 = tree_map_only(torch.Tensor, torch.clone, inpts)

        # Compare outputs (and mutated inputs), with and without functionalization.
        out_ref = func(*inpts)
        out_functional = _functionalize(
            func, reapply_views=reapply_views, crossref=self.crossref
        )(*clones1)

        # The reinplacing pass is only valid to run with reapply_views=True.
        functional_func = make_fx(
            _functionalize(func, reapply_views=True, crossref=self.crossref)
        )(*clones2)
        reinplace_func = reinplace(functional_func, *clones2)

        # NOTE: for now, need to pass in fresh inputs here, because make_fx
        # will directly mutate the inputs that you trace with.
        # Once this is fixed we can clean this up.
        out_reinplace = reinplace_func(*clones3)

        # functionalize() deficiency: input metadata mutations aren't propagated properly,
        # so we just need to skip checks here for the tests that exercise that.
        if not mutated_input_metadata:
            flat_inpts = pytree.tree_leaves(inpts)
            flat_clones1 = pytree.tree_leaves(clones1)
            flat_clones3 = pytree.tree_leaves(clones3)
            for inpt, input_clone, input_clone3 in zip(
                flat_inpts, flat_clones1, flat_clones3
            ):
                self.assertEqual(
                    inpt, input_clone
                )  # input mutations should still occur
                self.assertEqual(inpt, input_clone3)

        # Handle tests with multi-tensor outputs
        if isinstance(out_ref, tuple):
            out_refs, out_functionals, out_reinplaces = (
                list(out_ref),
                list(out_functional),
                list(out_reinplace),
            )
        else:
            out_refs, out_functionals, out_reinplaces = (
                [out_ref],
                [out_functional],
                [out_reinplace],
            )

        for out_ref_, out_functional_, out_reinplace_ in zip(
            out_refs, out_functionals, out_reinplaces
        ):
            self.assertEqual(out_ref_, out_functional_)
            self.assertEqual(out_ref_, out_reinplace_)

    def test_save_for_backwards_segfault(self):
        inp = torch._to_functional_tensor(
            LoggingTensor(torch.randn(2, 2))
        ).requires_grad_(True)
        inp.exp()

    def test_multiple_views_of_same_base(self):
        def f(x):
            y = x.view(-1)
            z = x.view(-1)
            x.add_(1)
            # y should have been updated.
            y2 = y + 1
            # z should have been updated too.
            z2 = z + 1
            return z2

        self.assert_functionalization(f, torch.ones(4))

    def test_freeze(self):
        def f(x):
            y = x.clone()
            z = y[0]
            torch._freeze_functional_tensor(y)
            x.add_(1)
            self.assertRaises(RuntimeError, lambda: y.add_(1))
            self.assertRaises(RuntimeError, lambda: z.add_(1))
            return z

        _functionalize(f, reapply_views=True, crossref=self.crossref)(torch.ones(3, 3))

    def test_copy_stride_mismatch(self):
        def f(x):
            y = torch.empty_strided((2, 2), (5, 1))
            y.copy_(x)
            return y

        r = _functionalize(f, reapply_views=True, crossref=self.crossref)(
            torch.ones(2, 2)
        )
        self.assertEqual(r.stride(), (5, 1))

    def test_set_(self):
        def f(x):
            y = torch.ones(2)
            y.set_(x.storage())
            return y

        # We should probably get the crossref test to work,
        # but fixing it for Storage() objects is annoying.
        r = _functionalize(f, reapply_views=True, crossref=False)(torch.ones(2))
        self.assertEqual(str(r.device), "cpu")

    def test_advanced_indexing(self):
        def f():
            x = torch.zeros(3, 3)
            idx = torch.tensor([0])
            val = torch.ones(3, 1)
            x[:, idx] = val
            return x

        self.assert_functionalization(f)

    def test_view_clone_view_inplace(self):
        def f(input):
            shape = [1, 1024, 128, 128]
            input_reshaped = input.view(shape)
            out = input_reshaped.clone()
            r = out.view(input.shape)
            r.relu_()
            return r

        def g(x):
            loss = f(x).sum()
            import torch.fx.traceback as fx_traceback
            from torch._functorch.aot_autograd import (
                setup_stacktrace_preservation_hooks,
            )

            setup_stacktrace_preservation_hooks([loss.grad_fn])
            with fx_traceback.preserve_node_meta():
                loss.backward()
            return x.grad

        with torch.autograd.detect_anomaly(check_nan=False):
            logs = self.get_logs(g, torch.ones(16, 64, 128, 128, requires_grad=True))
        self.assertExpectedInline(
            logs,
            """\



def forward(self, arg0_1):
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [1, 1024, 128, 128]);  arg0_1 = None
    clone = torch.ops.aten.clone.default(view_copy);  view_copy = None
    view_copy_1 = torch.ops.aten.view_copy.default(clone, [16, 64, 128, 128])
    relu = torch.ops.aten.relu.default(view_copy_1);  view_copy_1 = None
    view_copy_2 = torch.ops.aten.view_copy.default(relu, [1, 1024, 128, 128]);  relu = None
    view_copy_3 = torch.ops.aten.view_copy.default(view_copy_2, [16, 64, 128, 128]);  view_copy_2 = None
    view_copy_4 = torch.ops.aten.view_copy.default(clone, [16, 64, 128, 128]);  clone = view_copy_4 = None
    sum_1 = torch.ops.aten.sum.default(view_copy_3)
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand_copy = torch.ops.aten.expand_copy.default(ones_like, [16, 64, 128, 128]);  ones_like = None
    view_copy_5 = torch.ops.aten.view_copy.default(expand_copy, [1, 1024, 128, 128]);  expand_copy = None
    new_empty_strided = torch.ops.aten.new_empty_strided.default(view_copy_5, [1, 1024, 128, 128], [16777216, 16384, 128, 1])
    copy = torch.ops.aten.copy.default(new_empty_strided, view_copy_5);  new_empty_strided = view_copy_5 = None
    view_copy_6 = torch.ops.aten.view_copy.default(copy, [16, 64, 128, 128]);  view_copy_6 = None
    view_copy_7 = torch.ops.aten.view_copy.default(copy, [16, 64, 128, 128])
    clone_1 = torch.ops.aten.clone.default(view_copy_7, memory_format = torch.contiguous_format)
    threshold_backward = torch.ops.aten.threshold_backward.default(clone_1, view_copy_3, 0);  clone_1 = view_copy_3 = None
    copy_1 = torch.ops.aten.copy.default(view_copy_7, threshold_backward);  view_copy_7 = threshold_backward = None
    view_copy_8 = torch.ops.aten.view_copy.default(copy_1, [1, 1024, 128, 128]);  copy_1 = None
    view_copy_9 = torch.ops.aten.view_copy.default(view_copy_8, [16, 64, 128, 128]);  view_copy_9 = None
    view_copy_10 = torch.ops.aten.view_copy.default(copy, [16, 64, 128, 128]);  copy = None
    detach_copy = torch.ops.aten.detach_copy.default(view_copy_10);  view_copy_10 = detach_copy = None
    view_copy_11 = torch.ops.aten.view_copy.default(view_copy_8, [16, 64, 128, 128]);  view_copy_8 = None
    detach_copy_1 = torch.ops.aten.detach_copy.default(view_copy_11);  view_copy_11 = None
    return detach_copy_1
    """,
        )  # noqa: B950

    def test_simple(self):
        def f(x):
            # simple test: 1 view op, 1 inplace op
            tmp = torch.ones(4, 2)
            y = x.view(4, 2)
            y.add_(tmp)
            z = x * x
            return y

        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(
            logs,
            """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [4, 2])
    add = torch.ops.aten.add.Tensor(view_copy, ones);  view_copy = ones = None
    view_copy_1 = torch.ops.aten.view_copy.default(add, [4, 2]);  add = None
    view_copy_2 = torch.ops.aten.view_copy.default(view_copy_1, [4, 2])
    mul = torch.ops.aten.mul.Tensor(view_copy_1, view_copy_1);  mul = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, view_copy_1);  arg0_1 = view_copy_1 = copy_ = None
    return view_copy_2
    """,
        )

        reinplaced_logs = self.get_logs(
            f, torch.ones(4, 2), reapply_views=True, run_reinplace=True
        )
        self.assertExpectedInline(
            reinplaced_logs,
            """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view = torch.ops.aten.view.default(arg0_1, [4, 2])
    add = torch.ops.aten.add.Tensor(view, ones);  view = ones = None
    view_1 = torch.ops.aten.view.default(add, [4, 2]);  add = None
    view_2 = torch.ops.aten.view.default(view_1, [4, 2])
    mul = torch.ops.aten.mul.Tensor(view_1, view_1);  mul = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, view_1);  arg0_1 = view_1 = copy_ = None
    return view_2
    """,
        )

    def test_simple_out(self):
        def f(x):
            tmp = torch.ones(4, 2)
            y = x.view(4, 2)
            # the out= tensor will get resized, since it has size=0 to start.
            z = torch.empty(())
            torch.add(y, tmp, out=z)
            w = z * z
            return w

        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(
            logs,
            """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [4, 2]);  arg0_1 = None
    empty = torch.ops.aten.empty.memory_format([], device = device(type='cpu'), pin_memory = False);  empty = None
    add = torch.ops.aten.add.Tensor(view_copy, ones);  view_copy = ones = None
    mul = torch.ops.aten.mul.Tensor(add, add);  add = None
    return mul
    """,
        )

        reinplaced_logs = self.get_logs(
            f, torch.ones(4, 2), reapply_views=True, run_reinplace=True
        )
        self.assertExpectedInline(
            reinplaced_logs,
            """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view = torch.ops.aten.view.default(arg0_1, [4, 2]);  arg0_1 = None
    empty = torch.ops.aten.empty.memory_format([], device = device(type='cpu'), pin_memory = False);  empty = None
    add = torch.ops.aten.add.Tensor(view, ones);  view = ones = None
    mul = torch.ops.aten.mul.Tensor(add, add);  add = None
    return mul
    """,
        )

    def test_multi_out(self):
        def f(x):
            # aminmax.out returns a tuple of tensors.
            # functionalization should properly handle the tuple.
            out_min = torch.empty(4)
            out_max = torch.empty(4)
            torch.aminmax(x, dim=0, out=(out_max, out_min))
            return out_max

        self.assert_functionalization(f, torch.arange(8, dtype=torch.float32))
        logs = self.get_logs(f, torch.arange(8, dtype=torch.float32))
        self.assertExpectedInline(
            logs,
            """\



def forward(self, arg0_1):
    empty = torch.ops.aten.empty.memory_format([4], device = device(type='cpu'), pin_memory = False);  empty = None
    empty_1 = torch.ops.aten.empty.memory_format([4], device = device(type='cpu'), pin_memory = False);  empty_1 = None
    aminmax = torch.ops.aten.aminmax.default(arg0_1, dim = 0);  arg0_1 = None
    getitem = aminmax[0]
    getitem_1 = aminmax[1];  aminmax = getitem_1 = None
    return getitem
    """,
        )

        reinplaced_logs = self.get_logs(
            f,
            torch.arange(8, dtype=torch.float32),
            reapply_views=True,
            run_reinplace=True,
        )
        self.assertExpectedInline(
            reinplaced_logs,
            """\



def forward(self, arg0_1):
    empty = torch.ops.aten.empty.memory_format([4], device = device(type='cpu'), pin_memory = False);  empty = None
    empty_1 = torch.ops.aten.empty.memory_format([4], device = device(type='cpu'), pin_memory = False);  empty_1 = None
    aminmax = torch.ops.aten.aminmax.default(arg0_1, dim = 0);  arg0_1 = None
    getitem = aminmax[0]
    getitem_1 = aminmax[1];  aminmax = getitem_1 = None
    return getitem
    """,
        )

    def test_tensor_ctr(self):
        def f(x):
            y = torch.tensor((1, 2, 3))
            z = y.view(-1)
            z.add_(1)
            return y

        inpt = torch.arange(3, dtype=torch.float32)
        self.assert_functionalization(f, inpt)

        logs = self.get_logs(f, inpt)
        self.assertExpectedInline(
            logs,
            """\



def forward(self, arg0_1):
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    view_copy = torch.ops.aten.view_copy.default(lift_fresh_copy, [-1]);  lift_fresh_copy = None
    add = torch.ops.aten.add.Tensor(view_copy, 1);  view_copy = None
    view_copy_1 = torch.ops.aten.view_copy.default(add, [3]);  add = None
    view_copy_2 = torch.ops.aten.view_copy.default(view_copy_1, [-1]);  view_copy_2 = None
    return view_copy_1
    """,
        )

        reinplaced_logs = self.get_logs(f, inpt, reapply_views=True, run_reinplace=True)
        self.assertExpectedInline(
            reinplaced_logs,
            """\



def forward(self, arg0_1):
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    view = torch.ops.aten.view.default(lift_fresh_copy, [-1]);  lift_fresh_copy = None
    add = torch.ops.aten.add_.Tensor(view, 1);  add = None
    view_1 = torch.ops.aten.view.default(view, [3]);  view = None
    view_2 = torch.ops.aten.view.default(view_1, [-1]);  view_2 = None
    return view_1
    """,
        )

    def test_advanced_indexing_correct_strides(self):
        def f(a):
            # This test requires that *_scatter ops are able to return
            # non-contiguous tensors.
            b = a.clone()[:, 1]
            c = torch.ones_like(b, dtype=torch.bool)
            d = b.masked_fill_(c, 0)
            return d

        self.assert_functionalization(f, torch.ones(2, 2), reapply_views=True)

    def test_tensor_list_mixed_functional_nonfunctional(self):
        nonfunctional_tensor = torch.ones(2, dtype=torch.long)

        def f(x):
            # simple test: 1 view op, 1 inplace op
            functional_tensor = torch.ones(2, dtype=torch.long)
            out = x[functional_tensor, nonfunctional_tensor]
            return out

        out = f(torch.ones(2, 2))
        out_functional = _functionalize(f, reapply_views=True, crossref=self.crossref)(
            torch.ones(2, 2)
        )
        self.assertEqual(out, out_functional)

    def test_inplace_on_non_view(self):
        def f(x):
            # test for the case where we functionalize an inplace op on the other tensor - not a view.
            # This is worth checking because the tensor will have an empty ViewMeta stack, which needs to be special cased.
            tmp = torch.ones(4, 2)
            y = x.view(4, 2)
            x.add_(tmp)
            return y

        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(
            logs,
            """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [4, 2]);  view_copy = None
    add = torch.ops.aten.add.Tensor(arg0_1, ones);  ones = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = copy_ = None
    view_copy_1 = torch.ops.aten.view_copy.default(add, [4, 2]);  add = None
    return view_copy_1
    """,
        )

        reinplaced_logs = self.get_logs(
            f, torch.ones(4, 2), reapply_views=True, run_reinplace=True
        )
        self.assertExpectedInline(
            reinplaced_logs,
            """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False)
    view = torch.ops.aten.view.default(arg0_1, [4, 2]);  view = None
    add = torch.ops.aten.add.Tensor(arg0_1, ones);  ones = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = copy_ = None
    view_1 = torch.ops.aten.view.default(add, [4, 2]);  add = None
    return view_1
    """,
        )

    # Some ops that are mutable are neither inplace nor out= ops.
    # They also need special handling.
    def test_mutable_op_not_inplace_or_other(self):
        def f(x):
            return torch._fused_moving_avg_obs_fq_helper(
                x, x, x, x, x, x, x, 1.0, 0, 1, 0
            )

        logs = self.get_logs(f, torch.ones(1))
        self.assertExpectedInline(
            logs,
            """\



def forward(self, arg0_1):
    _fused_moving_avg_obs_fq_helper_functional = torch.ops.aten._fused_moving_avg_obs_fq_helper_functional.default(arg0_1, arg0_1, arg0_1, arg0_1, arg0_1, arg0_1, arg0_1, 1.0, 0, 1, 0)
    getitem = _fused_moving_avg_obs_fq_helper_functional[0]
    getitem_1 = _fused_moving_avg_obs_fq_helper_functional[1]
    getitem_2 = _fused_moving_avg_obs_fq_helper_functional[2];  getitem_2 = None
    getitem_3 = _fused_moving_avg_obs_fq_helper_functional[3];  getitem_3 = None
    getitem_4 = _fused_moving_avg_obs_fq_helper_functional[4];  getitem_4 = None
    getitem_5 = _fused_moving_avg_obs_fq_helper_functional[5];  _fused_moving_avg_obs_fq_helper_functional = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, getitem_5);  arg0_1 = getitem_5 = copy_ = None
    return (getitem, getitem_1)
    """,  # noqa: B950
        )

    def test_as_strided(self):
        def f(x):
            y = x.as_strided((2,), (2,), 1)
            y.add_(1)
            return x

        self.assert_functionalization(f, torch.ones(9))
        logs = self.get_logs(f, torch.ones(9))
        self.assertExpectedInline(
            logs,
            """\



def forward(self, arg0_1):
    as_strided_copy = torch.ops.aten.as_strided_copy.default(arg0_1, [2], [2], 1)
    add = torch.ops.aten.add.Tensor(as_strided_copy, 1);  as_strided_copy = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(arg0_1, add, [2], [2], 1);  add = None
    as_strided_copy_1 = torch.ops.aten.as_strided_copy.default(as_strided_scatter, [2], [2], 1);  as_strided_copy_1 = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, as_strided_scatter);  arg0_1 = copy_ = None
    return as_strided_scatter
    """,
        )

        # NB: even with reapply_views=True, we expect to see scatter op
        reinplaced_logs = self.get_logs(
            f, torch.ones(2, 2), reapply_views=True, run_reinplace=False
        )
        self.assertExpectedInline(
            reinplaced_logs,
            """\



def forward(self, arg0_1):
    as_strided = torch.ops.aten.as_strided.default(arg0_1, [2], [2], 1)
    add = torch.ops.aten.add.Tensor(as_strided, 1);  as_strided = None
    as_strided_scatter = torch.ops.aten.as_strided_scatter.default(arg0_1, add, [2], [2], 1);  add = None
    as_strided_1 = torch.ops.aten.as_strided.default(as_strided_scatter, [2], [2], 1);  as_strided_1 = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, as_strided_scatter);  arg0_1 = copy_ = None
    return as_strided_scatter
    """,
        )

    def test_tensor_list_composite(self):
        def f(x):
            # Test an op with TensorList input
            y = torch.block_diag(x, x)
            return y

        self.assert_functionalization(f, torch.ones(2, 2))
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline(
            logs,
            """\



def forward(self, arg0_1):
    block_diag = torch.ops.aten.block_diag.default([arg0_1, arg0_1]);  arg0_1 = None
    return block_diag
    """,
        )

    def test_cat(self):
        def f(x):
            out = torch.empty(0)
            torch.cat((x,), out=out)
            return out

        self.assert_functionalization(f, torch.ones(2, 2))
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline(
            logs,
            """\



def forward(self, arg0_1):
    empty = torch.ops.aten.empty.memory_format([0], device = device(type='cpu'), pin_memory = False);  empty = None
    cat = torch.ops.aten.cat.default([arg0_1]);  arg0_1 = None
    return cat
    """,
        )

        reinplaced_logs = self.get_logs(
            f, torch.ones(2, 2), reapply_views=True, run_reinplace=True
        )
        self.assertExpectedInline(
            reinplaced_logs,
            """\



def forward(self, arg0_1):
    empty = torch.ops.aten.empty.memory_format([0], device = device(type='cpu'), pin_memory = False);  empty = None
    cat = torch.ops.aten.cat.default([arg0_1]);  arg0_1 = None
    return cat
    """,
        )

    def test_diagonal(self):
        def f(x):
            # test: view ops that take a subset of the original tensor (select/diagonal)
            tmp = torch.ones(2)
            y = x.clone().diagonal()
            y.add_(tmp)
            z = x * x
            return z

        self.assert_functionalization(f, torch.ones(2, 2))
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline(
            logs,
            """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)
    clone = torch.ops.aten.clone.default(arg0_1)
    diagonal_copy = torch.ops.aten.diagonal_copy.default(clone)
    add = torch.ops.aten.add.Tensor(diagonal_copy, ones);  diagonal_copy = ones = None
    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(clone, add);  clone = add = None
    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(diagonal_scatter);  diagonal_scatter = diagonal_copy_1 = None
    mul = torch.ops.aten.mul.Tensor(arg0_1, arg0_1);  arg0_1 = None
    return mul
    """,
        )

        reinplaced_logs = self.get_logs(
            f, torch.ones(2, 2), reapply_views=True, run_reinplace=True
        )
        self.assertExpectedInline(
            reinplaced_logs,
            """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)
    clone = torch.ops.aten.clone.default(arg0_1)
    diagonal = torch.ops.aten.diagonal.default(clone)
    add = torch.ops.aten.add_.Tensor(diagonal, ones);  diagonal = ones = add = None
    diagonal_1 = torch.ops.aten.diagonal.default(clone);  clone = diagonal_1 = None
    mul = torch.ops.aten.mul.Tensor(arg0_1, arg0_1);  arg0_1 = None
    return mul
    """,
        )

    def test_diagonal_mutated_input(self):
        def f(x):
            # simple test: there are pending updates afterwards, which the test syncs manually
            tmp = torch.ones(2)
            y = x.diagonal()
            y.add_(tmp)
            return x

        x = torch.ones(2, 2)
        self.assert_functionalization(f, x)
        logs = self.get_logs(f, torch.ones(2, 2))
        self.assertExpectedInline(
            logs,
            """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)
    diagonal_copy = torch.ops.aten.diagonal_copy.default(arg0_1)
    add = torch.ops.aten.add.Tensor(diagonal_copy, ones);  diagonal_copy = ones = None
    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(arg0_1, add);  add = None
    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(diagonal_scatter);  diagonal_copy_1 = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, diagonal_scatter);  arg0_1 = copy_ = None
    return diagonal_scatter
    """,
        )

        # NB: even with reapply_views=True, we expect to see scatter op
        reinplaced_logs = self.get_logs(
            f, torch.ones(2, 2), reapply_views=True, run_reinplace=False
        )
        self.assertExpectedInline(
            reinplaced_logs,
            """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)
    diagonal = torch.ops.aten.diagonal.default(arg0_1)
    add = torch.ops.aten.add.Tensor(diagonal, ones);  diagonal = ones = None
    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(arg0_1, add);  add = None
    diagonal_1 = torch.ops.aten.diagonal.default(diagonal_scatter);  diagonal_1 = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, diagonal_scatter);  arg0_1 = copy_ = None
    return diagonal_scatter
    """,
        )

    def test_channels_last_contiguous(self):
        def f(x):
            return x.contiguous(memory_format=torch.channels_last)
            tmp = torch.ones(2)
            y = x.diagonal()
            y.add_(tmp)
            return x

        x = torch.randn(4, 8, 8, 3).permute(0, 3, 1, 2)
        self.assert_functionalization(f, x)
        logs = self.get_logs(f, x).strip()
        # There should be no clone in the graph
        self.assertExpectedInline(
            logs,
            """\
def forward(self, arg0_1):
    return arg0_1""",
        )

    def test_split(self):
        def f(x):
            # test: view ops that return multiple tensors (split)
            tmp = torch.ones(2)
            y1, y2 = x.split(2)
            y3 = y2.diagonal()
            y3.add_(tmp)
            z = x * x
            return y3

        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(
            logs,
            """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)
    split_copy = torch.ops.aten.split_copy.Tensor(arg0_1, 2)
    getitem = split_copy[0];  getitem = None
    getitem_1 = split_copy[1];  split_copy = None
    diagonal_copy = torch.ops.aten.diagonal_copy.default(getitem_1);  getitem_1 = None
    add = torch.ops.aten.add.Tensor(diagonal_copy, ones);  diagonal_copy = ones = None
    split_copy_1 = torch.ops.aten.split_copy.Tensor(arg0_1, 2)
    getitem_2 = split_copy_1[0];  getitem_2 = None
    getitem_3 = split_copy_1[1];  split_copy_1 = None
    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(getitem_3, add);  getitem_3 = add = None
    slice_scatter = torch.ops.aten.slice_scatter.default(arg0_1, diagonal_scatter, 0, 2, 4);  diagonal_scatter = None
    split_copy_2 = torch.ops.aten.split_copy.Tensor(slice_scatter, 2)
    getitem_4 = split_copy_2[0];  getitem_4 = None
    getitem_5 = split_copy_2[1];  split_copy_2 = None
    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(getitem_5);  getitem_5 = None
    mul = torch.ops.aten.mul.Tensor(slice_scatter, slice_scatter);  mul = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, slice_scatter);  arg0_1 = slice_scatter = copy_ = None
    return diagonal_copy_1
    """,
        )  # noqa: B950

        # NB: even with reapply_views=True, we expect to see scatter op
        reinplaced_logs = self.get_logs(
            f, torch.ones(4, 2), reapply_views=True, run_reinplace=False
        )
        self.assertExpectedInline(
            reinplaced_logs,
            """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)
    split = torch.ops.aten.split.Tensor(arg0_1, 2)
    getitem = split[0];  getitem = None
    getitem_1 = split[1];  split = None
    diagonal = torch.ops.aten.diagonal.default(getitem_1);  getitem_1 = None
    add = torch.ops.aten.add.Tensor(diagonal, ones);  diagonal = ones = None
    split_1 = torch.ops.aten.split.Tensor(arg0_1, 2)
    getitem_2 = split_1[0];  getitem_2 = None
    getitem_3 = split_1[1];  split_1 = None
    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(getitem_3, add);  getitem_3 = add = None
    slice_scatter = torch.ops.aten.slice_scatter.default(arg0_1, diagonal_scatter, 0, 2, 4);  diagonal_scatter = None
    split_2 = torch.ops.aten.split.Tensor(slice_scatter, 2)
    getitem_4 = split_2[0];  getitem_4 = None
    getitem_5 = split_2[1];  split_2 = None
    diagonal_1 = torch.ops.aten.diagonal.default(getitem_5);  getitem_5 = None
    mul = torch.ops.aten.mul.Tensor(slice_scatter, slice_scatter);  mul = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, slice_scatter);  arg0_1 = slice_scatter = copy_ = None
    return diagonal_1
    """,
        )  # noqa: B950

    def test_split_with_sizes(self):
        def f(x):
            # test: view ops that return multiple tensors (split_with_sizes)
            tmp = torch.ones(2)
            y1, y2 = x.split_with_sizes([2, 2])
            y3 = y1.diagonal()
            y3.add_(tmp)
            z = x * x
            return y3

        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(
            logs,
            """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)
    split_with_sizes_copy = torch.ops.aten.split_with_sizes_copy.default(arg0_1, [2, 2])
    getitem = split_with_sizes_copy[0]
    getitem_1 = split_with_sizes_copy[1];  split_with_sizes_copy = getitem_1 = None
    diagonal_copy = torch.ops.aten.diagonal_copy.default(getitem);  getitem = None
    add = torch.ops.aten.add.Tensor(diagonal_copy, ones);  diagonal_copy = ones = None
    split_with_sizes_copy_1 = torch.ops.aten.split_with_sizes_copy.default(arg0_1, [2, 2])
    getitem_2 = split_with_sizes_copy_1[0]
    getitem_3 = split_with_sizes_copy_1[1];  split_with_sizes_copy_1 = getitem_3 = None
    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(getitem_2, add);  getitem_2 = add = None
    slice_scatter = torch.ops.aten.slice_scatter.default(arg0_1, diagonal_scatter, 0, 0, 2);  diagonal_scatter = None
    split_with_sizes_copy_2 = torch.ops.aten.split_with_sizes_copy.default(slice_scatter, [2, 2])
    getitem_4 = split_with_sizes_copy_2[0]
    getitem_5 = split_with_sizes_copy_2[1];  split_with_sizes_copy_2 = getitem_5 = None
    diagonal_copy_1 = torch.ops.aten.diagonal_copy.default(getitem_4);  getitem_4 = None
    mul = torch.ops.aten.mul.Tensor(slice_scatter, slice_scatter);  mul = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, slice_scatter);  arg0_1 = slice_scatter = copy_ = None
    return diagonal_copy_1
    """,
        )  # noqa: B950

        # NB: even with reapply_views=True, we expect to see scatter op
        reinplaced_logs = self.get_logs(
            f, torch.ones(4, 2), reapply_views=True, run_reinplace=False
        )
        self.assertExpectedInline(
            reinplaced_logs,
            """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([2], device = device(type='cpu'), pin_memory = False)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(arg0_1, [2, 2])
    getitem = split_with_sizes[0]
    getitem_1 = split_with_sizes[1];  split_with_sizes = getitem_1 = None
    diagonal = torch.ops.aten.diagonal.default(getitem);  getitem = None
    add = torch.ops.aten.add.Tensor(diagonal, ones);  diagonal = ones = None
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(arg0_1, [2, 2])
    getitem_2 = split_with_sizes_1[0]
    getitem_3 = split_with_sizes_1[1];  split_with_sizes_1 = getitem_3 = None
    diagonal_scatter = torch.ops.aten.diagonal_scatter.default(getitem_2, add);  getitem_2 = add = None
    slice_scatter = torch.ops.aten.slice_scatter.default(arg0_1, diagonal_scatter, 0, 0, 2);  diagonal_scatter = None
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(slice_scatter, [2, 2])
    getitem_4 = split_with_sizes_2[0]
    getitem_5 = split_with_sizes_2[1];  split_with_sizes_2 = getitem_5 = None
    diagonal_1 = torch.ops.aten.diagonal.default(getitem_4);  getitem_4 = None
    mul = torch.ops.aten.mul.Tensor(slice_scatter, slice_scatter);  mul = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, slice_scatter);  arg0_1 = slice_scatter = copy_ = None
    return diagonal_1
    """,
        )  # noqa: B950

    def test_slice(self):
        def f(x):
            tmp = torch.ones(4)
            x.transpose_(1, 0)
            y = x[0:2]
            y.add_(tmp)
            return x

        self.assert_functionalization(f, torch.ones(4, 2), mutated_input_metadata=True)
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(
            logs,
            """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4], device = device(type='cpu'), pin_memory = False)
    transpose_copy = torch.ops.aten.transpose_copy.int(arg0_1, 1, 0)
    slice_copy = torch.ops.aten.slice_copy.Tensor(transpose_copy, 0, 0, 2);  transpose_copy = None
    add = torch.ops.aten.add.Tensor(slice_copy, ones);  slice_copy = ones = None
    transpose_copy_1 = torch.ops.aten.transpose_copy.int(arg0_1, 1, 0);  arg0_1 = None
    slice_scatter = torch.ops.aten.slice_scatter.default(transpose_copy_1, add, 0, 0, 2);  transpose_copy_1 = add = None
    transpose_copy_2 = torch.ops.aten.transpose_copy.int(slice_scatter, 1, 0);  slice_scatter = None
    transpose_copy_3 = torch.ops.aten.transpose_copy.int(transpose_copy_2, 1, 0)
    slice_copy_1 = torch.ops.aten.slice_copy.Tensor(transpose_copy_3, 0, 0, 2);  transpose_copy_3 = slice_copy_1 = None
    transpose_copy_4 = torch.ops.aten.transpose_copy.int(transpose_copy_2, 1, 0);  transpose_copy_2 = None
    return transpose_copy_4
    """,
        )  # noqa: B950

        # NB: even with reapply_views=True, we expect to see scatter op
        reinplaced_logs = self.get_logs(
            f, torch.ones(4, 2), reapply_views=True, run_reinplace=False
        )
        self.assertExpectedInline(
            reinplaced_logs,
            """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4], device = device(type='cpu'), pin_memory = False)
    transpose = torch.ops.aten.transpose.int(arg0_1, 1, 0)
    slice_1 = torch.ops.aten.slice.Tensor(transpose, 0, 0, 2);  transpose = None
    add = torch.ops.aten.add.Tensor(slice_1, ones);  slice_1 = ones = None
    transpose_1 = torch.ops.aten.transpose.int(arg0_1, 1, 0);  arg0_1 = None
    slice_scatter = torch.ops.aten.slice_scatter.default(transpose_1, add, 0, 0, 2);  transpose_1 = add = None
    transpose_2 = torch.ops.aten.transpose.int(slice_scatter, 1, 0);  slice_scatter = None
    transpose_3 = torch.ops.aten.transpose.int(transpose_2, 1, 0)
    slice_2 = torch.ops.aten.slice.Tensor(transpose_3, 0, 0, 2);  transpose_3 = slice_2 = None
    transpose_4 = torch.ops.aten.transpose.int(transpose_2, 1, 0);  transpose_2 = None
    return transpose_4
    """,
        )  # noqa: B950

    def test_view_inplace(self):
        def f(x):
            # test: view + inplace op (transpose_)
            tmp = torch.ones(4)
            x.transpose_(1, 0)
            y = x[0]
            y.add_(tmp)
            return x

        self.assert_functionalization(f, torch.ones(4, 2), mutated_input_metadata=True)
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(
            logs,
            """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4], device = device(type='cpu'), pin_memory = False)
    transpose_copy = torch.ops.aten.transpose_copy.int(arg0_1, 1, 0)
    select_copy = torch.ops.aten.select_copy.int(transpose_copy, 0, 0);  transpose_copy = None
    add = torch.ops.aten.add.Tensor(select_copy, ones);  select_copy = ones = None
    transpose_copy_1 = torch.ops.aten.transpose_copy.int(arg0_1, 1, 0);  arg0_1 = None
    select_scatter = torch.ops.aten.select_scatter.default(transpose_copy_1, add, 0, 0);  transpose_copy_1 = add = None
    transpose_copy_2 = torch.ops.aten.transpose_copy.int(select_scatter, 1, 0);  select_scatter = None
    transpose_copy_3 = torch.ops.aten.transpose_copy.int(transpose_copy_2, 1, 0)
    select_copy_1 = torch.ops.aten.select_copy.int(transpose_copy_3, 0, 0);  transpose_copy_3 = select_copy_1 = None
    transpose_copy_4 = torch.ops.aten.transpose_copy.int(transpose_copy_2, 1, 0);  transpose_copy_2 = None
    return transpose_copy_4
    """,
        )  # noqa: B950

        # NB: even with reapply_views=True, we expect to see scatter op
        reinplaced_logs = self.get_logs(
            f, torch.ones(4, 2), reapply_views=True, run_reinplace=False
        )
        self.assertExpectedInline(
            reinplaced_logs,
            """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4], device = device(type='cpu'), pin_memory = False)
    transpose = torch.ops.aten.transpose.int(arg0_1, 1, 0)
    select = torch.ops.aten.select.int(transpose, 0, 0);  transpose = None
    add = torch.ops.aten.add.Tensor(select, ones);  select = ones = None
    transpose_1 = torch.ops.aten.transpose.int(arg0_1, 1, 0);  arg0_1 = None
    select_scatter = torch.ops.aten.select_scatter.default(transpose_1, add, 0, 0);  transpose_1 = add = None
    transpose_2 = torch.ops.aten.transpose.int(select_scatter, 1, 0);  select_scatter = None
    transpose_3 = torch.ops.aten.transpose.int(transpose_2, 1, 0)
    select_1 = torch.ops.aten.select.int(transpose_3, 0, 0);  transpose_3 = select_1 = None
    transpose_4 = torch.ops.aten.transpose.int(transpose_2, 1, 0);  transpose_2 = None
    return transpose_4
    """,
        )  # noqa: B950

    def test_unbind(self):
        def f(x):
            # test: view + inplace op (transpose_)
            tmp = torch.ones(4)
            x.transpose_(1, 0)
            y, _ = x.unbind(0)
            y.add_(tmp)
            return x

        self.assert_functionalization(f, torch.ones(4, 2), mutated_input_metadata=True)
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(
            logs,
            """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4], device = device(type='cpu'), pin_memory = False)
    transpose_copy = torch.ops.aten.transpose_copy.int(arg0_1, 1, 0)
    unbind_copy = torch.ops.aten.unbind_copy.int(transpose_copy);  transpose_copy = None
    getitem = unbind_copy[0]
    getitem_1 = unbind_copy[1];  unbind_copy = getitem_1 = None
    add = torch.ops.aten.add.Tensor(getitem, ones);  getitem = ones = None
    transpose_copy_1 = torch.ops.aten.transpose_copy.int(arg0_1, 1, 0);  arg0_1 = None
    select_scatter = torch.ops.aten.select_scatter.default(transpose_copy_1, add, 0, 0);  transpose_copy_1 = add = None
    transpose_copy_2 = torch.ops.aten.transpose_copy.int(select_scatter, 1, 0);  select_scatter = None
    transpose_copy_3 = torch.ops.aten.transpose_copy.int(transpose_copy_2, 1, 0)
    unbind_copy_1 = torch.ops.aten.unbind_copy.int(transpose_copy_3);  transpose_copy_3 = None
    getitem_2 = unbind_copy_1[0];  getitem_2 = None
    getitem_3 = unbind_copy_1[1];  unbind_copy_1 = getitem_3 = None
    transpose_copy_4 = torch.ops.aten.transpose_copy.int(transpose_copy_2, 1, 0);  transpose_copy_2 = None
    return transpose_copy_4
    """,
        )  # noqa: B950

        # NB: even with reapply_views=True, we expect to see scatter op
        reinplaced_logs = self.get_logs(
            f, torch.ones(4, 2), reapply_views=True, run_reinplace=False
        )
        self.assertExpectedInline(
            reinplaced_logs,
            """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4], device = device(type='cpu'), pin_memory = False)
    transpose = torch.ops.aten.transpose.int(arg0_1, 1, 0)
    unbind = torch.ops.aten.unbind.int(transpose);  transpose = None
    getitem = unbind[0]
    getitem_1 = unbind[1];  unbind = getitem_1 = None
    add = torch.ops.aten.add.Tensor(getitem, ones);  getitem = ones = None
    transpose_1 = torch.ops.aten.transpose.int(arg0_1, 1, 0);  arg0_1 = None
    select_scatter = torch.ops.aten.select_scatter.default(transpose_1, add, 0, 0);  transpose_1 = add = None
    transpose_2 = torch.ops.aten.transpose.int(select_scatter, 1, 0);  select_scatter = None
    transpose_3 = torch.ops.aten.transpose.int(transpose_2, 1, 0)
    unbind_1 = torch.ops.aten.unbind.int(transpose_3);  transpose_3 = None
    getitem_2 = unbind_1[0];  getitem_2 = None
    getitem_3 = unbind_1[1];  unbind_1 = getitem_3 = None
    transpose_4 = torch.ops.aten.transpose.int(transpose_2, 1, 0);  transpose_2 = None
    return transpose_4
    """,
        )  # noqa: B950

    def test_optional_tensor_list(self):
        def f(x):
            # test: an operator that takes in a List[Optional[Tensor]] argument
            # (index_put)
            y = x.view(8)
            indices = torch.arange(4)
            values = torch.arange(4, dtype=y.dtype)
            y.index_put_((indices,), values, accumulate=False)
            return y

        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(
            logs,
            """\



def forward(self, arg0_1):
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [8])
    arange = torch.ops.aten.arange.default(4, device = device(type='cpu'), pin_memory = False)
    arange_1 = torch.ops.aten.arange.default(4, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    index_put = torch.ops.aten.index_put.default(view_copy, [arange], arange_1);  view_copy = arange = arange_1 = None
    view_copy_1 = torch.ops.aten.view_copy.default(index_put, [4, 2]);  index_put = None
    view_copy_2 = torch.ops.aten.view_copy.default(view_copy_1, [8])
    copy_ = torch.ops.aten.copy_.default(arg0_1, view_copy_1);  arg0_1 = view_copy_1 = copy_ = None
    return view_copy_2
    """,
        )  # noqa: B950

    def test_scalars(self):
        def f(x):
            # test: the pass can handle scalar inputs properly
            tmp = torch.ones(4, 2)
            y = x.view(4, 2)
            y.add_(1)
            z = 2 * y
            z.div_(1)
            return z

        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(
            logs,
            """\



def forward(self, arg0_1):
    ones = torch.ops.aten.ones.default([4, 2], device = device(type='cpu'), pin_memory = False);  ones = None
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [4, 2])
    add = torch.ops.aten.add.Tensor(view_copy, 1);  view_copy = None
    view_copy_1 = torch.ops.aten.view_copy.default(add, [4, 2]);  add = None
    view_copy_2 = torch.ops.aten.view_copy.default(view_copy_1, [4, 2])
    mul = torch.ops.aten.mul.Tensor(view_copy_2, 2);  view_copy_2 = None
    div = torch.ops.aten.div.Tensor(mul, 1);  mul = None
    copy_ = torch.ops.aten.copy_.default(arg0_1, view_copy_1);  arg0_1 = view_copy_1 = copy_ = None
    return div
    """,
        )

    @skipIfTorchDynamo("Test does not work with TorchDynamo")
    def test_metadata_change(self):
        def f(x):
            # ops like ge_() are allowed to change the dtype of the input.
            # functionalization should pick up on that.
            y = x.clone()
            out = y.ge_(0)
            return out

        self.assert_functionalization(f, torch.ones(4, 2))
        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(
            logs,
            """\



def forward(self, arg0_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    ge = torch.ops.aten.ge.Scalar(clone, 0);  clone = None
    _to_copy = torch.ops.aten._to_copy.default(ge, dtype = torch.float32, layout = torch.strided);  ge = None
    return _to_copy
    """,
        )

        reinplaced_logs = self.get_logs(
            f, torch.ones(2, 2), reapply_views=True, run_reinplace=True
        )
        self.assertExpectedInline(
            reinplaced_logs,
            """\



def forward(self, arg0_1):
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    ge = torch.ops.aten.ge.Scalar(clone, 0);  clone = None
    _to_copy = torch.ops.aten._to_copy.default(ge, dtype = torch.float32, layout = torch.strided);  ge = None
    return _to_copy
    """,
        )  # noqa: B950

    @skipIfTorchDynamo("Test does not work with TorchDynamo")
    def test_metadata_change_out_op(self):
        def f(t, y):
            out_1 = torch.ones(1)
            return torch.add(t, y, out=out_1)

        inpt1, inpt2 = torch.tensor([1]), torch.tensor([1])
        inpt1_func, inpt2_func = (
            torch._to_functional_tensor(inpt1),
            torch._to_functional_tensor(inpt2),
        )

        out_ref = f(inpt1, inpt2)
        torch._enable_functionalization(reapply_views=True)
        try:
            out_functional = f(inpt1_func, inpt2_func)
        finally:
            torch._disable_functionalization()
        self.assertEqual(out_ref, torch._from_functional_tensor(out_functional))

    def test_only_one_view(self):
        def f(x):
            # This tests that we don't have any unnecessary views in the trace.
            # If the input wasn't mutated, we don't need to regenerate it,
            # so there should be a total of 1 op in the output trace.
            return x.view(4, 2)

        logs = self.get_logs(f, torch.ones(4, 2))
        self.assertExpectedInline(
            logs,
            """\



def forward(self, arg0_1):
    view_copy = torch.ops.aten.view_copy.default(arg0_1, [4, 2]);  arg0_1 = None
    return view_copy
    """,
        )

    def test_everything(self):
        def f(x):
            # test: eve
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

- **Automatic Differentiation**: Uses autograd for gradient computation


## Performance Considerations

### Performance Notes

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
python docs/test/test_functionalization.py_docs.md
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

- **File Documentation**: `test_functionalization.py_docs.md_docs.md`
- **Keyword Index**: `test_functionalization.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
