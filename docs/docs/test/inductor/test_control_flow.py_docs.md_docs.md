# Documentation: `docs/test/inductor/test_control_flow.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_control_flow.py_docs.md`
- **Size**: 54,151 bytes (52.88 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_control_flow.py`

## File Metadata

- **Path**: `test/inductor/test_control_flow.py`
- **Size**: 78,771 bytes (76.92 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]

import itertools
import unittest

import torch
import torch._dynamo.testing
import torch.utils._pytree as pytree
from torch._higher_order_ops.associative_scan import associative_scan
from torch._higher_order_ops.map import _fake_map
from torch._higher_order_ops.scan import _fake_scan, scan
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import (
    decorateIf,
    instantiate_parametrized_tests,
    parametrize,
    skipIfXpu,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU
from torch.testing._internal.triton_utils import requires_gpu


def _prepend_product_of_values(inputs, possible_values, num_to_prepend=1, device=None):
    result = []
    if len(inputs) != 0:
        device = inputs[0].device
    assert device
    # iterate over the cartesian product of predicate values
    for values in itertools.product(*([possible_values] * num_to_prepend)):
        prepended = [torch.tensor(v, device=device) for v in values]
        result.append((*prepended, *inputs))
    return result


def prepend_predicates(inputs, num_predicates=1, device=None):
    return _prepend_product_of_values(inputs, [False, True], num_predicates, device)


def prepend_counters(inputs, num_counters=1, counter_values=(0, 1, 5)):
    return _prepend_product_of_values(inputs, counter_values, num_counters)


# a testing loss_fn
def loss_fn(result) -> torch.Tensor:
    flat_results, _ = pytree.tree_flatten(result)
    total_loss = torch.tensor(
        0.0, device=flat_results[0].device if flat_results else torch.device("cpu")
    )

    for res in flat_results:
        # Convert to float if integer tensor to avoid numerical issues
        if not res.dtype.is_floating_point:
            res = res.float()

        # Simple robust loss: abs values + small constant to avoid inf/nan
        total_loss = total_loss + (torch.abs(res) / (1.0 + torch.abs(res))).sum()

    return total_loss


class CondModels:
    class Simple(torch.nn.Module):
        def forward(self, p, a, b):
            def true_fn(x, y):
                return x + y

            def false_fn(x, y):
                return x - y

            return torch.cond(p, true_fn, false_fn, [a, b])

    class SimpleWithIntClosure(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num = 3

        def forward(self, p, a, b):
            return torch.cond(
                pred=p,
                true_fn=lambda a, b: [a + b + self.num],
                false_fn=lambda a, b: [a - b - self.num],
                operands=(a, b),
            )

    class Nested(torch.nn.Module):
        def forward(self, p0, p1, p2, a, b, c):
            def true_fn(x0, y0, z0):
                def true_true_fn(x1, y1, z1):
                    return (x1 - y1 * z1) * 3.14

                def true_false_fn(x1, y1, z1):
                    def true_false_true_fn(x2, y2, z2):
                        return (x2 * y2 * z2) / 2.71

                    def true_false_false_fn(x2, y2, z2):
                        return (x2 + y2 + z2) * 1.23

                    return torch.cond(
                        p2, true_false_true_fn, true_false_false_fn, [x1, y1, z1]
                    )

                return torch.cond(p1, true_true_fn, true_false_fn, [x0, y0, z0])

            def false_fn(x0, y0, z0):
                def false_true_fn(x1, y1, z1):
                    def false_true_true_fn(x2, y2, z2):
                        return (x2 - y2 - z2) + 1.23

                    def false_true_false_fn(x2, y2, z2):
                        return (x2 / y2 / z2) - 3.14

                    return torch.cond(
                        p2, false_true_true_fn, false_true_false_fn, [x1, y1, z1]
                    )

                def false_false_fn(x1, y1, z1):
                    return (x1 - y1 * z1) / 2.71

                return torch.cond(p1, false_true_fn, false_false_fn, [x0, y0, z0])

            return torch.cond(p0, true_fn, false_fn, [a, b, c])

    class Parameters(torch.nn.Module):
        class InnerModel1(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.layer = torch.nn.Linear(20, 30, device=device)

            def forward(self, x):
                return self.layer(x + 1) * 3.14

        class InnerModel2(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.layer1 = torch.nn.Linear(20, 10, device=device)
                self.layer2 = torch.nn.Linear(10, 30, device=device)

            def forward(self, x):
                return self.layer2(self.layer1(x - 2)) * 3.14

        def __init__(self, device):
            super().__init__()
            self.true_fn = self.InnerModel1(device)
            self.false_fn = self.InnerModel2(device)

        def forward(self, p, a):
            return torch.cond(p, self.true_fn, self.false_fn, [a])

    class ReinterpretView(torch.nn.Module):
        def forward(self, p, a, b):
            def true_fn(x, y):
                z1 = x + y
                z2 = x - y
                return z1[2:], z2[:, 4:].contiguous()

            def false_fn(x, y):
                z1 = x - y
                z2 = x + y
                return z1[2:], z2[:, 4:].contiguous()

            return torch.cond(p, true_fn, false_fn, [a[:-1], b[:-1]])

    class MultipleOutputs(torch.nn.Module):
        def forward(self, p, a, b, c):
            def true_fn(x, y, z):
                return x * y, z / 2.71, (y - x).sum(dim=1)

            def false_fn(x, y, z):
                return y / x, z * 3.14, (x + y).mean(dim=1)

            return torch.cond(p, true_fn, false_fn, [a, b, c])

    class OuterCode(torch.nn.Module):
        def forward(self, p, a, b):
            c = a * b + 3.14
            d = a / b - 2.71

            def true_fn(x, y):
                return x + y

            def false_fn(x, y):
                return x - y

            e = torch.cond(p, true_fn, false_fn, [c, d])

            return e * e / 1.41

    class OuterBuffers(torch.nn.Module):
        def forward(self, p, a, b, c):
            d = a * 2
            e = b / 2

            def true_fn(x):
                return x + d

            def false_fn(x):
                return x - e

            return torch.cond(p, true_fn, false_fn, [c])

    class WithNonTensorPredicate(torch.nn.Module):
        def forward(self, a, b):
            def true_fn(x, y):
                return x.sum(0) / 3.14

            def false_fn(x, y):
                return y.sum(0) * 2.71

            return torch.cond(a.size(0) > b.size(0), true_fn, false_fn, [a, b])

    class UnbackedSymIntClosure(torch.nn.Module):
        def forward(self, p, x, y, z):
            a = y.shape[0]
            b = z.sum().to(torch.int64).item()

            def true_fn(x):
                return x + a

            def false_fn(x):
                return x + b * z

            return torch.cond(x.shape[0] > 5, true_fn, false_fn, (x,))

    class MismatchedOutputSize(torch.nn.Module):
        def forward(self, p, x, y, z):
            a = y.shape[0]
            b = z.shape[0]

            def true_fn(x):
                return (x + a)[2:].sin()

            def false_fn(x):
                return (x + b * z)[:2].cos()

            return y.sum() - torch.cond(x.sum() > 0, true_fn, false_fn, (x,))

    class FunctionalCall(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, p, x):
            true_new_weight = torch.ones(x.size(0), x.size(0), device=x.device)
            false_new_weight = torch.zeros(x.size(0), x.size(0), device=x.device)
            true_new_bias = torch.ones(x.size(0), device=x.device)
            false_new_bias = torch.zeros(x.size(0), device=x.device)
            x = x.reshape(-1, x.size(0))

            def true_fn(x):
                return torch.func.functional_call(
                    self.linear,
                    {
                        "weight": true_new_weight,
                        "bias": true_new_bias,
                    },
                    x,
                )

            def false_fn(x):
                return torch.func.functional_call(
                    self.linear,
                    {
                        "weight": false_new_weight,
                        "bias": false_new_bias,
                    },
                    x,
                )

            return torch.cond(p, true_fn, false_fn, (x,))

    class SelectWithInputIdx(torch.nn.Module):
        def forward(self, p, x, idx):
            u0 = idx.item()
            x0 = x.select(0, u0)

            def fn():
                return x0.sin()

            return torch.cond(x0.sum() > 0, fn, fn)


class CondTests(TestCase):
    def _run_test(
        self,
        model,
        inputs,
        device,
        dynamic=False,
        num_predicates=1,
    ):
        cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        compiled_model = torch.compile(backend=cnt, fullgraph=True)(model)

        inputs = [inp.to(device=device) for inp in inputs]
        input_sets = [inputs]
        if dynamic:
            larger_inputs = []
            for inp in inputs:
                # only tile non-scalar tensor inputs
                if inp.ndim > 0:
                    # tile every first dim 5x
                    tiling = [5] + [1] * (inp.ndim - 1)
                    larger_inputs.append(torch.tile(inp, tiling))
                else:
                    larger_inputs.append(inp)
            input_sets.append(larger_inputs)
            for inputs in input_sets:
                for inp in inputs:
                    # mark every first dim as dynamic
                    torch._dynamo.mark_dynamic(inp, 0)

        for inputs in input_sets:
            for inputs_with_predicates in prepend_predicates(
                inputs, num_predicates, device=device
            ):
                cloned_inputs = [inp.clone() for inp in inputs_with_predicates]
                result = model(*inputs_with_predicates)
                result_compiled = compiled_model(*inputs_with_predicates)
                # inputs must not be mutated
                torch.testing.assert_close(cloned_inputs, inputs_with_predicates)
                torch.testing.assert_close(result, result_compiled)

        self.assertEqual(cnt.frame_count, 1, "only one compilation expected")

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_simple_control_flow(self, device, dynamic):
        # cond control flow without nesting
        self._run_test(
            model=CondModels.Simple(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_simple_with_int_closure(self, device):
        self._run_test(
            model=torch.compile(CondModels.SimpleWithIntClosure(), dynamic=True),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_cond_unbacked_symint_closure(self, device, dynamic):
        self._run_test(
            model=CondModels.UnbackedSymIntClosure(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
        )

    @skipIfXpu(msg="Remove this skip after issue #154949 resolved.")
    @requires_gpu
    def test_cond_control_flow_with_precomputed_size(self):
        class TestModel(torch.nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                self.conv2d = torch.nn.Conv2d(
                    512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
                )
                self.threshold = 20

            def forward(self, x: torch.Tensor, index) -> torch.Tensor:
                def true_fn(x: torch.Tensor):
                    return self.conv2d(x)

                def false_fn(x: torch.Tensor):
                    return self.conv2d(x)

                return torch.cond(
                    index < self.threshold and index >= 0, true_fn, false_fn, (x,)
                )

        main_model = TestModel().to(GPU_TYPE)
        x1 = torch.rand(2, 512, 128, 72).to(GPU_TYPE)
        x2 = torch.rand(2, 512, 96, 96).to(GPU_TYPE)

        opt_model = torch.compile(main_model)
        out1 = main_model(x1, 1)
        opt_out1 = opt_model(x1, 1)
        self.assertTrue(torch.allclose(out1, opt_out1, atol=1e-5))

        out2 = main_model(x2, 30)
        opt_out2 = opt_model(x2, 30)
        self.assertTrue(torch.allclose(out2, opt_out2, atol=1e-5))

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_nested_control_flow(self, device, dynamic):
        # cond control flow with nesting
        self._run_test(
            model=CondModels.Nested(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
            num_predicates=3,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_outer_code_before_after(self, device, dynamic):
        # some code before and after the conditional
        self._run_test(
            model=CondModels.OuterCode(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_multiple_outputs(self, device, dynamic):
        # multiple outputs with different shapes
        self._run_test(
            model=CondModels.MultipleOutputs(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
                torch.randn(30, 40),
            ),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_advanced_dynamic_shapes(self, device):
        # subgraphs input shapes include symbolic expressions
        class Model(torch.nn.Module):
            def forward(self, p, a, b):
                def true_fn(x, y):
                    return torch.cat([x - 3, y * 3], dim=1)

                def false_fn(x, y):
                    return torch.cat([x / 3, y - 3], dim=1)

                c = torch.cat([a, b], dim=0)
                d = c * 2
                e = c / 2

                return torch.cond(p, true_fn, false_fn, [d, e])

        self._run_test(
            model=Model(),
            inputs=(
                torch.randn(2, 3, 3),
                torch.randn(4, 3, 3),
            ),
            device=device,
            dynamic=True,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_unbacked_symint_outer_to_inner(self, device):
        class Model(torch.nn.Module):
            def forward(self, p, a):
                def true_fn(x):
                    return torch.cos(x)

                def false_fn(x):
                    return torch.sin(x)

                nz = torch.nonzero(a)
                b = torch.ones([nz.size(0), 8], device=nz.device)

                return torch.cond(p, true_fn, false_fn, [b])

        with torch._dynamo.config.patch(
            {
                "capture_dynamic_output_shape_ops": True,
            }
        ):
            self._run_test(
                model=Model(),
                inputs=(torch.randn(2, 3, 3),),
                device=device,
                dynamic=True,
            )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @torch._inductor.config.patch(size_asserts=False)
    # TODO: graph partition does not support creating tensor
    # with dynamic shape in conditional subgraph yet
    @torch._inductor.config.patch(graph_partition=False)
    def test_cond_unbacked_symint_inner(self, device):
        class Model(torch.nn.Module):
            def forward(self, p, a):
                def true_fn(x):
                    nz = torch.nonzero(x)
                    b = torch.ones([nz.size(0), 8], device=nz.device)
                    return torch.cos(b)

                def false_fn(x):
                    nz = torch.nonzero(x)
                    b = torch.ones([nz.size(0), 8], device=nz.device)
                    return torch.sin(b)

                b = torch.sin(a)

                return torch.cond(p, true_fn, false_fn, [b])

        with torch._dynamo.config.patch(
            {
                "capture_dynamic_output_shape_ops": True,
            }
        ):
            self._run_test(
                model=Model(),
                inputs=(torch.randn(2, 3, 3),),
                device=device,
                dynamic=True,
            )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_unbacked_symint_inner_to_outer(self, device):
        class Model(torch.nn.Module):
            def forward(self, p, a):
                def true_fn(x):
                    nz = torch.nonzero(x)
                    b = torch.ones([nz.size(0), 8], device=nz.device)
                    return torch.cos(b)

                def false_fn(x):
                    nz = torch.nonzero(x)
                    b = torch.ones([nz.size(0), 8], device=nz.device)
                    return torch.sin(b)

                b = torch.sin(a)

                y = torch.cond(p, true_fn, false_fn, [b])
                return torch.sin(y)

        with torch._dynamo.config.patch(
            {
                "capture_dynamic_output_shape_ops": True,
            }
        ):
            self._run_test(
                model=Model(),
                inputs=(torch.randn(2, 3, 3),),
                device=device,
                dynamic=True,
            )

    @requires_gpu
    def test_cond_use_buffers_from_outer_scope(self):
        # subgraphs input shapes include symbolic expressions
        self._run_test(
            model=CondModels.OuterBuffers(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=GPU_TYPE,
            dynamic=False,
        )

    @requires_gpu
    def test_cond_reintepret_view_inputs_outputs(self):
        # ReinterpretView in inputs and outputs of the subgraphs
        self._run_test(
            model=CondModels.ReinterpretView(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=GPU_TYPE,
            dynamic=True,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_subgraphs_with_parameters(self, device, dynamic):
        # nested Modules with parameters
        self._run_test(
            model=CondModels.Parameters(device),
            inputs=(torch.randn(10, 20),),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    def test_cond_non_tensor_predicates(self, device, dynamic):
        # model with a boolean predicate
        for b_size_0 in [5, 15]:
            torch._dynamo.reset()
            self._run_test(
                model=CondModels.WithNonTensorPredicate(),
                inputs=(
                    torch.randn(10, 20),
                    torch.randn(b_size_0, 20),
                ),
                device=device,
                dynamic=dynamic,
                num_predicates=0,
            )

    @requires_gpu
    def test_cond_aliasing_outputs(self):
        # output aliasing in subgraphs: not supported
        class Model(torch.nn.Module):
            def forward(self, p, a, b):
                def true_fn(x, y):
                    z = x + y
                    return z, z[1:]

                def false_fn(x, y):
                    z = x - y
                    return z, z[1:]

                return torch.cond(p, true_fn, false_fn, [a, b])

        # AssertionError: Output aliasing is currently not supported...
        with self.assertRaises(torch._dynamo.exc.UncapturedHigherOrderOpError):
            torch.compile(Model())(
                torch.tensor(True),
                torch.randn(10, 20),
                torch.randn(10, 20),
            )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_decompose_ops_in_subgraph(self, device):
        class Model(torch.nn.Module):
            def forward(self, p, a):
                def true_fn(x):
                    return torch.zeros_like(x)

                def false_fn(x):
                    return torch.ones_like(x)

                b = torch.ones_like(a)
                c = torch.cond(p, true_fn, false_fn, [b])
                return c

        self._run_test(
            model=Model(),
            inputs=(torch.rand(10, 20),),
            device=device,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    def test_cond_decompose_ops_in_subgraph_recursive(self, device):
        def inner_fn1(x):
            return torch.zeros_like(x)

        def inner_fn2(x):
            return torch.ones_like(x)

        class Model(torch.nn.Module):
            def forward(self, p, a):
                def true_fn(x):
                    return torch.cond(p, inner_fn2, inner_fn1, [x])

                def false_fn(x):
                    return torch.cond(p, inner_fn1, inner_fn2, [x])

                b = torch.ones_like(a)
                c = torch.cond(p, true_fn, false_fn, [b])
                return c

        self._run_test(
            model=Model(),
            inputs=(torch.rand(10, 20),),
            device=device,
        )

    @requires_gpu
    def test_cond_inductor_fx_passes_recursively_applied(self):
        counters = {"pre_grad": 0, "post_grad": 0}

        def pre_grad_pass_counter(gm):
            counters["pre_grad"] += 1

        def post_grad_pass_counter(gm):
            counters["post_grad"] += 1

        with torch._inductor.config.patch(
            {
                "pre_grad_custom_pass": pre_grad_pass_counter,
                "post_grad_custom_pre_pass": post_grad_pass_counter,
                # The above patches don't pickle
                "fx_graph_cache": False,
            }
        ):
            self._run_test(
                model=CondModels.Nested(),
                inputs=(
                    torch.randn(10, 20),
                    torch.randn(10, 20),
                    torch.randn(10, 20),
                ),
                device=GPU_TYPE,
                dynamic=True,
                num_predicates=3,
            )

        self.assertEqual(counters["pre_grad"], 11)
        self.assertEqual(counters["post_grad"], 11)

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    def test_cond_mismatched_branch_output_size(self, device, dynamic):
        self._run_test(
            model=CondModels.MismatchedOutputSize(),
            inputs={
                torch.randn(10, 20),
                torch.randn(10, 20),
                torch.randn(10, 20),
            },
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    def test_cond_functional_call(self, device, dynamic):
        self._run_test(
            model=CondModels.FunctionalCall(),
            inputs=(torch.randn(10, 20),),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_cond_select_with_input_idx(self, device, dynamic):
        self._run_test(
            model=CondModels.SelectWithInputIdx(),
            inputs=(torch.randn(10, 20), torch.tensor(0, dtype=torch.int64)),
            device=device,
            dynamic=dynamic,
        )

    @requires_gpu
    def test_output_on_different_device(self):
        class FactoryBranches(torch.nn.Module):
            def forward(self, pred):
                tensor = torch.cond(
                    pred,
                    lambda: torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).to(
                        GPU_TYPE
                    ),
                    lambda: torch.zeros(5, dtype=torch.float32).to(GPU_TYPE),
                )
                return tensor + 1

        self._run_test(
            model=FactoryBranches(),
            inputs=(),
            device="cpu",  # device for predicate
            dynamic=True,
        )


class WhileLoopModels:
    class Simple(torch.nn.Module):
        def forward(self, ci, a, b):
            def cond_fn(i, x, y):
                return i > 0

            def body_fn(i, x, y):
                return i - 1, x + y, y - x

            return torch._higher_order_ops.while_loop(cond_fn, body_fn, [ci, a, b])

    class Nested(torch.nn.Module):
        def forward(self, ci, cj, a, b):
            def cond_fn(i1, j1, x1, y1):
                return i1 > 0

            def body_fn(i1, j1, x1, y1):
                def cond_fn_nested(i2, j2, x2, y2):
                    return j2 > 0

                def body_fn_nested(i2, j2, x2, y2):
                    return i2.clone(), j2 - 1, x2 + 3.14, y2 - 2.71

                i1, j1, x1, y1 = torch._higher_order_ops.while_loop(
                    cond_fn_nested, body_fn_nested, [i1, j1, x1, y1]
                )

                return i1 - 1, j1.clone(), x1 * 2, y1 / 2

            return torch._higher_order_ops.while_loop(cond_fn, body_fn, (ci, cj, a, b))

    class Parameters(torch.nn.Module):
        class InnerModel(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.layer1 = torch.nn.Linear(
                    20, 30, device=device, dtype=torch.float64
                )
                self.layer2 = torch.nn.Linear(
                    30, 20, device=device, dtype=torch.float64
                )

            def forward(self, c, x):
                return c - 1, self.layer2(self.layer1(x - 2)) * 3.14

        def __init__(self, device):
            super().__init__()
            self.body_fn = self.InnerModel(device)
            self.cond_fn = lambda c, x: c > 0

        def forward(self, c, a):
            return torch._higher_order_ops.while_loop(
                self.cond_fn, self.body_fn, [c, a]
            )

    class OuterCode(torch.nn.Module):
        def forward(self, c, a, b):
            d = a * b + 3.14
            e = a / b - 2.71

            def cond_fn(c, x, y):
                return c > 0

            def body_fn(c, x, y):
                return c - 1, y - x, x + y

            _, f, g = torch._higher_order_ops.while_loop(cond_fn, body_fn, [c, d, e])

            return f * g / 1.41

    # TODO(aakhundov): add while_loop test with outer buffers
    # with dynamic=True once dynamo / export allows while_loop
    # closure capture with mark_dynamic:
    # https://github.com/pytorch/pytorch/issues/123596
    class OuterBuffers(torch.nn.Module):
        def forward(self, c, a, b):
            d = a * 2
            e = b / 2

            def cond_fn(c, x, y):
                return c > 0

            def body_fn(c, x, y):
                return c - 1, x + d, y - e

            return torch._higher_order_ops.while_loop(cond_fn, body_fn, [c, a, b])

    class PytreeCarry(torch.nn.Module):
        def forward(self, it, pytree_input):
            def cond_fn(it, pytree_input):
                return it > 0

            def body_fn(it, pytree_input):
                x = pytree_input[0][0]
                y = pytree_input[1]["x"]
                z = pytree_input[1]["y"]
                new_x = y.sin()
                new_y = z.cos()
                new_z = x + 1
                return it - 1, ([new_x], {"x": new_y, "y": new_z})

            return torch._higher_order_ops.while_loop(
                cond_fn, body_fn, (it, pytree_input)
            )

    class DataDependentOpInSubgraph(torch.nn.Module):
        def forward(self, c, a, b):
            def cond_fn(c, reduced_carry):
                return c > 0

            def body_fn(c, reduced_carry):
                k = torch.masked_select(a, b)
                d = torch.concat([k, k * 2])
                return c - 1, torch.min(d).unsqueeze(0) + reduced_carry

            return torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                [c, torch.zeros([1], dtype=torch.int64, device=c.device)],
            )

    class DataDependentInOut(torch.nn.Module):
        def forward(self, c, a, b):
            inp = torch.zeros(
                a.sum().to(torch.int64).item(), 3, device=a.device, dtype=torch.int64
            )

            def cond_fn(c, inp):
                return c > 0

            def body_fn(c, inp):
                return c - 1, (inp.sin() + 1).to(torch.int64)

            return torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                [c, inp],
            )

    class DataDependentInOutMismatch(torch.nn.Module):
        def forward(self, c, a, b):
            def cond_fn(c, a, b):
                return c > 0

            def body_fn(c, a, b):
                return c - 1, a.nonzero(), b.nonzero()

            return torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                [c, a, b],
            )

    class InfiniteLoop(torch.nn.Module):
        def forward(self, c, a):
            a_view = a.view(-1, 1)

            def cond_fn(c, a_view):
                return a_view.size(-1) > 0

            def body_fn(c, a_view):
                return c - 1, a_view + 1

            return torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                [c, a_view],
            )

    class ZeroLoop(torch.nn.Module):
        def forward(self, c, a):
            a_view = torch.sin(a.view(-1, 1))

            def cond_fn(c, a_view):
                return a_view.size(-1) == 0

            def body_fn(c, a_view):
                return c - 1, a_view + 1

            out1, out2 = torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                [c, a_view],
            )
            return out1 + 1, out2 + 2

    class ZeroLoop2(torch.nn.Module):
        def forward(self, c, a):
            a_view = torch.sin(a.view(-1, 1))

            def cond_fn(c, a_view):
                return False

            def body_fn(c, a_view):
                return c - 1, a_view + 1

            out1, out2 = torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                [c, a_view],
            )
            return out1 + 1, out2 + 2

    class ZeroLoop3(torch.nn.Module):
        def forward(self, c, a):
            a_view = torch.sin(a.view(-1, 1))

            def cond_fn(c, a_view):
                return 0

            def body_fn(c, a_view):
                return c - 1, a_view + 1

            out1, out2 = torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                [c, a_view],
            )
            return out1 + 1, out2 + 2

    class ZeroLoop4(torch.nn.Module):
        def forward(self, c, a):
            a_view = torch.sin(a.view(-1, 1))

            def cond_fn(c, a_view):
                return torch.clip(a_view.sum(), 0, 1) < 0

            def body_fn(c, a_view):
                return c - 1, a_view + 1

            out1, out2 = torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                [c, a_view],
            )
            return out2.sin_(), a_view.cos_()

    class UnbackedSymIntClosure(torch.nn.Module):
        def forward(self, c, a, b):
            d = a.sum().to(torch.int64).item()
            e = torch.nonzero(b).size(0)

            def cond_fn(c, a, b):
                return c > d + e + a.shape[0] - b.shape[0]

            def body_fn(c, a, b):
                return c - 1, a + e, b + d

            return torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                [c, a, b],
            )

    class SymExprCond(torch.nn.Module):
        def forward(self, c, a, b):
            d = a.sum().to(torch.int64).item()
            e = torch.nonzero(b).size(0)

            def cond_fn(c, a, b):
                return c + d + e + a.shape[0] - b.shape[0] < 10

            def body_fn(c, a, b):
                return c + 1, a + e, b + d

            return torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                [c, a, b],
            )

    class MixedDevice(torch.nn.Module):
        def forward(self, c, a, b):
            # Force the loop idx on cpu
            c = c.to(torch.device("cpu"))

            def cond_fn(loop_idx, a, b):
                return loop_idx < a.shape[0]

            def body_fn(loop_idx, a, b):
                return loop_idx + 1, a + b, a - b

            return torch._higher_order_ops.while_loop(cond_fn, body_fn, (c, a, b))

    class MixedDevice2(torch.nn.Module):
        def forward(self, c, a, b):
            # Force the loop idx on cpu
            c.to(torch.device("cpu"))

            def cond_fn(loop_idx, a, b):
                return loop_idx < a.shape[0]

            def body_fn(loop_idx, a, b):
                return loop_idx + a.sum(), a + b, a - b

            return torch._higher_order_ops.while_loop(cond_fn, body_fn, (c, a, b))

    class Conv(torch.nn.Module):
        def __init__(self, device):
            super().__init__()
            self.conv2d = torch.nn.Conv2d(
                4,
                4,
                (3, 3),
                stride=(1, 1),
                padding=(1, 1),
                device=device,
                dtype=torch.float64,
            )

        def forward(self, c, x):
            def cond_fn(loop_idx, x):
                return loop_idx < x.size(0)

            def body_fn(loop_idx, x):
                return loop_idx + 1, self.conv2d(x) + 1

            return torch._higher_order_ops.while_loop(
                cond_fn,
                body_fn,
                (c, x),
            )

    class WhileLoopStackOutputSimple(torch.nn.Module):
        def __init__(self, device):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3, device=device)

        def forward(self, c, x):
            def cond_fn(c, x):
                return c < x.size(0)

            def body_fn(c, x):
                return c + 1, self.linear(x)

            stacked_c, stacked_x = torch.ops.higher_order.while_loop_stack_output(
                cond_fn, body_fn, (c, x), tuple()
            )
            return stacked_c, stacked_x


class WhileLoopTests(TestCase):
    def _run_test(
        self, model, inputs, device, dynamic=False, num_counters=1, autograd=False
    ):
        import torch.utils._pytree as pytree

        cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        import copy

        if not autograd:
            for p in model.parameters():
                p.requires_grad_(False)

        compiled_model = copy.deepcopy(model)
        compiled_fn = torch.compile(backend=cnt, fullgraph=True)(compiled_model)

        inputs = pytree.tree_map(lambda t: t.to(device=device), inputs)
        input_sets = [inputs]

        def mark_first_dim_dyn(inp):
            torch._dynamo.mark_dynamic(inp, 0)

        if dynamic:

            def tile_fn(inp):
                # tile every first dim 5x
                tiling = [5] + [1] * (inp.ndim - 1)
                t = torch.tile(inp, tiling)
                return t

            larger_inputs = pytree.tree_map(tile_fn, inputs)
            input_sets.append(larger_inputs)

        for inputs in input_sets:
            flat_inputs, inp_spec = pytree.tree_flatten(inputs)
            for flat_inputs_with_counters in prepend_counters(
                flat_inputs, num_counters
            ):
                counters, flat = (
                    flat_inputs_with_counters[:num_counters],
                    flat_inputs_with_counters[num_counters:],
                )
                unflat_inputs = pytree.tree_unflatten(flat, inp_spec)
                inputs_with_counters = counters + unflat_inputs

                def process_inputs(inp):
                    inp = inp.clone()
                    if dynamic:
                        mark_first_dim_dyn(inp)

                    if autograd and inp.dtype.is_floating_point:
                        inp.requires_grad_(True)
                    return inp

                cloned_inputs = pytree.tree_map(process_inputs, inputs_with_counters)
                cloned_inputs2 = pytree.tree_map(process_inputs, inputs_with_counters)

                result = model(*cloned_inputs)
                result_compiled = compiled_fn(*cloned_inputs2)
                # inputs must not be mutated
                torch.testing.assert_close(cloned_inputs, inputs_with_counters)
                torch.testing.assert_close(
                    result, result_compiled, atol=1e-4, rtol=1e-4
                )

                if autograd and any(
                    pytree.tree_map_only(
                        torch.Tensor, lambda t: t.requires_grad, cloned_inputs
                    )
                ):
                    result_loss = loss_fn(pytree.tree_flatten(result)[0])
                    compiled_loss = loss_fn(pytree.tree_flatten(result_compiled)[0])
                    self.assertTrue(
                        not torch.isnan(result_loss) and not torch.isinf(compiled_loss)
                    )
                    self.assertTrue(
                        not torch.isnan(compiled_loss)
                        and not torch.isinf(compiled_loss)
                    )

                    self.assertEqual(result_loss, compiled_loss)

                    result_loss.backward()
                    compiled_loss.backward()

                    model_parameters = dict(model.named_parameters())
                    compiled_parameters = dict(compiled_model.named_parameters())
                    for name, param in model_parameters.items():
                        self.assertEqual(param, compiled_parameters[name])
                        self.assertEqual(
                            param.grad,
                            compiled_parameters[name].grad,
                            atol=1e-4,
                            rtol=1e-4,
                        )

                    for inp1, inp2 in zip(
                        pytree.tree_flatten(cloned_inputs)[0],
                        pytree.tree_flatten(cloned_inputs2)[0],
                    ):
                        if inp1.requires_grad:
                            self.assertEqual(
                                inp1.grad,
                                inp2.grad,
                                atol=1e-4,
                                rtol=1e-4,
                            )

        self.assertEqual(cnt.frame_count, 1, "only one compilation expected")

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    @parametrize("autograd", [False, True])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_while_loop_simple_control_flow(self, device, dynamic, autograd):
        # while_loop control flow without nesting
        self._run_test(
            model=WhileLoopModels.Simple(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    @parametrize("autograd", [False, True])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_while_loop_nested_control_flow(self, device, dynamic, autograd):
        # while_loop control flow with nesting
        self._run_test(
            model=WhileLoopModels.Nested(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
            num_counters=2,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    @parametrize("autograd", [False, True])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_while_loop_with_outer_code(self, device, dynamic, autograd):
        # while_loop control flow with outer code
        self._run_test(
            model=WhileLoopModels.OuterCode(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [False, True])
    @parametrize("autograd", [False, True])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_while_loop_with_parameters(self, device, dynamic, autograd):
        # while_loop control flow with parameters
        self._run_test(
            model=WhileLoopModels.Parameters(device),
            inputs=(torch.randn(10, 20, dtype=torch.float64),),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    # dynamic=True doesn't work now due to
    # https://github.com/pytorch/pytorch/issues/123596
    @parametrize("dynamic", [False])
    @parametrize("autograd", [False, True])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_while_loop_with_outer_buffers(self, device, dynamic, autograd):
        # while_loop control flow with outer code
        self._run_test(
            model=WhileLoopModels.OuterBuffers(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("autograd", [False, True])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_while_loop_with_pytree_inputs(self, device, dynamic, autograd):
        self._run_test(
            model=WhileLoopModels.PytreeCarry(),
            inputs=(
                (
                    [torch.randn(10, 20)],
                    {"x": torch.randn(10, 20), "y": torch.randn(10, 20)},
                ),
            ),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("autograd", [False, True])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_while_loop_with_data_dependent_ops(self, device, dynamic, autograd):
        with torch._dynamo.config.patch(
            {
                "capture_dynamic_output_shape_ops": True,
            }
        ):
            self._run_test(
                model=WhileLoopModels.DataDependentOpInSubgraph(),
                inputs=(
                    torch.tensor([1, 2, 3, 4, 5]),
                    torch.tensor(
                        [True, True, True, True, True],
                    ),
                ),
                device=device,
                dynamic=dynamic,
                autograd=autograd,
            )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("autograd", [False, True])
    @torch._dynamo.config.patch("capture_scalar_outputs", True)
    def test_while_loop_with_data_dependent_in_out(self, device, dynamic, autograd):
        with torch._dynamo.config.patch(
            {
                "capture_dynamic_output_shape_ops": True,
                "capture_scalar_outputs": True,
            }
        ):
            self._run_test(
                model=WhileLoopModels.DataDependentInOut(),
                inputs=(
                    torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]),
                    torch.tensor(
                        [True, True, True, True, True],
                    ),
                ),
                device=device,
                dynamic=dynamic,
                autograd=autograd,
            )

    @parametrize("dynamic", [True, False])
    def test_while_loop_with_data_dependent_in_out_mismatch(self, dynamic):
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Expected body_fn_output and carried_inputs to have same metadata but found",
        ):
            with torch._dynamo.config.patch(
                {
                    "capture_dynamic_output_shape_ops": True,
                }
            ):
                self._run_test(
                    model=WhileLoopModels.DataDependentInOutMismatch(),
                    inputs=(
                        torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]),
                        torch.tensor(
                            [True, True, True, True, True],
                        ),
                    ),
                    device="cpu",
                    dynamic=dynamic,
                )

    def test_while_loop_infinite_loop_error(self):
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "while_loop doesn't work unless it is captured completely",
        ):
            self._run_test(
                model=WhileLoopModels.InfiniteLoop(),
                inputs=(torch.tensor([1, 2, 3, 4, 5]),),
                device="cpu",
                dynamic=False,
            )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    def test_while_loop_zero_loop(self, device, dynamic):
        for model in [
            WhileLoopModels.ZeroLoop(),
            WhileLoopModels.ZeroLoop2(),
            WhileLoopModels.ZeroLoop3(),
            WhileLoopModels.ZeroLoop4(),
        ]:
            self._run_test(
                model=model,
                inputs=(torch.tensor([1, 2, 3, 4, 5]),),
                device=device,
                dynamic=dynamic,
            )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @torch._dynamo.config.patch(
        {"capture_scalar_outputs": True, "capture_dynamic_output_shape_ops": True}
    )
    @parametrize("autograd", [False, True])
    def test_while_loop_with_unbacked_symint_closure(self, device, dynamic, autograd):
        self._run_test(
            model=WhileLoopModels.UnbackedSymIntClosure(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=dynamic,
            autograd=autograd,
        )

    @requires_gpu
    @parametrize("device", [GPU_TYPE])
    def test_while_loop_models_with_mixed_device(self, device):
        self._run_test(
            model=WhileLoopModels.MixedDevice(),
            inputs=(
                torch.randn(10, 20),
                torch.randn(10, 20),
            ),
            device=device,
            dynamic=True,
        )

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Expected body_fn_output and carried_inputs to have same metadata but found",
        ):
            # Error at front end because device are promoted to a different one
            # after the first iteration
            self._run_test(
                model=WhileLoopModels.MixedDevice2(),
                inputs=(
                    torch.randn(10, 20),
                    torch.randn(10, 20),
                ),
                device=device,
                dynamic=True,
            )

    @requires_gpu
    @parametrize("device", ["cpu", GPU_TYPE])
    @parametrize("dynamic", [True, False])
    @parametrize("autograd", [False, True])
    @
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/inductor/test_control_flow.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/inductor`):

- [`test_snode_runtime.py_kw.md_docs.md`](./test_snode_runtime.py_kw.md_docs.md)
- [`test_metrics.py_docs.md_docs.md`](./test_metrics.py_docs.md_docs.md)
- [`test_flex_attention.py_kw.md_docs.md`](./test_flex_attention.py_kw.md_docs.md)
- [`test_cuda_repro.py_kw.md_docs.md`](./test_cuda_repro.py_kw.md_docs.md)
- [`test_fxir_backend.py_kw.md_docs.md`](./test_fxir_backend.py_kw.md_docs.md)
- [`test_split_cat_fx_passes.py_kw.md_docs.md`](./test_split_cat_fx_passes.py_kw.md_docs.md)
- [`test_mmdecomp.py_kw.md_docs.md`](./test_mmdecomp.py_kw.md_docs.md)
- [`test_torchinductor_codegen_config_overrides.py_kw.md_docs.md`](./test_torchinductor_codegen_config_overrides.py_kw.md_docs.md)
- [`test_aot_inductor_custom_ops.py_kw.md_docs.md`](./test_aot_inductor_custom_ops.py_kw.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_control_flow.py_docs.md_docs.md`
- **Keyword Index**: `test_control_flow.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
