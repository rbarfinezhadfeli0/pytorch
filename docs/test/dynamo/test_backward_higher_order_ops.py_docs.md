# Documentation: `test/dynamo/test_backward_higher_order_ops.py`

## File Metadata

- **Path**: `test/dynamo/test_backward_higher_order_ops.py`
- **Size**: 12,475 bytes (12.18 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]
# flake8: noqa: B950

import functools
import itertools
from unittest import mock

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils
from torch import _inductor as inductor
from torch._dynamo import compiled_autograd
from torch._dynamo._trace_wrapped_higher_order_op import trace_wrapped
from torch._dynamo.testing import normalize_gm
from torch.fx.experimental.proxy_tensor import make_fx


def _multiply(x):
    return x * x


def _multiply_invoke(grad):
    return trace_wrapped(grad, fn=_multiply)


class BackwardHigherOrderOpTests(torch._dynamo.test_case.TestCase):
    def test_invoke_in_eager(self):
        x = torch.tensor([0.5, 0.5], requires_grad=True)
        y = torch.tensor([0.5, 0.5], requires_grad=True)

        def fn(x, y):
            x.register_hook(_multiply_invoke)
            return x * y

        out = fn(x, y)
        grad_out = torch.tensor([2.0, 2.0])
        out.backward(grad_out)
        self.assertEqual(x.grad, y * grad_out)

    def test_invoke_in_pt2(self):
        for backend in ["eager", "aot_eager", "inductor"]:
            torch._dynamo.reset()
            x = torch.tensor([0.5, 0.5], requires_grad=True)
            y = torch.tensor([0.5, 0.5], requires_grad=True)

            def fn(x, y):
                x.register_hook(_multiply_invoke)
                return x * y

            fn = torch.compile(fn, backend=backend)
            out = fn(x, y)
            grad_out = torch.tensor([2.0, 2.0])
            out.backward(grad_out)
            self.assertEqual(x.grad, grad_out * y)

    def test_invoke_make_fx_forward_contrived(self):
        x = torch.tensor([0.5, 0.5], requires_grad=True)
        out = make_fx(_multiply_invoke)(x)
        self.assertEqual(out(x), torch.tensor([0.25, 0.25]))
        actual = normalize_gm(out.print_readable(False))
        self.assertExpectedInline(
            actual,
            """\
class _multiply_invoke(torch.nn.Module):
    def forward(self, grad_1: "f32[2]"):
        trace_wrapped: "f32[2]" = torch__dynamo__trace_wrapped_higher_order_op_self_invoke(grad_1);  grad_1 = None
        return trace_wrapped
""",
        )

    def test_invoke_make_bw(self):
        x = torch.tensor([0.5, 0.5], requires_grad=True)

        def fwd(x):
            z = x * x
            return z + z

        res = fwd(x)
        res.backward(torch.tensor([1.0, 1.0]))
        out = make_fx(_multiply_invoke)(x.grad)
        self.assertEqual(out(x.grad), torch.tensor([4.0, 4.0]))
        actual = normalize_gm(out.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class _multiply_invoke(torch.nn.Module):
    def forward(self, grad_1: "f32[2]"):
        trace_wrapped: "f32[2]" = torch__dynamo__trace_wrapped_higher_order_op_self_invoke(grad_1);  grad_1 = None
        return trace_wrapped
""",
        )

    @mock.patch(
        "torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count
    )
    def test_invoke_in_pt2_compiled_autograd(self, _):
        graph = None

        def compiler_fn(gm):
            def inner_compiler(gm_, example_inputs_):
                nonlocal graph
                self.assertEqual(graph, None)
                graph = gm_
                return inductor.compile(gm_, example_inputs_)

            return torch.compile(
                gm, backend=inner_compiler, fullgraph=True, dynamic=True
            )

        for backend in ["eager", "aot_eager", "inductor"]:
            torch._dynamo.reset()
            x = torch.tensor([0.5, 0.5], requires_grad=True)
            y = torch.tensor([0.5, 0.5], requires_grad=True)

            def fn(x, y):
                x.register_hook(_multiply_invoke)
                return x + y

            fn = torch.compile(fn, backend=backend)
            out = fn(x, y)
            grad_out = torch.tensor([2.0, 2.0])
            with compiled_autograd._enable(compiler_fn):
                out.backward(grad_out)
            actual = normalize_gm(graph.print_readable(False))
            self.assertEqual(x.grad, grad_out * grad_out)
            if backend == "aot_eager":
                self.assertExpectedInline(
                    actual,
                    """\
class GraphModule(torch.nn.Module):
    def forward(self, L_inputs_ : list, s69: "Sym(s21)", L_sizes_0_: "f32[0, s21]"):
        l_inputs_ = L_inputs_
        l_sizes_0_ = L_sizes_0_

        getitem: "f32[s21]" = l_inputs_[0]
        getitem_1: "f32[s21]" = l_inputs_[1]
        getitem_2: "f32[s21]" = l_inputs_[2];  l_inputs_ = None

        size: "Sym(s21)" = l_sizes_0_.size(1);  l_sizes_0_ = None

        validate_outputs = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem], [((None, None, device(type='cpu'), 6, 0, None), [size], False, 6)]);  getitem = size = None
        getitem_9: "f32[s21]" = validate_outputs[0];  validate_outputs = None

        call_aot_bwd_prologue = torch__dynamo_compiled_autograd_call_aot_bwd_prologue((), [], getitem_9);  getitem_9 = None
        aot1_tangents_1: "f32[s21]" = call_aot_bwd_prologue[0];  call_aot_bwd_prologue = None

        accumulate_grad = torch__dynamo_compiled_autograd_ops_AccumulateGrad([aot1_tangents_1], getitem_1, None, False);  getitem_1 = None
        getitem_11: "f32[s21]" = accumulate_grad[0];  accumulate_grad = None

        result: "f32[s21]" = aot1_tangents_1 * aot1_tangents_1;  aot1_tangents_1 = None

        accumulate_grad_1 = torch__dynamo_compiled_autograd_ops_AccumulateGrad([result], getitem_2, None, False);  result = getitem_2 = None
        getitem_12: "f32[s21]" = accumulate_grad_1[0];  accumulate_grad_1 = None
        return (getitem_11, getitem_12)
""",
                )
            elif backend == "inductor":
                self.assertExpectedInline(
                    actual,
                    """\
class GraphModule(torch.nn.Module):
    def forward(self, L_inputs_ : list, s69: "Sym(s21)", L_sizes_0_: "f32[0, s21]"):
        l_inputs_ = L_inputs_
        l_sizes_0_ = L_sizes_0_

        getitem: "f32[s21]" = l_inputs_[0]
        getitem_1: "f32[s21]" = l_inputs_[1]
        getitem_2: "f32[s21]" = l_inputs_[2];  l_inputs_ = None

        size: "Sym(s21)" = l_sizes_0_.size(1);  l_sizes_0_ = None

        validate_outputs = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem], [((None, None, device(type='cpu'), 6, 0, None), [size], False, 6)]);  getitem = size = None
        getitem_9: "f32[s21]" = validate_outputs[0];  validate_outputs = None

        call_aot_bwd_prologue = torch__dynamo_compiled_autograd_call_aot_bwd_prologue((), [], getitem_9);  getitem_9 = None
        aot3_tangents_1: "f32[s21]" = call_aot_bwd_prologue[0];  call_aot_bwd_prologue = None

        accumulate_grad = torch__dynamo_compiled_autograd_ops_AccumulateGrad([aot3_tangents_1], getitem_1, None, False);  getitem_1 = None
        getitem_11: "f32[s21]" = accumulate_grad[0];  accumulate_grad = None

        result: "f32[s21]" = aot3_tangents_1 * aot3_tangents_1;  aot3_tangents_1 = None

        accumulate_grad_1 = torch__dynamo_compiled_autograd_ops_AccumulateGrad([result], getitem_2, None, False);  result = getitem_2 = None
        getitem_12: "f32[s21]" = accumulate_grad_1[0];  accumulate_grad_1 = None
        return (getitem_11, getitem_12)
""",
                )

            graph = None

    @mock.patch(
        "torch._functorch.aot_autograd.AOT_COUNTER", new_callable=itertools.count
    )
    def test_invoke_in_pt2_compiled_autograd_side_effect(self, _):
        def _side_effect_stateful_fn2(x, obj):
            obj.counter = obj.counter + 1
            return _multiply(x)

        def _side_effectful_invoke2(grad, fn):
            return trace_wrapped(grad, fn=fn)

        graph = None

        def compiler_fn(gm):
            def inner_compiler(gm_, example_inputs_):
                nonlocal graph
                self.assertEqual(graph, None)
                graph = gm_
                return inductor.compile(gm_, example_inputs_)

            return torch.compile(
                gm, backend=inner_compiler, fullgraph=True, dynamic=True
            )

        for backend in ["inductor"]:
            torch._dynamo.reset()
            x = torch.tensor([0.5, 0.5], requires_grad=True)
            y = torch.tensor([0.5, 0.5], requires_grad=True)

            class MyObj:
                def __init__(self) -> None:
                    self.counter = 0

            obj = MyObj()
            inner_fn = functools.partial(_side_effect_stateful_fn2, obj=obj)
            hook_fn = functools.partial(_side_effectful_invoke2, fn=inner_fn)
            x.register_hook(hook_fn)

            def fn(x, y):
                return x + y

            fn = torch.compile(fn, backend=backend, fullgraph=True)
            out = fn(x, y)
            grad_out = torch.tensor([2.0, 2.0])
            with compiled_autograd._enable(compiler_fn):
                out.backward(grad_out)
            actual = normalize_gm(graph.print_readable(False))
            self.assertEqual(obj.counter, 1)
            self.assertEqual(x.grad, grad_out + grad_out)
            if backend in ["aot_eager", "inductor"]:
                self.assertExpectedInline(
                    actual,
                    """\
class GraphModule(torch.nn.Module):
    def forward(self, L_inputs_ : list, s69: "Sym(s21)", L_sizes_0_: "f32[0, s21]", L_hooks_1_keywords_fn_keywords_obj_counter: "Sym(s45)"):
        l_inputs_ = L_inputs_
        l_sizes_0_ = L_sizes_0_
        l_hooks_1_keywords_fn_keywords_obj_counter = L_hooks_1_keywords_fn_keywords_obj_counter

        getitem: "f32[s21]" = l_inputs_[0]
        getitem_1: "f32[s21]" = l_inputs_[1]
        getitem_2: "f32[s21]" = l_inputs_[2];  l_inputs_ = None

        size: "Sym(s21)" = l_sizes_0_.size(1);  l_sizes_0_ = None

        validate_outputs = torch__dynamo_compiled_autograd_ops_validate_outputs([getitem], [((None, None, device(type='cpu'), 6, 0, None), [size], False, 6)]);  getitem = size = None
        getitem_9: "f32[s21]" = validate_outputs[0];  validate_outputs = None

        call_aot_bwd_prologue = torch__dynamo_compiled_autograd_call_aot_bwd_prologue((), [], getitem_9);  getitem_9 = None
        aot0_tangents_1: "f32[s21]" = call_aot_bwd_prologue[0];  call_aot_bwd_prologue = None

        accumulate_grad = torch__dynamo_compiled_autograd_ops_AccumulateGrad([aot0_tangents_1], getitem_1, None, False);  getitem_1 = None
        getitem_11: "f32[s21]" = accumulate_grad[0];  accumulate_grad = None

        add: "Sym(s45 + 1)" = l_hooks_1_keywords_fn_keywords_obj_counter + 1;  l_hooks_1_keywords_fn_keywords_obj_counter = None

        result: "f32[s21]" = aot0_tangents_1 * aot0_tangents_1;  aot0_tangents_1 = None

        accumulate_grad_1 = torch__dynamo_compiled_autograd_ops_AccumulateGrad([result], getitem_2, None, False);  result = getitem_2 = None
        getitem_12: "f32[s21]" = accumulate_grad_1[0];  accumulate_grad_1 = None
        return (getitem_11, getitem_12, add)
""",
                )

            out = fn(x, y)
            out.backward(grad_out)
            self.assertEqual(obj.counter, 2)

            out = fn(x, y)
            out.backward(grad_out)
            self.assertEqual(obj.counter, 3)
            graph = None

    def test_invoke_in_pt2_compiled_autograd_graph_breaks(self):
        def _graph_breaking_fn(x):
            print("Boo!")
            return _multiply(x)

        def _graph_break_invoke(grad):
            return trace_wrapped(grad, fn=_graph_breaking_fn)

        def compiler_fn(gm):
            return torch.compile(gm, backend="inductor", fullgraph=True, dynamic=True)

        for backend in ["eager", "aot_eager", "inductor"]:
            torch._dynamo.reset()
            x = torch.tensor([0.5, 0.5], requires_grad=True)
            y = torch.tensor([0.5, 0.5], requires_grad=True)

            def fn(x, y):
                x.register_hook(_graph_break_invoke)
                return x + y

            fn = torch.compile(fn, backend=backend, fullgraph=True)
            out = fn(x, y)
            grad_out = torch.tensor([2.0, 2.0])
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported,
                "print",
            ):
                with compiled_autograd._enable(compiler_fn):
                    out.backward(grad_out)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

```



## High-Level Overview


This Python file contains 7 class(es) and 30 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BackwardHigherOrderOpTests`, `_multiply_invoke`, `_multiply_invoke`, `GraphModule`, `GraphModule`, `MyObj`, `GraphModule`

**Functions defined**: `_multiply`, `_multiply_invoke`, `test_invoke_in_eager`, `fn`, `test_invoke_in_pt2`, `fn`, `test_invoke_make_fx_forward_contrived`, `forward`, `test_invoke_make_bw`, `fwd`, `forward`, `test_invoke_in_pt2_compiled_autograd`, `compiler_fn`, `inner_compiler`, `fn`, `forward`, `forward`, `test_invoke_in_pt2_compiled_autograd_side_effect`, `_side_effect_stateful_fn2`, `_side_effectful_invoke2`

**Key imports**: functools, itertools, mock, torch, torch._dynamo.test_case, torch._dynamo.testing, torch._dynamo.utils, _inductor as inductor, compiled_autograd, trace_wrapped


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `functools`
- `itertools`
- `unittest`: mock
- `torch`
- `torch._dynamo.test_case`
- `torch._dynamo.testing`
- `torch._dynamo.utils`
- `torch._dynamo`: compiled_autograd
- `torch._dynamo._trace_wrapped_higher_order_op`: trace_wrapped
- `torch.fx.experimental.proxy_tensor`: make_fx


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/dynamo/test_backward_higher_order_ops.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/dynamo`):

- [`test_guard_serialization.py_docs.md`](./test_guard_serialization.py_docs.md)
- [`test_subgraphs.py_docs.md`](./test_subgraphs.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_unspec.py_docs.md`](./test_unspec.py_docs.md)
- [`test_trace_rules.py_docs.md`](./test_trace_rules.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_pre_dispatch.py_docs.md`](./test_pre_dispatch.py_docs.md)
- [`test_autograd_function.py_docs.md`](./test_autograd_function.py_docs.md)
- [`test_optimizers.py_docs.md`](./test_optimizers.py_docs.md)
- [`test_callback.py_docs.md`](./test_callback.py_docs.md)


## Cross-References

- **File Documentation**: `test_backward_higher_order_ops.py_docs.md`
- **Keyword Index**: `test_backward_higher_order_ops.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
