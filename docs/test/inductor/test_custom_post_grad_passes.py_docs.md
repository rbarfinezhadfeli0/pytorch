# Documentation: `test/inductor/test_custom_post_grad_passes.py`

## File Metadata

- **Path**: `test/inductor/test_custom_post_grad_passes.py`
- **Size**: 10,726 bytes (10.47 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
import contextlib
import operator
from collections import defaultdict

import torch
import torch._inductor.pattern_matcher as pattern_matcher
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codegen.common import get_custom_backend_pass_for_device
from torch._inductor.custom_graph_pass import (
    CustomGraphModulePass,
    CustomGraphPass,
    get_hash_for_files,
)
from torch._inductor.lowering import lowerings as L
from torch._inductor.pattern_matcher import Arg, CallFunction, PatternMatcherPass
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CPU, patch_inductor_backend


@config.patch({"freezing": True})
class TestCustomPassBase(TestCase):
    def _clone_inputs(self, inputs):
        def clone(x):
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone()

        return tuple(clone(x) for x in inputs)

    def _test_common(
        self,
        mod,
        inputs,
        matcher_count,
        matcher_nodes,
        atol=1e-5,
        rtol=1.3e-6,
    ):
        counters.clear()
        maybe_autocast = contextlib.nullcontext()
        with torch.no_grad(), maybe_autocast:
            clone_inputs = self._clone_inputs(inputs)
            expected = mod(*inputs)
            actual = torch.compile(mod)(*clone_inputs)
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            self.assertEqual(
                counters["inductor"]["pattern_matcher_count"], matcher_count
            )
            self.assertEqual(
                counters["inductor"]["pattern_matcher_nodes"],
                matcher_nodes,
            )


aten = torch.ops.aten
mkldnn = torch.ops.mkldnn


def change_cos_pass(graph):
    for node in graph.nodes:
        if node.op == "call_function" and node.target == aten.cos.default:
            node.target = aten.sin.default


class ChangeCosCustomPass(CustomGraphPass):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, g: torch.fx.graph.Graph):
        change_cos_pass(g)

    def uuid(self) -> bytes:
        return get_hash_for_files((__file__,))


class TestPostGradCustomPrePostPass(TestCustomPassBase):
    #  mkldnn fusion's pattern_matcher
    # (torch/_inductor/fx_passes/mkldnn_fusion.py),
    # and apply it to custom post_grad_passes.
    def _register_mkldnn_conv_relu_fusion(self, custom_pass_dict):
        # pattern
        def _mkldnn_conv_relu_pattern():
            return CallFunction(
                aten.relu,
                CallFunction(
                    mkldnn._convolution_pointwise.default,
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    Arg(),
                    _users=1,
                ),
            )

        # utils of pattern matcher registration
        def _register_fusion_lowering(pattern, custom_pass_dict):
            def dummy_check(m):
                return True

            def register_custom_lowering_pattern(
                pattern, extra_check, custom_pass_dict
            ):
                return pattern_matcher.register_lowering_pattern(
                    pattern, extra_check, pass_dict=custom_pass_dict
                )

            @register_custom_lowering_pattern(pattern, dummy_check, custom_pass_dict)
            def fn(match, *args, **kwargs):
                computation_args = list(args)[:-3] + ["relu", [], ""]
                return L[mkldnn._convolution_pointwise.default](*computation_args)

            return fn

        _register_fusion_lowering(_mkldnn_conv_relu_pattern(), custom_pass_dict)

    # custom post grad pass
    class _CustomPass(PatternMatcherPass, CustomGraphPass):
        def __init__(self) -> None:
            super().__init__()

        def __call__(self, g: torch.fx.graph.Graph):
            self.apply(g)

        def uuid(self) -> bytes:
            return get_hash_for_files((__file__,))

    # case model
    class _ConvReLU(torch.nn.Module):
        def __init__(self, ic, oc):
            super().__init__()
            self.conv = torch.nn.Conv2d(ic, oc, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            x1 = self.conv(x)
            return x1.relu()

    def test_custom_joint_pass_pre(self):
        with config.patch(joint_custom_pre_pass=ChangeCosCustomPass()):

            def g(x):
                return x.sin().sin().sin()

            def f(x):
                return x.cos().cos().cos()

            x = torch.randn(8, dtype=torch.float32)
            torch.testing.assert_close(torch.compile(f)(x), g(x))

    def test_custom_joint_pass_post(self):
        with config.patch(joint_custom_post_pass=ChangeCosCustomPass()):

            def g(x):
                return x.sin().sin().sin()

            def f(x):
                return x.cos().cos().cos()

            x = torch.randn(8, dtype=torch.float32)
            torch.testing.assert_close(torch.compile(f)(x), g(x))

    def test_custom_pre_pass(self):
        with config.patch(
            # leave custom pass only in post_grad_passes()
            pattern_matcher=False,
            post_grad_custom_pre_pass=self._CustomPass(),
            # define pattern match as custom post grad opt pass
            post_grad_custom_post_pass=None,
        ):
            # init mkldnn fusion on custom_matcher
            self._register_mkldnn_conv_relu_fusion(config.post_grad_custom_pre_pass)

            mod = self._ConvReLU(16, 16).eval()
            x = torch.randn((1, 16, 56, 56), dtype=torch.float32)

            match_count = 1
            match_nodes = 2
            other_match_count = 1  # conv prepack weight
            other_match_nodes = 1  # conv prepack weight
            self._test_common(
                mod,
                (x,),
                match_count + other_match_count,
                match_nodes + other_match_nodes,
            )

    def test_custom_post_pass(self):
        with config.patch(
            # leave custom pass only in post_grad_passes()
            pattern_matcher=False,
            # define pattern match as custom post grad opt pass
            post_grad_custom_pre_pass=None,
            post_grad_custom_post_pass=self._CustomPass(),
        ):
            # init mkldnn fusion on custom_matcher
            self._register_mkldnn_conv_relu_fusion(config.post_grad_custom_post_pass)

            mod = self._ConvReLU(16, 16).eval()
            x = torch.randn((1, 16, 56, 56), dtype=torch.float32)

            match_count = 1
            match_nodes = 2
            other_match_count = 1  # conv prepack weight
            other_match_nodes = 1  # conv prepack weight
            self._test_common(
                mod,
                (x,),
                match_count + other_match_count,
                match_nodes + other_match_nodes,
            )

    def test_custom_pre_grad_pass(self):
        saved_graph = [None]

        def merge_mm_shared_rhs(graph: fx.Graph):
            """
            Bad POC of merging mm with a shared RHS.
            i.e. [mm(x, W), mm(x2, W)] => mm(cat(x, x2), W).split()

            Isn't actually safe for a couple reasons. For example, it doesn't handle the
            case where the LHS inputs depend on each other
            """
            saved_graph[0] = graph
            matmuls = [n for n in graph.nodes if n.target == torch.mm]
            rhs_vals = defaultdict(set)
            for m in matmuls:
                rhs_vals[m.args[1]].add(m)

            order = {n: idx for idx, n in enumerate(graph.nodes)}

            for rhs, matmuls in rhs_vals.items():
                if len(matmuls) == 1:
                    continue
                matmuls = sorted(matmuls, key=lambda x: order[x])
                with graph.inserting_before(matmuls[0]):
                    lhs_vals = [m.args[0] for m in matmuls]
                    new_cat = graph.create_node(
                        "call_function", torch.cat, args=(lhs_vals, 0)
                    )
                    new_mm = graph.create_node(
                        "call_function", torch.mm, args=(new_cat, rhs)
                    )
                    split_vals = graph.create_node(
                        "call_function",
                        torch.split,
                        args=(
                            new_mm,
                            [l.meta["example_value"].shape[0] for l in lhs_vals],
                        ),
                    )
                for idx, m in enumerate(matmuls):
                    m.target = operator.getitem
                    m.args = (split_vals, idx)

        @config.patch(pre_grad_custom_pass=merge_mm_shared_rhs)
        def inner_test():
            @torch.compile
            def f(W, nested_seqs):
                outs = [torch.mm(s, W) for s in nested_seqs]
                return outs

            W = torch.randn(16, 16, dtype=torch.bfloat16)
            nested_seqs = [
                torch.randn(l, 16, dtype=torch.bfloat16) for l in [4, 8, 5, 3]
            ]

            f(W, nested_seqs)
            assert saved_graph[0] is not None
            matmuls = [n for n in saved_graph[0].nodes if n.target == torch.mm]
            assert len(matmuls) == 1

        inner_test()

    def test_custom_backend_pass(self):
        class CustomBackendPass(CustomGraphModulePass):
            def __init__(self, existing_pass: CustomGraphModulePass = None):
                super().__init__()
                self.existing_pass = existing_pass

            def __call__(self, gm: fx.GraphModule) -> None:
                if self.existing_pass:
                    self.existing_pass(gm)

                change_cos_pass(gm.graph)

            def uuid(self) -> bytes:
                return get_hash_for_files((__file__,))

        custom_backend_pass = CustomBackendPass(
            get_custom_backend_pass_for_device("cpu")
        )
        with patch_inductor_backend("cpu", custom_pass=custom_backend_pass):

            def g(x):
                return x.sin().sin().sin()

            def f(x):
                return x.cos().cos().cos()

            x = torch.randn(8, dtype=torch.float32)
            torch.testing.assert_close(torch.compile(f)(x), g(x))


if __name__ == "__main__":
    if IS_LINUX and HAS_CPU and torch.backends.mkldnn.is_available():
        run_tests()

```



## High-Level Overview


This Python file contains 6 class(es) and 36 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestCustomPassBase`, `ChangeCosCustomPass`, `TestPostGradCustomPrePostPass`, `_CustomPass`, `_ConvReLU`, `CustomBackendPass`

**Functions defined**: `_clone_inputs`, `clone`, `_test_common`, `change_cos_pass`, `__init__`, `__call__`, `uuid`, `_register_mkldnn_conv_relu_fusion`, `_mkldnn_conv_relu_pattern`, `_register_fusion_lowering`, `dummy_check`, `register_custom_lowering_pattern`, `fn`, `__init__`, `__call__`, `uuid`, `__init__`, `forward`, `test_custom_joint_pass_pre`, `g`

**Key imports**: contextlib, operator, defaultdict, torch, torch._inductor.pattern_matcher as pattern_matcher, torch.fx as fx, counters, config, get_custom_backend_pass_for_device, lowerings as L


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `operator`
- `collections`: defaultdict
- `torch`
- `torch._inductor.pattern_matcher as pattern_matcher`
- `torch.fx as fx`
- `torch._dynamo.utils`: counters
- `torch._inductor`: config
- `torch._inductor.codegen.common`: get_custom_backend_pass_for_device
- `torch._inductor.lowering`: lowerings as L
- `torch._inductor.pattern_matcher`: Arg, CallFunction, PatternMatcherPass
- `torch._inductor.test_case`: run_tests, TestCase
- `torch.testing._internal.common_utils`: IS_LINUX
- `torch.testing._internal.inductor_utils`: HAS_CPU, patch_inductor_backend


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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/inductor/test_custom_post_grad_passes.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/inductor`):

- [`test_benchmark_fusion.py_docs.md`](./test_benchmark_fusion.py_docs.md)
- [`test_op_dtype_prop.py_docs.md`](./test_op_dtype_prop.py_docs.md)
- [`test_custom_op_autotune.py_docs.md`](./test_custom_op_autotune.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_inductor_freezing.py_docs.md`](./test_inductor_freezing.py_docs.md)
- [`test_b2b_gemm.py_docs.md`](./test_b2b_gemm.py_docs.md)
- [`test_minifier_isolate.py_docs.md`](./test_minifier_isolate.py_docs.md)
- [`test_move_constructors_to_cuda.py_docs.md`](./test_move_constructors_to_cuda.py_docs.md)
- [`test_cutlass_backend.py_docs.md`](./test_cutlass_backend.py_docs.md)
- [`test_cache.py_docs.md`](./test_cache.py_docs.md)


## Cross-References

- **File Documentation**: `test_custom_post_grad_passes.py_docs.md`
- **Keyword Index**: `test_custom_post_grad_passes.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
