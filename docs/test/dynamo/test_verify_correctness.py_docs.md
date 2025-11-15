# Documentation: `test/dynamo/test_verify_correctness.py`

## File Metadata

- **Path**: `test/dynamo/test_verify_correctness.py`
- **Size**: 4,160 bytes (4.06 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]
import operator

import torch
import torch._dynamo
import torch._dynamo.config as config
import torch._dynamo.test_case
from torch._dynamo.testing import same
from torch.fx._lazy_graph_module import _force_skip_lazy_graph_module


class Seq(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


class Conv_Bn_Relu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b


def transform(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        # Checks if we're calling a function (i.e:
        # operator.add)
        if node.op == "call_function":
            # The target attribute is the function
            # that call_function calls.
            if node.target == operator.mul:
                node.target = operator.add

    gm.graph.lint()  # Does some checks to make sure the
    # Graph is well-formed.

    gm.recompile()
    return gm


@config.patch("verify_correctness", True)
class TestVerifyCorrectness(torch._dynamo.test_case.TestCase):
    def test_example_inputs(self):
        def fn(a, bc, d):
            b, c = bc
            return a / d - b / c

        def compiler_fn(graph, example_inputs):
            nonlocal r1
            r1 = graph(*example_inputs)[0]
            return graph.forward

        a = torch.empty(2).fill_(1)
        b = torch.empty(2).fill_(2)
        c = torch.empty(2).fill_(3)
        d = 4
        r1 = None
        r2 = fn(a, (b, c), d)
        opt_fn = torch._dynamo.optimize_assert(compiler_fn)(fn)
        r3 = opt_fn(a, (b, c), d)

        self.assertIsNotNone(r1)

        self.assertEqual(r1.shape, r2.shape)
        self.assertEqual(r1.shape, r3.shape)
        self.assertEqual(r1.device, r2.device)
        self.assertEqual(r1.device, r3.device)

    @_force_skip_lazy_graph_module()
    def test_torchscript(self):
        s = Seq()
        i = torch.randn(10)
        r1 = s(i)
        opt_s = torch.compile(s, backend="ts")
        r2 = opt_s(i)
        self.assertTrue(same(r1, r2))

    def test_incorrect_verify_true(self):
        """
        If a bad optimization return a graph that
        is not functionally equal to the original graph;
        When config.verify_correctness=True, it will
        check the correctness of outputs and raise an error
        """
        i1 = torch.randn(10)
        i2 = torch.randn(10)

        def incorrect_compile_fn(gm, example_inputs):
            return transform(gm).forward

        toy_example(i1, i2)
        try:
            opt_toy_example = torch.compile(toy_example, backend=incorrect_compile_fn)
            opt_toy_example(i1, i2)
        except RuntimeError:
            pass
        else:
            self.fail("expected failure")

    @config.patch("verify_correctness", False)
    def test_incorrect_verify_false(self):
        """
        The bad optimization return a graph that
        is not functionally equal to the original graph;
        When config.verify_correctness=False, wrong outputs
        will return
        """
        i1 = torch.randn(10)
        i2 = torch.randn(10)

        def incorrect_compile_fn(gm, example_inputs):
            return transform(gm).forward

        r1 = toy_example(i1, i2)
        opt_toy_example = torch.compile(toy_example, backend=incorrect_compile_fn)
        r2 = opt_toy_example(i1, i2)
        self.assertTrue(not same(r1, r2))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

```



## High-Level Overview


This Python file contains 3 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Seq`, `Conv_Bn_Relu`, `TestVerifyCorrectness`

**Functions defined**: `__init__`, `forward`, `__init__`, `forward`, `toy_example`, `transform`, `test_example_inputs`, `fn`, `compiler_fn`, `test_torchscript`, `test_incorrect_verify_true`, `incorrect_compile_fn`, `test_incorrect_verify_false`, `incorrect_compile_fn`

**Key imports**: operator, torch, torch._dynamo, torch._dynamo.config as config, torch._dynamo.test_case, same, _force_skip_lazy_graph_module, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `operator`
- `torch`
- `torch._dynamo`
- `torch._dynamo.config as config`
- `torch._dynamo.test_case`
- `torch._dynamo.testing`: same
- `torch.fx._lazy_graph_module`: _force_skip_lazy_graph_module


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
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
python test/dynamo/test_verify_correctness.py
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

- **File Documentation**: `test_verify_correctness.py_docs.md`
- **Keyword Index**: `test_verify_correctness.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
