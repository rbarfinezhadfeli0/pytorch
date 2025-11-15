# Documentation: `test/functorch/test_minifier.py`

## File Metadata

- **Path**: `test/functorch/test_minifier.py`
- **Size**: 3,563 bytes (3.48 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: functorch"]

import torch
from functorch import make_fx
from functorch.compile import minifier
from torch._functorch.compile_utils import get_outputs, get_placeholders
from torch.testing._internal.common_utils import run_tests, TestCase


class TestMinifier(TestCase):
    def test_has_mul_minifier(self):
        def failing_f(x, y):
            y = y / 3
            x = x + 3
            x = x * y
            return x + y

        inps = [torch.randn(3), torch.randn(3)]
        failing_f = make_fx(failing_f)(*inps)

        def has_mul(fx_g, inps):
            return torch.ops.aten.mul.Tensor in (i.target for i in fx_g.graph.nodes)

        min_f, inps = minifier(failing_f, inps, has_mul)
        self.assertEqual(len(min_f.graph.nodes), 4)
        self.assertEqual(len(inps), 2)

    def test_has_add_mul(self):
        def failing_f(x):
            x = x * 3
            x = x + 5
            x = x.cos()
            zero = x - x
            result = zero / zero
            result = result + 3
            return (result * 2,)

        inps = [torch.randn(3)]
        failing_f = make_fx(failing_f)(*inps)

        def has_nans(fx_g, inps):
            # Basically, make sure none of the nodes are computing nans
            for i in inps:
                if torch.isnan(i).any():
                    return False
            return torch.isnan(fx_g(*inps)[0]).any()

        min_f, inps = minifier(failing_f, inps, has_nans)
        self.assertEqual(len(min_f.graph.nodes), 3)
        self.assertEqual(len(inps), 1)

    def test_input_returned(self):
        def f(a, b, c):
            a = a.sin()
            c = c.cos()
            d = a * c
            return (a, b, c, d)

        inps = [torch.randn(3) for _ in range(3)]

        def inputs_returned(fx_g, inps):
            inps = set(get_placeholders(fx_g.graph))
            outs = set(get_outputs(fx_g.graph))
            return len(inps & outs) > 0

        failing_f = make_fx(f)(*inps)
        min_f, inps = minifier(failing_f, inps, inputs_returned)
        self.assertEqual(len(min_f.graph.nodes), 2)
        self.assertEqual(len(inps), 1)

    def test_tup_use(self):
        def f(a, b):
            tup = torch.std_mean(a)
            return (tup[0] + b * tup[1],)

        inps = [torch.randn(3), torch.randn(3)]

        def has_add(fx_g, inps):
            return torch.ops.aten.add.Tensor in (i.target for i in fx_g.graph.nodes)

        failing_f = make_fx(f)(*inps)
        min_f, inps = minifier(failing_f, inps, has_add)

        self.assertEqual(len(min_f.graph.nodes), 4)
        self.assertEqual(len(inps), 2)

    def test_module(self):
        class MockModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                y = self.relu(x)
                zero = y - y
                result = zero / zero
                result = result + 3
                return result

        mod = MockModule()
        failing_f = torch.fx.symbolic_trace(mod)

        inps = [torch.randn(3)]

        def pass_checker(fx_g, inps):
            # Basically, make sure none of the inputs are nans
            for i in inps:
                if torch.isnan(i).any():
                    return False
            return torch.isnan(fx_g(*inps)[0]).any()

        min_f, inps = minifier(failing_f, inps, pass_checker)
        assert len(min_f.graph.nodes) == 3
        assert len(inps) == 1


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 16 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestMinifier`, `MockModule`

**Functions defined**: `test_has_mul_minifier`, `failing_f`, `has_mul`, `test_has_add_mul`, `failing_f`, `has_nans`, `test_input_returned`, `f`, `inputs_returned`, `test_tup_use`, `f`, `has_add`, `test_module`, `__init__`, `forward`, `pass_checker`

**Key imports**: torch, make_fx, minifier, get_outputs, get_placeholders, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/functorch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `functorch`: make_fx
- `functorch.compile`: minifier
- `torch._functorch.compile_utils`: get_outputs, get_placeholders
- `torch.testing._internal.common_utils`: run_tests, TestCase


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
python test/functorch/test_minifier.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/functorch`):

- [`test_vmap.py_docs.md`](./test_vmap.py_docs.md)
- [`test_rearrange.py_docs.md`](./test_rearrange.py_docs.md)
- [`test_aot_joint_with_descriptors.py_docs.md`](./test_aot_joint_with_descriptors.py_docs.md)
- [`functorch_additional_op_db.py_docs.md`](./functorch_additional_op_db.py_docs.md)
- [`xfail_suggester.py_docs.md`](./xfail_suggester.py_docs.md)
- [`discover_coverage.py_docs.md`](./discover_coverage.py_docs.md)
- [`test_eager_transforms.py_docs.md`](./test_eager_transforms.py_docs.md)
- [`test_ac.py_docs.md`](./test_ac.py_docs.md)
- [`common_utils.py_docs.md`](./common_utils.py_docs.md)
- [`test_logging.py_docs.md`](./test_logging.py_docs.md)


## Cross-References

- **File Documentation**: `test_minifier.py_docs.md`
- **Keyword Index**: `test_minifier.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
