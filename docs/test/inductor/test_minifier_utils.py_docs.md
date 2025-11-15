# Documentation: `test/inductor/test_minifier_utils.py`

## File Metadata

- **Path**: `test/inductor/test_minifier_utils.py`
- **Size**: 3,247 bytes (3.17 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
import torch
from torch._dynamo.repro.aoti import (
    AOTIMinifierError,
    export_for_aoti_minifier,
    get_module_string,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class MinifierUtilsTests(TestCase):
    def test_invalid_output(self):
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                # return a graph module
                return self.linear

        model = SimpleModel()
        # Here we obtained a graph with invalid output by symbolic_trace for simplicity,
        # it can also obtained from running functorch.compile.minifier on an exported graph.
        traced = torch.fx.symbolic_trace(model)
        for strict in [True, False]:
            gm = export_for_aoti_minifier(traced, (torch.randn(2, 2),), strict=strict)
            self.assertTrue(gm is None)

    def test_non_exportable(self):
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return x.sum()

        model = SimpleModel()
        # Force export failure by providing an input with in-compatible shapes
        inputs = (torch.randn(2), torch.randn(2))
        for strict in [True, False]:
            gm = export_for_aoti_minifier(
                model, inputs, strict=strict, skip_export_error=True
            )
            print(gm)
            self.assertTrue(gm is None)

            with self.assertRaises(AOTIMinifierError):
                export_for_aoti_minifier(
                    model, inputs, strict=strict, skip_export_error=False
                )

    def test_convert_module_to_string(self):
        class M(torch.nn.Module):
            def forward(self, x, flag):
                flag = flag.item()

                def true_fn(x):
                    return x.clone()

                return torch.cond(flag > 0, true_fn, true_fn, [x])

        inputs = (
            torch.rand(28, 28),
            torch.tensor(1),
        )

        model = M()
        gm = torch.export.export(model, inputs, strict=False).module(check_guards=False)

        # TODO: make NNModuleToString.convert() generate string for nested submodules.
        model_string = get_module_string(gm)
        self.assertExpectedInline(
            model_string.strip(),
            """\
# from torch.nn import *
# class Repro(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.true_graph_0 = <lambda>()
#         self.false_graph_0 = <lambda>()



#     def forward(self, x, flag):
#         x, flag, = fx_pytree.tree_flatten_spec(([x, flag], {}), self._in_spec)
#         item = torch.ops.aten.item.default(flag);  flag = None
#         gt = item > 0;  item = None
#         true_graph_0 = self.true_graph_0
#         false_graph_0 = self.false_graph_0
#         cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (x,));  gt = true_graph_0 = false_graph_0 = x = None
#         getitem = cond[0];  cond = None
#         return pytree.tree_unflatten((getitem,), self._out_spec)""",
        )


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 5 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MinifierUtilsTests`, `SimpleModel`, `SimpleModel`, `M`, `Repro`

**Functions defined**: `test_invalid_output`, `__init__`, `forward`, `test_non_exportable`, `forward`, `test_convert_module_to_string`, `forward`, `true_fn`, `__init__`, `forward`

**Key imports**: torch, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
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
python test/inductor/test_minifier_utils.py
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

- **File Documentation**: `test_minifier_utils.py_docs.md`
- **Keyword Index**: `test_minifier_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
