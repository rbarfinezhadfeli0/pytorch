# Documentation: `test/lazy/test_functionalization.py`

## File Metadata

- **Path**: `test/lazy/test_functionalization.py`
- **Size**: 2,936 bytes (2.87 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import re

import torch
import torch._lazy.metrics as metrics
import torch._lazy.ts_backend
from torch.testing._internal.common_utils import run_tests, TestCase


torch._lazy.ts_backend.init()

NODE_TYPE_PATTERN = re.compile(r", NodeType=[^\n]+")


class LazyFuncionalizationTest(TestCase):
    def test_lazy_init_with_view(self):
        def f(device, reset_storage=False):
            torch.manual_seed(2023)

            if device == "lazy":
                metrics.reset()

            class Model(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.fc1 = torch.nn.Linear(4, 2, bias=False)

                def forward(self, x):
                    return x @ self.fc1.weight.transpose(0, 1)

            with torch.device(device):
                model = Model()

                if device == "lazy":
                    if reset_storage:
                        torch._C._unsafe_reset_storage(model.fc1.weight)

                    torch._lazy.mark_step()

                    sync_tensors = metrics.counter_value("SyncedTensorsWithIR")
                    if reset_storage:
                        assert sync_tensors == 1
                    else:
                        # There is an extra tensor being unnecessarily synced if
                        # the functional storage is not reset.
                        assert sync_tensors == 2

                x = torch.ones(4)
                out = model(x)

                if device == "lazy":
                    torch._lazy.mark_step()

                return out

        cpu_out = f("cpu")
        lazy_out_1 = f("lazy", reset_storage=False)
        lazy_out_2 = f("lazy", reset_storage=True)

        self.assertEqual(cpu_out, lazy_out_1.to("cpu"))
        self.assertEqual(cpu_out, lazy_out_2.to("cpu"))

    def test_data_assign(self):
        def text(lazyt):
            raw = torch._C._lazy._get_tensors_text([lazyt])
            return NODE_TYPE_PATTERN.sub("", raw)

        origin = torch.rand(3, dtype=torch.float32)
        tensor = origin.to("lazy")

        self.assertExpectedInline(
            text(tensor),
            """\
IR {
  %0 = [Float[3]] lazy_tensors::device_data(), device=CPU0, ROOT=0
}
""",
        )

        # Modify the data-type of tensor, and assign it to 'data'.
        # This should update the inner tensor of FunctionalTensorWrapper,
        # changing the corresponding IR node.
        modified_tensor = tensor.to(torch.bfloat16)
        tensor.data = modified_tensor

        self.assertExpectedInline(
            text(tensor),
            """\
IR {
  %0 = [Float[3]] lazy_tensors::device_data(), device=CPU0
  %1 = [BFloat16[3]] aten::_to_copy(%0), dtype=BFloat16, layout=null, device=null, pin_memory=null, non_blocking=0, memory_format=null, ROOT=0
}
""",  # noqa: B950
        )


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LazyFuncionalizationTest`, `Model`

**Functions defined**: `test_lazy_init_with_view`, `f`, `__init__`, `forward`, `test_data_assign`, `text`

**Key imports**: re, torch, torch._lazy.metrics as metrics, torch._lazy.ts_backend, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/lazy`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `re`
- `torch`
- `torch._lazy.metrics as metrics`
- `torch._lazy.ts_backend`
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
python test/lazy/test_functionalization.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/lazy`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_ts_opinfo.py_docs.md`](./test_ts_opinfo.py_docs.md)
- [`test_meta_kernel.py_docs.md`](./test_meta_kernel.py_docs.md)
- [`test_generator.py_docs.md`](./test_generator.py_docs.md)
- [`test_bindings.py_docs.md`](./test_bindings.py_docs.md)
- [`test_extract_compiled_graph.py_docs.md`](./test_extract_compiled_graph.py_docs.md)
- [`test_reuse_ir.py_docs.md`](./test_reuse_ir.py_docs.md)
- [`test_step_closures.py_docs.md`](./test_step_closures.py_docs.md)
- [`test_debug_util.py_docs.md`](./test_debug_util.py_docs.md)


## Cross-References

- **File Documentation**: `test_functionalization.py_docs.md`
- **Keyword Index**: `test_functionalization.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
