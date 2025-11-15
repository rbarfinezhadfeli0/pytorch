# Documentation: `test/distributed/fsdp/test_fsdp_fx.py`

## File Metadata

- **Path**: `test/distributed/fsdp/test_fsdp_fx.py`
- **Size**: 4,720 bytes (4.61 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]
import torch
from torch.distributed.fsdp._trace_utils import _ExecOrderTracer
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight1 = torch.nn.Parameter(torch.randn(6, 6))
        self.weight2 = torch.nn.Parameter(torch.randn(6, 6))
        self.weight_unused = torch.nn.Parameter(torch.randn(2, 2))
        self.layer0 = torch.nn.Linear(6, 6)
        self.layer1 = torch.nn.Linear(6, 6, bias=False)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(6, 3, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 6, bias=False),
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor, run_all_layers: bool) -> torch.Tensor:
        z = self.relu(self.layer0(x))
        z = self.relu(self.layer2(z))
        z = z @ self.weight1
        if run_all_layers:
            z = self.relu(self.layer1(z))
            z = z @ self.weight2
            # Use `layer0` twice to check the handling of multiplicity in the
            # saved data structures
            z = self.relu(self.layer0(x))
        return z


class TestSymbolicTracing(TestCase):
    def test_symbolic_tracing_outputs(self):
        """
        Tests running ``tracer.trace()`` inside ``patch_tracer()`` by checking
        the saved data structures.
        """
        model = Model()
        tracer = torch.fx.Tracer()
        orig_call_module = tracer.call_module
        orig_create_proxy = tracer.create_proxy
        exec_order_tracer = _ExecOrderTracer()
        with exec_order_tracer.patch_tracer(tracer=tracer, root_module=model):
            concrete_args = {"run_all_layers": True}
            tracer.trace(model, concrete_args)
        # Check that the tracer methods are unchanged after exiting the context
        self.assertEqual(orig_call_module, tracer.call_module)
        self.assertEqual(orig_create_proxy, tracer.create_proxy)
        # Check `module_forward_order`
        correct_module_forward_order = [
            model,
            model.layer0,
            model.relu,
            model.layer2,
            model.layer2[0],
            model.layer2[1],
            model.layer2[2],
            model.relu,
            model.layer1,
            model.relu,
            model.layer0,
            model.relu,
        ]
        exec_info = exec_order_tracer.exec_info
        self.assertEqual(exec_info.module_forward_order, correct_module_forward_order)
        # Check `module_to_param_usage_infos`
        self.assertEqual(
            exec_info.module_to_param_usage_infos[model],
            [
                (model.layer0, list(model.layer0.named_parameters())),
                (model.layer2, list(model.layer2.named_parameters())),
                (model, [("weight1", model.weight1)]),
                (model.layer1, list(model.layer1.named_parameters())),
                (model, [("weight2", model.weight2)]),
                (model.layer0, list(model.layer0.named_parameters())),
            ],
        )
        self.assertEqual(
            exec_info.module_to_param_usage_infos[model.layer0],
            [(model.layer0, list(model.layer0.named_parameters()))],
        )
        self.assertEqual(
            exec_info.module_to_param_usage_infos[model.layer1],
            [(model.layer1, list(model.layer1.named_parameters()))],
        )
        self.assertEqual(
            exec_info.module_to_param_usage_infos[model.layer2],
            [
                (model.layer2[0], list(model.layer2[0].named_parameters())),
                (model.layer2[2], list(model.layer2[2].named_parameters())),
            ],
        )
        self.assertEqual(exec_info.module_to_param_usage_infos[model.relu], [])
        # Check `param_forward_order`
        correct_param_order = [
            model.layer0.weight,
            model.layer0.bias,
            model.layer2[0].weight,
            model.layer2[2].weight,
            model.weight1,
            model.layer1.weight,
            model.weight2,
        ]
        self.assertEqual(exec_info.param_forward_order, correct_param_order)
        # Check `visited_params`
        self.assertEqual(
            len(exec_info.visited_params), len(exec_info.param_forward_order)
        )
        self.assertEqual(exec_info.visited_params, set(exec_info.param_forward_order))


devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(
    TestSymbolicTracing, globals(), only_for=devices, allow_xpu=True
)
if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""        Tests running ``tracer.trace()`` inside ``patch_tracer()`` by checking        the saved data structures.

This Python file contains 2 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Model`, `TestSymbolicTracing`

**Functions defined**: `__init__`, `forward`, `test_symbolic_tracing_outputs`

**Key imports**: torch, _ExecOrderTracer, instantiate_device_type_tests, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/fsdp`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.distributed.fsdp._trace_utils`: _ExecOrderTracer
- `torch.testing._internal.common_device_type`: instantiate_device_type_tests
- `torch.testing._internal.common_utils`: run_tests, TestCase


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/fsdp/test_fsdp_fx.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/fsdp`):

- [`test_fsdp_memory.py_docs.md`](./test_fsdp_memory.py_docs.md)
- [`test_fsdp_mixed_precision.py_docs.md`](./test_fsdp_mixed_precision.py_docs.md)
- [`test_fsdp_uneven.py_docs.md`](./test_fsdp_uneven.py_docs.md)
- [`test_fsdp_dtensor_state_dict.py_docs.md`](./test_fsdp_dtensor_state_dict.py_docs.md)
- [`test_fsdp_tp_integration.py_docs.md`](./test_fsdp_tp_integration.py_docs.md)
- [`test_distributed_checkpoint.py_docs.md`](./test_distributed_checkpoint.py_docs.md)
- [`test_fsdp_multiple_forward.py_docs.md`](./test_fsdp_multiple_forward.py_docs.md)
- [`test_checkpoint_wrapper.py_docs.md`](./test_checkpoint_wrapper.py_docs.md)
- [`test_fsdp_clip_grad_norm.py_docs.md`](./test_fsdp_clip_grad_norm.py_docs.md)
- [`test_fsdp_use_orig_params.py_docs.md`](./test_fsdp_use_orig_params.py_docs.md)


## Cross-References

- **File Documentation**: `test_fsdp_fx.py_docs.md`
- **Keyword Index**: `test_fsdp_fx.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
