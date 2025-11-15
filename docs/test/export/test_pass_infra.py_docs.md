# Documentation: `test/export/test_pass_infra.py`

## File Metadata

- **Path**: `test/export/test_pass_infra.py`
- **Size**: 7,104 bytes (6.94 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: export"]
import copy
import unittest

import torch
from functorch.experimental import control_flow
from torch._dynamo.eval_frame import is_dynamo_supported
from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse
from torch.export import export
from torch.fx.passes.infra.pass_base import PassResult
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase


@unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
class TestPassInfra(TestCase):
    def test_export_pass_base(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x):
                y = torch.cat([x, x])
                return torch.ops.aten.tensor_split.sections(y, 2)

        f = Foo()

        class NullPass(_ExportPassBaseDeprecatedDoNotUse):
            pass

        ep = export(f, (torch.ones(3, 2),), strict=True)
        old_nodes = ep.graph.nodes

        ep = ep._transform_do_not_use(NullPass())
        new_nodes = ep.graph.nodes

        for node in new_nodes:
            if node.op != "call_function":
                continue
            self.assertTrue(hasattr(node, "stack_trace"))
            self.assertIsNotNone(node.stack_trace)

        self.assertEqual(len(new_nodes), len(old_nodes))
        for new_node, old_node in zip(new_nodes, old_nodes):
            self.assertEqual(new_node.op, old_node.op)
            self.assertEqual(new_node.target, old_node.target)

    @unittest.skipIf(IS_WINDOWS, "Windows not supported")
    def test_cond(self) -> None:
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, pred, x, y):
                def true_fn(x, y):
                    b = x.item()
                    torch._check(b >= 2)
                    torch._check(b <= 5)
                    return x - y

                def false_fn(x, y):
                    c = y.item()
                    torch._check(c >= 2)
                    torch._check(c <= 5)
                    return x + y

                ret = control_flow.cond(pred, true_fn, false_fn, [x, y])
                return ret

        x = torch.tensor([2])
        y = torch.tensor([5])
        mod = M()
        _ = export(mod, (torch.tensor(True), x, y), strict=True)._transform_do_not_use(
            _ExportPassBaseDeprecatedDoNotUse()
        )

    def test_node_name_stability(self) -> None:
        # Tests that graph nodes stay the same for nodes that are not touched
        # during transformation
        class CustomModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

                # Define a parameter
                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                # Define two buffers
                self.my_buffer1 = torch.nn.Buffer(torch.tensor(3.0))
                self.my_buffer2 = torch.nn.Buffer(torch.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2

                # Mutate one of the buffers (e.g., increment it by 1)
                self.my_buffer2.add_(1.0)

                return output

        inps = (torch.rand(1), torch.rand(1))
        m = CustomModule()

        ep_before = export(m, inps, strict=True)

        # No op transformation that doesn't perform any meaningful changes to node
        ep_after = ep_before._transform_do_not_use(_ExportPassBaseDeprecatedDoNotUse())

        for before_node, after_node in zip(ep_before.graph.nodes, ep_after.graph.nodes):
            self.assertEqual(before_node.name, after_node.name)

    def test_graph_signature_updated_after_transformation(self) -> None:
        # Checks that pass infra correctly updates graph signature
        # after transformations.
        class CustomModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                self.my_buffer1 = torch.nn.Buffer(torch.tensor(3.0))
                self.my_buffer2 = torch.nn.Buffer(torch.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2
                return output

        my_module = CustomModule()

        # Test the custom module with two input tensors
        input_tensor1 = torch.tensor(5.0)
        input_tensor2 = torch.tensor(6.0)

        ep_before = torch.export.export(
            my_module, (input_tensor1, input_tensor2), strict=True
        )
        from torch.fx.passes.infra.pass_base import PassResult

        def modify_input_output_pass(gm):
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    node.name = node.name + "_modified"
            gm.recompile()
            return PassResult(gm, True)

        ep_after = ep_before._transform_do_not_use(modify_input_output_pass)
        new_signature = ep_after.graph_signature

        for node_name in new_signature.user_outputs:
            self.assertTrue("_modified" in node_name)

        old_signature = ep_before.graph_signature
        self.assertNotEqual(new_signature.user_outputs, old_signature.user_outputs)

    def test_replace_hook_basic(self) -> None:
        class CustomModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

                self.my_parameter = torch.nn.Parameter(torch.tensor(2.0))

                self.my_buffer1 = torch.nn.Buffer(torch.tensor(3.0))
                self.my_buffer2 = torch.nn.Buffer(torch.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2
                return output

        my_module = CustomModule()
        inputs = (torch.tensor(6.0), torch.tensor(7.0))
        ep_before = export(my_module, inputs, strict=True)

        def replace_pass(gm):
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    node.name = node.name + "_modified"
            gm.recompile()
            return PassResult(gm, True)

        gm = copy.deepcopy(ep_before.graph_module)
        sig = copy.deepcopy(ep_before.graph_signature)

        with gm._set_replace_hook(sig.get_replace_hook()):
            replace_pass(gm)

        for node_name in sig.user_outputs:
            self.assertTrue("_modified" in node_name)

        old_signature = ep_before.graph_signature
        self.assertNotEqual(sig.user_outputs, old_signature.user_outputs)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 7 class(es) and 18 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestPassInfra`, `Foo`, `NullPass`, `M`, `CustomModule`, `CustomModule`, `CustomModule`

**Functions defined**: `test_export_pass_base`, `forward`, `test_cond`, `__init__`, `forward`, `true_fn`, `false_fn`, `test_node_name_stability`, `__init__`, `forward`, `test_graph_signature_updated_after_transformation`, `__init__`, `forward`, `modify_input_output_pass`, `test_replace_hook_basic`, `__init__`, `forward`, `replace_pass`

**Key imports**: copy, unittest, torch, control_flow, is_dynamo_supported, _ExportPassBaseDeprecatedDoNotUse, export, PassResult, IS_WINDOWS, run_tests, TestCase, PassResult


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/export`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `unittest`
- `torch`
- `functorch.experimental`: control_flow
- `torch._dynamo.eval_frame`: is_dynamo_supported
- `torch._export.pass_base`: _ExportPassBaseDeprecatedDoNotUse
- `torch.export`: export
- `torch.fx.passes.infra.pass_base`: PassResult
- `torch.testing._internal.common_utils`: IS_WINDOWS, run_tests, TestCase


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
python test/export/test_pass_infra.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/export`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_schema.py_docs.md`](./test_schema.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_cpp_serdes.py_docs.md`](./test_cpp_serdes.py_docs.md)
- [`test_export_opinfo.py_docs.md`](./test_export_opinfo.py_docs.md)
- [`test_lift_unlift.py_docs.md`](./test_lift_unlift.py_docs.md)
- [`test_retraceability.py_docs.md`](./test_retraceability.py_docs.md)
- [`test_converter.py_docs.md`](./test_converter.py_docs.md)
- [`test_nativert.py_docs.md`](./test_nativert.py_docs.md)
- [`test_export.py_docs.md`](./test_export.py_docs.md)


## Cross-References

- **File Documentation**: `test_pass_infra.py_docs.md`
- **Keyword Index**: `test_pass_infra.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
