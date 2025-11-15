# Documentation: `test/export/test_verifier.py`

## File Metadata

- **Path**: `test/export/test_verifier.py`
- **Size**: 7,969 bytes (7.78 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: export"]
import unittest

import torch
from functorch.experimental import control_flow
from torch import Tensor
from torch._dynamo.eval_frame import is_dynamo_supported
from torch._export.verifier import SpecViolationError, Verifier
from torch.export import export
from torch.export.exported_program import InputKind, InputSpec, TensorArgument
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase


@unittest.skipIf(not is_dynamo_supported(), "dynamo isn't supported")
class TestVerifier(TestCase):
    def test_verifier_basic(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        f = Foo()

        ep = export(f, (torch.randn(100), torch.randn(100)), strict=True)

        verifier = Verifier()
        verifier.check(ep)

    def test_verifier_call_module(self) -> None:
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x: Tensor) -> Tensor:
                return self.linear(x)

        gm = torch.fx.symbolic_trace(M())

        verifier = Verifier()
        with self.assertRaises(SpecViolationError):
            verifier._check_graph_module(gm)

    def test_verifier_no_functional(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        f = Foo()

        ep = export(
            f, (torch.randn(100), torch.randn(100)), strict=True
        ).run_decompositions({})
        for node in ep.graph.nodes:
            if node.target == torch.ops.aten.add.Tensor:
                node.target = torch.ops.aten.add_.Tensor

        verifier = Verifier()
        with self.assertRaises(SpecViolationError):
            verifier.check(ep)

    @unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
    def test_verifier_higher_order(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                def true_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    return x + y

                def false_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    return x - y

                return control_flow.cond(x.sum() > 2, true_fn, false_fn, [x, y])

        f = Foo()

        ep = export(f, (torch.randn(3, 3), torch.randn(3, 3)), strict=True)

        verifier = Verifier()
        verifier.check(ep)

    @unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
    def test_verifier_nested_invalid_module(self) -> None:
        class Foo(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                def true_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    return x + y

                def false_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                    return x - y

                return control_flow.cond(x.sum() > 2, true_fn, false_fn, [x, y])

        f = Foo()

        ep = export(
            f, (torch.randn(3, 3), torch.randn(3, 3)), strict=True
        ).run_decompositions({})
        for node in ep.graph_module.true_graph_0.graph.nodes:
            if node.target == torch.ops.aten.add.Tensor:
                node.target = torch.ops.aten.add_.Tensor

        verifier = Verifier()
        with self.assertRaises(SpecViolationError):
            verifier.check(ep)

    def test_ep_verifier_basic(self) -> None:
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x: Tensor) -> Tensor:
                return self.linear(x)

        ep = export(M(), (torch.randn(10, 10),), strict=True)
        ep.validate()

    def test_ep_verifier_invalid_param(self) -> None:
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_parameter(
                    name="a", param=torch.nn.Parameter(torch.randn(100))
                )

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y + self.a

        ep = export(M(), (torch.randn(100), torch.randn(100)), strict=True)

        # Parameter doesn't exist in the state dict
        ep.graph_signature.input_specs[0] = InputSpec(
            kind=InputKind.PARAMETER, arg=TensorArgument(name="p_a"), target="bad_param"
        )
        with self.assertRaisesRegex(SpecViolationError, "not in the state dict"):
            ep.validate()

        # Add non-torch.nn.Parameter parameter to the state dict
        ep.state_dict["bad_param"] = torch.randn(100)
        with self.assertRaisesRegex(
            SpecViolationError, "not an instance of torch.nn.Parameter"
        ):
            ep.validate()

    def test_ep_verifier_invalid_buffer(self) -> None:
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor(3.0)

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y + self.a

        ep = export(M(), (torch.randn(100), torch.randn(100)), strict=True)

        # Buffer doesn't exist in the state dict
        ep.graph_signature.input_specs[0] = InputSpec(
            kind=InputKind.BUFFER,
            arg=TensorArgument(name="c_a"),
            target="bad_buffer",
            persistent=True,
        )
        with self.assertRaisesRegex(SpecViolationError, "not in the state dict"):
            ep.validate()

    def test_ep_verifier_buffer_mutate(self) -> None:
        class M(torch.nn.Module):
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

                # Mutate one of the buffers (e.g., increment it by 1)
                self.my_buffer2.add_(1.0)
                return output

        ep = export(M(), (torch.tensor(5.0), torch.tensor(6.0)), strict=True)
        ep.validate()

    def test_ep_verifier_invalid_output(self) -> None:
        class M(torch.nn.Module):
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

                # Mutate one of the buffers (e.g., increment it by 1)
                self.my_buffer2.add_(1.0)
                return output

        ep = export(M(), (torch.tensor(5.0), torch.tensor(6.0)), strict=True)

        output_node = list(ep.graph.nodes)[-1]
        output_node.args = (
            (
                output_node.args[0][0],
                next(iter(ep.graph.nodes)),
            ),
        )

        with self.assertRaisesRegex(SpecViolationError, "Number of output nodes"):
            ep.validate()


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 11 class(es) and 30 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestVerifier`, `Foo`, `M`, `Foo`, `Foo`, `Foo`, `M`, `M`, `M`, `M`, `M`

**Functions defined**: `test_verifier_basic`, `forward`, `test_verifier_call_module`, `__init__`, `forward`, `test_verifier_no_functional`, `forward`, `test_verifier_higher_order`, `forward`, `true_fn`, `false_fn`, `test_verifier_nested_invalid_module`, `forward`, `true_fn`, `false_fn`, `test_ep_verifier_basic`, `__init__`, `forward`, `test_ep_verifier_invalid_param`, `__init__`

**Key imports**: unittest, torch, control_flow, Tensor, is_dynamo_supported, SpecViolationError, Verifier, export, InputKind, InputSpec, TensorArgument, IS_WINDOWS, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/export`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `torch`
- `functorch.experimental`: control_flow
- `torch._dynamo.eval_frame`: is_dynamo_supported
- `torch._export.verifier`: SpecViolationError, Verifier
- `torch.export`: export
- `torch.export.exported_program`: InputKind, InputSpec, TensorArgument
- `torch.testing._internal.common_utils`: IS_WINDOWS, run_tests, TestCase


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/export/test_verifier.py
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

- **File Documentation**: `test_verifier.py_docs.md`
- **Keyword Index**: `test_verifier.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
