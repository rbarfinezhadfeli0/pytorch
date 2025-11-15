# Documentation: `test/jit/xnnpack/test_xnnpack_delegate.py`

## File Metadata

- **Path**: `test/jit/xnnpack/test_xnnpack_delegate.py`
- **Size**: 5,782 bytes (5.65 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import unittest

import torch
import torch._C


torch.ops.load_library("//caffe2:xnnpack_backend")


class TestXNNPackBackend(unittest.TestCase):
    def test_xnnpack_constant_data(self):
        class Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._constant = torch.ones(4, 4, 4)

            def forward(self, x):
                return x + self._constant

        scripted_module = torch.jit.script(Module())

        lowered_module = torch._C._jit_to_backend(
            "xnnpack",
            scripted_module,
            {
                "forward": {
                    "inputs": [torch.randn(4, 4, 4)],
                    "outputs": [torch.randn(4, 4, 4)],
                }
            },
        )

        for _ in range(20):
            sample_input = torch.randn(4, 4, 4)
            actual_output = scripted_module(sample_input)
            expected_output = lowered_module(sample_input)
            self.assertTrue(
                torch.allclose(actual_output, expected_output, atol=1e-03, rtol=1e-03)
            )

    def test_xnnpack_lowering(self):
        class Module(torch.nn.Module):
            def forward(self, x):
                return x + x

        scripted_module = torch.jit.script(Module())

        faulty_compile_spec = {
            "backward": {
                "inputs": [torch.zeros(1)],
                "outputs": [torch.zeros(1)],
            }
        }
        error_msg = 'method_compile_spec does not contain the "forward" key.'

        with self.assertRaisesRegex(
            RuntimeError,
            error_msg,
        ):
            _ = torch._C._jit_to_backend(
                "xnnpack",
                scripted_module,
                faulty_compile_spec,
            )

        mismatch_compile_spec = {
            "forward": {
                "inputs": [torch.zeros(1), torch.zeros(1)],
                "outputs": [torch.zeros(1)],
            }
        }
        error_msg = (
            "method_compile_spec inputs do not match expected number of forward inputs"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            error_msg,
        ):
            _ = torch._C._jit_to_backend(
                "xnnpack", scripted_module, mismatch_compile_spec
            )

        lowered = torch._C._jit_to_backend(
            "xnnpack",
            scripted_module,
            {
                "forward": {
                    "inputs": [torch.zeros(1)],
                    "outputs": [torch.zeros(1)],
                }
            },
        )
        lowered(torch.zeros(1))

    def test_xnnpack_backend_add(self):
        class AddModule(torch.nn.Module):
            def forward(self, x, y):
                z = x + y
                z = z + x
                z = z + x
                return z

        add_module = AddModule()
        sample_inputs = (torch.rand(1, 512, 512, 3), torch.rand(1, 512, 512, 3))
        sample_output = torch.zeros(1, 512, 512, 3)

        add_module = torch.jit.script(add_module)
        expected_output = add_module(sample_inputs[0], sample_inputs[1])

        lowered_add_module = torch._C._jit_to_backend(
            "xnnpack",
            add_module,
            {
                "forward": {
                    "inputs": [sample_inputs[0].clone(), sample_inputs[1].clone()],
                    "outputs": [sample_output],
                }
            },
        )

        actual_output = lowered_add_module.forward(sample_inputs[0], sample_inputs[1])
        self.assertTrue(
            torch.allclose(actual_output, expected_output, atol=1e-03, rtol=1e-03)
        )

    def test_xnnpack_broadcasting(self):
        class AddModule(torch.nn.Module):
            def forward(self, x, y):
                return x + y

        add_module = AddModule()
        sample_inputs = (torch.rand(5, 1, 4, 1), torch.rand(3, 1, 1))
        sample_output = torch.zeros(5, 3, 4, 1)

        add_module = torch.jit.script(add_module)
        expected_output = add_module(sample_inputs[0], sample_inputs[1])

        lowered_add_module = torch._C._jit_to_backend(
            "xnnpack",
            add_module,
            {
                "forward": {
                    "inputs": [sample_inputs[0], sample_inputs[1]],
                    "outputs": [sample_output],
                }
            },
        )

        actual_output = lowered_add_module.forward(sample_inputs[0], sample_inputs[1])
        self.assertTrue(
            torch.allclose(actual_output, expected_output, atol=1e-03, rtol=1e-03)
        )

    def test_xnnpack_unsupported(self):
        class AddSpliceModule(torch.nn.Module):
            def forward(self, x, y):
                z = x + y[:, :, 1, :]
                return z

        sample_inputs = (torch.rand(1, 512, 512, 3), torch.rand(1, 512, 512, 3))
        sample_output = torch.zeros(1, 512, 512, 3)

        error_msg = (
            "the module contains the following unsupported ops:\n"
            "aten::select\n"
            "aten::slice\n"
        )

        add_module = torch.jit.script(AddSpliceModule())
        with self.assertRaisesRegex(
            RuntimeError,
            error_msg,
        ):
            _ = torch._C._jit_to_backend(
                "xnnpack",
                add_module,
                {
                    "forward": {
                        "inputs": [sample_inputs[0], sample_inputs[1]],
                        "outputs": [sample_output],
                    }
                },
            )


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )

```



## High-Level Overview


This Python file contains 6 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestXNNPackBackend`, `Module`, `Module`, `AddModule`, `AddModule`, `AddSpliceModule`

**Functions defined**: `test_xnnpack_constant_data`, `__init__`, `forward`, `test_xnnpack_lowering`, `forward`, `test_xnnpack_backend_add`, `forward`, `test_xnnpack_broadcasting`, `forward`, `test_xnnpack_unsupported`, `forward`

**Key imports**: unittest, torch, torch._C


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit/xnnpack`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `torch`
- `torch._C`


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
python test/jit/xnnpack/test_xnnpack_delegate.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/jit/xnnpack`):



## Cross-References

- **File Documentation**: `test_xnnpack_delegate.py_docs.md`
- **Keyword Index**: `test_xnnpack_delegate.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
