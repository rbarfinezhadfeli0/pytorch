# Documentation: `docs/test/distributed/pipelining/test_backward.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/pipelining/test_backward.py_docs.md`
- **Size**: 11,441 bytes (11.17 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/pipelining/test_backward.py`

## File Metadata

- **Path**: `test/distributed/pipelining/test_backward.py`
- **Size**: 8,459 bytes (8.26 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import copy

from model_registry import MLPModule

import torch
from torch.distributed.pipelining._backward import (
    stage_backward,
    stage_backward_input,
    stage_backward_weight,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skipXPUIf,
)
from torch.testing._internal.common_utils import run_tests, TestCase


d_hid = 512
batch_size = 256


class StageBackwardTests(TestCase):
    @skipXPUIf(True, "https://github.com/intel/torch-xpu-ops/issues/1682")
    def test_stage_backward(self, device):
        # MLP as a stage module
        mod = MLPModule(d_hid).to(device)
        x = torch.randn(batch_size, d_hid, device=device)
        # As in a pipeline stage, the inputs to this stage requires gradients
        x.requires_grad_(True)
        target = torch.randn(batch_size, d_hid, device=device)
        loss_fn = torch.nn.MSELoss(reduction="sum")

        # Make a copy
        ref_mod = copy.deepcopy(mod).to(device)
        ref_x = x.detach().requires_grad_(x.requires_grad).to(device)
        ref_target = target.detach().to(device)

        # Forward and backward in stage manner
        out = mod(x)
        loss = loss_fn(out, target)
        grad_inputs = stage_backward(
            stage_output=loss,
            output_grads=None,
            input_values=(x,),
        )

        # Run reference
        ref_out = ref_mod(ref_x)
        ref_loss = loss_fn(ref_out, ref_target)
        ref_loss.backward()

        torch.testing.assert_close(grad_inputs[0], ref_x.grad)

        # Every rank checks gradients
        for name, p in mod.named_parameters():
            ref_p = ref_mod.get_parameter(name)
            try:
                torch.testing.assert_close(p.grad, ref_p.grad)
            except AssertionError:
                print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                raise

    def test_stage_backward_input(self, device):
        # MLP as a stage module
        mod = MLPModule(d_hid).to(device)
        x = torch.randn(batch_size, d_hid, device=device)
        # As in a pipeline stage, the inputs to this stage requires gradients
        x.requires_grad_(True)
        target = torch.randn(batch_size, d_hid, device=device)
        loss_fn = torch.nn.MSELoss(reduction="sum")

        # Make a copy
        ref_mod = copy.deepcopy(mod).to(device)
        ref_x = x.detach().requires_grad_(x.requires_grad).to(device)
        ref_target = target.detach().to(device)

        # Forward, then backward of loss with respect to inputs
        out = mod(x)
        loss = loss_fn(out, target)
        dinputs, _param_groups = stage_backward_input(
            stage_outputs_or_loss=(loss,),
            output_grads=None,
            input_values=[x],
            weights=mod.parameters(),
        )

        # Run reference
        ref_out = ref_mod(ref_x)
        ref_loss = loss_fn(ref_out, ref_target)
        ref_loss.backward()

        torch.testing.assert_close(x.grad, ref_x.grad)
        torch.testing.assert_close(dinputs[0], ref_x.grad)
        for _, p in mod.named_parameters():
            # Check that the weight gradients were not updated
            self.assertEqual(p.grad, None)

    @skipXPUIf(True, "https://github.com/intel/torch-xpu-ops/issues/1682")
    def test_stage_backward_weight(self, device):
        # MLP as a stage module
        mod = MLPModule(d_hid).to(device)
        x = torch.randn(batch_size, d_hid, device=device)
        # As in a pipeline stage, the inputs to this stage requires gradients
        x.requires_grad_(True)
        target = torch.randn(batch_size, d_hid, device=device)
        loss_fn = torch.nn.MSELoss(reduction="sum")

        # Make a copy
        ref_mod = copy.deepcopy(mod).to(device)
        ref_x = x.detach().requires_grad_(x.requires_grad).to(device)
        ref_target = target.detach().to(device)
        # Forward, then backward of loss with respect to inputs
        out = mod(x)
        loss = loss_fn(out, target)
        _dinputs, param_groups = stage_backward_input(
            stage_outputs_or_loss=(loss,),
            output_grads=None,
            input_values=[x],
            weights=mod.parameters(),
        )

        # backward of loss with respect to weights
        stage_backward_weight(mod.parameters(), param_groups, retain_graph=True)

        # Run reference
        ref_out = ref_mod(ref_x)
        ref_loss = loss_fn(ref_out, ref_target)
        ref_loss.backward()

        # Every rank checks gradients
        for name, p in mod.named_parameters():
            ref_p = ref_mod.get_parameter(name)
            try:
                torch.testing.assert_close(p.grad, ref_p.grad)
            except AssertionError:
                print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                raise

    @skipXPUIf(True, "https://github.com/intel/torch-xpu-ops/issues/1682")
    def test_stage_backward_weight_multiple_iters(self, device):
        # MLP as a stage module
        mod = MLPModule(d_hid).to(device)
        inputs = []
        for _ in range(10):
            x = torch.randn(batch_size, d_hid, device=device)
            inputs.append(x)
            # As in a pipeline stage, the inputs to this stage requires gradients
            x.requires_grad_(True)

        target = torch.randn(batch_size, d_hid, device=device)
        loss_fn = torch.nn.MSELoss(reduction="sum")

        # Make a copy
        ref_mod = copy.deepcopy(mod).to(device)
        ref_inputs = []
        for x in inputs:
            ref_x = x.detach().requires_grad_(x.requires_grad).to(device)
            ref_inputs.append(ref_x)
        ref_target = target.detach().to(device)

        # Forward, then backward of loss with respect to inputs
        for x in inputs:
            out = mod(x)
            loss = loss_fn(out, target)
            _dinputs, param_groups = stage_backward_input(
                stage_outputs_or_loss=(loss,),
                output_grads=None,
                input_values=[x],
                weights=mod.parameters(),
            )

            # backward of loss with respect to weights
            stage_backward_weight(mod.parameters(), param_groups)

        # Run reference
        for ref_x in ref_inputs:
            ref_out = ref_mod(ref_x)
            ref_loss = loss_fn(ref_out, ref_target)
            ref_loss.backward()

        # Every rank checks gradients
        for name, p in mod.named_parameters():
            ref_p = ref_mod.get_parameter(name)
            try:
                torch.testing.assert_close(p.grad, ref_p.grad)
            except AssertionError:
                print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                raise

    def test_stage_backward_weight_grad_validation(self, device):
        test_cases = [
            (
                "size >= 2",
                lambda: [
                    (
                        torch.randn(batch_size, d_hid, device=device),
                        torch.randn(batch_size, d_hid, device=device),
                    )
                ],
            ),
            ("size = 1", lambda: [(torch.randn(batch_size, d_hid, device=device),)]),
            (
                "1 grad, 1 None",
                lambda: [(torch.randn(batch_size, d_hid, device=device), None)],
            ),
        ]

        for description, mock_grads_factory in test_cases:
            with self.subTest(description=description):
                mod = MLPModule(d_hid).to(device)
                x = torch.randn(batch_size, d_hid, device=device)
                x.requires_grad_(True)
                out = mod(x)
                loss = torch.sum(out)
                dinputs, param_groups = stage_backward_input(
                    stage_outputs_or_loss=[loss],
                    output_grads=None,
                    input_values=[x],
                    weights=mod.parameters(),
                )

                # Set up mock grads
                for param_group in param_groups:
                    param_group["grads"] = mock_grads_factory()

                stage_backward_weight(mod.parameters(), param_groups)


devices = ["cpu", "cuda", "hpu", "xpu"]
instantiate_device_type_tests(
    StageBackwardTests, globals(), only_for=devices, allow_xpu=True
)

if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `StageBackwardTests`

**Functions defined**: `test_stage_backward`, `test_stage_backward_input`, `test_stage_backward_weight`, `test_stage_backward_weight_multiple_iters`, `test_stage_backward_weight_grad_validation`

**Key imports**: copy, MLPModule, torch, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/pipelining`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `model_registry`: MLPModule
- `torch`
- `torch.testing._internal.common_utils`: run_tests, TestCase


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling
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
python test/distributed/pipelining/test_backward.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/pipelining`):

- [`test_schedule_multiproc.py_docs.md`](./test_schedule_multiproc.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_microbatch.py_docs.md`](./test_microbatch.py_docs.md)
- [`test_schedule.py_docs.md`](./test_schedule.py_docs.md)
- [`test_pipe.py_docs.md`](./test_pipe.py_docs.md)
- [`model_registry.py_docs.md`](./model_registry.py_docs.md)
- [`test_transformer.py_docs.md`](./test_transformer.py_docs.md)
- [`test_stage.py_docs.md`](./test_stage.py_docs.md)
- [`schedule_registry.py_docs.md`](./schedule_registry.py_docs.md)
- [`test_unflatten.py_docs.md`](./test_unflatten.py_docs.md)


## Cross-References

- **File Documentation**: `test_backward.py_docs.md`
- **Keyword Index**: `test_backward.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/pipelining`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/pipelining`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/distributed/pipelining/test_backward.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/pipelining`):

- [`test_transformer.py_kw.md_docs.md`](./test_transformer.py_kw.md_docs.md)
- [`test_schedule_multiproc.py_docs.md_docs.md`](./test_schedule_multiproc.py_docs.md_docs.md)
- [`model_registry.py_kw.md_docs.md`](./model_registry.py_kw.md_docs.md)
- [`test_unflatten.py_docs.md_docs.md`](./test_unflatten.py_docs.md_docs.md)
- [`schedule_registry.py_docs.md_docs.md`](./schedule_registry.py_docs.md_docs.md)
- [`test_stage.py_docs.md_docs.md`](./test_stage.py_docs.md_docs.md)
- [`schedule_registry.py_kw.md_docs.md`](./schedule_registry.py_kw.md_docs.md)
- [`test_schedule_multiproc.py_kw.md_docs.md`](./test_schedule_multiproc.py_kw.md_docs.md)
- [`test_unflatten.py_kw.md_docs.md`](./test_unflatten.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_backward.py_docs.md_docs.md`
- **Keyword Index**: `test_backward.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
