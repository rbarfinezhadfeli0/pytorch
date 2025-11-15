# Documentation: `docs/test/distributed/pipelining/test_unflatten.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/pipelining/test_unflatten.py_docs.md`
- **Size**: 5,416 bytes (5.29 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/pipelining/test_unflatten.py`

## File Metadata

- **Path**: `test/distributed/pipelining/test_unflatten.py`
- **Size**: 2,449 bytes (2.39 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import torch
from torch.distributed.pipelining import pipe_split, pipeline
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


# Building block for model
class Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding=1
        )
        self.lin0 = torch.nn.Linear(256, 256)
        self.relu = torch.nn.ReLU()
        self.lin1 = torch.nn.Linear(256, 256)

    def forward(self, x: torch.Tensor, constant=None) -> torch.Tensor:
        x = self.conv(x)
        x = self.lin0(x)
        pipe_split()
        x.add(constant)
        x = self.lin1(x)
        return self.relu(x)


# Full model
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block0 = Block()
        self.block1 = Block()

    def forward(self, x: torch.Tensor, constant=None) -> torch.Tensor:
        x = self.block0(x, constant=constant)
        pipe_split()
        x = self.block1(x, constant=constant)
        return x


class UnflattenTests(TestCase):
    def test_unflatten(self, device):
        x = torch.randn(1, 16, 256, 256, device=device)
        constant = torch.ones(1, 16, 256, 256, device=device)

        mod = M().to(device)

        pipe = pipeline(
            mod,
            (x,),
            {"constant": constant},
        )

        assert pipe.num_stages == 4
        orig_state_dict = mod.state_dict()

        # Check qualnames
        for stage_idx in range(pipe.num_stages):
            stage_mod = pipe.get_stage_module(stage_idx)
            for param_name, _ in stage_mod.named_parameters():
                assert param_name in orig_state_dict, (
                    f"{param_name} not in original state dict"
                )
        print("Param qualname test passed")

        # Check equivalence
        ref = mod(x, constant)
        out = pipe(x, constant)[0]
        torch.testing.assert_close(out, ref)
        print(f"Equivalence test passed {torch.sum(out)} ref {torch.sum(ref)}")


devices = ["cpu", "cuda", "hpu", "xpu"]
instantiate_device_type_tests(
    UnflattenTests, globals(), only_for=devices, allow_xpu=True
)

if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 3 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Block`, `M`, `UnflattenTests`

**Functions defined**: `__init__`, `forward`, `__init__`, `forward`, `test_unflatten`

**Key imports**: torch, pipe_split, pipeline, instantiate_device_type_tests, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/pipelining`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.distributed.pipelining`: pipe_split, pipeline
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
python test/distributed/pipelining/test_unflatten.py
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


## Cross-References

- **File Documentation**: `test_unflatten.py_docs.md`
- **Keyword Index**: `test_unflatten.py_kw.md`
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

- **Object-Oriented Design**: Uses classes and constructors
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
python docs/test/distributed/pipelining/test_unflatten.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/pipelining`):

- [`test_transformer.py_kw.md_docs.md`](./test_transformer.py_kw.md_docs.md)
- [`test_schedule_multiproc.py_docs.md_docs.md`](./test_schedule_multiproc.py_docs.md_docs.md)
- [`model_registry.py_kw.md_docs.md`](./model_registry.py_kw.md_docs.md)
- [`schedule_registry.py_docs.md_docs.md`](./schedule_registry.py_docs.md_docs.md)
- [`test_stage.py_docs.md_docs.md`](./test_stage.py_docs.md_docs.md)
- [`schedule_registry.py_kw.md_docs.md`](./schedule_registry.py_kw.md_docs.md)
- [`test_schedule_multiproc.py_kw.md_docs.md`](./test_schedule_multiproc.py_kw.md_docs.md)
- [`test_unflatten.py_kw.md_docs.md`](./test_unflatten.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_unflatten.py_docs.md_docs.md`
- **Keyword Index**: `test_unflatten.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
