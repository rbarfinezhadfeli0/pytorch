# Documentation: `docs/test/distributed/pipelining/test_transformer.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/pipelining/test_transformer.py_docs.md`
- **Size**: 5,438 bytes (5.31 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/pipelining/test_transformer.py`

## File Metadata

- **Path**: `test/distributed/pipelining/test_transformer.py`
- **Size**: 2,438 bytes (2.38 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import torch
from torch.distributed.pipelining import pipeline, SplitPoint
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


d_hid = 16
n_layers = 8
microbatch_size = 4


class MLPModule(torch.nn.Module):
    def __init__(self, d_hid):
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


class TransformerLike(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(*[MLPModule(d_hid) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerTests(TestCase):
    def test_ir(self, device):
        transformer = TransformerLike().to(device)
        x = torch.randn(microbatch_size, d_hid, device=device)

        # Split into 2 stages
        num_stages = 2
        split_spec = {f"layers.{n_layers // num_stages}": SplitPoint.BEGINNING}

        pipe = pipeline(
            transformer,
            (x,),
            split_spec=split_spec,
        )
        assert pipe.num_stages == num_stages, f"{pipe.num_stages=}, expect {num_stages}"

        def get_layers(module):
            layers = [name for name, _ in module.layers.named_children()]
            return layers

        # Collect all layers in pipe
        layers = []
        for stage_idx in range(pipe.num_stages):
            stage_mod = pipe.get_stage_module(stage_idx)
            layers += get_layers(stage_mod)

        # Check layer completeness
        orig_layers = get_layers(transformer)
        assert sorted(layers) == sorted(orig_layers), f"{layers} != {orig_layers}"
        print("Layers matched!")

        # Check equivalence
        ref = transformer(x)
        out = pipe(x)[0]
        torch.testing.assert_close(out, ref)
        print(f"Equivalence test passed {torch.sum(out)} ref {torch.sum(ref)}")


devices = ["cpu", "cuda", "hpu", "xpu"]
instantiate_device_type_tests(
    TransformerTests, globals(), only_for=devices, allow_xpu=True
)

if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 3 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MLPModule`, `TransformerLike`, `TransformerTests`

**Functions defined**: `__init__`, `forward`, `__init__`, `forward`, `test_ir`, `get_layers`

**Key imports**: torch, pipeline, SplitPoint, instantiate_device_type_tests, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/pipelining`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.distributed.pipelining`: pipeline, SplitPoint
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
python test/distributed/pipelining/test_transformer.py
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
- [`test_stage.py_docs.md`](./test_stage.py_docs.md)
- [`schedule_registry.py_docs.md`](./schedule_registry.py_docs.md)
- [`test_unflatten.py_docs.md`](./test_unflatten.py_docs.md)


## Cross-References

- **File Documentation**: `test_transformer.py_docs.md`
- **Keyword Index**: `test_transformer.py_kw.md`
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
python docs/test/distributed/pipelining/test_transformer.py_docs.md
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

- **File Documentation**: `test_transformer.py_docs.md_docs.md`
- **Keyword Index**: `test_transformer.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
