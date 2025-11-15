# Documentation: `docs/torch/distributed/tensor/examples/visualize_sharding_example.py_docs.md`

## File Metadata

- **Path**: `docs/torch/distributed/tensor/examples/visualize_sharding_example.py_docs.md`
- **Size**: 6,746 bytes (6.59 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This file contains **examples or benchmarks**.

## Original Source

```markdown
# Documentation: `torch/distributed/tensor/examples/visualize_sharding_example.py`

## File Metadata

- **Path**: `torch/distributed/tensor/examples/visualize_sharding_example.py`
- **Size**: 4,224 bytes (4.12 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**.

## Original Source

```python
"""
To run the example, use the following command:
TERM=xterm-256color torchrun --nproc-per-node=4 visualize_sharding_example.py
"""

import os

import rich
import rich.rule

import torch
import torch.distributed as dist
import torch.distributed.tensor as dt
import torch.distributed.tensor.debug


assert int(os.getenv("WORLD_SIZE", "1")) >= 4, "We need at least 4 devices"
rank = int(os.environ["RANK"])


device_type = getattr(torch.accelerator.current_accelerator(), "type", "cpu")


def section(msg: str) -> None:
    if rank == 0:
        rich.print(rich.rule.Rule(msg))


def visualize(t: dt.DTensor, msg: str = "") -> None:
    if rank == 0:
        rich.print(msg)
        dt.debug.visualize_sharding(t, use_rich=False)
        dt.debug.visualize_sharding(t, use_rich=True)


section("[bold]1D Tensor; 1D Mesh[/bold]")
m = dist.init_device_mesh(device_type, (4,))
t = torch.ones(4)
visualize(
    dt.distribute_tensor(t, m, [dt.Replicate()]),
    "Replicate along the only mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=0)]),
    "Shard along the only tensor dimension",
)

section("[bold]2D Tensor; 1D Mesh[/bold]")
m = dist.init_device_mesh(device_type, (4,))
t = torch.ones(4, 4)
visualize(
    dt.distribute_tensor(t, m, [dt.Replicate()]),
    "Replicate along the only mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=0)]),
    "Shard alone the first tensor dimension along the only mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=1)]),
    "Shard along the second tensor dimension along the only mesh dimension",
)

section("[bold]1D Tensor; 2D Mesh[/bold]")
m = dist.init_device_mesh(device_type, (2, 2))
t = torch.ones(4)
visualize(
    dt.distribute_tensor(t, m, [dt.Replicate(), dt.Replicate()]),
    "Replicate along both mesh dimensions",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=0), dt.Shard(dim=0)]),
    "Shard the only tensor dimension along both mesh dimensions",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=0), dt.Replicate()]),
    "Shard the only tensor dimension along the first mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Replicate(), dt.Shard(dim=0)]),
    "Shard the only tensor dimension along the second mesh dimension",
)

section("[bold]2D Tensor; 2D Mesh[/bold]")
m = dist.init_device_mesh(device_type, (2, 2))
t = torch.ones(4, 4)
visualize(
    dt.distribute_tensor(t, m, [dt.Replicate(), dt.Replicate()]),
    "Replicate along both mesh dimensions",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=0), dt.Shard(dim=0)]),
    "Shard the first tensor dimension along both mesh dimensions",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=1), dt.Shard(dim=1)]),
    "Shard the second tensor dimension along both mesh dimensions",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=0), dt.Shard(dim=1)]),
    "Shard the first tensor dimension along the first mesh dimension, "
    + "the second tensor dimension along the second mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=1), dt.Shard(dim=0)]),
    "Shard the first tensor dimension along the second mesh dimension, "
    + "the second tensor dimension along the first mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=0), dt.Replicate()]),
    "Shard the first tensor dimension along the first mesh dimension, "
    + "replicate the second tensor dimension along the second mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Replicate(), dt.Shard(dim=0)]),
    "Shard the first tensor dimension along the second mesh dimension, "
    + "replicate the second tensor dimension along the first mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Shard(dim=1), dt.Replicate()]),
    "Shard the second tensor dimension along the first mesh dimension, "
    + "replicate the second tensor dimension along the second mesh dimension",
)
visualize(
    dt.distribute_tensor(t, m, [dt.Replicate(), dt.Shard(dim=1)]),
    "Shard the second tensor dimension along the second mesh dimension, "
    + "replicate the second tensor dimension along the first mesh dimension",
)


dist.destroy_process_group()

```



## High-Level Overview

"""To run the example, use the following command:TERM=xterm-256color torchrun --nproc-per-node=4 visualize_sharding_example.py

This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `section`, `visualize`

**Key imports**: os, rich, rich.rule, torch, torch.distributed as dist, torch.distributed.tensor as dt, torch.distributed.tensor.debug


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/distributed/tensor/examples`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `rich`
- `rich.rule`
- `torch`
- `torch.distributed as dist`
- `torch.distributed.tensor as dt`
- `torch.distributed.tensor.debug`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/distributed/tensor/examples`):

- [`comm_mode_features_example.py_docs.md`](./comm_mode_features_example.py_docs.md)
- [`torchrec_sharding_example.py_docs.md`](./torchrec_sharding_example.py_docs.md)
- [`flex_attention_cp.py_docs.md`](./flex_attention_cp.py_docs.md)
- [`convnext_example.py_docs.md`](./convnext_example.py_docs.md)


## Cross-References

- **File Documentation**: `visualize_sharding_example.py_docs.md`
- **Keyword Index**: `visualize_sharding_example.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/distributed/tensor/examples`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/distributed/tensor/examples`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/distributed/tensor/examples`):

- [`visualize_sharding_example.py_kw.md_docs.md`](./visualize_sharding_example.py_kw.md_docs.md)
- [`convnext_example.py_kw.md_docs.md`](./convnext_example.py_kw.md_docs.md)
- [`flex_attention_cp.py_kw.md_docs.md`](./flex_attention_cp.py_kw.md_docs.md)
- [`convnext_example.py_docs.md_docs.md`](./convnext_example.py_docs.md_docs.md)
- [`torchrec_sharding_example.py_kw.md_docs.md`](./torchrec_sharding_example.py_kw.md_docs.md)
- [`comm_mode_features_example.py_docs.md_docs.md`](./comm_mode_features_example.py_docs.md_docs.md)
- [`comm_mode_features_example.py_kw.md_docs.md`](./comm_mode_features_example.py_kw.md_docs.md)
- [`torchrec_sharding_example.py_docs.md_docs.md`](./torchrec_sharding_example.py_docs.md_docs.md)
- [`flex_attention_cp.py_docs.md_docs.md`](./flex_attention_cp.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `visualize_sharding_example.py_docs.md_docs.md`
- **Keyword Index**: `visualize_sharding_example.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
