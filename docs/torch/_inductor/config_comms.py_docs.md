# Documentation: `torch/_inductor/config_comms.py`

## File Metadata

- **Path**: `torch/_inductor/config_comms.py`
- **Size**: 3,078 bytes (3.01 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file handles **configuration or setup**.

## Original Source

```python
import os
import sys
from typing import Optional

from torch.utils._config_module import install_config_module


# Whether to use c10d._time_estimator for collectives runtime estimations.
runtime_estimations_use_nccl_lib_estimations: bool = False

# Config to enable sync of runtime estimations across distributed ranks,
# To prevent passes using this runtime estimations to make different
# decisions on different distributed ranks.
runtime_estimations_align_across_all_distributed_ranks: bool = False

reorder_iterative_debug_memory_recompute: bool = False
reorder_iterative_debug_limit_to_reorder: Optional[int] = (
    None
    # pyrefly: ignore[unbound-name]
    if (env_str := os.getenv("PYTORCH_REORDER_COLLECTIVES_LIMIT")) is None
    else int(env_str)
)
sink_waits_iterative_debug_limit_to_sink: Optional[int] = (
    # pyrefly: ignore[unbound-name]
    None if (env_str := os.getenv("PYTORCH_SINK_WAITS_LIMIT")) is None else int(env_str)
)


# Should be used with config.runtime_estimations_mms_benchmark = True
reorder_iterative_use_runtime_estimations: bool = False
sink_iterative_use_runtime_estimations: bool = False

# Broadcast runtime estimations doing real Collective operation between all ranks.
# If non-deterministic runtime estimations are used this must be used to make
# all ranks to do identical decisions and prevent global Collectives reordering,
# (that will result un NCCL hangs)
reorder_for_compute_comm_overlap_broadcast_runtime_estimations: bool = False

# Block of Ratios to workaround imperfection of current runtime estimations
# for collectives and compute for different scenarios.
# Multiplier of collectives estimated durations
reorder_sink_runtime_estimations_comm_mult: float = 2.0
# Multiplier of compute estimated durations
reorder_sink_runtime_estimations_non_comm_mult: float = 1.0
# The reordering will stop to reorder
# when overlap_comp >= (1 + extra_overlap_ratio) * comm_time
# Allows to configure more aggressive overlap
reorder_iterative_extra_comm_comp_overlap: float = 0.5
# The sink waits reordering will stop to reorder
# when overlap_comp >= (1 + extra_overlap_ratio) * comm_time
# Allows to configure more aggressive sink waits
sink_iterative_extra_comm_comp_overlap: float = 0.5

# Allow reorder iterative pass to increase peak memory
# up to peak_memory_before_pass * (1 + budget)
reorder_iterative_peak_memory_budget: float = 0.2
# Allow sink waits iterative pass to increase peak memory
# up to peak_memory_before_pass * (1 + budget)
sink_iterative_peak_memory_budget: float = 0.2

# Experimental unsafe configuration that allows changing relative collectives order.
# Must be used with runtime_estimations_align_across_all_distributed_ranks = True
reorder_iterative_unsafe_collectives_reorder: bool = True
sink_waits_iterative_unsafe_collectives_reorder: bool = True

# Allow group and move other collectives during reordering
reorder_iterative_group_with_collectives: bool = False
sink_waits_iterative_swap_with_collectives: bool = False

# adds patch, save_config, etc
install_config_module(sys.modules[__name__])

```



## High-Level Overview


This Python file contains 0 class(es) and 0 function(s).

## Detailed Analysis

### Code Structure

**Key imports**: os, sys, Optional, install_config_module


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `sys`
- `typing`: Optional
- `torch.utils._config_module`: install_config_module


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

Files in the same folder (`torch/_inductor`):

- [`freezing_utils.py_docs.md`](./freezing_utils.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`mkldnn_ir.py_docs.md`](./mkldnn_ir.py_docs.md)
- [`async_compile.py_docs.md`](./async_compile.py_docs.md)
- [`invert_expr_analysis.py_docs.md`](./invert_expr_analysis.py_docs.md)
- [`extern_node_serializer.py_docs.md`](./extern_node_serializer.py_docs.md)
- [`loop_body.py_docs.md`](./loop_body.py_docs.md)
- [`debug.py_docs.md`](./debug.py_docs.md)
- [`freezing.py_docs.md`](./freezing.py_docs.md)
- [`optimize_indexing.py_docs.md`](./optimize_indexing.py_docs.md)


## Cross-References

- **File Documentation**: `config_comms.py_docs.md`
- **Keyword Index**: `config_comms.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
