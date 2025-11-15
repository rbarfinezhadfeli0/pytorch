# Documentation: `torch/_inductor/template_heuristics/cutedsl.py`

## File Metadata

- **Path**: `torch/_inductor/template_heuristics/cutedsl.py`
- **Size**: 4,455 bytes (4.35 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from dataclasses import dataclass
from enum import auto, Enum
from itertools import product

import torch._inductor.config as config


class TensorMapUpdateMode(Enum):
    """Enum mirroring cutlass.utils.TensorMapUpdateMode to decouple this file from a cutlass dependency."""

    SMEM = auto()
    GMEM = auto()


@dataclass(frozen=True)
class CuTeGemmConfig:
    TILE_M: int = 128
    TILE_N: int = 192
    CLUSTER_M: int = 2
    CLUSTER_N: int = 1
    USE_2_CTA: bool = False
    TENSORMAP_UPDATE_MODE: TensorMapUpdateMode = TensorMapUpdateMode.SMEM


def get_exhaustive_groupgemm_configs() -> list[CuTeGemmConfig]:
    """
    Returns the exhaustive configuration set for the Blackwell CuTeDSL Grouped GEMM kernel.
    For information regarding valid config sets, see:
    https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/grouped_gemm.py
    """

    # Tile_n is always the same regardless of 2cta
    tile_n_vals = [32, 64, 96, 128, 160, 192, 224, 256]

    # Valid clusters
    clusters_no_2cta = [
        (1, 1),
        (1, 2),
        (1, 4),
        (1, 8),
        (1, 16),
        (2, 1),
        (2, 2),
        (2, 4),
        (2, 8),
        (4, 1),
        (4, 2),
        (4, 4),
        (8, 1),
        (8, 2),
        (16, 1),
    ]
    clusters_2cta = [
        (2, 1),
        (2, 2),
        (2, 4),
        (2, 8),
        (4, 1),
        (4, 2),
        (4, 4),
        (8, 1),
        (8, 2),
        (16, 1),
    ]

    configs: list[CuTeGemmConfig] = []

    for use_2cta, cluster_set, tile_m_range in [
        (False, clusters_no_2cta, [64, 128]),
        (True, clusters_2cta, [128, 256]),
    ]:
        for tensormap_update_mode, tile_m, tile_n, (cluster_m, cluster_n) in product(
            [TensorMapUpdateMode.SMEM, TensorMapUpdateMode.GMEM],
            tile_m_range,
            tile_n_vals,
            cluster_set,
        ):
            configs.append(
                CuTeGemmConfig(
                    tile_m,
                    tile_n,
                    cluster_m,
                    cluster_n,
                    USE_2_CTA=use_2cta,
                    TENSORMAP_UPDATE_MODE=tensormap_update_mode,
                )
            )

    return configs


def get_default_groupgemm_configs() -> list[CuTeGemmConfig]:
    """
    Returns the default configuration set for the Blackwell CuTeDSL Grouped GEMM kernel.
    """

    config_tuples = [
        (128, 256, 2, 1, False, TensorMapUpdateMode.SMEM),
        (256, 160, 2, 1, True, TensorMapUpdateMode.GMEM),
        (256, 256, 2, 1, True, TensorMapUpdateMode.GMEM),
        (64, 32, 1, 1, False, TensorMapUpdateMode.GMEM),
        (64, 256, 1, 2, False, TensorMapUpdateMode.SMEM),
        (128, 256, 1, 2, False, TensorMapUpdateMode.SMEM),
        (256, 256, 2, 2, True, TensorMapUpdateMode.GMEM),
        (128, 256, 1, 2, False, TensorMapUpdateMode.GMEM),
        (64, 32, 1, 1, False, TensorMapUpdateMode.SMEM),
        (256, 256, 2, 1, True, TensorMapUpdateMode.SMEM),
        (128, 256, 1, 1, False, TensorMapUpdateMode.GMEM),
        (256, 256, 8, 1, True, TensorMapUpdateMode.GMEM),
        (64, 32, 1, 2, False, TensorMapUpdateMode.SMEM),
        (256, 192, 2, 1, True, TensorMapUpdateMode.GMEM),
        (256, 256, 2, 2, True, TensorMapUpdateMode.SMEM),
        (128, 96, 1, 2, False, TensorMapUpdateMode.SMEM),
        (64, 192, 1, 1, False, TensorMapUpdateMode.SMEM),
        (64, 64, 1, 1, False, TensorMapUpdateMode.GMEM),
        (64, 192, 1, 1, False, TensorMapUpdateMode.GMEM),
        (128, 64, 1, 1, False, TensorMapUpdateMode.GMEM),
        (64, 160, 1, 1, False, TensorMapUpdateMode.GMEM),
        (64, 256, 1, 1, False, TensorMapUpdateMode.GMEM),
    ]

    return [CuTeGemmConfig(*args) for args in config_tuples]


def get_groupgemm_configs() -> list[CuTeGemmConfig]:
    """
    Returns the configuration set for the Blackwell CuTeDSL Grouped GEMM kernel.

    Note: CuTeDSL autotuning is still experimental — enabling it may trigger kernel launch failures
    or unstable results. By default, autotuning is disabled and we return only
    a single baseline config.
    """
    if (
        config.cutedsl_enable_autotuning
        and config.max_autotune_gemm_search_space == "EXHAUSTIVE"
    ):
        return get_exhaustive_groupgemm_configs()
    elif config.cutedsl_enable_autotuning:
        return get_default_groupgemm_configs()
    else:
        return [get_default_groupgemm_configs()[0]]

```



## High-Level Overview

"""Enum mirroring cutlass.utils.TensorMapUpdateMode to decouple this file from a cutlass dependency."""    SMEM = auto()    GMEM = auto()@dataclass(frozen=True)class CuTeGemmConfig:    TILE_M: int = 128    TILE_N: int = 192    CLUSTER_M: int = 2    CLUSTER_N: int = 1    USE_2_CTA: bool = False    TENSORMAP_UPDATE_MODE: TensorMapUpdateMode = TensorMapUpdateMode.SMEMdef get_exhaustive_groupgemm_configs() -> list[CuTeGemmConfig]:

This Python file contains 3 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TensorMapUpdateMode`, `CuTeGemmConfig`

**Functions defined**: `get_exhaustive_groupgemm_configs`, `get_default_groupgemm_configs`, `get_groupgemm_configs`

**Key imports**: dataclass, auto, Enum, product, torch._inductor.config as config


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/template_heuristics`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `dataclasses`: dataclass
- `enum`: auto, Enum
- `itertools`: product
- `torch._inductor.config as config`


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

Files in the same folder (`torch/_inductor/template_heuristics`):

- [`aten.py_docs.md`](./aten.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`params.py_docs.md`](./params.py_docs.md)
- [`registry.py_docs.md`](./registry.py_docs.md)
- [`decompose_k.py_docs.md`](./decompose_k.py_docs.md)
- [`base.py_docs.md`](./base.py_docs.md)
- [`contiguous_mm.py_docs.md`](./contiguous_mm.py_docs.md)
- [`triton.py_docs.md`](./triton.py_docs.md)
- [`triton_addmm.py_docs.md`](./triton_addmm.py_docs.md)


## Cross-References

- **File Documentation**: `cutedsl.py_docs.md`
- **Keyword Index**: `cutedsl.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
