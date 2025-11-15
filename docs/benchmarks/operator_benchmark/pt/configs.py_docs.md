# Documentation: `benchmarks/operator_benchmark/pt/configs.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/configs.py`
- **Size**: 3,788 bytes (3.70 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This file handles **configuration or setup**.

## Original Source

```python
import operator_benchmark as op_bench


"""
Configs shared by multiple benchmarks
"""


def remove_cuda(config_list):
    cuda_config = {"device": "cuda"}
    return [config for config in config_list if cuda_config not in config]


def remove_cpu(config_list):
    cpu_config = {"device": "cpu"}
    return [config for config in config_list if cpu_config not in config]


# Configs for conv-1d ops
conv_1d_configs_short = op_bench.config_list(
    attr_names=["IC", "OC", "kernel", "stride", "N", "L"],
    attrs=[
        [128, 256, 3, 1, 1, 64],
        [256, 256, 3, 2, 4, 64],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

conv_1d_configs_long = op_bench.cross_product_configs(
    IC=[128, 512],
    OC=[128, 512],
    kernel=[3],
    stride=[1, 2],
    N=[8],
    L=[128],
    device=["cpu", "cuda"],
    tags=["long"],
)

convtranspose_1d_configs_short = op_bench.config_list(
    attr_names=["IC", "OC", "kernel", "stride", "N", "L"],
    attrs=[
        [2016, 1026, 1024, 256, 1, 224],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

# Configs for Conv2d and ConvTranspose1d
conv_2d_configs_short = op_bench.config_list(
    attr_names=[
        "IC",
        "OC",
        "kernel",
        "stride",
        "N",
        "H",
        "W",
        "G",
        "pad",
    ],
    attrs=[
        [256, 256, 3, 1, 1, 16, 16, 1, 0],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

conv_2d_configs_long = op_bench.cross_product_configs(
    IC=[128, 256],
    OC=[128, 256],
    kernel=[3],
    stride=[1, 2],
    N=[4],
    H=[32],
    W=[32],
    G=[1],
    pad=[0],
    device=["cpu", "cuda"],
    tags=["long"],
)

# Configs for Conv2dPointwise
conv_2d_pw_configs_short = op_bench.config_list(
    attr_names=[
        "IC",
        "OC",
        "stride",
        "N",
        "H",
        "W",
        "G",
        "pad",
    ],
    attrs=[
        [256, 256, 1, 1, 16, 16, 1, 0],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

conv_2d_pw_configs_long = op_bench.cross_product_configs(
    IC=[128, 256],
    OC=[128, 256],
    stride=[1, 2],
    N=[4],
    H=[32],
    W=[32],
    G=[1],
    pad=[0],
    device=["cpu", "cuda"],
    tags=["long"],
)

# Configs for Conv3d and ConvTranspose3d
conv_3d_configs_short = op_bench.config_list(
    attr_names=["IC", "OC", "kernel", "stride", "N", "D", "H", "W"],
    attrs=[
        [64, 64, 3, 1, 8, 4, 16, 16],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)
conv_3d_configs_long = op_bench.cross_product_configs(
    IC=[16, 32],
    OC=[32, 64],
    kernel=[3, 5],
    stride=[1, 2],
    N=[1],
    D=[128],
    H=[128],
    W=[128],
    device=["cpu", "cuda"],
    tags=["long"],
)

linear_configs_short = op_bench.config_list(
    attr_names=["N", "IN", "OUT"],
    attrs=[
        [1, 1, 1],
        [4, 256, 128],
        [16, 512, 256],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)


linear_configs_long = op_bench.cross_product_configs(
    N=[32, 64], IN=[128, 512], OUT=[64, 128], device=["cpu", "cuda"], tags=["long"]
)

embeddingbag_short_configs = op_bench.cross_product_configs(
    embeddingbags=[10, 120, 1000, 2300],
    dim=[64],
    mode=["sum"],
    input_size=[8, 16, 64],
    offset=[0],
    sparse=[True, False],
    include_last_offset=[True, False],
    device=["cpu"],
    tags=["short"],
)

embedding_short_configs = op_bench.cross_product_configs(
    num_embeddings=[10, 120, 1000, 2300],
    embedding_dim=[64],
    input_size=[8, 16, 64],
    device=["cpu"],
    tags=["short"],
)

```



## High-Level Overview

"""Configs shared by multiple benchmarks

This Python file contains 0 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `remove_cuda`, `remove_cpu`

**Key imports**: operator_benchmark as op_bench


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/operator_benchmark/pt`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `operator_benchmark as op_bench`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`benchmarks/operator_benchmark/pt`):

- [`qarithmetic_test.py_docs.md`](./qarithmetic_test.py_docs.md)
- [`bmm_test.py_docs.md`](./bmm_test.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`gather_test.py_docs.md`](./gather_test.py_docs.md)
- [`clip_ranges_test.py_docs.md`](./clip_ranges_test.py_docs.md)
- [`split_test.py_docs.md`](./split_test.py_docs.md)
- [`groupnorm_test.py_docs.md`](./groupnorm_test.py_docs.md)
- [`sum_test.py_docs.md`](./sum_test.py_docs.md)
- [`matrix_mult_test.py_docs.md`](./matrix_mult_test.py_docs.md)
- [`pool_test.py_docs.md`](./pool_test.py_docs.md)


## Cross-References

- **File Documentation**: `configs.py_docs.md`
- **Keyword Index**: `configs.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
