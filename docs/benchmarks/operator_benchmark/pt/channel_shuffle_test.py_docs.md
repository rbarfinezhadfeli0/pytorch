# Documentation: `benchmarks/operator_benchmark/pt/channel_shuffle_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/channel_shuffle_test.py`
- **Size**: 1,626 bytes (1.59 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch


"""Microbenchmarks for channel_shuffle operator."""


# Configs for PT channel_shuffle operator
channel_shuffle_long_configs = op_bench.cross_product_configs(
    batch_size=[4, 8],
    channels_per_group=[32, 64],
    height=[32, 64],
    width=[32, 64],
    groups=[4, 8],
    channel_last=[True, False],
    tags=["long"],
)


channel_shuffle_short_configs = op_bench.config_list(
    attr_names=["batch_size", "channels_per_group", "height", "width", "groups"],
    attrs=[
        [2, 16, 16, 16, 2],
        [2, 32, 32, 32, 2],
        [4, 32, 32, 32, 4],
        [4, 64, 64, 64, 4],
        [8, 64, 64, 64, 8],
        [16, 64, 64, 64, 16],
    ],
    cross_product_configs={
        "channel_last": [True, False],
    },
    tags=["short"],
)


class ChannelSHuffleBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, batch_size, channels_per_group, height, width, groups, channel_last):
        channels = channels_per_group * groups
        data_shape = (batch_size, channels, height, width)
        input_data = torch.rand(data_shape)
        if channel_last:
            input_data = input_data.contiguous(memory_format=torch.channels_last)
        self.inputs = {"input_data": input_data, "groups": groups}
        self.set_module_name("channel_shuffle")

    def forward(self, input_data, groups: int):
        return torch.channel_shuffle(input_data, groups)


op_bench.generate_pt_test(
    channel_shuffle_short_configs + channel_shuffle_long_configs,
    ChannelSHuffleBenchmark,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for channel_shuffle operator."""# Configs for PT channel_shuffle operatorchannel_shuffle_long_configs = op_bench.cross_product_configs(    batch_size=[4, 8],    channels_per_group=[32, 64],    height=[32, 64],    width=[32, 64],    groups=[4, 8],    channel_last=[True, False],    tags=["long"],)channel_shuffle_short_configs = op_bench.config_list(    attr_names=["batch_size", "channels_per_group", "height", "width", "groups"],    attrs=[        [2, 16, 16, 16, 2],        [2, 32, 32, 32, 2],        [4, 32, 32, 32, 4],        [4, 64, 64, 64, 4],        [8, 64, 64, 64, 8],        [16, 64, 64, 64, 16],    ],    cross_product_configs={        "channel_last": [True, False],    },    tags=["short"],)class ChannelSHuffleBenchmark(op_bench.TorchBenchmarkBase):    def init(self, batch_size, channels_per_group, height, width, groups, channel_last):        channels = channels_per_group * groups        data_shape = (batch_size, channels, height, width)        input_data = torch.rand(data_shape)        if channel_last:            input_data = input_data.contiguous(memory_format=torch.channels_last)        self.inputs = {"input_data": input_data, "groups": groups}        self.set_module_name("channel_shuffle")    def forward(self, input_data, groups: int):        return torch.channel_shuffle(input_data, groups)

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ChannelSHuffleBenchmark`

**Functions defined**: `init`, `forward`

**Key imports**: operator_benchmark as op_bench, torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/operator_benchmark/pt`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `operator_benchmark as op_bench`
- `torch`


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

This is a test file. Run it with:

```bash
python benchmarks/operator_benchmark/pt/channel_shuffle_test.py
```

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

- **File Documentation**: `channel_shuffle_test.py_docs.md`
- **Keyword Index**: `channel_shuffle_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
