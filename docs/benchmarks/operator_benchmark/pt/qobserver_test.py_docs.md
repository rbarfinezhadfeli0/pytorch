# Documentation: `benchmarks/operator_benchmark/pt/qobserver_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/qobserver_test.py`
- **Size**: 4,284 bytes (4.18 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import operator_benchmark as op_bench

import torch
import torch.ao.quantization.observer as obs


qobserver_short_configs_dict = {
    "attr_names": ("C", "M", "N", "dtype", "device"),
    "attrs": (
        (3, 512, 512, torch.quint8, "cpu"),
        (3, 512, 512, torch.quint8, "cuda"),
    ),
    "tags": ("short",),
}

q_hist_observer_short_configs_dict = {
    "attr_names": ("C", "M", "N", "dtype", "device"),
    "attrs": ((3, 512, 512, torch.quint8, "cpu"),),
    "tags": ("short",),
}

qobserver_long_configs_dict = {
    "C": (32, 64),
    "M": (256, 1024),
    "N": (256, 1024),
    "device": ("cpu", "cuda"),
    "dtype": (torch.quint8,),  # dtype doesn't change the timing, keep the same
    "tags": ("long",),
}

q_hist_observer_long_configs_dict = {
    "C": (1, 3, 8),
    "M": (256, 1024),
    "N": (256, 1024),
    "device": ("cpu",),
    "dtype": (torch.quint8,),  # dtype doesn't change the timing, keep the same
    "tags": ("long",),
}


qobserver_per_tensor_configs_short = op_bench.config_list(
    cross_product_configs={
        "qscheme": (torch.per_tensor_affine, torch.per_tensor_symmetric)
    },
    **qobserver_short_configs_dict,
)

qobserver_per_tensor_configs_long = op_bench.cross_product_configs(
    qscheme=(torch.per_tensor_affine, torch.per_tensor_symmetric),
    **qobserver_long_configs_dict,
)

qobserver_per_channel_configs_short = op_bench.config_list(
    cross_product_configs={
        "qscheme": (torch.per_channel_affine, torch.per_channel_symmetric)
    },
    **qobserver_short_configs_dict,
)

qobserver_per_channel_configs_long = op_bench.cross_product_configs(
    qscheme=(torch.per_channel_affine, torch.per_channel_symmetric),
    **qobserver_long_configs_dict,
)

q_hist_observer_per_tensor_configs_short = op_bench.config_list(
    cross_product_configs={
        "qscheme": (torch.per_tensor_affine, torch.per_tensor_symmetric)
    },
    **q_hist_observer_short_configs_dict,
)

q_hist_observer_per_tensor_configs_long = op_bench.cross_product_configs(
    qscheme=(torch.per_tensor_affine, torch.per_tensor_symmetric),
    **q_hist_observer_long_configs_dict,
)


qobserver_per_tensor_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["MinMaxObserver", obs.MinMaxObserver],
        ["MovingAverageMinMaxObserver", obs.MovingAverageMinMaxObserver],
    ],
)

qobserver_per_channel_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["PerChannelMinMaxObserver", obs.PerChannelMinMaxObserver],
        [
            "MovingAveragePerChannelMinMaxObserver",
            obs.MovingAveragePerChannelMinMaxObserver,
        ],
    ],
)

q_hist_observer_list = op_bench.op_list(
    attr_names=["op_name", "op_func"],
    attrs=[
        ["HistogramObserver", obs.HistogramObserver],
        ["HistogramObserverCalculateQparams", obs.HistogramObserver],
    ],
)


class QObserverBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, C, M, N, dtype, qscheme, op_func, device):
        self.inputs = {"f_input": torch.rand(C, M, N, device=device)}
        self.op_func = op_func(dtype=dtype, qscheme=qscheme).to(device)

    def forward(self, f_input):
        self.op_func(f_input)
        return self.op_func.calculate_qparams()


class QObserverBenchmarkCalculateQparams(op_bench.TorchBenchmarkBase):
    def init(self, C, M, N, dtype, qscheme, op_func, device):
        self.f_input = torch.rand(C, M, N, device=device)
        self.q_observer = op_func(dtype=dtype, qscheme=qscheme).to(device)
        self.q_observer(self.f_input)
        self.inputs = {}

    def forward(self):
        return self.q_observer.calculate_qparams()


op_bench.generate_pt_tests_from_op_list(
    qobserver_per_tensor_list,
    qobserver_per_tensor_configs_short + qobserver_per_tensor_configs_long,
    QObserverBenchmark,
)

op_bench.generate_pt_tests_from_op_list(
    qobserver_per_channel_list,
    qobserver_per_channel_configs_short + qobserver_per_channel_configs_long,
    QObserverBenchmark,
)

op_bench.generate_pt_tests_from_op_list(
    q_hist_observer_list,
    q_hist_observer_per_tensor_configs_short + q_hist_observer_per_tensor_configs_long,
    QObserverBenchmarkCalculateQparams,
)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview


This Python file contains 2 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `QObserverBenchmark`, `QObserverBenchmarkCalculateQparams`

**Functions defined**: `init`, `forward`, `init`, `forward`

**Key imports**: operator_benchmark as op_bench, torch, torch.ao.quantization.observer as obs


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/operator_benchmark/pt`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `operator_benchmark as op_bench`
- `torch`
- `torch.ao.quantization.observer as obs`


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

This is a test file. Run it with:

```bash
python benchmarks/operator_benchmark/pt/qobserver_test.py
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

- **File Documentation**: `qobserver_test.py_docs.md`
- **Keyword Index**: `qobserver_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
