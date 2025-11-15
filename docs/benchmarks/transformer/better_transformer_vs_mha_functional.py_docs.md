# Documentation: `benchmarks/transformer/better_transformer_vs_mha_functional.py`

## File Metadata

- **Path**: `benchmarks/transformer/better_transformer_vs_mha_functional.py`
- **Size**: 7,233 bytes (7.06 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
"""
Tests the performance of torch.nn.MultiheadAttention's fast path (BetterTransformer)
vs the slow path (torch.nn.functional.multi_head_attention)

To run this script install these dependencies:

pip install tqdm
pip install prettytable
"""

import argparse
import itertools
import json
import random
import warnings
from collections import defaultdict, OrderedDict
from pathlib import Path
from pprint import pprint
from typing import Optional

import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm

import torch


warnings.filterwarnings("ignore")

error_dict = defaultdict(int)


def benchmark_torch_function(iters, f, *args, **kwargs):
    f(*args, **kwargs)
    f(*args, **kwargs)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        f(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    # elapsed_time has a resolution of 0.5 microseconds:
    # but returns milliseconds, so we need to multiply it to increase resolution
    return start_event.elapsed_time(end_event) * 1000 / iters, *f(*args, **kwargs)


def run(
    a: int,
    b: int,
    iters: int,
    batch_size: int,
    sequence_length: int,
    embed_dim: int,
    num_heads: int,
    device: str,
    dtype: str,
    block_size: int,
    seed,
):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    from scipy.stats import beta

    lengths = (
        beta.rvs(a, b, size=batch_size)
        * (sequence_length + block_size - 1)
        // block_size
    )
    lengths = list(map(int, list(lengths)))
    lengths = [l * block_size for l in lengths]
    lengths = [max(l, block_size) for l in lengths]

    # Used to enforce no padding
    # lengths = [sequence_length] * batch_size

    # Ensure one row in the batch of ele has the max_sequence_length
    lengths[random.randint(0, batch_size - 1)] = sequence_length

    q = [torch.randn(l, embed_dim, device=device, dtype=dtype) for l in lengths]
    q = torch.nested.nested_tensor(q, device=device, dtype=dtype)
    k, v = q, q

    qkv = torch.nn.Linear(embed_dim, 3 * embed_dim, device=device, dtype=dtype)
    proj = torch.nn.Linear(embed_dim, embed_dim, device=device, dtype=dtype)

    native_mha = torch.nn.MultiheadAttention(
        embed_dim, num_heads, batch_first=True, device=device, dtype=dtype
    ).eval()
    native_mha.in_proj_weight = qkv.weight
    native_mha.in_proj_bias = qkv.bias
    native_mha.out_proj.weight = proj.weight
    native_mha.out_proj.bias = proj.bias

    # Create query mask
    q_mask = torch.nested.to_padded_tensor(
        torch.nested.nested_tensor(
            [torch.tensor([True] * length, dtype=torch.bool) for length in lengths]
        ),
        0,
    )
    q_mask = q_mask.cuda()

    if q_mask.size(1) == 0:
        return None

    # Benchmark the native MHA in core
    with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True):
        with torch.inference_mode():
            time_native_mha_fast, y_native_mha_fast, _ = benchmark_torch_function(
                iters, native_mha, q, k, v, need_weights=False
            )
    q = q.to_padded_tensor(0)
    k = q
    v = q
    # Internal Flash Attention
    time_native_mha_slow, y_native_mha_slow, _ = benchmark_torch_function(
        iters, native_mha, q, k, v, key_padding_mask=~q_mask, need_weights=False
    )

    # Convert to padded for comparison
    if y_native_mha_fast.is_nested:
        y_native_mha_fast = torch.nested.to_padded_tensor(y_native_mha_fast, 0)
    y_native_mha_fast = y_native_mha_fast * q_mask.unsqueeze(-1)

    if y_native_mha_slow.is_nested:
        y_native_mha_slow = torch.nested.to_padded_tensor(y_native_mha_slow, 0)
    y_native_mha_slow = y_native_mha_slow * q_mask.unsqueeze(-1)

    # Correctness check
    entry_name = f"batch:{batch_size}_seq_len:{sequence_length}_n_heads:{num_heads}_embed_dim:{embed_dim}"
    try:
        torch.testing.assert_close(
            y_native_mha_fast, y_native_mha_slow, atol=1e-3, rtol=1e-3
        )
    except AssertionError:
        error_dict[entry_name] += 1
        pprint(error_dict)

    # Calculate amount of padding
    padding = 1 - q_mask.float().mean().item()

    # Calculate the speedup for flash attention
    speedup_fast_internal = time_native_mha_slow / time_native_mha_fast

    result_entry = OrderedDict()
    result_entry["dtype"] = dtype
    result_entry["batch_size"] = batch_size
    result_entry["sequence_length"] = sequence_length
    result_entry["n_heads"] = num_heads
    result_entry["embed_dim"] = embed_dim
    result_entry["time_native_mha_slow(\u00b5s)"] = f"{time_native_mha_slow:.3f}"
    result_entry["time_native_mha_fast (\u00b5s)"] = f"{time_native_mha_fast:.3f}"
    result_entry["speedup flash_mha v native_mha"] = f"{speedup_fast_internal:.3f}"
    result_entry["padding"] = f"{padding:.3f}"
    return result_entry


def main(save_path: Optional[Path], error_path: Optional[Path]):
    table = PrettyTable()
    entries = defaultdict(list)

    print("CUDA device: ", torch.cuda.get_device_name(0))
    iters = 100
    header = None
    batch_sizes = [16, 32, 64, 128, 256]
    sequence_lengths = [64, 128, 256, 512]
    embed_dims = [512, 1024]
    num_heads_list = [8, 16]
    betas = range(1, 64, 4)

    for batch_size, sequence_length, embed_dim, num_heads, block_size, b in tqdm(
        list(
            itertools.product(
                batch_sizes, sequence_lengths, embed_dims, num_heads_list, [2], betas
            )
        )
    ):
        seed = 26214  # Magic number that works well for higher b values
        entry = run(
            1,
            b * 0.05,
            iters,
            batch_size,
            sequence_length,
            embed_dim,
            num_heads,
            "cuda",
            torch.float16,
            block_size,
            seed,
        )
        if entry is None:
            continue
        if header is None:
            table.field_names = list(entry.keys())
            header = list(entry.keys())
        row = []
        for k, v in entry.items():
            row.append(v)
            entries[k].append(v)
        table.add_row(row)

    # Print the full table to console
    print(table)
    pprint(error_dict)

    csv_string = table.get_csv_string()
    if save_path is not None:
        with open(save_path, "w") as csvfile:
            csvfile.write(csv_string)

    print(f"Total errors: {sum(error_dict.values())}")
    if error_path is not None:
        with open(error_path, "w") as file:
            file.write(json.dumps(error_dict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path", "--save_path", type=str, help="Path to save the results"
    )
    parser.add_argument(
        "--error-save-path",
        "--error_save_path",
        type=str,
        help="Path to save the errors",
    )

    args = parser.parse_args()
    save_path = Path(args.save_path) if args.save_path else None
    error_path = Path(args.error_save_path) if args.error_save_path else None

    main(save_path, error_path)

```



## High-Level Overview

"""Tests the performance of torch.nn.MultiheadAttention's fast path (BetterTransformer)vs the slow path (torch.nn.functional.multi_head_attention)To run this script install these dependencies:pip install tqdmpip install prettytable

This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `benchmark_torch_function`, `run`, `main`

**Key imports**: argparse, itertools, json, random, warnings, defaultdict, OrderedDict, Path, pprint, Optional, numpy as np


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/transformer`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `itertools`
- `json`
- `random`
- `warnings`
- `collections`: defaultdict, OrderedDict
- `pathlib`: Path
- `pprint`: pprint
- `typing`: Optional
- `numpy as np`
- `prettytable`: PrettyTable
- `tqdm`: tqdm
- `torch`
- `scipy.stats`: beta


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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`benchmarks/transformer`):

- [`attention_bias_benchmarks.py_docs.md`](./attention_bias_benchmarks.py_docs.md)
- [`config_utils.py_docs.md`](./config_utils.py_docs.md)
- [`sdp.py_docs.md`](./sdp.py_docs.md)
- [`score_mod.py_docs.md`](./score_mod.py_docs.md)
- [`sdpa.py_docs.md`](./sdpa.py_docs.md)


## Cross-References

- **File Documentation**: `better_transformer_vs_mha_functional.py_docs.md`
- **Keyword Index**: `better_transformer_vs_mha_functional.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
