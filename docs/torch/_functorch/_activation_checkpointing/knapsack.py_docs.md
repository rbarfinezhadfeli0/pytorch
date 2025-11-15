# Documentation: `torch/_functorch/_activation_checkpointing/knapsack.py`

## File Metadata

- **Path**: `torch/_functorch/_activation_checkpointing/knapsack.py`
- **Size**: 3,945 bytes (3.85 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
import torch


def greedy_knapsack(
    memory: list[float], runtimes: list[float], max_memory: float
) -> tuple[float, list[int], list[int]]:
    n = len(runtimes)
    items = list(range(n))

    # Sort items based on the ratio of runtime to memory in descending order
    items = sorted(items, key=lambda i: runtimes[i] / memory[i], reverse=True)

    total_memory = 0.0
    total_runtime = 0.0
    items_to_save = []
    items_to_allow_recomputing = []

    for i in items:
        if total_memory + memory[i] <= max_memory:
            total_memory += memory[i]
            total_runtime += runtimes[i]
            items_to_save.append(i)
        else:
            items_to_allow_recomputing.append(i)
    return total_runtime, items_to_save, items_to_allow_recomputing


def ilp_knapsack(
    memory: list[float], runtimes: list[float], max_memory: float
) -> tuple[float, list[int], list[int]]:
    import numpy as np

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError:
        raise RuntimeError(
            "To use the ILP for memory budget checkpointing you need to install scipy"
        ) from None

    np_memory = np.array(memory)
    np_runtimes = np.array(runtimes)
    c = -np_runtimes  # type: ignore[operator]

    memory_constraint = LinearConstraint(A=np_memory, ub=np.array(max_memory))
    constraints = [memory_constraint]

    integrality = np.ones_like(c)
    res = milp(
        c=c, constraints=constraints, integrality=integrality, bounds=Bounds(0, 1)
    )
    if not res.success:
        raise RuntimeError("Somehow scipy solving failed")

    items_to_save = []
    items_to_allow_recomputing = []
    for idx, i in enumerate(res.x):
        if i == 1:
            items_to_save.append(idx)
        else:
            items_to_allow_recomputing.append(idx)
    return -res.fun, items_to_save, items_to_allow_recomputing


def dp_knapsack(
    memory: list[float], runtime: list[float], max_memory: float
) -> tuple[float, list[int], list[int]]:
    # Scaling factor to convert floating point weights to integers
    S = 10000

    # Quantize the memory weights
    quantized_memory = torch.tensor(
        [round(m * S) for m in memory], dtype=torch.long, device="cpu"
    )
    runtimes = torch.tensor(runtime, dtype=torch.float32, device="cpu")

    # Quantized pseudopolynomial DP for 0-1 Knapsack
    quantized_max_memory = round(max_memory * S)

    n = len(memory)

    # Initialize the DP table
    # TODO(chilli): I think if needed, this memory can be optimized with sliding
    # window trick + Hirschberg trick:
    # https://codeforces.com/blog/entry/47247?#comment-316200
    dp = torch.zeros(
        (n + 1, quantized_max_memory + 1), dtype=torch.float32, device="cpu"
    )

    for i in range(1, n + 1):
        current_memory = quantized_memory[i - 1]
        current_runtime = runtimes[i - 1]

        # Copy the previous row
        dp[i, :] = dp[i - 1, :]

        # Update dp[i, j] for all j >= current_memory
        if current_memory == 0:
            dp[i, :] = dp[i - 1, :] + current_runtime
        else:
            dp[i, current_memory:] = torch.maximum(
                dp[i - 1, current_memory:],
                dp[i - 1, :-current_memory] + current_runtime,
            )

    # Backtrack to find the items included in the knapsack
    saved_items = []
    recomputable_items = []
    j: int = quantized_max_memory
    for i in range(n, 0, -1):
        if dp[i][j] != dp[i - 1][j]:
            saved_items.append(i - 1)  # Include this item (indexing from 0)
            j -= int(quantized_memory[i - 1].item())
        else:
            recomputable_items.append(i - 1)

    saved_items.reverse()  # To get items in the order they were added

    # The maximum runtime that can be achieved within the max_memory constraint
    max_runtime = dp[n][quantized_max_memory].item()

    return max_runtime, saved_items, recomputable_items

```



## High-Level Overview


This Python file contains 0 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `greedy_knapsack`, `ilp_knapsack`, `dp_knapsack`

**Key imports**: torch, numpy as np, Bounds, LinearConstraint, milp


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_functorch/_activation_checkpointing`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `numpy as np`
- `scipy.optimize`: Bounds, LinearConstraint, milp


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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

Files in the same folder (`torch/_functorch/_activation_checkpointing`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`ac_logging_utils.py_docs.md`](./ac_logging_utils.py_docs.md)
- [`graph_info_provider.py_docs.md`](./graph_info_provider.py_docs.md)
- [`knapsack_evaluator.py_docs.md`](./knapsack_evaluator.py_docs.md)


## Cross-References

- **File Documentation**: `knapsack.py_docs.md`
- **Keyword Index**: `knapsack.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
