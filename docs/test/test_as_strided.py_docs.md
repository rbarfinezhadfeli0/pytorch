# Documentation: `test/test_as_strided.py`

## File Metadata

- **Path**: `test/test_as_strided.py`
- **Size**: 6,030 bytes (5.89 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: pt2"]

from collections import deque
from typing import Optional

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


def get_state(t: torch.Tensor) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Extract (sizes, strides) tuple from a tensor."""
    return (tuple(t.size()), tuple(t.stride()))


def enumerate_reachable_states(
    initial_size: int,
) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
    """
    Use BFS with DP to enumerate all reachable (size, stride) states from
    a 1D contiguous tensor via valid view operations.

    We only explore states with offset=0 (you can retroactively change the offset).
    We reject states with size=0 or size=1 dimensions as they are degenerate.
    """
    # Create initial 1D contiguous tensor
    initial_tensor = torch.arange(initial_size)

    initial_state = get_state(initial_tensor)

    # Map from state to tensor for that state
    state_to_tensor: dict[tuple[tuple[int, ...], tuple[int, ...]], torch.Tensor] = {
        initial_state: initial_tensor
    }
    visited: set[tuple[tuple[int, ...], tuple[int, ...]]] = {initial_state}
    queue: deque[tuple[tuple[int, ...], tuple[int, ...]]] = deque([initial_state])

    while queue:
        state = queue.popleft()
        t = state_to_tensor[state]
        sizes, strides = state
        ndim = len(sizes)

        def add_state(new_t: torch.Tensor) -> None:
            new_state = get_state(new_t)
            sizes, strides = new_state
            # Skip if has size-0 or size-1 dimensions
            if any(s == 0 or s == 1 for s in sizes):
                return
            # Only accept states where strides are in descending order
            if list(strides) != sorted(strides, reverse=True):
                return
            if new_state not in visited:
                visited.add(new_state)
                queue.append(new_state)
                state_to_tensor[new_state] = new_t

        # 1. Unflatten: try factoring each dimension
        for dim in range(ndim):
            size = sizes[dim]
            assert size > 1
            # Try all factorizations x * y = size where both x, y >= 2
            # We only need to check x up to size // 2 since when x > size // 2,
            # y = size // x < 2, which we reject
            for x in range(2, size // 2 + 1):
                if size % x == 0:
                    y = size // x
                    add_state(t.unflatten(dim, (x, y)))

        # 2. Slice: exhaustively check all possible slicing parameters
        for dim in range(ndim):
            size = sizes[dim]
            for start in range(size):
                for stop in range(start + 1, size + 1):
                    for step in range(1, size + 1):
                        slices = [slice(None)] * ndim
                        slices[dim] = slice(start, stop, step)
                        add_state(t[tuple(slices)])

        # 3. Flatten: merge adjacent dimensions
        for dim in range(ndim - 1):
            add_state(t.flatten(dim, dim + 1))

    return visited


class TestAsStrided(TestCase):
    def test_size_10_exhaustive(self) -> None:
        """Test that size 10 produces exactly the expected 54 states."""
        expected_states = {
            ((2,), (1,)),
            ((2,), (2,)),
            ((2,), (3,)),
            ((2,), (4,)),
            ((2,), (5,)),
            ((2,), (6,)),
            ((2,), (7,)),
            ((2,), (8,)),
            ((2,), (9,)),
            ((2, 2), (2, 1)),
            ((2, 2), (3, 1)),
            ((2, 2), (3, 2)),
            ((2, 2), (4, 1)),
            ((2, 2), (4, 2)),
            ((2, 2), (4, 3)),
            ((2, 2), (5, 1)),
            ((2, 2), (5, 2)),
            ((2, 2), (5, 3)),
            ((2, 2), (5, 4)),
            ((2, 2), (6, 1)),
            ((2, 2), (6, 2)),
            ((2, 2), (6, 3)),
            ((2, 2), (8, 1)),
            ((2, 2, 2), (4, 2, 1)),
            ((2, 2, 2), (5, 2, 1)),
            ((2, 3), (3, 1)),
            ((2, 3), (4, 1)),
            ((2, 3), (5, 1)),
            ((2, 3), (5, 2)),
            ((2, 3), (6, 1)),
            ((2, 4), (4, 1)),
            ((2, 4), (5, 1)),
            ((2, 5), (5, 1)),
            ((3,), (1,)),
            ((3,), (2,)),
            ((3,), (3,)),
            ((3,), (4,)),
            ((3, 2), (2, 1)),
            ((3, 2), (3, 1)),
            ((3, 2), (3, 2)),
            ((3, 2), (4, 1)),
            ((3, 3), (3, 1)),
            ((4,), (1,)),
            ((4,), (2,)),
            ((4,), (3,)),
            ((4, 2), (2, 1)),
            ((5,), (1,)),
            ((5,), (2,)),
            ((5, 2), (2, 1)),
            ((6,), (1,)),
            ((7,), (1,)),
            ((8,), (1,)),
            ((9,), (1,)),
            ((10,), (1,)),
        }

        actual_states = enumerate_reachable_states(10)

        self.assertEqual(len(actual_states), 54)
        self.assertEqual(actual_states, expected_states)

    def test_subset_property(self) -> None:
        """
        Test that for sizes 2..10, each smaller tensor results in a strict
        subset of possible states compared to the next one.
        """
        prev_states: Optional[set[tuple[tuple[int, ...], tuple[int, ...]]]] = None
        for size in range(2, 11):
            current_states = enumerate_reachable_states(size)

            if prev_states is not None:
                # Check that prev_states is a strict subset of current_states
                self.assertTrue(
                    prev_states.issubset(current_states),
                    f"States from size {size - 1} are not a subset of size {size}",
                )
                # Check that it's a strict subset (not equal)
                self.assertTrue(
                    len(prev_states) < len(current_states),
                    f"States from size {size - 1} should be strictly fewer than size {size}",
                )

            prev_states = current_states


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Extract (sizes, strides) tuple from a tensor."""    return (tuple(t.size()), tuple(t.stride()))def enumerate_reachable_states(    initial_size: int,) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:

This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestAsStrided`

**Functions defined**: `get_state`, `enumerate_reachable_states`, `add_state`, `test_size_10_exhaustive`, `test_subset_property`

**Key imports**: deque, Optional, torch, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `collections`: deque
- `typing`: Optional
- `torch`
- `torch.testing._internal.common_utils`: run_tests, TestCase


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

This is a test file. Run it with:

```bash
python test/test_as_strided.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_mkldnn.py_docs.md`](./test_mkldnn.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `test_as_strided.py_docs.md`
- **Keyword Index**: `test_as_strided.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
