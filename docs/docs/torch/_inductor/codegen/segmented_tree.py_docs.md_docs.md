# Documentation: `docs/torch/_inductor/codegen/segmented_tree.py_docs.md`

## File Metadata

- **Path**: `docs/torch/_inductor/codegen/segmented_tree.py_docs.md`
- **Size**: 11,368 bytes (11.10 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/_inductor/codegen/segmented_tree.py`

## File Metadata

- **Path**: `torch/_inductor/codegen/segmented_tree.py`
- **Size**: 8,193 bytes (8.00 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
from collections.abc import Callable
from typing import Generic, Optional, TypeVar


T = TypeVar("T")


def _value_or(opt: Optional[T], default: T) -> T:
    return opt if opt is not None else default


class SegmentedTree(Generic[T]):
    def __init__(
        self,
        values: list[T],
        update_op: Callable[[T, T], T],
        summary_op: Callable[[T, T], T],
        identity_element: T,
    ):
        """
        Initialize a segment tree with the given values and operations.

        Args:
            values: list of initial values
            update_op: Function to apply when updating a value (e.g., addition)
            summary_op: Function to summarize two values (e.g., min, max, sum)
            identity_element: Identity element for the summary_op (e.g., 0 for sum, float('inf') for min)

        Raises:
            ValueError: If the input values list is empty
        """
        if not values:
            raise ValueError("Cannot create a segment tree with empty values list")

        self.n = len(values)
        self.update_op = update_op
        self.summary_op = summary_op
        self.identity = identity_element

        # Size of segment tree array (next power of 2 * 2)
        # The tree follows a standard heap layout where
        # node `n`'s children are at `2*n` and `2*n+1`.
        # Index 0 is unused.
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        self.size *= 2

        # Initialize tree and lazy arrays
        self.tree = [identity_element] * self.size
        # The lazy array contains updates to the given node
        # Upon update, we only push updates to the top-most
        # nodes that fully receive the update. We then
        # propagate the update down as required (i.e., when
        # we receive an interval query that neither fully
        # contains the node nor fully doesn't contain the
        # node
        self.lazy: list[Optional[T]] = [None] * self.size

        # Build the tree
        self._build(values, 1, 0, self.n - 1)

    def _build(self, values: list[T], node: int, start: int, end: int) -> None:
        """
        Build the segment tree recursively.

        Args:
            values: Original array of values
            node: Current node index in the segment tree
            start: Start index of the segment
            end: End index of the segment
        """
        if start == end:
            # Leaf node
            if start < len(values):
                self.tree[node] = values[start]
            return

        mid = (start + end) // 2
        left_child = 2 * node
        right_child = 2 * node + 1

        # Recursively build left and right subtrees
        self._build(values, left_child, start, mid)
        self._build(values, right_child, mid + 1, end)

        # Update current node with summary of children
        self.tree[node] = self.summary_op(self.tree[left_child], self.tree[right_child])

    def _children(self, node: int) -> list[int]:
        return [2 * node, 2 * node + 1]

    def _push_lazy(self, node: int, start: int, end: int) -> None:
        """
        Push lazy updates down to children.

        Args:
            node: Current node index
            start: Start index of the segment
            end: End index of the segment
        """
        lazy_node = self.lazy[node]
        if lazy_node is None:
            return

        # Apply lazy update to current node
        self.tree[node] = self.update_op(self.tree[node], lazy_node)

        if start != end:  # Not a leaf node
            # Propagate to children
            for child in self._children(node):
                self.lazy[child] = self.update_op(
                    _value_or(self.lazy[child], self.identity), lazy_node
                )

        # Clear the lazy value
        self.lazy[node] = None

    def _update_range_helper(
        self, node: int, start: int, end: int, left: int, right: int, value: T
    ) -> None:
        """
        Helper method to update a range of values in the segment tree.

        Args:
            node: Current node index
            start: Start index of the current segment
            end: End index of the current segment
            left: Start index of the range to update
            right: End index of the range to update
            value: Value to apply to the range
        """
        # Push lazy updates before processing this node
        self._push_lazy(node, start, end)

        # No overlap
        if start > right or end < left:
            return

        # Complete overlap
        if start >= left and end <= right:
            # Apply update to current node
            self.lazy[node] = value
            self._push_lazy(node, start, end)
            return

        # Partial overlap, recurse to children
        mid = (start + end) // 2
        left_child = 2 * node
        right_child = 2 * node + 1

        self._update_range_helper(left_child, start, mid, left, right, value)
        self._update_range_helper(right_child, mid + 1, end, left, right, value)

        # Update current node based on children
        self.tree[node] = self.summary_op(self.tree[left_child], self.tree[right_child])

    def _query_range_helper(
        self, node: int, start: int, end: int, left: int, right: int
    ) -> T:
        """
        Helper method to query a range of values in the segment tree.

        Args:
            node: Current node index
            start: Start index of the current segment
            end: End index of the current segment
            left: Start index of the range to query
            right: End index of the range to query

        Returns:
            Summary value for the range
        """
        # No overlap
        if start > right or end < left:
            return self.identity

        # Push lazy updates before processing this node
        self._push_lazy(node, start, end)

        # Complete overlap
        if start >= left and end <= right:
            return self.tree[node]

        # Partial overlap, recurse to children
        mid = (start + end) // 2
        left_child = 2 * node
        right_child = 2 * node + 1

        left_result = self._query_range_helper(left_child, start, mid, left, right)
        right_result = self._query_range_helper(right_child, mid + 1, end, left, right)

        # Combine results from children
        return self.summary_op(left_result, right_result)

    def update_range(self, start: int, end: int, value: T) -> None:
        """
        Update a range of values in the segment tree.

        Args:
            start: Start index of the range to update (inclusive)
            end: End index of the range to update (inclusive)
            value: Value to apply to the range

        Raises:
            ValueError: If start > end or indices are out of bounds
        """
        if start > end:
            raise ValueError("Start index must be less than or equal to end index")

        if start < 0 or start >= self.n:
            raise ValueError(f"Start index {start} out of bounds [0, {self.n - 1}]")

        if end < 0 or end >= self.n:
            raise ValueError(f"End index {end} out of bounds [0, {self.n - 1}]")

        self._update_range_helper(1, 0, self.n - 1, start, end, value)

    def summarize_range(self, start: int, end: int) -> T:
        """
        Query a range of values in the segment tree.

        Args:
            start: Start index of the range to query (inclusive)
            end: End index of the range to query (inclusive)

        Returns:
            Summary value for the range according to the summary operation

        Raises:
            ValueError: If start > end or indices are out of bounds
        """
        if start > end:
            raise ValueError("Start index must be less than or equal to end index")

        if start < 0 or start >= self.n:
            raise ValueError(f"Start index {start} out of bounds [0, {self.n - 1}]")

        if end < 0 or end >= self.n:
            raise ValueError(f"End index {end} out of bounds [0, {self.n - 1}]")

        return self._query_range_helper(1, 0, self.n - 1, start, end)

```



## High-Level Overview

"""        Initialize a segment tree with the given values and operations.        Args:            values: list of initial values            update_op: Function to apply when updating a value (e.g., addition)            summary_op: Function to summarize two values (e.g., min, max, sum)            identity_element: Identity element for the summary_op (e.g., 0 for sum, float('inf') for min)        Raises:            ValueError: If the input values list is empty

This Python file contains 1 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SegmentedTree`

**Functions defined**: `_value_or`, `__init__`, `_build`, `_children`, `_push_lazy`, `_update_range_helper`, `_query_range_helper`, `update_range`, `summarize_range`

**Key imports**: Callable, Generic, Optional, TypeVar


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/codegen`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Callable
- `typing`: Generic, Optional, TypeVar


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`torch/_inductor/codegen`):

- [`cpp_wrapper_mps.py_docs.md`](./cpp_wrapper_mps.py_docs.md)
- [`wrapper_fxir.py_docs.md`](./wrapper_fxir.py_docs.md)
- [`cpp_flex_attention_template.py_docs.md`](./cpp_flex_attention_template.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`simd_kernel_features.py_docs.md`](./simd_kernel_features.py_docs.md)
- [`block_analysis.py_docs.md`](./block_analysis.py_docs.md)
- [`cpp_wrapper_cpu_array_ref.py_docs.md`](./cpp_wrapper_cpu_array_ref.py_docs.md)
- [`cpp_bmm_template.py_docs.md`](./cpp_bmm_template.py_docs.md)
- [`python_wrapper_mtia.py_docs.md`](./python_wrapper_mtia.py_docs.md)
- [`cpp_template.py_docs.md`](./cpp_template.py_docs.md)


## Cross-References

- **File Documentation**: `segmented_tree.py_docs.md`
- **Keyword Index**: `segmented_tree.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/_inductor/codegen`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/_inductor/codegen`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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

Files in the same folder (`docs/torch/_inductor/codegen`):

- [`wrapper_fxir.py_kw.md_docs.md`](./wrapper_fxir.py_kw.md_docs.md)
- [`simd.py_docs.md_docs.md`](./simd.py_docs.md_docs.md)
- [`mps_device_op_overrides.py_docs.md_docs.md`](./mps_device_op_overrides.py_docs.md_docs.md)
- [`simd_kernel_features.py_docs.md_docs.md`](./simd_kernel_features.py_docs.md_docs.md)
- [`triton_utils.py_docs.md_docs.md`](./triton_utils.py_docs.md_docs.md)
- [`wrapper.py_kw.md_docs.md`](./wrapper.py_kw.md_docs.md)
- [`mps.py_kw.md_docs.md`](./mps.py_kw.md_docs.md)
- [`cpu_device_op_overrides.py_kw.md_docs.md`](./cpu_device_op_overrides.py_kw.md_docs.md)
- [`cpp_gemm_template.py_kw.md_docs.md`](./cpp_gemm_template.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `segmented_tree.py_docs.md_docs.md`
- **Keyword Index**: `segmented_tree.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
