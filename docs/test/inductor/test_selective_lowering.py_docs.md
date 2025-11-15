# Documentation: `test/inductor/test_selective_lowering.py`

## File Metadata

- **Path**: `test/inductor/test_selective_lowering.py`
- **Size**: 2,918 bytes (2.85 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
"""
Test selective lowering control via node metadata annotations.
"""

from collections.abc import Callable

import torch
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


@instantiate_parametrized_tests
class SelectiveLoweringTest(InductorTestCase):
    """
    Tests for user-controllable selective lowering using node.meta annotations.
    """

    device = GPU_TYPE

    def _mark_nodes_for_fallback(
        self, gm: torch.fx.GraphModule, predicate: Callable[[torch.fx.Node], bool]
    ) -> torch.fx.GraphModule:
        """
        Helper method to mark nodes with should_fallback metadata based on a predicate.
        """
        for node in gm.graph.nodes:
            if node.op == "call_function" and predicate(node):
                node.meta["should_fallback"] = True
        return gm

    def test_basic_selective_lowering(self):
        """
        Test that nodes marked for fallback use fallback handlers instead of lowerings.
        """

        def foo(x, y):
            a = x + y  # This will be marked for fallback
            b = a * 2  # This will use normal lowering
            return b

        x = torch.randn(10, device=self.device)
        y = torch.randn(10, device=self.device)

        def custom_backend(gm: torch.fx.GraphModule, example_inputs):
            # Mark all add operations for fallback
            def should_fallback_add(node: torch.fx.Node) -> bool:
                return node.target == torch.ops.aten.add.Tensor

            self._mark_nodes_for_fallback(gm, should_fallback_add)

            from torch._inductor.compile_fx import compile_fx

            return compile_fx(gm, example_inputs)

        compiled_fn = torch.compile(foo, backend=custom_backend)
        result = compiled_fn(x, y)
        expected = foo(x, y)

        self.assertTrue(torch.allclose(result, expected))

    def test_no_fallback_when_unmarked(self):
        """
        Test that operations without fallback annotation use normal lowering.
        """

        def foo(x, y):
            return x + y

        x = torch.randn(10, device=self.device)
        y = torch.randn(10, device=self.device)

        def custom_backend(gm: torch.fx.GraphModule, example_inputs):
            # Don't mark anything - all operations should use normal lowering
            from torch._inductor.compile_fx import compile_fx

            return compile_fx(gm, example_inputs)

        compiled_fn = torch.compile(foo, backend=custom_backend)
        result = compiled_fn(x, y)
        expected = foo(x, y)

        self.assertTrue(torch.allclose(result, expected))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests(needs="filelock")

```



## High-Level Overview

"""Test selective lowering control via node metadata annotations.

This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SelectiveLoweringTest`

**Functions defined**: `_mark_nodes_for_fallback`, `test_basic_selective_lowering`, `foo`, `custom_backend`, `should_fallback_add`, `test_no_fallback_when_unmarked`, `foo`, `custom_backend`

**Key imports**: Callable, torch, TestCase as InductorTestCase, instantiate_parametrized_tests, GPU_TYPE, HAS_GPU, compile_fx, compile_fx, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `collections.abc`: Callable
- `torch`
- `torch._inductor.test_case`: TestCase as InductorTestCase
- `torch.testing._internal.common_utils`: instantiate_parametrized_tests
- `torch.testing._internal.inductor_utils`: GPU_TYPE, HAS_GPU
- `torch._inductor.compile_fx`: compile_fx


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/inductor/test_selective_lowering.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/inductor`):

- [`test_benchmark_fusion.py_docs.md`](./test_benchmark_fusion.py_docs.md)
- [`test_op_dtype_prop.py_docs.md`](./test_op_dtype_prop.py_docs.md)
- [`test_custom_op_autotune.py_docs.md`](./test_custom_op_autotune.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_inductor_freezing.py_docs.md`](./test_inductor_freezing.py_docs.md)
- [`test_b2b_gemm.py_docs.md`](./test_b2b_gemm.py_docs.md)
- [`test_minifier_isolate.py_docs.md`](./test_minifier_isolate.py_docs.md)
- [`test_move_constructors_to_cuda.py_docs.md`](./test_move_constructors_to_cuda.py_docs.md)
- [`test_cutlass_backend.py_docs.md`](./test_cutlass_backend.py_docs.md)
- [`test_cache.py_docs.md`](./test_cache.py_docs.md)


## Cross-References

- **File Documentation**: `test_selective_lowering.py_docs.md`
- **Keyword Index**: `test_selective_lowering.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
