# Documentation: `test/dynamo/test_precompile_context.py`

## File Metadata

- **Path**: `test/dynamo/test_precompile_context.py`
- **Size**: 4,028 bytes (3.93 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]
import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._functorch
from torch._dynamo.precompile_context import BackendCacheArtifact, PrecompileContext
from torch._functorch import config as functorch_config
from torch._functorch._aot_autograd.autograd_cache import (
    BundledAOTAutogradCacheArtifact,
)
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_triton


@functorch_config.patch({"enable_autograd_cache": True})
@torch._dynamo.config.patch(
    {"caching_precompile": True}
)  # Requires bundledaotautograd cache for now
class PrecompileContextTests(InductorTestCase):
    def setUp(self):
        """
        Reset all counters and caches before each unit test
        """
        super().setUp()
        # Clear PrecompileContext cache artifacts
        PrecompileContext.clear()

    @requires_triton()
    def test_basic(self):
        """
        Test that after torch.compile, PrecompileContext._new_cache_artifacts length is 1
        """

        def simple_function(x):
            return x.sin() + x.cos()

        compiled_fn = torch.compile(simple_function)

        # Run the compiled function
        x = torch.randn(10, device=GPU_TYPE, requires_grad=True)
        result = compiled_fn(x)
        result.sum().backward()
        self.assertEqual(len(PrecompileContext._dynamo_cache_entries), 1)
        self.assertEqual(len(PrecompileContext._backend_artifacts_by_key), 1)
        cache_entries, _ = PrecompileContext.create_cache_entries()
        self.assertEqual(len(cache_entries), 1)

    @requires_triton()
    def test_serialize_by_key(self):
        def simple_function(x):
            return x.sin() + x.cos()

        compiled_fn = torch.compile(simple_function)

        # Run the compiled function
        x = torch.randn(10, device=GPU_TYPE, requires_grad=True)
        result = compiled_fn(x)
        result.sum().backward()
        self.assertEqual(len(PrecompileContext._dynamo_cache_entries), 1)
        self.assertEqual(len(PrecompileContext._backend_artifacts_by_key), 1)
        for key in PrecompileContext._backend_artifacts_by_key.keys():
            result = PrecompileContext.serialize_artifact_by_key(key)
            assert isinstance(result, BackendCacheArtifact)
            self.assertEqual(result.key, key)

        # This should still work
        result, _ = PrecompileContext.create_cache_entries()
        assert len(result) == 1

    @requires_triton()
    def test_editable(self):
        """
        Test that after torch.compile, PrecompileContext._new_cache_artifacts length is 1
        """

        def simple_function(x):
            return x.sin() + x.cos()

        compiled_fn = torch.compile(simple_function)

        # Run the compiled function
        x = torch.randn(10, device=GPU_TYPE, requires_grad=True)
        result = compiled_fn(x)
        result.sum().backward()
        self.assertEqual(len(PrecompileContext._dynamo_cache_entries), 1)
        self.assertEqual(len(PrecompileContext._backend_artifacts_by_key), 1)
        # Find the key for the artifact of type "precompile_aot_autograd"
        key = next(iter(PrecompileContext._backend_artifacts_by_key))

        def edit_fn(x):
            x._my_private_field = 42
            return x

        PrecompileContext.edit_artifact(key, edit_fn)

        result = PrecompileContext.serialize_artifact_by_key(key)
        assert isinstance(result, BundledAOTAutogradCacheArtifact)
        self.assertEqual(result.key, key)

        result, _ = PrecompileContext.create_cache_entries()
        assert len(result) == 1
        aot_autograd_artifacts = next(iter(result.values())).backends
        assert len(aot_autograd_artifacts) == 1
        entry = next(iter(aot_autograd_artifacts.values())).content
        self.assertEqual(entry._my_private_field, 42)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

```



## High-Level Overview

"""        Reset all counters and caches before each unit test

This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `PrecompileContextTests`

**Functions defined**: `setUp`, `test_basic`, `simple_function`, `test_serialize_by_key`, `simple_function`, `test_editable`, `simple_function`, `edit_fn`

**Key imports**: torch, torch._dynamo, torch._dynamo.test_case, torch._functorch, BackendCacheArtifact, PrecompileContext, config as functorch_config, TestCase as InductorTestCase, GPU_TYPE, requires_triton, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._dynamo`
- `torch._dynamo.test_case`
- `torch._functorch`
- `torch._dynamo.precompile_context`: BackendCacheArtifact, PrecompileContext
- `torch._inductor.test_case`: TestCase as InductorTestCase
- `torch.testing._internal.inductor_utils`: GPU_TYPE, requires_triton


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
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
python test/dynamo/test_precompile_context.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/dynamo`):

- [`test_guard_serialization.py_docs.md`](./test_guard_serialization.py_docs.md)
- [`test_subgraphs.py_docs.md`](./test_subgraphs.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_unspec.py_docs.md`](./test_unspec.py_docs.md)
- [`test_trace_rules.py_docs.md`](./test_trace_rules.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_pre_dispatch.py_docs.md`](./test_pre_dispatch.py_docs.md)
- [`test_autograd_function.py_docs.md`](./test_autograd_function.py_docs.md)
- [`test_optimizers.py_docs.md`](./test_optimizers.py_docs.md)
- [`test_callback.py_docs.md`](./test_callback.py_docs.md)


## Cross-References

- **File Documentation**: `test_precompile_context.py_docs.md`
- **Keyword Index**: `test_precompile_context.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
