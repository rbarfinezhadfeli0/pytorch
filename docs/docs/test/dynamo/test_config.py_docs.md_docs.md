# Documentation: `docs/test/dynamo/test_config.py_docs.md`

## File Metadata

- **Path**: `docs/test/dynamo/test_config.py_docs.md`
- **Size**: 7,214 bytes (7.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**. This file handles **configuration or setup**.

## Original Source

```markdown
# Documentation: `test/dynamo/test_config.py`

## File Metadata

- **Path**: `test/dynamo/test_config.py`
- **Size**: 4,268 bytes (4.17 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. This file handles **configuration or setup**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.utils import disable_cache_limit


# NB: do NOT include this test class in test_dynamic_shapes.py


class ConfigTests(torch._dynamo.test_case.TestCase):
    @disable_cache_limit()
    def test_no_automatic_dynamic(self):
        def fn(a, b):
            return a - b * 10

        torch._dynamo.reset()
        cnt_static = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=False, assume_static_by_default=True
        ):
            opt_fn = torch.compile(fn, backend=cnt_static)
            for i in range(2, 12):
                opt_fn(torch.randn(i), torch.randn(i))
        self.assertEqual(cnt_static.frame_count, 10)

    @disable_cache_limit()
    def test_automatic_dynamic(self):
        def fn(a, b):
            return a - b * 10

        torch._dynamo.reset()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=True, assume_static_by_default=True
        ):
            opt_fn = torch.compile(fn, backend=cnt_dynamic)
            # NB: must not do 0, 1 as they specialized
            for i in range(2, 12):
                opt_fn(torch.randn(i), torch.randn(i))
        # two graphs now rather than 10
        self.assertEqual(cnt_dynamic.frame_count, 2)

    @disable_cache_limit()
    def test_no_assume_static_by_default(self):
        def fn(a, b):
            return a - b * 10

        torch._dynamo.reset()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch(
            automatic_dynamic_shapes=True, assume_static_by_default=False
        ):
            opt_fn = torch.compile(fn, backend=cnt_dynamic)
            # NB: must not do 0, 1 as they specialized
            for i in range(2, 12):
                opt_fn(torch.randn(i), torch.randn(i))
        # one graph now, as we didn't wait for recompile
        self.assertEqual(cnt_dynamic.frame_count, 1)

    def test_config_compile_ignored(self):
        # Remove from this list if no longer relevant
        dynamo_guarded_config_ignorelist = {
            "log_file_name",
            "verbose",
            "verify_correctness",  # will not affect model, will raise RuntimeError
            # (no silent change to compilation behaviour)
            "recompile_limit",
            "accumulated_recompile_limit",
            "replay_record_enabled",
            "cprofile",  # only wraps _compile, not graph
            "repro_after",
            "repro_level",
            "repro_forward_only",
            "repro_tolerance",
            "same_two_models_use_fp64",
            "error_on_recompile",  # safe because: will throw error
            "report_guard_failures",
            "base_dir",  # used for minifying / logging
            "DEBUG_DIR_VAR_NAME",
            "debug_dir_root",
        }
        for k in dynamo_guarded_config_ignorelist:
            assert k in torch._dynamo.config._compile_ignored_keys, k

    def test_config_hash(self):
        config = torch._dynamo.config
        starting_hash = config.get_hash()

        with config.patch({"verbose": not config.verbose}):
            new_hash = config.get_hash()
            assert "verbose" in config._compile_ignored_keys
            assert new_hash == starting_hash

        new_hash = config.get_hash()
        assert new_hash == starting_hash

        with config.patch({"suppress_errors": not config.suppress_errors}):
            changed_hash = config.get_hash()
            assert "suppress_errors" not in config._compile_ignored_keys
            assert changed_hash != starting_hash

            # Test nested patch
            with config.patch({"verbose": not config.verbose}):
                inner_changed_hash = config.get_hash()
                assert inner_changed_hash == changed_hash
                assert inner_changed_hash != starting_hash

        newest_hash = config.get_hash()
        assert changed_hash != newest_hash
        assert newest_hash == starting_hash


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ConfigTests`

**Functions defined**: `test_no_automatic_dynamic`, `fn`, `test_automatic_dynamic`, `fn`, `test_no_assume_static_by_default`, `fn`, `test_config_compile_ignored`, `test_config_hash`

**Key imports**: torch, torch._dynamo.test_case, torch._dynamo.testing, disable_cache_limit, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._dynamo.test_case`
- `torch._dynamo.testing`
- `torch._dynamo.utils`: disable_cache_limit


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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
python test/dynamo/test_config.py
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

- **File Documentation**: `test_config.py_docs.md`
- **Keyword Index**: `test_config.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/dynamo`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/dynamo`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
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
python docs/test/dynamo/test_config.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/dynamo`):

- [`test_error_messages.py_docs.md_docs.md`](./test_error_messages.py_docs.md_docs.md)
- [`test_hooks.py_kw.md_docs.md`](./test_hooks.py_kw.md_docs.md)
- [`test_unittest.py_docs.md_docs.md`](./test_unittest.py_docs.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)
- [`test_aot_autograd.py_kw.md_docs.md`](./test_aot_autograd.py_kw.md_docs.md)
- [`test_einops.py_docs.md_docs.md`](./test_einops.py_docs.md_docs.md)
- [`test_compile.py_kw.md_docs.md`](./test_compile.py_kw.md_docs.md)
- [`test_misc.py_docs.md_docs.md`](./test_misc.py_docs.md_docs.md)
- [`test_buffers_override.py_kw.md_docs.md`](./test_buffers_override.py_kw.md_docs.md)
- [`test_frame_init.py_docs.md_docs.md`](./test_frame_init.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_config.py_docs.md_docs.md`
- **Keyword Index**: `test_config.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
