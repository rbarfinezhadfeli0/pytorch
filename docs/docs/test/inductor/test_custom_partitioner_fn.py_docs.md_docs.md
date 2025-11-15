# Documentation: `docs/test/inductor/test_custom_partitioner_fn.py_docs.md`

## File Metadata

- **Path**: `docs/test/inductor/test_custom_partitioner_fn.py_docs.md`
- **Size**: 6,109 bytes (5.97 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/inductor/test_custom_partitioner_fn.py`

## File Metadata

- **Path**: `test/inductor/test_custom_partitioner_fn.py`
- **Size**: 2,700 bytes (2.64 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: pt2-dispatcher"]
import torch
from functorch.compile import min_cut_rematerialization_partition
from torch._C import FileCheck
from torch._inductor.custom_graph_pass import CustomPartitionerFn, get_hash_for_files
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_fw_bw_and_get_code
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


class MyCustomPartitionerFn(CustomPartitionerFn):
    """
    A custom partitioner function with static_lifetime_input_indices overwrites.
    """

    def __init__(self):
        super().__init__()
        self.called = False

    def __call__(self, gm, joint_inputs, **kwargs):
        self.called = True
        kwargs["static_lifetime_input_indices"] = [0, 1]
        return min_cut_rematerialization_partition(gm, joint_inputs, **kwargs)

    def uuid(self):
        return get_hash_for_files((__file__,))


class TestCustomPartitionerFn(TestCase):
    def test_custom_partitioner_fn(self):
        """
        For function f(a, b), with the  partitioner in the compile_fx stack,
        the addition `a+b` (equivalently `buf0`) is saved for backward.
        With the custom partitioner function, we indicate that
        `a` and `b` (equivalently `primals_1` and `primals_2`) do not take
        additional memory and thus, they are saved for backward.
        """

        # initialization
        @torch.compile
        def f(a, b):
            return (a + b).cos().cos()

        a = torch.randn((2, 2), requires_grad=True, device=GPU_TYPE)
        b = torch.randn((2, 2), requires_grad=True, device=GPU_TYPE)

        # CASE 1 -- default
        # addition `a + b` (i.e, `buf0`) is saved for backward.
        code_og = run_fw_bw_and_get_code(lambda: f(a, b))
        fwd_code_og = code_og[1][0]
        FileCheck().check("return (buf1, buf0, )").run(fwd_code_og)

        # CASE 2 -- custom partitioner function
        # `a` and `b` (i.e., `primals_1` and `primals_2`) are saved for backward.
        custom_partitioner_fn = MyCustomPartitionerFn()
        self.assertFalse(custom_partitioner_fn.called)
        self.assertIsNotNone(custom_partitioner_fn.uuid())

        with torch._inductor.config.patch(custom_partitioner_fn=custom_partitioner_fn):
            code_cp = run_fw_bw_and_get_code(lambda: f(a, b))
        fwd_code_cp = code_cp[1][0]
        FileCheck().check("return (buf0, primals_1, primals_2, )").run(fwd_code_cp)

        # make sure the custom partitioner function is indeed invoked
        self.assertTrue(custom_partitioner_fn.called)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests()

```



## High-Level Overview

"""    A custom partitioner function with static_lifetime_input_indices overwrites.

This Python file contains 2 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MyCustomPartitionerFn`, `TestCustomPartitionerFn`

**Functions defined**: `__init__`, `__call__`, `uuid`, `test_custom_partitioner_fn`, `f`

**Key imports**: torch, min_cut_rematerialization_partition, FileCheck, CustomPartitionerFn, get_hash_for_files, TestCase, run_fw_bw_and_get_code, GPU_TYPE, HAS_GPU, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `functorch.compile`: min_cut_rematerialization_partition
- `torch._C`: FileCheck
- `torch._inductor.custom_graph_pass`: CustomPartitionerFn, get_hash_for_files
- `torch._inductor.test_case`: TestCase
- `torch._inductor.utils`: run_fw_bw_and_get_code
- `torch.testing._internal.inductor_utils`: GPU_TYPE, HAS_GPU


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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
python test/inductor/test_custom_partitioner_fn.py
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

- **File Documentation**: `test_custom_partitioner_fn.py_docs.md`
- **Keyword Index**: `test_custom_partitioner_fn.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/inductor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python docs/test/inductor/test_custom_partitioner_fn.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/inductor`):

- [`test_snode_runtime.py_kw.md_docs.md`](./test_snode_runtime.py_kw.md_docs.md)
- [`test_metrics.py_docs.md_docs.md`](./test_metrics.py_docs.md_docs.md)
- [`test_flex_attention.py_kw.md_docs.md`](./test_flex_attention.py_kw.md_docs.md)
- [`test_cuda_repro.py_kw.md_docs.md`](./test_cuda_repro.py_kw.md_docs.md)
- [`test_fxir_backend.py_kw.md_docs.md`](./test_fxir_backend.py_kw.md_docs.md)
- [`test_split_cat_fx_passes.py_kw.md_docs.md`](./test_split_cat_fx_passes.py_kw.md_docs.md)
- [`test_mmdecomp.py_kw.md_docs.md`](./test_mmdecomp.py_kw.md_docs.md)
- [`test_torchinductor_codegen_config_overrides.py_kw.md_docs.md`](./test_torchinductor_codegen_config_overrides.py_kw.md_docs.md)
- [`test_aot_inductor_custom_ops.py_kw.md_docs.md`](./test_aot_inductor_custom_ops.py_kw.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_custom_partitioner_fn.py_docs.md_docs.md`
- **Keyword Index**: `test_custom_partitioner_fn.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
