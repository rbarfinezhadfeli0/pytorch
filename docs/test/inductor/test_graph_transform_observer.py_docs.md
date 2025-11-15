# Documentation: `test/inductor/test_graph_transform_observer.py`

## File Metadata

- **Path**: `test/inductor/test_graph_transform_observer.py`
- **Size**: 2,280 bytes (2.23 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: inductor"]
import glob
import math
import os
import shutil
import tempfile

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FUSED_ATTENTION
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


try:
    import pydot  # noqa: F401

    HAS_PYDOT = True
except ImportError:
    HAS_PYDOT = False


HAS_DOT = shutil.which("dot") is not None


class TestGraphTransformObserver(TestCase):
    def test_sdpa_rewriter(self):
        if not (
            HAS_CUDA_AND_TRITON
            and PLATFORM_SUPPORTS_FUSED_ATTENTION
            and HAS_PYDOT
            and HAS_DOT
        ):
            return

        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(value)
            )

        log_url = tempfile.mkdtemp()
        inductor_config.trace.log_url_for_graph_xform = log_url
        inductor_config.force_disable_caches = True
        compiled_fn = torch.compile(dot_prod_attention, fullgraph=True)

        tensor_shape = (4, 2, 16, 32)
        q = torch.randn(tensor_shape, device="cuda")
        k = torch.randn(tensor_shape, device="cuda")
        v = torch.randn(tensor_shape, device="cuda")
        compiled_fn(q, k, v)

        found_input_svg = False
        found_output_svg = False
        for filepath_object in glob.glob(log_url + "/*"):
            if os.path.isfile(filepath_object):
                if filepath_object.endswith("input_graph.dot"):
                    found_input_svg = True
                elif filepath_object.endswith("output_graph.dot"):
                    found_output_svg = True

        self.assertTrue(found_input_svg)
        self.assertTrue(found_output_svg)


if __name__ == "__main__":
    if IS_LINUX:
        run_tests()

```



## High-Level Overview

"""Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""            return (                torch.matmul(query, key.transpose(-2, -1))                .div(math.sqrt(key.shape[-1]))                .softmax(dim=-1)                .matmul(value)            )        log_url = tempfile.mkdtemp()        inductor_config.trace.log_url_for_graph_xform = log_url

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestGraphTransformObserver`

**Functions defined**: `test_sdpa_rewriter`, `dot_prod_attention`

**Key imports**: glob, math, os, shutil, tempfile, torch, torch._dynamo, torch._inductor.config as inductor_config, run_tests, TestCase, PLATFORM_SUPPORTS_FUSED_ATTENTION


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/inductor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `glob`
- `math`
- `os`
- `shutil`
- `tempfile`
- `torch`
- `torch._dynamo`
- `torch._inductor.config as inductor_config`
- `torch._inductor.test_case`: run_tests, TestCase
- `torch.testing._internal.common_cuda`: PLATFORM_SUPPORTS_FUSED_ATTENTION
- `torch.testing._internal.common_utils`: IS_LINUX
- `torch.testing._internal.inductor_utils`: HAS_CUDA_AND_TRITON
- `pydot  `


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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
python test/inductor/test_graph_transform_observer.py
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

- **File Documentation**: `test_graph_transform_observer.py_docs.md`
- **Keyword Index**: `test_graph_transform_observer.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
