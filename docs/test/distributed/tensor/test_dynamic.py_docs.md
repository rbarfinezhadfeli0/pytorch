# Documentation: `test/distributed/tensor/test_dynamic.py`

## File Metadata

- **Path**: `test/distributed/tensor/test_dynamic.py`
- **Size**: 2,174 bytes (2.12 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

from unittest.mock import patch

import torch
from torch.distributed.tensor import distribute_tensor, DTensor
from torch.distributed.tensor.placement_types import Replicate
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    create_local_tensor_test_class,
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_gpu


class TestDynamic(DTensorTestBase):
    @requires_gpu
    @with_comms
    @parametrize("fake_tensor_cache_enabled", [False, True])
    def test_embedding(self, fake_tensor_cache_enabled):
        with patch.object(
            torch._dynamo.config, "fake_tensor_cache_enabled", fake_tensor_cache_enabled
        ):
            device_mesh = self.build_device_mesh()

            placements = (Replicate(),)

            num_embeddings = 202048
            embedding_dim = 256
            weight = distribute_tensor(
                torch.rand(
                    [num_embeddings, embedding_dim],
                    dtype=torch.float32,
                    device=GPU_TYPE,
                    requires_grad=True,
                ),
                device_mesh,
                placements,  # [Replicate()],
            )

            def forward(input_batch_inputs_):
                to = weight.to(torch.float32)
                emb = torch.nn.functional.embedding(input_batch_inputs_, to)
                return emb

            arg0 = torch.randint(
                low=0, high=100, size=(2, 512), dtype=torch.int64, device=GPU_TYPE
            )
            arg0 = DTensor.from_local(arg0, device_mesh, placements)

            compiled_forward = torch.compile(forward, fullgraph=True, dynamic=True)
            _out = compiled_forward(arg0)


instantiate_parametrized_tests(TestDynamic)

TestDynamicWithLocalTensor = create_local_tensor_test_class(
    TestDynamic,
)

if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestDynamic`

**Functions defined**: `test_embedding`, `forward`

**Key imports**: patch, torch, distribute_tensor, DTensor, Replicate, GPU_TYPE, requires_gpu


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/tensor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest.mock`: patch
- `torch`
- `torch.distributed.tensor`: distribute_tensor, DTensor
- `torch.distributed.tensor.placement_types`: Replicate
- `torch.testing._internal.inductor_utils`: GPU_TYPE
- `torch.testing._internal.triton_utils`: requires_gpu


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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
python test/distributed/tensor/test_dynamic.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/tensor`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_dtensor.py_docs.md`](./test_dtensor.py_docs.md)
- [`test_dtensor_testbase.py_docs.md`](./test_dtensor_testbase.py_docs.md)
- [`test_optimizers.py_docs.md`](./test_optimizers.py_docs.md)
- [`test_dtensor_dispatch_overhead.py_docs.md`](./test_dtensor_dispatch_overhead.py_docs.md)
- [`test_tensor_ops.py_docs.md`](./test_tensor_ops.py_docs.md)
- [`test_matrix_ops.py_docs.md`](./test_matrix_ops.py_docs.md)
- [`test_op_schema.py_docs.md`](./test_op_schema.py_docs.md)
- [`test_utils.py_docs.md`](./test_utils.py_docs.md)
- [`test_attention.py_docs.md`](./test_attention.py_docs.md)


## Cross-References

- **File Documentation**: `test_dynamic.py_docs.md`
- **Keyword Index**: `test_dynamic.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
