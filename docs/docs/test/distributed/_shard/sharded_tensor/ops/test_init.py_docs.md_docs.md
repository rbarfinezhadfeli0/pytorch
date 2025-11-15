# Documentation: `docs/test/distributed/_shard/sharded_tensor/ops/test_init.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/_shard/sharded_tensor/ops/test_init.py_docs.md`
- **Size**: 7,040 bytes (6.88 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/_shard/sharded_tensor/ops/test_init.py`

## File Metadata

- **Path**: `test/distributed/_shard/sharded_tensor/ops/test_init.py`
- **Size**: 4,036 bytes (3.94 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import sys

import torch
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestShardedTensorNNInit(ShardedTensorTestBase):
    """Testing torch.nn.init functions for ShardedTensor"""

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_init_sharded_tensor_with_uniform(self):
        """Test torch.nn.init.uniform_(ShardedTensor, a, b)"""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 8, 2
        a, b = 10, 20

        seed = 1234
        dtype = torch.double

        st = sharded_tensor.empty(spec, h, w, dtype=dtype)
        self.assertEqual(1, len(st.local_shards()))

        # Clone local tensor to ensure torch.nn.init starts from the same input
        local_tensor_clone = torch.clone(st.local_shards()[0].tensor)
        torch.manual_seed(seed)
        torch.nn.init.uniform_(st, a=a, b=b)

        torch.manual_seed(seed)
        torch.nn.init.uniform_(local_tensor_clone, a=a, b=b)
        self.assertEqual(local_tensor_clone, st.local_shards()[0].tensor)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_init_sharded_tensor_with_normal(self):
        """Test torch.nn.init.normal_(ShardedTensor, mean, std)"""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 8, 2
        mean, std = 10, 5

        seed = 1234
        dtype = torch.double

        st = sharded_tensor.empty(spec, h, w, dtype=dtype)
        self.assertEqual(1, len(st.local_shards()))

        # Clone local tensor to ensure torch.nn.init starts from the same input
        local_tensor_clone = torch.clone(st.local_shards()[0].tensor)
        torch.manual_seed(seed)
        torch.nn.init.normal_(st, mean=mean, std=std)

        torch.manual_seed(seed)
        torch.nn.init.normal_(local_tensor_clone, mean=mean, std=std)
        self.assertEqual(local_tensor_clone, st.local_shards()[0].tensor)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_init_sharded_tensor_with_kaiming_uniform(self):
        """Test torch.nn.init.kaiming_uniform_(ShardedTensor, a, mode, nonlinearit)"""

        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        h, w = 8, 2
        a, mode, nonlinearity = 0, "fan_in", "leaky_relu"

        seed = 1234
        dtype = torch.double

        st = sharded_tensor.empty(spec, h, w, dtype=dtype)
        self.assertEqual(1, len(st.local_shards()))

        # Clone local tensor to ensure torch.nn.init starts from the same input
        local_tensor_clone = torch.clone(st.local_shards()[0].tensor)
        torch.manual_seed(seed)
        torch.nn.init.kaiming_uniform_(st, a=a, mode=mode, nonlinearity=nonlinearity)

        torch.manual_seed(seed)
        torch.nn.init.kaiming_uniform_(
            local_tensor_clone, a=a, mode=mode, nonlinearity=nonlinearity
        )
        self.assertEqual(local_tensor_clone, st.local_shards()[0].tensor)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Testing torch.nn.init functions for ShardedTensor"""    @with_comms    @skip_if_lt_x_gpu(4)    @requires_nccl()    def test_init_sharded_tensor_with_uniform(self):

This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestShardedTensorNNInit`

**Functions defined**: `test_init_sharded_tensor_with_uniform`, `test_init_sharded_tensor_with_normal`, `test_init_sharded_tensor_with_kaiming_uniform`

**Key imports**: sys, torch, sharded_tensor, ChunkShardingSpec, requires_nccl, skip_if_lt_x_gpu, run_tests, TEST_WITH_DEV_DBG_ASAN


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/_shard/sharded_tensor/ops`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `torch`
- `torch.distributed._shard`: sharded_tensor
- `torch.distributed._shard.sharding_spec`: ChunkShardingSpec
- `torch.testing._internal.common_distributed`: requires_nccl, skip_if_lt_x_gpu
- `torch.testing._internal.common_utils`: run_tests, TEST_WITH_DEV_DBG_ASAN


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/_shard/sharded_tensor/ops/test_init.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/_shard/sharded_tensor/ops`):

- [`test_tensor_ops.py_docs.md`](./test_tensor_ops.py_docs.md)
- [`test_binary_cmp.py_docs.md`](./test_binary_cmp.py_docs.md)
- [`test_embedding.py_docs.md`](./test_embedding.py_docs.md)
- [`test_embedding_bag.py_docs.md`](./test_embedding_bag.py_docs.md)


## Cross-References

- **File Documentation**: `test_init.py_docs.md`
- **Keyword Index**: `test_init.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/_shard/sharded_tensor/ops`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/_shard/sharded_tensor/ops`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python docs/test/distributed/_shard/sharded_tensor/ops/test_init.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/_shard/sharded_tensor/ops`):

- [`test_tensor_ops.py_docs.md_docs.md`](./test_tensor_ops.py_docs.md_docs.md)
- [`test_binary_cmp.py_docs.md_docs.md`](./test_binary_cmp.py_docs.md_docs.md)
- [`test_embedding.py_kw.md_docs.md`](./test_embedding.py_kw.md_docs.md)
- [`test_embedding_bag.py_kw.md_docs.md`](./test_embedding_bag.py_kw.md_docs.md)
- [`test_embedding_bag.py_docs.md_docs.md`](./test_embedding_bag.py_docs.md_docs.md)
- [`test_binary_cmp.py_kw.md_docs.md`](./test_binary_cmp.py_kw.md_docs.md)
- [`test_embedding.py_docs.md_docs.md`](./test_embedding.py_docs.md_docs.md)
- [`test_tensor_ops.py_kw.md_docs.md`](./test_tensor_ops.py_kw.md_docs.md)
- [`test_init.py_kw.md_docs.md`](./test_init.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_init.py_docs.md_docs.md`
- **Keyword Index**: `test_init.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
