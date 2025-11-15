# Documentation: `test/distributed/_shard/sharded_tensor/ops/test_tensor_ops.py`

## File Metadata

- **Path**: `test/distributed/_shard/sharded_tensor/ops/test_tensor_ops.py`
- **Size**: 4,427 bytes (4.32 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import copy

import torch
import torch.distributed._shard.sharded_tensor as sharded_tensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    TEST_GPU_NUM,
    with_comms,
)


class TestTensorOps(ShardedTensorTestBase):
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_deep_copy(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        st = sharded_tensor.rand(spec, (12, 5))
        copied_st = copy.deepcopy(st)
        self.assertTrue(type(copied_st) is type(st))
        self.assertEqual(copied_st.local_tensor(), st.local_tensor())
        self.assertFalse(copied_st is st)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_inplace_copy(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        st = sharded_tensor.rand(spec, (12, 5))
        ones_st = sharded_tensor.ones(spec, (12, 5))
        self.assertFalse(torch.equal(ones_st, st))
        st.copy_(ones_st)
        self.assertTrue(torch.equal(st, ones_st))

        # no grad inplace_copy should work between two with different requires_grad
        st_with_grad = sharded_tensor.rand(spec, (12, 5), requires_grad=True)
        self.assertTrue(st_with_grad.requires_grad)
        self.assertFalse(ones_st.requires_grad)
        with torch.no_grad():
            st_with_grad.copy_(ones_st)
            self.assertEqual(st_with_grad.local_tensor(), ones_st.local_tensor())

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_clone(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        st = sharded_tensor.rand(spec, (12, 5))
        copied_st = st.clone()
        self.assertTrue(type(copied_st) is type(st))
        self.assertEqual(copied_st.local_tensor(), st.local_tensor())
        self.assertFalse(copied_st is st)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_detach(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        st = sharded_tensor.rand(spec, (12, 5), requires_grad=True)
        local_shards = st.local_shards()
        # before set requires_grad, all local shards should not require grads
        for local_shard in local_shards:
            self.assertTrue(local_shard.tensor.requires_grad)

        detached_st = st.detach()
        self.assertFalse(detached_st.requires_grad)

        for local_shard in detached_st.local_shards():
            self.assertFalse(local_shard.tensor.requires_grad)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_set_requires_grad(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )
        st = sharded_tensor.rand(spec, (12, 5))
        local_shards = st.local_shards()
        # before set requires_grad, all local shards should not require grads
        for local_shard in local_shards:
            self.assertFalse(local_shard.tensor.requires_grad)

        st.requires_grad_()
        self.assertTrue(st.requires_grad)

        for local_shard in local_shards:
            self.assertTrue(local_shard.tensor.requires_grad)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestTensorOps`

**Functions defined**: `test_deep_copy`, `test_inplace_copy`, `test_clone`, `test_detach`, `test_set_requires_grad`

**Key imports**: copy, torch, torch.distributed._shard.sharded_tensor as sharded_tensor, ChunkShardingSpec, requires_nccl, skip_if_lt_x_gpu, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/_shard/sharded_tensor/ops`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `torch`
- `torch.distributed._shard.sharded_tensor as sharded_tensor`
- `torch.distributed._shard.sharding_spec`: ChunkShardingSpec
- `torch.testing._internal.common_distributed`: requires_nccl, skip_if_lt_x_gpu
- `torch.testing._internal.common_utils`: run_tests


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python test/distributed/_shard/sharded_tensor/ops/test_tensor_ops.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/_shard/sharded_tensor/ops`):

- [`test_binary_cmp.py_docs.md`](./test_binary_cmp.py_docs.md)
- [`test_embedding.py_docs.md`](./test_embedding.py_docs.md)
- [`test_init.py_docs.md`](./test_init.py_docs.md)
- [`test_embedding_bag.py_docs.md`](./test_embedding_bag.py_docs.md)


## Cross-References

- **File Documentation**: `test_tensor_ops.py_docs.md`
- **Keyword Index**: `test_tensor_ops.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
