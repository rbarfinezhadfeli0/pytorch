# Documentation: `docs/test/distributed/_shard/sharded_tensor/ops/test_binary_cmp.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/_shard/sharded_tensor/ops/test_binary_cmp.py_docs.md`
- **Size**: 9,204 bytes (8.99 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/_shard/sharded_tensor/ops/test_binary_cmp.py`

## File Metadata

- **Path**: `test/distributed/_shard/sharded_tensor/ops/test_binary_cmp.py`
- **Size**: 5,298 bytes (5.17 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
from torch.distributed._shard import sharded_tensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed.distributed_c10d import _get_default_group
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


class TestShardedTensorBinaryOps(ShardedTensorTestBase):
    """Test base for binary comparison functions such as torch.equal, torch.allclose etc. for ShardedTensor"""

    seed = 42

    def get_random_tensors(
        self, spec1, spec2, *sizes, pg1=None, pg2=None, seed_offset=0
    ):
        pg1 = _get_default_group() if pg1 is None else pg1
        pg2 = _get_default_group() if pg2 is None else pg2
        torch.manual_seed(TestShardedTensorBinaryOps.seed)
        st1 = sharded_tensor.rand(spec1, sizes, process_group=pg1)
        torch.manual_seed(TestShardedTensorBinaryOps.seed + seed_offset)
        st2 = sharded_tensor.rand(spec2, sizes, process_group=pg2)

        TestShardedTensorBinaryOps.seed += 1
        return st1, st2

    def get_gpu_specs(self):
        spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        alt_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:1/cuda:1",
                "rank:0/cuda:0",
                "rank:3/cuda:3",
                "rank:2/cuda:2",
            ],
        )
        return spec, alt_spec

    def _test_common_failures(self, cmp_op):
        spec, alt_spec = self.get_gpu_specs()

        st1, st2 = self.get_random_tensors(spec, spec, 10, 10)
        if self.rank == 0:
            torch.nn.init.uniform_(st1.local_shards()[0].tensor)
        self.assertFalse(cmp_op(st1, st2))

        st1 = sharded_tensor.ones(spec, 10, 10)
        st2 = sharded_tensor.ones(spec, 10, 5)
        self.assertFalse(cmp_op(st1, st2))

        st1, st2 = self.get_random_tensors(spec, alt_spec, 10, 10)
        self.assertFalse(cmp_op(st1, st2))

        st1 = sharded_tensor.ones(spec, 10, 10)
        st2 = sharded_tensor.zeros(spec, 10, 10)
        self.assertFalse(cmp_op(st1, st2))

        st1 = sharded_tensor.ones(spec, 10, 10)
        st2 = sharded_tensor.ones(spec, 10, 10, dtype=torch.double)
        self.assertFalse(cmp_op(st1, st2))

        st1 = sharded_tensor.ones(spec, 10, 10)
        st2 = sharded_tensor.ones(spec, 10, 10, requires_grad=True)
        self.assertFalse(cmp_op(st1, st2))

        cpu_spec = ChunkShardingSpec(
            dim=0,
            placements=[
                "rank:0/cpu",
                "rank:1/cpu",
                "rank:2/cpu",
                "rank:3/cpu",
            ],
        )
        st1 = sharded_tensor.ones(cpu_spec, 10, 10)
        st2 = sharded_tensor.ones(cpu_spec, 10, 10, pin_memory=True)
        self.assertFalse(cmp_op(st1, st2))

        pg = dist.new_group([1, 0, 3, 2])
        st1, st2 = self.get_random_tensors(spec, spec, 10, 10, pg2=pg)
        with self.assertRaisesRegex(
            RuntimeError, "All distributed tensors should use the same ProcessGroup"
        ):
            cmp_op(st1, st2)

        pg = dist.new_group([0, 1, 2, 3])
        st1, st2 = self.get_random_tensors(spec, spec, 10, 10, pg2=pg)
        with self.assertRaisesRegex(
            RuntimeError, "All distributed tensors should use the same ProcessGroup"
        ):
            cmp_op(st1, st2)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_torch_equal_tensor_specs(self):
        self._test_common_failures(torch.equal)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_torch_equal(self):
        """Test torch.equal(ShardedTensor, ShardedTensor)"""

        spec, _ = self.get_gpu_specs()
        st1, st2 = self.get_random_tensors(spec, spec, 10, 10)
        self.assertTrue(torch.equal(st1, st2))

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_torch_allclose_tensor_specs(self):
        self._test_common_failures(torch.allclose)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_torch_allclose(self):
        """Test torch.allclose(ShardedTensor, ShardedTensor)"""

        spec, _ = self.get_gpu_specs()

        st1, st2 = self.get_random_tensors(spec, spec, 10, 10)
        self.assertTrue(torch.allclose(st1, st2))
        self.assertTrue(torch.allclose(st1, st2, atol=0))

        # compare different arrays
        st1, st2 = self.get_random_tensors(spec, spec, 10, 10, seed_offset=1)
        self.assertFalse(torch.allclose(st1, st2))
        # sharded_tensor.rand produces uniform values in the [0,1] range.
        self.assertTrue(torch.allclose(st1, st2, atol=1))


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Test base for binary comparison functions such as torch.equal, torch.allclose etc. for ShardedTensor"""    seed = 42    def get_random_tensors(        self, spec1, spec2, *sizes, pg1=None, pg2=None, seed_offset=0    ):        pg1 = _get_default_group() if pg1 is None else pg1        pg2 = _get_default_group() if pg2 is None else pg2        torch.manual_seed(TestShardedTensorBinaryOps.seed)        st1 = sharded_tensor.rand(spec1, sizes, process_group=pg1)        torch.manual_seed(TestShardedTensorBinaryOps.seed + seed_offset)        st2 = sharded_tensor.rand(spec2, sizes, process_group=pg2)        TestShardedTensorBinaryOps.seed += 1        return st1, st2    def get_gpu_specs(self):        spec = ChunkShardingSpec(            dim=0,            placements=[                "rank:0/cuda:0",                "rank:1/cuda:1",                "rank:2/cuda:2",

This Python file contains 1 class(es) and 7 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestShardedTensorBinaryOps`

**Functions defined**: `get_random_tensors`, `get_gpu_specs`, `_test_common_failures`, `test_torch_equal_tensor_specs`, `test_torch_equal`, `test_torch_allclose_tensor_specs`, `test_torch_allclose`

**Key imports**: sys, torch, torch.distributed as dist, sharded_tensor, ChunkShardingSpec, _get_default_group, requires_nccl, skip_if_lt_x_gpu, run_tests, TEST_WITH_DEV_DBG_ASAN


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/_shard/sharded_tensor/ops`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `torch`
- `torch.distributed as dist`
- `torch.distributed._shard`: sharded_tensor
- `torch.distributed._shard.sharding_spec`: ChunkShardingSpec
- `torch.distributed.distributed_c10d`: _get_default_group
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
python test/distributed/_shard/sharded_tensor/ops/test_binary_cmp.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/_shard/sharded_tensor/ops`):

- [`test_tensor_ops.py_docs.md`](./test_tensor_ops.py_docs.md)
- [`test_embedding.py_docs.md`](./test_embedding.py_docs.md)
- [`test_init.py_docs.md`](./test_init.py_docs.md)
- [`test_embedding_bag.py_docs.md`](./test_embedding_bag.py_docs.md)


## Cross-References

- **File Documentation**: `test_binary_cmp.py_docs.md`
- **Keyword Index**: `test_binary_cmp.py_kw.md`
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
python docs/test/distributed/_shard/sharded_tensor/ops/test_binary_cmp.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/_shard/sharded_tensor/ops`):

- [`test_tensor_ops.py_docs.md_docs.md`](./test_tensor_ops.py_docs.md_docs.md)
- [`test_embedding.py_kw.md_docs.md`](./test_embedding.py_kw.md_docs.md)
- [`test_embedding_bag.py_kw.md_docs.md`](./test_embedding_bag.py_kw.md_docs.md)
- [`test_embedding_bag.py_docs.md_docs.md`](./test_embedding_bag.py_docs.md_docs.md)
- [`test_binary_cmp.py_kw.md_docs.md`](./test_binary_cmp.py_kw.md_docs.md)
- [`test_init.py_docs.md_docs.md`](./test_init.py_docs.md_docs.md)
- [`test_embedding.py_docs.md_docs.md`](./test_embedding.py_docs.md_docs.md)
- [`test_tensor_ops.py_kw.md_docs.md`](./test_tensor_ops.py_kw.md_docs.md)
- [`test_init.py_kw.md_docs.md`](./test_init.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_binary_cmp.py_docs.md_docs.md`
- **Keyword Index**: `test_binary_cmp.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
