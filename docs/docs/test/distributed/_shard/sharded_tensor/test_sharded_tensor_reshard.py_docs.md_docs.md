# Documentation: `docs/test/distributed/_shard/sharded_tensor/test_sharded_tensor_reshard.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/_shard/sharded_tensor/test_sharded_tensor_reshard.py_docs.md`
- **Size**: 6,384 bytes (6.23 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/_shard/sharded_tensor/test_sharded_tensor_reshard.py`

## File Metadata

- **Path**: `test/distributed/_shard/sharded_tensor/test_sharded_tensor_reshard.py`
- **Size**: 3,572 bytes (3.49 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import sys
from itertools import product

import torch
from torch.distributed._shard import _shard_tensor, sharded_tensor
from torch.distributed._shard.sharding_spec import EnumerableShardingSpec, ShardMetadata
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_st_common import (
    _chunk_sharding_specs_list_for_test,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestReshard(ShardedTensorTestBase):
    def _run_sharded_tensor_reshard(self, sharding_spec, reshard_spec, input_size):
        torch.manual_seed(0)
        local_tensor = torch.rand(*input_size).cuda(self.rank)
        st = _shard_tensor(local_tensor, sharding_spec)
        st_compare = _shard_tensor(local_tensor, reshard_spec)
        st.reshard(reshard_spec)
        self.assertEqual(1, len(st.local_shards()))
        self.assertEqual(1, len(st_compare.local_shards()))
        st_compare._metadata.shards_metadata.sort(
            key=lambda metadata: metadata.placement.rank()
        )
        self.assertEqual(st._metadata, st_compare._metadata)
        self.assertEqual(st.local_tensor(), st_compare.local_tensor())
        self.assertEqual(
            st.local_shards()[0].metadata, st_compare.local_shards()[0].metadata
        )

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_tensor_reshard(self):
        dims = [0, 1]
        for sharding_dim, reshard_dim in product(dims, dims):
            specs = _chunk_sharding_specs_list_for_test(
                [sharding_dim, reshard_dim], seed=5
            )
            spec, reshard_spec = specs[0], specs[1]
            self._run_sharded_tensor_reshard(spec, reshard_spec, [13, 21])
            self._run_sharded_tensor_reshard(spec, reshard_spec, [14, 23])
            self._run_sharded_tensor_reshard(spec, reshard_spec, [15, 26])
            self._run_sharded_tensor_reshard(spec, reshard_spec, [12, 24])

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_tensor_reshard_errors(self):
        specs = _chunk_sharding_specs_list_for_test([0, 1], seed=6)
        spec, reshard_spec = specs[0], specs[1]
        enumerable_sharding_spec = EnumerableShardingSpec(
            [
                ShardMetadata(
                    shard_offsets=[0, 0],
                    shard_sizes=[5, 5],
                    placement="rank:0/cuda:0",
                ),
                ShardMetadata(
                    shard_offsets=[5, 0],
                    shard_sizes=[5, 5],
                    placement="rank:1/cuda:1",
                ),
            ]
        )
        st = sharded_tensor.rand(spec, 24, 12)
        with self.assertRaisesRegex(
            NotImplementedError, "Only ChunkShardingSpec supported for reshard."
        ):
            st.reshard(enumerable_sharding_spec)
        st._local_shards = [st.local_shards()[0], st.local_shards()[0]]
        with self.assertRaisesRegex(
            NotImplementedError, "Only single local shard supported for reshard."
        ):
            st.reshard(reshard_spec)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestReshard`

**Functions defined**: `_run_sharded_tensor_reshard`, `test_sharded_tensor_reshard`, `test_sharded_tensor_reshard_errors`

**Key imports**: sys, product, torch, _shard_tensor, sharded_tensor, EnumerableShardingSpec, ShardMetadata, requires_nccl, skip_if_lt_x_gpu, run_tests, TEST_WITH_DEV_DBG_ASAN


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/_shard/sharded_tensor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `itertools`: product
- `torch`
- `torch.distributed._shard`: _shard_tensor, sharded_tensor
- `torch.distributed._shard.sharding_spec`: EnumerableShardingSpec, ShardMetadata
- `torch.testing._internal.common_distributed`: requires_nccl, skip_if_lt_x_gpu
- `torch.testing._internal.common_utils`: run_tests, TEST_WITH_DEV_DBG_ASAN


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
python test/distributed/_shard/sharded_tensor/test_sharded_tensor_reshard.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/_shard/sharded_tensor`):

- [`test_logger.py_docs.md`](./test_logger.py_docs.md)
- [`test_sharded_tensor.py_docs.md`](./test_sharded_tensor.py_docs.md)


## Cross-References

- **File Documentation**: `test_sharded_tensor_reshard.py_docs.md`
- **Keyword Index**: `test_sharded_tensor_reshard.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/_shard/sharded_tensor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/_shard/sharded_tensor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python docs/test/distributed/_shard/sharded_tensor/test_sharded_tensor_reshard.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/_shard/sharded_tensor`):

- [`test_sharded_tensor.py_kw.md_docs.md`](./test_sharded_tensor.py_kw.md_docs.md)
- [`test_sharded_tensor_reshard.py_kw.md_docs.md`](./test_sharded_tensor_reshard.py_kw.md_docs.md)
- [`test_logger.py_kw.md_docs.md`](./test_logger.py_kw.md_docs.md)
- [`test_sharded_tensor.py_docs.md_docs.md`](./test_sharded_tensor.py_docs.md_docs.md)
- [`test_logger.py_docs.md_docs.md`](./test_logger.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_sharded_tensor_reshard.py_docs.md_docs.md`
- **Keyword Index**: `test_sharded_tensor_reshard.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
