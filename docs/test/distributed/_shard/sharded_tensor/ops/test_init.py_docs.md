# Documentation: test_init.py

## File Metadata
- **Path**: `test/distributed/_shard/sharded_tensor/ops/test_init.py`
- **Size**: 4036 bytes
- **Lines**: 130
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 1 class(es): TestShardedTensorNNInit

### Functions
This file defines 3 function(s): test_init_sharded_tensor_with_uniform, test_init_sharded_tensor_with_normal, test_init_sharded_tensor_with_kaiming_uniform


## Key Components

The file contains 275 words across 130 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4036 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
