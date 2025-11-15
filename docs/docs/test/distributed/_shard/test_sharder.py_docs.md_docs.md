# Documentation: `docs/test/distributed/_shard/test_sharder.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/_shard/test_sharder.py_docs.md`
- **Size**: 9,215 bytes (9.00 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/_shard/test_sharder.py`

## File Metadata

- **Path**: `test/distributed/_shard/test_sharder.py`
- **Size**: 6,366 bytes (6.22 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]
import copy
import sys

import torch
import torch.nn as nn
from torch.distributed._shard import shard_module
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharder import Sharder
from torch.distributed._shard.sharding_plan import ShardingPlan
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    TEST_GPU_NUM,
    with_comms,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


# a simple collection of embedding bag implementation
class CustomEmbeddingBagCollection(nn.Module):
    def __init__(self, num_bags, num_embeddings_per_bag, num_dims):
        super().__init__()
        self.num_bags = num_bags
        self.embedding_bags: nn.ModuleDict = nn.ModuleDict()

        for i in range(num_bags):
            self.embedding_bags[f"embedding_bag_{i}"] = nn.EmbeddingBag(
                num_embeddings_per_bag, num_dims, mode="sum"
            )

    def forward(self, inputs):
        outputs = []
        for bag in self.embedding_bags.values():
            outputs.append(bag(inputs))
        return torch.cat(outputs)


# a simple sharded version of EBC
class CustomShardedEBC(nn.Module):
    def __init__(self, ebc, split_idx, specs):
        super().__init__()
        self.split_idx = split_idx
        row_spec, col_spec = specs

        # create embedding bags base on the spec
        self.embedding_bags: nn.ModuleDict = nn.ModuleDict()

        assert self.split_idx < ebc.num_bags
        for i in range(ebc.num_bags):
            bag_key = f"embedding_bag_{i}"
            if i < self.split_idx:
                shard_module(
                    ebc,
                    plan=ShardingPlan(
                        plan={f"embedding_bags.{bag_key}.weight": row_spec}
                    ),
                )
            else:
                shard_module(
                    ebc,
                    plan=ShardingPlan(
                        plan={f"embedding_bags.{bag_key}.weight": col_spec}
                    ),
                )

            self.embedding_bags[bag_key] = ebc.embedding_bags[bag_key]


class CustomSharder(Sharder):
    def __init__(self, devices, split_sharding_idx):
        self.devices = devices
        self.split_sharding_idx = split_sharding_idx
        self.rowwise_spec = ChunkShardingSpec(dim=0, placements=devices)
        self.colwise_spec = ChunkShardingSpec(dim=1, placements=devices)

    def shard(self, ebc: nn.Module) -> nn.Module:
        if not isinstance(ebc, CustomEmbeddingBagCollection):
            raise RuntimeError(
                "The custom sharder only supports CustomEmbeddingBagCollection"
            )

        return CustomShardedEBC(
            ebc, self.split_sharding_idx, (self.rowwise_spec, self.colwise_spec)
        )


class TestCustomSharder(ShardedTensorTestBase):
    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_custom_sharder(self):
        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ebc = CustomEmbeddingBagCollection(10, 10, 8)

            def forward(self, inputs):
                return self.ebc(inputs)

        custom_sharder = CustomSharder(
            devices=[f"rank:{i}/cuda:{i}" for i in range(TEST_GPU_NUM)],
            split_sharding_idx=TEST_GPU_NUM // 2,
        )

        sharding_plan = ShardingPlan(
            plan={
                "ebc": custom_sharder,
            }
        )

        local_model = MyModule().cuda(self.rank)
        sharded_model = copy.deepcopy(local_model)

        # shard the module with the provided sharding plan
        shard_module(sharded_model, sharding_plan)

        # check to make sure the module already been sharded
        emb_bags = sharded_model.ebc.embedding_bags
        self.assertTrue(isinstance(emb_bags["embedding_bag_0"].weight, ShardedTensor))
        self.assertTrue(isinstance(emb_bags["embedding_bag_9"].weight, ShardedTensor))
        self.assertEqual(
            emb_bags["embedding_bag_0"].weight.sharding_spec(),
            custom_sharder.rowwise_spec,
        )
        self.assertEqual(
            emb_bags["embedding_bag_9"].weight.sharding_spec(),
            custom_sharder.colwise_spec,
        )

        # make sure we can run sharded computation and compare outputs
        # with the local model version
        input = torch.arange(8).reshape((2, 4)).cuda(self.rank)
        local_output = local_model(input)
        sharded_output = sharded_model(input)

        self.assertEqual(local_output, sharded_output)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_custom_sharder_errors(self):
        custom_sharder = CustomSharder(
            devices=[f"rank:{i}/cuda:{i}" for i in range(TEST_GPU_NUM)],
            split_sharding_idx=TEST_GPU_NUM // 2,
        )

        sharding_plan = ShardingPlan(
            plan={
                "": custom_sharder,
            }
        )

        sharded_model = CustomEmbeddingBagCollection(10, 10, 8).cuda(self.rank)

        with self.assertRaisesRegex(
            KeyError, "path must not be empty for custom sharder!"
        ):
            # shard the module with the provided sharding plan
            shard_module(sharded_model, sharding_plan)

        # test conflicted sharding plan
        spec = ChunkShardingSpec(dim=0, placements=["rank:0/cuda:0", "rank:1/cuda:1"])
        sharding_plan = ShardingPlan(
            plan={
                "embedding_bags.embedding_bag_0.weight": spec,
                "embedding_bags": custom_sharder,
            }
        )

        with self.assertRaisesRegex(
            RuntimeError, "should not conflict with the submodule tree"
        ):
            # shard the module with the provided sharding plan
            shard_module(sharded_model, sharding_plan)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 5 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `CustomEmbeddingBagCollection`, `CustomShardedEBC`, `CustomSharder`, `TestCustomSharder`, `MyModule`

**Functions defined**: `__init__`, `forward`, `__init__`, `__init__`, `shard`, `test_custom_sharder`, `__init__`, `forward`, `test_custom_sharder_errors`

**Key imports**: copy, sys, torch, torch.nn as nn, shard_module, ShardedTensor, Sharder, ShardingPlan, ChunkShardingSpec, requires_nccl, skip_if_lt_x_gpu


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/_shard`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `sys`
- `torch`
- `torch.nn as nn`
- `torch.distributed._shard`: shard_module
- `torch.distributed._shard.sharded_tensor`: ShardedTensor
- `torch.distributed._shard.sharder`: Sharder
- `torch.distributed._shard.sharding_plan`: ShardingPlan
- `torch.distributed._shard.sharding_spec`: ChunkShardingSpec
- `torch.testing._internal.common_distributed`: requires_nccl, skip_if_lt_x_gpu
- `torch.testing._internal.common_utils`: run_tests, TEST_WITH_DEV_DBG_ASAN


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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
python test/distributed/_shard/test_sharder.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/_shard`):



## Cross-References

- **File Documentation**: `test_sharder.py_docs.md`
- **Keyword Index**: `test_sharder.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/_shard`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/_shard`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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
python docs/test/distributed/_shard/test_sharder.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/_shard`):

- [`test_sharder.py_kw.md_docs.md`](./test_sharder.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_sharder.py_docs.md_docs.md`
- **Keyword Index**: `test_sharder.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
