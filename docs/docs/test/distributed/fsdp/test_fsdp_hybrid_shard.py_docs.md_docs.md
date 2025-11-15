# Documentation: `docs/test/distributed/fsdp/test_fsdp_hybrid_shard.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/fsdp/test_fsdp_hybrid_shard.py_docs.md`
- **Size**: 21,160 bytes (20.66 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/fsdp/test_fsdp_hybrid_shard.py`

## File Metadata

- **Path**: `test/distributed/fsdp/test_fsdp_hybrid_shard.py`
- **Size**: 17,097 bytes (16.70 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import contextlib
import sys
from collections import Counter
from enum import auto, Enum
from functools import partial
from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.distributed_c10d import _rank_not_in_group
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp._init_utils import (
    _init_intra_and_inter_node_groups,
    HYBRID_SHARDING_STRATEGIES,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    DEVICEInitMode,
    FSDPInitMode,
    FSDPTest,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


@contextlib.contextmanager
def patch_allreduce(new_allreduce):
    """
    Patches dist.all_reduce with a new all_reduce and
    restores upon exiting.
    """
    orig_ar = dist.all_reduce
    dist.all_reduce = new_allreduce
    try:
        yield
    finally:
        dist.all_reduce = orig_ar


@contextlib.contextmanager
def patch_reduce_scatter(new_reduce_scatter):
    """
    Patches dist.reduce_scatter_tensor with a new reduce_scatter_tensor and
    restores upon exiting.
    """
    orig_reduce_scatter = dist.reduce_scatter_tensor
    dist.reduce_scatter_tensor = new_reduce_scatter
    try:
        yield
    finally:
        dist.reduce_scatter_tensor = orig_reduce_scatter


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(10, 10)
        self.lin2 = nn.Linear(10, 10)
        self.lin3 = nn.Linear(10, 10)

    def forward(self, x):
        return self.lin3(self.lin2(self.lin1(x)))


class ShardingStrategyMode(Enum):
    ALL_HYBRID_SHARD = auto()
    MIXED_HYBRID_FULL_SHARD = auto()


class TestFSDPHybridShard(FSDPTest):
    @property
    def world_size(self):
        return max(torch.accelerator.device_count(), 2)

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @skip_if_lt_x_gpu(2)
    def test_raises_manual_wrap_hybrid_shard_when_none_policy(self):
        model = MyModel().to(device_type)
        err_ctx = self.assertRaisesRegex(
            ValueError,
            "requires explicit specification of process group or device_mesh.",
        )

        with err_ctx:
            model = FSDP(model, sharding_strategy=ShardingStrategy.HYBRID_SHARD)

        with err_ctx:
            model = FSDP(model, sharding_strategy=ShardingStrategy._HYBRID_SHARD_ZERO2)

    @skip_if_lt_x_gpu(4)
    def test_hsdp_save_load_state_dict(self):
        model = MyModel().to(device_type)
        num_node_devices = torch.accelerator.device_count()
        shard_rank_lists = (
            list(range(num_node_devices // 2)),
            list(range(num_node_devices // 2, num_node_devices)),
        )
        shard_groups = (
            dist.new_group(shard_rank_lists[0]),
            dist.new_group(shard_rank_lists[1]),
        )
        my_shard_group = (
            shard_groups[0] if self.rank in shard_rank_lists[0] else shard_groups[1]
        )
        my_replicate_group = None
        my_rank = self.rank
        # Create groups like (0, 4), (1, 5), (2, 6) etc and assign appropriately
        shard_factor = len(shard_rank_lists[0])
        for i in range(num_node_devices // 2):
            replicate_group_ranks = list(range(i, num_node_devices, shard_factor))
            replicate_group = dist.new_group(replicate_group_ranks)
            if my_rank in replicate_group_ranks:
                my_replicate_group = replicate_group

        fsdp_ctor = partial(
            FSDP,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            use_orig_params=True,
            process_group=(my_shard_group, my_replicate_group),
        )
        model = fsdp_ctor(model)
        optim = torch.optim.AdamW(model.parameters())
        # Initialize optimizer states
        model(torch.randn(2, 10)).sum().backward()
        optim.step()
        shard_g = model.process_group
        replicate_g = model._inter_node_pg
        assert shard_g == my_shard_group
        assert replicate_g == my_replicate_group
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            msd = model.state_dict()
            osd = FSDP.optim_state_dict(model, optim)

        load_model = fsdp_ctor(MyModel().to(device_type))
        load_optim = torch.optim.AdamW(load_model.parameters())
        with FSDP.state_dict_type(load_model, StateDictType.SHARDED_STATE_DICT):
            load_model.load_state_dict(msd)
            FSDP.optim_state_dict_to_load(load_model, load_optim, osd)
        load_optim.load_state_dict(osd)

    @skip_if_lt_x_gpu(4)
    def test_hsdp_sync_module_state(self):
        model = MyModel().to(device_type)
        num_node_devices = torch.accelerator.device_count()
        shard_rank_lists = (
            list(range(num_node_devices // 2)),
            list(range(num_node_devices // 2, num_node_devices)),
        )
        shard_groups = (
            dist.new_group(shard_rank_lists[0]),
            dist.new_group(shard_rank_lists[1]),
        )
        my_shard_group = (
            shard_groups[0] if self.rank in shard_rank_lists[0] else shard_groups[1]
        )
        my_replicate_group = None
        my_rank = self.rank
        # Create groups like (0, 4), (1, 5), (2, 6) etc and assign appropriately
        shard_factor = len(shard_rank_lists[0])
        for i in range(num_node_devices // 2):
            replicate_group_ranks = list(range(i, num_node_devices, shard_factor))
            replicate_group = dist.new_group(replicate_group_ranks)
            if my_rank in replicate_group_ranks:
                my_replicate_group = replicate_group

        nn.init.constant_(model.lin1.weight, self.rank)
        nn.init.constant_(model.lin2.weight, self.rank)
        nn.init.constant_(model.lin3.weight, self.rank)

        fsdp_ctor = partial(
            FSDP,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            use_orig_params=True,
            sync_module_states=True,
            process_group=(my_shard_group, my_replicate_group),
        )
        model = fsdp_ctor(model)

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            self.assertTrue((model.lin1.weight == 0).all())
            self.assertTrue((model.lin2.weight == 0).all())
            self.assertTrue((model.lin3.weight == 0).all())

    @skip_if_lt_x_gpu(2)
    def test_invalid_pg_specification_raises(self):
        pol = ModuleWrapPolicy({nn.Linear})
        model = MyModel().to(device_type)
        with self.assertRaisesRegex(
            ValueError, "Expected process_group to be passed in"
        ):
            model = FSDP(
                model,
                auto_wrap_policy=pol,
                process_group=self.process_group,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            )

    # TODO - add test for ZeRO-2 style sharding ensure params are not
    # resharded after forward.

    @skip_if_lt_x_gpu(2)
    def test_fsdp_hybrid_shard_basic_setup(self):
        """
        Tests basic functionality of HYBRID_SHARD and _HYBRID_SHARD_ZERO2:
            1. Inter and intra-node process groups are correctly setup
            2. Process groups are the same across FSDP wrapped instances
            3. reduce_scatter and allreduce called the expected no. of times
        """
        self.run_subtests(
            {
                "hsdp_sharding_strategy": [
                    ShardingStrategy.HYBRID_SHARD,
                    ShardingStrategy._HYBRID_SHARD_ZERO2,
                ],
                "sharding_strategy_mode": [
                    ShardingStrategyMode.ALL_HYBRID_SHARD,
                    ShardingStrategyMode.MIXED_HYBRID_FULL_SHARD,
                ],
                "use_orig_params": [False, True],
                "use_device_mesh": [False, True],
            },
            self._test_fsdp_hybrid_shard_basic_setup,
        )

    def _test_fsdp_hybrid_shard_basic_setup(
        self,
        hsdp_sharding_strategy: ShardingStrategy,
        sharding_strategy_mode: ShardingStrategyMode,
        use_orig_params: bool,
        use_device_mesh: bool,
    ):
        if use_device_mesh:
            device_mesh = init_device_mesh(device_type, (1, self.world_size))
        else:
            device_mesh = None
        hsdp_model = self._init_hsdp_model(
            hsdp_sharding_strategy,
            sharding_strategy_mode,
            use_orig_params,
            hsdp_device_mesh=device_mesh,
        )
        # All FSDP modules should have state.process_group as the process group over which to
        # shard (default process group), and state._inter_node_pg (process group containing only
        # this rank)
        intra_node_pgs = set()
        inter_node_pgs = set()
        for fsdp_module in hsdp_model.fsdp_modules(hsdp_model):
            # TODO: This needs to be replaced if we deprecate
            # `FSDP.sharding_strategy` to only use the handle one.
            # https://github.com/pytorch/pytorch/issues/90857
            if fsdp_module.sharding_strategy not in HYBRID_SHARDING_STRATEGIES:
                self.assertEqual(
                    sharding_strategy_mode, ShardingStrategyMode.MIXED_HYBRID_FULL_SHARD
                )
                self.assertEqual(
                    fsdp_module.sharding_strategy, ShardingStrategy.FULL_SHARD
                )
                continue
            # process_group should be across the node, which is just the
            # whole world here.
            self.assertEqual(
                dist.get_world_size(fsdp_module.process_group),
                dist.get_world_size(self.process_group),
            )
            intra_node_pgs.add(fsdp_module.process_group)
            inter_node_pg = fsdp_module._inter_node_pg
            inter_node_pgs.add(inter_node_pg)
            self.assertEqual(1, dist.get_world_size(inter_node_pg))
            self.assertFalse(_rank_not_in_group(inter_node_pg))
            self.assertEqual(hsdp_sharding_strategy, fsdp_module.sharding_strategy)
        # All fsdp modules should share the same process groups
        self.assertEqual(1, len(intra_node_pgs))
        self.assertEqual(1, len(inter_node_pgs))

        orig_ar = dist.all_reduce
        orig_rs = dist.reduce_scatter_tensor

        def patched_collective(orig_collective, counter, *args, **kwargs):
            counter[orig_collective] += 1
            return orig_collective(*args, **kwargs)

        cntr = Counter()
        patched_allreduce = partial(patched_collective, orig_ar, cntr)
        patched_reduce_scatter = partial(patched_collective, orig_rs, cntr)
        with (
            patch_allreduce(patched_allreduce),
            patch_reduce_scatter(patched_reduce_scatter),
        ):
            inp = hsdp_model.get_input(device=torch.accelerator.current_device_index())
            out = hsdp_model(inp[0], inp[1])
            loss = hsdp_model.get_loss(inp, out)
            loss.backward()

        if sharding_strategy_mode == ShardingStrategyMode.ALL_HYBRID_SHARD:
            num_flat_params = len(list(traversal_utils._get_fsdp_handles(hsdp_model)))
            self.assertEqual(num_flat_params, cntr[orig_ar])
            self.assertEqual(num_flat_params, cntr[orig_rs])
        elif sharding_strategy_mode == ShardingStrategyMode.MIXED_HYBRID_FULL_SHARD:
            num_hsdp_flat_params = len(
                list(traversal_utils._get_fsdp_handles(hsdp_model.transformer))
            )
            num_flat_params = len(list(traversal_utils._get_fsdp_handles(hsdp_model)))
            self.assertEqual(num_hsdp_flat_params, cntr[orig_ar])
            self.assertEqual(num_flat_params, cntr[orig_rs])

    @skip_if_lt_x_gpu(4)
    def test_fsdp_hybrid_shard_parity(self):
        self.run_subtests(
            {
                "hsdp_sharding_strategy": [
                    ShardingStrategy.HYBRID_SHARD,
                    ShardingStrategy._HYBRID_SHARD_ZERO2,
                ],
                "use_orig_params": [False, True],
            },
            self._test_fsdp_hybrid_shard_parity,
        )

    def _test_fsdp_hybrid_shard_parity(
        self, hsdp_sharding_strategy: ShardingStrategy, use_orig_params: bool
    ):
        fsdp_model = self._init_fsdp_model(use_orig_params)
        global_pg = dist.distributed_c10d._get_default_group()
        hsdp_pgs = _init_intra_and_inter_node_groups(global_pg, 2)
        hsdp_model = self._init_hsdp_model(
            hsdp_sharding_strategy,
            ShardingStrategyMode.ALL_HYBRID_SHARD,
            use_orig_params,
            hsdp_process_groups=hsdp_pgs,
        )
        assert hsdp_model._inter_node_pg.size() > 1, (
            "HSDP model initialized without replication"
        )
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=1e-2)
        hsdp_optim = torch.optim.Adam(hsdp_model.parameters(), lr=1e-2)
        torch.manual_seed(global_pg.rank() + 1)
        for _ in range(5):
            inp = fsdp_model.module.get_input(torch.device(device_type))
            losses: list[torch.Tensor] = []
            for model, optim in ((fsdp_model, fsdp_optim), (hsdp_model, hsdp_optim)):
                optim.zero_grad()
                loss = model(*inp).sum()
                losses.append(loss)
                loss.backward()
                optim.step()
            self.assertEqual(losses[0], losses[1])

    def _init_fsdp_model(self, use_orig_params: bool) -> nn.Module:
        auto_wrap_policy = ModuleWrapPolicy(
            {TransformerEncoderLayer, TransformerDecoderLayer},
        )
        hsdp_kwargs = {
            "auto_wrap_policy": auto_wrap_policy,
            "device_id": torch.accelerator.current_device_index(),
            "use_orig_params": use_orig_params,
        }
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_BEFORE,
            hsdp_kwargs,
            deterministic=True,
        )
        return fsdp_model

    def _init_hsdp_model(
        self,
        hsdp_sharding_strategy: ShardingStrategy,
        sharding_strategy_mode: str,
        use_orig_params: bool,
        hsdp_process_groups: Optional[
            tuple[dist.ProcessGroup, dist.ProcessGroup]
        ] = None,
        hsdp_device_mesh: Optional = None,
    ):
        assert hsdp_process_groups is None or hsdp_device_mesh is None
        auto_wrap_policy = ModuleWrapPolicy(
            {TransformerEncoderLayer, TransformerDecoderLayer},
        )
        hsdp_kwargs = {
            "device_id": torch.accelerator.current_device_index(),
            "auto_wrap_policy": auto_wrap_policy,
            "sharding_strategy": hsdp_sharding_strategy,
            "use_orig_params": use_orig_params,
            "device_mesh": hsdp_device_mesh,
        }
        if sharding_strategy_mode == ShardingStrategyMode.ALL_HYBRID_SHARD:
            hsdp_model = TransformerWithSharedParams.init(
                hsdp_process_groups or self.process_group,
                FSDPInitMode.RECURSIVE,
                DEVICEInitMode.DEVICE_BEFORE,
                hsdp_kwargs,
                deterministic=True,
            )
        elif sharding_strategy_mode == ShardingStrategyMode.MIXED_HYBRID_FULL_SHARD:
            model = TransformerWithSharedParams.init(
                hsdp_process_groups or self.process_group,
                FSDPInitMode.NO_FSDP,
                DEVICEInitMode.DEVICE_BEFORE,
                {},
                deterministic=True,
            )
            # Use the HSDP strategy for the transformer module
            model.transformer = FSDP(model.transformer, **hsdp_kwargs)
            # Use `FULL_SHARD` for the embedding and output projection
            hsdp_model = FSDP(
                model,
                device_id=torch.accelerator.current_device_index(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                use_orig_params=use_orig_params,
            )
        return hsdp_model


instantiate_parametrized_tests(TestFSDPHybridShard)

if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 3 class(es) and 17 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MyModel`, `ShardingStrategyMode`, `TestFSDPHybridShard`

**Functions defined**: `patch_allreduce`, `patch_reduce_scatter`, `__init__`, `forward`, `world_size`, `process_group`, `test_raises_manual_wrap_hybrid_shard_when_none_policy`, `test_hsdp_save_load_state_dict`, `test_hsdp_sync_module_state`, `test_invalid_pg_specification_raises`, `test_fsdp_hybrid_shard_basic_setup`, `_test_fsdp_hybrid_shard_basic_setup`, `patched_collective`, `test_fsdp_hybrid_shard_parity`, `_test_fsdp_hybrid_shard_parity`, `_init_fsdp_model`, `_init_hsdp_model`

**Key imports**: contextlib, sys, Counter, auto, Enum, partial, Optional, torch, torch.distributed as dist, torch.distributed.fsdp._traversal_utils as traversal_utils, torch.nn as nn


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/fsdp`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `sys`
- `collections`: Counter
- `enum`: auto, Enum
- `functools`: partial
- `typing`: Optional
- `torch`
- `torch.distributed as dist`
- `torch.distributed.fsdp._traversal_utils as traversal_utils`
- `torch.nn as nn`
- `torch.distributed.device_mesh`: init_device_mesh
- `torch.distributed.distributed_c10d`: _rank_not_in_group
- `torch.distributed.fsdp.wrap`: ModuleWrapPolicy
- `torch.nn`: TransformerDecoderLayer, TransformerEncoderLayer
- `torch.testing._internal.common_distributed`: skip_if_lt_x_gpu


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
python test/distributed/fsdp/test_fsdp_hybrid_shard.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/fsdp`):

- [`test_fsdp_memory.py_docs.md`](./test_fsdp_memory.py_docs.md)
- [`test_fsdp_mixed_precision.py_docs.md`](./test_fsdp_mixed_precision.py_docs.md)
- [`test_fsdp_uneven.py_docs.md`](./test_fsdp_uneven.py_docs.md)
- [`test_fsdp_dtensor_state_dict.py_docs.md`](./test_fsdp_dtensor_state_dict.py_docs.md)
- [`test_fsdp_tp_integration.py_docs.md`](./test_fsdp_tp_integration.py_docs.md)
- [`test_distributed_checkpoint.py_docs.md`](./test_distributed_checkpoint.py_docs.md)
- [`test_fsdp_multiple_forward.py_docs.md`](./test_fsdp_multiple_forward.py_docs.md)
- [`test_checkpoint_wrapper.py_docs.md`](./test_checkpoint_wrapper.py_docs.md)
- [`test_fsdp_clip_grad_norm.py_docs.md`](./test_fsdp_clip_grad_norm.py_docs.md)
- [`test_fsdp_use_orig_params.py_docs.md`](./test_fsdp_use_orig_params.py_docs.md)


## Cross-References

- **File Documentation**: `test_fsdp_hybrid_shard.py_docs.md`
- **Keyword Index**: `test_fsdp_hybrid_shard.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/fsdp`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/fsdp`, which is part of the **testing infrastructure**.



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
python docs/test/distributed/fsdp/test_fsdp_hybrid_shard.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/fsdp`):

- [`test_fsdp_grad_acc.py_docs.md_docs.md`](./test_fsdp_grad_acc.py_docs.md_docs.md)
- [`test_fsdp_ignored_modules.py_kw.md_docs.md`](./test_fsdp_ignored_modules.py_kw.md_docs.md)
- [`test_fsdp_meta.py_kw.md_docs.md`](./test_fsdp_meta.py_kw.md_docs.md)
- [`test_fsdp_apply.py_docs.md_docs.md`](./test_fsdp_apply.py_docs.md_docs.md)
- [`test_fsdp_tp_integration.py_kw.md_docs.md`](./test_fsdp_tp_integration.py_kw.md_docs.md)
- [`test_fsdp_fx.py_docs.md_docs.md`](./test_fsdp_fx.py_docs.md_docs.md)
- [`test_fsdp_memory.py_kw.md_docs.md`](./test_fsdp_memory.py_kw.md_docs.md)
- [`test_fsdp_apply.py_kw.md_docs.md`](./test_fsdp_apply.py_kw.md_docs.md)
- [`test_fsdp_tp_integration.py_docs.md_docs.md`](./test_fsdp_tp_integration.py_docs.md_docs.md)
- [`test_fsdp_multiple_forward.py_kw.md_docs.md`](./test_fsdp_multiple_forward.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_fsdp_hybrid_shard.py_docs.md_docs.md`
- **Keyword Index**: `test_fsdp_hybrid_shard.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
