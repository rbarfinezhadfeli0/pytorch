# Documentation: `docs/torch/testing/_internal/distributed/common_state_dict.py_docs.md`

## File Metadata

- **Path**: `docs/torch/testing/_internal/distributed/common_state_dict.py_docs.md`
- **Size**: 9,785 bytes (9.56 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/testing/_internal/distributed/common_state_dict.py`

## File Metadata

- **Path**: `torch/testing/_internal/distributed/common_state_dict.py`
- **Size**: 6,725 bytes (6.57 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
# mypy: allow-untyped-defs

# Owner(s): ["oncall: distributed"]

import copy
from itertools import chain
from typing import Any

import torch
import torch.nn as nn
from torch.distributed._sharded_tensor import ShardedTensor
from torch.distributed._state_dict_utils import _gather_state_dict
from torch.distributed.checkpoint.state_dict import (
    _PG,
    _STATE,
    set_state_dict,
    StateDictOptions,
)
from torch.distributed.tensor import DTensor


class VerifyStateDictMixin:
    def _compare_tensor(self, orig_tensor, dist_tensor, offload_to_cpu=False):
        if isinstance(dist_tensor, (DTensor, ShardedTensor)):
            dist_tensor = _gather_state_dict({"mykey": dist_tensor}).pop("mykey")

        if offload_to_cpu:
            orig_tensor = orig_tensor.cpu()
            dist_tensor = dist_tensor.cpu()
        self.assertTrue(isinstance(dist_tensor, torch.Tensor))
        self.assertTrue(torch.allclose(orig_tensor, dist_tensor))

    def _verify_msd(
        self,
        msd: dict[str, Any],
        dist_msd: dict[str, Any],
        options: StateDictOptions = StateDictOptions(),
        offload_to_cpu=False,
    ) -> None:
        if not options.ignore_frozen_params:
            self.assertEqual(len(msd), len(dist_msd))
        for fqn, param in msd.items():
            dist_param = dist_msd.get(fqn)
            if not options.ignore_frozen_params:
                self.assertIsNotNone(dist_param, f"{fqn=}")
                try:
                    self._compare_tensor(param, dist_param, offload_to_cpu)
                except AssertionError as e:
                    raise AssertionError(
                        f"{fqn} has mismatched value {param} {dist_param}"
                    ) from e
            elif dist_param is None:
                self.assertFalse(param.requires_grad, f"{fqn=}")

    def _verify_osd(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        osd: dict[str, Any],
        dist_osd: dict[str, Any],
    ) -> None:
        params = list(chain.from_iterable(g["params"] for g in optim.param_groups))
        param_pid_mapping = dict(zip(params, range(len(params)), strict=True))
        fqn_pid_mapping = {}
        for fqn, param in model.named_parameters():
            pid = param_pid_mapping[param]
            fqn_pid_mapping[fqn] = pid
            fqn_pid_mapping[pid] = fqn
        # Check optimizer_state_dict state

        self.assertEqual(len(osd[_STATE]), len(dist_osd[_STATE]))
        for pid, states in osd[_STATE].items():
            fqn = fqn_pid_mapping[pid]
            dist_states = dist_osd[_STATE].get(fqn, None)
            self.assertIsNotNone(dist_states, fqn)
            self.assertEqual(len(states), len(dist_states))
            for key, state in states.items():
                dist_state = states.get(key, None)
                self.assertIsNotNone(dist_state)
                self._compare_tensor(state, dist_state)

        # Check optimizer_state_dict param_group
        old_dist_osd_pg = dist_osd[_PG]
        if len(osd[_PG]) != len(dist_osd[_PG]):
            self.assertTrue(len(dist_osd[_PG]) > len(osd[_PG]))
            new_pg = copy.deepcopy(dist_osd[_PG][0])
            new_pg["params"] = []
            for dist_group in dist_osd[_PG]:
                new_pg["params"].extend(dist_group["params"])
            dist_osd[_PG] = [new_pg]

        self.assertEqual(len(osd[_PG]), len(dist_osd[_PG]))
        for group, dist_group in zip(osd[_PG], dist_osd[_PG], strict=True):
            self.assertEqual(len(group), len(dist_group))
            for key, value in group.items():
                # Below doesn't work because param_groups can have None
                # values.
                # dist_value = dist_group.get(key, None)
                # self.assertIsNotNone(dist_value, (dist_group, group))
                dist_value = dist_group[key]
                if key == "params":
                    fqns = [fqn_pid_mapping[pid] for pid in value]
                    self.assertEqual(sorted(fqns), sorted(dist_value))
                else:
                    self.assertEqual(value, dist_value)
        dist_osd[_PG] = old_dist_osd_pg

    def _verify_osd_by_load(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        new_optim: torch.optim.Optimizer,
        dist_osd: dict[str, Any],
    ) -> None:
        new_dist_osd = _gather_state_dict(dist_osd)
        set_state_dict(
            model,
            optimizers=new_optim,
            model_state_dict={},
            optim_state_dict=new_dist_osd,
        )
        self.assertEqual(optim.state_dict(), new_optim.state_dict())


class FusionEmbedding(nn.Module):
    def __init__(self, vocab_size: int, fusion_vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fusion_embedding = nn.Embedding(fusion_vocab_size, embed_dim)


class FusionEmbeddingWithHook(nn.Module):
    def __init__(self, vocab_size: int, fusion_vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fusion_embedding = nn.Embedding(fusion_vocab_size, embed_dim)
        self._register_state_dict_hook(FusionEmbeddingWithHook._state_dict_hook)
        self._register_load_state_dict_pre_hook(
            FusionEmbeddingWithHook._load_state_dict_hook, with_module=True
        )

    def _state_dict_hook(self, destination, prefix, keep_vars):
        """Remove "embedding" from the original embedding in the state_dict
        name. This keeps the original state dict name for the embedding
        from before fusing with the FusionEmbedding.
        """
        key = prefix + "embedding.weight"
        new_key = prefix + "weight"
        destination[new_key] = destination[key]
        del destination[key]

    def _load_state_dict_hook(self, state_dict, prefix, *args, **kwargs):
        """Apply extra "embedding" prefix to the state_dict key to
        account for the FusionEmbedding wrapping.
        """
        if state_dict:
            key = prefix + "weight"
            new_key = prefix + "embedding.weight"
            state_dict[new_key] = state_dict[key]
            del state_dict[key]


class FusionEmbeddingWithModifier(FusionEmbeddingWithHook):
    # _fqn_modifiers is a private function as a contract between DSD. When users change the state_dict
    # keys, they need to provide a mapping from the new key to the original key. This is used to ensure
    # consistency between the state_dict keys and fqn.
    def _fqn_modifiers(self) -> dict[str, str]:
        return {
            "weight": "embedding",
        }

```



## High-Level Overview


This Python file contains 4 class(es) and 9 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `VerifyStateDictMixin`, `FusionEmbedding`, `FusionEmbeddingWithHook`, `FusionEmbeddingWithModifier`

**Functions defined**: `_compare_tensor`, `_verify_msd`, `_verify_osd`, `_verify_osd_by_load`, `__init__`, `__init__`, `_state_dict_hook`, `_load_state_dict_hook`, `_fqn_modifiers`

**Key imports**: copy, chain, Any, torch, torch.nn as nn, ShardedTensor, _gather_state_dict, DTensor


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/testing/_internal/distributed`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `itertools`: chain
- `typing`: Any
- `torch`
- `torch.nn as nn`
- `torch.distributed._sharded_tensor`: ShardedTensor
- `torch.distributed._state_dict_utils`: _gather_state_dict
- `torch.distributed.tensor`: DTensor


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python torch/testing/_internal/distributed/common_state_dict.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/testing/_internal/distributed`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`ddp_under_dist_autograd_test.py_docs.md`](./ddp_under_dist_autograd_test.py_docs.md)
- [`checkpoint_utils.py_docs.md`](./checkpoint_utils.py_docs.md)
- [`fake_pg.py_docs.md`](./fake_pg.py_docs.md)
- [`multi_threaded_pg.py_docs.md`](./multi_threaded_pg.py_docs.md)
- [`distributed_utils.py_docs.md`](./distributed_utils.py_docs.md)
- [`rpc_utils.py_docs.md`](./rpc_utils.py_docs.md)
- [`distributed_test.py_docs.md`](./distributed_test.py_docs.md)


## Cross-References

- **File Documentation**: `common_state_dict.py_docs.md`
- **Keyword Index**: `common_state_dict.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/testing/_internal/distributed`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/testing/_internal/distributed`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
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
python docs/torch/testing/_internal/distributed/common_state_dict.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/testing/_internal/distributed`):

- [`ddp_under_dist_autograd_test.py_kw.md_docs.md`](./ddp_under_dist_autograd_test.py_kw.md_docs.md)
- [`ddp_under_dist_autograd_test.py_docs.md_docs.md`](./ddp_under_dist_autograd_test.py_docs.md_docs.md)
- [`multi_threaded_pg.py_docs.md_docs.md`](./multi_threaded_pg.py_docs.md_docs.md)
- [`distributed_utils.py_kw.md_docs.md`](./distributed_utils.py_kw.md_docs.md)
- [`distributed_utils.py_docs.md_docs.md`](./distributed_utils.py_docs.md_docs.md)
- [`distributed_test.py_docs.md_docs.md`](./distributed_test.py_docs.md_docs.md)
- [`checkpoint_utils.py_docs.md_docs.md`](./checkpoint_utils.py_docs.md_docs.md)
- [`common_state_dict.py_kw.md_docs.md`](./common_state_dict.py_kw.md_docs.md)
- [`rpc_utils.py_docs.md_docs.md`](./rpc_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `common_state_dict.py_docs.md_docs.md`
- **Keyword Index**: `common_state_dict.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
