# Documentation: `test/distributed/_composable/fsdp/test_fully_shard_state.py`

## File Metadata

- **Path**: `test/distributed/_composable/fsdp/test_fully_shard_state.py`
- **Size**: 3,197 bytes (3.12 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import copy

import torch.nn as nn
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTestMultiThread, MLP
from torch.testing._internal.common_utils import run_tests


class TestFullyShardState(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 1

    @skip_if_lt_x_gpu(1)
    def test_fully_shard_state(self):
        """
        Tests the ability to get the state object from a fully sharded module.
        """
        num_mlps = 3
        model = nn.Sequential(*[MLP(8) for _ in range(num_mlps)])
        for mlp in model:
            fully_shard(mlp)
        fully_shard(model)
        root_state = fully_shard.state(model)
        self.assertTrue(root_state is not None)
        all_states = [root_state] + [fully_shard.state(mlp) for mlp in model]
        # Check that each `fully_shard` call constructs a distinct state object
        self.assertEqual(len(set(all_states)), num_mlps + 1)

    @skip_if_lt_x_gpu(1)
    def test_fully_shard_reapply(self):
        model = MLP(8)
        fully_shard(model)
        with self.assertRaisesRegex(
            AssertionError,
            "Each distinct composable distributed API can only be applied to a module once.",
        ):
            fully_shard(model)

    @skip_if_lt_x_gpu(1)
    def test_fully_shard_cls(self):
        # Check that we only swap class for the module passed to `fully_shard`
        model = MLP(8)
        fully_shard(model)
        self.assertTrue(isinstance(model, MLP))
        self.assertTrue(isinstance(model, FSDPModule))
        self.assertEqual(model.__class__.__name__, "FSDPMLP")
        for module in model.modules():
            if module is model:
                continue
            self.assertFalse(isinstance(module, FSDPModule))

        # Check that slicing into a `Sequential` does not preserve FSDP
        model = nn.Sequential(*[MLP(8) for _ in range(3)])
        fully_shard(model)
        self.assertTrue(isinstance(model, nn.Sequential))
        self.assertTrue(isinstance(model, FSDPModule))
        self.assertEqual(model.__class__.__name__, "FSDPSequential")
        sliced_model = model[:2]
        self.assertTrue(isinstance(sliced_model, nn.Sequential))
        self.assertFalse(isinstance(sliced_model, FSDPModule))

    @skip_if_lt_x_gpu(1)
    def test_fully_shard_unsupported_module_cls(self):
        regex = (
            r"fully\_shard does not support containers that do not implement forward"
        )
        model = nn.ModuleList([MLP(8) for _ in range(3)])
        with self.assertRaisesRegex(ValueError, regex):
            fully_shard(model)
        model = nn.ModuleDict({"1": MLP(8), "2": MLP(8)})
        with self.assertRaisesRegex(ValueError, regex):
            fully_shard(model)

    @skip_if_lt_x_gpu(1)
    def test_fully_shard_deepcopy(self):
        model = MLP(8)
        fully_shard(model)
        with self.assertRaisesRegex(AssertionError, "FSDP does not support deepcopy"):
            copy.deepcopy(model)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""        Tests the ability to get the state object from a fully sharded module.

This Python file contains 2 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestFullyShardState`

**Functions defined**: `world_size`, `test_fully_shard_state`, `test_fully_shard_reapply`, `test_fully_shard_cls`, `test_fully_shard_unsupported_module_cls`, `test_fully_shard_deepcopy`

**Key imports**: copy, torch.nn as nn, FSDPModule, fully_shard, skip_if_lt_x_gpu, FSDPTestMultiThread, MLP, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/_composable/fsdp`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `torch.nn as nn`
- `torch.distributed.fsdp`: FSDPModule, fully_shard
- `torch.testing._internal.common_distributed`: skip_if_lt_x_gpu
- `torch.testing._internal.common_fsdp`: FSDPTestMultiThread, MLP
- `torch.testing._internal.common_utils`: run_tests


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
python test/distributed/_composable/fsdp/test_fully_shard_state.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/_composable/fsdp`):

- [`test_fully_shard_extensions.py_docs.md`](./test_fully_shard_extensions.py_docs.md)
- [`test_fully_shard_logging.py_docs.md`](./test_fully_shard_logging.py_docs.md)
- [`test_fully_shard_mixed_precision.py_docs.md`](./test_fully_shard_mixed_precision.py_docs.md)
- [`test_fully_shard_ignore_params.py_docs.md`](./test_fully_shard_ignore_params.py_docs.md)
- [`test_fully_shard_frozen.py_docs.md`](./test_fully_shard_frozen.py_docs.md)
- [`test_fully_shard_clip_grad_norm_.py_docs.md`](./test_fully_shard_clip_grad_norm_.py_docs.md)
- [`test_fully_shard_overlap.py_docs.md`](./test_fully_shard_overlap.py_docs.md)
- [`test_fully_shard_state_dict.py_docs.md`](./test_fully_shard_state_dict.py_docs.md)
- [`test_fully_shard_init.py_docs.md`](./test_fully_shard_init.py_docs.md)


## Cross-References

- **File Documentation**: `test_fully_shard_state.py_docs.md`
- **Keyword Index**: `test_fully_shard_state.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
