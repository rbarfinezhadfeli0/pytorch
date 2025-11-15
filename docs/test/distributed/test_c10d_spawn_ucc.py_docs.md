# Documentation: `test/distributed/test_c10d_spawn_ucc.py`

## File Metadata

- **Path**: `test/distributed/test_c10d_spawn_ucc.py`
- **Size**: 2,493 bytes (2.43 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]


from test_c10d_spawn import _torch_dist_nn_available, TestDistributedNNFunctions

import torch.distributed as c10d
from torch.testing._internal.common_distributed import requires_ucc, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
)


NO_UCC = not hasattr(c10d, "ProcessGroupUCC")

# Fails on Python-3.9, see https://github.com/pytorch/pytorch/issues/51619


# Skip dev-asan as torch + multiprocessing spawn have known issues
if not TEST_WITH_DEV_DBG_ASAN:

    class TestDistributedNNFunctionsUcc(TestDistributedNNFunctions):
        # Test Common Ops First.
        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_broadcast(self):
            self._test_broadcast("ucc")

        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_reduce(self):
            self._test_reduce("ucc")

        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_allreduce(self):
            self._test_allreduce("ucc")

        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        @skip_but_pass_in_sandcastle(
            "runs into illegal memory access on first assertEqual check when run locally"
        )
        def test_all_gather(self):
            self._test_all_gather("ucc")

        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_to_all(self):
            self._test_all_to_all("ucc")

        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_to_all_single(self):
            self._test_all_to_all_single("ucc")


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestDistributedNNFunctionsUcc`

**Functions defined**: `test_broadcast`, `test_reduce`, `test_allreduce`, `test_all_gather`, `test_all_to_all`, `test_all_to_all_single`

**Key imports**: _torch_dist_nn_available, TestDistributedNNFunctions, torch.distributed as c10d, requires_ucc, skip_if_lt_x_gpu


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `test_c10d_spawn`: _torch_dist_nn_available, TestDistributedNNFunctions
- `torch.distributed as c10d`
- `torch.testing._internal.common_distributed`: requires_ucc, skip_if_lt_x_gpu


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
python test/distributed/test_c10d_spawn_ucc.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed`):

- [`test_run.py_docs.md`](./test_run.py_docs.md)
- [`test_c10d_logger.py_docs.md`](./test_c10d_logger.py_docs.md)
- [`test_dist2.py_docs.md`](./test_dist2.py_docs.md)
- [`test_c10d_functional_native.py_docs.md`](./test_c10d_functional_native.py_docs.md)
- [`test_c10d_object_collectives.py_docs.md`](./test_c10d_object_collectives.py_docs.md)
- [`test_c10d_ucc.py_docs.md`](./test_c10d_ucc.py_docs.md)
- [`test_serialization.py_docs.md`](./test_serialization.py_docs.md)
- [`test_nccl.py_docs.md`](./test_nccl.py_docs.md)
- [`test_multi_threaded_pg.py_docs.md`](./test_multi_threaded_pg.py_docs.md)


## Cross-References

- **File Documentation**: `test_c10d_spawn_ucc.py_docs.md`
- **Keyword Index**: `test_c10d_spawn_ucc.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
