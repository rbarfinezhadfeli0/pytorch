# Documentation: `docs/test/distributed/test_p2p_ipc.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/test_p2p_ipc.py_docs.md`
- **Size**: 5,270 bytes (5.15 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/test_p2p_ipc.py`

## File Metadata

- **Path**: `test/distributed/test_p2p_ipc.py`
- **Size**: 2,062 bytes (2.01 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

# To run:
# python test/distributed/test_p2p_ipc.py


import torch
from torch.multiprocessing.reductions import reduce_tensor
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import requires_cuda_p2p_access, run_tests


# So that tests are written in device-agnostic way
device_type = "cuda"
device_module = torch.get_device_module(device_type)


@requires_cuda_p2p_access()
class P2PIpcTest(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls):
        return "gloo"

    def _init_device(self) -> None:
        # init and pin the process to the device
        device_module.set_device(self.device)
        torch.empty(1, device=self.device)

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)

    def test_p2p_ipc(self) -> None:
        """
        Test that cross-process P2P access works, by reducing a tensor,
        and then constructing a new tensor from the reduced tensor,
        while modifying the 6-th argument.

        This test is here to help stabilize the P2P share mechanism,
        preventing bc-breakage.
        """
        self._init_device()

        tensor: torch.Tensor

        if self.rank == 0:
            tensor = torch.randn(2333, device=self.device)
            tensor_meta = reduce_tensor(tensor)
            torch.distributed.broadcast_object_list([tensor_meta], src=0)
        else:
            recv_list = [None]
            torch.distributed.broadcast_object_list(recv_list, src=0)
            tensor_meta = recv_list[0]
            func, args = tensor_meta
            args = list(args)
            args[6] = self.rank
            tensor = func(*args)

        torch.distributed.barrier()

        if self.rank == 0:
            tensor.fill_(1)

        device_module.synchronize()
        torch.distributed.barrier()

        assert tensor.allclose(tensor, 1)

        torch.distributed.barrier()


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""        Test that cross-process P2P access works, by reducing a tensor,        and then constructing a new tensor from the reduced tensor,        while modifying the 6-th argument.        This test is here to help stabilize the P2P share mechanism,        preventing bc-breakage.

This Python file contains 1 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `P2PIpcTest`

**Functions defined**: `backend_str`, `_init_device`, `device`, `test_p2p_ipc`

**Key imports**: torch, reduce_tensor, MultiProcContinuousTest, requires_cuda_p2p_access, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.multiprocessing.reductions`: reduce_tensor
- `torch.testing._internal.common_distributed`: MultiProcContinuousTest
- `torch.testing._internal.common_utils`: requires_cuda_p2p_access, run_tests


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
python test/distributed/test_p2p_ipc.py
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
- [`test_c10d_spawn_ucc.py_docs.md`](./test_c10d_spawn_ucc.py_docs.md)
- [`test_c10d_ucc.py_docs.md`](./test_c10d_ucc.py_docs.md)
- [`test_serialization.py_docs.md`](./test_serialization.py_docs.md)
- [`test_nccl.py_docs.md`](./test_nccl.py_docs.md)
- [`test_multi_threaded_pg.py_docs.md`](./test_multi_threaded_pg.py_docs.md)


## Cross-References

- **File Documentation**: `test_p2p_ipc.py_docs.md`
- **Keyword Index**: `test_p2p_ipc.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed`, which is part of the **testing infrastructure**.



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
python docs/test/distributed/test_p2p_ipc.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed`):

- [`test_run.py_kw.md_docs.md`](./test_run.py_kw.md_docs.md)
- [`test_inductor_collectives.py_docs.md_docs.md`](./test_inductor_collectives.py_docs.md_docs.md)
- [`test_control_collectives.py_kw.md_docs.md`](./test_control_collectives.py_kw.md_docs.md)
- [`test_c10d_gloo.py_docs.md_docs.md`](./test_c10d_gloo.py_docs.md_docs.md)
- [`test_collective_utils.py_kw.md_docs.md`](./test_collective_utils.py_kw.md_docs.md)
- [`test_data_parallel.py_kw.md_docs.md`](./test_data_parallel.py_kw.md_docs.md)
- [`test_overlap_bucketing_unit.py_kw.md_docs.md`](./test_overlap_bucketing_unit.py_kw.md_docs.md)
- [`test_c10d_nccl.py_kw.md_docs.md`](./test_c10d_nccl.py_kw.md_docs.md)
- [`test_multi_threaded_pg.py_docs.md_docs.md`](./test_multi_threaded_pg.py_docs.md_docs.md)
- [`argparse_util_test.py_kw.md_docs.md`](./argparse_util_test.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_p2p_ipc.py_docs.md_docs.md`
- **Keyword Index**: `test_p2p_ipc.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
