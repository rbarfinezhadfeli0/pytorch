# Documentation: `docs/test/distributed/test_c10d_spawn_nccl.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/test_c10d_spawn_nccl.py_docs.md`
- **Size**: 10,971 bytes (10.71 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/test_c10d_spawn_nccl.py`

## File Metadata

- **Path**: `test/distributed/test_c10d_spawn_nccl.py`
- **Size**: 7,676 bytes (7.50 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]


from test_c10d_spawn import _torch_dist_nn_available, TestDistributedNNFunctions

import torch
import torch.distributed as c10d
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
)


NO_NCCL = not hasattr(c10d, "ProcessGroupNCCL")

# Fails on Python-3.9, see https://github.com/pytorch/pytorch/issues/51619


# Skip dev-asan as torch + multiprocessing spawn have known issues
if not TEST_WITH_DEV_DBG_ASAN:

    class TestDistributedNNFunctionsNccl(TestDistributedNNFunctions):
        # Test Common Ops First.
        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_broadcast(self):
            self._test_broadcast("nccl")

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_reduce(self):
            self._test_reduce("nccl")

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_allreduce(self):
            self._test_allreduce("nccl")

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_gather(self):
            self._test_all_gather("nccl")

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_to_all(self):
            self._test_all_to_all("nccl")

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_to_all_single(self):
            self._test_all_to_all_single("nccl")

        # Test Ops only supported in NCCL.
        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_reduce_scatter(self):
            store = c10d.FileStore(self.file_name, self.world_size)
            # This is required because these functions calls directly to the .dist and needs
            # the world to be initialized
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )
            device = torch.device(f"cuda:{self.rank}")
            x0 = torch.ones(5, 5, device=device) + self.rank
            x1 = torch.ones(5, 5, device=device) + self.rank + 1
            x0.requires_grad = True
            x1.requires_grad = True
            y = torch.empty_like(x0)
            expected = (
                1 + self.world_size
            ) * self.world_size / 2 + self.world_size * self.rank
            y = torch.distributed.nn.reduce_scatter(y, [x0, x1])
            self.assertEqual(y, torch.ones(5, 5, device=device) * expected)
            z = y.sin().sum()
            z.backward()
            expected_0 = (1 + self.world_size) * self.world_size / 2
            expected_1 = expected_0 + self.world_size
            x_s_0 = (expected_0 * torch.ones(5, 5, device=device)).cos()
            x_s_1 = (expected_1 * torch.ones(5, 5, device=device)).cos()
            self.assertEqual(x0.grad, x_s_0)
            self.assertEqual(x1.grad, x_s_1)

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_reduce_scatter_non_contiguous(self):
            store = c10d.FileStore(self.file_name, self.world_size)
            # This is required because these functions calls directly to the .dist and needs
            # the world to be initialized
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )
            device = torch.device(f"cuda:{self.rank}")

            class NonContiguousGrad(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input):
                    return input

                @staticmethod
                def backward(ctx, grad_output):
                    # Make grad non-contiguous
                    return grad_output.clone().transpose(0, 1)

            x0 = torch.rand(5, 5, device=device, requires_grad=True)
            x1 = torch.rand(5, 5, device=device, requires_grad=True)
            y = torch.empty(5, 5, device=device)

            y = torch.distributed.nn.reduce_scatter(y, [x0, x1])
            NonContiguousGrad.apply(y).sum().backward()

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_reduce_non_contiguous(self):
            store = c10d.FileStore(self.file_name, self.world_size)
            # This is required because these functions calls directly to the .dist and needs
            # the world to be initialized
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )
            device = torch.device(f"cuda:{self.rank}")

            class NonContiguousGrad(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input):
                    return input

                @staticmethod
                def backward(ctx, grad_output):
                    # Make grad non-contiguous
                    return grad_output.clone().transpose(0, 1)

            x = torch.rand(5, 5, device=device, requires_grad=True)
            y = torch.distributed.nn.all_reduce(x)
            NonContiguousGrad.apply(y).sum().backward()

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_gather_base(self):
            store = c10d.FileStore(self.file_name, self.world_size)
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="nccl"
            )

            device = torch.device(f"cuda:{self.rank}")
            x = torch.ones(5, 5, device=device) + self.rank
            x.requires_grad = True

            output = torch.empty(5 * self.world_size, 5, device=device)
            output = torch.distributed.nn.functional._all_gather_base(output, x)
            self.assertEqual(output.size(), torch.Size((5 * self.world_size, 5)))

            for idx in range(self.world_size):
                self.assertEqual(
                    output[5 * idx : 5 * (idx + 1)],
                    torch.ones(5, 5, device=device) + idx,
                )

            y = torch.sum(output.view(self.world_size, 5, 5), axis=0)
            z = y.sin().sum()
            z.backward()

            x_s = 2 * (3 * torch.ones(5, 5, device=device)).cos()
            self.assertEqual(x.grad, x_s)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 3 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestDistributedNNFunctionsNccl`, `NonContiguousGrad`, `NonContiguousGrad`

**Functions defined**: `test_broadcast`, `test_reduce`, `test_allreduce`, `test_all_gather`, `test_all_to_all`, `test_all_to_all_single`, `test_reduce_scatter`, `test_reduce_scatter_non_contiguous`, `forward`, `backward`, `test_all_reduce_non_contiguous`, `forward`, `backward`, `test_all_gather_base`

**Key imports**: _torch_dist_nn_available, TestDistributedNNFunctions, torch, torch.distributed as c10d, requires_nccl, skip_if_lt_x_gpu


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `test_c10d_spawn`: _torch_dist_nn_available, TestDistributedNNFunctions
- `torch`
- `torch.distributed as c10d`
- `torch.testing._internal.common_distributed`: requires_nccl, skip_if_lt_x_gpu


## Code Patterns & Idioms

### Common Patterns

- **Automatic Differentiation**: Uses autograd for gradient computation


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
python test/distributed/test_c10d_spawn_nccl.py
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

- **File Documentation**: `test_c10d_spawn_nccl.py_docs.md`
- **Keyword Index**: `test_c10d_spawn_nccl.py_kw.md`
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

- **Automatic Differentiation**: Uses autograd for gradient computation


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
python docs/test/distributed/test_c10d_spawn_nccl.py_docs.md
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

- **File Documentation**: `test_c10d_spawn_nccl.py_docs.md_docs.md`
- **Keyword Index**: `test_c10d_spawn_nccl.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
