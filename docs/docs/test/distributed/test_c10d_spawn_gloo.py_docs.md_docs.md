# Documentation: `docs/test/distributed/test_c10d_spawn_gloo.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/test_c10d_spawn_gloo.py_docs.md`
- **Size**: 12,238 bytes (11.95 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/test_c10d_spawn_gloo.py`

## File Metadata

- **Path**: `test/distributed/test_c10d_spawn_gloo.py`
- **Size**: 8,717 bytes (8.51 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import copy
import os
import tempfile

from test_c10d_spawn import _torch_dist_nn_available, TestDistributedNNFunctions

import torch
import torch.distributed as c10d
import torch.nn as nn
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_distributed import requires_gloo, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)


# Fails on Python-3.9, see https://github.com/pytorch/pytorch/issues/51619


class DistributedDataParallelSingleProcessTest(TestCase):
    def setUp(self):
        self.rank = 0
        self.world_size = 1
        self.file = tempfile.NamedTemporaryFile(delete=False)  # noqa: P201

    def tearDown(self):
        try:
            os.remove(self.file.name)
        except OSError:
            pass

    def _test_base(self, net, inp, check_allclose=True):
        store = c10d.FileStore(self.file.name, self.world_size)
        c10d.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        if inp[0].is_cuda:
            device_ids = [torch.cuda.current_device()]
        else:
            device_ids = None

        ddp = nn.parallel.DistributedDataParallel(
            copy.deepcopy(net), device_ids=device_ids, process_group=process_group
        )

        net_opt = torch.optim.Adam(net.parameters(), lr=0.001)
        ddp_opt = torch.optim.Adam(ddp.parameters(), lr=0.001)

        for i, j in zip(ddp.parameters(), net.parameters()):
            self.assertTrue(i.allclose(j))

        for _ in range(10):
            net_out = net(*inp)
            ddp_out = ddp(*inp)

            net_out.sum().backward()
            ddp_out.sum().backward()

            net_opt.step()
            ddp_opt.step()

        if check_allclose:
            for i, j in zip(ddp.parameters(), net.parameters()):
                self.assertTrue(i.allclose(j))

    @requires_gloo()
    def test_cpu(self):
        self._test_base(nn.Linear(2, 2), [torch.randn(30, 2)])

    @requires_gloo()
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "At least 1 CUDA GPUS needed")
    def test_cuda(self):
        self._test_base(nn.Linear(2, 2).to(0), [torch.randn(30, 2).to(0)])

    @requires_gloo()
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "At least 1 CUDA GPUS needed")
    def test_rnn(self):
        # This test is inspired by the bug reported in
        # https://github.com/pytorch/pytorch/issues/36268
        BATCH_SIZE = 12  # Divisible by 2, 3, 4
        INPUT_DIM = 256
        OUTPUT_DIM = 256
        HIDDEN_DIM = 256
        N_LAYERS = 3
        SEQ_LEN = 100

        class Net(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, hidden_layers):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.output_dim = output_dim
                self.hidden_layers = hidden_layers

                self.lstm = nn.LSTM(
                    input_dim, hidden_dim, hidden_layers, batch_first=True
                )
                self.h2o = nn.Linear(hidden_dim, output_dim)

            def forward(self, x, y):
                self.lstm.flatten_parameters()
                h_t, _ = self.lstm(x)
                output = self.h2o(h_t)
                loss = nn.functional.mse_loss(output, y)
                return loss

        net = Net(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS).to(0)
        inp = [
            torch.randn((BATCH_SIZE, SEQ_LEN, INPUT_DIM)).to(0),
            torch.rand((BATCH_SIZE, SEQ_LEN, OUTPUT_DIM)).to(0),
        ]

        # Not checking result allclose as the parameter inconsistency exist
        # prior to this change. See #37079
        self._test_base(net, inp, check_allclose=False)


# Skip dev-asan as torch + multiprocessing spawn have known issues
if not TEST_WITH_DEV_DBG_ASAN:

    class TestDistributedNNFunctionsGloo(TestDistributedNNFunctions):
        # Test Common Ops First.
        @requires_gloo()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_broadcast(self):
            self._test_broadcast("gloo")

        @requires_gloo()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_reduce(self):
            self._test_reduce("gloo")

        @requires_gloo()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_allreduce(self):
            self._test_allreduce("gloo")

        @requires_gloo()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_gather(self):
            self._test_all_gather("gloo")

        @requires_gloo()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_to_all(self):
            self._test_all_to_all("gloo")

        @requires_gloo()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_to_all_single(self):
            self._test_all_to_all_single("gloo")

        # Test Ops only supported in GLOO.
        @requires_gloo()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_gather(self):
            store = c10d.FileStore(self.file_name, self.world_size)
            # This is required because these functions calls directly to the .dist and needs
            # the world to be initialized
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="gloo"
            )
            device = torch.device(f"cuda:{self.rank}")
            x = torch.ones(5, 5, device=device) + self.rank
            x.requires_grad = True
            tensors = torch.distributed.nn.gather(x, 1)
            if self.rank == 1:
                for i, t in enumerate(tensors):
                    self.assertEqual(t, torch.ones(5, 5, device=device) + i)
            elif self.rank == 0:
                for t in tensors:
                    zeros = torch.zeros(5, 5, device=device)
                    self.assertEqual(t, zeros)
            y = torch.sum(torch.stack(tensors), axis=0)
            z = y.sin().sum()
            z.backward()

            # Test gradient
            x_s = 3 * torch.ones(5, 5, device=device)
            self.assertEqual(x.grad, x_s.cos())

        @requires_gloo()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_scatter(self):
            store = c10d.FileStore(self.file_name, self.world_size)
            # This is required because these functions calls directly to the .dist and needs
            # the world to be initialized
            c10d.init_process_group(
                store=store, rank=self.rank, world_size=self.world_size, backend="gloo"
            )
            device = torch.device(f"cuda:{self.rank}")
            x0 = torch.ones(5, 5, device=device)
            x1 = torch.ones(5, 5, device=device) + 1
            x0.requires_grad = True
            x1.requires_grad = True

            y = torch.distributed.nn.scatter([x0, x1], 1)
            if self.rank == 1:
                self.assertEqual(y, 1 + torch.ones(5, 5, device=device))
            elif self.rank == 0:
                self.assertEqual(y, torch.ones(5, 5, device=device))
            z = y.sin().sum()
            z.backward()

            # Test gradient
            if self.rank == 1:
                x0_s = torch.ones(5, 5, device=device).cos()
                x1_s = (2 * torch.ones(5, 5, device=device)).cos()
                self.assertEqual(x0.grad, x0_s)
                self.assertEqual(x1.grad, x1_s)
            if self.rank == 0:
                self.assertEqual(x0.grad, torch.zeros(5, 5, device=device))


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 3 class(es) and 16 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `DistributedDataParallelSingleProcessTest`, `Net`, `TestDistributedNNFunctionsGloo`

**Functions defined**: `setUp`, `tearDown`, `_test_base`, `test_cpu`, `test_cuda`, `test_rnn`, `__init__`, `forward`, `test_broadcast`, `test_reduce`, `test_allreduce`, `test_all_gather`, `test_all_to_all`, `test_all_to_all_single`, `test_gather`, `test_scatter`

**Key imports**: copy, os, tempfile, _torch_dist_nn_available, TestDistributedNNFunctions, torch, torch.distributed as c10d, torch.nn as nn, TEST_CUDA, requires_gloo, skip_if_lt_x_gpu


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `os`
- `tempfile`
- `test_c10d_spawn`: _torch_dist_nn_available, TestDistributedNNFunctions
- `torch`
- `torch.distributed as c10d`
- `torch.nn as nn`
- `torch.testing._internal.common_cuda`: TEST_CUDA
- `torch.testing._internal.common_distributed`: requires_gloo, skip_if_lt_x_gpu


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
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
python test/distributed/test_c10d_spawn_gloo.py
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

- **File Documentation**: `test_c10d_spawn_gloo.py_docs.md`
- **Keyword Index**: `test_c10d_spawn_gloo.py_kw.md`
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
python docs/test/distributed/test_c10d_spawn_gloo.py_docs.md
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

- **File Documentation**: `test_c10d_spawn_gloo.py_docs.md_docs.md`
- **Keyword Index**: `test_c10d_spawn_gloo.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
