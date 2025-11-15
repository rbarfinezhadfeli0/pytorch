# Documentation: `docs/torch/testing/_internal/distributed/rpc/examples/parameter_server_test.py_docs.md`

## File Metadata

- **Path**: `docs/torch/testing/_internal/distributed/rpc/examples/parameter_server_test.py_docs.md`
- **Size**: 7,428 bytes (7.25 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This file contains **examples or benchmarks**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `torch/testing/_internal/distributed/rpc/examples/parameter_server_test.py`

## File Metadata

- **Path**: `torch/testing/_internal/distributed/rpc/examples/parameter_server_test.py`
- **Size**: 4,568 bytes (4.46 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file contains **examples or benchmarks**. This appears to be a **test file**.

## Original Source

```python
# mypy: allow-untyped-defs

# If you need to modify this file to make this test pass, please also apply same edits accordingly to
# https://github.com/pytorch/examples/blob/master/distributed/rpc/batch/parameter_server.py
# and https://pytorch.org/tutorials/intermediate/rpc_async_execution.html#batch-updating-parameter-server

import threading
from datetime import datetime
from time import perf_counter

import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch import optim
from torch.testing._internal.dist_utils import dist_init, worker_name
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


batch_size = 20
in_features = 100
out_features = 30
num_batches = 4


def timed_log(text):
    print(f"{datetime.now().strftime('%H:%M:%S')} {text}")


class BatchUpdateParameterServer:
    def __init__(self, batch_update_size):
        self.model = nn.Linear(in_features, out_features)
        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()
        self.batch_update_size = batch_update_size
        self.curr_update_size = 0
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p)

    def get_model(self):
        return self.model

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(ps_rref, grads):
        self = ps_rref.local_value()
        for p, g in zip(self.model.parameters(), grads, strict=True):
            if p.grad is None:
                p.grad = g
            else:
                p.grad += g
        with self.lock:
            timed_log(
                f"PS got {self.curr_update_size}/{self.batch_update_size} updates"
            )
            self.curr_update_size += 1
            fut = self.future_model

            if self.curr_update_size >= self.batch_update_size:
                for p in self.model.parameters():
                    p.grad /= self.batch_update_size
                self.curr_update_size = 0
                self.optimizer.step()
                self.optimizer.zero_grad()
                fut.set_result(self.model)
                timed_log("PS updated model")
                self.future_model = torch.futures.Future()

        return fut


class Trainer:
    def __init__(self, ps_rref):
        self.ps_rref = ps_rref
        self.loss_fn = nn.L1Loss()

    def get_next_batch(self):
        for _ in range(num_batches):
            inputs = torch.randn(batch_size, in_features)
            labels = torch.zeros(batch_size, out_features)
            yield inputs, labels

    def train(self):
        name = rpc.get_worker_info().name
        m = self.ps_rref.rpc_sync().get_model()
        for inputs, labels in self.get_next_batch():
            timed_log(f"{name} processing one batch")
            self.loss_fn(m(inputs), labels).backward()
            timed_log(f"{name} reporting grads")
            m = rpc.rpc_sync(
                self.ps_rref.owner(),
                BatchUpdateParameterServer.update_and_fetch_model,
                args=(self.ps_rref, [p.grad for p in m.cpu().parameters()]),
            )
            timed_log(f"{name} got updated model")


def run_trainer(ps_rref):
    trainer = Trainer(ps_rref)
    trainer.train()


def run_ps(trainers):
    timed_log("Start training")
    start = perf_counter()
    ps_rref = rpc.RRef(BatchUpdateParameterServer(len(trainers)))
    futs = [
        rpc.rpc_async(trainer, run_trainer, args=(ps_rref,)) for trainer in trainers
    ]

    torch.futures.wait_all(futs)
    stop = perf_counter()
    timed_log("Finish training")
    timed_log(f"Time spent training: {stop - start}s")


class ParameterServerTest(RpcAgentTestFixture):
    @dist_init(setup_rpc=False)
    def test_batch_updating_parameter_server(self):
        if self.rank != 0:
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )
        else:
            rpc.init_rpc(
                name=worker_name(self.rank),
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )
            run_ps([f"{worker_name(r)}" for r in range(1, self.world_size)])

        rpc.shutdown()

```



## High-Level Overview


This Python file contains 3 class(es) and 10 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `BatchUpdateParameterServer`, `Trainer`, `ParameterServerTest`

**Functions defined**: `timed_log`, `__init__`, `get_model`, `update_and_fetch_model`, `__init__`, `get_next_batch`, `train`, `run_trainer`, `run_ps`, `test_batch_updating_parameter_server`

**Key imports**: threading, datetime, perf_counter, torch, torch.distributed.rpc as rpc, torch.nn as nn, optim, dist_init, worker_name


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/testing/_internal/distributed/rpc/examples`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `threading`
- `datetime`: datetime
- `time`: perf_counter
- `torch`
- `torch.distributed.rpc as rpc`
- `torch.nn as nn`
- `torch.testing._internal.dist_utils`: dist_init, worker_name


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
python torch/testing/_internal/distributed/rpc/examples/parameter_server_test.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/testing/_internal/distributed/rpc/examples`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`reinforcement_learning_rpc_test.py_docs.md`](./reinforcement_learning_rpc_test.py_docs.md)


## Cross-References

- **File Documentation**: `parameter_server_test.py_docs.md`
- **Keyword Index**: `parameter_server_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/testing/_internal/distributed/rpc/examples`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/testing/_internal/distributed/rpc/examples`, which is part of the **core PyTorch library**.



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
python docs/torch/testing/_internal/distributed/rpc/examples/parameter_server_test.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/testing/_internal/distributed/rpc/examples`):

- [`reinforcement_learning_rpc_test.py_docs.md_docs.md`](./reinforcement_learning_rpc_test.py_docs.md_docs.md)
- [`reinforcement_learning_rpc_test.py_kw.md_docs.md`](./reinforcement_learning_rpc_test.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)
- [`__init__.py_kw.md_docs.md`](./__init__.py_kw.md_docs.md)
- [`parameter_server_test.py_kw.md_docs.md`](./parameter_server_test.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `parameter_server_test.py_docs.md_docs.md`
- **Keyword Index**: `parameter_server_test.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
