# Documentation: `docs/torch/csrc/lazy/test_mnist.py_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/lazy/test_mnist.py_docs.md`
- **Size**: 5,107 bytes (4.99 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `torch/csrc/lazy/test_mnist.py`

## File Metadata

- **Path**: `torch/csrc/lazy/test_mnist.py`
- **Size**: 2,721 bytes (2.66 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# mypy: ignore-errors

import os

from torchvision import datasets, transforms

import torch
import torch._lazy
import torch._lazy.metrics
import torch._lazy.ts_backend
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


torch._lazy.ts_backend.init()


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        torch._lazy.mark_step()

        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]"
                f"\tLoss: {loss.item():.6f}"
            )


if __name__ == "__main__":
    bsz = 64
    device = "lazy"
    epochs = 14
    log_interval = 10
    lr = 1
    gamma = 0.7
    train_kwargs = {"batch_size": bsz}
    # if we want to use CUDA
    if "LTC_TS_CUDA" in os.environ:
        cuda_kwargs = {
            "num_workers": 1,
            "pin_memory": True,
            "shuffle": True,
            "batch_size": bsz,
        }
        train_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(log_interval, model, device, train_loader, optimizer, epoch)
        scheduler.step()

```



## High-Level Overview


This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Net`

**Functions defined**: `__init__`, `forward`, `train`

**Key imports**: os, datasets, transforms, torch, torch._lazy, torch._lazy.metrics, torch._lazy.ts_backend, torch.nn as nn, torch.nn.functional as F, torch.optim as optim, StepLR


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/lazy`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `torchvision`: datasets, transforms
- `torch`
- `torch._lazy`
- `torch._lazy.metrics`
- `torch._lazy.ts_backend`
- `torch.nn as nn`
- `torch.nn.functional as F`
- `torch.optim as optim`
- `torch.optim.lr_scheduler`: StepLR


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
python torch/csrc/lazy/test_mnist.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/lazy`):

- [`tutorial.md_docs.md`](./tutorial.md_docs.md)


## Cross-References

- **File Documentation**: `test_mnist.py_docs.md`
- **Keyword Index**: `test_mnist.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/lazy`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/lazy`, which is part of the **core PyTorch library**.



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
python docs/torch/csrc/lazy/test_mnist.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/lazy`):

- [`tutorial.md_docs.md_docs.md`](./tutorial.md_docs.md_docs.md)
- [`test_mnist.py_kw.md_docs.md`](./test_mnist.py_kw.md_docs.md)
- [`tutorial.md_kw.md_docs.md`](./tutorial.md_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_mnist.py_docs.md_docs.md`
- **Keyword Index**: `test_mnist.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
