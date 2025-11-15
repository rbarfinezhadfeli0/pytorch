# Documentation: `functorch/docs/source/tutorials/_src/plot_per_sample_gradients.py`

## File Metadata

- **Path**: `functorch/docs/source/tutorials/_src/plot_per_sample_gradients.py`
- **Size**: 5,065 bytes (4.95 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **documentation**.

## Original Source

```python
"""
==========================
Per-sample-gradients
==========================

What is it?
--------------------------------------------------------------------
Per-sample-gradient computation is computing the gradient for each and every
sample in a batch of data. It is a useful quantity in differential privacy
and optimization research.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(0)


# Here's a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        output = x
        return output


def loss_fn(predictions, targets):
    return F.nll_loss(predictions, targets)


# Let's generate a batch of dummy data. Pretend that we're working with an
# MNIST dataset where the images are 28 by 28 and we have a minibatch of size 64.
device = "cuda"
num_models = 10
batch_size = 64
data = torch.randn(batch_size, 1, 28, 28, device=device)
targets = torch.randint(10, (64,), device=device)

# In regular model training, one would forward the batch of examples and then
# call .backward() to compute gradients:

model = SimpleCNN().to(device=device)
predictions = model(data)
loss = loss_fn(predictions, targets)
loss.backward()


# Conceptually, per-sample-gradient computation is equivalent to: for each sample
# of the data, perform a forward and a backward pass to get a gradient.
def compute_grad(sample, target):
    sample = sample.unsqueeze(0)
    target = target.unsqueeze(0)
    prediction = model(sample)
    loss = loss_fn(prediction, target)
    return torch.autograd.grad(loss, list(model.parameters()))


def compute_sample_grads(data, targets):
    sample_grads = [compute_grad(data[i], targets[i]) for i in range(batch_size)]
    sample_grads = zip(*sample_grads)
    sample_grads = [torch.stack(shards) for shards in sample_grads]
    return sample_grads


per_sample_grads = compute_sample_grads(data, targets)

# sample_grads[0] is the per-sample-grad for model.conv1.weight
# model.conv1.weight.shape is [32, 1, 3, 3]; notice how there is one gradient
# per sample in the batch for a total of 64.
print(per_sample_grads[0].shape)


######################################################################
# Per-sample-grads using functorch
# --------------------------------------------------------------------
# We can compute per-sample-gradients efficiently by using function transforms.
# First, let's create a stateless functional version of ``model`` by using
# ``functorch.make_functional_with_buffers``.
from functorch import grad, make_functional_with_buffers, vmap


fmodel, params, buffers = make_functional_with_buffers(model)


# Next, let's define a function to compute the loss of the model given a single
# input rather than a batch of inputs. It is important that this function accepts the
# parameters, the input, and the target, because we will be transforming over them.
# Because the model was originally written to handle batches, we'll use
# ``torch.unsqueeze`` to add a batch dimension.
def compute_loss(params, buffers, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)
    predictions = fmodel(params, buffers, batch)
    loss = loss_fn(predictions, targets)
    return loss


# Now, let's use ``grad`` to create a new function that computes the gradient
# with respect to the first argument of compute_loss (i.e. the params).
ft_compute_grad = grad(compute_loss)

# ``ft_compute_grad`` computes the gradient for a single (sample, target) pair.
# We can use ``vmap`` to get it to compute the gradient over an entire batch
# of samples and targets. Note that in_dims=(None, None, 0, 0) because we wish
# to map ``ft_compute_grad`` over the 0th dimension of the data and targets
# and use the same params and buffers for each.
ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

# Finally, let's used our transformed function to compute per-sample-gradients:
ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data, targets)
for per_sample_grad, ft_per_sample_grad in zip(per_sample_grads, ft_per_sample_grads):
    assert torch.allclose(per_sample_grad, ft_per_sample_grad, atol=1e-6, rtol=1e-6)

# A quick note: there are limitations around what types of functions can be
# transformed by vmap. The best functions to transform are ones that are
# pure functions: a function where the outputs are only determined by the inputs
# that have no side effects (e.g. mutation). vmap is unable to handle mutation of
# arbitrary Python data structures, but it is able to handle many in-place
# PyTorch operations.

```



## High-Level Overview

"""==========================Per-sample-gradients==========================What is it?--------------------------------------------------------------------Per-sample-gradient computation is computing the gradient for each and everysample in a batch of data. It is a useful quantity in differential privacyand optimization research.

This Python file contains 1 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SimpleCNN`

**Functions defined**: `__init__`, `forward`, `loss_fn`, `compute_grad`, `compute_sample_grads`, `compute_loss`

**Key imports**: torch, torch.nn as nn, torch.nn.functional as F, grad, make_functional_with_buffers, vmap


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `functorch/docs/source/tutorials/_src`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.nn as nn`
- `torch.nn.functional as F`
- `functorch`: grad, make_functional_with_buffers, vmap


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components
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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`functorch/docs/source/tutorials/_src`):

- [`plot_jacobians_and_hessians.py_docs.md`](./plot_jacobians_and_hessians.py_docs.md)
- [`plot_ensembling.py_docs.md`](./plot_ensembling.py_docs.md)


## Cross-References

- **File Documentation**: `plot_per_sample_gradients.py_docs.md`
- **Keyword Index**: `plot_per_sample_gradients.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
