# Documentation: `docs/torch/nn/modules/adaptive.py_docs.md`

## File Metadata

- **Path**: `docs/torch/nn/modules/adaptive.py_docs.md`
- **Size**: 15,568 bytes (15.20 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/nn/modules/adaptive.py`

## File Metadata

- **Path**: `torch/nn/modules/adaptive.py`
- **Size**: 12,606 bytes (12.31 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: allow-untyped-defs

import itertools
from collections import namedtuple
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .container import ModuleList, Sequential
from .linear import Linear
from .module import Module


__all__ = ["AdaptiveLogSoftmaxWithLoss"]

_ASMoutput = namedtuple("_ASMoutput", ["output", "loss"])


class AdaptiveLogSoftmaxWithLoss(Module):
    (
        """Efficient softmax approximation.

    As described in
    `Efficient softmax approximation for GPUs by Edouard Grave, Armand Joulin,
    Moustapha Ciss\u00e9, David Grangier, and Herv\u00e9 J\u00e9gou
    <https://arxiv.org/abs/1609.04309>`__.
"""
        r"""
    Adaptive softmax is an approximate strategy for training models with large
    output spaces. It is most effective when the label distribution is highly
    imbalanced, for example in natural language modelling, where the word
    frequency distribution approximately follows the `Zipf's law`_.

    Adaptive softmax partitions the labels into several clusters, according to
    their frequency. These clusters may contain different number of targets
    each.
    Additionally, clusters containing less frequent labels assign lower
    dimensional embeddings to those labels, which speeds up the computation.
    For each minibatch, only clusters for which at least one target is
    present are evaluated.

    The idea is that the clusters which are accessed frequently
    (like the first one, containing most frequent labels), should also be cheap
    to compute -- that is, contain a small number of assigned labels.

    We highly recommend taking a look at the original paper for more details.

    * :attr:`cutoffs` should be an ordered Sequence of integers sorted
      in the increasing order.
      It controls number of clusters and the partitioning of targets into
      clusters. For example setting ``cutoffs = [10, 100, 1000]``
      means that first `10` targets will be assigned
      to the 'head' of the adaptive softmax, targets `11, 12, ..., 100` will be
      assigned to the first cluster, and targets `101, 102, ..., 1000` will be
      assigned to the second cluster, while targets
      `1001, 1002, ..., n_classes - 1` will be assigned
      to the last, third cluster.

    * :attr:`div_value` is used to compute the size of each additional cluster,
      which is given as
      :math:`\left\lfloor\frac{\texttt{in\_features}}{\texttt{div\_value}^{idx}}\right\rfloor`,
      where :math:`idx` is the cluster index (with clusters
      for less frequent words having larger indices,
      and indices starting from :math:`1`).

    * :attr:`head_bias` if set to True, adds a bias term to the 'head' of the
      adaptive softmax. See paper for details. Set to False in the official
      implementation.

    .. warning::
        Labels passed as inputs to this module should be sorted according to
        their frequency. This means that the most frequent label should be
        represented by the index `0`, and the least frequent
        label should be represented by the index `n_classes - 1`.

    .. note::
        This module returns a ``NamedTuple`` with ``output``
        and ``loss`` fields. See further documentation for details.

    .. note::
        To compute log-probabilities for all classes, the ``log_prob``
        method can be used.

    Args:
        in_features (int): Number of features in the input tensor
        n_classes (int): Number of classes in the dataset
        cutoffs (Sequence): Cutoffs used to assign targets to their buckets
        div_value (float, optional): value used as an exponent to compute sizes
            of the clusters. Default: 4.0
        head_bias (bool, optional): If ``True``, adds a bias term to the 'head' of the
            adaptive softmax. Default: ``False``

    Returns:
        ``NamedTuple`` with ``output`` and ``loss`` fields:
            * **output** is a Tensor of size ``N`` containing computed target
              log probabilities for each example
            * **loss** is a Scalar representing the computed negative
              log likelihood loss

    Shape:
        - input: :math:`(N, \texttt{in\_features})` or :math:`(\texttt{in\_features})`
        - target: :math:`(N)` or :math:`()` where each value satisfies :math:`0 <= \texttt{target[i]} <= \texttt{n\_classes}`
        - output1: :math:`(N)` or :math:`()`
        - output2: ``Scalar``

    .. _Zipf's law: https://en.wikipedia.org/wiki/Zipf%27s_law
    """
    )

    in_features: int
    n_classes: int
    cutoffs: list[int]
    div_value: float
    head_bias: bool
    head: Linear
    tail: ModuleList

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        cutoffs: Sequence[int],
        div_value: float = 4.0,
        head_bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        cutoffs = list(cutoffs)

        if len(cutoffs) == 0:
            raise ValueError("cutoffs should be a sequence of length larger than 0")

        if (
            (cutoffs != sorted(cutoffs))
            or (min(cutoffs) <= 0)
            or (max(cutoffs) > (n_classes - 1))
            or (len(set(cutoffs)) != len(cutoffs))
            or any(int(c) != c for c in cutoffs)
        ):
            raise ValueError(
                "cutoffs should be a sequence of unique, positive "
                "integers sorted in an increasing order, where "
                "each value is between 1 and n_classes-1"
            )

        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]
        self.div_value = div_value
        self.head_bias = head_bias

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        self.head = Linear(
            self.in_features, self.head_size, bias=self.head_bias, **factory_kwargs
        )
        self.tail = ModuleList()

        for i in range(self.n_clusters):
            hsz = int(self.in_features // (self.div_value ** (i + 1)))
            osz = self.cutoffs[i + 1] - self.cutoffs[i]

            projection = Sequential(
                Linear(self.in_features, hsz, bias=False, **factory_kwargs),
                Linear(hsz, osz, bias=False, **factory_kwargs),
            )

            self.tail.append(projection)

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        self.head.reset_parameters()
        for i2h, h2o in self.tail:  # type: ignore[misc]
            i2h.reset_parameters()  # type: ignore[has-type]
            h2o.reset_parameters()  # type: ignore[has-type]

    def forward(self, input_: Tensor, target_: Tensor) -> _ASMoutput:
        """
        Runs the forward pass.
        """
        targ_dim = target_.dim()

        if targ_dim == 1:
            if input_.size(0) != target_.size(0):
                raise RuntimeError(
                    "Input and target should have the same size in the batch dimension."
                )
            if input_.dim() != 2:
                raise RuntimeError(
                    "1D target tensor expects 2D input tensors, "
                    "but found inputs with size",
                    input_.size(),
                )
        elif targ_dim == 0:
            if input_.dim() != 1:
                raise RuntimeError(
                    "0D target tensor expects 1D input tensors, "
                    "but found inputs with size",
                    input_.size(),
                )
        else:
            raise RuntimeError(
                "0D or 1D target tensor expected, multi-target not supported"
            )

        is_batched = targ_dim > 0
        input = input_ if is_batched else input_.unsqueeze(0)
        target = target_ if is_batched else target_.unsqueeze(0)

        used_rows = 0
        batch_size = target.size(0)

        output = input.new_zeros(batch_size)
        gather_inds = target.new_empty(batch_size)

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):
            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            target_mask = (target >= low_idx) & (target < high_idx)
            row_indices = target_mask.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue

            if i == 0:
                gather_inds.index_copy_(0, row_indices, target[target_mask])

            else:
                relative_target = target[target_mask] - low_idx
                input_subset = input.index_select(0, row_indices)

                cluster_output = self.tail[i - 1](input_subset)
                cluster_index = self.shortlist_size + i - 1

                gather_inds.index_fill_(0, row_indices, cluster_index)
                cluster_logprob = F.log_softmax(cluster_output, dim=1)
                local_logprob = cluster_logprob.gather(1, relative_target.unsqueeze(1))
                output.index_copy_(0, row_indices, local_logprob.squeeze(1))

            used_rows += row_indices.numel()

        if used_rows != batch_size:
            raise RuntimeError(
                f"Target values should be in [0, {self.n_classes - 1}], "
                f"but values in range [{target.min().item()}, {target.max().item()}] "
                "were found. "
            )

        head_output = self.head(input)
        head_logprob = F.log_softmax(head_output, dim=1)
        output += head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze()
        loss = (-output).mean()

        if not is_batched:
            output = output.squeeze(0)

        return _ASMoutput(output, loss)

    def _get_full_log_prob(self, input, head_output):
        """Given input tensor, and output of ``self.head``, compute the log of the full distribution."""
        out = input.new_empty((head_output.size(0), self.n_classes))
        head_logprob = F.log_softmax(head_output, dim=1)

        out[:, : self.shortlist_size] = head_logprob[:, : self.shortlist_size]

        for i, (start_idx, stop_idx) in enumerate(itertools.pairwise(self.cutoffs)):
            cluster_output = self.tail[i](input)
            cluster_logprob = F.log_softmax(cluster_output, dim=1)
            output_logprob = cluster_logprob + head_logprob[
                :, self.shortlist_size + i
            ].unsqueeze(1)

            out[:, start_idx:stop_idx] = output_logprob

        return out

    def log_prob(self, input: Tensor) -> Tensor:
        r"""Compute log probabilities for all :math:`\texttt{n\_classes}`.

        Args:
            input (Tensor): a minibatch of examples

        Returns:
            log-probabilities of for each class :math:`c`
            in range :math:`0 <= c <= \texttt{n\_classes}`, where :math:`\texttt{n\_classes}` is a
            parameter passed to ``AdaptiveLogSoftmaxWithLoss`` constructor.

        Shape:
            - Input: :math:`(N, \texttt{in\_features})`
            - Output: :math:`(N, \texttt{n\_classes})`

        """
        head_output = self.head(input)
        return self._get_full_log_prob(input, head_output)

    def predict(self, input: Tensor) -> Tensor:
        r"""Return the class with the highest probability for each example in the input minibatch.

        This is equivalent to ``self.log_prob(input).argmax(dim=1)``, but is more efficient in some cases.

        Args:
            input (Tensor): a minibatch of examples

        Returns:
            output (Tensor): a class with the highest probability for each example

        Shape:
            - Input: :math:`(N, \texttt{in\_features})`
            - Output: :math:`(N)`
        """
        head_output = self.head(input)
        output = torch.argmax(head_output, dim=1)
        not_in_shortlist = output >= self.shortlist_size
        all_in_shortlist = not (not_in_shortlist.any())

        if all_in_shortlist:
            return output

        elif not_in_shortlist.all():
            log_prob = self._get_full_log_prob(input, head_output)
            return torch.argmax(log_prob, dim=1)

        else:
            log_prob = self._get_full_log_prob(
                input[not_in_shortlist], head_output[not_in_shortlist]
            )
            output[not_in_shortlist] = torch.argmax(log_prob, dim=1)
            return output

```



## High-Level Overview

"""Efficient softmax approximation.    As described in    `Efficient softmax approximation for GPUs by Edouard Grave, Armand Joulin,    Moustapha Ciss\u00e9, David Grangier, and Herv\u00e9 J\u00e9gou    <https://arxiv.org/abs/1609.04309>`__.

This Python file contains 3 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `AdaptiveLogSoftmaxWithLoss`

**Functions defined**: `__init__`, `reset_parameters`, `forward`, `_get_full_log_prob`, `log_prob`, `predict`

**Key imports**: itertools, namedtuple, Sequence, torch, torch.nn.functional as F, Tensor, ModuleList, Sequential, Linear, Module


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/nn/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `itertools`
- `collections`: namedtuple
- `collections.abc`: Sequence
- `torch`
- `torch.nn.functional as F`
- `.container`: ModuleList, Sequential
- `.linear`: Linear
- `.module`: Module


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/nn/modules`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`fold.py_docs.md`](./fold.py_docs.md)
- [`rnn.py_docs.md`](./rnn.py_docs.md)
- [`channelshuffle.py_docs.md`](./channelshuffle.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`conv.py_docs.md`](./conv.py_docs.md)
- [`distance.py_docs.md`](./distance.py_docs.md)
- [`linear.py_docs.md`](./linear.py_docs.md)
- [`normalization.py_docs.md`](./normalization.py_docs.md)


## Cross-References

- **File Documentation**: `adaptive.py_docs.md`
- **Keyword Index**: `adaptive.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/nn/modules`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/nn/modules`, which is part of the **core PyTorch library**.



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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/nn/modules`):

- [`sparse.py_docs.md_docs.md`](./sparse.py_docs.md_docs.md)
- [`instancenorm.py_kw.md_docs.md`](./instancenorm.py_kw.md_docs.md)
- [`activation.py_kw.md_docs.md`](./activation.py_kw.md_docs.md)
- [`container.py_docs.md_docs.md`](./container.py_docs.md_docs.md)
- [`distance.py_kw.md_docs.md`](./distance.py_kw.md_docs.md)
- [`pixelshuffle.py_kw.md_docs.md`](./pixelshuffle.py_kw.md_docs.md)
- [`module.py_docs.md_docs.md`](./module.py_docs.md_docs.md)
- [`batchnorm.py_docs.md_docs.md`](./batchnorm.py_docs.md_docs.md)
- [`transformer.py_kw.md_docs.md`](./transformer.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `adaptive.py_docs.md_docs.md`
- **Keyword Index**: `adaptive.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
