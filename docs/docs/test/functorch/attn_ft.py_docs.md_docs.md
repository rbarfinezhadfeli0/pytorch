# Documentation: `docs/test/functorch/attn_ft.py_docs.md`

## File Metadata

- **Path**: `docs/test/functorch/attn_ft.py_docs.md`
- **Size**: 10,306 bytes (10.06 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/functorch/attn_ft.py`

## File Metadata

- **Path**: `test/functorch/attn_ft.py`
- **Size**: 7,703 bytes (7.52 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch
from functorch.dim import cat, dimlists, dims
from torch import nn


class Linear(nn.Linear):
    def forward(self, input):
        ci, co = dims()
        b = dimlists()
        result = (input[b, ci] * self.weight[co, ci]).sum(ci) + self.bias[co]
        return result.order(b, co)


class BertSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        position_embedding_type=None,
        max_position_embeddings=None,
        linear=Linear,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = linear(hidden_size, self.all_head_size)
        self.key = linear(hidden_size, self.all_head_size)
        self.value = linear(hidden_size, self.all_head_size)

        self.dropout_prob = attention_probs_dropout_prob
        self.position_embedding_type = position_embedding_type

        if self.position_embedding_type is not None:
            assert max_position_embeddings is not None
            self.max_position_embeddings = max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * max_position_embeddings - 1, self.attention_head_size
            )

    def forward(
        self,
        hidden_states,
        past_key_value=None,
    ):
        # first run the encoding linear layers for q, k, v normally
        # the meaning of a linear layer is well understood, so no need to use explicit dimensions
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        # introduce values that represent each dimension. dimensions are 'first class'
        # because they are actual python values introduced here
        batch, query_sequence, key_sequence, heads, features = dims()
        heads.size = self.num_attention_heads

        # bind the positional dimensions in k, q, and v against
        # our values. the sizes of each dimension are determined by this binding
        # and when a dimension is used twice (e.g. batch), its size against both
        # uses is checked for consistency.
        # The group (heads, features) splits apart a single positional dimension
        # into two dimensions. Since heads.size*features.size == q.size(2)
        # and we specified heads.size, features.size is inferred here.
        q = q[batch, query_sequence, [heads, features]]
        k = k[batch, key_sequence, [heads, features]]
        v = v[batch, key_sequence, [heads, features]]

        # this option allows the model to attend to not just the elements of the current sequence
        # but the previous elements as well as additional tokens.
        if past_key_value is not None:
            extended_key_sequence = dims()
            key_past = past_key_value[0][batch, heads, key_sequence, features]
            value_past = past_key_value[1][batch, heads, key_sequence, features]
            # cat introduces a new dimension extended_key_sequence, because it is twice as long
            # as the original key_sequence
            k = cat([key_past, k], key_sequence, extended_key_sequence)
            v = cat([value_past, v], key_sequence, extended_key_sequence)
            # for the rest of the function, we will just use extended_key_sequence in lieu of
            # key_sequence
            key_sequence = extended_key_sequence

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # The actual outer-product and summation are explicitly represented here,
        # and like einsum, will be pattern matched to an efficient matrix multiply op.
        attention_scores = (q * k).sum(features) / math.sqrt(features.size)

        # relative positional embeddings gave a unique embedding based on the distance between
        # key and value tokens in the sequence, e.g.
        #  0  1  2  3
        # -1  0  1  2
        # -2 -1  0  1
        # -3 -2 -1  0
        if self.position_embedding_type is not None:
            # the value of a dimension object when used as a tensor is the indices along its dimension
            # so we can directly subtract the two dimensions to get a 2D tensor of (query_sequence x key_sequence)
            # with the distance between them
            distance = query_sequence - key_sequence

            assert key_sequence.size <= self.max_position_embeddings

            # we can then use that as an indirect index into the embedding table values to look up the features for that index
            # this is just a `gather` primitive op. The resulting tensor will
            # have all the dimensions of embedding_idx (query_sequence x key_sequence),
            # plus all the dimensions of `embed` that were not indirectly accessed (`embedding_range`).
            # this form of indirect indexing is more straightforward than either advanced indexing or torch.gather which both
            # have a lot of dependencies on the positions of indexing tensors.

            positional_embedding = self.distance_embedding.weight[
                self.max_position_embeddings - 1 + distance, features
            ]

            if self.position_embedding_type == "relative_key":
                # these were einsum ops in the positional code because they are not easy to fit to existing matmul operators
                # even though they are degenerate matmuls
                relative_position_scores = (q * positional_embedding).sum(features)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = (q * positional_embedding).sum(
                    features
                )
                relative_position_scores_key = (k * positional_embedding).sum(features)
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_probs = attention_scores
        # Normalize the attention scores to probabilities.
        attention_probs = torch.softmax(attention_scores, dim=key_sequence)
        # # This is actually dropping out entire tokens to attend to, which might
        # # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = torch.nn.functional.dropout(
            attention_probs, p=self.dropout_prob
        )

        # similarly, we can replace the matmul with a direct listing of the outer product, which makes it clear
        # we are weighting the values v across all keys with the attention scores.
        context_layer = (attention_probs * v).sum(key_sequence)

        # finally, we convert back to a standard tensor by describing the layout of dimensions.
        # working in reverse to with_dims, the (heads, features) group flattens the dimensions into a single one.
        return context_layer.order(batch, query_sequence, [heads, features])

```



## High-Level Overview


This Python file contains 2 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `Linear`, `BertSelfAttention`

**Functions defined**: `forward`, `__init__`, `forward`

**Key imports**: math, torch, cat, dimlists, dims, nn


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/functorch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `math`
- `torch`
- `functorch.dim`: cat, dimlists, dims


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
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
python test/functorch/attn_ft.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/functorch`):

- [`test_vmap.py_docs.md`](./test_vmap.py_docs.md)
- [`test_rearrange.py_docs.md`](./test_rearrange.py_docs.md)
- [`test_aot_joint_with_descriptors.py_docs.md`](./test_aot_joint_with_descriptors.py_docs.md)
- [`functorch_additional_op_db.py_docs.md`](./functorch_additional_op_db.py_docs.md)
- [`xfail_suggester.py_docs.md`](./xfail_suggester.py_docs.md)
- [`discover_coverage.py_docs.md`](./discover_coverage.py_docs.md)
- [`test_eager_transforms.py_docs.md`](./test_eager_transforms.py_docs.md)
- [`test_ac.py_docs.md`](./test_ac.py_docs.md)
- [`common_utils.py_docs.md`](./common_utils.py_docs.md)
- [`test_logging.py_docs.md`](./test_logging.py_docs.md)


## Cross-References

- **File Documentation**: `attn_ft.py_docs.md`
- **Keyword Index**: `attn_ft.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/functorch`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/functorch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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
python docs/test/functorch/attn_ft.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/functorch`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_aot_joint_with_descriptors.py_kw.md_docs.md`](./test_aot_joint_with_descriptors.py_kw.md_docs.md)
- [`test_minifier.py_kw.md_docs.md`](./test_minifier.py_kw.md_docs.md)
- [`test_eager_transforms.py_docs.md_docs.md`](./test_eager_transforms.py_docs.md_docs.md)
- [`functorch_additional_op_db.py_kw.md_docs.md`](./functorch_additional_op_db.py_kw.md_docs.md)
- [`test_ac_knapsack.py_docs.md_docs.md`](./test_ac_knapsack.py_docs.md_docs.md)
- [`common_utils.py_kw.md_docs.md`](./common_utils.py_kw.md_docs.md)
- [`test_logging.py_kw.md_docs.md`](./test_logging.py_kw.md_docs.md)
- [`test_rearrange.py_kw.md_docs.md`](./test_rearrange.py_kw.md_docs.md)
- [`test_dims.py_kw.md_docs.md`](./test_dims.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `attn_ft.py_docs.md_docs.md`
- **Keyword Index**: `attn_ft.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
