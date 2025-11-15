# Documentation: `benchmarks/operator_benchmark/pt/qatembedding_ops_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/qatembedding_ops_test.py`
- **Size**: 2,787 bytes (2.72 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
import numpy
from pt import configs

import operator_benchmark as op_bench

import torch
import torch.ao.nn.qat as nnqat
from torch.ao.quantization import default_embedding_qat_qconfig


"""
Microbenchmarks for QAT Embedding + EmbeddingBag operators.
"""


class QATEmbeddingBagBenchmark(op_bench.TorchBenchmarkBase):
    def init(
        self,
        embeddingbags,
        dim,
        mode,
        input_size,
        offset,
        sparse,
        include_last_offset,
        device,
    ):
        qconfig = default_embedding_qat_qconfig
        self.embedding = nnqat.EmbeddingBag(
            num_embeddings=embeddingbags,
            embedding_dim=dim,
            mode=mode,
            include_last_offset=include_last_offset,
            sparse=sparse,
            device=device,
            qconfig=qconfig,
        )
        numpy.random.seed((1 << 32) - 1)
        offsets = torch.LongTensor([offset], device=device)
        input = torch.tensor(
            numpy.random.randint(0, embeddingbags, input_size), device=device
        ).long()
        self.inputs = {
            "input": input,
            "offset": torch.cat(
                (offsets, torch.tensor([input.size(0)], dtype=torch.long)), 0
            ),
        }
        self.set_module_name("qatEmbeddingBag")

    def forward(self, input, offset):
        return self.embedding(input, offset)


# Currently, EmbeddingBag QAT does not support sparse embeddings.
embeddingbag_short_dense_configs = [
    config
    for config in configs.embeddingbag_short_configs
    if {"sparse": True} not in config
]

op_bench.generate_pt_test(embeddingbag_short_dense_configs, QATEmbeddingBagBenchmark)
op_bench.generate_pt_gradient_test(
    embeddingbag_short_dense_configs, QATEmbeddingBagBenchmark
)


class QATEmbeddingBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, num_embeddings, embedding_dim, input_size, device):
        qconfig = default_embedding_qat_qconfig
        self.embedding = nnqat.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            qconfig=qconfig,
            device=device,
        )
        self.embedding.qconfig = default_embedding_qat_qconfig
        numpy.random.seed((1 << 32) - 1)
        self.input = torch.tensor(
            numpy.random.randint(0, num_embeddings, input_size), device=device
        ).long()
        self.inputs = {"input": self.input}
        self.set_module_name("qatEmbedding")

    def forward(self, input):
        return self.embedding(input)


op_bench.generate_pt_test(configs.embedding_short_configs, QATEmbeddingBenchmark)
op_bench.generate_pt_gradient_test(
    configs.embedding_short_configs, QATEmbeddingBenchmark
)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Microbenchmarks for QAT Embedding + EmbeddingBag operators.

This Python file contains 2 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `QATEmbeddingBagBenchmark`, `QATEmbeddingBenchmark`

**Functions defined**: `init`, `forward`, `init`, `forward`

**Key imports**: numpy, configs, operator_benchmark as op_bench, torch, torch.ao.nn.qat as nnqat, default_embedding_qat_qconfig


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/operator_benchmark/pt`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `numpy`
- `pt`: configs
- `operator_benchmark as op_bench`
- `torch`
- `torch.ao.nn.qat as nnqat`
- `torch.ao.quantization`: default_embedding_qat_qconfig


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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
python benchmarks/operator_benchmark/pt/qatembedding_ops_test.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`benchmarks/operator_benchmark/pt`):

- [`qarithmetic_test.py_docs.md`](./qarithmetic_test.py_docs.md)
- [`bmm_test.py_docs.md`](./bmm_test.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`gather_test.py_docs.md`](./gather_test.py_docs.md)
- [`clip_ranges_test.py_docs.md`](./clip_ranges_test.py_docs.md)
- [`split_test.py_docs.md`](./split_test.py_docs.md)
- [`groupnorm_test.py_docs.md`](./groupnorm_test.py_docs.md)
- [`sum_test.py_docs.md`](./sum_test.py_docs.md)
- [`matrix_mult_test.py_docs.md`](./matrix_mult_test.py_docs.md)
- [`pool_test.py_docs.md`](./pool_test.py_docs.md)


## Cross-References

- **File Documentation**: `qatembedding_ops_test.py_docs.md`
- **Keyword Index**: `qatembedding_ops_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
