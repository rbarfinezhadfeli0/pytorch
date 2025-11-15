# Documentation: `benchmarks/operator_benchmark/pt/embeddingbag_test.py`

## File Metadata

- **Path**: `benchmarks/operator_benchmark/pt/embeddingbag_test.py`
- **Size**: 2,189 bytes (2.14 KB)
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


"""Embedding and EmbeddingBag Operator Benchmark"""


class EmbeddingBagBenchmark(op_bench.TorchBenchmarkBase):
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
        self.embedding = torch.nn.EmbeddingBag(
            num_embeddings=embeddingbags,
            embedding_dim=dim,
            mode=mode,
            include_last_offset=include_last_offset,
            sparse=sparse,
        ).to(device=device)
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
        self.set_module_name("embeddingbag")

    def forward(self, input, offset):
        return self.embedding(input, offset)


op_bench.generate_pt_test(configs.embeddingbag_short_configs, EmbeddingBagBenchmark)
op_bench.generate_pt_gradient_test(
    configs.embeddingbag_short_configs, EmbeddingBagBenchmark
)


class EmbeddingBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, num_embeddings, embedding_dim, input_size, device):
        self.embedding = torch.nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        ).to(device=device)
        numpy.random.seed((1 << 32) - 1)
        input = torch.tensor(
            numpy.random.randint(0, num_embeddings, input_size), device=device
        ).long()
        self.inputs = {"input": input}
        self.set_module_name("embedding")

    def forward(self, input):
        return self.embedding(input)


op_bench.generate_pt_test(configs.embedding_short_configs, EmbeddingBenchmark)
op_bench.generate_pt_gradient_test(configs.embedding_short_configs, EmbeddingBenchmark)

if __name__ == "__main__":
    op_bench.benchmark_runner.main()

```



## High-Level Overview

"""Embedding and EmbeddingBag Operator Benchmark"""class EmbeddingBagBenchmark(op_bench.TorchBenchmarkBase):    def init(        self,        embeddingbags,        dim,        mode,        input_size,        offset,        sparse,        include_last_offset,        device,    ):        self.embedding = torch.nn.EmbeddingBag(            num_embeddings=embeddingbags,            embedding_dim=dim,            mode=mode,            include_last_offset=include_last_offset,            sparse=sparse,        ).to(device=device)        numpy.random.seed((1 << 32) - 1)        offsets = torch.LongTensor([offset], device=device)        input = torch.tensor(            numpy.random.randint(0, embeddingbags, input_size), device=device        ).long()        self.inputs = {            "input": input,            "offset": torch.cat(                (offsets, torch.tensor([input.size(0)], dtype=torch.long)), 0            ),        }        self.set_module_name("embeddingbag")    def forward(self, input, offset):        return self.embedding(input, offset)op_bench.generate_pt_test(configs.embeddingbag_short_configs, EmbeddingBagBenchmark)op_bench.generate_pt_gradient_test(    configs.embeddingbag_short_configs, EmbeddingBagBenchmark

This Python file contains 2 class(es) and 4 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `EmbeddingBagBenchmark`, `EmbeddingBenchmark`

**Functions defined**: `init`, `forward`, `init`, `forward`

**Key imports**: numpy, configs, operator_benchmark as op_bench, torch


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


## Code Patterns & Idioms

### Common Patterns

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
python benchmarks/operator_benchmark/pt/embeddingbag_test.py
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

- **File Documentation**: `embeddingbag_test.py_docs.md`
- **Keyword Index**: `embeddingbag_test.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
