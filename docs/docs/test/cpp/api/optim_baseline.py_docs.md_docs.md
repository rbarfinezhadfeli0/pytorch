# Documentation: `docs/test/cpp/api/optim_baseline.py_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/api/optim_baseline.py_docs.md`
- **Size**: 7,117 bytes (6.95 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/cpp/api/optim_baseline.py`

## File Metadata

- **Path**: `test/cpp/api/optim_baseline.py`
- **Size**: 4,555 bytes (4.45 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. Can be **executed as a standalone script**.

## Original Source

```python
"""Script to generate baseline values from PyTorch optimization algorithms"""

import argparse
import math
import sys

import torch
import torch.optim


HEADER = """
#include <torch/types.h>

#include <vector>

namespace expected_parameters {
"""

FOOTER = "} // namespace expected_parameters"

PARAMETERS = "inline std::vector<std::vector<torch::Tensor>> {}() {{"

OPTIMIZERS = {
    "LBFGS": lambda p: torch.optim.LBFGS(p, 1.0),
    "LBFGS_with_line_search": lambda p: torch.optim.LBFGS(
        p, 1.0, line_search_fn="strong_wolfe"
    ),
    "Adam": lambda p: torch.optim.Adam(p, 1.0),
    "Adam_with_weight_decay": lambda p: torch.optim.Adam(p, 1.0, weight_decay=1e-2),
    "Adam_with_weight_decay_and_amsgrad": lambda p: torch.optim.Adam(
        p, 1.0, weight_decay=1e-6, amsgrad=True
    ),
    "AdamW": lambda p: torch.optim.AdamW(p, 1.0),
    "AdamW_without_weight_decay": lambda p: torch.optim.AdamW(p, 1.0, weight_decay=0),
    "AdamW_with_amsgrad": lambda p: torch.optim.AdamW(p, 1.0, amsgrad=True),
    "Adagrad": lambda p: torch.optim.Adagrad(p, 1.0),
    "Adagrad_with_weight_decay": lambda p: torch.optim.Adagrad(
        p, 1.0, weight_decay=1e-2
    ),
    "Adagrad_with_weight_decay_and_lr_decay": lambda p: torch.optim.Adagrad(
        p, 1.0, weight_decay=1e-6, lr_decay=1e-3
    ),
    "RMSprop": lambda p: torch.optim.RMSprop(p, 0.1),
    "RMSprop_with_weight_decay": lambda p: torch.optim.RMSprop(
        p, 0.1, weight_decay=1e-2
    ),
    "RMSprop_with_weight_decay_and_centered": lambda p: torch.optim.RMSprop(
        p, 0.1, weight_decay=1e-6, centered=True
    ),
    "RMSprop_with_weight_decay_and_centered_and_momentum": lambda p: torch.optim.RMSprop(
        p, 0.1, weight_decay=1e-6, centered=True, momentum=0.9
    ),
    "SGD": lambda p: torch.optim.SGD(p, 0.1),
    "SGD_with_weight_decay": lambda p: torch.optim.SGD(p, 0.1, weight_decay=1e-2),
    "SGD_with_weight_decay_and_momentum": lambda p: torch.optim.SGD(
        p, 0.1, momentum=0.9, weight_decay=1e-2
    ),
    "SGD_with_weight_decay_and_nesterov_momentum": lambda p: torch.optim.SGD(
        p, 0.1, momentum=0.9, weight_decay=1e-6, nesterov=True
    ),
}


def weight_init(module):
    if isinstance(module, torch.nn.Linear):
        stdev = 1.0 / math.sqrt(module.weight.size(1))
        for p in module.parameters():
            p.data.uniform_(-stdev, stdev)


def run(optimizer_name, iterations, sample_every):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3),
        torch.nn.Sigmoid(),
        torch.nn.Linear(3, 1),
        torch.nn.Sigmoid(),
    )
    model = model.to(torch.float64).apply(weight_init)

    optimizer = OPTIMIZERS[optimizer_name](model.parameters())

    input = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float64)

    values = []
    for i in range(iterations):
        optimizer.zero_grad()

        output = model.forward(input)
        loss = output.sum()
        loss.backward()

        def closure():
            return torch.tensor([10.0])

        optimizer.step(closure)

        if i % sample_every == 0:
            values.append(
                [p.clone().flatten().data.numpy() for p in model.parameters()]
            )

    return values


def emit(optimizer_parameter_map):
    # Don't write generated with an @ in front, else this file is recognized as generated.
    print("// @{} from {}".format("generated", __file__))
    print(HEADER)
    for optimizer_name, parameters in optimizer_parameter_map.items():
        print(PARAMETERS.format(optimizer_name))
        print("  return {")
        for sample in parameters:
            print("    {")
            for parameter in sample:
                parameter_values = "{{{}}}".format(", ".join(map(str, parameter)))
                print(f"      torch::tensor({parameter_values}),")
            print("    },")
        print("  };")
        print("}\n")
    print(FOOTER)


def main():
    parser = argparse.ArgumentParser(
        "Produce optimization output baseline from PyTorch"
    )
    parser.add_argument("-i", "--iterations", default=1001, type=int)
    parser.add_argument("-s", "--sample-every", default=100, type=int)
    options = parser.parse_args()

    optimizer_parameter_map = {}
    for optimizer in OPTIMIZERS:
        sys.stderr.write(f"Evaluating {optimizer} ...\n")
        optimizer_parameter_map[optimizer] = run(
            optimizer, options.iterations, options.sample_every
        )

    emit(optimizer_parameter_map)


if __name__ == "__main__":
    main()

```



## High-Level Overview

"""Script to generate baseline values from PyTorch optimization algorithms"""import argparseimport mathimport sysimport torchimport torch.optim

This Python file contains 0 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `weight_init`, `run`, `closure`, `emit`, `main`

**Key imports**: argparse, math, sys, torch, torch.optim


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/api`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `math`
- `sys`
- `torch`
- `torch.optim`


## Code Patterns & Idioms

### Common Patterns

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
python test/cpp/api/optim_baseline.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/api`):

- [`fft.cpp_docs.md`](./fft.cpp_docs.md)
- [`tensor_options.cpp_docs.md`](./tensor_options.cpp_docs.md)
- [`any.cpp_docs.md`](./any.cpp_docs.md)
- [`torch_include.cpp_docs.md`](./torch_include.cpp_docs.md)
- [`rnn.cpp_docs.md`](./rnn.cpp_docs.md)
- [`jit.cpp_docs.md`](./jit.cpp_docs.md)
- [`nn_utils.cpp_docs.md`](./nn_utils.cpp_docs.md)
- [`nested.cpp_docs.md`](./nested.cpp_docs.md)
- [`meta_tensor.cpp_docs.md`](./meta_tensor.cpp_docs.md)
- [`nested_int.cpp_docs.md`](./nested_int.cpp_docs.md)


## Cross-References

- **File Documentation**: `optim_baseline.py_docs.md`
- **Keyword Index**: `optim_baseline.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp/api`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp/api`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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
python docs/test/cpp/api/optim_baseline.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/api`):

- [`init_baseline.py_kw.md_docs.md`](./init_baseline.py_kw.md_docs.md)
- [`support.cpp_kw.md_docs.md`](./support.cpp_kw.md_docs.md)
- [`memory.cpp_docs.md_docs.md`](./memory.cpp_docs.md_docs.md)
- [`parallel_benchmark.cpp_docs.md_docs.md`](./parallel_benchmark.cpp_docs.md_docs.md)
- [`dataloader.cpp_docs.md_docs.md`](./dataloader.cpp_docs.md_docs.md)
- [`moduledict.cpp_kw.md_docs.md`](./moduledict.cpp_kw.md_docs.md)
- [`support.h_kw.md_docs.md`](./support.h_kw.md_docs.md)
- [`ordered_dict.cpp_docs.md_docs.md`](./ordered_dict.cpp_docs.md_docs.md)
- [`functional.cpp_docs.md_docs.md`](./functional.cpp_docs.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)


## Cross-References

- **File Documentation**: `optim_baseline.py_docs.md_docs.md`
- **Keyword Index**: `optim_baseline.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
