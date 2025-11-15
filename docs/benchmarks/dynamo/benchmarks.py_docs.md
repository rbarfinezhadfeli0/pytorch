# Documentation: `benchmarks/dynamo/benchmarks.py`

## File Metadata

- **Path**: `benchmarks/dynamo/benchmarks.py`
- **Size**: 3,239 bytes (3.16 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file contains **examples or benchmarks**. Can be **executed as a standalone script**.

## Original Source

```python
#!/usr/bin/env python3

import argparse
import os
import sys


# Run only this selected group of models, leave this empty to run everything
TORCHBENCH_ONLY_MODELS = [
    m.strip() for m in os.getenv("TORCHBENCH_ONLY_MODELS", "").split(",") if m.strip()
]


# Note - hf and timm have their own version of this, torchbench does not
# TODO(voz): Someday, consolidate all the files into one runner instead of a shim like this...
def model_names(filename: str) -> set[str]:
    names = set()
    with open(filename) as fh:
        lines = fh.readlines()
        lines = [line.rstrip() for line in lines]
        for line in lines:
            line_parts = line.split(" ")
            if len(line_parts) == 1:
                line_parts = line.split(",")
            model_name = line_parts[0]
            if TORCHBENCH_ONLY_MODELS and model_name not in TORCHBENCH_ONLY_MODELS:
                continue
            names.add(model_name)
    return names


TIMM_MODEL_NAMES = model_names(
    os.path.join(os.path.dirname(__file__), "timm_models_list.txt")
)
HF_MODELS_FILE_NAME = model_names(
    os.path.join(os.path.dirname(__file__), "huggingface_models_list.txt")
)
TORCHBENCH_MODELS_FILE_NAME = model_names(
    os.path.join(os.path.dirname(__file__), "all_torchbench_models_list.txt")
)

# timm <> HF disjoint
assert TIMM_MODEL_NAMES.isdisjoint(HF_MODELS_FILE_NAME)
# timm <> torch disjoint
assert TIMM_MODEL_NAMES.isdisjoint(TORCHBENCH_MODELS_FILE_NAME)
# torch <> hf disjoint
assert TORCHBENCH_MODELS_FILE_NAME.isdisjoint(HF_MODELS_FILE_NAME)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        help="""Run just one model from whichever model suite it belongs to. Or
        specify the path and class name of the model in format like:
        --only=path:<MODEL_FILE_PATH>,class:<CLASS_NAME>

        Due to the fact that dynamo changes current working directory,
        the path should be an absolute path.

        The class should have a method get_example_inputs to return the inputs
        for the model. An example looks like
        ```
        class LinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

            def get_example_inputs(self):
                return (torch.randn(2, 10),)
        ```
    """,
    )
    return parser.parse_known_args(args)


if __name__ == "__main__":
    args, unknown = parse_args()
    if args.only:
        name = args.only
        if name in TIMM_MODEL_NAMES:
            import timm_models

            timm_models.timm_main()
        elif name in HF_MODELS_FILE_NAME:
            import huggingface

            huggingface.huggingface_main()
        elif name in TORCHBENCH_MODELS_FILE_NAME:
            import torchbench

            torchbench.torchbench_main()
        else:
            print(f"Illegal model name? {name}")
            sys.exit(-1)
    else:
        import torchbench

        torchbench.torchbench_main()

        import huggingface

        huggingface.huggingface_main()

        import timm_models

        timm_models.timm_main()

```



## High-Level Overview


This Python file contains 3 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `LinearModel`

**Functions defined**: `model_names`, `parse_args`, `__init__`, `forward`, `get_example_inputs`

**Key imports**: argparse, os, sys, timm_models, huggingface, torchbench, torchbench, huggingface, timm_models


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `benchmarks/dynamo`, which is part of the PyTorch project infrastructure.



## Dependencies

### Import Dependencies

This file imports:

- `argparse`
- `os`
- `sys`
- `timm_models`
- `huggingface`
- `torchbench`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes


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

Files in the same folder (`benchmarks/dynamo`):

- [`timm_models_list_cpu.txt_docs.md`](./timm_models_list_cpu.txt_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test.py_docs.md`](./test.py_docs.md)
- [`check_graph_breaks.py_docs.md`](./check_graph_breaks.py_docs.md)
- [`check_csv.py_docs.md`](./check_csv.py_docs.md)
- [`all_torchbench_models_list.txt_docs.md`](./all_torchbench_models_list.txt_docs.md)
- [`check_accuracy.py_docs.md`](./check_accuracy.py_docs.md)
- [`torchbench_models_list_cpu.txt_docs.md`](./torchbench_models_list_cpu.txt_docs.md)
- [`timm_models.py_docs.md`](./timm_models.py_docs.md)


## Cross-References

- **File Documentation**: `benchmarks.py_docs.md`
- **Keyword Index**: `benchmarks.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
