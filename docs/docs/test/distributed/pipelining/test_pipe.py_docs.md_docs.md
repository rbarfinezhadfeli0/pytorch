# Documentation: `docs/test/distributed/pipelining/test_pipe.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/pipelining/test_pipe.py_docs.md`
- **Size**: 6,223 bytes (6.08 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/pipelining/test_pipe.py`

## File Metadata

- **Path**: `test/distributed/pipelining/test_pipe.py`
- **Size**: 3,442 bytes (3.36 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
from model_registry import MLPModule, ModelWithParamAlias

import torch
from torch.distributed.pipelining import pipe_split, pipeline
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


d_hid = 512
microbatch_size = 16

torch.manual_seed(0)


# Basic example
class ExampleCode(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin1 = torch.nn.Linear(d_hid, d_hid)
        self.lin2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x, y):
        x = torch.mm(x, self.mm_param1)  # mutli-use param
        skip_connection = x
        x = x + y
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param1)  # mutli-use param
        x = self.lin1(x)
        pipe_split()
        x = torch.relu(x)
        x = x + skip_connection
        x = torch.mm(x, self.mm_param2)
        pipe_split()
        x = self.lin2(x)
        x = torch.relu(x)
        return x


class MultiMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp0 = MLPModule(d_hid)
        self.mlp1 = MLPModule(d_hid)
        self.mlp2 = MLPModule(d_hid)
        self.mlp3 = MLPModule(d_hid)

    def forward(self, x, y):
        x = self.mlp0(x)
        pipe_split()
        x = self.mlp1(x)
        pipe_split()
        x = self.mlp2(x)
        pipe_split()
        x = self.mlp3(x)
        return x - y


EXPECTED_N_STAGES = {
    ExampleCode: 4,
    MultiMLP: 4,
    ModelWithParamAlias: 2,
}

# Currently, we don't enforce full set equality on the FQNs between the original
# and pipelined models, because in the multi-use param case, PP will deduplicate
# the FQNs from the state_dict.
# TODO
CHECK_FQN_SET_EQUALITY = False


class PipeTests(TestCase):
    @parametrize("ModelClass", [ExampleCode, MultiMLP, ModelWithParamAlias])
    def test_model_split(self, ModelClass):
        mod = ModelClass()
        x = torch.randn(microbatch_size, d_hid)
        y = torch.randn(microbatch_size, d_hid)

        pipe = pipeline(
            mod,
            mb_args=(x, y),
        )

        assert pipe.num_stages == EXPECTED_N_STAGES[ModelClass], (
            f"nstages = {pipe.num_stages}, expect {EXPECTED_N_STAGES[ModelClass]}"
        )

        ref_out = mod(x, y)
        out = pipe(x, y)[0]
        torch.testing.assert_close(out, ref_out)
        print(f"equivalence test passed {torch.sum(out)} ref {torch.sum(ref_out)}")

        # Check qualname
        # state_dict.keys include both parameters and persistent buffers
        old_names = set(mod.state_dict().keys())
        new_names = set()
        for idx in range(pipe.num_stages):
            stage_mod = pipe.get_stage_module(idx)
            stage_fqns = set(stage_mod.state_dict().keys())
            assert stage_fqns.issubset(old_names)
            new_names.update(stage_fqns)

        if CHECK_FQN_SET_EQUALITY:
            assert old_names == new_names, f"""
            old names {old_names}
            new names {new_names}
            """
        print("Qualname check passed")


instantiate_parametrized_tests(PipeTests)

if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 3 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `ExampleCode`, `MultiMLP`, `PipeTests`

**Functions defined**: `__init__`, `forward`, `__init__`, `forward`, `test_model_split`

**Key imports**: MLPModule, ModelWithParamAlias, torch, pipe_split, pipeline


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/pipelining`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `model_registry`: MLPModule, ModelWithParamAlias
- `torch`
- `torch.distributed.pipelining`: pipe_split, pipeline


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
python test/distributed/pipelining/test_pipe.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/pipelining`):

- [`test_schedule_multiproc.py_docs.md`](./test_schedule_multiproc.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_microbatch.py_docs.md`](./test_microbatch.py_docs.md)
- [`test_schedule.py_docs.md`](./test_schedule.py_docs.md)
- [`model_registry.py_docs.md`](./model_registry.py_docs.md)
- [`test_transformer.py_docs.md`](./test_transformer.py_docs.md)
- [`test_stage.py_docs.md`](./test_stage.py_docs.md)
- [`schedule_registry.py_docs.md`](./schedule_registry.py_docs.md)
- [`test_unflatten.py_docs.md`](./test_unflatten.py_docs.md)


## Cross-References

- **File Documentation**: `test_pipe.py_docs.md`
- **Keyword Index**: `test_pipe.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/pipelining`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/pipelining`, which is part of the **testing infrastructure**.



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
python docs/test/distributed/pipelining/test_pipe.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/pipelining`):

- [`test_transformer.py_kw.md_docs.md`](./test_transformer.py_kw.md_docs.md)
- [`test_schedule_multiproc.py_docs.md_docs.md`](./test_schedule_multiproc.py_docs.md_docs.md)
- [`model_registry.py_kw.md_docs.md`](./model_registry.py_kw.md_docs.md)
- [`test_unflatten.py_docs.md_docs.md`](./test_unflatten.py_docs.md_docs.md)
- [`schedule_registry.py_docs.md_docs.md`](./schedule_registry.py_docs.md_docs.md)
- [`test_stage.py_docs.md_docs.md`](./test_stage.py_docs.md_docs.md)
- [`schedule_registry.py_kw.md_docs.md`](./schedule_registry.py_kw.md_docs.md)
- [`test_schedule_multiproc.py_kw.md_docs.md`](./test_schedule_multiproc.py_kw.md_docs.md)
- [`test_unflatten.py_kw.md_docs.md`](./test_unflatten.py_kw.md_docs.md)
- [`__init__.py_docs.md_docs.md`](./__init__.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_pipe.py_docs.md_docs.md`
- **Keyword Index**: `test_pipe.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
