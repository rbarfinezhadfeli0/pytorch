# Documentation: `test/mobile/model_test/android_api_module.py`

## File Metadata

- **Path**: `test/mobile/model_test/android_api_module.py`
- **Size**: 3,776 bytes (3.69 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
from typing import Optional

import torch
from torch import Tensor


class AndroidAPIModule(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, input):
        return None

    @torch.jit.script_method
    def eqBool(self, input: bool) -> bool:
        return input

    @torch.jit.script_method
    def eqInt(self, input: int) -> int:
        return input

    @torch.jit.script_method
    def eqFloat(self, input: float) -> float:
        return input

    @torch.jit.script_method
    def eqStr(self, input: str) -> str:
        return input

    @torch.jit.script_method
    def eqTensor(self, input: Tensor) -> Tensor:
        return input

    @torch.jit.script_method
    def eqDictStrKeyIntValue(self, input: dict[str, int]) -> dict[str, int]:
        return input

    @torch.jit.script_method
    def eqDictIntKeyIntValue(self, input: dict[int, int]) -> dict[int, int]:
        return input

    @torch.jit.script_method
    def eqDictFloatKeyIntValue(self, input: dict[float, int]) -> dict[float, int]:
        return input

    @torch.jit.script_method
    def listIntSumReturnTuple(self, input: list[int]) -> tuple[list[int], int]:
        sum = 0
        for x in input:
            sum += x
        return (input, sum)

    @torch.jit.script_method
    def listBoolConjunction(self, input: list[bool]) -> bool:
        res = True
        for x in input:
            res = res and x
        return res

    @torch.jit.script_method
    def listBoolDisjunction(self, input: list[bool]) -> bool:
        res = False
        for x in input:
            res = res or x
        return res

    @torch.jit.script_method
    def tupleIntSumReturnTuple(
        self, input: tuple[int, int, int]
    ) -> tuple[tuple[int, int, int], int]:
        sum = 0
        for x in input:
            sum += x
        return (input, sum)

    @torch.jit.script_method
    def optionalIntIsNone(self, input: Optional[int]) -> bool:
        return input is None

    @torch.jit.script_method
    def intEq0None(self, input: int) -> Optional[int]:
        if input == 0:
            return None
        return input

    @torch.jit.script_method
    def str3Concat(self, input: str) -> str:
        return input + input + input

    @torch.jit.script_method
    def newEmptyShapeWithItem(self, input):
        return torch.tensor([int(input.item())])[0]

    @torch.jit.script_method
    def testAliasWithOffset(self) -> list[Tensor]:
        x = torch.tensor([100, 200])
        a = [x[0], x[1]]
        return a

    @torch.jit.script_method
    def testNonContiguous(self):
        x = torch.tensor([100, 200, 300])[::2]
        assert not x.is_contiguous()
        assert x[0] == 100
        assert x[1] == 300
        return x

    @torch.jit.script_method
    def conv2d(self, x: Tensor, w: Tensor, toChannelsLast: bool) -> Tensor:
        r = torch.nn.functional.conv2d(x, w)
        if toChannelsLast:
            r = r.contiguous(memory_format=torch.channels_last)
        else:
            r = r.contiguous()
        return r

    @torch.jit.script_method
    def conv3d(self, x: Tensor, w: Tensor, toChannelsLast: bool) -> Tensor:
        r = torch.nn.functional.conv3d(x, w)
        if toChannelsLast:
            r = r.contiguous(memory_format=torch.channels_last_3d)
        else:
            r = r.contiguous()
        return r

    @torch.jit.script_method
    def contiguous(self, x: Tensor) -> Tensor:
        return x.contiguous()

    @torch.jit.script_method
    def contiguousChannelsLast(self, x: Tensor) -> Tensor:
        return x.contiguous(memory_format=torch.channels_last)

    @torch.jit.script_method
    def contiguousChannelsLast3d(self, x: Tensor) -> Tensor:
        return x.contiguous(memory_format=torch.channels_last_3d)

```



## High-Level Overview


This Python file contains 1 class(es) and 24 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `AndroidAPIModule`

**Functions defined**: `forward`, `eqBool`, `eqInt`, `eqFloat`, `eqStr`, `eqTensor`, `eqDictStrKeyIntValue`, `eqDictIntKeyIntValue`, `eqDictFloatKeyIntValue`, `listIntSumReturnTuple`, `listBoolConjunction`, `listBoolDisjunction`, `tupleIntSumReturnTuple`, `optionalIntIsNone`, `intEq0None`, `str3Concat`, `newEmptyShapeWithItem`, `testAliasWithOffset`, `testNonContiguous`, `conv2d`

**Key imports**: Optional, torch, Tensor


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/mobile/model_test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `typing`: Optional
- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/mobile/model_test/android_api_module.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/mobile/model_test`):

- [`torchvision_models.py_docs.md`](./torchvision_models.py_docs.md)
- [`gen_test_model.py_docs.md`](./gen_test_model.py_docs.md)
- [`update_production_ops.py_docs.md`](./update_production_ops.py_docs.md)
- [`math_ops.py_docs.md`](./math_ops.py_docs.md)
- [`builtin_ops.py_docs.md`](./builtin_ops.py_docs.md)
- [`README.md_docs.md`](./README.md_docs.md)
- [`nn_ops.py_docs.md`](./nn_ops.py_docs.md)
- [`model_ops.yaml_docs.md`](./model_ops.yaml_docs.md)
- [`quantization_ops.py_docs.md`](./quantization_ops.py_docs.md)


## Cross-References

- **File Documentation**: `android_api_module.py_docs.md`
- **Keyword Index**: `android_api_module.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
