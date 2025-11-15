# Documentation: generate_test_torchscripts.py

## File Metadata
- **Path**: `android/pytorch_android/generate_test_torchscripts.py`
- **Size**: 4211 bytes
- **Lines**: 151
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
from typing import Optional

import torch
from torch import Tensor


OUTPUT_DIR = "src/androidTest/assets/"


def scriptAndSave(module, fileName):
    print("-" * 80)
    script_module = torch.jit.script(module)
    print(script_module.graph)
    outputFileName = OUTPUT_DIR + fileName
    # note that the lite interpreter model can also be used in full JIT
    script_module._save_for_lite_interpreter(outputFileName)
    print("Saved to " + outputFileName)
    print("=" * 80)


class Test(torch.jit.ScriptModule):
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


scriptAndSave(Test(), "test.pt")

```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 1 class(es): Test

### Functions
This file defines 25 function(s): scriptAndSave, forward, eqBool, eqInt, eqFloat, eqStr, eqTensor, eqDictStrKeyIntValue, eqDictIntKeyIntValue, eqDictFloatKeyIntValue, listIntSumReturnTuple, listBoolConjunction, listBoolDisjunction, tupleIntSumReturnTuple, optionalIntIsNone, intEq0None, str3Concat, newEmptyShapeWithItem, testAliasWithOffset, testNonContiguous, conv2d, conv3d, contiguous, contiguousChannelsLast, contiguousChannelsLast3d


## Key Components

The file contains 389 words across 151 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 4211 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
