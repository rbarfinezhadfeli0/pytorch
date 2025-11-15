# Documentation: `torch/_inductor/fx_passes/serialized_patterns/_sfdp_pattern_13.py`

## File Metadata

- **Path**: `torch/_inductor/fx_passes/serialized_patterns/_sfdp_pattern_13.py`
- **Size**: 7,862 bytes (7.68 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This is a python source code that is part of the PyTorch project.

## Original Source

```python
# mypy: ignore-errors

# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

import torch
import torch._inductor
import operator

aten = torch.ops.aten
prims = torch.ops.prims

from torch._inductor.pattern_matcher import (
   Arg,
   CallFunction,
   CallFunctionVarArgs,
   CallMethod,
   CallMethodVarArgs,
   CallModule,
   CallModuleVarArgs,
   ExclusiveKeywordArg,
   Ignored,
   KeywordArg,
   ListOf,
   MultiOutputPattern,
   PatternExpr,
   RepeatedExpr,
   _TargetArgsExpr,
   _TargetExpr,
   _TargetExprVarArgs,
)
rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored(), _users=2)
bmm_default = CallFunction(aten.bmm.default, KeywordArg('query'), permute_default, _users=2)
amax_default = CallFunction(aten.amax.default, bmm_default, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, bmm_default, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, div_Tensor)
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored(), _users=2)
bmm_default_1 = CallFunction(aten.bmm.default, mul_Tensor_1, KeywordArg('value'))
neg_default = CallFunction(aten.neg.default, div_Tensor)
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
bmm_default_2 = CallFunction(aten.bmm.default, KeywordArg('tangents_1'), permute_default_1)
convert_element_type_default = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default, Ignored())
mul_Tensor_3 = CallFunction(aten.mul.Tensor, bmm_default_2, mul_Tensor_2)
mul_Tensor_4 = CallFunction(aten.mul.Tensor, mul_Tensor_3, div_Tensor, _users=2)
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4, _users=2)
permute_default_2 = CallFunction(aten.permute.default, permute_default, Ignored())
bmm_default_3 = CallFunction(aten.bmm.default, fma_default, permute_default_2)
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_3, fma_default)
permute_default_4 = CallFunction(aten.permute.default, bmm_default_4, Ignored())
permute_default_5 = CallFunction(aten.permute.default, mul_Tensor_1, Ignored())
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_5, KeywordArg('tangents_1'))
_sfdp_pattern_13_training = MultiOutputPattern([bmm_default_1,
  bmm_default_3,
  permute_default_4,
  bmm_default_5,
  None
])


permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
bmm_default = CallFunction(aten.bmm.default, KeywordArg('query'), permute_default, _users=2)
amax_default = CallFunction(aten.amax.default, bmm_default, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, bmm_default, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
_sfdp_pattern_13_inference = CallFunction(aten.bmm.default, div_Tensor, KeywordArg('value'), _users=0)


rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored(), _users=2)
bmm_default = CallFunction(aten.bmm.default, KeywordArg('query'), permute_default)
convert_element_type_default = CallFunction(prims.convert_element_type.default, bmm_default, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored(), _users=2)
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, convert_element_type_default_1)
mul_Tensor_1 = CallFunction(aten.mul.Tensor, mul_Tensor, Ignored(), _users=2)
bmm_default_1 = CallFunction(aten.bmm.default, mul_Tensor_1, KeywordArg('value'))
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, convert_element_type_default_1, Ignored(), _users=2)
neg_default = CallFunction(aten.neg.default, convert_element_type_default_2)
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
bmm_default_2 = CallFunction(aten.bmm.default, KeywordArg('tangents_1'), permute_default_1)
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default_3, Ignored())
mul_Tensor_3 = CallFunction(aten.mul.Tensor, bmm_default_2, mul_Tensor_2)
convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, mul_Tensor_3, Ignored())
mul_Tensor_4 = CallFunction(aten.mul.Tensor, convert_element_type_default_4, convert_element_type_default_2, _users=2)
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_4, Ignored(), True)
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_4)
convert_element_type_default_5 = CallFunction(prims.convert_element_type.default, fma_default, Ignored(), _users=2)
permute_default_2 = CallFunction(aten.permute.default, permute_default, Ignored())
bmm_default_3 = CallFunction(aten.bmm.default, convert_element_type_default_5, permute_default_2)
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_3, convert_element_type_default_5)
permute_default_4 = CallFunction(aten.permute.default, bmm_default_4, Ignored())
permute_default_5 = CallFunction(aten.permute.default, mul_Tensor_1, Ignored())
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_5, KeywordArg('tangents_1'))
_sfdp_pattern_13_half_training = MultiOutputPattern([bmm_default_1,
  bmm_default_3,
  permute_default_4,
  bmm_default_5,
  None
])


permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
bmm_default = CallFunction(aten.bmm.default, KeywordArg('query'), permute_default)
convert_element_type_default = CallFunction(prims.convert_element_type.default, bmm_default, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored())
_sfdp_pattern_13_half_inference = CallFunction(aten.bmm.default, convert_element_type_default_1, KeywordArg('value'), _users=0)

```



## High-Level Overview


This Python file contains 0 class(es) and 0 function(s).

## Detailed Analysis

### Code Structure

**Key imports**: torch, torch._inductor, operator


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/_inductor/fx_passes/serialized_patterns`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._inductor`
- `operator`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


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

Files in the same folder (`torch/_inductor/fx_passes/serialized_patterns`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`_sfdp_pattern_21.py_docs.md`](./_sfdp_pattern_21.py_docs.md)
- [`_sfdp_pattern_4.py_docs.md`](./_sfdp_pattern_4.py_docs.md)
- [`_sfdp_pattern_18.py_docs.md`](./_sfdp_pattern_18.py_docs.md)
- [`_sfdp_pattern_12.py_docs.md`](./_sfdp_pattern_12.py_docs.md)
- [`_sfdp_pattern_16.py_docs.md`](./_sfdp_pattern_16.py_docs.md)
- [`_sfdp_pattern_14.py_docs.md`](./_sfdp_pattern_14.py_docs.md)
- [`_sfdp_pattern_8.py_docs.md`](./_sfdp_pattern_8.py_docs.md)
- [`_sfdp_pattern_10.py_docs.md`](./_sfdp_pattern_10.py_docs.md)
- [`_sfdp_pattern_1.py_docs.md`](./_sfdp_pattern_1.py_docs.md)


## Cross-References

- **File Documentation**: `_sfdp_pattern_13.py_docs.md`
- **Keyword Index**: `_sfdp_pattern_13.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
