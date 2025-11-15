# Documentation: `docs/test/distributed/tensor/test_common_rules.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/tensor/test_common_rules.py_docs.md`
- **Size**: 20,066 bytes (19.60 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/tensor/test_common_rules.py`

## File Metadata

- **Path**: `test/distributed/tensor/test_common_rules.py`
- **Size**: 16,758 bytes (16.37 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSchema
from torch.distributed.tensor._ops._common_rules import einop_rule, pointwise_rule
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorContinuousTestBase,
)


aten = torch.ops.aten


class CommonRulesTest(DTensorContinuousTestBase):
    # hard code world size to 4 as we need to test
    # at least with 2d mesh
    world_size = 4

    def _gen_tensor_meta(self, shape):
        empty_tensor = torch.empty(shape)
        return TensorMeta(
            empty_tensor.shape,
            empty_tensor.stride(),
            empty_tensor.dtype,
        )

    def test_einop_basic_propagation(self):
        # plain einsum, mm
        mesh = DeviceMesh(self.device_type(), torch.arange(self.world_size))

        mm_call = aten.mm.default
        # propagate col-wise sharding
        mat1, mat2 = [-1, -1], [-1, 0]

        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([4, 8]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [-1, 0])

        # propagate row-wise sharding
        mat1, mat2 = [0, -1], [-1, -1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [0, -1])

        # generate partial
        mat1, mat2 = [-1, 0], [0, -1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertTrue(output_spec.placements[0].is_partial())

    def test_einop_pointwise_propagation(self):
        mesh = DeviceMesh(self.device_type(), torch.arange(self.world_size))

        add_call = aten.add.Tensor
        # addition
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 8]))
        mat1 = [0, -1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        output_sharding = einop_rule(
            "ij,ij->ij", OpSchema(add_call, (mat1_spec, mat1_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [0, -1])

        # broadcast addition
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 8]))
        mat1 = [-1, 0, -1]
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )

        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([2]))
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, [-1], [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = einop_rule(
            "ijk,k->ijk", OpSchema(add_call, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [-1, 0, -1])

        # broadcast to a common shape
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 8, 8]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([1, 8]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, [0, -1, -1], [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, [-1, -1], [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = einop_rule(
            "ijk,1k->ijk", OpSchema(add_call, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [0, -1, -1])

    def test_einop_merge_sharding(self):
        # 2d mesh einop merge sharding
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        mesh = DeviceMesh(self.device_type(), mesh_shape)

        mm_call = aten.mm.default

        mat1, mat2 = [0, -1], [-1, 1]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([4, 8]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [0, 1])

    def test_einop_linearity(self):
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        mesh = DeviceMesh(self.device_type(), mesh_shape)

        mm_call = aten.mm.default

        mat1, mat2 = [0, -1], [-1, -1]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([4, 8]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [1], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        # if not turn on linearity, partial sum is not eligible to propagate, we return
        # suggestion to reshard inputs with no partial sum (i.e. all_reduce one input)
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )
        self.assertIsNone(output_sharding.output_spec)
        suggestions = output_sharding.redistribute_schema
        self.assertIsNotNone(suggestions)
        suggested_spec = suggestions.args_schema[0]
        self.assertFalse(suggested_spec.placements[1].is_partial())

        # einop prop with linearity on mm, should give back suggestion
        # on converting placements to partial
        output_sharding = einop_rule(
            "mk,kn->mn",
            OpSchema(mm_call, (mat1_spec, mat2_spec), {}),
            linearity=True,
        )
        self.assertIsNone(output_sharding.output_spec)
        suggestions = output_sharding.redistribute_schema
        self.assertIsNotNone(suggestions)
        mat2_spec = suggestions.args_schema[1]
        # mat2 mesh dim 1 should become partial now!
        self.assertTrue(mat2_spec.placements[1].is_partial())

        # einop prop with linearity on point-wise, should give back suggestion
        # on converting placements to partial
        add_call = aten.add.Tensor
        mat1, mat2 = [0, -1], [0, -1]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 6]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([8, 6]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [1], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )

        output_sharding = einop_rule(
            "ij,ij->ij",
            OpSchema(add_call, (mat1_spec, mat2_spec), {}),
            linearity=True,
        )
        self.assertIsNone(output_sharding.output_spec)
        suggestions = output_sharding.redistribute_schema
        self.assertIsNotNone(suggestions)
        mat2_spec = suggestions.args_schema[1]
        # mat2 mesh dim 1 should become partial now!
        self.assertTrue(mat2_spec.placements[1].is_partial())

    def test_einop_multi_sharding_on_mesh_dim(self):
        # einop prop with multi sharding on same mesh dim
        mesh_shape = torch.arange(self.world_size)
        mesh = DeviceMesh(self.device_type(), mesh_shape)

        mm_call = aten.mm.default
        mat1, mat2 = [0, -1], [0, -1]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 12]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([12, 4]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = einop_rule(
            "mk,kn->mn", OpSchema(mm_call, (mat1_spec, mat2_spec), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNone(output_spec)
        self.assertIsNotNone(output_sharding.redistribute_schema)

        # ensure that the suggestion is to reshard the second
        # arg by all_gather its tensor dim sharding
        schema_suggestion = output_sharding.redistribute_schema
        self.assertEqual(schema_suggestion.args_schema[0].dim_map, [0, -1])
        self.assertEqual(schema_suggestion.args_schema[1].dim_map, [-1, -1])

    def test_einop_errors(self):
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        mesh = DeviceMesh(self.device_type(), mesh_shape)

        add_call = aten.add.Tensor
        mat1, mat2 = [0, -1], [1, -1]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )

        with self.assertRaisesRegex(RuntimeError, "sharded two different ways:"):
            einop_rule("ij,ij->ij", OpSchema(add_call, (mat1_spec, mat2_spec), {}))

    def test_pointwise_rules_broadcasting(self):
        mesh = DeviceMesh(self.device_type(), torch.arange(self.world_size))

        where_call = aten.where.self
        inp1, inp2, inp3 = [0], [], [-1, -1]
        inp1_tensor_meta = self._gen_tensor_meta(torch.Size([8]))
        inp2_tensor_meta = self._gen_tensor_meta(torch.Size([]))
        inp3_tensor_meta = self._gen_tensor_meta(torch.Size([1, 1]))
        condition = DTensorSpec.from_dim_map(
            mesh, inp1, [], tensor_meta=inp1_tensor_meta
        )
        self_tensor = DTensorSpec.from_dim_map(
            mesh, inp2, [], tensor_meta=inp2_tensor_meta
        )
        other_tensor = DTensorSpec.from_dim_map(
            mesh, inp3, [], tensor_meta=inp3_tensor_meta
        )
        # propagate point-wise sharding with broadcasting
        output_sharding = pointwise_rule(
            OpSchema(where_call, (condition, self_tensor, other_tensor), {})
        )
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [-1, 0])

    def test_pointwise_rules_suggestion(self):
        mesh = DeviceMesh(self.device_type(), torch.arange(self.world_size))

        lerp_call = aten.lerp.Scalar
        # propagate point-wise sharding
        inp1, inp2 = [-1, -1], [-1, 0]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([8, 4]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, inp1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, inp2, [], tensor_meta=mat2_tensor_meta
        )
        # adding a positional argument -1 to arg schema
        output_sharding = pointwise_rule(
            OpSchema(lerp_call, (mat1_spec, mat2_spec, -1), {})
        )
        self.assertIsNone(output_sharding.output_spec)
        self.assertIsNotNone(output_sharding.redistribute_schema)

        # ensure that the suggestion from pointwise rules still have
        # the positional args that are not DTensorSpec
        schema_suggestion = output_sharding.redistribute_schema
        self.assertEqual(len(schema_suggestion.args_schema), 3)
        self.assertEqual(schema_suggestion.args_schema[2], -1)

    def test_pointwise_multi_sharding_on_mesh_dim(self):
        # 2d mesh pointwise sharding
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        mesh = DeviceMesh(self.device_type(), mesh_shape)

        add_call = aten.add.Tensor

        # basic case to test implicit broadcasting shape alignment
        mat1, mat2 = [-1, 0], [0]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([20, 6]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([6]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = pointwise_rule(OpSchema(add_call, (mat1_spec, mat2_spec), {}))
        output_spec = output_sharding.output_spec
        self.assertIsNotNone(output_spec)
        self.assertEqual(output_spec.dim_map, [-1, 0])

        # more advanced case that needs reshard one input to align sharding
        mat1, mat2 = [0, -1, -1, 1], [0, -1, 1]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([12, 1, 1, 8]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([12, 4, 8]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = pointwise_rule(OpSchema(add_call, (mat1_spec, mat2_spec), {}))
        output_spec = output_sharding.output_spec
        self.assertIsNone(output_spec)
        self.assertIsNotNone(output_sharding.redistribute_schema)

        # ensure that the suggestion is to reshard the first
        # arg by all_gather first tensor dim sharding
        schema_suggestion = output_sharding.redistribute_schema
        self.assertEqual(schema_suggestion.args_schema[0].dim_map, [-1, -1, -1, 1])
        self.assertEqual(schema_suggestion.args_schema[1].dim_map, mat2)

    def test_pointwise_enforce_sharding_multi_sharding_on_mesh_dim(self):
        # 2d mesh pointwise sharding
        mesh_shape = torch.arange(self.world_size).reshape(
            self.world_size // 2, self.world_size // 2
        )
        mesh = DeviceMesh(self.device_type(), mesh_shape)

        add_call = aten.add_.Tensor

        # more advanced case that needs reshard one input to align sharding
        mat1, mat2 = [0, -1, 1], [-1, -1, 0]
        mat1_tensor_meta = self._gen_tensor_meta(torch.Size([12, 4, 8]))
        mat2_tensor_meta = self._gen_tensor_meta(torch.Size([12, 1, 8]))
        mat1_spec = DTensorSpec.from_dim_map(
            mesh, mat1, [], tensor_meta=mat1_tensor_meta
        )
        mat2_spec = DTensorSpec.from_dim_map(
            mesh, mat2, [], tensor_meta=mat2_tensor_meta
        )
        output_sharding = pointwise_rule(OpSchema(add_call, (mat1_spec, mat2_spec), {}))
        output_spec = output_sharding.output_spec
        self.assertIsNone(output_spec)
        self.assertIsNotNone(output_sharding.redistribute_schema)

        # ensure that the suggestion is to reshard the second
        # arg as we should enforce the sharding of the first arg
        schema_suggestion = output_sharding.redistribute_schema
        self.assertEqual(schema_suggestion.args_schema[0].dim_map, mat1)
        self.assertEqual(schema_suggestion.args_schema[1].dim_map, mat1)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 1 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `CommonRulesTest`

**Functions defined**: `_gen_tensor_meta`, `test_einop_basic_propagation`, `test_einop_pointwise_propagation`, `test_einop_merge_sharding`, `test_einop_linearity`, `test_einop_multi_sharding_on_mesh_dim`, `test_einop_errors`, `test_pointwise_rules_broadcasting`, `test_pointwise_rules_suggestion`, `test_pointwise_multi_sharding_on_mesh_dim`, `test_pointwise_enforce_sharding_multi_sharding_on_mesh_dim`

**Key imports**: torch, DeviceMesh, DTensorSpec, TensorMeta, OpSchema, einop_rule, pointwise_rule, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/tensor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.distributed.tensor`: DeviceMesh
- `torch.distributed.tensor._dtensor_spec`: DTensorSpec, TensorMeta
- `torch.distributed.tensor._op_schema`: OpSchema
- `torch.distributed.tensor._ops._common_rules`: einop_rule, pointwise_rule
- `torch.testing._internal.common_utils`: run_tests


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

This is a test file. Run it with:

```bash
python test/distributed/tensor/test_common_rules.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/tensor`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_dtensor.py_docs.md`](./test_dtensor.py_docs.md)
- [`test_dtensor_testbase.py_docs.md`](./test_dtensor_testbase.py_docs.md)
- [`test_optimizers.py_docs.md`](./test_optimizers.py_docs.md)
- [`test_dtensor_dispatch_overhead.py_docs.md`](./test_dtensor_dispatch_overhead.py_docs.md)
- [`test_tensor_ops.py_docs.md`](./test_tensor_ops.py_docs.md)
- [`test_matrix_ops.py_docs.md`](./test_matrix_ops.py_docs.md)
- [`test_op_schema.py_docs.md`](./test_op_schema.py_docs.md)
- [`test_utils.py_docs.md`](./test_utils.py_docs.md)
- [`test_attention.py_docs.md`](./test_attention.py_docs.md)


## Cross-References

- **File Documentation**: `test_common_rules.py_docs.md`
- **Keyword Index**: `test_common_rules.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/tensor`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/tensor`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


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
python docs/test/distributed/tensor/test_common_rules.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/tensor`):

- [`test_math_ops.py_docs.md_docs.md`](./test_math_ops.py_docs.md_docs.md)
- [`test_view_ops.py_kw.md_docs.md`](./test_view_ops.py_kw.md_docs.md)
- [`test_dtensor_export.py_docs.md_docs.md`](./test_dtensor_export.py_docs.md_docs.md)
- [`test_placement_types.py_docs.md_docs.md`](./test_placement_types.py_docs.md_docs.md)
- [`test_convolution_ops.py_kw.md_docs.md`](./test_convolution_ops.py_kw.md_docs.md)
- [`test_placement_types.py_kw.md_docs.md`](./test_placement_types.py_kw.md_docs.md)
- [`test_common_rules.py_kw.md_docs.md`](./test_common_rules.py_kw.md_docs.md)
- [`test_dtensor_compile.py_kw.md_docs.md`](./test_dtensor_compile.py_kw.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`test_api.py_docs.md_docs.md`](./test_api.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_common_rules.py_docs.md_docs.md`
- **Keyword Index**: `test_common_rules.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
