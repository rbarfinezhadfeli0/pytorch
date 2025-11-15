# Documentation: `docs/test/export/test_sparse.py_docs.md`

## File Metadata

- **Path**: `docs/test/export/test_sparse.py_docs.md`
- **Size**: 12,294 bytes (12.01 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/export/test_sparse.py`

## File Metadata

- **Path**: `test/export/test_sparse.py`
- **Size**: 9,228 bytes (9.01 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: sparse"]
#
# Test to ensure sparsity information propagates properly into traced graph.
#

import sys
import unittest

import torch
from torch._environment import is_fbcode
from torch._subclasses.fake_tensor import FakeTensor
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)


# Various data types (preserved over operations).
DTYPES = [
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
]

# Various index types.
ITYPES = [torch.int32, torch.int64]


# Constructs a subtest for every sparse layout currently supported in torch.sparse.
def all_sparse_layouts(test_name="layout"):
    return parametrize(
        test_name,
        [
            subtest(torch.sparse_coo, name="SparseCOO"),
            subtest(torch.sparse_csr, name="SparseCSR"),
            subtest(torch.sparse_csc, name="SparseCSC"),
            subtest(torch.sparse_bsr, name="SparseBSR"),
            subtest(torch.sparse_bsc, name="SparseBSC"),
        ],
    )


#
# Various network examples.
#


class IdNet(torch.nn.Module):
    def forward(self, x):
        return x


class SumNet(torch.nn.Module):
    def forward(self, x):
        return x.sum()


class EltwiseNet(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.relu(2 * torch.abs(-x))


class ToDenseNet(torch.nn.Module):
    def forward(self, x):
        return x.to_dense()


class AddNet(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y)


class SparseActivationCOO(torch.nn.Module):
    def forward(self, x):
        return [xi.to_sparse() for xi in x]


class SparseActivationCSR(torch.nn.Module):
    def forward(self, x):
        return [xi.to_sparse_csr() for xi in x]


#
# The test driver.
#


@unittest.skipIf(is_fbcode(), "See torch._dynamo.config")
@unittest.skipIf(
    sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
)
class TestSparseProp(TestCase):
    def setUp(self):
        super().setUp()

    def assertEqualMeta(self, x, y):
        self.assertIsInstance(x, FakeTensor)
        self.assertIsInstance(y, torch.Tensor)

        # Convert expected value to meta for comparison.
        y = y.to("meta")
        self.assertEqual(x, y, exact_layout=True, exact_is_coalesced=True)

        # When x or y is a meta tensor (say, `x.device == "meta"`), then
        # assertEqual(x, y) compares only x and y attributes but skips
        # comparing their values. In the case of sparse tensors, this means
        # that comparing indices and values attributes are skipped as well,
        # which is why we are doing that explicitly below.
        if x.layout is torch.strided:
            pass
        elif x.layout is torch.sparse_coo:
            self.assertEqual(x._indices(), y._indices(), exact_layout=True)
            self.assertEqual(x._values(), y._values(), exact_layout=True)
        else:
            if x.layout in {torch.sparse_csr, torch.sparse_bsr}:
                x_meta1, y_meta1 = (x.crow_indices(), y.crow_indices())
                x_meta2, y_meta2 = (x.col_indices(), y.col_indices())
            elif x.layout in {torch.sparse_csc, torch.sparse_bsc}:
                x_meta1, y_meta1 = (x.ccol_indices(), y.ccol_indices())
                x_meta2, y_meta2 = (x.row_indices(), y.row_indices())
            else:
                assert 0  # unreachable
            self.assertEqual(x_meta1, y_meta1, exact_layout=True)
            self.assertEqual(x_meta2, y_meta2, exact_layout=True)
            self.assertEqual(x.values(), y.values(), exact_layout=True)

    @parametrize("dtype", DTYPES)
    @parametrize("itype", ITYPES)
    @all_sparse_layouts("layout")
    def test_idnet(self, dtype, itype, layout):
        net = IdNet()
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            # Build the traced graph.
            prog = torch.export.export(net, (sparse_input,), strict=True)
            # Test arg/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("val", None)
                if i == 0:
                    self.assertEqualMeta(meta, sparse_input)
                else:
                    self.assertEqual(meta, None)

    @parametrize("dtype", DTYPES)
    @parametrize("itype", ITYPES)
    @all_sparse_layouts("layout")
    def test_sumnet(self, dtype, itype, layout):
        net = SumNet()
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            result = net(sparse_input)
            # Build the traced graph.
            prog = torch.export.export(net, (sparse_input,), strict=True)
            # Test arg/sum/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("val", None)
                if i == 0:
                    self.assertEqualMeta(meta, sparse_input)
                elif i == 1:
                    self.assertEqualMeta(meta, result)
                else:
                    self.assertEqual(meta, None)

    @parametrize("dtype", DTYPES)
    @parametrize("itype", ITYPES)
    @all_sparse_layouts("layout")
    def test_eltwisenet(self, dtype, itype, layout):
        net = EltwiseNet()
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            result = net(sparse_input)
            # Build the traced graph.
            prog = torch.export.export(net, (sparse_input,), strict=True)
            # Test arg/neg/abs/mul/relu/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("val", None)
                if i <= 4:
                    self.assertEqualMeta(meta, result)
                else:
                    self.assertEqual(meta, None)

    @parametrize("dtype", DTYPES)
    @parametrize("itype", ITYPES)
    @all_sparse_layouts("layout")
    def test_todensenet(self, dtype, itype, layout):
        net = ToDenseNet()
        for sparse_input in self.generate_simple_inputs(
            layout,
            device="cpu",
            dtype=dtype,
            index_dtype=itype,
        ):
            result = net(sparse_input)
            # Build the traced graph.
            prog = torch.export.export(net, (sparse_input,), strict=True)
            # Test arg/todense/output.
            for i, node in enumerate(prog.graph.nodes):
                meta = node.meta.get("val", None)
                if i == 0:
                    self.assertEqualMeta(meta, sparse_input)
                elif i == 1:
                    self.assertEqualMeta(meta, result)
                else:
                    self.assertEqual(meta, None)

    def test_add(self):
        net = AddNet()
        Y = torch.arange(16, 32, dtype=torch.float32).view(4, 4)
        A = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 1.0],
                [3.0, 0.0, 3.0, 0.0],
            ],
            dtype=torch.float32,
        )
        S = A.to_sparse_csr()
        result = net(S, Y)
        # Build the traced graph.
        prog = torch.export.export(net, (S, Y), strict=True)
        # Test args/add/output.
        for i, node in enumerate(prog.graph.nodes):
            meta = node.meta.get("val", None)
            if i == 0:
                self.assertEqualMeta(meta, S)
            elif i == 1:
                self.assertEqualMeta(meta, Y)
            elif i == 2:
                self.assertEqualMeta(meta, result)
            else:
                self.assertEqual(meta, None)

    def test_activation_coo(self):
        net = SparseActivationCOO()
        x = [torch.randn(3, 3) for _ in range(3)]
        result = net(x)
        # Build the traced graph.
        prog = torch.export.export(net, args=(x,), strict=True)
        # Test args/to_sparse/output.
        for i, node in enumerate(prog.graph.nodes):
            meta = node.meta.get("val", None)
            if i <= 2:
                self.assertEqualMeta(meta, x[i])
            elif i <= 5:
                self.assertEqualMeta(meta, result[i - 3])
            else:
                self.assertEqual(meta, None)

    def test_activation_csr(self):
        net = SparseActivationCSR()
        x = [torch.randn(3, 3) for _ in range(3)]
        result = net(x)
        # Build the traced graph.
        prog = torch.export.export(net, args=(x,), strict=True)
        # Test args/to_sparse/output.
        for i, node in enumerate(prog.graph.nodes):
            meta = node.meta.get("val", None)
            if i <= 2:
                self.assertEqualMeta(meta, x[i])
            elif i <= 5:
                self.assertEqualMeta(meta, result[i - 3])
            else:
                self.assertEqual(meta, None)


instantiate_parametrized_tests(TestSparseProp)

if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 8 class(es) and 17 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `IdNet`, `SumNet`, `EltwiseNet`, `ToDenseNet`, `AddNet`, `SparseActivationCOO`, `SparseActivationCSR`, `TestSparseProp`

**Functions defined**: `all_sparse_layouts`, `forward`, `forward`, `forward`, `forward`, `forward`, `forward`, `forward`, `setUp`, `assertEqualMeta`, `test_idnet`, `test_sumnet`, `test_eltwisenet`, `test_todensenet`, `test_add`, `test_activation_coo`, `test_activation_csr`

**Key imports**: sys, unittest, torch, is_fbcode, FakeTensor


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/export`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `unittest`
- `torch`
- `torch._environment`: is_fbcode
- `torch._subclasses.fake_tensor`: FakeTensor


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
python test/export/test_sparse.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/export`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_schema.py_docs.md`](./test_schema.py_docs.md)
- [`test_package.py_docs.md`](./test_package.py_docs.md)
- [`test_cpp_serdes.py_docs.md`](./test_cpp_serdes.py_docs.md)
- [`test_export_opinfo.py_docs.md`](./test_export_opinfo.py_docs.md)
- [`test_lift_unlift.py_docs.md`](./test_lift_unlift.py_docs.md)
- [`test_retraceability.py_docs.md`](./test_retraceability.py_docs.md)
- [`test_converter.py_docs.md`](./test_converter.py_docs.md)
- [`test_nativert.py_docs.md`](./test_nativert.py_docs.md)
- [`test_export.py_docs.md`](./test_export.py_docs.md)


## Cross-References

- **File Documentation**: `test_sparse.py_docs.md`
- **Keyword Index**: `test_sparse.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/export`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/export`, which is part of the **testing infrastructure**.



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
python docs/test/export/test_sparse.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/export`):

- [`test_serialize.py_docs.md_docs.md`](./test_serialize.py_docs.md_docs.md)
- [`test_verifier.py_kw.md_docs.md`](./test_verifier.py_kw.md_docs.md)
- [`test_upgrader.py_kw.md_docs.md`](./test_upgrader.py_kw.md_docs.md)
- [`test_db.py_docs.md_docs.md`](./test_db.py_docs.md_docs.md)
- [`test_export.py_docs.md_docs.md`](./test_export.py_docs.md_docs.md)
- [`test_dynamic_shapes.py_kw.md_docs.md`](./test_dynamic_shapes.py_kw.md_docs.md)
- [`test_passes.py_kw.md_docs.md`](./test_passes.py_kw.md_docs.md)
- [`test_unflatten.py_docs.md_docs.md`](./test_unflatten.py_docs.md_docs.md)
- [`test_pass_infra.py_kw.md_docs.md`](./test_pass_infra.py_kw.md_docs.md)
- [`test_functionalized_assertions.py_kw.md_docs.md`](./test_functionalized_assertions.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_sparse.py_docs.md_docs.md`
- **Keyword Index**: `test_sparse.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
