# Documentation: `docs/test/test_nestedtensor.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_nestedtensor.py_docs.md`
- **Size**: 54,428 bytes (53.15 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_nestedtensor.py`

## File Metadata

- **Path**: `test/test_nestedtensor.py`
- **Size**: 359,210 bytes (350.79 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: nestedtensor"]
# ruff: noqa: F841
import ast
import io
import itertools
import math
import os
import random
import sys
import tempfile
import unittest
from functools import partial
from typing import Optional

import numpy as np

import torch
import torch._dynamo
import torch._dynamo.testing
import torch.nn
import torch.nn.functional as F
from torch.nested._internal.nested_tensor import (
    buffer_from_jagged,
    jagged_from_list,
    nested_view_from_values_offsets,
    NestedTensor,
    ViewNestedFromBuffer,
)
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FUSED_ATTENTION,
    SM70OrLater,
    SM80OrLater,
    tf32_on_and_off,
)
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
    ops,
    PYTORCH_CUDA_MEMCHECK,
    skipCPUIf,
    skipCUDAIf,
    skipCUDAIfRocm,
    skipMeta,
)
from torch.testing._internal.common_dtype import floating_types_and_half
from torch.testing._internal.common_utils import (
    decorateIf,
    freeze_rng_state,
    gradcheck,
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_WINDOWS,
    markDynamoStrictTest,
    NestedTensorTestCase,
    parametrize,
    run_tests,
    serialTest,
    skipIfSlowGradcheckEnv,
    skipIfTorchDynamo,
    subtest,
    TEST_WITH_ROCM,
    xfailIfTorchDynamo,
)
from torch.testing._internal.opinfo.core import (
    BinaryUfuncInfo,
    ReductionOpInfo,
    sample_skips_and_xfails,
    SkipRule,
    XFailRule,
)
from torch.testing._internal.opinfo.definitions.nested import _sample_njts, njt_op_db
from torch.utils._pytree import tree_flatten, tree_map_only
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts


# Tests are ported from pytorch/nestedtensor.
# This makes porting as_nested_tensor easier in the future.


def _iter_constructors():
    # yield as_nested_tensor
    yield torch.nested.nested_tensor


# Returns True if the function recompiles between inputs1 and inputs2 with the
# specified dynamic setting.
def _recompiles_for_inputs(fn, inputs1, inputs2, dynamic=True):
    compile_count = [0]

    def counter(gm, example_inputs):
        compile_count[0] += 1
        return gm

    compiled_f = torch.compile(fn, fullgraph=True, backend=counter, dynamic=dynamic)
    compiled_f(*inputs1)
    compiled_f(*inputs2)
    return compile_count[0] > 1


# Helper function to generate a pair of random nested tensors
# one is contiguous, the other is not, but they appear to have same entries
# an output nested tensor consists of
# * `len(ragged_sizes)` matrices
# * matrices[i].shape == (20, ragged_sizes[i])


def random_nt_noncontiguous_pair(ragged_sizes, device="cpu", dtype=torch.float16):
    xs = []
    for size in ragged_sizes:
        xs.append(torch.randn((size, 20), device=device, dtype=dtype))
    # contiguous nested tensor
    ys = []
    for x in xs:
        ys.append(x.transpose(-1, -2))
    nt_contiguous = torch.nested.nested_tensor(ys)
    # noncontiguous nested tensor
    n = len(ragged_sizes)
    nt_noncontiguous = torch.nested.nested_tensor(xs).transpose(-1, -2)
    return nt_contiguous, nt_noncontiguous


# Helper functions to pad a noncontiguous nested tensor
# can be replaced once to_padded_tensor supports noncontiguous memory


def noncontiguous_to_padded_tensor(input, shape=None):
    tensors = input.unbind()
    ntensors = len(tensors)
    assert ntensors > 0
    if shape is None:
        shape = []
        for size in tensors[0].shape:
            shape.append(size)
        for i in range(1, ntensors):
            new_shape = tensors[i].shape
            for j in range(len(shape)):
                shape[j] = max(shape[j], new_shape[j])
        shape = [ntensors] + shape
    result = tensors[0].new_zeros(shape)
    for itensor in range(ntensors):
        tensor = tensors[itensor]
        view = result[itensor]
        for idim in range(tensor.dim()):
            view = view.narrow(idim, 0, tensor.size(idim))
        view.copy_(tensor)
    return result


# Helper function to generate a random nested tensor


def random_nt(
    device,
    dtype,
    num_tensors,
    max_dims,
    min_dims=None,
    layout=torch.strided,
    require_non_empty=True,
):
    if min_dims is None:
        min_dims = tuple([0] * len(max_dims))

    assert len(max_dims) == len(min_dims)
    for min_dim, max_dim in zip(min_dims, max_dims):
        assert max_dim > min_dim, "random_nt: max_dim must be greater than min_dim"
        assert min_dim >= 0, "random_nt: min_dim must be non-negative"
        if require_non_empty:
            assert not (min_dim == 0 and max_dim == 1), (
                "random_nt: zero cannot be the only possible value if require_non_empty is True"
            )

    if require_non_empty:
        # Select a random idx that will be required to be non-empty
        non_zero_idx = torch.randint(low=0, high=num_tensors, size=(1,)).item()

    ts1 = []
    for i, _ in enumerate(range(num_tensors)):
        tensor_dims = []
        for min_dim, max_dim in zip(min_dims, max_dims):
            new_min_dim = min_dim
            if require_non_empty and i == non_zero_idx and min_dim == 0:
                new_min_dim = 1
            tensor_dims.append(
                torch.randint(low=new_min_dim, high=max_dim, size=(1,)).item()
            )
        t1 = torch.randn(tensor_dims, device=device, dtype=dtype)
        ts1.append(t1)

    return torch.nested.nested_tensor(ts1, device=device, dtype=dtype, layout=layout)


# Alternate approach to generating a random NT.
# dims should be something like [5, None, 10], with None indicating that a
# random ragged structure should be used
def random_nt_from_dims(
    dims, device=None, dtype=None, layout=torch.strided, requires_grad=False
):
    sizes = [
        [
            d if d is not None else torch.randint(2, 10, size=(1,)).item()
            for d in dims[1:]
        ]
        for d in range(dims[0])
    ]
    return torch.nested.nested_tensor(
        [torch.randn(*size) for size in sizes],
        device=device,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
    )


# Creates an NT matching another NT's number of components and
# shape / ragged structure for all dims specified to be -1.
def random_nt_from_similar(other, dims=None):
    if dims is None:
        return torch.randn_like(other)
    assert len(dims) == other.dim()
    assert dims[0] == -1 or dims[0] == other.size(0)

    ret_sizes = []
    for t in other.unbind():
        other_size = t.shape
        ret_size = []
        for i, d in enumerate(dims[1:]):
            if d == -1:
                ret_size.append(other_size[i])
            else:
                ret_size.append(d)
        ret_sizes.append(ret_size)

    return torch.nested.nested_tensor(
        [torch.randn(*size) for size in ret_sizes], device=other.device
    )


# makes naming nice for tests that parametrize over layout.
def layout_name(layout):
    # e.g. "torch.jagged" -> "jagged"
    return layout.__repr__().split(".")[-1]


def get_op_name(layout):
    # e.g. "<OpOverload(op='aten.sum', overload='dim_IntList')>" -> "sum"
    return layout.__name__.split(".")[0].split("_")[-1]


# Helper function for test_dummy_mha_with_nt
@torch.fx.wrap
def convert_dense_to_nested_tensor_legacy(values):
    offsets = torch.arange(
        0, values.shape[0] * values.shape[1] + 1, values.shape[1], device=values.device
    )
    metadata_cache = {"max_seqlen": values.shape[1], "min_seqlen": 1}
    nt = ViewNestedFromBuffer.apply(
        values.view(-1, values.shape[-1]), offsets, metadata_cache
    )
    return nt


# Helper function for test_dummy_mha_with_nt
@torch.fx.wrap
def convert_jagged_to_nested_tensor_legacy(
    values: torch.Tensor, offsets: torch.Tensor, max_length: int
) -> torch.Tensor:
    metadata_cache = {"max_seqlen": max_length, "min_seqlen": 1}
    nt = ViewNestedFromBuffer.apply(values, offsets, metadata_cache)
    return nt


# Helper function for test_dummy_mha_with_nt
@torch.fx.wrap
def convert_nt_to_jagged_legacy(nt):
    return buffer_from_jagged(nt)


# Helper function for test_dummy_mha_with_nt
@torch.fx.wrap
def convert_dense_to_nested_tensor(values):
    nt = torch.nested.as_nested_tensor(values, layout=torch.jagged)
    return nt


# Helper function for test_dummy_mha_with_nt
@torch.fx.wrap
def convert_jagged_to_nested_tensor(
    values: torch.Tensor, offsets: torch.Tensor, max_length: int
) -> torch.Tensor:
    nt = torch.nested.nested_tensor_from_jagged(
        values, offsets, lengths=None, min_seqlen=1, max_seqlen=max_length
    )
    return nt


# Helper function for test_dummy_mha_with_nt
def convert_nt_to_jagged(nt):
    return nt.values()


@markDynamoStrictTest
class TestNestedTensor(NestedTensorTestCase):
    @parametrize("batch_size", [2, 4])
    @parametrize("max_seq_len", [3, 5])
    @parametrize("vocab_size", [10, 20])
    def test_2d_nested_tensor(self, batch_size, max_seq_len, vocab_size):
        data = []
        nested_tensor_ref_list = []
        for _ in range(batch_size):
            if max_seq_len == 0:
                length = 0
            else:
                length = np.random.randint(low=1, high=max_seq_len)
            row = list(np.random.randint(low=0, high=vocab_size, size=(length,)))
            data.append(row)
            nested_tensor_ref_list.append(torch.Tensor(row))
        nested_tensor = torch.nested.nested_tensor(data, dtype=torch.int64)
        nested_tensor_list = nested_tensor.unbind()
        for id in range(batch_size):
            self.assertEqual(
                nested_tensor_list[id], nested_tensor_ref_list[id].type(torch.int64)
            )

    @parametrize("batch_size", [2, 4])
    @parametrize("max_seq_len", [3, 5])
    @parametrize("vocab_size", [10, 20])
    def test_3d_nested_tensor(self, batch_size, max_seq_len, vocab_size):
        data = []
        nested_tensor_ref_list = []
        for _ in range(batch_size):
            if max_seq_len == 0:
                length = 0
            else:
                length = np.random.randint(low=1, high=max_seq_len)
            row = list(np.random.randint(low=0, high=vocab_size, size=(length,)))
            row = [list(item * np.arange(max_seq_len)) for item in row]
            data.append(row)
            nested_tensor_ref_list.append(torch.Tensor(row))
        nested_tensor = torch.nested.nested_tensor(data, dtype=torch.int64)
        nested_tensor_list = nested_tensor.unbind()
        for id in range(batch_size):
            self.assertEqual(
                nested_tensor_list[id], nested_tensor_ref_list[id].type(torch.int64)
            )

    @parametrize("batch_size", [2, 4])
    @parametrize("max_seq_len", [3, 5])
    @parametrize("vocab_size", [10, 20])
    def test_3d_nested_tensor_float(self, batch_size, max_seq_len, vocab_size):
        data = []
        nested_tensor_ref_list = []
        for _ in range(batch_size):
            if max_seq_len == 0:
                length = 0
            else:
                length = np.random.randint(low=1, high=max_seq_len)
            row = list(
                np.random.randint(low=0, high=vocab_size, size=(length,)).astype(float)
            )
            row = [list(item * np.arange(max_seq_len)) for item in row]
            data.append(row)
            nested_tensor_ref_list.append(torch.Tensor(row))
        nested_tensor = torch.nested.nested_tensor(data, dtype=torch.float)
        nested_tensor_list = nested_tensor.unbind()
        for id in range(batch_size):
            self.assertEqual(
                nested_tensor_list[id], nested_tensor_ref_list[id].type(torch.float)
            )

    @torch.inference_mode()
    def _test_unbind_case(self, a, b):
        nt = torch.nested.nested_tensor([a, b])
        a1, b1 = nt.unbind()
        self.assertTrue(a is not a1)
        self.assertTrue(b is not b1)

        nt = torch.nested.nested_tensor([a, b], dtype=a.dtype)
        a1, b1 = nt.unbind(0)
        self.assertEqual(a, a1)
        self.assertEqual(b, b1)

        a = torch.randn((2, 3)).add_(1)
        nt = torch.nested.nested_tensor([a])
        self.assertEqual(a, nt.unbind(0)[0])

    @torch.inference_mode()
    def test_unbind_0(self):
        self._test_unbind_case(torch.tensor([1, 2]), torch.tensor([7, 8]))

    @torch.inference_mode()
    def test_unbind_1(self):
        self._test_unbind_case(torch.tensor([1]), torch.tensor([7]))

    @torch.inference_mode()
    def test_unbind_3(self):
        self._test_unbind_case(torch.tensor([1.0]), torch.tensor([]))

    @torch.inference_mode()
    def test_unbind_4(self):
        self._test_unbind_case(torch.tensor([]), torch.tensor([]))

    @torch.inference_mode()
    def test_unbind_dim(self):
        def _test_fn(unbind_fn):
            a = torch.rand(3, 2)
            b = torch.rand(2, 3)
            nt = torch.nested.nested_tensor([a, b])
            self.assertRaises(RuntimeError, lambda: unbind_fn(nt, 1))

        # Both of these tests are necessary, because we're using
        # torch_function.
        _test_fn(lambda x, dim: x.unbind(dim))
        # TODO: Re-enable this once using torch_dispatch
        # _test_fn(lambda x, dim: torch.unbind(x, dim))

    @torch.inference_mode()
    def test_nested_tensor(self):
        self.assertRaises(
            TypeError, lambda: torch.nested.nested_tensor(torch.tensor([3.0]))
        )
        self.assertRaises(TypeError, lambda: torch.nested.nested_tensor(4.0))

    @torch.inference_mode()
    def test_nested_tensor_matching_dim(self):
        self.assertRaisesRegex(
            RuntimeError,
            "Found dimension 1 for Tensor at index 1 and dimension 0 for Tensor at index 0.",
            lambda: torch.nested.nested_tensor([torch.tensor(1.0), torch.tensor([])]),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Found dimension 1 for Tensor at index 2 and dimension 0 for Tensor at index 1.",
            lambda: torch.nested.nested_tensor(
                [torch.tensor(1.0), torch.tensor(2.0), torch.tensor([])]
            ),
        )

    @torch.inference_mode()
    def test_default_nested_tensor(self):
        self.assertRaises(TypeError, lambda: torch.nested.nested_tensor())
        default_nested_tensor = torch.nested.nested_tensor([])
        default_tensor = torch.tensor([])
        # self.assertEqual(default_nested_tensor.nested_dim(), 1)
        # self.assertEqual(default_nested_tensor.nested_size(), ())
        self.assertEqual(default_nested_tensor.dim(), default_tensor.dim())
        self.assertEqual(default_nested_tensor.layout, default_tensor.layout)
        self.assertEqual(default_nested_tensor.device, default_tensor.device)
        self.assertEqual(default_nested_tensor.dtype, default_tensor.dtype)
        self.assertEqual(
            default_nested_tensor.requires_grad, default_tensor.requires_grad
        )
        self.assertIsNone(default_tensor.grad)
        # TODO: Re-enable once we have a performance driven
        # use case and implementation.
        # self.assertEqual(default_nested_tensor.is_pinned(),
        #                  default_tensor.is_pinned())

    @torch.inference_mode()
    def test_dim(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertEqual(a1.dim(), 1)
            a1 = constructor([torch.tensor(3.0)])
            self.assertEqual(a1.dim(), 1)
            a1 = constructor([torch.tensor([1, 2, 3, 4])])
            self.assertEqual(a1.dim(), 2)

    @unittest.skipIf(IS_FBCODE, "numel is not virtual in fbcode.")
    @torch.inference_mode()
    def test_numel(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertEqual(a1.numel(), 0)
            a1 = constructor([torch.tensor(3.0), torch.tensor(4.0)])
            self.assertEqual(a1.numel(), 2)
            a1 = constructor([torch.randn(2, 2, 2)])
            self.assertEqual(a1.numel(), 8)
            a1 = constructor([torch.randn([1, 2, 3]), torch.randn(3, 2, 1)])
            self.assertEqual(a1.numel(), 12)
            a1 = constructor([torch.randn([1, 1, 3]), torch.randn(3, 2, 4)])
            self.assertEqual(a1.numel(), 27)
            a1 = constructor([torch.randn([5, 5, 5]), torch.randn(6, 6, 6)])
            self.assertEqual(a1.numel(), 341)

            # Interesting edge case
            a1 = constructor([torch.randn([1, 2, 3]), torch.randn(1, 2, 0)])
            self.assertEqual(a1.numel(), 6)

    @torch.inference_mode()
    def test_size(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertRaisesRegex(
                RuntimeError,
                "NestedTensorImpl doesn't support sizes",
                lambda: a1.size(),
            )

    def test_size_dim(self):
        a = torch.nested.nested_tensor([])
        self.assertEqual(a.size(0), 0)

        a = torch.nested.nested_tensor([torch.tensor(1)])
        self.assertEqual(a.size(0), 1)

        a = torch.nested.nested_tensor([torch.tensor(1), torch.tensor(2)])
        self.assertEqual(a.size(0), 2)

        a = torch.nested.nested_tensor([torch.rand(1, 2), torch.rand(1, 8)])
        self.assertEqual(a.size(0), 2)
        self.assertEqual(a.size(1), 1)
        self.assertRaisesRegex(
            RuntimeError,
            "Given dimension 2 is irregular and does not have a size",
            lambda: a.size(2),
        )

        a = torch.nested.nested_tensor([torch.rand(3, 4), torch.rand(5, 4)])
        self.assertEqual(a.size(0), 2)
        self.assertRaisesRegex(
            RuntimeError,
            "Given dimension 1 is irregular and does not have a size",
            lambda: a.size(1),
        )
        self.assertEqual(a.size(2), 4)

    @unittest.skipIf(IS_FBCODE, "stride is not virtual in fbcode.")
    @torch.inference_mode()
    def test_stride(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertRaisesRegex(
                RuntimeError,
                "NestedTensorImpl doesn't support strides",
                lambda: a1.stride(),
            )

    @unittest.skipIf(IS_FBCODE, "is_contiguous is not virtual in fbcode.")
    @torch.inference_mode()
    def test_is_contiguous(self):
        # Test empty case
        nt_empty = torch.nested.nested_tensor([])
        assert nt_empty.is_contiguous()
        self.assertEqual(nt_empty, nt_empty.contiguous())

        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3, 6, 7))

        # Test contiguous case
        assert nt_contiguous.is_contiguous()
        self.assertEqual(nt_contiguous, nt_contiguous.contiguous())

        # Test non_contiguous case
        assert not nt_noncontiguous.is_contiguous()
        self.assertEqual(nt_contiguous, nt_noncontiguous.contiguous())

        # Test querying by memory_format
        self.assertTrue(
            nt_contiguous.is_contiguous(memory_format=torch.contiguous_format)
        )
        self.assertTrue(
            not nt_noncontiguous.is_contiguous(memory_format=torch.contiguous_format)
        )

    @torch.inference_mode()
    def test_repr_string(self):
        a = torch.nested.nested_tensor([])
        expected = "nested_tensor([\n\n])"
        self.assertEqual(str(a), expected)
        self.assertEqual(repr(a), expected)

        a = torch.nested.nested_tensor([torch.tensor(1.0)])
        expected = "nested_tensor([\n  tensor(1.)\n])"
        self.assertEqual(str(a), expected)
        self.assertEqual(repr(a), expected)

        a = torch.nested.nested_tensor([torch.tensor([[1, 2]]), torch.tensor([[4, 5]])])
        expected = "nested_tensor([\n  tensor([[1, 2]]),\n  tensor([[4, 5]])\n])"
        self.assertEqual(str(a), expected)
        self.assertEqual(repr(a), expected)

    def test_to_padded_tensor_on_empty_tensor(self):
        nt = torch.nested.nested_tensor([])
        empty = torch.nested.to_padded_tensor(nt, 4)
        self.assertEqual(empty, torch.tensor([]))

    def test_nested_namespace(self):
        nt = torch.nested.nested_tensor([torch.randn(2, 3), torch.randn(4, 5)])
        result = nt.to_padded_tensor(4)
        nested_namespace_result = torch.nested.to_padded_tensor(nt, 4)
        self.assertEqual(result, nested_namespace_result)

    def test_to(self):
        ntensors = 4
        nt = random_nt(torch.device("cpu"), torch.float32, ntensors, (4, 4))

        def test_copy_behavior(t, non_blocking=False):
            self.assertIs(t, t.to(t, non_blocking=non_blocking))
            self.assertIs(t, t.to(t.dtype, non_blocking=non_blocking))
            self.assertIs(t, t.to(torch.empty_like(t), non_blocking=non_blocking))
            self.assertIsNot(t, t.to(t, non_blocking=non_blocking, copy=True))
            self.assertIsNot(t, t.to(t.dtype, non_blocking=non_blocking, copy=True))
            self.assertIsNot(
                t, t.to(torch.empty_like(t), non_blocking=non_blocking, copy=True)
            )

            devices = [t.device]
            if t.device.type == "cuda":
                if t.device.index == -1:
                    devices.append(f"cuda:{torch.cuda.current_device()}")
                elif t.device.index == torch.cuda.current_device():
                    devices.append("cuda")
            for device in devices:
                self.assertIs(t, t.to(device, non_blocking=non_blocking))
                self.assertIs(t, t.to(device, t.dtype, non_blocking=non_blocking))
                self.assertIsNot(t, t.to(device, non_blocking=non_blocking, copy=True))
                self.assertIsNot(
                    t, t.to(device, t.dtype, non_blocking=non_blocking, copy=True)
                )

        test_copy_behavior(nt)
        self.assertEqual(nt.device, nt.to("cpu").device)
        self.assertEqual(nt.device, nt.to("cpu", dtype=torch.float32).device)
        self.assertIs(torch.float32, nt.to("cpu", dtype=torch.float32).dtype)
        self.assertEqual(nt.device, nt.to(torch.float32).device)
        self.assertIs(torch.float32, nt.to(dtype=torch.float32).dtype)

        def test_data_ptr(getter):
            self.assertEqual(getter(nt), getter(nt.to("cpu")))
            self.assertEqual(
                getter(nt), getter(nt.to(dtype=nt.dtype, device=nt.device, copy=False))
            )
            self.assertEqual(getter(nt), getter(nt.to("cpu", copy=False)))
            self.assertNotEqual(getter(nt), getter(nt.to("cpu", copy=True)))

        test_data_ptr(lambda nt: nt.data_ptr())

        if torch.cuda.is_available():
            for non_blocking in [True, False]:
                for cuda in [
                    "cuda",
                    "cuda:0" if torch.cuda.device_count() == 1 else "cuda:1",
                ]:
                    nt2 = random_nt(cuda, torch.float32, ntensors, (4, 4))
                    test_copy_behavior(nt2, non_blocking)
                    self.assertEqual(
                        nt2.device, nt2.to(cuda, non_blocking=non_blocking).device
                    )
                    self.assertEqual(
                        nt.device, nt2.to("cpu", non_blocking=non_blocking).device
                    )
                    self.assertEqual(
                        nt2.device, nt.to(cuda, non_blocking=non_blocking).device
                    )
                    self.assertIs(
                        torch.int32,
                        nt2.to(
                            "cpu", dtype=torch.int32, non_blocking=non_blocking
                        ).dtype,
                    )
                    self.assertEqual(
                        nt.device,
                        nt2.to(
                            "cpu", dtype=torch.int32, non_blocking=non_blocking
                        ).device,
                    )
                    self.assertIs(torch.int32, nt2.to(dtype=torch.int32).dtype)
                    self.assertEqual(nt2.device, nt2.to(dtype=torch.int32).device)

    def test_copy_(self):
        ntensors = 4
        nt = random_nt(torch.device("cpu"), torch.float32, ntensors, (4, 4))
        nt_copy = torch.empty_like(nt)
        nt_copy.copy_(nt)

        for nt_ub, nt_copy_ub in zip(nt.unbind(), nt_copy):
            self.assertEqual(nt_ub, nt_copy_ub)

        nt_error = torch.nested.nested_tensor([torch.tensor([0, 0])])
        self.assertRaisesRegex(
            RuntimeError,
            "copy_ only supports tensors that are the same size for Nested implementations",
            lambda: nt_error.copy_(nt),
        )

        if torch.cuda.is_available():
            nt = random_nt(torch.device("cuda"), torch.float32, ntensors, (4, 4))
            nt_copy = torch.empty_like(nt, device=torch.device("cpu"))
            nt_copy.copy_(nt, non_blocking=True)
            torch.cuda.current_stream(torch.cuda.current_device()).synchronize()
            for nt_ub, nt_copy_ub in zip(nt.unbind(), nt_copy):
                self.assertEqual(nt_ub, nt_copy_ub)

            nt_copy = torch.empty_like(nt, device=torch.device("cpu"))
            nt_copy.copy_(nt, non_blocking=False)
            for nt_ub, nt_copy_ub in zip(nt.unbind(), nt_copy):
                self.assertEqual(nt_ub, nt_copy_ub)

    def test_fill_(self):
        ntensors = 4
        nt = random_nt(torch.device("cpu"), torch.float32, ntensors, (4, 4))
        nt.fill_(10.0)
        for nt_ub in nt.unbind():
            t = torch.empty_like(nt_ub)
            t.fill_(10.0)
            self.assertEqual(nt_ub, t)

        fill_tensor = torch.tensor([11.0])
        self.assertRaisesRegex(
            RuntimeError,
            "fill_ only supports 0-dimension value tensor",
            lambda: nt.fill_(fill_tensor),
        )

        nt.fill_(fill_tensor[0])
        for nt_ub in nt.unbind():
            t = torch.empty_like(nt_ub)
            t.fill_(11.0)
            self.assertEqual(nt_ub, t)

    def test_zero_(self):
        ntensors = 4
        nt = random_nt(torch.device("cpu"), torch.float32, ntensors, (4, 4))
        nt.zero_()
        for nt_ub in nt.unbind():
            t = torch.empty_like(nt_ub)
            t.fill_(0.0)
            self.assertEqual(nt_ub, t)

    @parametrize(
        "func",
        [torch.ones_like, torch.zeros_like, torch.randn_like],
        name_fn=lambda f: f.__name__,
    )
    def test_like_functions(self, func):
        ntensors = 4
        nt = random_nt(torch.device("cpu"), torch.float32, ntensors, (4, 4))
        torch.manual_seed(1)
        nt_like = func(nt)

        torch.manual_seed(1)
        for nt_ub in nt_like.unbind():
            t_like = func(nt_ub)
            self.assertEqual(nt_ub, t_like)

    def test_cat(self):
        # dim=0 success case
        # No constraints on ragged structures matching.
        x = random_nt_from_dims([5, None, 10])
        y = random_nt_from_dims([3, 4, None])
        output = torch.cat([x, y], dim=0)
        for out_component, xy_component in zip(
            output.unbind(), itertools.chain(x.unbind(), y.unbind())
        ):
            self.assertEqual(out_component, xy_component)

        # dim=-1 success case
        # shape (B, *, D)
        x = random_nt_from_dims([5, None, 10])
        # shape (B, *, D'); same structure as x but dim=-1 differs
        y = random_nt_from_similar(x, dims=[-1, -1, 8])
        # should be shape (B, *, D + D') when supported
        output = torch.cat([x, y], dim=-1)
        for out_component, x_component, y_component in zip(
            output.unbind(), x.unbind(), y.unbind()
        ):
            self.assertEqual(
                out_component, torch.cat([x_component, y_component], dim=-1)
            )

        # dim between 0 and -1 success case
        x = random_nt_from_dims([5, None, 2, 3])
        # same structure as x but dim=2 differs
        y = random_nt_from_similar(x, dims=[-1, -1, 4, -1])
        output = torch.cat([x, y], dim=2)
        for out_component, x_component, y_component in zip(
            output.unbind(), x.unbind(), y.unbind()
        ):
            self.assertEqual(
                out_component, torch.cat([x_component, y_component], dim=1)
            )

        # error case: mixed NT / dense inputs
        x = random_nt_from_dims([5, None, 2])
        y = torch.randn(5, 3, 2)
        with self.assertRaisesRegex(
            RuntimeError, "expected each tensor in given list to be nested"
        ):
            torch.cat([x, y], dim=-1)

        # error case: NTs with different dims
        x = random_nt_from_dims([5, None, 2])
        y = random_nt_from_dims([5, None, 2, 3])
        with self.assertRaisesRegex(
            RuntimeError,
            "expected all nested tensors to have matching ragged structures outside of the concatenated dim",
        ):
            torch.cat([x, y], dim=-1)

        # error case: non-contiguous NT
        x, y = random_nt_noncontiguous_pair((2, 3, 4), dtype=torch.float32)
        # transpose to put ragged dim next to batch dim
        x, y = x.transpose(-2, -1), y.transpose(-2, -1)
        with self.assertRaisesRegex(
            RuntimeError, "only contiguous nested tensors are supported"
        ):
            torch.cat([x, y], dim=-1)

        # error case: multiple ragged dims in inputs
        x = random_nt_from_dims([5, None, None, 2])
        y = random_nt_from_similar(x)
        with self.assertRaisesRegex(
            RuntimeError,
            "only nested tensors with a single ragged dim next to the batch dim are supported",
        ):
            torch.cat([x, y], dim=-1)

        # error case: ragged dim not next to batch dim
        x = random_nt_from_dims([5, 2, None])
        y = random_nt_from_similar(x)
        with self.assertRaisesRegex(
            RuntimeError,
            "only nested tensors with a single ragged dim next to the batch dim are supported",
        ):
            torch.cat([x, y], dim=1)

        # error case: NTs with different batch sizes
        x = random_nt_from_dims([5, None, 2])
        y = random_nt_from_dims([3, None, 2])
        with self.assertRaisesRegex(
            RuntimeError,
            "expected all nested tensors to have matching ragged structures outside of the concatenated dim",
        ):
            torch.cat([x, y], dim=-1)

        # error case: NTs with different ragged structures
        x = torch.nested.nested_tensor(
            [
                torch.randn(2, 6),
                torch.randn(4, 6),
                torch.randn(5, 6),
            ]
        )
        y = torch.nested.nested_tensor(
            [
                torch.randn(5, 6),
                torch.randn(4, 6),
                torch.randn(2, 6),
            ]
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "expected all nested tensors to have matching ragged structures outside of the concatenated dim",
        ):
            torch.cat([x, y], dim=-1)

    # https://github.com/pytorch/pytorch/issues/161812
    def test_jagged_with_dim_error(self):
        x = torch.nested.nested_tensor(
            [torch.ones(3, 2, 3), torch.ones(4, 2, 3)], layout=torch.jagged
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "not supported for NestedTensor on dim=0",
        ):
            torch.cat([x, x])
        with self.assertRaisesRegex(
            RuntimeError,
            "not supported for NestedTensor on dim=0",
        ):
            torch.stack([x, x])

    def test_nested_view_from_buffer_overflow_errors(self):
        buffer = torch.tensor([1])
        sizes = torch.tensor([[2**63 - 1], [2**63 - 1], [3]], dtype=torch.int64)
        strides = torch.tensor(
            [[0x41414141], [0x41414141], [0x41414141]], dtype=torch.int64
        )
        offsets = torch.tensor(
            [[0x41414141], [0x41414141], [0x41414141]], dtype=torch.int64
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"Storage size calculation overflowed with sizes=\[9223372036854775807\] and strides=\[1094795585\]",
        ):
            nt = torch._nested_view_from_buffer(buffer, sizes, strides, offsets)


@markDynamoStrictTest
class TestNestedTensorDeviceType(NestedTensorTestCase):
    # Helper function to generate a pair of random nested tensors
    # the 2 nested tensors have same shapes
    def random_nt_pair(self, device, dtype, num_tensors, max_dims):
        ts1 = []
        ts2 = []
        for _ in range(num_tensors):
            tensor_dims = tuple(
                [
                    torch.randint(low=0, high=max_dim, size=(1,)).item()
                    for max_dim in max_dims
                ]
            )
            t1 = torch.randn(tensor_dims, device=device, dtype=dtype)
            t2 = torch.randn(tensor_dims, device=device, dtype=dtype)
            ts1.append(t1)
            ts2.append(t2)
        return (
            torch.nested.nested_tensor(ts1, device=device, dtype=dtype),
            torch.nested.nested_tensor(ts2, device=device, dtype=dtype),
        )

    @dtypes(*floating_types_and_half())
    def test_detach(self, device, dtype):
        a = torch.randn(2, 4, device=device, dtype=dtype, requires_grad=False)
        b = torch.randn(5, 4, device=device, dtype=dtype, requires_grad=False)
        x = torch.nested.nested_tensor([a, b], requires_grad=True)

        x_detach = x.detach()

        z = x_detach * 4
        self.assertFalse(x_detach.requires_grad)
        self.assertFalse(z.requires_grad)

        a = torch.randn(2, 4, device=device, dtype=dtype, requires_grad=True)
        b = torch.randn(5, 4, device=device, dtype=dtype, requires_grad=True)
        x = torch.nested.as_nested_tensor([a, b])

        y = x * 2
        y = y.detach()
        self.assertFalse(y.requires_grad)
        self.assertIsNone(y.grad_fn)

        z = x + y
        torch.nested.to_padded_tensor(z, 0).sum().backward()
        # This is an incorrect gradient, but we assume that's what the user
        # wanted. detach() is an advanced option.
        self.assertEqual(a.grad, torch.ones(2, 4, device=device, dtype=dtype))
        self.assertEqual(b.grad, torch.ones(5, 4, device=device, dtype=dtype))

    @dtypes(torch.float, torch.double, torch.half)
    @parametrize("requires_grad", [False, True])
    @parametrize("weights_only", [False, True])
    def test_serialization(self, device, dtype, requires_grad, weights_only):
        def compare_metadata(nt1, nt2):
            self.assertEqual(nt1._nested_tensor_size(), nt2._nested_tensor_size())
            self.assertEqual(nt1._nested_tensor_strides(), nt2._nested_tensor_strides())
            self.assertEqual(
                nt1._nested_tensor_storage_offsets(),
                nt2._nested_tensor_storage_offsets(),
            )

        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3, 6, 7))
        for a in [nt_contiguous, nt_noncontiguous]:
            buffer = io.BytesIO()
            serialized = torch.save(a, buffer)
            buffer.seek(0)
            b = torch.load(buffer, weights_only=weights_only)
            # should be both conceptually equal and metadata equivalent
            self.assertEqual(a, b)
            compare_metadata(a, b)
            # should be conceptually equal but not necessarily metadata equivalent
            self.assertEqual(b, nt_contiguous)
            self.assertEqual(b, nt_noncontiguous)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_unbind_noncontiguous(self, device, dtype):
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3, 6, 7), device, dtype
        )
        ub_contiguous = nt_contiguous.unbind()
        ub_noncontiguous = nt_noncontiguous.unbind()
        self.assertEqual(len(ub_contiguous), len(ub_noncontiguous))
        n = len(ub_contiguous)
        for i in range(n):
            self.assertEqual(ub_contiguous[i], ub_noncontiguous[i])

    @dtypes(torch.float)
    @skipMeta
    def test_to_then_from_padded_tensor_no_transform0213(self, device, dtype):
        t = torch.randn(4, 4, 4, device=device, dtype=dtype)
        ts = list(torch.unbind(t))
        ts[0] = ts[0][:-1]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        padded = torch.nested.to_padded_tensor(nt, 0)

        nt_to = torch._nested_from_padded_and_nested_example(padded, nt)

        for t1, t2 in zip(nt.unbind(), nt_to.unbind()):
            self.assertEqual(t1, t2)
        self.assertEqual(nt.device, nt_to.device)

    @dtypes(torch.float)
    @dtypesIfCUDA(torch.float, torch.half)
    @skipMeta
    @torch.inference_mode()
    def test_layer_norm(self, device, dtype):
        def _test(size):
            # Simple shapes test
            t0 = torch.randn(2, size, device=device, dtype=dtype, requires_grad=False)
            t1 = torch.randn(2, size, device=device, dtype=dtype, requires_grad=False)
            ts = [t0, t1, t0, t1]
            nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
            layer_norm = torch.nn.LayerNorm(size, device=device, dtype=dtype)
            nt_result = layer_norm(nt)
            for nt_subresult, t in zip(nt_result.unbind(), ts):
                t_result = layer_norm(t.reshape(1, -1, size).squeeze(0))
                self.assertEqual(nt_subresult, t_result)

            # More complex nt test with different lengths for each tensor
            t0 = torch.randn(4, size, device=device, dtype=dtype, requires_grad=False)
            t1 = torch.randn(10, size, device=device, dtype=dtype, requires_grad=False)
            t2 = torch.randn(7, size, device=device, dtype=dtype, requires_grad=False)
            ts = [t0, t1, t2, t0, t2]
            nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
            layer_norm = torch.nn.LayerNorm(size, device=device, dtype=dtype)
            nt_result = layer_norm(nt)
            for nt_subresult, t in zip(nt_result.unbind(), ts):
                t_result = layer_norm(t.reshape(1, -1, size).squeeze(0))
                self.assertEqual(nt_subresult, t_result)

            if size <= 128:
                # Test with multidimensional tensors after irregular dim
                # (run only with smaller dimensions to ensure fast execution)
                t0 = torch.randn(
                    4, size, size, 4, device=device, dtype=dtype, requires_grad=False
                )
                t1 = torch.randn(
                    10, size, size, 4, device=device, dtype=dtype, requires_grad=False
                )
                t2 = torch.randn(
                    7, size, size, 4, device=device, dtype=dtype, requires_grad=False
                )
                ts = [t0, t1, t2, t0, t2]
                nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
                layer_norm = torch.nn.LayerNorm(
                    (size, size, 4), device=device, dtype=dtype
                )
                nt_result = layer_norm(nt)
                for nt_subresult, t in zip(nt_result.unbind(), ts):
                    t_result = layer_norm(t.reshape(1, -1, size, size, 4).squeeze(0))
                    self.assertEqual(nt_subresult, t_result)

                # Test where the normalizing dimensions are not all
                layer_norm = torch.nn.LayerNorm((size, 4), device=device, dtype=dtype)
                nt_result = layer_norm(nt)
                for nt_subresult, t in zip(nt_result.unbind(), ts):
                    t_result = layer_norm(t.reshape(1, -1, size, size, 4).squeeze(0))
                    self.assertEqual(nt_subresult, t_result)

        for size in (1024, 1023, 513, 512, 256, 128, 2, 4, 32):
            _test(size)

    @dtypes(torch.float)
    @dtypesIfCUDA(torch.float, torch.half)
    @skipMeta
    @torch.inference_mode()
    def test_layer_norm_breaking(self, device, dtype):
        size = 128
        t0 = torch.randn(
            4, size, size, 4, device=device, dtype=dtype, requires_grad=False
        )
        t1 = torch.randn(
            10, size, size, 4, device=device, dtype=dtype, requires_grad=False
        )
        t2 = torch.randn(
            7, size, size, 4, device=device, dtype=dtype, requires_grad=False
        )
        ts = [t0, t1, t2, t0, t2]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        layer_norm = torch.nn.LayerNorm((4, size, size, 4), device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "normalized_shape extends into irregular dimensions for the nested tensor",
            lambda: layer_norm(nt),
        )
        layer_norm = torch.nn.LayerNorm((size + 1, size, 4), device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "The shape at dimension 0",
            lambda: layer_norm(nt),
        )

    @parametrize("layout", [torch.strided, torch.jagged], name_fn=layout_name)
    def test_embedding(self, device, layout):
        inputs = [
            torch.randint(100, (L,), device=device, dtype=torch.int64)
            for L in torch.randint(5, 50, (8,))
        ]
        x = torch.nested.nested_tensor(
            inputs, device=device, dtype=torch.int64, layout=layout
        )
        emb = torch.nn.Embedding(100, 8, device=device)
        y = emb(x)
        if layout == torch.jagged:
            y.backward(torch.randn_like(y))

        @torch._dynamo.disable
        def check(inputs, y):
            ys = y.unbind()
            for i, inp in enumerate(inputs):
                self.assertEqual(emb(inp), ys[i])

        check(inputs, y)

    @dtypes(
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.float,
        torch.float16,
        torch.bfloat16,
        torch.double,
    )
    def test_jagged_max_dtypes(self, device, dtype):
        x = torch.nested.nested_tensor(
            [torch.arange(0, n, dtype=dtype, device=device) for n in (10, 20, 30)],
            layout=torch.jagged,
        )

        result_max = x.max(dim=1)
        expected_max = torch.tensor([9, 19, 29], dtype=dtype, device=device)

        self.assertEqual(result_max.values, expected_max)

    @dtypes(
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.float,
        torch.float16,
        torch.bfloat16,
        torch.double,
    )
    def test_jagged_min_dtypes(self, device, dtype):
        x = torch.nested.nested_tensor(
            [torch.arange(0, n, dtype=dtype, device=device) for n in (10, 20, 30)],
            layout=torch.jagged,
        )

        result_min = x.min(dim=1)
        expected_min = torch.tensor([0, 0, 0], dtype=dtype, device=device)

        self.assertEqual(result_min.values, expected_min)

    @dtypes(
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.float,
        torch.float16,
        torch.bfloat16,
        torch.double,
    )
    def test_jagged_amax_dtypes(self, device, dtype):
        x = torch.nested.nested_tensor(
            [torch.arange(0, n, dtype=dtype, device=device) for n in (10, 20, 30)],
            layout=torch.jagged,
        )

        result_amax = x.amax(dim=1)
        expected_amax = torch.tensor([9, 19, 29], dtype=dtype, device=device)

        self.assertEqual(result_amax, expected_amax)

    @dtypes(
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.float,
        torch.float16,
        torch.bfloat16,
        torch.double,
    )
    def test_jagged_amin_dtypes(self, device, dtype):
        x = torch.nested.nested_tensor(
            [torch.arange(0, n, dtype=dtype, device=device) for n in (10, 20, 30)],
            layout=torch.jagged,
        )

        result_amin = x.amin(dim=1)
        expected_amin = torch.tensor([0, 0, 0], dtype=dtype, device=device)

        self.assertEqual(result_amin, expected_amin)

    @dtypes(
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.float,
        torch.float16,
        torch.bfloat16,
        torch.double,
    )
    def test_jagged_argmax_dtypes(self, device, dtype):
        x = torch.nested.nested_tensor(
            [torch.arange(0, n, dtype=dtype, device=device) for n in (10, 20, 30)],
            layout=torch.jagged,
        )

        result_argmax = x.argmax(dim=1)
        expected_argmax = torch.tensor([9, 19, 29], dtype=torch.long, device=device)

        self.assertEqual(result_argmax, expected_argmax)

    @dtypes(
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.float,
        torch.float16,
        torch.bfloat16,
        torch.double,
    )
    def test_jagged_argmin_dtypes(self, device, dtype):
        x = torch.nested.nested_tensor(
            [torch.arange(0, n, dtype=dtype, device=device) for n in (10, 20, 30)],
            layout=torch.jagged,
        )

        result_argmin = x.argmin(dim=1)
        expected_argmin = torch.tensor([0, 0, 0], dtype=torch.long, device=device)

        self.assertEqual(result_argmin, expected_argmin)

    @skipMeta
    @torch.inference_mode()
    @dtypes(*floating_types_and_half())
    def test_masked_fill(self, device, dtype):
        # nested tensor * nested tensor
        (nt, mask) = self.random_nt_pair(device, dtype, 4, (4, 4))
        mask = torch.nested.nested_tensor([m < 0 for m in mask.unbind()])
        ref = torch.nested.nested_tensor(
            [t.masked_fill(m, 0) for (t, m) in zip(nt.unbind(), mask.unbind())]
        )
        out = nt.masked_fill(mask, 0)
        self.assertEqual(ref, out)

    @dtypes(torch.float, torch.float16)
    def test_to_padded_tensor_simple(self, device, dtype):
        t = torch.randn(4, 4, 4, device=device, dtype=dtype)
        ts = list(torch.unbind(t))
        ts[0] = ts[0][:-1]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        for padding_value in (0, 1):
            padded = torch.nested.to_padded_tensor(nt, padding_value)

            correct_output = t.clone()
            if padding_value == 0:
                correct_output[0][-1] = torch.zeros_like(correct_output[0][-1])
            else:
                correct_output[0][-1] = torch.ones_like(correct_output[0][-1])

            self.assertEqual(padded, correct_output)
            self.assertEqual(padded.device, torch.device(device))
            self.assertEqual(padded.dtype, dtype)

    @dtypes(torch.float, torch.float16)
    def test_to_padded_tensor_output_size(self, device, dtype):
        t = torch.randn(4, 4, 4, device=device, dtype=dtype)
        output_size = (4, 6, 5)
        ts = list(torch.unbind(t))
        ts[0] = ts[0][:-1]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        for padding_value in (0, 1):
            padded = torch.nested.to_padded_tensor(
                nt, padding_value, output_size=output_size
            )
            correct_output = (
                torch.ones(output_size, device=device, dtype=dtype) * padding_value
            )
            correct_output[:4:, :4, :4] = t.clone()
            if padding_value == 0:
                correct_output[0][3] = torch.zeros_like(correct_output[0][3])
            else:
                correct_output[0][3] = torch.ones_like(correct_output[0][3])

            self.assertEqual(padded, correct_output)
            self.assertEqual(padded.device, torch.device(device))
            self.assertEqual(padded.dtype, dtype)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_to_padded_tensor_dim2(self, device, dtype):
        ts = [
            torch.randn(160, device=device, dtype=dtype),
            torch.randn(1240, device=device, dtype=dtype),
            torch.randn(2400, device=device, dtype=dtype),
        ]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        pad = 42
        correct_output = []
        for t in ts:
            next_output = torch.ones_like(ts[2]) * pad
            correct_output.append(next_output)
            next_output[: t.size(0)].copy_(t)
        correct_output = torch.stack(correct_output)
        padded = torch.nested.to_padded_tensor(nt, pad)
        self.assertEqual(padded, correct_output)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_to_padded_tensor_dim3(self, device, dtype):
        ts = [
            torch.randn(16, 21, device=device, dtype=dtype),
            torch.randn(24, 32, device=device, dtype=dtype),
            torch.randn(40, 53, device=device, dtype=dtype),
        ]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        pad = 42
        correct_output = []
        for t in ts:
            next_output = torch.ones_like(ts[2]) * pad
            correct_output.append(next_output)
            next_output[: t.size(0), : t.size(1)].copy_(t)
        correct_output = torch.stack(correct_output)
        padded = torch.nested.to_padded_tensor(nt, pad)
        self.assertEqual(padded, correct_output)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_to_padded_tensor_dim4(self, device, dtype):
        ts = [
            torch.randn(16, 21, 13, device=device, dtype=dtype),
            torch.randn(24, 32, 14, device=device, dtype=dtype),
            torch.ra
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data
- **Command Execution**: Executes system commands - validate inputs

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_nestedtensor.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_tensorexpr.py_docs.md_docs.md`](./test_tensorexpr.py_docs.md_docs.md)
- [`pytest_shard_custom.py_docs.md_docs.md`](./pytest_shard_custom.py_docs.md_docs.md)
- [`test_weak.py_kw.md_docs.md`](./test_weak.py_kw.md_docs.md)
- [`test_view_ops.py_kw.md_docs.md`](./test_view_ops.py_kw.md_docs.md)
- [`test_varlen_attention.py_kw.md_docs.md`](./test_varlen_attention.py_kw.md_docs.md)
- [`test_namedtensor.py_docs.md_docs.md`](./test_namedtensor.py_docs.md_docs.md)
- [`test_binary_ufuncs.py_docs.md_docs.md`](./test_binary_ufuncs.py_docs.md_docs.md)
- [`test_ops_gradients.py_kw.md_docs.md`](./test_ops_gradients.py_kw.md_docs.md)
- [`test_torchfuzz_repros.py_docs.md_docs.md`](./test_torchfuzz_repros.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_nestedtensor.py_docs.md_docs.md`
- **Keyword Index**: `test_nestedtensor.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
