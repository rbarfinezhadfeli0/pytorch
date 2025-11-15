# Documentation: test_nestedtensor.py

## File Metadata
- **Path**: `test/test_nestedtensor.py`
- **Size**: 359210 bytes
- **Lines**: 9132
- **Extension**: .py
- **Type**: Regular file

## Original Source

```py
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
            torch.randn(40, 53, 16, device=device, dtype=dtype),
        ]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        pad = 42
        correct_output = []
        for t in ts:
            next_output = torch.ones_like(ts[2]) * pad
            correct_output.append(next_output)
            next_output[: t.size(0), : t.size(1), : t.size(2)].copy_(t)
        correct_output = torch.stack(correct_output)
        padded = torch.nested.to_padded_tensor(nt, pad)
        self.assertEqual(padded, correct_output)

    # TODO: test noncontiguous to_padded_tensor
    # For now this tests the functionality of noncontiguous_to_padded_tensor
    # and the error message of to_padded_tensor
    # since to_padded_tensor does not support noncontiguous buffer yet
    @dtypes(torch.float, torch.float16, torch.double)
    @torch.inference_mode()
    def test_to_padded_tensor_noncontiguous(self, device, dtype):
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3, 6, 7), device, dtype
        )
        # test noncontiguous_to_padded_tensor functionality
        self.assertEqual(
            torch.nested.to_padded_tensor(nt_contiguous, 0.0),
            noncontiguous_to_padded_tensor(nt_noncontiguous),
        )
        # test to_padded_tensor error message
        self.assertRaisesRegex(
            RuntimeError,
            r"for now to_padded_tensor only supports contiguous nested tensor",
            lambda: torch.nested.to_padded_tensor(nt_noncontiguous, 0.0),
        )

    @skipMeta
    def test_device_checks(self, device):
        nt = torch.nested.nested_tensor([], device=device)
        is_cuda = "cuda" in str(device)
        self.assertEqual(nt.is_cuda, is_cuda)

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_share_memory(self, device):
        a = torch.randn(3, 4, device=device)
        b = torch.randn(5, 4, device=device)
        nt = torch.nested.nested_tensor([a, b], layout=torch.jagged)

        # Guard CUDA tensors
        if "cuda" in device:
            result = nt.share_memory_()
            self.assertIs(result, nt)
            return

        result = nt.share_memory_()
        self.assertIs(result, nt)

        # Verify in shared memory
        self.assertTrue(nt.is_shared())

    @dtypes(torch.float, torch.float16, torch.double)
    def test_nested_tensor_indexing(self, device, dtype):
        # edge case: empty nested tensor
        nt0 = torch.nested.nested_tensor([])
        self.assertRaises(IndexError, lambda: nt0[0])
        # normal case
        x0 = torch.randn((2, 5), device=device, dtype=dtype)
        x1 = torch.randn((3, 4), device=device, dtype=dtype)
        nt = torch.nested.nested_tensor([x0, x1])
        # single index: only support integer in the batch dimension
        self.assertEqual(nt[0], x0)
        self.assertEqual(nt[-1], x1)
        self.assertRaises(IndexError, lambda: nt[2])
        self.assertRaises(IndexError, lambda: nt[-3])
        self.assertRaises(NotImplementedError, lambda: nt[:])
        self.assertEqual(nt[...], nt)
        # tuple of indices: only support integer in the batch dimension
        #                 + all possible indexing in the original tensor dimensions
        self.assertEqual(nt[0, 0, 0], x0[0, 0])
        self.assertEqual(nt[0, 1, :], x0[1, :])
        self.assertEqual(nt[1, ...], x1)
        self.assertRaises(IndexError, lambda: nt[1, 4, 2])
        self.assertRaises(NotImplementedError, lambda: nt[:, 1, 1])
        # test select on non-batch dimensions
        self.assertEqual(nt.select(1, 0)[0], x0.select(0, 0))
        self.assertEqual(nt.select(1, 0)[1], x1.select(0, 0))
        self.assertRaises(IndexError, lambda: nt.select(1, 3))
        self.assertEqual(nt.select(2, 0)[0], x0.select(1, 0))
        self.assertEqual(nt.select(2, 0)[1], x1.select(1, 0))
        self.assertRaises(IndexError, lambda: nt.select(2, 5))
        # make sure indexing returns a view
        nt[0].fill_(100.0)
        answer = torch.tensor(100.0, device=device, dtype=dtype).expand((2, 5))
        self.assertEqual(nt[0], answer)
        nt[1, 1, :].fill_(200.0)
        answer = torch.tensor(200.0, device=device, dtype=dtype).expand(4)
        self.assertEqual(nt[1, 1, :], answer)

        # Test that indexing works when requires_grad_(True)
        # previously this was failing because the backward kernel for select.int uses .sizes()
        nt = torch.nested.nested_tensor([x0, x1]).requires_grad_(True)
        self.assertEqual(nt[0], x0)
        self.assertEqual(nt[-1], x1)
        grad_x0 = torch.randn((2, 5), device=device, dtype=dtype)
        nt[0].backward(grad_x0)
        expected_grad = torch.nested.nested_tensor(
            [grad_x0, torch.zeros((3, 4), device=device, dtype=dtype)]
        )
        self.assertEqual(nt.grad, expected_grad)

    @parametrize(
        "func",
        [
            subtest(torch.nn.functional.relu, name="relu"),
            subtest(torch.nn.functional.relu_, name="relu_"),
            subtest(torch.nn.functional.gelu, name="gelu"),
            subtest(torch._C._nn.gelu_, name="gelu_"),
            subtest(torch.tanh, name="tanh"),
            subtest(torch.tanh_, name="tanh_"),
            subtest(torch.neg, name="neg"),
            subtest(torch.nn.functional.silu, name="silu"),
            subtest(partial(torch.nn.functional.silu, inplace=True), name="silu_"),
            subtest(torch.abs, name="abs"),
            subtest(torch.abs_, name="abs_"),
            subtest(torch.sgn, name="sgn"),
            subtest(torch.logical_not, name="logical_not"),
            subtest(torch.sin, name="sin"),
            subtest(torch.cos, name="cos"),
            subtest(torch.isinf, name="isinf"),
            subtest(torch.isposinf, name="isposinf"),
            subtest(torch.isneginf, name="isneginf"),
            subtest(torch.isnan, name="isnan"),
            subtest(torch.sqrt, name="sqrt"),
        ],
    )
    def test_unary_funcs(self, device, func):
        nt, nt_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3, 6, 7), device=device, dtype=torch.float32
        )
        nested_result = func(nt)
        self.assertTrue(nested_result.is_nested)
        for t, t_res in zip(nt.unbind(), nested_result.unbind()):
            self.assertEqual(func(t), t_res)
        self.assertRaisesRegex(
            RuntimeError,
            "NestedTensor must be contiguous to get buffer.",
            lambda: func(nt_noncontiguous),
        )

    def test_is_any_true_jagged(self, device):
        B, Fin = 2, 6
        start = torch.zeros(B, dtype=torch.int64, device=device)
        lengths = torch.tensor([3, 2], dtype=torch.int64, device=device)

        # NestedTensor reduction should operate on same data as .values().
        with self.subTest("dispatch_matches_values_buffer"):
            cond = torch.tensor(
                [
                    [True, False, False, True, True, False],
                    [False, False, True, False, False, False],
                ],
                dtype=torch.bool,
                device=device,
            )
            nt = torch.nested.narrow(
                cond, dim=1, start=start, length=lengths, layout=torch.jagged
            )
            out_nt = torch.ops.aten._is_any_true.default(nt).item()
            out_vals = torch.ops.aten._is_any_true.default(nt.values()).item()
            self.assertEqual(out_nt, out_vals)

        # Verify jagged boolean behavior.
        with self.subTest("all_false_returns_false"):
            cond_false = torch.zeros(B, Fin, dtype=torch.bool, device=device)
            nt_false = torch.nested.narrow(
                cond_false, dim=1, start=start, length=lengths, layout=torch.jagged
            )
            self.assertFalse(torch.ops.aten._is_any_true.default(nt_false).item())

        with self.subTest("one_true_returns_true"):
            cond_mixed = torch.zeros(B, Fin, dtype=torch.bool, device=device)
            cond_mixed[0, 0] = True
            nt_mixed = torch.nested.narrow(
                cond_mixed, dim=1, start=start, length=lengths, layout=torch.jagged
            )
            self.assertTrue(torch.ops.aten._is_any_true.default(nt_mixed).item())

    def test_is_all_true_jagged(self, device):
        B, Fin = 2, 6
        start = torch.zeros(B, dtype=torch.int64, device=device)
        lengths = torch.tensor([3, 2], dtype=torch.int64, device=device)

        # NestedTensor reduction should operate on same data as .values().
        with self.subTest("dispatch_matches_values_buffer"):
            cond = torch.tensor(
                [
                    [True, True, True, False, False, False],
                    [True, True, False, False, False, False],
                ],
                dtype=torch.bool,
                device=device,
            )
            nt = torch.nested.narrow(
                cond, dim=1, start=start, length=lengths, layout=torch.jagged
            )
            out_nt = torch.ops.aten._is_all_true.default(nt).item()
            out_vals = torch.ops.aten._is_all_true.default(nt.values()).item()
            self.assertEqual(out_nt, out_vals)

        # Verify jagged boolean behavior.
        with self.subTest("all_true_returns_true"):
            cond_true = torch.ones(B, Fin, dtype=torch.bool, device=device)
            nt_true = torch.nested.narrow(
                cond_true, dim=1, start=start, length=lengths, layout=torch.jagged
            )
            self.assertTrue(torch.ops.aten._is_all_true.default(nt_true).item())

        with self.subTest("any_false_returns_false"):
            cond_mixed = torch.ones(B, Fin, dtype=torch.bool, device=device)
            cond_mixed[0, 1] = False
            nt_mixed = torch.nested.narrow(
                cond_mixed, dim=1, start=start, length=lengths, layout=torch.jagged
            )
            self.assertFalse(torch.ops.aten._is_all_true.default(nt_mixed).item())

    @parametrize("func", [subtest(torch.ge, name="ge"), subtest(torch.eq, name="eq")])
    def test_binary_ops_with_scalar(self, device, func):
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3, 6, 7), device=device, dtype=torch.float32
        )
        scalar = 0.0

        # should work regardless of contiguity
        for nt in (nt_contiguous, nt_noncontiguous):
            nested_result = func(nt, scalar)
            self.assertTrue(nested_result.is_nested)
            for t, t_res in zip(nt.unbind(), nested_result.unbind()):
                self.assertEqual(func(t, scalar), t_res)

    @dtypes(*floating_types_and_half())
    def test_nested_tensor_chunk(self, device, dtype):
        # Transformer use case
        a = torch.randn(3, 3 * 4, device=device, dtype=dtype)
        b = torch.randn(2, 3 * 4, device=device, dtype=dtype)
        c = torch.randn(1, 3 * 4, device=device, dtype=dtype)
        a_chunks = a.chunk(3, dim=-1)
        b_chunks = b.chunk(3, dim=-1)
        c_chunks = c.chunk(3, dim=-1)

        a_nt = [a_chunks[0], b_chunks[0], c_chunks[0]]
        b_nt = [a_chunks[1], b_chunks[1], c_chunks[1]]
        c_nt = [a_chunks[2], b_chunks[2], c_chunks[2]]

        nt = torch.nested.nested_tensor([a, b, c])
        chunked = nt.chunk(3, dim=-1)

        self.assertEqual(chunked[0], torch.nested.nested_tensor(a_nt))
        self.assertEqual(chunked[1], torch.nested.nested_tensor(b_nt))
        self.assertEqual(chunked[2], torch.nested.nested_tensor(c_nt))

        for chunk in chunked:
            self.assertFalse(chunk.is_contiguous())

        # Failure chunking on ragged dimensions
        self.assertRaisesRegex(
            RuntimeError,
            "Chunk for nested tensors is currently only supported for the last dimension.",
            lambda: torch.chunk(nt, 5, dim=1),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Chunk for nested tensors is currently only supported for the last dimension.",
            lambda: torch.chunk(nt, 5, dim=0),
        )

        # Failure on non-contiguous nt
        _, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3), device, dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "chunk expects `self` to be contiguous.",
            lambda: torch.chunk(nt_noncontiguous, 5, dim=-1),
        )

        # Failure when calling non divisible n_chunks
        self.assertRaisesRegex(
            RuntimeError,
            "Chunk for nested tensors is only supported for "
            "nested tensors with trailing dimension divisible by chunks.",
            lambda: torch.chunk(nt, 5, dim=-1),
        )

        # Failure when calling backward on a chunk
        a = torch.randn(3, 3 * 4, device=device, dtype=dtype, requires_grad=True)
        b = torch.randn(2, 3 * 4, device=device, dtype=dtype, requires_grad=True)
        nt_grad = torch.nested.as_nested_tensor([a, b])
        chunked = torch.chunk(nt_grad, 2, dim=-1)
        self.assertRaisesRegex(
            RuntimeError,
            "Nested Strided Tensor doesn't support chunk backward.",
            lambda: chunked[0].backward(chunked[0].clone()),
        )

    @dtypes(*floating_types_and_half())
    def test_nested_tensor_split_with_sizes(self, device, dtype):
        a = torch.randn(3, 20, device=device, dtype=dtype)
        b = torch.randn(2, 20, device=device, dtype=dtype)
        c = torch.randn(1, 20, device=device, dtype=dtype)

        split_sizes = [4, 6, 10]
        a_splits = a.split_with_sizes(split_sizes, dim=-1)
        b_splits = b.split_with_sizes(split_sizes, dim=-1)
        c_splits = c.split_with_sizes(split_sizes, dim=-1)

        nt = torch.nested.nested_tensor([a, b, c])
        nt_splits = nt.split_with_sizes(split_sizes, dim=-1)

        for i, nt_split in enumerate(nt_splits):
            self.assertEqual(
                nt_split,
                torch.nested.nested_tensor([a_splits[i], b_splits[i], c_splits[i]]),
            )
            dense_strides = torch.stack(
                [
                    torch.tensor(a_splits[i].stride()),
                    torch.tensor(b_splits[i].stride()),
                    torch.tensor(c_splits[i].stride()),
                ]
            )
            self.assertEqual(nt_split._nested_tensor_strides(), dense_strides)
            self.assertFalse(nt_split.is_contiguous())

        # Failure calling on ragged dimensions
        self.assertRaisesRegex(
            RuntimeError,
            "split_with_sizes for nested tensors is currently only supported for the last dimension.",
            lambda: torch.split_with_sizes(nt, split_sizes, dim=1),
        )

        # Failure calling on non-last dimension
        self.assertRaisesRegex(
            RuntimeError,
            "split_with_sizes for nested tensors is currently only supported for the last dimension.",
            lambda: torch.split_with_sizes(nt, split_sizes, dim=0),
        )

        # Failure on non-contiguous nt
        _, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3), device, dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "split_with_sizes expects `self` to be contiguous.",
            lambda: torch.split_with_sizes(nt_noncontiguous, split_sizes, dim=-1),
        )

        # Failure when calling with split_sizes that don't cover the full dim size
        bad_split_sizes = [4, 6, 9]  # don't add up to 20
        self.assertRaisesRegex(
            RuntimeError,
            "split_with_sizes expects split_sizes to sum exactly to 20",
            lambda: torch.split_with_sizes(nt, bad_split_sizes, dim=-1),
        )

    @dtypes(torch.float, torch.float16, torch.double)
    @torch.inference_mode()
    def test_nested_tensor_indexing_noncontiguous(self, device, dtype):
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3, 6, 7), device, dtype
        )
        self.assertEqual(nt_contiguous.size(0), nt_noncontiguous.size(0))
        n = nt_contiguous.size(0)
        for i in range(n):
            self.assertEqual(nt_contiguous[i], nt_noncontiguous[i])

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    @parametrize("transpose", [True, False])
    def test_nested_tensor_add(self, device, dtype, transpose):
        if transpose:
            a = torch.randn(2, 2, 2, device=device, dtype=dtype)
            b = torch.rand(2, 2, 2, device=device, dtype=dtype)
            c = a.transpose(-1, -2).contiguous()
            d = b.transpose(-1, -2).contiguous()
            nt1 = torch.nested.nested_tensor([a, b, a, b])
            nt2 = torch.nested.nested_tensor([c, d, c, d]).transpose(-1, -2)
        else:
            (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        ref = torch.nested.nested_tensor(
            [t1 + t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())]
        )
        out = nt1 + nt2
        self.assertEqual(ref, out)

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    @parametrize("transpose", [True, False])
    def test_nested_tensor_sub(self, device, dtype, transpose):
        if transpose:
            a = torch.randn(2, 2, 2, device=device, dtype=dtype)
            b = torch.rand(2, 2, 2, device=device, dtype=dtype)
            c = a.transpose(-1, -2).contiguous()
            d = b.transpose(-1, -2).contiguous()
            nt1 = torch.nested.nested_tensor([a, b, a, b])
            nt2 = torch.nested.nested_tensor([c, d, c, d]).transpose(-1, -2)
        else:
            (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        ref = torch.nested.nested_tensor(
            [t1 - t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())]
        )
        out = nt1 - nt2
        self.assertEqual(ref, out)

    @onlyCUDA
    @dtypes(torch.float, torch.float16)
    @torch.inference_mode()
    @parametrize("embedding_dim", [8, 128, 256, 384])
    def test_nested_tensor_dense_elementwise(self, device, dtype, embedding_dim):
        def _test_add_mul(nt, t):
            ref_add = torch.nested.nested_tensor(
                [t1 + t2 for (t1, t2) in zip(nt.unbind(), t.unbind())]
            )
            ref_mul = torch.nested.nested_tensor(
                [t1 * t2 for (t1, t2) in zip(nt.unbind(), t.unbind())]
            )
            self.assertEqual(nt.add(t), ref_add)
            self.assertEqual(nt.mul(t), ref_mul)

        batch_size = 32
        seq_lens = torch.randint(low=0, high=10, size=(batch_size,))

        # [B, *, D], [B, 1, D] case
        ts = [torch.randn((seq_len, embedding_dim)) for seq_len in seq_lens]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        t = torch.randn((batch_size, 1, embedding_dim), device=device, dtype=dtype)
        _test_add_mul(nt, t)

        # [B, *], [B, 1] case
        ts = [torch.randn(seq_len) for seq_len in seq_lens]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        t = torch.randn((batch_size, 1), device=device, dtype=dtype)
        _test_add_mul(nt, t)

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    def test_nested_tensor_mul(self, device, dtype):
        # nested tensor * nested tensor
        (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        ref = torch.nested.nested_tensor(
            [t1 * t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())]
        )
        out = nt1 * nt2
        self.assertEqual(ref, out)
        # nested tensor * scalar
        number = 10.0
        scalar = torch.tensor(number).to(dtype).to(device)
        ref = torch.nested.nested_tensor([t * number for t in nt1.unbind()])
        out_number0 = nt1 * number
        out_number1 = number * nt1
        out_scalar0 = nt1 * scalar
        out_scalar1 = scalar * nt1
        self.assertEqual(out_number0, ref)
        self.assertEqual(out_number1, ref)
        self.assertEqual(out_scalar0, ref)
        self.assertEqual(out_scalar1, ref)
        # error case: numel == 1 but dim > 0
        vector = torch.tensor([number]).to(dtype).to(device)
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both self and other to be nested, but got a nested self and non-nested other",
            lambda: nt1.mul(vector),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both self and other to be nested, but got a non-nested self and nested other",
            lambda: vector.mul(nt1),
        )

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    def test_nested_tensor_div(self, device, dtype):
        nt, nt2 = self.random_nt_pair(device, dtype, 4, (4, 4))
        scale = 4.0
        ref = torch.nested.nested_tensor([t / scale for t in nt.unbind()])
        out = nt / 4.0
        self.assertEqual(ref, out)
        ref_transposed = ref.transpose(1, 2)
        out = nt.transpose(1, 2) / 4.0
        self.assertEqual(ref_transposed, out)

        ref = torch.nested.nested_tensor(
            [t / t2 for (t, t2) in zip(nt.unbind(), nt2.unbind())]
        )
        out = nt / nt2
        self.assertEqual(ref, out)

        out = nt.transpose(1, 2) / nt2.transpose(1, 2)
        self.assertEqual(ref.transpose(1, 2), out)

        nt_transpose_copy = torch.nested.nested_tensor(
            [t.transpose(0, 1) for t in nt.unbind()]
        )

        self.assertRaisesRegex(
            RuntimeError,
            "div requires strides to match when given NestedTensors",
            lambda: nt_transpose_copy.transpose(1, 2) / nt2,
        )

        nt = torch.nested.nested_tensor(
            [torch.randn(i, 4) for i in [3, 4, 5]], device=device, dtype=dtype
        )
        nt_chunks = nt.chunk(2, -1)
        self.assertRaisesRegex(
            RuntimeError,
            "div requires offsets to match when given NestedTensors",
            lambda: nt_chunks[0] / nt_chunks[1],
        )

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    def test_nested_tensor_add_in_place(self, device, dtype):
        (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        ref = torch.nested.nested_tensor(
            [t1 + t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())]
        )
        nt1 += nt2
        self.assertEqual(ref, nt1)

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    def test_nested_tensor_mul_in_place(self, device, dtype):
        # nested tensor * nested tensor
        (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        ref = torch.nested.nested_tensor(
            [t1 * t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())]
        )
        nt1 *= nt2
        self.assertEqual(ref, nt1)
        # nested tensor * scalar
        number = 10.0
        scalar = torch.tensor(number).to(dtype).to(device)
        ref = torch.nested.nested_tensor([t * number for t in nt1.unbind()])
        out_number = nt1.clone()
        out_number *= number
        out_scalar = nt1.clone()
        out_scalar *= scalar
        self.assertEqual(out_number, ref)
        self.assertEqual(out_scalar, ref)
        self.assertRaisesRegex(
            RuntimeError,
            r"output with shape \[.*\] doesn't match the broadcast shape \[.*\]",
            lambda: scalar.mul_(nt1),
        )
        # error case: numel == 1 but dim > 0
        vector = torch.tensor([number]).to(dtype).to(device)
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both self and other to be nested, but got a nested self and non-nested other",
            lambda: nt1.mul_(vector),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both self and other to be nested, but got a non-nested self and nested other",
            lambda: vector.mul_(nt1),
        )

    @onlyCPU
    @skipMeta
    @dtypes(torch.float)
    def test_nested_tensor_sum_dim(self, device, dtype):
        params = ((2, (1, 1)), ((4), (4, 4)), (10, (3, 5, 7)))

        def test_sum(device, dtype, ntensors, max_sizes, dim, keepdim=True):
            nt = random_nt(device, dtype, ntensors, max_sizes, require_non_empty=False)
            nt2 = nt.clone()
            ub2 = nt2.unbind()
            nt.requires_grad_(True)
            [t.requires_grad_(True) for t in ub2]
            nt_sum = nt.sum(dim=dim, keepdim=keepdim)
            ub2_sum = [t.sum(-1, keepdim=keepdim) for t in ub2]
            self.assertEqual(nt_sum, torch.nested.nested_tensor(ub2_sum))

            # test backward
            # generate gradient tensor that has the same size as the output
            size = nt_sum._nested_tensor_size()
            gt2 = []
            for i in range(ntensors):
                gt2.append(torch.randn(size[i].tolist(), device=device, dtype=dtype))
            gt = torch.nested.nested_tensor(gt2).clone()
            nt_sum.backward(gt)
            for t2, g2 in zip(ub2_sum, gt2):
                t2.backward(g2)
            self.assertEqual(nt.grad, torch.nested.nested_tensor([t.grad for t in ub2]))
            return

        for ntensors, max_sizes in params:
            test_sum(device, dtype, ntensors, max_sizes, len(max_sizes))

        # Test error inputs
        with self.assertRaisesRegex(
            RuntimeError, "NestedTensor can only be reduced across the last"
        ):
            torch.nested.nested_tensor(
                [torch.tensor([3, 4, 5]), torch.tensor([1, 2])]
            ).sum(0, keepdim=True)

        with self.assertRaisesRegex(
            RuntimeError, "NestedTensor only allows reduction of a single"
        ):
            torch.nested.nested_tensor(
                [torch.tensor([[3, 4, 5]]), torch.tensor([[1, 2]])]
            ).sum([0, 1], keepdim=True)

        with self.assertRaisesRegex(
            RuntimeError, "NestedTensor always requires keepdim=True for now."
        ):
            torch.nested.nested_tensor(
                [torch.tensor([3, 4, 5]), torch.tensor([1, 2])]
            ).sum(-1)

    @dtypes(torch.float, torch.float16)
    def test_contiguous(self, device, dtype):
        # Since we don't have access to the buffer in python this is harder to show what
        # we are testing for. When we call chunk on a consistent dim of a NT
        # for chunk_size > 1 the resulting tensors are views of the original NT
        # whose numels is now less than the size of the buffer. Clone was
        # previously creating a new NT with a buffer that was the same size as the
        # original.
        nt_contiguous = torch.nested.nested_tensor(
            [
                torch.randn(2, 20, device=device, dtype=dtype),
                torch.randn(4, 20, device=device, dtype=dtype),
            ]
        )
        # Split up the last dimension which has a consistent size of 20 into 5 chunks
        chunks = nt_contiguous.chunk(5, dim=-1)

        # # Check chunks are contiguous after calling contiguous
        for chunk in chunks:
            self.assertFalse(chunk.is_contiguous())
            self.assertTrue(chunk.contiguous().is_contiguous())

    @dtypes(torch.float, torch.float16)
    @skipMeta
    def test_clone(self, device, dtype):
        nt1 = random_nt(device, dtype, 4, (4, 4), (1, 1))
        nt2 = nt1.clone()
        # Verify the values match
        self.assertEqual(nt1, nt2)
        # Verify modifying nt2 doesn't affect nt1
        nt2.mul_(nt1)
        ub1 = nt1.unbind()
        ub2 = nt2.unbind()
        for i in range(len(ub1)):
            self.assertNotEqual(ub1[i], ub2[i])

        nt1.clone(memory_format=torch.preserve_format)
        msg = "Nested tensor clone supports Preserve and Contiguous memory formats, called clone with memory format: ChannelsLast"
        with self.assertRaisesRegex(RuntimeError, msg):
            nt1.clone(memory_format=torch.channels_last)

    # cannot test torch.float16 because: RuntimeError: "bernoulli_scalar_cpu_" not implemented for 'Half'
    @decorateIf(xfailIfTorchDynamo, lambda params: params["layout"] == torch.jagged)
    @dtypes(torch.float, torch.double)
    @parametrize("layout", [torch.strided, torch.jagged], name_fn=layout_name)
    def test_dropout(self, device, dtype, layout):
        # edge case: empty nested tensor
        # TODO: support empty NT in jagged layout
        if layout == torch.strided:
            nt0 = torch.nested.nested_tensor([], layout=layout)
            y = torch.nn.functional.dropout(nt0, 0.5)
            self.assertEqual(nt0, y)
        # normal nested tensor
        ntensors = 4
        if layout == torch.jagged:
            nt = random_nt(device, dtype, ntensors, (4, 4), (0, 3), layout=layout)
        else:
            nt = random_nt(device, dtype, ntensors, (4, 4), layout=layout)
        # edge case: invalid dropout
        self.assertRaises(ValueError, lambda: torch.nn.Dropout(-0.1))
        self.assertRaises(ValueError, lambda: torch.nn.Dropout(1.1))
        self.assertRaises(ValueError, lambda: torch.nn.functional.dropout(nt, -0.1))
        self.assertRaises(ValueError, lambda: torch.nn.functional.dropout(nt, 1.1))
        # edge case: no dropout
        dropouter = torch.nn.Dropout(0.0)
        y0 = dropouter(nt)
        y1 = torch.nn.functional.dropout(nt, 0.0)
        self.assertEqual(nt, y0)
        self.assertEqual(nt, y1)
        # edge case: all dropout
        dropouter = torch.nn.Dropout(1.0)
        y0 = dropouter(nt)
        y1 = torch.nn.functional.dropout(nt, 1.0)
        nt0 = torch.zeros_like(nt)
        self.assertEqual(nt0, y0)
        self.assertEqual(nt0, y1)
        # normal case: normal dropout
        p = 0.2
        y = torch.nn.functional.dropout(nt, p)
        expect = nt.clone()
        if layout == torch.jagged:
            expect = torch.where(y == 0.0, y, nt)
            expect /= 1.0 - p
            self.assertEqual(y, expect)
        else:
            expect = nt.clone()
            for i in range(ntensors):
                actual_tensor = y[i].view(-1)
                expect_tensor = expect[i].view(-1)
                for j in range(actual_tensor.shape[0]):
                    if actual_tensor[j].item() == 0.0:
                        expect_tensor[j] = 0.0
                    else:
                        expect_tensor[j] /= 1.0 - p
            self.assertEqual(y, expect)
        with freeze_rng_state():
            dropouter = torch.nn.Dropout(p)
            y0 = dropouter(nt)
        with freeze_rng_state():
            y1 = torch.nn.functional.dropout(nt, p)
        self.assertEqual(y0, y1)

    @dtypes(torch.float, torch.double)
    def test_dropout_noncontiguous(self, device, dtype):
        ntensors = 4
        nt0 = random_nt(device, dtype, ntensors, (4, 4))
        nt1 = nt0.transpose(-1, -2)
        p = 0.3
        with freeze_rng_state():
            dropouter = torch.nn.Dropout(p)
            y0 = dropouter(nt0)
        with freeze_rng_state():
            y1 = torch.nn.functional.dropout(nt1, p).transpose(-1, -2)
        self.assertEqual(y0, y1)

    # cannot test torch.float16 because: RuntimeError: "softmax_kernel_impl" not implemented for 'Half'
    @dtypes(torch.float, torch.double)
    def test_softmax(self, device, dtype):
        # normal nested tensor
        ntensors = 4
        nt = random_nt(device, dtype, ntensors, (4, 4))
        # error case: softmax across nested dimension
        self.assertRaisesRegex(
            RuntimeError,
            "Cannot apply softmax across nested dimension 0",
            lambda: torch.nn.functional.softmax(nt, 0),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Cannot apply softmax across nested dimension 0",
            lambda: torch.nn.functional.softmax(nt, -3),
        )
        # error case: dimension out of range
        self.assertRaises(IndexError, lambda: torch.nn.functional.softmax(nt, 3))
        self.assertRaises(IndexError, lambda: torch.nn.functional.softmax(nt, -4))
        # normal case: should equal to padding -inf
        softmaxer = torch.nn.Softmax(1)
        y0 = softmaxer(nt)
        y1 = torch.nn.functional.softmax(nt, 1)
        self.assertEqual(y0, y1)
        pt = torch.nested.to_padded_tensor(nt, float("-inf"))
        # if an entire slice is padded, then softmax will return 0.0 / 0.0 = nan
        # however, physically speaking that should be 0.0
        expect = torch.nn.functional.softmax(pt, 1).nan_to_num_(0.0)
        self.assertEqual(torch.nested.to_padded_tensor(y0, 0.0), expect)
        # edge case: empty nested tensor
        nt0 = torch.nested.nested_tensor([])
        y = torch.nn.functional.softmax(nt0, 1)
        self.assertEqual(nt0, y)
        # edge case: nesting scalars
        nt1 = torch.nested.nested_tensor([torch.tensor(0.0), torch.tensor(1.0)])
        self.assertRaises(RuntimeError, lambda: torch.nn.functional.softmax(nt1, 0))
        self.assertRaises(IndexError, lambda: torch.nn.functional.softmax(nt1, 1))

    @dtypes(torch.float, torch.double)
    @torch.inference_mode()
    def test_softmax_noncontiguous(self, device, dtype):
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3, 6, 7), device, dtype
        )
        self.assertEqual(
            torch.nn.functional.softmax(nt_contiguous, -1),
            torch.nn.functional.softmax(nt_noncontiguous, -1),
        )

    def _test_bmm(self, device, dtype):
        # error case: not 3D tensors
        nt0 = torch.nested.nested_tensor([], device=device, dtype=dtype)
        nt1 = torch.nested.nested_tensor(
            [torch.randn(2), torch.randn(3)], device=device, dtype=dtype
        )
        nt2 = torch.nested.nested_tensor(
            [torch.randn((2, 4)), torch.randn((3, 4))], device=device, dtype=dtype
        )
        self.assertRaisesRegex(
            RuntimeError, "batch1 must be a 3D tensor", lambda: nt0.bmm(nt0)
        )
        self.assertRaisesRegex(
            RuntimeError, "batch1 must be a 3D tensor", lambda: nt0.bmm(nt1)
        )
        self.assertRaisesRegex(
            RuntimeError, "batch1 must be a 3D tensor", lambda: nt0.bmm(nt2)
        )
        self.assertRaisesRegex(
            RuntimeError, "batch1 must be a 3D tensor", lambda: nt1.bmm(nt0)
        )
        self.assertRaisesRegex(
            RuntimeError, "batch1 must be a 3D tensor", lambda: nt1.bmm(nt1)
        )
        self.assertRaisesRegex(
            RuntimeError, "batch1 must be a 3D tensor", lambda: nt1.bmm(nt2)
        )
        self.assertRaisesRegex(
            RuntimeError, "batch2 must be a 3D tensor", lambda: nt2.bmm(nt0)
        )
        self.assertRaisesRegex(
            RuntimeError, "batch2 must be a 3D tensor", lambda: nt2.bmm(nt1)
        )
        # error case: incompatible batch size
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 4)), torch.randn((3, 4))], device=device, dtype=dtype
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((4, 6)), torch.randn((4, 5)), torch.randn((4, 7))],
            device=device,
            dtype=dtype,
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Expected size for the 1st dimension of batch2 tensor to be: 2 but got: 3.",
            lambda: nt0.bmm(nt1),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Expected size for the 1st dimension of batch2 tensor to be: 3 but got: 2.",
            lambda: nt1.bmm(nt0),
        )
        # error case: underlying matrices cannot be multiplied
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 4)), torch.randn((3, 4))], device=device, dtype=dtype
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"0-th nested matrices in batch cannot be multiplied \(2x4 and 2x4\)",
            lambda: nt0.bmm(nt0),
        )
        # normal nested tensor
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 4)), torch.randn((3, 7))], device=device, dtype=dtype
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((4, 6)), torch.randn((7, 5))], device=device, dtype=dtype
        )
        actual = torch.nested.to_padded_tensor(nt0.bmm(nt1), 0.0)
        expect = torch.nested.to_padded_tensor(nt0, 0.0).bmm(
            torch.nested.to_padded_tensor(nt1, 0.0)
        )
        if dtype == torch.float16:
            self.assertEqual(actual, expect, rtol=1e-3, atol=1e-3)
        else:
            self.assertEqual(actual, expect)

        # nested tensor bmm normal tensor
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 7)), torch.randn((3, 7))], device=device, dtype=dtype
        )
        nt1 = torch.rand(2, 7, 5, dtype=dtype, device=device)
        actual = torch.nested.to_padded_tensor(nt0.bmm(nt1), 0.0)
        expect = torch.nested.to_padded_tensor(nt0, 0.0).bmm(nt1)
        if dtype == torch.float16:
            self.assertEqual(actual, expect, rtol=1e-3, atol=1e-3)
        else:
            self.assertEqual(actual, expect)

        # nested tensor bmm normal tensor with non-contiguous view
        nt1 = torch.rand(2, 5, 7, dtype=dtype, device=device)
        nt1 = nt1.transpose(1, 2)
        actual = torch.nested.to_padded_tensor(nt0.bmm(nt1), 0.0)
        expect = torch.nested.to_padded_tensor(nt0, 0.0).bmm(nt1)
        if dtype == torch.float16:
            self.assertEqual(actual, expect, rtol=1e-3, atol=1e-3)
        else:
            self.assertEqual(actual, expect)

        # normal tensor bmm nested tensor
        nt0 = torch.rand(2, 5, 7, dtype=dtype, device=device)
        nt1 = torch.nested.nested_tensor(
            [torch.randn((7, 6)), torch.randn((7, 5))], device=device, dtype=dtype
        )
        actual = torch.nested.to_padded_tensor(nt0.bmm(nt1), 0.0)
        expect = nt0.bmm(torch.nested.to_padded_tensor(nt1, 0.0))
        if dtype == torch.float16:
            self.assertEqual(actual, expect, rtol=1e-3, atol=1e-3)
        else:
            self.assertEqual(actual, expect)

        # test tensorcore path
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 8)), torch.randn((3, 16))], device=device, dtype=dtype
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((8, 8)), torch.randn((16, 8))], device=device, dtype=dtype
        )
        actual = torch.nested.to_padded_tensor(nt0.bmm(nt1), 0.0)
        expect = torch.nested.to_padded_tensor(nt0, 0.0).bmm(
            torch.nested.to_padded_tensor(nt1, 0.0)
        )
        if dtype == torch.float16:
            self.assertEqual(actual, expect, rtol=1e-3, atol=1e-3)
        else:
            self.assertEqual(actual, expect)

    @onlyCUDA
    @dtypes(torch.float, torch.double, torch.float16, torch.bfloat16)
    @tf32_on_and_off(0.005)
    def test_bmm_cuda(self, device, dtype):
        self._test_bmm(device, dtype)

    @onlyCPU
    # cannot test torch.float16 because: RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'
    @dtypes(torch.float, torch.double)
    def test_bmm_cpu(self, device, dtype):
        self._test_bmm(device, dtype)

    # cannot test torch.float16 because: RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'
    @dtypes(torch.float, torch.double)
    def test_bmm_noncontiguous(self, device, dtype):
        nt0_contiguous, nt0_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3), device, dtype
        )
        nt1_contiguous, nt1_noncontiguous = random_nt_noncontiguous_pair(
            (6, 7), device, dtype
        )
        self.assertEqual(
            nt0_contiguous.transpose(-1, -2).bmm(nt1_contiguous),
            nt0_noncontiguous.transpose(-1, -2).bmm(nt1_noncontiguous),
        )

    @dtypes(torch.float, torch.double)
    @tf32_on_and_off(0.005)
    def test_matmul_with_bmm_path(self, device, dtype):
        def unbind_rebind_matmul(nt1, nt2):
            t1s = nt1.unbind()
            t2s = nt2.unbind()
            out_ts = [t1.matmul(t2) for t1, t2 in zip(t1s, t2s)]
            return torch.nested.nested_tensor(out_ts)

        # [N, n_head, *, head_dim], [N, n_head, head_dim, *]
        Ns = [1, 2, 5]
        n_heads = np.random.randint(2, 5)
        head_dim = 3
        t1s = []
        t2s = []
        for N in Ns:
            for _ in range(N):
                seq_len1 = np.random.randint(2, 5)
                seq_len2 = np.random.randint(2, 5)
                t1s.append(torch.randn(n_heads, seq_len1, head_dim))
                t2s.append(torch.randn(n_heads, head_dim, seq_len2))
            nt1 = torch.nested.nested_tensor(t1s, device=device, dtype=dtype)
            nt2 = torch.nested.nested_tensor(t2s, device=device, dtype=dtype)
            self.assertEqual(torch.matmul(nt1, nt2), unbind_rebind_matmul(nt1, nt2))

        # test with noncontiguous
        t3s = []
        t4s = []
        for _ in range(N):
            seq_len = np.random.randint(2, 5)
            t3s.append(torch.randn(seq_len, n_heads, head_dim))
            t4s.append(torch.randn(seq_len, n_heads, head_dim))
        nt3 = torch.nested.nested_tensor(t3s, device=device, dtype=dtype).transpose(
            1, 2
        )
        nt4 = (
            torch.nested.nested_tensor(t4s, device=device, dtype=dtype)
            .transpose(1, 2)
            .transpose(2, 3)
        )
        self.assertEqual(torch.matmul(nt3, nt4), unbind_rebind_matmul(nt3, nt4))

    # cannot test torch.float16 because: RuntimeError: "bmm" not implemented for 'Half'
    @dtypes(torch.float, torch.double)
    def test_matmul(self, device, dtype):
        # error case: one is nested but the other is not
        nt = torch.nested.nested_tensor(
            [torch.randn(2), torch.randn(3)], device=device, dtype=dtype
        )
        t = torch.randn(4, device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both to be nested, but got a nested self and non-nested other",
            lambda: torch.matmul(nt, t),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both to be nested, but got a non-nested self and nested other",
            lambda: torch.matmul(t, nt),
        )
        # error case: not 3+D tensors
        nt0 = torch.nested.nested_tensor([], device=device, dtype=dtype)
        nt1 = torch.nested.nested_tensor(
            [torch.randn(2), torch.randn(3)], device=device, dtype=dtype
        )
        nt2 = torch.nested.nested_tensor(
            [torch.randn((2, 4)), torch.randn((3, 4))], device=device, dtype=dtype
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 1st input has rank: [0-9]+",
            lambda: torch.matmul(nt0, nt0),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 1st input has rank: [0-9]+",
            lambda: torch.matmul(nt0, nt1),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 1st input has rank: [0-9]+",
            lambda: torch.matmul(nt0, nt2),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 1st input has rank: [0-9]+",
            lambda: torch.matmul(nt1, nt0),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 1st input has rank: [0-9]+",
            lambda: torch.matmul(nt1, nt1),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 1st input has rank: [0-9]+",
            lambda: torch.matmul(nt1, nt2),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 2nd input has rank: [0-9]+",
            lambda: torch.matmul(nt2, nt0),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 2nd input has rank: [0-9]+",
            lambda: torch.matmul(nt2, nt1),
        )
        # error case: incompatible batch size
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 4)), torch.randn((3, 4))], device=device, dtype=dtype
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((4, 6)), torch.randn((4, 5)), torch.randn((4, 7))],
            device=device,
            dtype=dtype,
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: Expected size for the 1st dimension of 2nd input tensor to be: [0-9]+ but got: [0-9]+.",
            lambda: torch.matmul(nt0, nt1),
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: Expected size for the 1st dimension of 2nd input tensor to be: [0-9]+ but got: [0-9]+.",
            lambda: torch.matmul(nt1, nt0),
        )
        # error case: incompatible (wrong) batch sizes that shouldn't even broadcast?
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 2, 4)), torch.randn((2, 3, 4))], device=device, dtype=dtype
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((3, 4, 6)), torch.randn((3, 4, 5))], device=device, dtype=dtype
        )
        self.assertRaisesRegex(
            RuntimeError,
            "matmul(): For nested tensors, batch dimensions must have the same sizes,",
            lambda: torch.matmul(nt0, nt1),
        )
        # error case: incompatible batch sizes that should technically broadcast
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 2, 4)), torch.randn((1, 3, 4))], device=device, dtype=dtype
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((1, 4, 6)), torch.randn((3, 4, 5))], device=device, dtype=dtype
        )
        self.assertRaisesRegex(
            RuntimeError,
            "matmul(): For nested tensors, batch dimensions must have the same sizes,",
            lambda: torch.matmul(nt0, nt1),
        )
        # error case: underlying matrices cannot be multiplied
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 4)), torch.randn((3, 4))], device=device, dtype=dtype
        )
        self.assertRaisesRegex(
            RuntimeError,
            "matmul(): Nested tensors cannot be matrix multiplied",
            lambda: torch.matmul(nt0, nt0),
        )
        # normal nested tensor: 3D
        nt0 = torch.nested.nested_tensor(
            [torch.randn((2, 4)), torch.randn((3, 7))], device=device, dtype=dtype
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((4, 6)), torch.randn((7, 5))], device=device, dtype=dtype
        )
        actual = torch.nested.to_padded_tensor(torch.matmul(nt0, nt1), 0.0)
        expect = torch.matmul(
            torch.nested.to_padded_tensor(nt0, 0.0),
            torch.nested.to_padded_tensor(nt1, 0.0),
        )
        self.assertEqual(actual, expect)
        # normal nested tensor: 4D (with testing for batch_size=1)
        nt0 = torch.nested.nested_tensor(
            [torch.randn((1, 2, 4)), torch.randn((8, 3, 7))], device=device, dtype=dtype
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((1, 4, 6)), torch.randn((8, 7, 5))], device=device, dtype=dtype
        )
        actual = torch.nested.to_padded_tensor(torch.matmul(nt0, nt1), 0.0)
        expect = torch.matmul(
            torch.nested.to_padded_tensor(nt0, 0.0),
            torch.nested.to_padded_tensor(nt1, 0.0),
        )
        self.assertEqual(actual, expect)
        # normal nested tensor: 5D
        nt0 = torch.nested.nested_tensor(
            [torch.randn((8, 9, 2, 4)), torch.randn((8, 9, 3, 7))],
            device=device,
            dtype=dtype,
        )
        nt1 = torch.nested.nested_tensor(
            [torch.randn((8, 9, 4, 6)), torch.randn((8, 9, 7, 5))],
            device=device,
            dtype=dtype,
        )
        actual = torch.nested.to_padded_tensor(torch.matmul(nt0, nt1), 0.0)
        expect = torch.matmul(
            torch.nested.to_padded_tensor(nt0, 0.0),
            torch.nested.to_padded_tensor(nt1, 0.0),
        )
        self.assertEqual(actual, expect)

    # only supported on CUDA for now
    @dtypes(torch.float, torch.double)
    def test_matmul_nt_with_broadcasted_t(self, device, dtype):
        # NT (B, *, C, D) with T (D, E) broadcasting case
        nt = random_nt_from_dims([3, None, 4, 5], device=device, dtype=dtype)
        t = torch.randn(5, 6, device=device, dtype=dtype)
        output = torch.matmul(nt, t)

        # should be equivalent to matmul-ing each component with the dense tensor
        self.assertEqual(nt.size(0), output.size(0))
        for component, out_component in zip(nt, output):
            self.assertEqual(out_component, torch.matmul(component, t))

    # cannot test torch.float16 because: RuntimeError: "bmm" not implemented for 'Half'
    @dtypes(torch.float, torch.double)
    def test_matmul_noncontiguous(self, device, dtype):
        nt0_contiguous, nt0_noncontiguous = random_nt_noncontiguous_pair(
            (2, 3), device, dtype
        )
        nt1_contiguous, nt1_noncontiguous = random_nt_noncontiguous_pair(
            (6, 7), device, dtype
        )
        self.assertEqual(
            torch.matmul(nt0_contiguous.transpose(-1, -2), nt1_contiguous),
            torch.matmul(nt0_noncontiguous.transpose(-1, -2), nt1_noncontiguous),
        )

    @dtypes(torch.float, torch.double)
    def test_linear(self, device, dtype):
        a = torch.randn(1, 2, device=device, dtype=dtype)
        b = torch.randn(2, 2, device=device, dtype=dtype)
        c = torch.randn(3, 2, device=device, dtype=dtype)
        nt = torch.nested.nested_tensor([a, b, c])

        weight = torch.randn(2, 2, device=device, dtype=dtype)
        bias = torch.randn(2, device=device, dtype=dtype)
        # success case
        torch.functional.F.linear(nt, weight, bias)

        # invalid nested tensor di

... (truncated, file too large)
```

## High-Level Overview

This file is part of the PyTorch repository. It is a Python source file that may contain classes, functions, and module-level code.

## Detailed Walkthrough

### Classes
This file defines 9 class(es): TestNestedTensor, TestNestedTensorDeviceType, TestNestedTensorAutograd, TestNestedTensorSubclass, mha, MyFunction, CustomDispatchMode, TestNestedTensorOpInfo, TestNestedInt

### Functions
This file defines 366 function(s): _iter_constructors, _recompiles_for_inputs, counter, random_nt_noncontiguous_pair, noncontiguous_to_padded_tensor, random_nt, random_nt_from_dims, random_nt_from_similar, layout_name, get_op_name, convert_dense_to_nested_tensor_legacy, convert_jagged_to_nested_tensor_legacy, convert_nt_to_jagged_legacy, convert_dense_to_nested_tensor, convert_jagged_to_nested_tensor, convert_nt_to_jagged, test_2d_nested_tensor, test_3d_nested_tensor, test_3d_nested_tensor_float, _test_unbind_case, test_unbind_0, test_unbind_1, test_unbind_3, test_unbind_4, test_unbind_dim, _test_fn, test_nested_tensor, test_nested_tensor_matching_dim, test_default_nested_tensor, test_dim


## Key Components

The file contains 28259 words across 9132 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 359210 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
