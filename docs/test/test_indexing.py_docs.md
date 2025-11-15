# Documentation: `test/test_indexing.py`

## File Metadata

- **Path**: `test/test_indexing.py`
- **Size**: 98,299 bytes (96.00 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: tests"]

import operator
import random
import unittest
import warnings
from functools import reduce
from itertools import product

import numpy as np

import torch
from torch import tensor
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCPU,
    dtypesIfCUDA,
    dtypesIfMPS,
    dtypesIfXPU,
    expectedFailureMPS,
    instantiate_device_type_tests,
    onlyCPU,
    onlyNativeDeviceTypes,
    onlyOn,
    skipXLA,
    skipXPUIf,
)
from torch.testing._internal.common_dtype import (
    all_mps_types_and,
    all_types_and,
    all_types_and_complex_and,
    all_types_complex_float8_and,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    parametrize,
    run_tests,
    serialTest,
    skipIfTorchDynamo,
    TEST_CUDA,
    TEST_MPS,
    TEST_XPU,
    TestCase,
    xfailIfTorchDynamo,
)


class TestIndexing(TestCase):
    def test_index(self, device):
        def consec(size, start=1):
            sequence = torch.ones(torch.tensor(size).prod(0)).cumsum(0)
            sequence.add_(start - 1)
            return sequence.view(*size)

        reference = consec((3, 3, 3)).to(device)

        # empty tensor indexing
        self.assertEqual(
            reference[torch.LongTensor().to(device)], reference.new(0, 3, 3)
        )

        self.assertEqual(reference[0], consec((3, 3)), atol=0, rtol=0)
        self.assertEqual(reference[1], consec((3, 3), 10), atol=0, rtol=0)
        self.assertEqual(reference[2], consec((3, 3), 19), atol=0, rtol=0)
        self.assertEqual(reference[0, 1], consec((3,), 4), atol=0, rtol=0)
        self.assertEqual(reference[0:2], consec((2, 3, 3)), atol=0, rtol=0)
        self.assertEqual(reference[2, 2, 2], 27, atol=0, rtol=0)
        self.assertEqual(reference[:], consec((3, 3, 3)), atol=0, rtol=0)

        # indexing with Ellipsis
        self.assertEqual(
            reference[..., 2],
            torch.tensor([[3.0, 6.0, 9.0], [12.0, 15.0, 18.0], [21.0, 24.0, 27.0]]),
            atol=0,
            rtol=0,
        )
        self.assertEqual(
            reference[0, ..., 2], torch.tensor([3.0, 6.0, 9.0]), atol=0, rtol=0
        )
        self.assertEqual(reference[..., 2], reference[:, :, 2], atol=0, rtol=0)
        self.assertEqual(reference[0, ..., 2], reference[0, :, 2], atol=0, rtol=0)
        self.assertEqual(reference[0, 2, ...], reference[0, 2], atol=0, rtol=0)
        self.assertEqual(reference[..., 2, 2, 2], 27, atol=0, rtol=0)
        self.assertEqual(reference[2, ..., 2, 2], 27, atol=0, rtol=0)
        self.assertEqual(reference[2, 2, ..., 2], 27, atol=0, rtol=0)
        self.assertEqual(reference[2, 2, 2, ...], 27, atol=0, rtol=0)
        self.assertEqual(reference[...], reference, atol=0, rtol=0)

        reference_5d = consec((3, 3, 3, 3, 3)).to(device)
        self.assertEqual(
            reference_5d[..., 1, 0], reference_5d[:, :, :, 1, 0], atol=0, rtol=0
        )
        self.assertEqual(
            reference_5d[2, ..., 1, 0], reference_5d[2, :, :, 1, 0], atol=0, rtol=0
        )
        self.assertEqual(
            reference_5d[2, 1, 0, ..., 1], reference_5d[2, 1, 0, :, 1], atol=0, rtol=0
        )
        self.assertEqual(reference_5d[...], reference_5d, atol=0, rtol=0)

        # LongTensor indexing
        reference = consec((5, 5, 5)).to(device)
        idx = torch.LongTensor([2, 4]).to(device)
        self.assertEqual(reference[idx], torch.stack([reference[2], reference[4]]))
        # TODO: enable one indexing is implemented like in numpy
        # self.assertEqual(reference[2, idx], torch.stack([reference[2, 2], reference[2, 4]]))
        # self.assertEqual(reference[3, idx, 1], torch.stack([reference[3, 2], reference[3, 4]])[:, 1])

        # None indexing
        self.assertEqual(reference[2, None], reference[2].unsqueeze(0))
        self.assertEqual(
            reference[2, None, None], reference[2].unsqueeze(0).unsqueeze(0)
        )
        self.assertEqual(reference[2:4, None], reference[2:4].unsqueeze(1))
        self.assertEqual(
            reference[None, 2, None, None],
            reference.unsqueeze(0)[:, 2].unsqueeze(0).unsqueeze(0),
        )
        self.assertEqual(
            reference[None, 2:5, None, None],
            reference.unsqueeze(0)[:, 2:5].unsqueeze(2).unsqueeze(2),
        )

        # indexing 0-length slice
        self.assertEqual(torch.empty(0, 5, 5), reference[slice(0)])
        self.assertEqual(torch.empty(0, 5), reference[slice(0), 2])
        self.assertEqual(torch.empty(0, 5), reference[2, slice(0)])
        self.assertEqual(torch.tensor([]), reference[2, 1:1, 2])

        # indexing with step
        reference = consec((10, 10, 10)).to(device)
        self.assertEqual(reference[1:5:2], torch.stack([reference[1], reference[3]], 0))
        self.assertEqual(
            reference[1:6:2], torch.stack([reference[1], reference[3], reference[5]], 0)
        )
        self.assertEqual(reference[1:9:4], torch.stack([reference[1], reference[5]], 0))
        self.assertEqual(
            reference[2:4, 1:5:2],
            torch.stack([reference[2:4, 1], reference[2:4, 3]], 1),
        )
        self.assertEqual(
            reference[3, 1:6:2],
            torch.stack([reference[3, 1], reference[3, 3], reference[3, 5]], 0),
        )
        self.assertEqual(
            reference[None, 2, 1:9:4],
            torch.stack([reference[2, 1], reference[2, 5]], 0).unsqueeze(0),
        )
        self.assertEqual(
            reference[:, 2, 1:6:2],
            torch.stack(
                [reference[:, 2, 1], reference[:, 2, 3], reference[:, 2, 5]], 1
            ),
        )

        lst = [list(range(i, i + 10)) for i in range(0, 100, 10)]
        _make_tensor = (
            torch.DoubleTensor if not device.startswith("mps") else torch.FloatTensor
        )
        tensor = _make_tensor(lst).to(device)
        for _ in range(100):
            idx1_start = random.randrange(10)
            idx1_end = idx1_start + random.randrange(1, 10 - idx1_start + 1)
            idx1_step = random.randrange(1, 8)
            idx1 = slice(idx1_start, idx1_end, idx1_step)
            if random.randrange(2) == 0:
                idx2_start = random.randrange(10)
                idx2_end = idx2_start + random.randrange(1, 10 - idx2_start + 1)
                idx2_step = random.randrange(1, 8)
                idx2 = slice(idx2_start, idx2_end, idx2_step)
                lst_indexed = [l[idx2] for l in lst[idx1]]
                tensor_indexed = tensor[idx1, idx2]
            else:
                lst_indexed = lst[idx1]
                tensor_indexed = tensor[idx1]
            self.assertEqual(_make_tensor(lst_indexed), tensor_indexed)

        self.assertRaises(ValueError, lambda: reference[1:9:0])
        self.assertRaises(ValueError, lambda: reference[1:9:-1])

        self.assertRaises(IndexError, lambda: reference[1, 1, 1, 1])
        self.assertRaises(IndexError, lambda: reference[1, 1, 1, 1:1])
        self.assertRaises(IndexError, lambda: reference[3, 3, 3, 3, 3, 3, 3, 3])

        self.assertRaises(IndexError, lambda: reference[0.0])
        self.assertRaises(TypeError, lambda: reference[0.0:2.0])
        self.assertRaises(IndexError, lambda: reference[0.0, 0.0:2.0])
        self.assertRaises(IndexError, lambda: reference[0.0, :, 0.0:2.0])
        self.assertRaises(IndexError, lambda: reference[0.0, ..., 0.0:2.0])
        self.assertRaises(IndexError, lambda: reference[0.0, :, 0.0])

        def delitem():
            del reference[0]

        self.assertRaises(TypeError, delitem)

    @onlyNativeDeviceTypes
    @dtypes(torch.half, torch.double)
    @dtypesIfMPS(torch.half)  # TODO: add bf16 there?
    def test_advancedindex(self, device, dtype):
        # Tests for Integer Array Indexing, Part I - Purely integer array
        # indexing

        def consec(size, start=1):
            # Creates the sequence in float since CPU half doesn't support the
            # needed operations. Converts to dtype before returning.
            numel = reduce(operator.mul, size, 1)
            sequence = torch.ones(numel, dtype=torch.float, device=device).cumsum(0)
            sequence.add_(start - 1)
            return sequence.view(*size).to(dtype=dtype)

        # pick a random valid indexer type
        def ri(indices):
            choice = random.randint(0, 2)
            if choice == 0:
                return torch.LongTensor(indices).to(device)
            elif choice == 1:
                return list(indices)
            else:
                return tuple(indices)

        def validate_indexing(x):
            self.assertEqual(x[[0]], consec((1,)))
            self.assertEqual(x[ri([0]),], consec((1,)))
            self.assertEqual(x[ri([3]),], consec((1,), 4))
            self.assertEqual(x[[2, 3, 4]], consec((3,), 3))
            self.assertEqual(x[ri([2, 3, 4]),], consec((3,), 3))
            self.assertEqual(
                x[ri([0, 2, 4]),], torch.tensor([1, 3, 5], dtype=dtype, device=device)
            )

        def validate_setting(x):
            x[[0]] = -2
            self.assertEqual(x[[0]], torch.tensor([-2], dtype=dtype, device=device))
            x[[0]] = -1
            self.assertEqual(
                x[ri([0]),], torch.tensor([-1], dtype=dtype, device=device)
            )
            x[[2, 3, 4]] = 4
            self.assertEqual(
                x[[2, 3, 4]], torch.tensor([4, 4, 4], dtype=dtype, device=device)
            )
            x[ri([2, 3, 4]),] = 3
            self.assertEqual(
                x[ri([2, 3, 4]),], torch.tensor([3, 3, 3], dtype=dtype, device=device)
            )
            x[ri([0, 2, 4]),] = torch.tensor([5, 4, 3], dtype=dtype, device=device)
            self.assertEqual(
                x[ri([0, 2, 4]),], torch.tensor([5, 4, 3], dtype=dtype, device=device)
            )

        # Only validates indexing and setting for Halves
        if dtype == torch.half:
            reference = consec((10,))
            validate_indexing(reference)
            validate_setting(reference)
            return

        # Case 1: Purely Integer Array Indexing
        reference = consec((10,))
        validate_indexing(reference)

        # setting values
        validate_setting(reference)

        # Tensor with stride != 1
        # strided is [1, 3, 5, 7]
        reference = consec((10,))
        strided = torch.tensor((), dtype=dtype, device=device)
        strided.set_(
            reference.untyped_storage(),
            storage_offset=0,
            size=torch.Size([4]),
            stride=[2],
        )

        self.assertEqual(strided[[0]], torch.tensor([1], dtype=dtype, device=device))
        self.assertEqual(
            strided[ri([0]),], torch.tensor([1], dtype=dtype, device=device)
        )
        self.assertEqual(
            strided[ri([3]),], torch.tensor([7], dtype=dtype, device=device)
        )
        self.assertEqual(
            strided[[1, 2]], torch.tensor([3, 5], dtype=dtype, device=device)
        )
        self.assertEqual(
            strided[ri([1, 2]),], torch.tensor([3, 5], dtype=dtype, device=device)
        )
        self.assertEqual(
            strided[ri([[2, 1], [0, 3]]),],
            torch.tensor([[5, 3], [1, 7]], dtype=dtype, device=device),
        )

        # stride is [4, 8]
        strided = torch.tensor((), dtype=dtype, device=device)
        strided.set_(
            reference.untyped_storage(),
            storage_offset=4,
            size=torch.Size([2]),
            stride=[4],
        )
        self.assertEqual(strided[[0]], torch.tensor([5], dtype=dtype, device=device))
        self.assertEqual(
            strided[ri([0]),], torch.tensor([5], dtype=dtype, device=device)
        )
        self.assertEqual(
            strided[ri([1]),], torch.tensor([9], dtype=dtype, device=device)
        )
        self.assertEqual(
            strided[[0, 1]], torch.tensor([5, 9], dtype=dtype, device=device)
        )
        self.assertEqual(
            strided[ri([0, 1]),], torch.tensor([5, 9], dtype=dtype, device=device)
        )
        self.assertEqual(
            strided[ri([[0, 1], [1, 0]]),],
            torch.tensor([[5, 9], [9, 5]], dtype=dtype, device=device),
        )

        # reference is 1 2
        #              3 4
        #              5 6
        reference = consec((3, 2))
        self.assertEqual(
            reference[ri([0, 1, 2]), ri([0])],
            torch.tensor([1, 3, 5], dtype=dtype, device=device),
        )
        self.assertEqual(
            reference[ri([0, 1, 2]), ri([1])],
            torch.tensor([2, 4, 6], dtype=dtype, device=device),
        )
        self.assertEqual(reference[ri([0]), ri([0])], consec((1,)))
        self.assertEqual(reference[ri([2]), ri([1])], consec((1,), 6))
        self.assertEqual(
            reference[(ri([0, 0]), ri([0, 1]))],
            torch.tensor([1, 2], dtype=dtype, device=device),
        )
        self.assertEqual(
            reference[(ri([0, 1, 1, 0, 2]), ri([1]))],
            torch.tensor([2, 4, 4, 2, 6], dtype=dtype, device=device),
        )
        self.assertEqual(
            reference[(ri([0, 0, 1, 1]), ri([0, 1, 0, 0]))],
            torch.tensor([1, 2, 3, 3], dtype=dtype, device=device),
        )

        rows = ri([[0, 0], [1, 2]])
        columns = ([0],)
        self.assertEqual(
            reference[rows, columns],
            torch.tensor([[1, 1], [3, 5]], dtype=dtype, device=device),
        )

        rows = ri([[0, 0], [1, 2]])
        columns = ri([1, 0])
        self.assertEqual(
            reference[rows, columns],
            torch.tensor([[2, 1], [4, 5]], dtype=dtype, device=device),
        )
        rows = ri([[0, 0], [1, 2]])
        columns = ri([[0, 1], [1, 0]])
        self.assertEqual(
            reference[rows, columns],
            torch.tensor([[1, 2], [4, 5]], dtype=dtype, device=device),
        )

        # setting values
        reference[ri([0]), ri([1])] = -1
        self.assertEqual(
            reference[ri([0]), ri([1])], torch.tensor([-1], dtype=dtype, device=device)
        )
        reference[ri([0, 1, 2]), ri([0])] = torch.tensor(
            [-1, 2, -4], dtype=dtype, device=device
        )
        self.assertEqual(
            reference[ri([0, 1, 2]), ri([0])],
            torch.tensor([-1, 2, -4], dtype=dtype, device=device),
        )
        reference[rows, columns] = torch.tensor(
            [[4, 6], [2, 3]], dtype=dtype, device=device
        )
        self.assertEqual(
            reference[rows, columns],
            torch.tensor([[4, 6], [2, 3]], dtype=dtype, device=device),
        )

        # Verify still works with Transposed (i.e. non-contiguous) Tensors

        reference = torch.tensor(
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype=dtype, device=device
        ).t_()

        # Transposed: [[0, 4, 8],
        #              [1, 5, 9],
        #              [2, 6, 10],
        #              [3, 7, 11]]

        self.assertEqual(
            reference[ri([0, 1, 2]), ri([0])],
            torch.tensor([0, 1, 2], dtype=dtype, device=device),
        )
        self.assertEqual(
            reference[ri([0, 1, 2]), ri([1])],
            torch.tensor([4, 5, 6], dtype=dtype, device=device),
        )
        self.assertEqual(
            reference[ri([0]), ri([0])], torch.tensor([0], dtype=dtype, device=device)
        )
        self.assertEqual(
            reference[ri([2]), ri([1])], torch.tensor([6], dtype=dtype, device=device)
        )
        self.assertEqual(
            reference[(ri([0, 0]), ri([0, 1]))],
            torch.tensor([0, 4], dtype=dtype, device=device),
        )
        self.assertEqual(
            reference[(ri([0, 1, 1, 0, 3]), ri([1]))],
            torch.tensor([4, 5, 5, 4, 7], dtype=dtype, device=device),
        )
        self.assertEqual(
            reference[(ri([0, 0, 1, 1]), ri([0, 1, 0, 0]))],
            torch.tensor([0, 4, 1, 1], dtype=dtype, device=device),
        )

        rows = ri([[0, 0], [1, 2]])
        columns = ([0],)
        self.assertEqual(
            reference[rows, columns],
            torch.tensor([[0, 0], [1, 2]], dtype=dtype, device=device),
        )

        rows = ri([[0, 0], [1, 2]])
        columns = ri([1, 0])
        self.assertEqual(
            reference[rows, columns],
            torch.tensor([[4, 0], [5, 2]], dtype=dtype, device=device),
        )
        rows = ri([[0, 0], [1, 3]])
        columns = ri([[0, 1], [1, 2]])
        self.assertEqual(
            reference[rows, columns],
            torch.tensor([[0, 4], [5, 11]], dtype=dtype, device=device),
        )

        # setting values
        reference[ri([0]), ri([1])] = -1
        self.assertEqual(
            reference[ri([0]), ri([1])], torch.tensor([-1], dtype=dtype, device=device)
        )
        reference[ri([0, 1, 2]), ri([0])] = torch.tensor(
            [-1, 2, -4], dtype=dtype, device=device
        )
        self.assertEqual(
            reference[ri([0, 1, 2]), ri([0])],
            torch.tensor([-1, 2, -4], dtype=dtype, device=device),
        )
        reference[rows, columns] = torch.tensor(
            [[4, 6], [2, 3]], dtype=dtype, device=device
        )
        self.assertEqual(
            reference[rows, columns],
            torch.tensor([[4, 6], [2, 3]], dtype=dtype, device=device),
        )

        # stride != 1

        # strided is [[1 3 5 7],
        #             [9 11 13 15]]

        reference = torch.arange(0.0, 24, dtype=dtype, device=device).view(3, 8)
        strided = torch.tensor((), dtype=dtype, device=device)
        strided.set_(
            reference.untyped_storage(), 1, size=torch.Size([2, 4]), stride=[8, 2]
        )

        self.assertEqual(
            strided[ri([0, 1]), ri([0])],
            torch.tensor([1, 9], dtype=dtype, device=device),
        )
        self.assertEqual(
            strided[ri([0, 1]), ri([1])],
            torch.tensor([3, 11], dtype=dtype, device=device),
        )
        self.assertEqual(
            strided[ri([0]), ri([0])], torch.tensor([1], dtype=dtype, device=device)
        )
        self.assertEqual(
            strided[ri([1]), ri([3])], torch.tensor([15], dtype=dtype, device=device)
        )
        self.assertEqual(
            strided[(ri([0, 0]), ri([0, 3]))],
            torch.tensor([1, 7], dtype=dtype, device=device),
        )
        self.assertEqual(
            strided[(ri([1]), ri([0, 1, 1, 0, 3]))],
            torch.tensor([9, 11, 11, 9, 15], dtype=dtype, device=device),
        )
        self.assertEqual(
            strided[(ri([0, 0, 1, 1]), ri([0, 1, 0, 0]))],
            torch.tensor([1, 3, 9, 9], dtype=dtype, device=device),
        )

        rows = ri([[0, 0], [1, 1]])
        columns = ([0],)
        self.assertEqual(
            strided[rows, columns],
            torch.tensor([[1, 1], [9, 9]], dtype=dtype, device=device),
        )

        rows = ri([[0, 1], [1, 0]])
        columns = ri([1, 2])
        self.assertEqual(
            strided[rows, columns],
            torch.tensor([[3, 13], [11, 5]], dtype=dtype, device=device),
        )
        rows = ri([[0, 0], [1, 1]])
        columns = ri([[0, 1], [1, 2]])
        self.assertEqual(
            strided[rows, columns],
            torch.tensor([[1, 3], [11, 13]], dtype=dtype, device=device),
        )

        # setting values

        # strided is [[10, 11],
        #             [17, 18]]

        reference = torch.arange(0.0, 24, dtype=dtype, device=device).view(3, 8)
        strided = torch.tensor((), dtype=dtype, device=device)
        strided.set_(
            reference.untyped_storage(), 10, size=torch.Size([2, 2]), stride=[7, 1]
        )
        self.assertEqual(
            strided[ri([0]), ri([1])], torch.tensor([11], dtype=dtype, device=device)
        )
        strided[ri([0]), ri([1])] = -1
        self.assertEqual(
            strided[ri([0]), ri([1])], torch.tensor([-1], dtype=dtype, device=device)
        )

        reference = torch.arange(0.0, 24, dtype=dtype, device=device).view(3, 8)
        strided = torch.tensor((), dtype=dtype, device=device)
        strided.set_(
            reference.untyped_storage(), 10, size=torch.Size([2, 2]), stride=[7, 1]
        )
        self.assertEqual(
            strided[ri([0, 1]), ri([1, 0])],
            torch.tensor([11, 17], dtype=dtype, device=device),
        )
        strided[ri([0, 1]), ri([1, 0])] = torch.tensor(
            [-1, 2], dtype=dtype, device=device
        )
        self.assertEqual(
            strided[ri([0, 1]), ri([1, 0])],
            torch.tensor([-1, 2], dtype=dtype, device=device),
        )

        reference = torch.arange(0.0, 24, dtype=dtype, device=device).view(3, 8)
        strided = torch.tensor((), dtype=dtype, device=device)
        strided.set_(
            reference.untyped_storage(), 10, size=torch.Size([2, 2]), stride=[7, 1]
        )

        rows = ri([[0], [1]])
        columns = ri([[0, 1], [0, 1]])
        self.assertEqual(
            strided[rows, columns],
            torch.tensor([[10, 11], [17, 18]], dtype=dtype, device=device),
        )
        strided[rows, columns] = torch.tensor(
            [[4, 6], [2, 3]], dtype=dtype, device=device
        )
        self.assertEqual(
            strided[rows, columns],
            torch.tensor([[4, 6], [2, 3]], dtype=dtype, device=device),
        )

        # Tests using less than the number of dims, and ellipsis

        # reference is 1 2
        #              3 4
        #              5 6
        reference = consec((3, 2))
        self.assertEqual(
            reference[ri([0, 2]),],
            torch.tensor([[1, 2], [5, 6]], dtype=dtype, device=device),
        )
        self.assertEqual(
            reference[ri([1]), ...], torch.tensor([[3, 4]], dtype=dtype, device=device)
        )
        self.assertEqual(
            reference[..., ri([1])],
            torch.tensor([[2], [4], [6]], dtype=dtype, device=device),
        )

        # verify too many indices fails
        with self.assertRaises(IndexError):
            reference[ri([1]), ri([0, 2]), ri([3])]

        # test invalid index fails
        reference = torch.empty(10, dtype=dtype, device=device)
        # can't test cuda/xpu because it is a device assert
        if reference.device.type == "cpu":
            for err_idx in (10, -11):
                with self.assertRaisesRegex(IndexError, r"out of"):
                    reference[err_idx]
                with self.assertRaisesRegex(IndexError, r"out of"):
                    reference[torch.LongTensor([err_idx]).to(device)]
                with self.assertRaisesRegex(IndexError, r"out of"):
                    reference[[err_idx]]

        def tensor_indices_to_np(tensor, indices):
            # convert the Torch Tensor to a numpy array
            tensor = tensor.to(device="cpu")
            npt = tensor.numpy()

            # convert indices
            idxs = tuple(
                i.tolist() if isinstance(i, torch.LongTensor) else i for i in indices
            )

            return npt, idxs

        def get_numpy(tensor, indices):
            npt, idxs = tensor_indices_to_np(tensor, indices)

            # index and return as a Torch Tensor
            return torch.tensor(npt[idxs], dtype=dtype, device=device)

        def set_numpy(tensor, indices, value):
            if not isinstance(value, int):
                if self.device_type != "cpu":
                    value = value.cpu()
                value = value.numpy()

            npt, idxs = tensor_indices_to_np(tensor, indices)
            npt[idxs] = value
            return npt

        def assert_get_eq(tensor, indexer):
            self.assertEqual(tensor[indexer], get_numpy(tensor, indexer))

        def assert_set_eq(tensor, indexer, val):
            pyt = tensor.clone()
            numt = tensor.clone()
            pyt[indexer] = val
            numt = torch.tensor(
                set_numpy(numt, indexer, val), dtype=dtype, device=device
            )
            self.assertEqual(pyt, numt)

        def assert_backward_eq(tensor, indexer):
            cpu = tensor.float().detach().clone().requires_grad_(True)
            outcpu = cpu[indexer]
            gOcpu = torch.rand_like(outcpu)
            outcpu.backward(gOcpu)
            dev = cpu.to(device).detach().requires_grad_(True)
            outdev = dev[indexer]
            outdev.backward(gOcpu.to(device))
            self.assertEqual(cpu.grad, dev.grad)

        def get_set_tensor(indexed, indexer):
            set_size = indexed[indexer].size()
            set_count = indexed[indexer].numel()
            set_tensor = torch.randperm(set_count).view(set_size).double().to(device)
            return set_tensor

        # Tensor is  0  1  2  3  4
        #            5  6  7  8  9
        #           10 11 12 13 14
        #           15 16 17 18 19
        reference = torch.arange(0.0, 20, dtype=dtype, device=device).view(4, 5)

        indices_to_test = [
            # grab the second, fourth columns
            (slice(None), [1, 3]),
            # first, third rows,
            ([0, 2], slice(None)),
            # weird shape
            (slice(None), [[0, 1], [2, 3]]),
            # negatives
            ([-1], [0]),
            ([0, 2], [-1]),
            (slice(None), [-1]),
        ]

        # only test dupes on gets
        get_indices_to_test = indices_to_test + [(slice(None), [0, 1, 1, 2, 2])]

        for indexer in get_indices_to_test:
            assert_get_eq(reference, indexer)
            if self.device_type != "cpu":
                assert_backward_eq(reference, indexer)

        for indexer in indices_to_test:
            assert_set_eq(reference, indexer, 44)
            assert_set_eq(reference, indexer, get_set_tensor(reference, indexer))

        reference = torch.arange(0.0, 160, dtype=dtype, device=device).view(4, 8, 5)

        indices_to_test = [
            (slice(None), slice(None), (0, 3, 4)),
            (slice(None), (2, 4, 5, 7), slice(None)),
            ((2, 3), slice(None), slice(None)),
            (slice(None), (0, 2, 3), (1, 3, 4)),
            (slice(None), (0,), (1, 2, 4)),
            (slice(None), (0, 1, 3), (4,)),
            (slice(None), ((0, 1), (1, 0)), ((2, 3),)),
            (slice(None), ((0, 1), (2, 3)), ((0,),)),
            (slice(None), ((5, 6),), ((0, 3), (4, 4))),
            ((0, 2, 3), (1, 3, 4), slice(None)),
            ((0,), (1, 2, 4), slice(None)),
            ((0, 1, 3), (4,), slice(None)),
            (((0, 1), (1, 0)), ((2, 1), (3, 5)), slice(None)),
            (((0, 1), (1, 0)), ((2, 3),), slice(None)),
            (((0, 1), (2, 3)), ((0,),), slice(None)),
            (((2, 1),), ((0, 3), (4, 4)), slice(None)),
            (((2,),), ((0, 3), (4, 1)), slice(None)),
            # non-contiguous indexing subspace
            ((0, 2, 3), slice(None), (1, 3, 4)),
            # [...]
            # less dim, ellipsis
            ((0, 2),),
            ((0, 2), slice(None)),
            ((0, 2), Ellipsis),
            ((0, 2), slice(None), Ellipsis),
            ((0, 2), Ellipsis, slice(None)),
            ((0, 2), (1, 3)),
            ((0, 2), (1, 3), Ellipsis),
            (Ellipsis, (1, 3), (2, 3)),
            (Ellipsis, (2, 3, 4)),
            (Ellipsis, slice(None), (2, 3, 4)),
            (slice(None), Ellipsis, (2, 3, 4)),
            # ellipsis counts for nothing
            (Ellipsis, slice(None), slice(None), (0, 3, 4)),
            (slice(None), Ellipsis, slice(None), (0, 3, 4)),
            (slice(None), slice(None), Ellipsis, (0, 3, 4)),
            (slice(None), slice(None), (0, 3, 4), Ellipsis),
            (Ellipsis, ((0, 1), (1, 0)), ((2, 1), (3, 5)), slice(None)),
            (((0, 1), (1, 0)), ((2, 1), (3, 5)), Ellipsis, slice(None)),
            (((0, 1), (1, 0)), ((2, 1), (3, 5)), slice(None), Ellipsis),
        ]

        for indexer in indices_to_test:
            assert_get_eq(reference, indexer)
            assert_set_eq(reference, indexer, 212)
            assert_set_eq(reference, indexer, get_set_tensor(reference, indexer))
            if torch.accelerator.is_available():
                assert_backward_eq(reference, indexer)

        reference = torch.arange(0.0, 1296, dtype=dtype, device=device).view(3, 9, 8, 6)

        indices_to_test = [
            (slice(None), slice(None), slice(None), (0, 3, 4)),
            (slice(None), slice(None), (2, 4, 5, 7), slice(None)),
            (slice(None), (2, 3), slice(None), slice(None)),
            ((1, 2), slice(None), slice(None), slice(None)),
            (slice(None), slice(None), (0, 2, 3), (1, 3, 4)),
            (slice(None), slice(None), (0,), (1, 2, 4)),
            (slice(None), slice(None), (0, 1, 3), (4,)),
            (slice(None), slice(None), ((0, 1), (1, 0)), ((2, 3),)),
            (slice(None), slice(None), ((0, 1), (2, 3)), ((0,),)),
            (slice(None), slice(None), ((5, 6),), ((0, 3), (4, 4))),
            (slice(None), (0, 2, 3), (1, 3, 4), slice(None)),
            (slice(None), (0,), (1, 2, 4), slice(None)),
            (slice(None), (0, 1, 3), (4,), slice(None)),
            (slice(None), ((0, 1), (3, 4)), ((2, 3), (0, 1)), slice(None)),
            (slice(None), ((0, 1), (3, 4)), ((2, 3),), slice(None)),
            (slice(None), ((0, 1), (3, 2)), ((0,),), slice(None)),
            (slice(None), ((2, 1),), ((0, 3), (6, 4)), slice(None)),
            (slice(None), ((2,),), ((0, 3), (4, 2)), slice(None)),
            ((0, 1, 2), (1, 3, 4), slice(None), slice(None)),
            ((0,), (1, 2, 4), slice(None), slice(None)),
            ((0, 1, 2), (4,), slice(None), slice(None)),
            (((0, 1), (0, 2)), ((2, 4), (1, 5)), slice(None), slice(None)),
            (((0, 1), (1, 2)), ((2, 0),), slice(None), slice(None)),
            (((2, 2),), ((0, 3), (4, 5)), slice(None), slice(None)),
            (((2,),), ((0, 3), (4, 5)), slice(None), slice(None)),
            (slice(None), (3, 4, 6), (0, 2, 3), (1, 3, 4)),
            (slice(None), (2, 3, 4), (1, 3, 4), (4,)),
            (slice(None), (0, 1, 3), (4,), (1, 3, 4)),
            (slice(None), (6,), (0, 2, 3), (1, 3, 4)),
            (slice(None), (2, 3, 5), (3,), (4,)),
            (slice(None), (0,), (4,), (1, 3, 4)),
            (slice(None), (6,), (0, 2, 3), (1,)),
            (slice(None), ((0, 3), (3, 6)), ((0, 1), (1, 3)), ((5, 3), (1, 2))),
            ((2, 2, 1), (0, 2, 3), (1, 3, 4), slice(None)),
            ((2, 0, 1), (1, 2, 3), (4,), slice(None)),
            ((0, 1, 2), (4,), (1, 3, 4), slice(None)),
            ((0,), (0, 2, 3), (1, 3, 4), slice(None)),
            ((0, 2, 1), (3,), (4,), slice(None)),
            ((0,), (4,), (1, 3, 4), slice(None)),
            ((1,), (0, 2, 3), (1,), slice(None)),
            (((1, 2), (1, 2)), ((0, 1), (2, 3)), ((2, 3), (3, 5)), slice(None)),
            # less dim, ellipsis
            (Ellipsis, (0, 3, 4)),
            (Ellipsis, slice(None), (0, 3, 4)),
            (Ellipsis, slice(None), slice(None), (0, 3, 4)),
            (slice(None), Ellipsis, (0, 3, 4)),
            (slice(None), slice(None), Ellipsis, (0, 3, 4)),
            (slice(None), (0, 2, 3), (1, 3, 4)),
            (slice(None), (0, 2, 3), (1, 3, 4), Ellipsis),
            (Ellipsis, (0, 2, 3), (1, 3, 4), slice(None)),
            ((0,), (1, 2, 4)),
            ((0,), (1, 2, 4), slice(None)),
            ((0,), (1, 2, 4), Ellipsis),
            ((0,), (1, 2, 4), Ellipsis, slice(None)),
            ((1,),),
            ((0, 2, 1), (3,), (4,)),
            ((0, 2, 1), (3,), (4,), slice(None)),
            ((0, 2, 1), (3,), (4,), Ellipsis),
            (Ellipsis, (0, 2, 1), (3,), (4,)),
        ]

        for indexer in indices_to_test:
            assert_get_eq(reference, indexer)
            assert_set_eq(reference, indexer, 1333)
            assert_set_eq(reference, indexer, get_set_tensor(reference, indexer))
        indices_to_test += [
            (slice(None), slice(None), [[0, 1], [1, 0]], [[2, 3], [3, 0]]),
            (slice(None), slice(None), [[2]], [[0, 3], [4, 4]]),
        ]
        for indexer in indices_to_test:
            assert_get_eq(reference, indexer)
            assert_set_eq(reference, indexer, 1333)
            if self.device_type != "cpu":
                assert_backward_eq(reference, indexer)

    def test_advancedindex_big(self, device):
        reference = torch.arange(0, 123344, dtype=torch.int, device=device)

        self.assertEqual(
            reference[[0, 123, 44488, 68807, 123343],],
            torch.tensor([0, 123, 44488, 68807, 123343], dtype=torch.int),
        )

    def test_set_item_to_scalar_tensor(self, device):
        m = random.randint(1, 10)
        n = random.randint(1, 10)
        z = torch.randn([m, n], device=device)
        a = 1.0
        w = torch.tensor(a, requires_grad=True, device=device)
        z[:, 0] = w
        z.sum().backward()
        self.assertEqual(w.grad, m * a)

    def test_single_int(self, device):
        v = torch.randn(5, 7, 3, device=device)
        self.assertEqual(v[4].shape, (7, 3))

    def test_multiple_int(self, device):
        v = torch.randn(5, 7, 3, device=device)
        self.assertEqual(v[4].shape, (7, 3))
        self.assertEqual(v[4, :, 1].shape, (7,))

    def test_none(self, device):
        v = torch.randn(5, 7, 3, device=device)
        self.assertEqual(v[None].shape, (1, 5, 7, 3))
        self.assertEqual(v[:, None].shape, (5, 1, 7, 3))
        self.assertEqual(v[:, None, None].shape, (5, 1, 1, 7, 3))
        self.assertEqual(v[..., None].shape, (5, 7, 3, 1))

    def test_step(self, device):
        v = torch.arange(10, device=device)
        self.assertEqual(v[::1], v)
        self.assertEqual(v[::2].tolist(), [0, 2, 4, 6, 8])
        self.assertEqual(v[::3].tolist(), [0, 3, 6, 9])
        self.assertEqual(v[::11].tolist(), [0])
        self.assertEqual(v[1:6:2].tolist(), [1, 3, 5])

    def test_step_assignment(self, device):
        v = torch.zeros(4, 4, device=device)
        v[0, 1::2] = torch.tensor([3.0, 4.0], device=device)
        self.assertEqual(v[0].tolist(), [0, 3, 0, 4])
        self.assertEqual(v[1:].sum(), 0)

    def test_bool_indices(self, device):
        v = torch.randn(5, 7, 3, device=device)
        boolIndices = torch.tensor(
            [True, False, True, True, False], dtype=torch.bool, device=device
        )
        self.assertEqual(v[boolIndices].shape, (3, 7, 3))
        self.assertEqual(v[boolIndices], torch.stack([v[0], v[2], v[3]]))

        v = torch.tensor([True, False, True], dtype=torch.bool, device=device)
        boolIndices = torch.tensor(
            [True, False, False], dtype=torch.bool, device=device
        )
        uint8Indices = torch.tensor([1, 0, 0], dtype=torch.uint8, device=device)
        with warnings.catch_warnings(record=True) as w:
            v1 = v[boolIndices]
            v2 = v[uint8Indices]
            self.assertEqual(v1.shape, v2.shape)
            self.assertEqual(v1, v2)
            self.assertEqual(
                v[boolIndices], tensor([True], dtype=torch.bool, device=device)
            )
            self.assertEqual(len(w), 1)

    def test_list_indices(self, device):
        N = 1000
        t = torch.randn(N, device=device)
        # Set window size
        W = 10
        # Generate a list of lists, containing overlapping window indices
        indices = [range(i, i + W) for i in range(N - W)]

        for i in [len(indices), 100, 32]:
            windowed_data = t[indices[:i]]
            self.assertEqual(windowed_data.shape, (i, W))

        with self.assertRaisesRegex(IndexError, "too many indices"):
            windowed_data = t[indices[:31]]

    def test_bool_indices_accumulate(self, device):
        mask = torch.zeros(size=(10,), dtype=torch.bool, device=device)
        y = torch.ones(size=(10, 10), device=device)
        y.index_put_((mask,), y[mask], accumulate=True)
        self.assertEqual(y, torch.ones(size=(10, 10), device=device))

    def test_multiple_bool_indices(self, device):
        v = torch.randn(5, 7, 3, device=device)
        # note: these broadcast together and are transposed to the first dim
        mask1 = torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool, device=device)
        mask2 = torch.tensor([1, 1, 1], dtype=torch.bool, device=device)
        self.assertEqual(v[mask1, :, mask2].shape, (3, 7))

    def test_multi_dimensional_bool_mask(self, device):
        x = torch.randn(2, 2, 3, device=device)
        b = ((True, False), (False, False))
        m = torch.tensor(b, dtype=torch.bool, device=device)
        z = torch.tensor(0)
        t = torch.tensor(True)
        f = torch.tensor(False)

        # Using boolean sequence
        self.assertEqual(x[b,].shape, (1, 3))
        self.assertEqual(x[b, ::2].shape, (1, 2))
        self.assertEqual(x[b, None].shape, (1, 1, 3))
        self.assertEqual(x[b, 0].shape, (1,))
        self.assertEqual(x[b, z].shape, (1,))
        self.assertEqual(x[b, True].shape, (1, 3))
        self.assertEqual(x[b, True, True, True, True].shape, (1, 3))
        self.assertEqual(x[b, False].shape, (0, 3))
        self.assertEqual(x[b, True, True, False, True].shape, (0, 3))
        self.assertEqual(x[b, t].shape, (1, 3))
        self.assertEqual(x[b, f].shape, (0, 3))

        # Using boolean tensor
        self.assertEqual(x[m].shape, (1, 3))
        self.assertEqual(x[m, ::2].shape, (1, 2))
        self.assertEqual(x[m, None].shape, (1, 1, 3))
        self.assertEqual(x[m, 0].shape, (1,))
        self.assertEqual(x[m, z].shape, (1,))
        self.assertEqual(x[m, True].shape, (1, 3))
        self.assertEqual(x[m, True, True, True, True].shape, (1, 3))
        self.assertEqual(x[m, False].shape, (0, 3))
        self.assertEqual(x[m, True, True, False, True].shape, (0, 3))
        self.assertEqual(x[m, t].shape, (1, 3))
        self.assertEqual(x[m, f].shape, (0, 3))

        # Boolean mask in the middle of indices array
        x = torch.randn(3, 2, 2, 5, device=device)
        self.assertEqual(x[:, m, :].shape, (3, 1, 5))
        self.assertEqual(x[0, m, ::2].shape, (1, 3))
        self.assertEqual(x[..., m, ::2].shape, (3, 1, 3))
        self.assertEqual(x[None, ..., m, ::2].shape, (1, 3, 1, 3))

    def test_bool_mask_assignment(self, device):
        v = torch.tensor([[1, 2], [3, 4]], device=device)
        mask = torch.tensor([1, 0], dtype=torch.bool, device=device)
        v[mask, :] = 0
        self.assertEqual(v, torch.tensor([[0, 0], [3, 4]], device=device))

        v = torch.tensor([[1, 2], [3, 4]], device=device)
        v[:, mask] = 0
        self.assertEqual(v, torch.tensor([[0, 2], [0, 4]], device=device))

    def test_multi_dimensional_bool_mask_assignment(self, device):
        v = torch.tensor([[[[1], [2]], [[3], [4]]]], device=device)
        mask = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool, device=device)
        v[:, mask, :] = 0
        self.assertEqual(v, torch.tensor([[[[0], [2]], [[3], [0]]]], device=device))
        v = torch.tensor([[[[1], [2]], [[3], [4]]]], device=device)
        torch.ops.aten.index_put_(v, [None, mask, None], torch.tensor(0))
        self.assertEqual(v, torch.tensor([[[[0], [2]], [[3], [0]]]], device=device))

    def test_byte_mask(self, device):
        v = torch.randn(5, 7, 3, device=device)
        mask = torch.ByteTensor([1, 0, 1, 1, 0]).to(device)
        with warnings.catch_warnings(record=True) as w:
            res = v[mask]
            self.assertEqual(res.shape, (3, 7, 3))
            self.assertEqual(res, torch.stack([v[0], v[2], v[3]]))
            self.assertEqual(len(w), 1)

        v = torch.tensor([1.0], device=device)
        self.assertEqual(v[v == 0], torch.tensor([], device=device))

    def test_byte_mask_accumulate(self, device):
        mask = torch.zeros(size=(10,), dtype=torch.uint8, device=device)
        y = torch.ones(size=(10, 10), device=device)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y.index_put_((mask,), y[mask], accumulate=True)
            self.assertEqual(y, torch.ones(size=(10, 10), device=device))
            self.assertEqual(len(w), 2)

    # MPS: Fails locally, but passes in CI...
    @skipIfTorchDynamo(
        "This test causes SIGKILL when running with dynamo, https://github.com/pytorch/pytorch/issues/88472"
    )
    @serialTest(TEST_CUDA or TEST_XPU or TEST_MPS)
    def test_index_put_accumulate_large_tensor(self, device):
        # This test is for tensors with number of elements >= INT_MAX (2^31 - 1).
        N = (1 << 31) + 5
        dt = torch.int8
        a = torch.ones(N, dtype=dt, device=device)
        indices = torch.tensor(
            [-2, 0, -2, -1, 0, -1, 1], device=device, dtype=torch.long
        )
        values = torch.tensor([6, 5, 6, 6, 5, 7, 11], dtype=dt, device=device)

        a.index_put_((indices,), values, accumulate=True)

        self.assertEqual(a[0], 11)
        self.assertEqual(a[1], 12)
        self.assertEqual(a[2], 1)
        self.assertEqual(a[-3], 1)
        self.assertEqual(a[-2], 13)
        self.assertEqual(a[-1], 14)

        a = torch.ones((2, N), dtype=dt, device=device)
        indices0 = torch.tensor([0, -1, 0, 1], device=device, dtype=torch.long)
        indices1 = torch.tensor([-2, -1, 0, 1], device=device, dtype=torch.long)
        values = torch.tensor([12, 13, 10, 11], dtype=dt, device=device)

        a.index_put_((indices0, indices1), values, accumulate=True)

        self.assertEqual(a[0, 0], 11)
        self.assertEqual(a[0, 1], 1)
        self.assertEqual(a[1, 0], 1)
        self.assertEqual(a[1, 1], 12)
        self.assertEqual(a[:, 2], torch.ones(2, dtype=torch.int8))
        self.assertEqual(a[:, -3], torch.ones(2, dtype=torch.int8))
        self.assertEqual(a[0, -2], 13)
        self.assertEqual(a[1, -2], 1)
        self.assertEqual(a[-1, -1], 14)
        self.assertEqual(a[0, -1], 1)

    @onlyNativeDeviceTypes
    def test_index_put_accumulate_expanded_values(self, device):
        # checks the issue with cuda: https://github.com/pytorch/pytorch/issues/39227
        # and verifies consistency with CPU result
        t = torch.zeros((5, 2))
        t_dev = t.to(device)
        indices = [torch.tensor([0, 1, 2, 3]), torch.tensor([1])]
        indices_dev = [i.to(device) for i in indices]
        values0d = torch.tensor(1.0)
        values1d = torch.tensor([1.0])

        out_cuda = t_dev.index_put_(indices_dev, values0d.to(device), accumulate=True)
        out_cpu = t.index_put_(indices, values0d, accumulate=True)
        self.assertEqual(out_cuda.cpu(), out_cpu)

        out_cuda = t_dev.index_put_(indices_dev, values1d.to(device), accumulate=True)
        out_cpu = t.index_put_(indices, values1d, accumulate=True)
        self.assertEqual(out_cuda.cpu(), out_cpu)

        t = torch.zeros(4, 3, 2)
        t_dev = t.to(device)

        indices = [
            torch.tensor([0]),
            torch.arange(3)[:, None],
            torch.arange(2)[None, :],
        ]
        indices_dev = [i.to(device) for i in indices]
        values1d = torch.tensor([-1.0, -2.0])
        values2d = torch.tensor([[-1.0, -2.0]])

        out_cuda = t_dev.index_put_(indices_dev, values1d.to(device), accumulate=True)
        out_cpu = t.index_put_(indices, values1d, accumulate=True)
        self.assertEqual(out_cuda.cpu(), out_cpu)

        out_cuda = t_dev.index_put_(indices_dev, values2d.to(device), accumulate=True)
        out_cpu = t.index_put_(indices, values2d, accumulate=True)
        self.assertEqual(out_cuda.cpu(), out_cpu)

    @onlyOn(["cuda", "xpu"])
    def test_index_put_large_indices(self, device):
        def generate_indices(num_indices: int, index_range: int):
            indices = []
            for _ in range(num_indices):
                x = random.randint(0, index_range - 1)
                indices.append(x)
            return torch.tensor(indices)

        num_indices = 401988
        max_index_range = 2000
        target_index_range = [16, 256, 2000]
        # BFloat16
        for generated_index_range in target_index_range:
            # create CPU tensors
            a_tensor_size = (max_index_range, 256)
            a = torch.randn(a_tensor_size, dtype=torch.bfloat16)
            b = generate_indices(
                num_indices=num_indices, index_range=generated_index_range
            )
            c_tensor_size = (num_indices, 256)
            c = torch.randn(c_tensor_size, dtype=torch.bfloat16)
            # create GPU copies
            a_dev = a.to(device)
            b_dev = b.to(device)
            c_dev = c.to(device)
            # run
            a.index_put_(indices=[b], values=c, accumulate=True)
            a_dev.index_put_(indices=[b_dev], values=c_dev, accumulate=True)
            self.assertEqual(a_dev.cpu(), a)

        # Float32
        for generated_index_range in target_index_range:
            # create CPU tensors
            a_tensor_size = (max_index_range, 256)
            a = torch.randn(a_tensor_size, dtype=torch.float32)
            b = generate_indices(
                num_indices=num_indices, index_range=generated_index_range
            )
            c_tensor_size = (num_indices, 256)
            c = torch.randn(c_tensor_size, dtype=torch.float32)
            # create GPU copies
            a_dev = a.to(device)
            b_dev = b.to(device)
            c_dev = c.to(device)
            # run
            torch.use_deterministic_algorithms(True)
            a.index_put_(indices=[b], values=c, accumulate=True)
            torch.use_deterministic_algorithms(False)
            a_dev.index_put_(indices=[b_dev], values=c_dev, accumulate=True)
            self.assertEqual(a_dev.cpu(), a)

    @onlyOn(["cuda", "xpu"])
    def test_index_put_accumulate_non_contiguous(self, device):
        t = torch.zeros((5, 2, 2))
        t_dev = t.to(device)
        t1 = t_dev[:, 0, :]
        t2 = t[:, 0, :]
        self.assertTrue(not t1.is_contiguous())
        self.assertTrue(not t2.is_contiguous())

        indices = [torch.tensor([0, 1])]
        indices_dev = [i.to(device) for i in indices]
        value = torch.randn(2, 2)
        out_cuda = t1.index_put_(indices_dev, value.to(device), accumulate=True)
        out_cpu = t2.index_put_(indices, value, accumulate=True)
        self.assertTrue(not t1.is_contiguous())
        self.assertTrue(not t2.is_contiguous())

        self.assertEqual(out_cuda.cpu(), out_cpu)

    @onlyOn(["cuda", "xpu"])
    def test_index_put_deterministic_with_optional_tensors(self, device):
        def func(x, i, v):
            with DeterministicGuard(True):
                x[..., i] = v
            return x

        def func1(x, i, v):
            with DeterministicGuard(True):
                x[i] = v
            return x

        n = 4
        t = torch.arange(n * 2, dtype=torch.float32).reshape(n, 2)
        t_dev = t.to(device)
        indices = torch.tensor([1, 0])
        indices_dev = indices.to(device)
        value0d = torch.tensor(10.0)
        value1d = torch.tensor([1.0, 2.0])
        values2d = torch.randn(n, 1)

        for val in (value0d, value1d, values2d):
            out_cuda = func(t_dev, indices_dev, val.to(device))
            out_cpu = func(t, indices, val)
            self.assertEqual(out_cuda.cpu(), out_cpu)

        t = torch.zeros((5, 4))
        t_dev = t.to(device)
        indices = torch.tensor([1, 4, 3])
        indices_dev = indices.to(device)
        val = torch.randn(4)
        out_cuda = func1(t_dev, indices_dev, val.to(device))
        out_cpu = func1(t, indices, val)
        self.assertEqual(out_cuda.cpu(), out_cpu)

        t = torch.zeros(2, 3, 4)
        ind = torch.tensor([0, 1])
        val = torch.randn(6, 2)
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            func(t, ind, val)

        with self.assertRaisesRegex(RuntimeError, "must match"):
            func(t.to(device), ind.to(device), val.to(device))

        val = torch.randn(2, 3, 1)
        out_cuda = func1(t.to(device), ind.to(device), val.to(device))
        out_cpu = func1(t, ind, val)
        self.assertEqual(out_cuda.cpu(), out_cpu)

    @onlyNativeDeviceTypes
    def test_index_put_accumulate_duplicate_indices(self, device):
        dtype = torch.float if device.startswith("mps") else torch.double
        for i in range(1, 512):
            # generate indices by random walk, this will create indices with
            # lots of duplicates interleaved with each other
            delta = torch.empty(i, dtype=dtype, device=device).uniform_(-1, 1)
            indices = delta.cumsum(0).long()

            input = torch.randn(indices.abs().max() + 1, device=device)
            values = torch.randn(indices.size(0), device=device)
            output = input.index_put((indices,), values, accumulate=True)

            input_list = input.tolist()
            indices_list = indices.tolist()
            values_list = values.tolist()
            for i, v in zip(indices_list, values_list):
                input_list[i] += v

            self.assertEqual(output, input_list)

    @onlyNativeDeviceTypes
    def test_index_ind_dtype(self, device):
        x = torch.randn(4, 4, device=device)
        ind_long = torch.randint(4, (4,), dtype=torch.long, device=device)
        ind_int = ind_long.int()
        src = torch.randn(4, device=device)
        ref = x[ind_long, ind_long]
        res = x[ind_int, ind_int]
        self.assertEqual(ref, res)
        ref = x[ind_long, :]
        res = x[ind_int, :]
        self.assertEqual(ref, res)
        ref = x[:, ind_long]
        res = x[:, ind_int]
        self.assertEqual(ref, res)
        # no repeating indices for index_put
        ind_long = torch.arange(4, dtype=torch.long, device=device)
        ind_int = ind_long.int()
        for accum in (True, False):
            inp_ref = x.clone()
            inp_res = x.clone()
            torch.index_put_
```



## High-Level Overview


This Python file contains 2 class(es) and 116 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestIndexing`, `NumpyTests`

**Functions defined**: `test_index`, `consec`, `delitem`, `test_advancedindex`, `consec`, `ri`, `validate_indexing`, `validate_setting`, `tensor_indices_to_np`, `get_numpy`, `set_numpy`, `assert_get_eq`, `assert_set_eq`, `assert_backward_eq`, `get_set_tensor`, `test_advancedindex_big`, `test_set_item_to_scalar_tensor`, `test_single_int`, `test_multiple_int`, `test_none`

**Key imports**: operator, random, unittest, warnings, reduce, product, numpy as np, torch, tensor, make_tensor


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `operator`
- `random`
- `unittest`
- `warnings`
- `functools`: reduce
- `itertools`: product
- `numpy as np`
- `torch`
- `torch.testing`: make_tensor


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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
python test/test_indexing.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_mkldnn.py_docs.md`](./test_mkldnn.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `test_indexing.py_docs.md`
- **Keyword Index**: `test_indexing.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
