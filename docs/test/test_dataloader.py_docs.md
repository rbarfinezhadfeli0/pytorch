# Documentation: `test/test_dataloader.py`

## File Metadata

- **Path**: `test/test_dataloader.py`
- **Size**: 133,493 bytes (130.36 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: dataloader"]
# ruff: noqa: F841

import ctypes
import errno
import faulthandler
import functools
import gc
import itertools
import math
import operator
import os
import signal
import sys
import tempfile
import time
import unittest
import warnings

import torch
import torch.utils.data.datapipes as dp
from torch import multiprocessing as mp
from torch._utils import ExceptionWrapper
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    IS_CI,
    IS_JETSON,
    IS_MACOS,
    IS_S390X,
    IS_SANDCASTLE,
    IS_WINDOWS,
    load_tests,
    parametrize,
    run_tests,
    skipIfNoDill,
    skipIfXpu,
    slowTest,
    TEST_CUDA,
    TEST_NUMPY,
    TEST_WITH_ASAN,
    TEST_WITH_TSAN,
    TestCase,
    xfailIfLinux,
)
from torch.utils.data import (
    _utils,
    ChainDataset,
    ConcatDataset,
    DataLoader,
    dataloader,
    Dataset,
    IterableDataset,
    IterDataPipe,
    StackDataset,
    Subset,
    TensorDataset,
)
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.utils.data.dataset import random_split


try:
    import psutil

    HAS_PSUTIL = True
except ModuleNotFoundError:
    HAS_PSUTIL = False
    psutil = None
    err_msg = (
        "psutil not found. Some critical data loader tests relying on it "
        "(e.g., TestDataLoader.test_proper_exit) will not run."
    )
    if IS_CI:
        raise ModuleNotFoundError(err_msg) from None
    else:
        warnings.warn(err_msg)


try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False
    np = None
skipIfNoNumpy = unittest.skipIf(not HAS_NUMPY, "no NumPy")

# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

TEST_CUDA_IPC = (
    torch.cuda.is_available()
    and sys.platform != "darwin"
    and sys.platform != "win32"
    and not IS_JETSON
    #    and not TEST_WITH_ROCM
)  # https://github.com/pytorch/pytorch/issues/90940

TEST_MULTIGPU = TEST_CUDA_IPC and torch.cuda.device_count() > 1

# We want to use `spawn` if able because some of our tests check that the
# data loader terminates gracefully. To prevent hanging in the testing
# process, such data loaders are run in a separate subprocess.
#
# We also want to test the `pin_memory=True` configuration, thus `spawn` is
# required to launch such processes and they initialize the CUDA context.
#
# Mixing different start method is a recipe for disaster (e.g., using a fork
# `mp.Event` with a spawn `mp.Process` segfaults). So we set this globally
# to avoid bugs.
#
# Get a multiprocessing context because some test / third party library will
# set start_method when imported, and setting again triggers `RuntimeError`.
mp = mp.get_context(method="spawn")


# 60s of timeout?
# Yes, in environments where physical CPU resources are shared, e.g., CI, the
# time for a inter-process communication can be highly varying.  With 15~17s of
# timeout, we have observed flakiness in some CI builds (see
# pytorch/pytorch#14501, pytorch/pytorch#16608).  We follow the CPython
# multiprocessing setup and set the timeout to 60s here:
#
# https://github.com/python/cpython/blob/e8113f51a8bdf33188ee30a1c038a298329e7bfa/Lib/test/_test_multiprocessing.py#L73
JOIN_TIMEOUT = 60.0  # seconds


supported_multiprocessing_contexts = [None] + list(
    torch.multiprocessing.get_all_start_methods()
)


# The following collate functions are defined globally here for pickle purposes.


# collate_fn that returns the batch cloned
def _clone_collate(b):
    return [x.clone() for x in b]


# collate_fn that returns the batch of sparse coo tensors cloned
def _sparse_coo_collate(b):
    lst = []
    for x in b:
        t = x.clone()
        lst.append(t)
        # Force sparse tensor invariants checks. check_pinning=True
        # reproduces gh-153143.
        torch._validate_sparse_coo_tensor_args(
            t._indices(), t._values(), t.size(), t.is_coalesced(), check_pinning=False
        )
    return lst


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestDatasetRandomSplit(TestCase):
    def test_lengths_must_equal_dataset_size(self):
        with self.assertRaises(ValueError):
            random_split([1, 2, 3, 4], [1, 2])

    def test_splits_have_correct_size(self):
        splits = random_split([1, 2, 3, 4, 5, 6], [2, 4])
        self.assertEqual(len(splits), 2)
        self.assertEqual(len(splits[0]), 2)
        self.assertEqual(len(splits[1]), 4)

        splits = random_split([1, 2, 3, 4, 5, 6], [0.5, 0.5])
        self.assertEqual(len(splits), 2)
        self.assertEqual(len(splits[0]), 3)
        self.assertEqual(len(splits[1]), 3)

        # Odd size splits
        self.assertEqual(
            len(
                random_split(
                    range(3), [0.5, 0.5], generator=torch.Generator().manual_seed(1)
                )
            ),
            2,
        )

        # Odd sized round-robin splits
        splits = random_split(
            range(106), [0.1, 0.2, 0.3, 0.4], generator=torch.Generator().manual_seed(1)
        )
        self.assertEqual(len(splits[0]), 11)
        self.assertEqual(len(splits[1]), 22)
        self.assertEqual(len(splits[2]), 31)
        self.assertEqual(len(splits[3]), 42)

    def test_splits_are_mutually_exclusive(self):
        data = [5, 2, 3, 4, 1, 6]
        splits = random_split(data, [2, 4])
        all_values = []
        all_values.extend(list(splits[0]))
        all_values.extend(list(splits[1]))
        data.sort()
        all_values.sort()
        self.assertListEqual(data, all_values)

        splits = random_split(data, [0.33, 0.67])
        all_values = []
        all_values.extend(list(splits[0]))
        all_values.extend(list(splits[1]))
        data.sort()
        all_values.sort()
        self.assertListEqual(data, all_values)

        data = [1, 2, 3, 4]
        splits = random_split(data, [0.25, 0.75])
        all_values = []
        all_values.extend(list(splits[0]))
        all_values.extend(list(splits[1]))
        data.sort()
        all_values.sort()
        self.assertListEqual(data, all_values)

    def test_splits_indexing_type(self):
        r"""Indices generated by random_split
        should be of integer type
        """

        class CustomDataset:
            def __init__(self, test_object, custom_list):
                self.data = custom_list
                self.test_object = test_object

            def __getitem__(self, key):
                self.test_object.assertEqual(type(key), int)
                return self.data[key]

            def __len__(self):
                return len(self.data)

        x = [1, 2, 3, 4, 5]
        dataset = CustomDataset(self, x)
        dataset = random_split(dataset, [5])[0]
        data_loader = DataLoader(dataset)
        for _batch in data_loader:
            pass

        # fractional splitting
        dataset = CustomDataset(self, x)
        dataset = random_split(dataset, [1.0])[0]
        data_loader = DataLoader(dataset)
        for _batch in data_loader:
            pass

    def test_splits_reproducibility(self):
        self.assertEqual(
            [
                list(x)
                for x in random_split(
                    range(10), [3, 7], generator=torch.Generator().manual_seed(1)
                )
            ],
            [[5, 6, 1], [2, 0, 8, 9, 3, 7, 4]],
        )
        self.assertEqual(
            random_split(
                range(100), [60, 40], generator=torch.Generator().manual_seed(42)
            ),
            random_split(
                range(100), [60, 40], generator=torch.Generator().manual_seed(42)
            ),
        )
        self.assertEqual(
            random_split(
                range(100), [0.5, 0.5], generator=torch.Generator().manual_seed(42)
            ),
            random_split(
                range(100), [0.5, 0.5], generator=torch.Generator().manual_seed(42)
            ),
        )
        self.assertEqual(
            random_split(
                range(100),
                [0.33, 0.33, 0.34],
                generator=torch.Generator().manual_seed(42),
            ),
            random_split(
                range(100),
                [0.33, 0.33, 0.34],
                generator=torch.Generator().manual_seed(42),
            ),
        )

    def test_incomplete_fractional_splits(self):
        with self.assertRaises(ValueError):
            # should raise since the sum of fractions is not 1
            random_split([1, 2, 3, 4], [0.1])

        with self.assertRaises(ValueError):
            # should raise since fraction > 1
            random_split([1, 2, 3, 4], [1.1])

    def test_splits_generator(self):
        # A random_split without a specific generator should affect the default one
        state = torch.get_rng_state()
        a = torch.rand(10)
        torch.set_rng_state(state)
        random_split(range(10), [5, 5])
        b = torch.rand(10)
        self.assertNotEqual(a, b)

        # A random_split with a specific generator should not affect the default one
        state = torch.get_rng_state()
        a = torch.rand(10)
        torch.set_rng_state(state)
        random_split(range(10), [5, 5], generator=torch.Generator().manual_seed(42))
        b = torch.rand(10)
        self.assertEqual(a, b)

    def test_slicing_of_subset_of_dataset(self):
        # Testing slicing a subset initialized with a dataset
        dataset = TensorDataset(torch.tensor([1, 2, 3, 4, 5]))
        subset_of_dataset = Subset(dataset, [0, 1, 2, 3, 4])
        self.assertEqual(subset_of_dataset[:], dataset[:])
        self.assertEqual(subset_of_dataset[1:2], dataset[1:2])
        self.assertEqual(subset_of_dataset[0:-1:2], dataset[0:-1:2])
        # Testing slicing of subset from random split
        subset1, subset2 = random_split(dataset, [3, 2])
        self.assertEqual(subset1[:], dataset[subset1.indices[:]])
        self.assertEqual(subset1[0:2], dataset[subset1.indices[0:2]])
        self.assertEqual(subset1[0:-1:2], dataset[subset1.indices[0:-1:2]])

    def test_slicing_of_subset_of_subset(self):
        # Testing slicing a subset initialized with a subset
        dataset = TensorDataset(torch.tensor([1, 2, 3, 4, 5]))
        subset_of_dataset = Subset(dataset, [0, 1, 2, 3, 4])
        subset_of_subset = Subset(subset_of_dataset, [0, 1, 2, 3, 4])
        self.assertEqual(subset_of_subset[:], dataset[:])
        self.assertEqual(subset_of_subset[0:2], dataset[0:2])
        self.assertEqual(subset_of_subset[0:-1:2], dataset[0:-1:2])
        # Testing slicing of subset of subset from random split
        subset1, subset2 = random_split(dataset, [4, 1])
        subset_of_subset1, subset_of_subset2 = random_split(subset1, [3, 1])
        idx = [subset1.indices[i] for i in subset_of_subset1.indices]
        self.assertEqual(subset_of_subset1[:], dataset[idx.copy()])
        self.assertEqual(subset_of_subset1[0:2], dataset[idx[0:2]])
        self.assertEqual(subset_of_subset1[0:-1:2], dataset[idx[0:-1:2]])


class CUDACountingDataset(Dataset):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __getitem__(self, i):
        return torch.as_tensor(i, device="cuda")

    def __len__(self):
        return self.n


class CountingDataset(Dataset):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __getitem__(self, i):
        return i

    def __len__(self):
        return self.n


class CountingIterableDataset(IterableDataset):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __iter__(self):
        return iter(range(self.n))

    def __len__(self):
        return self.n


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestTensorDataset(TestCase):
    def test_len(self):
        source = TensorDataset(torch.randn(15, 10, 2, 3, 4, 5), torch.randperm(15))
        self.assertEqual(len(source), 15)

    def test_getitem(self):
        t = torch.randn(15, 10, 2, 3, 4, 5)
        l = torch.randn(15, 10)
        source = TensorDataset(t, l)
        for i in range(15):
            self.assertEqual(t[i], source[i][0])
            self.assertEqual(l[i], source[i][1])

    def test_getitem_1d(self):
        t = torch.randn(15)
        l = torch.randn(15)
        source = TensorDataset(t, l)
        for i in range(15):
            self.assertEqual(t[i], source[i][0])
            self.assertEqual(l[i], source[i][1])

    def test_single_tensor(self):
        t = torch.randn(5, 10)
        source = TensorDataset(t)
        self.assertEqual(len(source), 5)
        for i in range(5):
            self.assertEqual(t[i], source[i][0])

    def test_many_tensors(self):
        t0 = torch.randn(5, 10, 2, 3, 4, 5)
        t1 = torch.randn(5, 10)
        t2 = torch.randn(5, 10, 2, 5)
        t3 = torch.randn(5, 10, 3, 7)
        source = TensorDataset(t0, t1, t2, t3)
        self.assertEqual(len(source), 5)
        for i in range(5):
            self.assertEqual(t0[i], source[i][0])
            self.assertEqual(t1[i], source[i][1])
            self.assertEqual(t2[i], source[i][2])
            self.assertEqual(t3[i], source[i][3])


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestStackDataset(TestCase):
    def test_empty(self):
        with self.assertRaisesRegex(
            ValueError, "At least one dataset should be passed"
        ):
            StackDataset()

    def test_mixed(self):
        with self.assertRaisesRegex(ValueError, "Supported either"):
            StackDataset(
                TensorDataset(torch.randn(15, 10)), a=TensorDataset(torch.randn(10, 15))
            )

    def test_size_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Size mismatch between datasets"):
            StackDataset(
                TensorDataset(torch.randn(15, 10)), TensorDataset(torch.randn(10, 15))
            )
        with self.assertRaisesRegex(ValueError, "Size mismatch between datasets"):
            StackDataset(
                a=TensorDataset(torch.randn(15, 10)),
                b=TensorDataset(torch.randn(10, 15)),
            )

    def test_len(self):
        source = StackDataset(
            TensorDataset(torch.randn(15, 10)), TensorDataset(torch.randn(15))
        )
        self.assertEqual(len(source), 15)
        source = StackDataset(TensorDataset(torch.randn(15, 10)))
        self.assertEqual(len(source), 15)
        source = StackDataset(
            a=TensorDataset(torch.randn(15, 10)), b=TensorDataset(torch.randn(15))
        )
        self.assertEqual(len(source), 15)
        source = StackDataset(a=TensorDataset(torch.randn(15, 10)))
        self.assertEqual(len(source), 15)

    def test_single(self):
        t = TensorDataset(torch.randn(15, 10))
        source = StackDataset(t)
        for i in range(15):
            self.assertEqual(t[i], source[i][0])
        source = StackDataset(a=t)
        for i in range(15):
            self.assertEqual(t[i], source[i]["a"])

    def test_getitem(self):
        t = TensorDataset(torch.randn(15, 10))
        l = TensorDataset(torch.randn(15, 5, 4))
        source = StackDataset(t, l)
        for i in range(15):
            self.assertEqual(t[i], source[i][0])
            self.assertEqual(l[i], source[i][1])
        source = StackDataset(a=t, b=l)
        for i in range(15):
            self.assertEqual(t[i], source[i]["a"])
            self.assertEqual(l[i], source[i]["b"])

    def test_getitems(self):
        class GetItemsDataset(Dataset):
            def __init__(self) -> None:
                self.data = torch.randn(4)

            def __getitem__(self, item):
                return self.data[item]

            def __getitems__(self, items):
                return self.data[items]

            def __len__(self):
                return 4

        t = GetItemsDataset()
        l = [1, 2, 3, 4]

        source = StackDataset(t, l)
        batch = source.__getitems__([0, 1, 2, 3])
        for i in range(4):
            self.assertEqual(t[i], batch[i][0])
            self.assertEqual(l[i], batch[i][1])

        source = StackDataset(t=t, l=l)
        batch = source.__getitems__([0, 1, 2, 3])
        for i in range(4):
            self.assertEqual(t[i], batch[i]["t"])
            self.assertEqual(l[i], batch[i]["l"])

    def test_getitems_raises_index_error(self):
        class GetItemsDataset(Dataset):
            def __init__(self) -> None:
                self.data = torch.randn(4)

            def __getitem__(self, item):
                return self.data[item]

            def __getitems__(self, items):
                return self.data[items]

            def __len__(self):
                return 4

        t = GetItemsDataset()
        l = [1, 2, 3, 4]

        source = StackDataset(t, l)

        with self.assertRaises(IndexError):
            source.__getitems__([0, 4])

    def test_getitems_value_error(self):
        class GetItemsDataset(Dataset):
            def __init__(self) -> None:
                self.data = torch.randn(4)

            def __getitem__(self, item):
                return self.data[item]

            def __getitems__(self, items):
                return self.data[items][:-1]  # return less

            def __len__(self):
                return 4

        t = GetItemsDataset()
        l = [1, 2, 3, 4]

        source = StackDataset(t, l)

        with self.assertRaisesRegex(
            ValueError, "Nested dataset's output size mismatch. Expected 4, got 3"
        ):
            source.__getitems__([0, 1, 2, 3])


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
class TestConcatDataset(TestCase):
    def test_concat_two_singletons(self):
        result = ConcatDataset([[0], [1]])
        self.assertEqual(2, len(result))
        self.assertEqual(0, result[0])
        self.assertEqual(1, result[1])

    def test_concat_two_non_singletons(self):
        result = ConcatDataset([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        self.assertEqual(10, len(result))
        self.assertEqual(0, result[0])
        self.assertEqual(5, result[5])

    def test_concat_two_non_singletons_with_empty(self):
        # Adding an empty dataset somewhere is correctly handled
        result = ConcatDataset([[0, 1, 2, 3, 4], [], [5, 6, 7, 8, 9]])
        self.assertEqual(10, len(result))
        self.assertEqual(0, result[0])
        self.assertEqual(5, result[5])

    def test_concat_raises_index_error(self):
        result = ConcatDataset([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        with self.assertRaises(IndexError):
            # this one goes to 11
            result[11]

    def test_add_dataset(self):
        d1 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        d2 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        d3 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        result = d1 + d2 + d3
        self.assertEqual(21, len(result))
        self.assertEqual(0, (d1[0][0] - result[0][0]).abs().sum())
        self.assertEqual(0, (d2[0][0] - result[7][0]).abs().sum())
        self.assertEqual(0, (d3[0][0] - result[14][0]).abs().sum())

    def test_iterable_dataset_err(self):
        d1 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        it1 = CountingIterableDataset(5)
        it2 = CountingIterableDataset(10)

        with self.assertRaisesRegex(AssertionError, "does not support IterableDataset"):
            ConcatDataset([d1, it2, it1])

        with self.assertRaisesRegex(AssertionError, "does not support IterableDataset"):
            ConcatDataset([it2])

        with self.assertRaisesRegex(AssertionError, "does not support IterableDataset"):
            ConcatDataset([it1, d1])


# takes in dummy var so this can also be used as a `worker_init_fn`
def set_faulthander_if_available(_=None):
    faulthandler.enable(sys.__stderr__)
    if not IS_WINDOWS:
        # windows does not have faulthandler.register
        # chain=False prevents the default behavior of killing the process
        faulthandler.register(signal.SIGUSR1, file=sys.__stderr__, chain=False)


set_faulthander_if_available()


# Process `pid` must have called `set_faulthander_if_available`
def print_traces_of_all_threads(pid):
    if not IS_WINDOWS:
        # use the custom signal if available
        os.kill(pid, signal.SIGUSR1)
    else:
        # otherwise we can still use the handler given by faulthandler.enable()
        # at the cost of killing the process.
        os.kill(pid, signal.SIGSEGV)

    # wait in parent process to give subprocess some time to print
    time.sleep(5)


# The following `ErrorTrackingProcess` stores the first encountered exception in
# its `.exception` attribute.
# Inspired by https://stackoverflow.com/a/33599967
class ErrorTrackingProcess(mp.Process):
    # Why no *args?
    #   py2 doesn't support def fn(x, *args, key=val, **kwargs)
    # Setting disable_stderr=True may generate a lot of unrelated error outputs
    # but could be helpful for debugging.
    def __init__(self, disable_stderr=True, **kwargs):
        super().__init__(**kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None
        self.disable_stderr = disable_stderr

    def run(self):
        set_faulthander_if_available()
        if self.disable_stderr:
            # Disable polluting stderr with errors that are supposed to happen.
            with open(os.devnull, "w") as devnull:
                os.dup2(devnull.fileno(), sys.stderr.fileno())
        try:
            super().run()
            self._cconn.send(None)
        except Exception:
            self._cconn.send(ExceptionWrapper(sys.exc_info()))
            raise

    def print_traces_of_all_threads(self):
        assert self.is_alive(), (
            "can only use print_traces_of_all_threads if the process is alive"
        )
        assert not self.disable_stderr, (
            "do not disable stderr if you use print_traces_of_all_threads"
        )
        # On platforms without `SIGUSR1`, `set_faulthander_if_available` sets
        # `faulthandler.enable()`, and `print_traces_of_all_threads` may kill
        # the process. So let's poll the exception first
        _ = self.exception
        print_traces_of_all_threads(self.pid)

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        if self._exception is None:
            return None
        else:
            return self._exception.exc_type(self._exception.exc_msg)

    # ESRCH means that os.kill can't finds alive proc
    def send_signal(self, signum, ignore_ESRCH=False):
        try:
            os.kill(self.pid, signum)
        except OSError as e:
            if not ignore_ESRCH or e.errno != errno.ESRCH:
                raise


class ErrorDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class SegfaultDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __getitem__(self, idx):
        return ctypes.string_at(0)

    def __len__(self):
        return self.size


class SleepDataset(Dataset):
    def __init__(self, size, sleep_sec):
        self.size = size
        self.sleep_sec = sleep_sec
        self.slept = False

    def __getitem__(self, idx):
        if not self.slept:
            time.sleep(self.sleep_sec)
            self.slept = True
        return idx

    def __len__(self):
        return self.size


class SeedDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __getitem__(self, idx):
        return torch.initial_seed()

    def __len__(self):
        return self.size


class WorkerSpecificIterableDataset(IterableDataset):
    def __init__(self, sizes_for_all_workers):
        self.sizes_for_all_workers = sizes_for_all_workers

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is not None
        return iter(range(self.sizes_for_all_workers[worker_info.id]))

    def __len__(self):
        return sum(self.sizes_for_all_workers)


# Inspired by https://stackoverflow.com/a/26703365
# If all workers will call `sync_once`, they will be blocked until all workers
# reach the call (i.e., acting like a barrier).
# This can be used to ensure that each worker at least processes one data.
class SynchronizedDataset(Dataset):
    def __init__(self, size, batch_size, num_workers):
        assert size >= num_workers * batch_size
        self.count = mp.Value("i", 0, lock=True)
        self.barrier = mp.Semaphore(0)
        self.num_workers = num_workers
        self.size = size

    def sync_once(self):
        with self.count.get_lock():
            self.count.value += 1
            if self.count.value == self.num_workers:
                self.barrier.release()
        self.barrier.acquire()
        self.barrier.release()

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        return self.size


class EmptyTensorDataset(torch.utils.data.Dataset):
    def __init__(self, len):
        self.len = len

    def __len__(self):
        return self.len

    def __getitem__(self, any):
        return torch.empty(0)


class SynchronizedSeedDataset(SynchronizedDataset):
    def __getitem__(self, idx):
        self.sync_once()
        return torch.initial_seed()


def _test_timeout(persistent_workers):
    dataset = SleepDataset(10, 3)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        timeout=1,
        persistent_workers=persistent_workers,
    )
    _ = next(iter(dataloader))


def _test_timeout_pin_memory(persistent_workers):
    dataset = SleepDataset(10, 3)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        timeout=1,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    _ = next(iter(dataloader))


def _test_large_sampler_indices(persistent_workers):
    # See
    #   test_large_sampler_indices
    #   https://github.com/pytorch/pytorch/issues/48666

    dataloader = torch.utils.data.DataLoader(
        EmptyTensorDataset(10000000),
        batch_size=40960,
        persistent_workers=persistent_workers,
        num_workers=1,
    )

    it = iter(dataloader)

    for x in it:
        assert x.numel() == 0
        raise RuntimeError("My Error")


def disable_stderr(worker_id):
    r"""
    Avoids printing "ERROR: Unexpected segmentation fault encountered in worker."
    from workers. Since worker signal handler prints with low-level write(),
    this has to be done on OS level via dup.

    This is used as worker_init_fn for test_segfault.
    """
    sys.stderr.flush()  # flush library buffers that dup2 knows nothing about
    # Can't use a with-block because otherwise the fd will be closed when this
    # function ends.
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), sys.stderr.fileno())


def _test_segfault():
    dataset = SegfaultDataset(10)
    dataloader = DataLoader(
        dataset, batch_size=2, num_workers=2, worker_init_fn=disable_stderr
    )
    _ = next(iter(dataloader))


def _test_no_segfault():
    dataset = [1, 2, 3]
    num_threads = torch.get_num_threads()
    if num_threads < 4:
        torch.set_num_threads(4)
    else:
        torch.set_num_threads(num_threads)
    mp_ctx = torch.multiprocessing.get_context(method="fork")
    dataloader = DataLoader(
        dataset,
        num_workers=1,
        worker_init_fn=disable_stderr,
        multiprocessing_context=mp_ctx,
    )
    _ = next(iter(dataloader))


class TestProperExitDataset(Dataset):
    def __init__(self, size, error_event):
        self.size = size
        self.error_event = error_event

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        if (
            self.error_event is not None
            and self.error_event.is_set()
            and worker_info.id == worker_info.num_workers - 1
        ):
            # only error in the last worker
            raise RuntimeError("Worker error")
        return torch.tensor([idx])


class TestProperExitIterableDataset(IterableDataset):
    def __init__(self, size, error_event):
        self.error_event = error_event
        self.size = size
        self.remaining = size

    def __len__(self):
        return self.size

    def __iter__(self):
        return self

    def __next__(self):
        worker_info = torch.utils.data.get_worker_info()
        if (
            self.error_event is not None
            and self.error_event.is_set()
            and worker_info.id == worker_info.num_workers - 1
        ):
            # only error in the last worker
            raise RuntimeError("Worker error")
        self.remaining -= 1
        if self.remaining < 0:
            raise StopIteration
        return torch.tensor(-1000)


# See TestDataLoader.test_proper_exit for usage
def _test_proper_exit(
    is_iterable_dataset,
    use_workers,
    pin_memory,
    exit_method,
    hold_iter_reference,
    loader_setup_event,
    tester_setup_event,
    persistent_workers,
):
    num_workers = 2 if use_workers else 0

    if exit_method == "worker_error" or exit_method == "worker_kill":
        assert use_workers is True

    if exit_method == "worker_error":
        worker_error_event = mp.Event()
    else:
        worker_error_event = None

    if is_iterable_dataset:
        ds = TestProperExitIterableDataset(7, worker_error_event)
    else:
        ds = TestProperExitDataset(12, worker_error_event)

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=set_faulthander_if_available,
        persistent_workers=persistent_workers,
    )

    error_it = 2

    if use_workers:
        # 2 is the magical per-worker prefetch number...
        # FIXME: change this after the number becomes configurable.
        if is_iterable_dataset:
            assert len(ds) * num_workers > (error_it + 2 + 1)
        else:
            assert len(loader) > (error_it + 2 + 1) * num_workers
    else:
        if is_iterable_dataset:
            assert len(ds) > error_it + 1
        else:
            assert len(loader) > error_it + 1

    it = iter(loader)
    if use_workers:
        workers = it._workers

    def kill_pid(pid):
        psutil_p = psutil.Process(pid)
        psutil_p.kill()
        psutil_p.wait(JOIN_TIMEOUT)
        assert not psutil_p.is_running()

    for i, _ in enumerate(it):
        if i == 0:
            if not hold_iter_reference:
                del it
                del loader
            loader_setup_event.set()
            tester_setup_event.wait()
            # ensure that the workers are still alive
            if use_workers:
                for w in workers:
                    assert w.is_alive()
            if worker_error_event is not None:
                worker_error_event.set()

        if i == error_it:
            if exit_method == "loader_error":
                raise RuntimeError("Loader error")
            elif exit_method == "loader_kill":
                kill_pid(os.getpid())
            elif exit_method == "worker_kill":
                kill_pid(workers[-1].pid)  # kill last worker

    if not hold_iter_reference:
        # Tries to trigger the __del__ clean-up rather than the automatic
        # exiting of daemonic children. Technically it should be automatically
        # triggered, but I don't want to rely on the implementation detail of
        # Python gc.
        gc.collect()


class TestWorkerInfoDataset(SynchronizedDataset):
    def __getitem__(self, idx):
        self.sync_once()
        return torch.tensor(self.value)


# Should be used as worker_init_fn with TestWorkerInfoDataset.
# See _test_get_worker_info below for usage.
def _test_worker_info_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_id == worker_info.id, (
        "worker_init_fn and worker_info should have consistent id"
    )
    assert worker_id < worker_info.num_workers, (
        "worker_init_fn and worker_info should have valid id"
    )
    assert worker_info.seed == torch.initial_seed(), (
        "worker_init_fn and worker_info should have consistent seed"
    )
    dataset = worker_info.dataset
    assert isinstance(dataset, TestWorkerInfoDataset), (
        "worker_info should have correct dataset copy"
    )
    assert not hasattr(dataset, "value"), "worker_info should have correct dataset copy"
    # test that WorkerInfo attributes are read-only
    try:
        worker_info.id = 3999
    except RuntimeError as e:
        assert str(e) == "Cannot assign attributes to WorkerInfo objects"
    try:
        worker_info.a = 3
    except RuntimeError as e:
        assert str(e) == "Cannot assign attributes to WorkerInfo objects"
    for k in ["id", "num_workers", "seed", "dataset"]:
        assert f"{k}=" in repr(worker_info)
    dataset.value = [worker_id, os.getpid()]


def _test_get_worker_info():
    # get_worker_info returns None in main proc
    assert torch.utils.data.get_worker_info() is None
    num_workers = 2
    batch_size = 2
    dataset = TestWorkerInfoDataset(6, batch_size, num_workers)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=_test_worker_info_init_fn,
    )
    it = iter(dataloader)
    data = []
    for d in it:
        data.append(d)  # noqa: PERF402
    worker_pids = [w.pid for w in it._workers]
    data = torch.cat(data, 0)
    for d in data:
        # each `d` is a [worker_id, worker_pid] pair, which is set in
        # _test_worker_info_init_fn
        assert d[1] == worker_pids[d[0]]
    # get_worker_info returns None in main proc after data loading
    assert torch.utils.data.get_worker_info() is None
    # main proc dataset was never assigned this attribute
    assert not hasattr(dataset, "value")
    try:
        _ = dataset[0]
    except AttributeError:
        return
    raise RuntimeError("Expected AttributeError")


# test custom init function
def init_fn(worker_id):
    torch.manual_seed(12345)


# used with test_error_in_init
class ErrorIterableDataset(IterableDataset):
    def __iter__(self):
        raise RuntimeError("Error in __iter__")


# used with test_error_in_init
def error_worker_init_fn(_):
    raise RuntimeError("Error in worker_init_fn")


class BulkLoadingDataset(Dataset):
    def __init__(self, length):
        self.length = length

    def __getitem__(self, indices):
        assert isinstance(indices, (list, tuple))
        return torch.as_tensor(indices)

    def __len__(self):
        return self.length


class BulkLoadingSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        for x in torch.randperm(len(self.dataset)).split(self.batch_size):
            yield x.tolist()

    def __len__(self):
        return int(math.ceil(len(self.dataset) / float(self.batch_size)))


class TestMultiEpochDataset(IterableDataset):
    def __init__(self, length):
        self.length = length

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is not None
        worker_id = worker_info.id
        for _ in range(self.length // worker_info.num_workers):
            yield worker_id

    def __len__(self):
        return self.length


class CustomList(list):
    pass


class CustomDict(dict):
    pass


def row_processor(row):
    return np.add(row, 1)


def filter_len(row):
    return len(row) == 4


@unittest.skipIf(
    TEST_WITH_TSAN,
    "Fails with TSAN with the following error: starting new threads after multi-threaded "
    "fork is not supported. Dying (set die_after_fork=0 to override)",
)
@unittest.skipIf(
    TEST_WITH_ASAN,
    "DataLoader tests hang in ASAN, see: https://github.com/pytorch/pytorch/issues/66223",
)
class TestDataLoader(TestCase):
    def setUp(self):
        super().setUp()
        self.data = torch.randn(100, 2, 3, 5)
        self.labels = torch.randperm(50).repeat(2)
        self.dataset = TensorDataset(self.data, self.labels)
        self.persistent_workers = False

    def _get_data_loader(self, dataset, **kwargs):
        persistent_workers = kwargs.get("persistent_workers", self.persistent_workers)
        if persistent_workers and kwargs.get("num_workers", 0) == 0:
            persistent_workers = False
        kwargs["persistent_workers"] = persistent_workers
        return DataLoader(dataset, **kwargs)

    def _test_sequential(self, loader):
        batch_size = loader.batch_size
        if batch_size is None:
            for idx, (sample, target) in enumerate(loader):
                self.assertEqual(sample, self.data[idx])
                self.assertEqual(target, self.labels[idx])
            self.assertEqual(idx, len(self.dataset) - 1)
        else:
            for i, (sample, target) in enumerate(loader):
                idx = i * batch_size
                self.assertEqual(sample, self.data[idx : idx + batch_size])
                self.assertEqual(target, self.labels[idx : idx + batch_size])
            self.assertEqual(i, math.floor((len(self.dataset) - 1) / batch_size))

    def _test_shuffle(self, loader):
        found_data = dict.fromkeys(range(self.data.size(0)), 0)
        found_labels = dict.fromkeys(range(self.labels.size(0)), 0)
        batch_size = loader.batch_size
        if batch_size is None:
            for i, (batch_samples, batch_targets) in enumerate(loader):
                sample, target = (batch_samples, batch_targets)
                for data_point_idx, data_point in enumerate(self.data):
                    if data_point.eq(sample).all():
                        self.assertFalse(found_data[data_point_idx])
                        found_data[data_point_idx] += 1
                        break
                self.assertEqual(target, self.labels[data_point_idx])
                found_labels[data_point_idx] += 1
                self.assertEqual(sum(found_data.values()), (i + 1))
                self.assertEqual(sum(found_labels.values()), (i + 1))
            self.assertEqual(i, (len(self.dataset) - 1))
        else:
            for i, (batch_samples, batch_targets) in enumerate(loader):
                for sample, target in zip(batch_samples, batch_targets):
                    for data_point_idx, data_point in enumerate(self.data):
                        if data_point.eq(sample).all():
                            self.assertFalse(found_data[data_point_idx])
                            found_data[data_point_idx] += 1
                            break
                    self.assertEqual(target, self.labels[data_point_idx])
                    found_labels[data_point_idx] += 1
                self.assertEqual(sum(found_data.values()), (i + 1) * batch_size)
                self.assertEqual(sum(found_labels.values()), (i + 1) * batch_size)
            self.assertEqual(i, math.floor((len(self.dataset) - 1) / batch_size))

    def _test_error(self, loader):
        it = iter(loader)
        errors = 0
        while True:
            try:
                next(it)
            except NotImplementedError:
                errors += 1
            except StopIteration:
                self.assertEqual(
                    errors, math.ceil(float(len(loader.dataset)) / loader.batch_size)
                )
                return

    def test_error_in_init(self):
        for num_workers in [0, 2]:
            loader = self._get_data_loader(
                ErrorIterableDataset(), num_workers=num_workers
            )
            with self.assertRaisesRegex(RuntimeError, "Error in __iter__"):
                list(iter(loader))

        loader = self._get_data_loader(
            self.dataset, num_workers=2, worker_init_fn=error_worker_init_fn
        )
        with self.assertRaisesRegex(RuntimeError, "Error in worker_init_fn"):
            list(iter(loader))

    def test_typing(self):
        # Make sure there is no TypeError

        class SomeDatasetClass(Dataset[list[torch.Tensor]]):
            pass

        def _create_dataloader(is_train: bool) -> DataLoader[list[torch.Tensor]]:
            pass

    @unittest.skipIf(IS_SANDCASTLE, "subprocess doesn't work in FB internal CI")
    @unittest.skipIf(IS_WINDOWS, "No 'resource' module on Windows")
    def test_fd_limit_exceeded(self):
        # See NOTE [ DataLoader on Linux and open files limit ]
        import subprocess

        subprocess.check_output(
            [
                sys.executable,
                "-c",
                """\
import torch
import resource
from torch.utils.data import DataLoader, IterableDataset

class RandomDataset(IterableDataset):
    def __init__(self, len, size):
        super(RandomDataset).__init__()
        self.len = len
        self.size = size

    def __iter__(self):
        return self

    def __next__(self):
        if self.len <= 0:
            raise StopIteration
        self.len -= 1
        return torch.randn(self.size)

try:
    keep_fds_alive = []
    resource.setrlimit(resource.RLIMIT_NOFILE, (100, 100))
    for random_t in DataLoader(RandomDataset(200, (2,2)), multiprocessing_context="fork",
                               num_workers=1):
      random_t.max(dim=0)
      keep_fds_alive.append(random_t)
except RuntimeError as e:
    assert "ulimit -n" in str(e)
    assert "set_sharing_strategy" in str(e)
""",
            ]
        )

    def test_invalid_assign_after_init(self):
        dl = self._get_data_loader(self.dataset)
        for attr in ("batch_size", "sampler", "batch_sampler", "drop_last", "dataset"):

            def fn():
                setattr(dl, attr, {})

            self.assertRaises(ValueError, fn)

    def test_sequential_nonbatch(self):
        self._test_sequential(self._get_data_loader(self.dataset, batch_size=None))

    def test_sequential_batch(self):
        self._test_sequential(self._get_data_loader(self.dataset))
        self._test_sequential(self._get_data_loader(self.dataset, batch_size=2))

    def test_bulk_loading_nobatch(self):
        n = 35
        bs = 4
        ds = BulkLoadingDataset(n)
        sampler = BulkLoadingSampler(ds, batch_size=4)

        for num_workers in [0, 4]:
            dl = self._get_data_loader(
                ds,
                num_workers=num_workers,
                batch_size=None,
                sampler=sampler,
                pin_memory=TEST_CUDA,
            )
            self.assertFalse(dl._auto_collation)
            samples = list(dl)
            self.assertEqual(samples[0].is_pinned(), TEST_CUDA)
            self.assertEqual(set(torch.cat(samples, 0).tolist()), set(range(n)))

    def test_growing_dataset(self):
        dataset = [torch.ones(4) for _ in range(4)]
        dataloader_seq = self._get_data_loader(dataset, shuffle=False)
        dataloader_shuffle = self._get_data_loader(dataset, shuffle=True)
        dataset.append(torch.ones(4))
        self.assertEqual(len(dataloader_seq), 5)
        self.assertEqual(len(dataloader_shuffle), 5)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_sequential_pin_memory(self):
        loader = self._get_data_loader(self.dataset, batch_size=2, pin_memory=True)
        for input, target in loader:
            self.assertTrue(input.is_pinned())
            self.assertTrue(target.is_pinned())

    @unittest.skipIf(not TEST_CUDA_IPC, "CUDA IPC not available")
    def test_multiple_dataloaders(self):
        for multiprocessing_context in supported_multiprocessing_contexts:
            loader1_it = iter(self._get_data_loader(self.dataset, num_workers=1))
            loader2_it = iter(
                self._get_data_loader(
                    self.dataset,
                    num_workers=2,
                    multiprocessing_context=multiprocessing_context,
                )
            )
            next(loader1_it)
            next(loader1_it)
            next(loader2_it)
            next(loader2_it)
            next(loader1_it)
            next(loader2_it)
            del loader1_it
            del loader2_it

    # This case pass on Intel GPU, but currently expected failure on other device,
    # please don't forget to remove this skip when remove the xfailIfLinux.
    @skipIfXpu
    # This case passes on s390x too.
    # please don't forget to remove this skip when remove the xfailIfLinux.
    @unittest.skipIf(IS_S390X, "Unexpectedly succeeds on s390x")
    # https://github.com/pytorch/pytorch/issues/128551
    @xfailIfLinux
    def test_segfault(self):
        p = ErrorTrackingProcess(target=_test_segfault)
        p.start()
        p.join(JOIN_TIMEOUT)
        try:
            self.assertFalse(p.is_alive())
            self.assertNotEqual(p.exitcode, 0)
            if IS_WINDOWS:
                self.assertIsInstance(p.exception, OSError)
                self.assertRegex(str(p.exception), r"access violation reading ")
            else:
                self.assertIsInstance(p.exception, RuntimeError)
                self.assertRegex(
                    str(p.exception),
                    r"DataLoader worker \(pid \d+\) is killed by signal: ",
                )
        finally:
            p.terminate()

    # Tests if the child process forked by the DataLoader segfaults due to having more than 3 threads
    # in the parent process after at least one set_num_threads invocation in the parent process.
    # After forking, set_num_threads(1) in the child process entails handling some inherited data-structures
    # of the Caffe2 thread-pool of the parent process, culminating in a segfault.
    # Reference: https://github.com/pytorch/pytorch/issues/54752
    @unittest.skipIf(IS_WINDOWS, "Needs fork")
    def test_no_segfault(self):
        p = ErrorTrackingProcess(target=_test_no_segfault)
        p.start()
        p.join(JOIN_TIMEOUT)
        try:
            self.assertFalse(p.is_alive())
            if p.exception:
                self.assertIsInstance(p.exception, RuntimeError)
                self.assertRegex(
                    str(p.exception),
                    r"DataLoader worker \(pid \d+\) is killed by signal: ",
                )
                self.fail("Segfault occurred in worker process after fork")
        finally:
            p.terminate()

    def test_timeout(self):
        if TEST_CUDA:
            # This test runs in a subprocess, which can only initialize CUDA with spawn.
            # _test_timeout_pin_memory with pin_memory=True initializes CUDA when the iterator is
            # constructed.
            targets = (_test_timeout, _test_timeout_pin_memory)
        else:
            targets = (_test_timeout,)
        for target in targets:
            p = ErrorTrackingProcess(target=target, args=(self.persistent_workers,))
            p.start()
            p.join(JOIN_TIMEOUT)
            try:
                self.assertFalse(p.is_alive())
                self.assertNotEqual(p.exitcode, 0)
                self.assertIsInstance(p.exception, RuntimeError)
                self.assertRegex(
                    str(p.exception), r"DataLoader timed out after \d+ seconds"
                )
            finally:
                p.terminate()

    def test_large_sampler_indices(self):
        # Test that the data loader cleanly exit when the process errors
        #   1. having an reference to the iterator
        #   2. using a sampler that yields big elements s.t. _index_queues putters block
        #
        # More context: https://github.com/pytorch/pytorch/issues/48666

        p = ErrorTrackingProcess(
            target=_test_large_sampler_indices, args=(self.persistent_workers,)
        )
        p.start()
        p.join(JOIN_TIMEOUT)
        try:
            self.assertFalse(p.is_alive())
            self.assertNotEqual(p.exitcode, 0)
            self.assertIsInstance(p.exception, RuntimeError)
            self.assertRegex(str(p.exception), r"My Error")
        finally:
            p.terminate()

    def test_invalid_ctor_args_combinations(self):
        # general
        with self.assertRaisesRegex(
            ValueError, "num_workers option should be non-negative"
        ):
            self._get_data_loader(self.dataset, num_workers=-1)
        with self.assertRaisesRegex(
            ValueError, "timeout option should be non-negative"
        ):
            self._get_data_loader(self.dataset, timeout=-1)

        # disable auto-batching
        with self.assertRaisesRegex(
            ValueError,
            "batch_size=None option disables auto-batching and is mutually exclusive",
        ):
            self._get_data_loader(self.dataset, batch_size=None, drop_last=True)

        valid_ctx = list(torch.multiprocessing.get_all_start_methods())[-1]
        with self.assertRaisesRegex(
            ValueError, r"multi-process loading
```



## High-Level Overview


This Python file contains 58 class(es) and 276 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestDatasetRandomSplit`, `CustomDataset`, `CUDACountingDataset`, `CountingDataset`, `CountingIterableDataset`, `TestTensorDataset`, `TestStackDataset`, `GetItemsDataset`, `GetItemsDataset`, `GetItemsDataset`, `TestConcatDataset`, `ErrorTrackingProcess`, `ErrorDataset`, `SegfaultDataset`, `SleepDataset`, `SeedDataset`, `WorkerSpecificIterableDataset`, `SynchronizedDataset`, `EmptyTensorDataset`, `SynchronizedSeedDataset`

**Functions defined**: `_clone_collate`, `_sparse_coo_collate`, `test_lengths_must_equal_dataset_size`, `test_splits_have_correct_size`, `test_splits_are_mutually_exclusive`, `test_splits_indexing_type`, `__init__`, `__getitem__`, `__len__`, `test_splits_reproducibility`, `test_incomplete_fractional_splits`, `test_splits_generator`, `test_slicing_of_subset_of_dataset`, `test_slicing_of_subset_of_subset`, `__init__`, `__getitem__`, `__len__`, `__init__`, `__getitem__`, `__len__`

**Key imports**: ctypes, errno, faulthandler, functools, gc, itertools, math, operator, os, signal


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `ctypes`
- `errno`
- `faulthandler`
- `functools`
- `gc`
- `itertools`
- `math`
- `operator`
- `os`
- `signal`
- `sys`
- `tempfile`
- `time`
- `unittest`
- `warnings`
- `torch`
- `torch.utils.data.datapipes as dp`
- `torch._utils`: ExceptionWrapper
- `torch.testing._internal.common_device_type`: instantiate_device_type_tests
- `torch.utils.data._utils`: MP_STATUS_CHECK_INTERVAL
- `torch.utils.data.datapipes.iter`: IterableWrapper
- `torch.utils.data.dataset`: random_split
- `psutil`
- `numpy as np`
- `subprocess`
- `resource`
- `torch.utils.data`: DataLoader, IterableDataset
- `multiprocessing as py_mp`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Implements or uses **caching** mechanisms.

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
python test/test_dataloader.py
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

- **File Documentation**: `test_dataloader.py_docs.md`
- **Keyword Index**: `test_dataloader.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
