# Documentation: `docs/test/test_datapipe.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_datapipe.py_docs.md`
- **Size**: 54,130 bytes (52.86 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_datapipe.py`

## File Metadata

- **Path**: `test/test_datapipe.py`
- **Size**: 155,653 bytes (152.00 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# mypy: ignore-errors

# Owner(s): ["module: dataloader"]

import copy
import itertools
import importlib.util
import os
import os.path
import pickle
import pydoc
import random
import tempfile
import warnings
from functools import partial
from typing import (
    Any,
    Optional,
    TypeVar,
    Union,
)
from collections.abc import Awaitable, Iterator

import operator
from unittest import skipIf

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data.datapipes as dp
import torch.utils.data.graph
import torch.utils.data.graph_settings
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfNoDill,
    skipIfTorchDynamo,
    suppress_warnings,
    TEST_DILL,
    TestCase,
)
from torch.utils._import_utils import import_dill
from torch.utils.data import (
    argument_validation,
    DataChunk,
    DataLoader,
    IterDataPipe,
    MapDataPipe,
    RandomSampler,
    runtime_validation,
    runtime_validation_disabled,
)
from torch.utils.data.datapipes.dataframe import (
    CaptureDataFrame,
    dataframe_wrapper as df_wrapper,
)
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from torch.utils.data.datapipes.utils.common import StreamWrapper
from torch.utils.data.datapipes.utils.decoder import (
    basichandlers as decoder_basichandlers,
)
from torch.utils.data.datapipes.utils.snapshot import _simple_graph_snapshot_restoration
from torch.utils.data.graph import traverse_dps

dill = import_dill()
HAS_DILL = TEST_DILL

HAS_PANDAS: bool = importlib.util.find_spec("pandas") is not None
skipIfNoDataFrames = skipIf(not HAS_PANDAS, "no dataframes (pandas)")

skipTyping = skipIf(True, "TODO: Fix typing bug")
T_co = TypeVar("T_co", covariant=True)


def create_temp_dir_and_files():
    # The temp dir and files within it will be released and deleted in tearDown().
    # Adding `noqa: P201` to avoid mypy's warning on not releasing the dir handle within this function.
    temp_dir = tempfile.TemporaryDirectory()  # noqa: P201
    temp_dir_path = temp_dir.name
    with tempfile.NamedTemporaryFile(
        dir=temp_dir_path, delete=False, suffix=".txt"
    ) as f:
        temp_file1_name = f.name
    with tempfile.NamedTemporaryFile(
        dir=temp_dir_path, delete=False, suffix=".byte"
    ) as f:
        temp_file2_name = f.name
    with tempfile.NamedTemporaryFile(
        dir=temp_dir_path, delete=False, suffix=".empty"
    ) as f:
        temp_file3_name = f.name

    with open(temp_file1_name, "w") as f1:
        f1.write("0123456789abcdef")
    with open(temp_file2_name, "wb") as f2:
        f2.write(b"0123456789abcdef")

    temp_sub_dir = tempfile.TemporaryDirectory(dir=temp_dir_path)  # noqa: P201
    temp_sub_dir_path = temp_sub_dir.name
    with tempfile.NamedTemporaryFile(
        dir=temp_sub_dir_path, delete=False, suffix=".txt"
    ) as f:
        temp_sub_file1_name = f.name
    with tempfile.NamedTemporaryFile(
        dir=temp_sub_dir_path, delete=False, suffix=".byte"
    ) as f:
        temp_sub_file2_name = f.name

    with open(temp_sub_file1_name, "w") as f1:
        f1.write("0123456789abcdef")
    with open(temp_sub_file2_name, "wb") as f2:
        f2.write(b"0123456789abcdef")

    return [
        (temp_dir, temp_file1_name, temp_file2_name, temp_file3_name),
        (temp_sub_dir, temp_sub_file1_name, temp_sub_file2_name),
    ]


def reset_after_n_next_calls(
    datapipe: Union[IterDataPipe[T_co], MapDataPipe[T_co]], n: int
) -> tuple[list[T_co], list[T_co]]:
    """
    Given a DataPipe and integer n, iterate the DataPipe for n elements and store the elements into a list
    Then, reset the DataPipe and return a tuple of two lists
        1. A list of elements yielded before the reset
        2. A list of all elements of the DataPipe after the reset
    """
    it = iter(datapipe)
    res_before_reset = []
    for _ in range(n):
        res_before_reset.append(next(it))
    return res_before_reset, list(datapipe)


def odd_or_even(x: int) -> int:
    return x % 2


class TestDataChunk(TestCase):
    def setUp(self):
        self.elements = list(range(10))
        random.shuffle(self.elements)
        self.chunk: DataChunk[int] = DataChunk(self.elements)

    def test_getitem(self):
        for i in range(10):
            self.assertEqual(self.elements[i], self.chunk[i])

    def test_iter(self):
        for ele, dc in zip(self.elements, iter(self.chunk)):
            self.assertEqual(ele, dc)

    def test_len(self):
        self.assertEqual(len(self.elements), len(self.chunk))

    def test_as_string(self):
        self.assertEqual(str(self.chunk), str(self.elements))

        batch = [self.elements] * 3
        chunks: list[DataChunk[int]] = [DataChunk(self.elements)] * 3
        self.assertEqual(str(batch), str(chunks))

    def test_sort(self):
        chunk: DataChunk[int] = DataChunk(self.elements)
        chunk.sort()
        self.assertTrue(isinstance(chunk, DataChunk))
        for i, d in enumerate(chunk):
            self.assertEqual(i, d)

    def test_reverse(self):
        chunk: DataChunk[int] = DataChunk(self.elements)
        chunk.reverse()
        self.assertTrue(isinstance(chunk, DataChunk))
        for i in range(10):
            self.assertEqual(chunk[i], self.elements[9 - i])

    def test_random_shuffle(self):
        elements = list(range(10))
        chunk: DataChunk[int] = DataChunk(elements)

        rng = random.Random(0)
        rng.shuffle(chunk)

        rng = random.Random(0)
        rng.shuffle(elements)

        self.assertEqual(chunk, elements)


class TestStreamWrapper(TestCase):
    class _FakeFD:
        def __init__(self, filepath):
            self.filepath = filepath
            self.opened = False
            self.closed = False

        def open(self):
            self.opened = True

        def read(self):
            if self.opened:
                return "".join(self)
            else:
                raise OSError("Cannot read from un-opened file descriptor")

        def __iter__(self):
            for i in range(5):
                yield str(i)

        def close(self):
            if self.opened:
                self.opened = False
                self.closed = True

        def __repr__(self):
            return "FakeFD"

    def test_dir(self):
        fd = TestStreamWrapper._FakeFD("")
        wrap_fd = StreamWrapper(fd)

        s = set(dir(wrap_fd))
        for api in ["open", "read", "close"]:
            self.assertTrue(api in s)

    @skipIfTorchDynamo()
    def test_api(self):
        fd = TestStreamWrapper._FakeFD("")
        wrap_fd = StreamWrapper(fd)

        self.assertFalse(fd.opened)
        self.assertFalse(fd.closed)
        with self.assertRaisesRegex(IOError, "Cannot read from"):
            wrap_fd.read()

        wrap_fd.open()
        self.assertTrue(fd.opened)
        self.assertEqual("01234", wrap_fd.read())

        del wrap_fd
        self.assertFalse(fd.opened)
        self.assertTrue(fd.closed)

    def test_pickle(self):
        with tempfile.TemporaryFile() as f:
            with self.assertRaises(TypeError) as ctx1:
                pickle.dumps(f)

            wrap_f = StreamWrapper(f)
            with self.assertRaises(TypeError) as ctx2:
                pickle.dumps(wrap_f)

            # Same exception when pickle
            self.assertEqual(str(ctx1.exception), str(ctx2.exception))

        fd = TestStreamWrapper._FakeFD("")
        wrap_fd = StreamWrapper(fd)
        _ = pickle.loads(pickle.dumps(wrap_fd))

    def test_repr(self):
        fd = TestStreamWrapper._FakeFD("")
        wrap_fd = StreamWrapper(fd)
        self.assertEqual(str(wrap_fd), "StreamWrapper<FakeFD>")

        with tempfile.TemporaryFile() as f:
            wrap_f = StreamWrapper(f)
            self.assertEqual(str(wrap_f), "StreamWrapper<" + str(f) + ">")


class TestIterableDataPipeBasic(TestCase):
    def setUp(self):
        ret = create_temp_dir_and_files()
        self.temp_dir = ret[0][0]
        self.temp_files = ret[0][1:]
        self.temp_sub_dir = ret[1][0]
        self.temp_sub_files = ret[1][1:]

    def tearDown(self):
        try:
            self.temp_sub_dir.cleanup()
            self.temp_dir.cleanup()
        except Exception as e:
            warnings.warn(
                f"TestIterableDatasetBasic was not able to cleanup temp dir due to {str(e)}"
            )

    def test_listdirfiles_iterable_datapipe(self):
        temp_dir = self.temp_dir.name
        datapipe: IterDataPipe = dp.iter.FileLister(temp_dir, "")

        count = 0
        for pathname in datapipe:
            count = count + 1
            self.assertTrue(pathname in self.temp_files)
        self.assertEqual(count, len(self.temp_files))

        count = 0
        datapipe = dp.iter.FileLister(temp_dir, "", recursive=True)
        for pathname in datapipe:
            count = count + 1
            self.assertTrue(
                (pathname in self.temp_files) or (pathname in self.temp_sub_files)
            )
        self.assertEqual(count, len(self.temp_files) + len(self.temp_sub_files))

        temp_files = self.temp_files
        datapipe = dp.iter.FileLister([temp_dir, *temp_files])
        count = 0
        for pathname in datapipe:
            count += 1
            self.assertTrue(pathname in self.temp_files)
        self.assertEqual(count, 2 * len(self.temp_files))

        # test functional API
        datapipe = datapipe.list_files()
        count = 0
        for pathname in datapipe:
            count += 1
            self.assertTrue(pathname in self.temp_files)
        self.assertEqual(count, 2 * len(self.temp_files))

    def test_listdirfilesdeterministic_iterable_datapipe(self):
        temp_dir = self.temp_dir.name

        datapipe = dp.iter.FileLister(temp_dir, "")
        # The output order should be always the same.
        self.assertEqual(list(datapipe), list(datapipe))

        datapipe = dp.iter.FileLister(temp_dir, "", recursive=True)
        # The output order should be always the same.
        self.assertEqual(list(datapipe), list(datapipe))

    def test_openfilesfromdisk_iterable_datapipe(self):
        # test import datapipe class directly
        from torch.utils.data.datapipes.iter import FileLister, FileOpener

        temp_dir = self.temp_dir.name
        datapipe1 = FileLister(temp_dir, "")
        datapipe2 = FileOpener(datapipe1, mode="b")

        count = 0
        for rec in datapipe2:
            count = count + 1
            self.assertTrue(rec[0] in self.temp_files)
            with open(rec[0], "rb") as f:
                self.assertEqual(rec[1].read(), f.read())
                rec[1].close()
        self.assertEqual(count, len(self.temp_files))

        # functional API
        datapipe3 = datapipe1.open_files(mode="b")

        count = 0
        for rec in datapipe3:
            count = count + 1
            self.assertTrue(rec[0] in self.temp_files)
            with open(rec[0], "rb") as f:
                self.assertEqual(rec[1].read(), f.read())
                rec[1].close()
        self.assertEqual(count, len(self.temp_files))

        # __len__ Test
        with self.assertRaises(TypeError):
            len(datapipe3)

    def test_routeddecoder_iterable_datapipe(self):
        temp_dir = self.temp_dir.name
        temp_pngfile_pathname = os.path.join(temp_dir, "test_png.png")
        png_data = np.array(
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            dtype=np.single,
        )
        np.save(temp_pngfile_pathname, png_data)
        datapipe1 = dp.iter.FileLister(temp_dir, ["*.png", "*.txt"])
        datapipe2 = dp.iter.FileOpener(datapipe1, mode="b")

        def _png_decoder(extension, data):
            if extension != "png":
                return None
            return np.load(data)

        def _helper(prior_dp, dp, channel_first=False):
            # Byte stream is not closed
            for inp in prior_dp:
                self.assertFalse(inp[1].closed)
            for inp, rec in zip(prior_dp, dp):
                ext = os.path.splitext(rec[0])[1]
                if ext == ".png":
                    expected = np.array(
                        [
                            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        ],
                        dtype=np.single,
                    )
                    if channel_first:
                        expected = expected.transpose(2, 0, 1)
                    self.assertEqual(rec[1], expected)
                else:
                    with open(rec[0], "rb") as f:
                        self.assertEqual(rec[1], f.read().decode("utf-8"))
                # Corresponding byte stream is closed by Decoder
                self.assertTrue(inp[1].closed)

        cached = list(datapipe2)
        with warnings.catch_warnings(record=True):
            datapipe3 = dp.iter.RoutedDecoder(cached, _png_decoder)
        datapipe3.add_handler(decoder_basichandlers)
        _helper(cached, datapipe3)

        cached = list(datapipe2)
        with warnings.catch_warnings(record=True):
            datapipe4 = dp.iter.RoutedDecoder(cached, decoder_basichandlers)
        datapipe4.add_handler(_png_decoder)
        _helper(cached, datapipe4, channel_first=True)

    def test_groupby_iterable_datapipe(self):
        file_list = [
            "a.png",
            "b.png",
            "c.json",
            "a.json",
            "c.png",
            "b.json",
            "d.png",
            "d.json",
            "e.png",
            "f.json",
            "g.png",
            "f.png",
            "g.json",
            "e.json",
            "h.txt",
            "h.json",
        ]

        import io

        datapipe1 = dp.iter.IterableWrapper(
            [(filename, io.BytesIO(b"12345abcde")) for filename in file_list]
        )

        def group_fn(data):
            filepath, _ = data
            return os.path.basename(filepath).split(".")[0]

        datapipe2 = dp.iter.Grouper(datapipe1, group_key_fn=group_fn, group_size=2)

        def order_fn(data):
            data.sort(key=lambda f: f[0], reverse=True)
            return data

        datapipe3 = dp.iter.Mapper(datapipe2, fn=order_fn)  # type: ignore[var-annotated]

        expected_result = [
            ("a.png", "a.json"),
            ("c.png", "c.json"),
            ("b.png", "b.json"),
            ("d.png", "d.json"),
            ("f.png", "f.json"),
            ("g.png", "g.json"),
            ("e.png", "e.json"),
            ("h.txt", "h.json"),
        ]

        count = 0
        for rec, expected in zip(datapipe3, expected_result):
            count = count + 1
            self.assertEqual(os.path.basename(rec[0][0]), expected[0])
            self.assertEqual(os.path.basename(rec[1][0]), expected[1])
            for i in [0, 1]:
                self.assertEqual(rec[i][1].read(), b"12345abcde")
                rec[i][1].close()
        self.assertEqual(count, 8)

        # testing the keep_key option
        datapipe4 = dp.iter.Grouper(
            datapipe1, group_key_fn=group_fn, keep_key=True, group_size=2
        )

        def order_fn(data):
            data[1].sort(key=lambda f: f[0], reverse=True)
            return data

        datapipe5 = dp.iter.Mapper(datapipe4, fn=order_fn)  # type: ignore[var-annotated]

        expected_result = [
            ("a", ("a.png", "a.json")),
            ("c", ("c.png", "c.json")),
            ("b", ("b.png", "b.json")),
            ("d", ("d.png", "d.json")),
            ("f", ("f.png", "f.json")),
            ("g", ("g.png", "g.json")),
            ("e", ("e.png", "e.json")),
            ("h", ("h.txt", "h.json")),
        ]

        count = 0
        for rec, expected in zip(datapipe5, expected_result):
            count = count + 1
            self.assertEqual(rec[0], expected[0])
            self.assertEqual(rec[1][0][0], expected[1][0])
            self.assertEqual(rec[1][1][0], expected[1][1])
            for i in [0, 1]:
                self.assertEqual(rec[1][i][1].read(), b"12345abcde")
                rec[1][i][1].close()
        self.assertEqual(count, 8)

    def test_demux_mux_datapipe(self):
        numbers = NumbersDataset(10)
        n1, n2 = numbers.demux(2, lambda x: x % 2)
        self.assertEqual([0, 2, 4, 6, 8], list(n1))
        self.assertEqual([1, 3, 5, 7, 9], list(n2))

        # Functional Test: demux and mux works sequentially as expected
        numbers = NumbersDataset(10)
        n1, n2, n3 = numbers.demux(3, lambda x: x % 3)
        n = n1.mux(n2, n3)
        self.assertEqual(list(range(9)), list(n))

        # Functional Test: Uneven DataPipes
        source_numbers = list(range(10)) + [10, 12]
        numbers_dp = dp.iter.IterableWrapper(source_numbers)
        n1, n2 = numbers_dp.demux(2, lambda x: x % 2)
        self.assertEqual([0, 2, 4, 6, 8, 10, 12], list(n1))
        self.assertEqual([1, 3, 5, 7, 9], list(n2))
        n = n1.mux(n2)
        self.assertEqual(list(range(10)), list(n))

    @suppress_warnings  # Suppress warning for lambda fn
    def test_map_with_col_file_handle_datapipe(self):
        temp_dir = self.temp_dir.name
        datapipe1 = dp.iter.FileLister(temp_dir, "")
        datapipe2 = dp.iter.FileOpener(datapipe1)

        def _helper(datapipe):
            dp1 = datapipe.map(lambda x: x.read(), input_col=1)
            dp2 = datapipe.map(lambda x: (x[0], x[1].read()))
            self.assertEqual(list(dp1), list(dp2))

        # tuple
        _helper(datapipe2)
        # list
        datapipe3 = datapipe2.map(lambda x: list(x))
        _helper(datapipe3)


@skipIfNoDataFrames
class TestCaptureDataFrame(TestCase):
    def get_new_df(self):
        return df_wrapper.create_dataframe([[1, 2]], columns=["a", "b"])

    def compare_capture_and_eager(self, operations):
        cdf = CaptureDataFrame()
        cdf = operations(cdf)
        df = self.get_new_df()
        cdf = cdf.apply_ops(df)

        df = self.get_new_df()
        df = operations(df)

        self.assertTrue(df.equals(cdf))

    def test_basic_capture(self):
        def operations(df):
            df["c"] = df.b + df["a"] * 7
            # somehow swallows pandas UserWarning when `df.c = df.b + df['a'] * 7`
            return df

        self.compare_capture_and_eager(operations)


class TestDataFramesPipes(TestCase):
    """
    Most of test will fail if pandas installed, but no dill available.
    Need to rework them to avoid multiple skips.
    """

    def _get_datapipe(self, range=10, dataframe_size=7):
        return NumbersDataset(range).map(lambda i: (i, i % 3))

    def _get_dataframes_pipe(self, range=10, dataframe_size=7):
        return (
            NumbersDataset(range)
            .map(lambda i: (i, i % 3))
            ._to_dataframes_pipe(columns=["i", "j"], dataframe_size=dataframe_size)
        )

    @skipIfNoDataFrames
    @skipIfNoDill  # TODO(VitalyFedyunin): Decouple tests from dill by avoiding lambdas in map
    def test_capture(self):
        dp_numbers = self._get_datapipe().map(lambda x: (x[0], x[1], x[1] + 3 * x[0]))
        df_numbers = self._get_dataframes_pipe()
        df_numbers["k"] = df_numbers["j"] + df_numbers.i * 3
        expected = list(dp_numbers)
        actual = list(df_numbers)
        self.assertEqual(expected, actual)

    @skipIfNoDataFrames
    @skipIfNoDill
    def test_shuffle(self):
        #  With non-zero (but extremely low) probability (when shuffle do nothing),
        #  this test fails, so feel free to restart
        df_numbers = self._get_dataframes_pipe(range=1000).shuffle()
        dp_numbers = self._get_datapipe(range=1000)
        df_result = [tuple(item) for item in df_numbers]
        self.assertNotEqual(list(dp_numbers), df_result)
        self.assertEqual(list(dp_numbers), sorted(df_result))

    @skipIfNoDataFrames
    @skipIfNoDill
    def test_batch(self):
        df_numbers = self._get_dataframes_pipe(range=100).batch(8)
        df_numbers_list = list(df_numbers)
        last_batch = df_numbers_list[-1]
        self.assertEqual(4, len(last_batch))
        unpacked_batch = [tuple(row) for row in last_batch]
        self.assertEqual([(96, 0), (97, 1), (98, 2), (99, 0)], unpacked_batch)

    @skipIfNoDataFrames
    @skipIfNoDill
    def test_unbatch(self):
        df_numbers = self._get_dataframes_pipe(range=100).batch(8).batch(3)
        dp_numbers = self._get_datapipe(range=100)
        self.assertEqual(list(dp_numbers), list(df_numbers.unbatch(2)))

    @skipIfNoDataFrames
    @skipIfNoDill
    def test_filter(self):
        df_numbers = self._get_dataframes_pipe(range=10).filter(lambda x: x.i > 5)
        actual = list(df_numbers)
        self.assertEqual([(6, 0), (7, 1), (8, 2), (9, 0)], actual)

    @skipIfNoDataFrames
    @skipIfNoDill
    def test_collate(self):
        def collate_i(column):
            return column.sum()

        def collate_j(column):
            return column.prod()

        df_numbers = self._get_dataframes_pipe(range=30).batch(3)
        df_numbers = df_numbers.collate({"j": collate_j, "i": collate_i})

        expected_i = [
            3,
            12,
            21,
            30,
            39,
            48,
            57,
            66,
            75,
            84,
        ]

        actual_i = []
        for i, _ in df_numbers:
            actual_i.append(i)
        self.assertEqual(expected_i, actual_i)

        actual_i = []
        for item in df_numbers:
            actual_i.append(item.i)
        self.assertEqual(expected_i, actual_i)


class IDP_NoLen(IterDataPipe):
    def __init__(self, input_dp):
        super().__init__()
        self.input_dp = input_dp

    # Prevent in-place modification
    def __iter__(self):
        input_dp = (
            self.input_dp
            if isinstance(self.input_dp, IterDataPipe)
            else copy.deepcopy(self.input_dp)
        )
        yield from input_dp


def _fake_fn(data):
    return data


def _fake_add(constant, data):
    return constant + data


def _fake_filter_fn(data):
    return True


def _simple_filter_fn(data):
    return data >= 5


def _fake_filter_fn_constant(constant, data):
    return data >= constant


def _mul_10(x):
    return x * 10


def _mod_3_test(x):
    return x % 3 == 1


def _to_list(x):
    return [x]


lambda_fn1 = lambda x: x  # noqa: E731
lambda_fn2 = lambda x: x % 2  # noqa: E731
lambda_fn3 = lambda x: x >= 5  # noqa: E731


class Add1Module(nn.Module):
    def forward(self, x):
        return x + 1


class Add1Callable:
    def __call__(self, x):
        return x + 1


class TestFunctionalIterDataPipe(TestCase):
    def _serialization_test_helper(self, datapipe, use_dill):
        if use_dill:
            serialized_dp = dill.dumps(datapipe)
            deserialized_dp = dill.loads(serialized_dp)
        else:
            serialized_dp = pickle.dumps(datapipe)
            deserialized_dp = pickle.loads(serialized_dp)
        try:
            self.assertEqual(list(datapipe), list(deserialized_dp))
        except AssertionError as e:
            print(f"{datapipe} is failing.")
            raise e

    def _serialization_test_for_single_dp(self, dp, use_dill=False):
        # 1. Testing for serialization before any iteration starts
        self._serialization_test_helper(dp, use_dill)
        # 2. Testing for serialization after DataPipe is partially read
        it = iter(dp)
        _ = next(it)
        self._serialization_test_helper(dp, use_dill)
        # 3. Testing for serialization after DataPipe is fully read
        it = iter(dp)
        _ = list(it)
        self._serialization_test_helper(dp, use_dill)

    def _serialization_test_for_dp_with_children(self, dp1, dp2, use_dill=False):
        # 1. Testing for serialization before any iteration starts
        self._serialization_test_helper(dp1, use_dill)
        self._serialization_test_helper(dp2, use_dill)

        # 2. Testing for serialization after DataPipe is partially read
        it1, it2 = iter(dp1), iter(dp2)
        _, _ = next(it1), next(it2)
        # Catch `fork`, `demux` "some child DataPipes are not exhausted" warning
        with warnings.catch_warnings(record=True):
            self._serialization_test_helper(dp1, use_dill)
            self._serialization_test_helper(dp2, use_dill)

        # 2.5. Testing for serialization after one child DataPipe is fully read
        #      (Only for DataPipes with children DataPipes)
        it1 = iter(dp1)
        _ = list(it1)  # fully read one child
        # Catch `fork`, `demux` "some child DataPipes are not exhausted" warning
        with warnings.catch_warnings(record=True):
            self._serialization_test_helper(dp1, use_dill)
            self._serialization_test_helper(dp2, use_dill)

        # 3. Testing for serialization after DataPipe is fully read
        it2 = iter(dp2)
        _ = list(it2)  # fully read the other child
        self._serialization_test_helper(dp1, use_dill)
        self._serialization_test_helper(dp2, use_dill)

    def test_serializable(self):
        picklable_datapipes: list = [
            (
                dp.iter.Batcher,
                None,
                (
                    3,
                    True,
                ),
                {},
            ),
            (dp.iter.Collator, None, (_fake_fn,), {}),
            (dp.iter.Concater, None, (dp.iter.IterableWrapper(range(5)),), {}),
            (dp.iter.Demultiplexer, None, (2, _simple_filter_fn), {}),
            (dp.iter.FileLister, ".", (), {}),
            (dp.iter.FileOpener, None, (), {}),
            (dp.iter.Filter, None, (_fake_filter_fn,), {}),
            (dp.iter.Filter, None, (partial(_fake_filter_fn_constant, 5),), {}),
            (dp.iter.Forker, None, (2,), {}),
            (dp.iter.Forker, None, (2,), {"copy": "shallow"}),
            (dp.iter.Grouper, None, (_fake_filter_fn,), {"group_size": 2}),
            (dp.iter.IterableWrapper, range(10), (), {}),
            (dp.iter.Mapper, None, (_fake_fn,), {}),
            (dp.iter.Mapper, None, (partial(_fake_add, 1),), {}),
            (dp.iter.Multiplexer, None, (dp.iter.IterableWrapper(range(10)),), {}),
            (dp.iter.Sampler, None, (), {}),
            (dp.iter.Shuffler, dp.iter.IterableWrapper([0] * 10), (), {}),
            (dp.iter.StreamReader, None, (), {}),
            (dp.iter.UnBatcher, None, (0,), {}),
            (dp.iter.Zipper, None, (dp.iter.IterableWrapper(range(10)),), {}),
        ]
        # Skipping comparison for these DataPipes
        dp_skip_comparison = {dp.iter.FileOpener, dp.iter.StreamReader}
        # These DataPipes produce multiple DataPipes as outputs and those should be compared
        dp_compare_children = {dp.iter.Demultiplexer, dp.iter.Forker}

        for dpipe, custom_input, dp_args, dp_kwargs in picklable_datapipes:
            if custom_input is None:
                custom_input = dp.iter.IterableWrapper(range(10))
            if (
                dpipe in dp_skip_comparison
            ):  # Merely make sure they are picklable and loadable (no value comparison)
                datapipe = dpipe(custom_input, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                serialized_dp = pickle.dumps(datapipe)
                _ = pickle.loads(serialized_dp)
            elif dpipe in dp_compare_children:  # DataPipes that have children
                dp1, dp2 = dpipe(custom_input, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                self._serialization_test_for_dp_with_children(dp1, dp2)
            else:  # Single DataPipe that requires comparison
                datapipe = dpipe(custom_input, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                self._serialization_test_for_single_dp(datapipe)

    @skipIfTorchDynamo("Dict with function as keys")
    def test_serializable_with_dill(self):
        """Only for DataPipes that take in a function as argument"""
        input_dp = dp.iter.IterableWrapper(range(10))

        datapipes_with_lambda_fn: list[
            tuple[type[IterDataPipe], tuple, dict[str, Any]]
        ] = [
            (dp.iter.Collator, (lambda_fn1,), {}),
            (
                dp.iter.Demultiplexer,
                (
                    2,
                    lambda_fn2,
                ),
                {},
            ),
            (dp.iter.Filter, (lambda_fn3,), {}),
            (dp.iter.Grouper, (lambda_fn3,), {}),
            (dp.iter.Mapper, (lambda_fn1,), {}),
        ]

        def _local_fns():
            def _fn1(x):
                return x

            def _fn2(x):
                return x % 2

            def _fn3(x):
                return x >= 5

            return _fn1, _fn2, _fn3

        fn1, fn2, fn3 = _local_fns()

        datapipes_with_local_fn: list[
            tuple[type[IterDataPipe], tuple, dict[str, Any]]
        ] = [
            (dp.iter.Collator, (fn1,), {}),
            (
                dp.iter.Demultiplexer,
                (
                    2,
                    fn2,
                ),
                {},
            ),
            (dp.iter.Filter, (fn3,), {}),
            (dp.iter.Grouper, (fn3,), {}),
            (dp.iter.Mapper, (fn1,), {}),
        ]

        dp_compare_children = {dp.iter.Demultiplexer}

        if HAS_DILL:
            for dpipe, dp_args, dp_kwargs in (
                datapipes_with_lambda_fn + datapipes_with_local_fn
            ):
                if dpipe in dp_compare_children:
                    dp1, dp2 = dpipe(input_dp, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                    self._serialization_test_for_dp_with_children(
                        dp1, dp2, use_dill=True
                    )
                else:
                    datapipe = dpipe(input_dp, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                    self._serialization_test_for_single_dp(datapipe, use_dill=True)
        else:
            msgs = (
                r"^Lambda function is not supported by pickle",
                r"^Local function is not supported by pickle",
            )
            for dps, msg in zip(
                (datapipes_with_lambda_fn, datapipes_with_local_fn), msgs
            ):
                for dpipe, dp_args, dp_kwargs in dps:
                    with self.assertWarnsRegex(UserWarning, msg):
                        datapipe = dpipe(input_dp, *dp_args, **dp_kwargs)  # type: ignore[call-arg]
                    with self.assertRaises((pickle.PicklingError, AttributeError)):
                        pickle.dumps(datapipe)

    def test_docstring(self):
        """
        Ensure functional form of IterDataPipe has the correct docstring from
        the class form.

        Regression test for https://github.com/pytorch/data/issues/792.
        """
        input_dp = dp.iter.IterableWrapper(range(10))

        for dp_funcname in [
            "batch",
            "collate",
            "concat",
            "demux",
            "filter",
            "fork",
            "map",
            "mux",
            "read_from_stream",
            # "sampler",
            "shuffle",
            "unbatch",
            "zip",
        ]:
            docstring = pydoc.render_doc(
                thing=getattr(input_dp, dp_funcname), forceload=True
            )

            assert f"(functional name: ``{dp_funcname}``)" in docstring
            assert "Args:" in docstring
            assert "Example:" in docstring or "Examples:" in docstring

    def test_iterable_wrapper_datapipe(self):
        input_ls = list(range(10))
        input_dp = dp.iter.IterableWrapper(input_ls)

        # Functional Test: values are unchanged and in the same order
        self.assertEqual(input_ls, list(input_dp))

        # Functional Test: deep copy by default when an iterator is initialized (first element is read)
        it = iter(input_dp)
        self.assertEqual(
            0, next(it)
        )  # The deep copy only happens when the first element is read
        input_ls.append(50)
        self.assertEqual(list(range(1, 10)), list(it))

        # Functional Test: shallow copy
        input_ls2 = [1, 2, 3]
        input_dp_shallow = dp.iter.IterableWrapper(input_ls2, deepcopy=False)
        input_ls2.append(10)
        self.assertEqual([1, 2, 3, 10], list(input_dp_shallow))

        # Reset Test: reset the DataPipe
        input_ls = list(range(10))
        input_dp = dp.iter.IterableWrapper(input_ls)
        n_elements_before_reset = 5
        res_before_reset, res_after_reset = reset_after_n_next_calls(
            input_dp, n_elements_before_reset
        )
        self.assertEqual(input_ls[:n_elements_before_reset], res_before_reset)
        self.assertEqual(input_ls, res_after_reset)

        # __len__ Test: inherits length from sequence
        self.assertEqual(len(input_ls), len(input_dp))

    def test_concat_iterdatapipe(self):
        input_dp1 = dp.iter.IterableWrapper(range(10))
        input_dp2 = dp.iter.IterableWrapper(range(5))

        # Functional Test: Raises exception for empty input
        with self.assertRaisesRegex(ValueError, r"Expected at least one DataPipe"):
            dp.iter.Concater()

        # Functional Test: Raises exception for non-IterDataPipe input
        with self.assertRaisesRegex(
            TypeError, r"Expected all inputs to be `IterDataPipe`"
        ):
            dp.iter.Concater(input_dp1, ())  # type: ignore[arg-type]

        # Functional Test: Concatenate DataPipes as expected
        concat_dp = input_dp1.concat(input_dp2)
        self.assertEqual(len(concat_dp), 15)
        self.assertEqual(list(concat_dp), list(range(10)) + list(range(5)))

        # Reset Test: reset the DataPipe
        n_elements_before_reset = 5
        res_before_reset, res_after_reset = reset_after_n_next_calls(
            concat_dp, n_elements_before_reset
        )
        self.assertEqual(list(range(5)), res_before_reset)
        self.assertEqual(list(range(10)) + list(range(5)), res_after_reset)

        # __len__ Test: inherits length from source DataPipe
        input_dp_nl = IDP_NoLen(range(5))
        concat_dp = input_dp1.concat(input_dp_nl)
        with self.assertRaisesRegex(TypeError, r"instance doesn't have valid length$"):
            len(concat_dp)

        self.assertEqual(list(concat_dp), list(range(10)) + list(range(5)))

    def test_fork_iterdatapipe(self):
        input_dp = dp.iter.IterableWrapper(range(10))

        with self.assertRaises(ValueError):
            input_dp.fork(num_instances=0)

        dp0 = input_dp.fork(num_instances=1, buffer_size=0)
        self.assertEqual(dp0, input_dp)

        # Functional Test: making sure all child DataPipe shares the same reference
        dp1, dp2, dp3 = input_dp.fork(num_instances=3)
        self.assertTrue(all(n1 is n2 and n1 is n3 for n1, n2, n3 in zip(dp1, dp2, dp3)))

        # Functional Test: one child DataPipe yields all value at a time
        output1, output2, output3 = list(dp1), list(dp2), list(dp3)
        self.assertEqual(list(range(10)), output1)
        self.assertEqual(list(range(10)), output2)
        self.assertEqual(list(range(10)), output3)

        # Functional Test: two child DataPipes yield value together
        dp1, dp2 = input_dp.fork(num_instances=2)
        output = []
        for n1, n2 in zip(dp1, dp2):
            output.append((n1, n2))
        self.assertEqual([(i, i) for i in range(10)], output)

        # Functional Test: one child DataPipe yields all value first, but buffer_size = 5 being too small
        dp1, dp2 = input_dp.fork(num_instances=2, buffer_size=4)
        it1 = iter(dp1)
        for _ in range(4):
            next(it1)
        with self.assertRaises(BufferError):
            next(it1)
        with self.assertRaises(BufferError):
            list(dp2)

        dp1, dp2 = input_dp.fork(num_instances=2, buffer_size=5)
        with self.assertRaises(BufferError):
            list(dp2)

        # Functional Test: one child DataPipe yields all value first with unlimited buffer
        with warnings.catch_warnings(record=True) as wa:
            dp1, dp2 = input_dp.fork(num_instances=2, buffer_size=-1)
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), r"Unlimited buffer size is set")
        l1, l2 = list(dp1), list(dp2)
        for d1, d2 in zip(l1, l2):
            self.assertEqual(d1, d2)

        # Functional Test: two child DataPipes yield value together with buffer size 1
        dp1, dp2 = input_dp.fork(num_instances=2, buffer_size=1)
        output = []
        for n1, n2 in zip(dp1, dp2):
            output.append((n1, n2))
        self.assertEqual([(i, i) for i in range(10)], output)

        # Functional Test: two child DataPipes yield shallow copies with copy equals shallow
        dp1, dp2 = input_dp.map(_to_list).fork(num_instances=2, copy="shallow")
        for n1, n2 in zip(dp1, dp2):
            self.assertIsNot(n1, n2)
            self.assertEqual(n1, n2)

        # Functional Test: two child DataPipes yield deep copies with copy equals deep
        dp1, dp2 = (
            input_dp.map(_to_list).map(_to_list).fork(num_instances=2, copy="deep")
        )
        for n1, n2 in zip(dp1, dp2):
            self.assertIsNot(n1[0], n2[0])
            self.assertEqual(n1, n2)

        # Functional Test: fork DataPipe raises error for unknown copy method
        with self.assertRaises(ValueError):
            input_dp.fork(num_instances=2, copy="unknown")

        # Functional Test: make sure logic related to slowest_ptr is working properly
        dp1, dp2, dp3 = input_dp.fork(num_instances=3)
        output1, output2, output3 = [], [], []
        for i, (n1, n2) in enumerate(zip(dp1, dp2)):
            output1.append(n1)
            output2.append(n2)
            if i == 4:  # yield all of dp3 when halfway through dp1, dp2
                output3 = list(dp3)
                break
        self.assertEqual(list(range(5)), output1)
        self.assertEqual(list(range(5)), output2)
        self.assertEqual(list(range(10)), output3)

        # Reset Test: DataPipe resets when a new iterator is created, even if this datapipe hasn't been read
        dp1, dp2 = input_dp.fork(num_instances=2)
        _ = iter(dp1)
        output2 = []
        with self.assertRaisesRegex(RuntimeError, r"iterator has been invalidated"):
            for i, n2 in enumerate(dp2):
                output2.append(n2)
                if i == 4:
                    with warnings.catch_warnings(record=True) as wa:
                        _ = iter(dp1)  # This will reset all child DataPipes
                        self.assertEqual(len(wa), 1)
                        self.assertRegex(
                            str(wa[0].message), r"child DataPipes are not exhausted"
                        )
        self.assertEqual(list(range(5)), output2)

        # Reset Test: DataPipe resets when some of it has been read
        dp1, dp2 = input_dp.fork(num_instances=2)
        output1, output2 = [], []
        for i, (n1, n2) in enumerate(zip(dp1, dp2)):
            output1.append(n1)
            output2.append(n2)
            if i == 4:
                with warnings.catch_warnings(record=True) as wa:
                    _ = iter(dp1)  # Reset both all child DataPipe
                    self.assertEqual(len(wa), 1)
                    self.assertRegex(
                        str(wa[0].message), r"Some child DataPipes are not exhausted"
                    )
                break
        with warnings.catch_warnings(record=True) as wa:
            for n1, n2 in zip(dp1, dp2):
                output1.append(n1)
                output2.append(n2)
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), r"child DataPipes are not exhausted")
        self.assertEqual(list(range(5)) + list(range(10)), output1)
        self.assertEqual(list(range(5)) + list(range(10)), output2)

        # Reset Test: DataPipe reset, even when some other child DataPipes are not read
        dp1, dp2, dp3 = input_dp.fork(num_instances=3)
        output1, output2 = list(dp1), list(dp2)
        self.assertEqual(list(range(10)), output1)
        self.assertEqual(list(range(10)), output2)
        with warnings.catch_warnings(record=True) as wa:
            self.assertEqual(
                list(range(10)), list(dp1)
            )  # Resets even though dp3 has not been read
            self.assertEqual(len(wa), 1)
            self.assertRegex(
                str(wa[0].message), r"Some child DataPipes are not exhausted"
            )
        output3 = []
        for i, n3 in enumerate(dp3):
            output3.append(n3)
            if i == 4:
                with warnings.catch_warnings(record=True) as wa:
                    output1 = list(dp1)  # Resets even though dp3 is only partially read
                    self.assertEqual(len(wa), 1)
                    self.assertRegex(
                        str(wa[0].message), r"Some child DataPipes are not exhausted"
                    )
                self.assertEqual(list(range(5)), output3)
                self.assertEqual(list(range(10)), output1)
                break
        self.assertEqual(
            list(range(10)), list(dp3)
        )  # dp3 has to read from the start again

        # __len__ Test: Each DataPipe inherits the source datapipe's length
        dp1, dp2, dp3 = input_dp.fork(num_instances=3)
        self.assertEqual(len(input_dp), len(dp1))
        self.assertEqual(len(input_dp), len(dp2))
        self.assertEqual(len(input_dp), len(dp3))

        # Pickle Test:
        dp1, dp2, dp3 = input_dp.fork(num_instances=3)
        traverse_dps(dp1)  # This should not raise any error
        for _ in zip(dp1, dp2, dp3):
            pass
        traverse_dps(dp2)  # This should not raise any error either

    def test_mux_iterdatapipe(self):
        # Functional Test: Elements are yielded one at a time from each DataPipe, until they are all exhausted
        input_dp1 = dp.iter.IterableWrapper(range(4))
        input_dp2 = dp.iter.IterableWrapper(range(4, 8))
        input_dp3 = dp.iter.IterableWrapper(range(8, 12))
        output_dp = input_dp1.mux(input_dp2, input_dp3)
        expected_output = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        self.assertEqual(len(expected_output), len(output_dp))
        self.assertEqual(expected_output, list(output_dp))

        # Functional Test: Uneven input Data Pipes
        input_dp1 = dp.iter.IterableWrapper([1, 2, 3, 4])
        input_dp2 = dp.iter.IterableWrapper([10])
        input_dp3 = dp.iter.IterableWrapper([100, 200, 300])
        output_dp = input_dp1.mux(input_dp2, input_dp3)
        expected_output = [1, 10, 100]
        self.assertEqual(len(expected_output), len(output_dp))
        self.assertEqual(expected_output, list(output_dp))

        # Functional Test: Empty Data Pipe
        input_dp1 = dp.iter.IterableWrapper([0, 1, 2, 3])
        input_dp2 = dp.iter.IterableWrapper([])
        output_dp = input_dp1.mux(input_dp2)
        self.assertEqual(len(input_dp2), len(output_dp))
        self.assertEqual(list(input_dp2), list(output_dp))

        # __len__ Test: raises TypeError when __len__ is called and an input doesn't have __len__
        input_dp1 = dp.iter.IterableWrapper(range(10))
        input_dp_no_len = IDP_NoLen(range(10))
        output_dp = input_dp1.mux(input_dp_no_len)
        with self.assertRaises(TypeError):
            len(output_dp)

    def test_demux_iterdatapipe(self):
        input_dp = dp.iter.IterableWrapper(range(10))

        with self.assertRaises(ValueError):
            input_dp.demux(num_instances=0, classifier_fn=lambda x: 0)

        # Functional Test: split into 2 DataPipes and output them one at a time
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        output1, output2 = list(dp1), list(dp2)
        self.assertEqual(list(range(0, 10, 2)), output1)
        self.assertEqual(list(range(1, 10, 2)), output2)

        # Functional Test: split into 2 DataPipes and output them together
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        output = []
        for n1, n2 in zip(dp1, dp2):
            output.append((n1, n2))
        self.assertEqual([(i, i + 1) for i in range(0, 10, 2)], output)

        # Functional Test: values of the same classification are lumped together, and buffer_size = 3 being too small
        dp1, dp2 = input_dp.demux(
            num_instances=2, classifier_fn=lambda x: 0 if x >= 5 else 1, buffer_size=4
        )
        it1 = iter(dp1)
        with self.assertRaises(BufferError):
            next(
                it1
            )  # Buffer raises because first 5 elements all belong to the a different child
        with self.assertRaises(BufferError):
            list(dp2)

        # Functional Test: values of the same classification are lumped together, and buffer_size = 5 is just enough
        dp1, dp2 = input_dp.demux(
            num_instances=2, classifier_fn=lambda x: 0 if x >= 5 else 1, buffer_size=5
        )
        output1, output2 = list(dp1), list(dp2)
        self.assertEqual(list(range(5, 10)), output1)
        self.assertEqual(list(range(5)), output2)

        # Functional Test: values of the same classification are lumped together, and unlimited buffer
        with warnings.catch_warnings(record=True) as wa:
            dp1, dp2 = input_dp.demux(
                num_instances=2,
                classifier_fn=lambda x: 0 if x >= 5 else 1,
                buffer_size=-1,
            )
            exp_l = 1 if HAS_DILL else 2
            self.assertEqual(len(wa), exp_l)
            self.assertRegex(str(wa[-1].message), r"Unlimited buffer size is set")
        output1, output2 = list(dp1), list(dp2)
        self.assertEqual(list(range(5, 10)), output1)
        self.assertEqual(list(range(5)), output2)

        # Functional Test: classifier returns a value outside of [0, num_instance - 1]
        dp0 = input_dp.demux(num_instances=1, classifier_fn=lambda x: x % 2)
        it = iter(dp0[0])
        with self.assertRaises(ValueError):
            next(it)
            next(it)

        # Reset Test: DataPipe resets when a new iterator is created, even if this datapipe hasn't been read
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        _ = iter(dp1)
        output2 = []
        with self.assertRaisesRegex(RuntimeError, r"iterator has been invalidated"):
            for i, n2 in enumerate(dp2):
                output2.append(n2)
                if i == 4:
                    with warnings.catch_warnings(record=True) as wa:
                        _ = iter(dp1)  # This will reset all child DataPipes
                        self.assertEqual(len(wa), 1)
                        self.assertRegex(
                            str(wa[0].message), r"child DataPipes are not exhausted"
                        )
        self.assertEqual(list(range(1, 10, 2)), output2)

        # Reset Test: DataPipe resets when some of it has been read
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        output1, output2 = [], []
        for n1, n2 in zip(dp1, dp2):
            output1.append(n1)
            output2.append(n2)
            if n1 == 4:
                break
        with warnings.catch_warnings(record=True) as wa:
            iter(dp1)  # Reset all child DataPipes
            self.assertEqual(len(wa), 1)
            self.assertRegex(
                str(wa[0].message), r"Some child DataPipes are not exhausted"
            )
            for n1, n2 in zip(dp1, dp2):
                output1.append(n1)
                output2.append(n2)
            self.assertEqual([0, 2, 4] + list(range(0, 10, 2)), output1)
            self.assertEqual([1, 3, 5] + list(range(1, 10, 2)), output2)
            self.assertEqual(len(wa), 1)
            self.assertRegex(str(wa[0].message), r"child DataPipes are not exhausted")

        # Reset Test: DataPipe reset, even when not all child DataPipes are exhausted
        dp1, dp2 = input_dp.demux(num_instances=2, classifier_fn=lambda x: x % 2)
        output1 = list(dp1)
        self.assertEqual(list(range(0, 10, 2)), output1)
        with warnings.catch_warnings(record=True) as wa:
            self.assertEqual(
                list(range(0, 10, 2)), list(dp1)
            )  # Reset even when dp2 is not read
            self.assertEqual(len(wa), 1)
            self.assertRegex(
                str(wa[0].message), r"Some child DataPipes are not exhausted"
            )
        output2 = []
        for i, n2 in enumerate(dp2):
            output2.append(n2)
            if i == 1:
                self.assertEqual(list(range(1, 5, 2)), output2)
                with warnings.catch_warnings(record=True) as wa:
                    self.assertEqual(
                        list(range(0, 10, 2)), list(dp1)
                    )  # Can reset even when dp2 is partially read
                    self.assertEqual(len(wa), 1)
                    self.assertRegex(
                        str(wa[0].message), r"Some child DataPipes are not exhausted"
                    )
                break
        output2 = list(dp2)  # output2 has to read from beginning again
        self.assertEqual(list(range(1, 10, 2)), output2)

        # Functional Test: drop_none = True
        dp1, dp2 = input_dp.demux(
            num_instances=2,
            classifier_fn=lambda x: x % 2 if x % 5 != 0 else
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

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_datapipe.py_docs.md
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

- **File Documentation**: `test_datapipe.py_docs.md_docs.md`
- **Keyword Index**: `test_datapipe.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
