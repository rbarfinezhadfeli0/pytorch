# Documentation: `test/test_serialization.py`

## File Metadata

- **Path**: `test/test_serialization.py`
- **Size**: 315,676 bytes (308.28 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# Owner(s): ["module: serialization"]
# ruff: noqa: F841

import contextlib
import copy
import functools
import gc
import gzip
import io
import os
import pathlib
import pickle
import platform
import re
import shutil
import sys
import tempfile
import unittest
import warnings
import zipfile
from collections import namedtuple, OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.serialization import config as serialization_config
from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensorConverter
from torch._utils import _rebuild_tensor
from torch._utils_internal import get_file_path_2
from torch.serialization import (
    check_module_version_greater_or_equal,
    get_default_load_endianness,
    LoadEndianness,
    safe_globals,
    set_default_load_endianness,
    skip_data,
    SourceChangeWarning,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_dtype import all_types_and_complex_and
from torch.testing._internal.common_utils import (
    AlwaysWarnTypedStorageRemoval,
    BytesIOContext,
    download_file,
    instantiate_parametrized_tests,
    IS_CI,
    IS_FBCODE,
    IS_FILESYSTEM_UTF8_ENCODING,
    IS_WINDOWS,
    parametrize,
    run_tests,
    serialTest,
    skipIfTorchDynamo,
    TemporaryDirectoryName,
    TemporaryFileName,
    TEST_DILL,
    TEST_WITH_MTIA,
    TestCase,
)
from torch.testing._internal.two_tensor import TwoTensor  # noqa: F401
from torch.utils._import_utils import import_dill
from pickle import UnpicklingError


if not IS_WINDOWS:
    from mmap import MAP_PRIVATE, MAP_SHARED
else:
    MAP_SHARED, MAP_PRIVATE = None, None

if TEST_WITH_MTIA:
    import mtia.host_runtime.torch_mtia.dynamic_library  # noqa: F401

# These tests were all copied from `test/test_torch.py` at some point, so see
# the actual blame, see this revision
# https://github.com/pytorch/pytorch/blame/9a2691f2fc948b9792686085b493c61793c2de30/test/test_torch.py

dill = import_dill()
HAS_DILL_AT_LEAST_0_3_1 = dill is not None and check_module_version_greater_or_equal(dill, (0, 3, 1))

can_retrieve_source = True
with warnings.catch_warnings(record=True) as warns:
    with tempfile.NamedTemporaryFile() as checkpoint:
        x = torch.save(torch.nn.Module(), checkpoint)
        for warn in warns:
            if "Couldn't retrieve source code" in warn.message.args[0]:
                can_retrieve_source = False
                break


class FilelikeMock:
    def __init__(self, data, has_fileno=True, has_readinto=False):
        if has_readinto:
            self.readinto = self.readinto_opt
        if has_fileno:
            # Python 2's StringIO.StringIO has no fileno attribute.
            # This is used to test that.
            self.fileno = self.fileno_opt

        self.calls = set()
        self.bytesio = io.BytesIO(data)

        def trace(fn, name):
            def result(*args, **kwargs):
                self.calls.add(name)
                return fn(*args, **kwargs)
            return result

        for attr in ['read', 'readline', 'seek', 'tell', 'write', 'flush']:
            traced_fn = trace(getattr(self.bytesio, attr), attr)
            setattr(self, attr, traced_fn)

    def fileno_opt(self):
        raise io.UnsupportedOperation('Not a real file')

    def readinto_opt(self, view):
        self.calls.add('readinto')
        return self.bytesio.readinto(view)

    def was_called(self, name):
        return name in self.calls

class ClassAMock:
    class Nested:
        pass

class ClassBMock:
    class Nested:
        pass

def up_size(size):
    return (*size[:-1], size[-1] * 2)

class UInt4Tensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem, **kwargs):
        assert elem.dtype is torch.uint8
        assert not kwargs.get("requires_grad", False)
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, up_size(elem.shape), dtype=torch.uint4, **kwargs)

    def __init__(self, elem):
        self.elem = elem

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        pass


class Int4Tensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem, **kwargs):
        assert elem.dtype is torch.uint8
        assert not kwargs.get("requires_grad", False)
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, up_size(elem.shape), dtype=torch.int4, **kwargs)

    def __init__(self, elem):
        self.elem = elem

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        pass


class SerializationMixin:
    def _test_serialization_data(self):
        a = [torch.randn(5, 5).float() for i in range(2)]
        b = [a[i % 2] for i in range(4)]  # 0-3
        b += [a[0].storage()]  # 4
        b += [a[0].reshape(-1)[1:4].storage()]  # 5
        b += [torch.arange(1, 11).int()]  # 6
        t1 = torch.FloatTensor().set_(a[0].reshape(-1)[1:4].clone().storage(), 0, (3,), (1,))
        t2 = torch.FloatTensor().set_(a[0].reshape(-1)[1:4].clone().storage(), 0, (3,), (1,))
        b += [(t1.storage(), t1.storage(), t2.storage())]  # 7
        b += [a[0].reshape(-1)[0:2].storage()]  # 8
        return b

    def _test_serialization_assert(self, b, c):
        self.assertEqual(b, c, atol=0, rtol=0)
        self.assertTrue(isinstance(c[0], torch.FloatTensor))
        self.assertTrue(isinstance(c[1], torch.FloatTensor))
        self.assertTrue(isinstance(c[2], torch.FloatTensor))
        self.assertTrue(isinstance(c[3], torch.FloatTensor))
        self.assertTrue(isinstance(c[4], torch.storage.TypedStorage))
        self.assertEqual(c[4].dtype, torch.float)
        c[0].fill_(10)
        self.assertEqual(c[0], c[2], atol=0, rtol=0)
        self.assertEqual(c[4], torch.FloatStorage(25).fill_(10), atol=0, rtol=0)
        c[1].fill_(20)
        self.assertEqual(c[1], c[3], atol=0, rtol=0)
        # I have to do it in this roundabout fashion, because there's no
        # way to slice storages
        for i in range(4):
            self.assertEqual(c[4][i + 1], c[5][i])

        # check that serializing the same storage view object unpickles
        # it as one object not two (and vice versa)
        views = c[7]
        self.assertEqual(views[0]._cdata, views[1]._cdata)
        self.assertEqual(views[0], views[2])
        self.assertNotEqual(views[0]._cdata, views[2]._cdata)

        rootview = c[8]
        self.assertEqual(rootview.data_ptr(), c[0].data_ptr())

    def test_serialization_zipfile_utils(self):
        data = {
            'a': b'12039810948234589',
            'b': b'1239081209484958',
            'c/d': b'94589480984058'
        }

        def test(name_or_buffer):
            with torch.serialization._open_zipfile_writer(name_or_buffer) as zip_file:
                for key in data:
                    zip_file.write_record(key, data[key], len(data[key]))

            if hasattr(name_or_buffer, 'seek'):
                name_or_buffer.seek(0)

            with torch.serialization._open_zipfile_reader(name_or_buffer) as zip_file:
                for key in data:
                    actual = zip_file.get_record(key)
                    expected = data[key]
                    self.assertEqual(expected, actual)

        with tempfile.NamedTemporaryFile() as f:
            test(f)

        with TemporaryFileName() as fname:
            test(fname)

        test(io.BytesIO())

    def _test_serialization(self, weights_only):
        # Test serialization with a real file
        b = self._test_serialization_data()
        with tempfile.NamedTemporaryFile() as f:
            torch.save(b, f)
            f.seek(0)
            c = torch.load(f, weights_only=weights_only)
            self._test_serialization_assert(b, c)
        with TemporaryFileName() as fname:
            torch.save(b, fname)
            c = torch.load(fname, weights_only=weights_only)
            self._test_serialization_assert(b, c)
        # test non-ascii encoding of bytes arrays/strings
        # The following bytes are produced by serializing
        #   [b'\xc5\xbc\xc4\x85\xc4\x85\xc3\xb3\xc5\xbc\xc4\x85\xc5\xbc', torch.zeros(1, dtype=torch.float), 2]
        # in Python 2.7.12 and PyTorch 0.4.1, where the first element contains
        # bytes of some utf-8 characters (i.e., `utf8_str.encode('utf-8')`).
        serialized = (
            b'\x80\x02\x8a\nl\xfc\x9cF\xf9 j\xa8P\x19.\x80\x02M\xe9\x03.'
            b'\x80\x02}q\x01(U\x10protocol_versionq\x02M\xe9\x03U\n'
            b'type_sizesq\x03}q\x04(U\x03intq\x05K\x04U\x05shortq\x06K\x02U'
            b'\x04longq\x07K\x04uU\rlittle_endianq\x08\x88u.\x80\x02]q'
            b'\x01(U\x0e\xc5\xbc\xc4\x85\xc4\x85\xc3\xb3\xc5\xbc\xc4\x85'
            b'\xc5\xbcq\x02ctorch._utils\n_rebuild_tensor_v2\nq\x03((U'
            b'\x07storageq\x04ctorch\nFloatStorage\nq\x05U\x0845640624q'
            b'\x06U\x03cpuq\x07\x8a\x01\x01NtQK\x00K\x01\x85K\x01\x85'
            b'\x89NtRq\x08K\x02e.\x80\x02]q\x01U\x0845640624q\x02a.\x01\x00'
            b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        )
        buf = io.BytesIO(serialized)
        utf8_bytes = b'\xc5\xbc\xc4\x85\xc4\x85\xc3\xb3\xc5\xbc\xc4\x85\xc5\xbc'
        utf8_str = utf8_bytes.decode('utf-8')
        loaded_utf8 = torch.load(buf, weights_only=weights_only, encoding='utf-8')
        self.assertEqual(loaded_utf8, [utf8_str, torch.zeros(1, dtype=torch.float), 2])
        buf.seek(0)
        loaded_bytes = torch.load(buf, weights_only=weights_only, encoding='bytes')
        self.assertEqual(loaded_bytes, [utf8_bytes, torch.zeros(1, dtype=torch.float), 2])

    def test_serialization(self):
        self._test_serialization(False)

    def test_serialization_safe(self):
        self._test_serialization(True)

    def test_serialization_filelike(self):
        # Test serialization (load and save) with a filelike object
        b = self._test_serialization_data()
        with BytesIOContext() as f:
            torch.save(b, f)
            f.seek(0)
            c = torch.load(f)
        self._test_serialization_assert(b, c)

    def test_serialization_fake_zip(self):
        data = [
            ord('P'),
            ord('K'),
            5,
            6
        ]
        for _ in range(100):
            data.append(0)
        t = torch.tensor(data, dtype=torch.uint8)

        with tempfile.NamedTemporaryFile() as f:
            torch.save(t, f)

            # If this check is False for all Python versions (i.e. the fix
            # has been backported), this test and torch.serialization._is_zipfile
            # can be deleted
            self.assertTrue(zipfile.is_zipfile(f))
            self.assertFalse(torch.serialization._is_zipfile(f))
            f.seek(0)
            self.assertEqual(torch.load(f), t)

    def test_serialization_gzip(self):
        # Test serialization with gzip file
        b = self._test_serialization_data()
        with tempfile.NamedTemporaryFile() as f1, tempfile.NamedTemporaryFile(delete=False) as f2:
            torch.save(b, f1)
            f1.seek(0)
            with gzip.open(f2.name, 'wb') as f_out:
                shutil.copyfileobj(f1, f_out)

            with gzip.open(f2.name, 'rb') as f:
                c = torch.load(f)
                self._test_serialization_assert(b, c)
            f2.close()
            os.unlink(f2.name)

    @unittest.skipIf(
        not TEST_DILL or HAS_DILL_AT_LEAST_0_3_1,
        '"dill" not found or is correct version'
    )
    def test_serialization_dill_version_not_supported(self):
        x = torch.randn(5, 5)

        with tempfile.NamedTemporaryFile() as f:
            with self.assertRaisesRegex(ValueError, 'supports dill >='):
                torch.save(x, f, pickle_module=dill)
            f.seek(0)
            with self.assertRaisesRegex(ValueError, 'supports dill >='):
                # weights_only=False as this is legacy code that saves the model
                x2 = torch.load(f, pickle_module=dill, encoding='utf-8', weights_only=False)

    def test_pickle_module(self):
        class ThrowingUnpickler(pickle.Unpickler):
            def load(self, *args, **kwargs):
                raise RuntimeError("rumpelstiltskin")

        class ThrowingModule:
            Unpickler = ThrowingUnpickler
            load = ThrowingUnpickler.load

        x = torch.eye(3)
        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f)
            f.seek(0)
            with self.assertRaisesRegex(RuntimeError, "rumpelstiltskin"):
                # weights_only=False as True does not support custom pickle module
                torch.load(f, pickle_module=ThrowingModule, weights_only=False)
            f.seek(0)
            z = torch.load(f)
        self.assertEqual(x, z)

    @unittest.skipIf(
        not TEST_DILL or not HAS_DILL_AT_LEAST_0_3_1,
        '"dill" not found or not correct version'
    )
    @skipIfTorchDynamo("Different behavior between 3.11 and 3.13, causing CI issues")
    def test_serialization_dill(self):
        x = torch.randn(5, 5)

        with tempfile.NamedTemporaryFile() as f:
            torch.save(x, f, pickle_module=dill)
            f.seek(0)
            # weights_only=False as True does not support custom pickle_module
            x2 = torch.load(f, pickle_module=dill, encoding='utf-8', weights_only=False)
            self.assertIsInstance(x2, type(x))
            self.assertEqual(x, x2)
            f.seek(0)
            # weights_only=False as True does not support custom pickle_module
            x3 = torch.load(f, pickle_module=dill, weights_only=False)
            self.assertIsInstance(x3, type(x))
            self.assertEqual(x, x3)

    def test_serialization_offset_gzip(self):
        a = torch.randn(5, 5)
        i = 41
        f2 = tempfile.NamedTemporaryFile(delete=False)
        with tempfile.NamedTemporaryFile() as f1:
            pickle.dump(i, f1)
            torch.save(a, f1)
            f1.seek(0)
            with gzip.open(f2.name, 'wb') as f_out:
                shutil.copyfileobj(f1, f_out)

            with gzip.open(f2.name, 'rb') as f:
                j = pickle.load(f)
                b = torch.load(f)
                self.assertTrue(torch.equal(a, b))
                self.assertEqual(i, j)

    def _test_serialization_sparse(self, weights_only):
        def _test_serialization(conversion):
            x = torch.zeros(3, 3)
            x[1][1] = 1
            x = conversion(x)
            with tempfile.NamedTemporaryFile() as f:
                torch.save({"tensor": x}, f)
                f.seek(0)
                y = torch.load(f, weights_only=weights_only)
                self.assertEqual(x, y["tensor"], exact_is_coalesced=True)
        _test_serialization(lambda x: x.to_sparse())
        _test_serialization(lambda x: x.to_sparse_csr())
        _test_serialization(lambda x: x.to_sparse_csc())
        _test_serialization(lambda x: x.to_sparse_bsr((1, 1)))
        _test_serialization(lambda x: x.to_sparse_bsc((1, 1)))

    def test_serialization_sparse(self):
        self._test_serialization(False)

    def test_serialization_sparse_safe(self):
        self._test_serialization(True)

    @unittest.skipIf(True, "Temporary skip due to gh-153143")
    def test_serialization_sparse_invalid(self):
        x = torch.zeros(3, 3)
        x[1][1] = 1
        x = x.to_sparse()

        class TensorSerializationSpoofer:
            def __init__(self, tensor):
                self.tensor = tensor

            def __reduce_ex__(self, proto):
                invalid_indices = self.tensor._indices().clone()
                invalid_indices[0][0] = 3
                return (
                    torch._utils._rebuild_sparse_tensor,
                    (
                        self.tensor.layout,
                        (
                            invalid_indices,
                            self.tensor._values(),
                            self.tensor.size())))

        with tempfile.NamedTemporaryFile() as f:
            torch.save({"spoofed": TensorSerializationSpoofer(x)}, f)
            for weights_only in (False, True):
                f.seek(0)
                with torch.sparse.check_sparse_tensor_invariants(), self.assertRaisesRegex(
                        RuntimeError,
                        "size is inconsistent with indices"):
                    y = torch.load(f, weights_only=weights_only)

    @unittest.skipIf(True, "Temporary skip due to gh-153143")
    def test_serialization_sparse_invalid_legacy_ctor(self):
        # This is set in test class setup but would not be check when running user code
        prev_invariant_check_enabled = torch.sparse.check_sparse_tensor_invariants.is_enabled()
        try:
            torch.sparse.check_sparse_tensor_invariants.disable()
            x = torch.zeros(3, 3)
            x[1][1] = 1
            x = x.to_sparse()
            x_legacy_ctor = torch.sparse.FloatTensor(x.indices(), x.values())

            # technically legacy ctor will still always be rebuilt with _rebuild_sparse_tensor
            # this is to test that legacy ctor in data.pkl will be validated by weights_only unpickler
            class LegacyCtorSerializationSpoofer:
                def __init__(self, tensor):
                    self.tensor = tensor

                def __reduce_ex__(self, proto):
                    indices = self.tensor._indices()
                    indices[0][0] = 3
                    return (torch.sparse.FloatTensor, (indices, self.tensor._values(), self.tensor.size()))

            with tempfile.NamedTemporaryFile() as f:
                sd = {"spoofed_legacy_ctor": LegacyCtorSerializationSpoofer(x_legacy_ctor)}
                torch.save(sd, f)
                for weights_only in (True,):
                    f.seek(0)
                    with torch.sparse.check_sparse_tensor_invariants(), self.assertRaisesRegex(
                            RuntimeError,
                            "size is inconsistent with indices|found negative index"):
                        y = torch.load(f, weights_only=weights_only)
        finally:
            if prev_invariant_check_enabled:
                torch.sparse.check_sparse_tensor_invariants.enable()

    @torch.sparse.check_sparse_tensor_invariants(enable=True)
    def _test_serialization_sparse_compressed_invalid(self,
                                                      conversion,
                                                      get_compressed_indices,
                                                      get_plain_indices):
        x = torch.zeros(3, 3)
        x[1][1] = 1
        x = conversion(x)

        class TensorSerializationSpoofer:
            def __init__(self, tensor):
                self.tensor = tensor

            def __reduce_ex__(self, proto):
                invalid_compressed_indices = get_compressed_indices(self.tensor).clone()
                invalid_compressed_indices[0] = 3
                return (
                    torch._utils._rebuild_sparse_tensor,
                    (
                        self.tensor.layout,
                        (
                            invalid_compressed_indices,
                            get_plain_indices(self.tensor),
                            self.tensor.values(),
                            self.tensor.size())))

        if x.layout in {torch.sparse_csr, torch.sparse_bsr}:
            compressed_indices_name = 'crow_indices'
        else:
            compressed_indices_name = 'ccol_indices'

        with tempfile.NamedTemporaryFile() as f:
            torch.save({"spoofed": TensorSerializationSpoofer(x)}, f)
            f.seek(0)
            with self.assertRaisesRegex(
                    RuntimeError,
                    f"`{compressed_indices_name}[[]..., 0[]] == 0` is not satisfied."):
                y = torch.load(f)

    @unittest.skipIf(True, "Temporary skip due to gh-153143")
    def test_serialization_sparse_csr_invalid(self):
        self._test_serialization_sparse_compressed_invalid(
            torch.Tensor.to_sparse_csr, torch.Tensor.crow_indices, torch.Tensor.col_indices)

    @unittest.skipIf(True, "Temporary skip due to gh-153143")
    def test_serialization_sparse_csc_invalid(self):
        self._test_serialization_sparse_compressed_invalid(
            torch.Tensor.to_sparse_csc, torch.Tensor.ccol_indices, torch.Tensor.row_indices)

    @unittest.skipIf(True, "Temporary skip due to gh-153143")
    def test_serialization_sparse_bsr_invalid(self):
        self._test_serialization_sparse_compressed_invalid(
            lambda x: x.to_sparse_bsr((1, 1)), torch.Tensor.crow_indices, torch.Tensor.col_indices)

    @unittest.skipIf(True, "Temporary skip due to gh-153143")
    def test_serialization_sparse_bsc_invalid(self):
        self._test_serialization_sparse_compressed_invalid(
            lambda x: x.to_sparse_bsc((1, 1)), torch.Tensor.ccol_indices, torch.Tensor.row_indices)

    def test_serialize_device(self):
        device_str = ['cpu', 'cpu:0', 'cuda', 'cuda:0']
        device_obj = [torch.device(d) for d in device_str]
        for device in device_obj:
            device_copied = copy.deepcopy(device)
            self.assertEqual(device, device_copied)

    def _test_serialization_backwards_compat(self, weights_only):
        a = [torch.arange(1 + i, 26 + i).view(5, 5).float() for i in range(2)]
        b = [a[i % 2] for i in range(4)]
        b += [a[0].storage()]
        b += [a[0].reshape(-1)[1:4].clone().storage()]
        path = download_file('https://download.pytorch.org/test_data/legacy_serialized.pt')
        if weights_only:
            with self.assertRaisesRegex(RuntimeError,
                                        "Cannot use ``weights_only=True`` with files saved in the legacy .tar format."):
                c = torch.load(path, weights_only=weights_only)
        c = torch.load(path, weights_only=False)
        self.assertEqual(b, c, atol=0, rtol=0)
        self.assertTrue(isinstance(c[0], torch.FloatTensor))
        self.assertTrue(isinstance(c[1], torch.FloatTensor))
        self.assertTrue(isinstance(c[2], torch.FloatTensor))
        self.assertTrue(isinstance(c[3], torch.FloatTensor))
        self.assertTrue(isinstance(c[4], torch.storage.TypedStorage))
        self.assertEqual(c[4].dtype, torch.float32)
        c[0].fill_(10)
        self.assertEqual(c[0], c[2], atol=0, rtol=0)
        self.assertEqual(c[4], torch.FloatStorage(25).fill_(10), atol=0, rtol=0)
        c[1].fill_(20)
        self.assertEqual(c[1], c[3], atol=0, rtol=0)

        # test some old tensor serialization mechanism
        class OldTensorBase:
            def __init__(self, new_tensor):
                self.new_tensor = new_tensor

            def __getstate__(self):
                return (self.new_tensor.storage(),
                        self.new_tensor.storage_offset(),
                        tuple(self.new_tensor.size()),
                        self.new_tensor.stride())

        class OldTensorV1(OldTensorBase):
            def __reduce__(self):
                return (torch.Tensor, (), self.__getstate__())

        class OldTensorV2(OldTensorBase):
            def __reduce__(self):
                return (_rebuild_tensor, self.__getstate__())

        x = torch.randn(30).as_strided([2, 3], [9, 3], 2)
        for old_cls in [OldTensorV1, OldTensorV2]:
            with tempfile.NamedTemporaryFile() as f:
                old_x = old_cls(x)
                torch.save(old_x, f)
                f.seek(0)
                load_x = torch.load(f, weights_only=weights_only)
                self.assertEqual(x.storage(), load_x.storage())
                self.assertEqual(x.storage_offset(), load_x.storage_offset())
                self.assertEqual(x.size(), load_x.size())
                self.assertEqual(x.stride(), load_x.stride())

    def test_serialization_backwards_compat(self):
        self._test_serialization_backwards_compat(False)

    def test_serialization_backwards_compat_safe(self):
        self._test_serialization_backwards_compat(True)

    @skipIfTorchDynamo("graph breaks messages collide with warnings")
    def test_serialization_save_warnings(self):
        with warnings.catch_warnings(record=True) as warns:
            with tempfile.NamedTemporaryFile() as checkpoint:
                x = torch.save(torch.nn.Linear(2, 3), checkpoint)
                self.assertEqual(len(warns), 0)

    def test_serialization_map_location(self):
        test_file_path = download_file('https://download.pytorch.org/test_data/gpu_tensors.pt')

        def map_location(storage, loc):
            return storage

        def generate_map_locations(device_type):
            return [
                {'cuda:0': device_type + ':0'},
                device_type,
                device_type + ':0',
                torch.device(device_type),
                torch.device(device_type, 0)
            ]

        def load_bytes():
            with open(test_file_path, 'rb') as f:
                return io.BytesIO(f.read())

        fileobject_lambdas = [lambda: test_file_path, load_bytes]
        cpu_map_locations = [
            map_location,
            {'cuda:0': 'cpu'},
            'cpu',
            torch.device('cpu'),
        ]
        gpu_0_map_locations = generate_map_locations('cuda')
        gpu_last_map_locations = [
            f'cuda:{torch.cuda.device_count() - 1}',
        ]
        xpu_0_map_locations = generate_map_locations('xpu')
        xpu_last_map_locations = [
            f'xpu:{torch.xpu.device_count() - 1}',
        ]
        mtia_0_map_locations = generate_map_locations('mtia')
        mtia_last_map_locations = [
            f'mtia:{torch.mtia.device_count() - 1}',
        ]

        def check_map_locations(map_locations, dtype, intended_device):
            for fileobject_lambda in fileobject_lambdas:
                for map_location in map_locations:
                    tensor = torch.load(fileobject_lambda(), map_location=map_location)

                    self.assertEqual(tensor.device, intended_device)
                    self.assertEqual(tensor.dtype, dtype)
                    self.assertEqual(tensor, torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype, device=intended_device))

        check_map_locations(cpu_map_locations, torch.float, torch.device('cpu'))
        if torch.cuda.is_available():
            check_map_locations(gpu_0_map_locations, torch.float, torch.device('cuda', 0))
            check_map_locations(
                gpu_last_map_locations,
                torch.float,
                torch.device('cuda', torch.cuda.device_count() - 1)
            )
        if torch.xpu.is_available():
            check_map_locations(xpu_0_map_locations, torch.float, torch.device('xpu', 0))
            check_map_locations(
                xpu_last_map_locations,
                torch.float,
                torch.device('xpu', torch.xpu.device_count() - 1)
            )
        if torch.mtia.is_available():
            check_map_locations(mtia_0_map_locations, torch.float, torch.device('mtia', 0))
            check_map_locations(
                mtia_last_map_locations,
                torch.float,
                torch.device('mtia', torch.mtia.device_count() - 1)
            )

    @unittest.skipIf(torch.cuda.is_available(), "Testing torch.load on CPU-only machine")
    def test_load_nonexistent_device(self):
        # Setup: create a serialized file object with a 'cuda:0' restore location
        # The following was generated by saving a torch.randn(2, device='cuda') tensor.
        serialized = (b'\x80\x02\x8a\nl\xfc\x9cF\xf9 j\xa8P\x19.\x80\x02M\xe9'
                      b'\x03.\x80\x02}q\x00(X\x10\x00\x00\x00protocol_versionq'
                      b'\x01M\xe9\x03X\r\x00\x00\x00little_endianq\x02\x88X\n'
                      b'\x00\x00\x00type_sizesq\x03}q\x04(X\x05\x00\x00\x00shortq'
                      b'\x05K\x02X\x03\x00\x00\x00intq\x06K\x04X\x04\x00\x00\x00'
                      b'longq\x07K\x04uu.\x80\x02ctorch._utils\n_rebuild_tensor_v2'
                      b'\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\nFloatStorage'
                      b'\nq\x02X\x0e\x00\x00\x0094919395964320q\x03X\x06\x00\x00'
                      b'\x00cuda:0q\x04K\x02Ntq\x05QK\x00K\x02\x85q\x06K\x01\x85q'
                      b'\x07\x89Ntq\x08Rq\t.\x80\x02]q\x00X\x0e\x00\x00\x00'
                      b'94919395964320q\x01a.\x02\x00\x00\x00\x00\x00\x00\x00\xbb'
                      b'\x1f\x82\xbe\xea\x81\xd1>')

        buf = io.BytesIO(serialized)

        error_msg = r'Attempting to deserialize object on a CUDA device'
        with self.assertRaisesRegex(RuntimeError, error_msg):
            _ = torch.load(buf)

    def test_serialization_filelike_api_requirements(self):
        filemock = FilelikeMock(b'', has_readinto=False)
        tensor = torch.randn(3, 5)
        torch.save(tensor, filemock)
        expected_superset = {'write', 'flush'}
        self.assertTrue(expected_superset.issuperset(filemock.calls))

        # Reset between save and load
        filemock.seek(0)
        filemock.calls.clear()

        _ = torch.load(filemock)
        expected_superset = {'read', 'readline', 'seek', 'tell'}
        self.assertTrue(expected_superset.issuperset(filemock.calls))

    def _test_serialization_filelike(self, tensor, mock, desc):
        f = mock(b'')
        torch.save(tensor, f)
        f.seek(0)
        data = mock(f.read())

        msg = 'filelike serialization with {}'

        b = torch.load(data)
        self.assertTrue(torch.equal(tensor, b), msg.format(desc))

    def test_serialization_filelike_missing_attrs(self):
        # Test edge cases where filelike objects are missing attributes.
        # The Python io docs suggests that these attributes should really exist
        # and throw io.UnsupportedOperation, but that isn't always the case.
        mocks = [
            ('no readinto', lambda x: FilelikeMock(x)),
            ('has readinto', lambda x: FilelikeMock(x, has_readinto=True)),
            ('no fileno', lambda x: FilelikeMock(x, has_fileno=False)),
        ]

        to_serialize = torch.randn(3, 10)
        for desc, mock in mocks:
            self._test_serialization_filelike(to_serialize, mock, desc)

    def test_serialization_filelike_stress(self):
        a = torch.randn(11 * (2 ** 9) + 1, 5 * (2 ** 9))

        # This one should call python read multiple times
        self._test_serialization_filelike(a, lambda x: FilelikeMock(x, has_readinto=False),
                                          'read() stress test')
        self._test_serialization_filelike(a, lambda x: FilelikeMock(x, has_readinto=True),
                                          'readinto() stress test')

    def test_serialization_filelike_uses_readinto(self):
        # For maximum efficiency, when reading a file-like object,
        # ensure the C API calls readinto instead of read.
        a = torch.randn(5, 4)

        f = io.BytesIO()
        torch.save(a, f)
        f.seek(0)
        data = FilelikeMock(f.read(), has_readinto=True)

        b = torch.load(data)
        self.assertTrue(data.was_called('readinto'))

    def test_serialization_filelike_exceptions(self):
        # Try to serialize to buffers that does not have write method
        # Or have a malfrormed one, and make sure it does not cause an abort
        # See https://github.com/pytorch/pytorch/issues/87997
        x = torch.rand(10)
        with self.assertRaises(AttributeError):
            # Tries to serialize str into tensor
            torch.save('foo', x)
        x.write = "bar"
        x.flush = "baz"
        with self.assertRaises(TypeError):
            # Tries to serialize str into tensor with write property
            torch.save('foo', x)
        x.write = str.__add__
        x.flush = str.__mul__
        with self.assertRaises(TypeError):
            # Tries to serialize str into tensor with wrong callable write property
            torch.save('foo', x)
        s_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        s = torch.CharStorage(s_data)
        with self.assertRaises(AttributeError):
            # Tries to serialize list into CharStorage
            torch.save(s_data, s)
        x = torch.randint(10, (3, 3), dtype=torch.float).cpu().numpy()
        with self.assertRaises(AttributeError):
            # Tries to serialize ndarray into ndarray
            torch.save(x, x)


    def test_serialization_storage_slice(self):
        # Generated using:
        #
        # t = torch.zeros(2);
        # s1 = t.storage()[:1]
        # s2 = t.storage()[1:]
        # torch.save((s1, s2), 'foo.ser')
        #
        # with PyTorch 0.3.1
        serialized = (b'\x80\x02\x8a\nl\xfc\x9cF\xf9 j\xa8P\x19.\x80\x02M\xe9\x03'
                      b'.\x80\x02}q\x00(X\n\x00\x00\x00type_sizesq\x01}q\x02(X\x03'
                      b'\x00\x00\x00intq\x03K\x04X\x05\x00\x00\x00shortq\x04K\x02X'
                      b'\x04\x00\x00\x00longq\x05K\x04uX\x10\x00\x00\x00protocol_versionq'
                      b'\x06M\xe9\x03X\r\x00\x00\x00little_endianq\x07\x88u.\x80\x02'
                      b'(X\x07\x00\x00\x00storageq\x00ctorch\nFloatStorage\nq\x01X\x0e'
                      b'\x00\x00\x0094279043900432q\x02X\x03\x00\x00\x00cpuq\x03K\x02'
                      b'X\x0e\x00\x00\x0094279029750368q\x04K\x00K\x01\x87q\x05tq\x06'
                      b'Q(h\x00h\x01X\x0e\x00\x00\x0094279043900432q\x07h\x03K\x02X'
                      b'\x0e\x00\x00\x0094279029750432q\x08K\x01K\x01\x87q\ttq\nQ'
                      b'\x86q\x0b.\x80\x02]q\x00X\x0e\x00\x00\x0094279043900432q'
                      b'\x01a.\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
                      b'\x00\x00\x00\x00')

        buf = io.BytesIO(serialized)
        (s1, s2) = torch.load(buf)
        self.assertEqual(s1[0], 0)
        self.assertEqual(s2[0], 0)
        self.assertEqual(s1.data_ptr() + 4, s2.data_ptr())

    def test_load_unicode_error_msg(self):
        # This Pickle contains a Python 2 module with Unicode data and the
        # loading should fail if the user explicitly specifies ascii encoding!
        path = download_file('https://download.pytorch.org/test_data/legacy_conv2d.pt')
        # weights_only=False as this is legacy code that saves the model
        self.assertRaises(UnicodeDecodeError, lambda: torch.load(path, encoding='ascii', weights_only=False))

    def test_load_python2_unicode_module(self):
        # This Pickle contains some Unicode data!
        path = download_file('https://download.pytorch.org/test_data/legacy_conv2d.pt')
        with warnings.catch_warnings(record=True) as w:
            # weights_only=False as this is legacy code that saves the model
            self.assertIsNotNone(torch.load(path, weights_only=False))

    def test_load_error_msg(self):
        expected_err_msg = (".*You can only torch.load from a file that is seekable. " +
                            "Please pre-load the data into a buffer like io.BytesIO and " +
                            "try to load from it instead.")

        resource = FilelikeMock(data=b"data")
        delattr(resource, "tell")
        delattr(resource, "seek")
        with self.assertRaisesRegex(AttributeError, expected_err_msg):
            # weights_only=False as this is legacy code that saves the model
            torch.load(resource, weights_only=False)

    def test_save_different_dtype_unallocated(self):
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')

        def save_load_check(a, b):
            with io.BytesIO() as f:
                torch.save([a, b], f)
                f.seek(0)
                a_loaded, b_loaded = torch.load(f)
            self.assertEqual(a, a_loaded)
            self.assertEqual(b, b_loaded)

        for device, dtype in product(devices, all_types_and_complex_and(torch.half,
                                                                        torch.bfloat16, torch.bool)):
            a = torch.tensor([], dtype=dtype, device=device)

            for other_dtype in all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool):
                s = torch.TypedStorage(
                    wrap_storage=a.storage().untyped(),
                    dtype=other_dtype)
                save_load_check(a, s)
                save_load_check(a.storage(), s)
                b = torch.tensor([], dtype=other_dtype, device=device)
                save_load_check(a, b)

    def test_save_different_dtype_error(self):
        error_msg = r"Cannot save multiple tensors or storages that view the same data as different types"

        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')

        for device in devices:
            a = torch.randn(10, dtype=torch.complex128, device=device)
            f = io.BytesIO()

            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.save([a, a.imag], f)

            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.save([a.storage(), a.imag], f)

            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.save([a, a.imag.storage()], f)

            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.save([a.storage(), a.imag.storage()], f)

            a = torch.randn(10, device=device)
            s_bytes = torch.TypedStorage(
                wrap_storage=a.storage().untyped(),
                dtype=torch.uint8)

            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.save([a, s_bytes], f)

            with self.assertRaisesRegex(RuntimeError, error_msg):
                torch.save([a.storage(), s_bytes], f)

    def test_safe_load_basic_types(self):
        with tempfile.NamedTemporaryFile() as f:
            data = {"int": 123, "str": "world", "float": 3.14, "bool": False}
            torch.save(data, f)
            f.seek(0)
            loaded_data = torch.load(f, weights_only=True)
            self.assertEqual(data, loaded_data)

    @unittest.skipIf(not IS_CI, "only check debug var is set in CI")
    def test_debug_set_in_ci(self):
        # This test is to make sure that the serialization debug flag is set in CI
        self.assertTrue(os.environ.get("TORCH_SERIALIZATION_DEBUG", "0") == "1")

    def test_skip_data_load(self):
        t_device = "cuda" if torch.cuda.is_available() else "cpu"
        t_v2 = torch.randn(2, 3, device=t_device)
        tt = TwoTensor(torch.randn(2, device=t_device), torch.randn(2, device=t_device))

        sd = {'t_v2': t_v2, 'tt': tt}
        sd_zeroed = {
            't_v2': torch.zeros(2, 3, device=t_device),
            'tt': TwoTensor(torch.zeros(2, device=t_device), torch.zeros(2, device=t_device)),
        }

        with BytesIOContext() as f:
            torch.save(sd, f)
            f.seek(0)
            with safe_globals([TwoTensor]), skip_data():
                sd_loaded = torch.load(f)
            self.assertNotEqual(sd_loaded, sd)
            for k in sd_loaded:
                sd_loaded[k] = sd_loaded[k].zero_()
            self.assertEqual(sd_loaded, sd_zeroed)


class serialization_method:
    def __init__(self, use_zip):
        self.use_zip = use_zip
        self.torch_save = torch.save

    def __enter__(self, *args, **kwargs):
        def wrapper(*args, **kwargs):
            if '_use_new_zipfile_serialization' in kwargs:
                raise RuntimeError("Cannot set method manually")
            kwargs['_use_new_zipfile_serialization'] = self.use_zip
            return self.torch_save(*args, **kwargs)

        torch.save = wrapper

    def __exit__(self, *args, **kwargs):
        torch.save = self.torch_save

Point = namedtuple('Point', ['x', 'y'])

class ClassThatUsesBuildInstruction:
    def __init__(self, num):
        self.num = num

    def __reduce_ex__(self, proto):
        # Third item, state here will cause pickle to push a BUILD instruction
        return ClassThatUsesBuildInstruction, (self.num,), {'foo': 'bar'}

@dataclass
class ClassThatUsesBuildInstructionAllSlots:
    __slots__ = ["x", "y"]
    x: int
    y: int

@dataclass
class ClassThatUsesBuildInstructionSomeSlots(ClassThatUsesBuildInstructionAllSlots):
    x: int
    y: int
    c: str

class TestBothSerialization(TestCase):
    @parametrize("weights_only", (True, False))
    def test_serialization_new_format_old_format_compat(self, device, weights_only):
        x = [torch.ones(200, 200, device=device) for i in range(30)]

        def test(f_new, f_old):
            torch.save(x, f_new, _use_new_zipfile_serialization=True)
            f_new.seek(0)
            x_new_load = torch.load(f_new, weights_only=weights_only)
            self.assertEqual(x, x_new_load)

            torch.save(x, f_old, _use_new_zipfile_serialization=False)
            f_old.seek(0)
            x_old_load = torch.load(f_old, weights_only=weights_only)
            self.assertEqual(x_old_load, x_new_load)

        with AlwaysWarnTypedStorageRemoval(True), warnings.catch_warnings(record=True) as w:
            with tempfile.NamedTemporaryFile() as f_new, tempfile.NamedTemporaryFile() as f_old:
                test(f_new, f_old)
            self.assertTrue(len(w) == 0, msg=f"Expected no warnings but got {[str(x) for x in w]}")


class TestOldSerialization(TestCase, SerializationMixin):
    # unique_key is necessary because on Python 2.7, if a warning passed to
    # the warning module is the same, it is not raised again.
    def _test_serialization_container(self, unique_key, filecontext_lambda):

        tmpmodule_name = f'tmpmodule{unique_key}'

        def import_module(name, filename):
            import importlib.util
            spec = importlib.util.spec_from_file_location(name, filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[module.__name__] = module
            return module

        with filecontext_lambda() as checkpoint:
            fname = get_file_path_2(os.path.dirname(os.path.dirname(torch.__file__)), 'torch', 'testing',
                                    '_internal', 'data', 'network1.py')
            module = import_module(tmpmodule_name, fname)
            torch.save(module.Net(), checkpoint)

            # First check that the checkpoint can be loaded without warning about unsafe loads
            checkpoint.seek(0)
            with warnings.catch_warnings(record=True) as w:
                # weights_only=False as this is legacy code that saves the model
                loaded = torch.load(checkpoint, weights_only=False)
                self.assertTrue(isinstance(loaded, module.Net))
                if can_retrieve_source:
                    self.assertEqual(len(w), 0)

            # Replace the module with different source
            fname = get_file_path_2(os.path.dirname(os.path.dirname(torch.__file__)), 'torch', 'testing',
                                    '_internal', 'data', 'network2.py')
            module = import_module(tmpmodule_name, fname)
            checkpoint.seek(0)
            with warnings.catch_warnings(record=True) as w:
                # weights_only=False as this is legacy code that saves the model
                loaded = torch.load(checkpoint, weights_only=False)
                self.assertTrue(isinstance(loaded, module.Net))
                if can_retrieve_source:
                    self.assertEqual(len(w), 1)
                    self.assertEqual(w[0].category, SourceChangeWarning)

    def test_serialization_container(self):
        self._test_serialization_container('file', tempfile.NamedTemporaryFile)

    def test_serialization_container_filelike(self):
        self._test_serialization_container('filelike', BytesIOContext)

    def test_serialization_offset(self):
        a = torch.randn(5, 5)
        b = torch.randn(1024, 1024, 512, dtype=torch.float32)
        m = torch.nn.Conv2d(1, 1, (1, 3))
        i, j = 41, 43
        with tempfile.NamedTemporaryFile() as f:
            pickle.dump(i, f)
            torch.save(a, f)
            pickle.dump(j, f)
            torch.save(b, f)
            torch.save(m, f)
            self.assertTrue(f.tell() > 2 * 1024 * 1024 * 1024)
            f.seek(0)
            i_loaded = pickle.load(f)
            a_loaded = torch.load(f)
            j_loaded = pickle.load(f)
            b_loaded = torch.load(f)
            # weights_only=False as this is legacy code that saves the model
            m_loaded = torch.load(f, weights_only=False)
        self.assertTrue(torch.equal(a, a_loaded))
        self.assertTrue(torch.equal(b, b_loaded))
        self.assertTrue(m.kernel_size == m_loaded.kernel_size)
        self.assertEqual(i, i_loaded)
        self.assertEqual(j, j_loaded)

    @parametrize('weights_only', (True, False))
    def test_serialization_offset_filelike(self, weights_only):
        a = torch.randn(5, 5)
        b = torch.randn(1024, 1024, 512, dtype=torch.float32)
        i, j = 41, 43
        with BytesIOContext() as f:
            pickle.dump(i, f)
            torch.save(a, f)
            pickle.dump(j, f)
            torch.save(b, f)
            self.assertTrue(f.tell() > 2 * 1024 * 1024 * 1024)
            f.seek(0)
            i_loaded = pickle.load(f)
            a_loaded = torch.load(f, weights_only=weights_only)
            j_loaded = pickle.load(f)
            b_loaded = torch.load(f, weights_only=weights_only)
        self.assertTrue(torch.equal(a, a_loaded))
        self.assertTrue(torch.equal(b, b_loaded))
        self.assertEqual(i, i_loaded)
        self.assertEqual(j, j_loaded)

    def run(self, *args, **kwargs):
        with serialization_method(use_zip=False):
            return super().run(*args, **kwargs)


class TestSerialization(TestCase, SerializationMixin):
    @parametrize('weights_only', (True, False))
    def test_serialization_zipfile(self, weights_only):
        data = self._test_serialization_data()

        def test(name_or_buffer):
            torch.save(data, name_or_buffer)

            if hasattr(name_or_buffer, 'seek'):
                name_or_buffer.seek(0)

            result = torch.load(name_or_buffer, weights_only=weights_only)
            self.assertEqual(result, data)

        with tempfile.NamedTemporaryFile() as f:
            test(f)

        with TemporaryFileName() as fname:
            test(fname)

        if IS_FILESYSTEM_UTF8_ENCODING:
            with TemporaryDirectoryName(suffix='\u975eASCII\u30d1\u30b9') as dname:
                with TemporaryFileName(dir=dname) as fname:
                    test(fname)

        test(io.BytesIO())

    def test_serialization_zipfile_actually_jit(self):
        with tempfile.NamedTemporaryFile() as f:
            torch.jit.save(torch.jit.script(torch.nn.Linear(3, 4)), f)
            f.seek(0)
            with self.assertRaisesRegex(
                RuntimeError,
                re.escape("Cannot use ``weights_only=True`` with TorchScript archives passed to ``torch.load``")
            ):
                torch.load(f, weights_only=True)
            f.seek(0)
            torch.load(f, weights_only=False)

    # Ensure large zip64 serialization works properly
    @serialTest()
    def test_serialization_2gb_file(self):
        # Run GC to clear up as much memory as possible before running this test
        gc.collect()
        big_model = torch.nn.Conv2d(20000, 3200, kernel_size=3)

        with BytesIOContext() as f:
            torch.save(big_model.state_dict(), f)
            f.seek(0)
            state = torch.load(f)

    @serialTest()
    def test_serialization_4gb_file(self):
        '''
        This is a specially engineered testcase that would fail if the data_descriptor size
        had been incorrectly set as data_descriptor_size32 when it should be data_descriptor_size64
        '''
        # Run GC to clear up as much memory as possible before running this test
        gc.collect()
        big_model = torch.nn.ModuleList([torch.nn.Linear(1, int(1024 * 1024 * 1024) + 12, bias=False),
                                         torch.nn.Linear(1, 1, bias=False).to(torch.float8_e4m3fn),
                                         torch.nn.Linear(1, 2, bias=False).to(torch.float8_e4m3fn)])

        with BytesIOContext() as f:
            torch.save(big_model.state_dict(), f)
            f.seek(0)
            torch.load(f)

    @parametrize('weights_only', (True, False))
    def test_pathlike_serialization(self, weights_only):
        model = torch.nn.Conv2d(20, 3200, kernel_size=3)

        with TemporaryFileName() as fname:
            path = Path(fname)
            torch.save(model.state_dict(), path)
            torch.load(path, weights_only=weights_only)

    @parametrize('weights_only', (True, False))
    def test_meta_serialization(self, weights_only):
        big_model = torch.nn.Conv2d(20000, 320000, kernel_size=3, device='meta')

        with BytesIOContext() as f:
            torch.save(big_model.state_dict(), f)
            f.seek(0)
            state = torch.load(f, weights_only=weights_only)

        self.assertEqual(state['weight'].size(), big_model.weight.size())

    def test_lr_scheduler_serialization(self):
        sgd = torch.optim.SGD([
            torch.tensor(torch.randn(100, 100, 2000), requires_grad=True)
        ], lr=0.1, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(sgd, 6.0, total_steps=10)

        with BytesIOContext() as f:
            torch.save(lr_scheduler.state_dict(), f)
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(0)
            lr_scheduler_state = torch.load(f)

        self.assertEqual(lr_scheduler_state['base_lrs'], lr_scheduler.b
```



## High-Level Overview


This Python file contains 40 class(es) and 176 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `FilelikeMock`, `ClassAMock`, `Nested`, `ClassBMock`, `Nested`, `UInt4Tensor`, `Int4Tensor`, `SerializationMixin`, `ThrowingUnpickler`, `ThrowingModule`, `TensorSerializationSpoofer`, `LegacyCtorSerializationSpoofer`, `TensorSerializationSpoofer`, `OldTensorBase`, `OldTensorV1`, `OldTensorV2`, `serialization_method`, `ClassThatUsesBuildInstruction`, `ClassThatUsesBuildInstructionAllSlots`, `ClassThatUsesBuildInstructionSomeSlots`

**Functions defined**: `__init__`, `trace`, `result`, `fileno_opt`, `readinto_opt`, `was_called`, `up_size`, `__new__`, `__init__`, `__torch_dispatch__`, `__new__`, `__init__`, `__torch_dispatch__`, `_test_serialization_data`, `_test_serialization_assert`, `test_serialization_zipfile_utils`, `test`, `_test_serialization`, `test_serialization`, `test_serialization_safe`

**Key imports**: contextlib, copy, functools, gc, gzip, io, os, pathlib, pickle, platform


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `contextlib`
- `copy`
- `functools`
- `gc`
- `gzip`
- `io`
- `os`
- `pathlib`
- `pickle`
- `platform`
- `re`
- `shutil`
- `sys`
- `tempfile`
- `unittest`
- `warnings`
- `zipfile`
- `collections`: namedtuple, OrderedDict
- `dataclasses`: dataclass
- `itertools`: product
- `unittest.mock`: patch
- `torch`
- `torch.utils.serialization`: config as serialization_config
- `torch._subclasses.fake_tensor`: FakeTensorMode, FakeTensorConverter
- `torch._utils`: _rebuild_tensor
- `torch._utils_internal`: get_file_path_2
- `torch.testing._internal.common_device_type`: instantiate_device_type_tests
- `torch.testing._internal.common_dtype`: all_types_and_complex_and


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Context Manager**: Implements context manager protocol
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

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
python test/test_serialization.py
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

- **File Documentation**: `test_serialization.py_docs.md`
- **Keyword Index**: `test_serialization.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
