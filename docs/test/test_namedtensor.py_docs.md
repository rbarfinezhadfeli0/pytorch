# Documentation: `test/test_namedtensor.py`

## File Metadata

- **Path**: `test/test_namedtensor.py`
- **Size**: 84,329 bytes (82.35 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# Owner(s): ["module: named tensor"]
# ruff: noqa: F841
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_NUMPY
from torch.testing._internal.common_utils import skipIfTorchDynamo
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import get_all_device_types
from collections import namedtuple, OrderedDict
import itertools
import functools
import torch
from torch import Tensor
import torch.nn.functional as F
from multiprocessing.reduction import ForkingPickler
import pickle
import io
import sys
import warnings


def pass_name_to_python_arg_parser(name):
    x = torch.empty(2, names=(name,))


def flatten(lst):
    return [item for sublist in lst for item in sublist]


Function = namedtuple('TestCase', ['name', 'lambd'])


def parse_compressed_namedshape(string):
    # This is a metalanguage for describing a shape of a tensor compactly.
    # 'N:3,C:2' -> size = [3, 2], names: ['N', 'C']
    # 'None:3,None:2' -> size = [3, 2], names: ['None', 'None']
    # '3,2' -> size = [3, 2], names=None passed to ctor.
    def parse_name(maybe_name):
        maybe_name = maybe_name.strip()
        if maybe_name == 'None':
            return None
        return maybe_name

    string = string.strip()

    # '' -> size: [], names:None
    if len(string) == 0:
        return None, []

    # '3, 2' -> size = [3, 2], None names.
    if ':' not in string:
        return None, [int(size) for size in string.split(',')]

    dims = string.split(',')
    tuples = [dim.split(':') for dim in dims]
    return zip(*[(parse_name(name), int(size)) for name, size in tuples])


def create(namedshape, factory=torch.randn):
    # namedshape: str
    names, shape = parse_compressed_namedshape(namedshape)
    return factory(shape, names=names)


def out_fn(operator):
    @functools.wraps(operator)
    def fn(*inputs):
        return operator(*inputs[1:], out=inputs[0])
    return fn


class TestNamedTensor(TestCase):
    def test_aaa_must_run_first_check_experimental_warning(self):
        # TODO(rzou): It would be nice for this to be a "real" python warning.
        # Right now this error message only prints once and doesn't respect
        # warnings.simplefilter behavior (where python users can control whether
        # or not to display warnings once, all the time, or never).
        with warnings.catch_warnings(record=True) as warns:
            x = torch.randn(3, 3, names=('N', 'C'))
            self.assertEqual(len(warns), 1)
            self.assertTrue(str(warns[0].message).startswith(
                'Named tensors and all their associated APIs are an experimental feature'))

    def test_trivial(self):
        pass

    def _test_name_inference(self, op, args=(), expected_names=(), device='cpu',
                             maybe_raises_regex=None):
        casted_args = [arg.to(device) if isinstance(arg, torch.Tensor) else arg
                       for arg in args]
        if maybe_raises_regex is not None:
            with self.assertRaisesRegex(RuntimeError, maybe_raises_regex):
                result = op(*args)
            return
        result = op(*args)
        self.assertEqual(result.names, expected_names,
                         msg=f'Name inference for {op.__name__} on device {device} failed')

    # TODO(rzou): Some form of this check should be added to self.assertEqual.
    # Right now I don't know what it should look like.
    def assertTensorDataAndNamesEqual(self, x, y):
        self.assertEqual(x.names, y.names)
        unnamed_x = x.rename(None)
        unnamed_y = y.rename(None)
        self.assertEqual(unnamed_x, unnamed_y)

    def _test_factory(self, factory, device):
        x = factory([], device=device)
        self.assertEqual(x.names, ())

        x = factory(1, 2, 3, device=device)
        self.assertEqual(x.names, (None, None, None))

        x = factory(1, 2, 3, names=None, device=device)
        self.assertEqual(x.names, (None, None, None))

        x = factory(1, 2, 3, names=('N', 'T', 'D'), device=device)
        self.assertEqual(x.names, ('N', 'T', 'D'))

        x = factory(1, 2, 3, names=('N', None, 'D'), device=device)
        self.assertEqual(x.names, ('N', None, 'D'))

        x = factory(1, 2, 3, names=('_1', 'batch9', 'BATCH_5'), device=device)
        self.assertEqual(x.names, ('_1', 'batch9', 'BATCH_5'))

        with self.assertRaisesRegex(RuntimeError,
                                    'a valid identifier contains only'):
            x = factory(2, names=('1',), device=device)

        with self.assertRaisesRegex(RuntimeError,
                                    'a valid identifier contains only'):
            x = factory(2, names=('?',), device=device)

        with self.assertRaisesRegex(RuntimeError, 'Number of names'):
            x = factory(2, 1, names=('N',), device=device)

        with self.assertRaisesRegex(TypeError, 'invalid combination of arguments'):
            x = factory(2, 1, names='N', device=device)

        with self.assertRaisesRegex(RuntimeError, 'construct a tensor with duplicate names'):
            x = factory(2, 1, 1, names=('N', 'C', 'N'), device=device)

        names64 = ['A' * i for i in range(1, 65)]
        x = factory([1] * 64, names=names64, device=device)
        self.assertEqual(x.names, names64)

        with self.assertRaisesRegex(
                RuntimeError,
                'only support up to 64 dims'):
            names65 = ['A' * i for i in range(1, 66)]
            x = factory([1] * 65, names=names64, device=device)

    @skipIfTorchDynamo("not a bug: Dynamo causes the refcounts to be different")
    def test_none_names_refcount(self, N=10):
        def scope():
            unnamed = torch.empty(2, 3)
            unnamed.names  # materialize [None, None]

        prev_none_refcnt = sys.getrefcount(None)
        # Ran it N times to reduce flakiness
        [scope() for i in range(N)]
        after_none_refcnt = sys.getrefcount(None)
        self.assertTrue(after_none_refcnt - prev_none_refcnt < N / 2,
                        msg='Using tensor.names should not change '
                            'the refcount of Py_None')

    def test_has_names(self):
        unnamed = torch.empty(2, 3)
        none_named = torch.empty(2, 3, names=(None, None))
        partially_named = torch.empty(2, 3, names=('N', None))
        fully_named = torch.empty(2, 3, names=('N', 'C'))

        self.assertFalse(unnamed.has_names())
        self.assertFalse(none_named.has_names())
        self.assertTrue(partially_named.has_names())
        self.assertTrue(fully_named.has_names())

    def test_py3_ellipsis(self):
        tensor = torch.randn(2, 3, 5, 7)
        output = tensor.refine_names('N', ..., 'C')
        self.assertEqual(output.names, ['N', None, None, 'C'])

    def test_refine_names(self):
        # Unnamed tensor -> Unnamed tensor
        self._test_name_inference(Tensor.refine_names,
                                  [create('None:1,None:2,None:3'), 'N', 'C', 'H'],
                                  ['N', 'C', 'H'])

        # Named tensor -> Named tensor
        self._test_name_inference(Tensor.refine_names,
                                  [create('N:1,C:2,H:3'), 'N', 'C', 'H'],
                                  ['N', 'C', 'H'])

        # Partially named tensor -> named tensor
        self._test_name_inference(Tensor.refine_names,
                                  [create('None:1,C:2,None:3'), None, 'C', 'H'],
                                  [None, 'C', 'H'])

        # Too few names
        self._test_name_inference(Tensor.refine_names,
                                  [create('None:2,None:3'), 'N', 'C', 'H'],
                                  maybe_raises_regex="different number of dims")

        # Cannot change Tensor[D] to Tensor[N]
        self._test_name_inference(Tensor.refine_names,
                                  [create('D:3'), 'N'],
                                  maybe_raises_regex="is different from")

        # Cannot change Tensor[D] to Tensor[None]
        self._test_name_inference(Tensor.refine_names,
                                  [create('D:3'), None],
                                  maybe_raises_regex="'D' is more specific than None")

        # globbing behavior exists
        self._test_name_inference(Tensor.refine_names,
                                  [create('None:1,None:1,None:2,None:3'), '...', 'C', 'H'],
                                  [None, None, 'C', 'H'])

    def test_detach(self):
        names = ['N']
        self._test_name_inference(
            Tensor.detach_,
            [torch.randn(3, requires_grad=True, names=names)],
            names)
        self._test_name_inference(
            Tensor.detach,
            [torch.randn(3, requires_grad=True, names=names)],
            names)

    def test_index_fill(self):
        for device in get_all_device_types():
            expected_names = ('N', 'C')
            x = torch.randn(3, 5, device=device, names=expected_names)

            output = x.index_fill_('C', torch.tensor([0, 1], device=device), 5)
            self.assertEqual(output.names, expected_names)

            output = x.index_fill_('C', torch.tensor([0, 1], device=device), torch.tensor(4.))
            self.assertEqual(output.names, expected_names)

            output = x.index_fill('C', torch.tensor([0, 1], device=device), 5)
            self.assertEqual(output.names, expected_names)

            output = x.index_fill('C', torch.tensor([0, 1], device=device), torch.tensor(4.))
            self.assertEqual(output.names, expected_names)

    def test_equal(self):
        for device in get_all_device_types():
            tensor = torch.randn(2, 3, device=device)
            other = tensor.clone()

            self.assertTrue(torch.equal(tensor.rename('N', 'C'), other.rename('N', 'C')))
            self.assertFalse(torch.equal(tensor.rename('M', 'C'), other.rename('N', 'C')))
            self.assertFalse(torch.equal(tensor.rename(None, 'C'), other.rename('N', 'C')))

    def test_squeeze(self):
        x = create('N:3,C:1,H:1,W:1')
        output = x.squeeze('C')
        self.assertEqual(output.names, ['N', 'H', 'W'])

        output = x.squeeze()
        self.assertEqual(output.names, ['N'])

    def test_repr(self):
        named_tensor = torch.zeros(2, 3).rename_('N', 'C')
        expected = "tensor([[0., 0., 0.],\n        [0., 0., 0.]], names=('N', 'C'))"
        self.assertEqual(repr(named_tensor), expected)

        unnamed_tensor = torch.zeros(2, 3)
        expected = "tensor([[0., 0., 0.],\n        [0., 0., 0.]])"
        self.assertEqual(repr(unnamed_tensor), expected)

        none_named_tensor = torch.zeros(2, 3).rename_(None, None)
        self.assertEqual(repr(none_named_tensor), expected)

    def test_diagonal(self):
        named_tensor = torch.zeros(2, 3, 5, 7, names=list('ABCD'))
        self.assertEqual(named_tensor.diagonal().names, ['C', 'D', None])
        self.assertEqual(named_tensor.diagonal(1, 3).names, ['A', 'C', None])

        self.assertEqual(named_tensor.diagonal(outdim='E', dim1='B', dim2='D').names,
                         ['A', 'C', 'E'])

    def test_empty_names(self):
        ref_tensor = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        empty_named_tensor = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], names=[])
        self.assertEqual(ref_tensor, empty_named_tensor)

    def test_max_pooling(self):
        def check_tuple_return(op, inputs, expected_names):
            values, indices = op(*inputs)
            self.assertEqual(values.names, expected_names)
            self.assertEqual(indices.names, expected_names)

        for device in get_all_device_types():

            named_tensor_1d = torch.zeros(2, 3, 5, device=device, names=list('ABC'))
            named_tensor_2d = torch.zeros(2, 3, 5, 7, device=device, names=list('ABCD'))
            named_tensor_3d = torch.zeros(2, 3, 5, 7, 9, device=device, names=list('ABCDE'))

            self.assertEqual(F.max_pool1d(named_tensor_1d, 2).names, named_tensor_1d.names)
            self.assertEqual(F.max_pool2d(named_tensor_2d, [2, 2]).names, named_tensor_2d.names)
            self.assertEqual(F.max_pool3d(named_tensor_3d, [2, 2, 2]).names, named_tensor_3d.names)

            check_tuple_return(F.max_pool1d_with_indices, [named_tensor_1d, 2], named_tensor_1d.names)
            check_tuple_return(F.max_pool2d_with_indices, [named_tensor_2d, [2, 2]], named_tensor_2d.names)
            check_tuple_return(F.max_pool3d_with_indices, [named_tensor_3d, [2, 2, 2]], named_tensor_3d.names)

    def test_max_pooling_without_names_does_not_warn(self):
        for device in get_all_device_types():
            tensor_2d = torch.zeros(2, 3, 5, 7, device=device, requires_grad=True)
            with warnings.catch_warnings(record=True) as warns:
                warnings.simplefilter("always")
                result = F.max_pool2d(tensor_2d, [2, 2])
                result.sum().backward()
                self.assertEqual(len(warns), 0)

    def test_no_save_support(self):
        named_tensor = torch.zeros(2, 3, names=('N', 'C'))
        buf = io.BytesIO()
        with self.assertRaisesRegex(RuntimeError, "NYI"):
            torch.save(named_tensor, buf)

    def test_no_pickle_support(self):
        named_tensor = torch.zeros(2, 3, names=('N', 'C'))
        with self.assertRaisesRegex(RuntimeError, "NYI"):
            serialized = pickle.dumps(named_tensor)

    def test_no_multiprocessing_support(self):
        named_tensor = torch.zeros(2, 3, names=('N', 'C'))
        buf = io.BytesIO()
        with self.assertRaisesRegex(RuntimeError, "NYI"):
            ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(named_tensor)

    def test_big_tensor_repr_has_names(self):
        def check_repr(named_tensor):
            unnamed_tensor = named_tensor.rename(None)
            names_tag = f'names={named_tensor.names}'
            self.assertIn(names_tag, repr(named_tensor))

        check_repr(torch.randn(128, 3, 64, 64, names=('N', 'C', 'H', 'W')))

    def test_noncontig_contiguous(self):
        # This type of contiguous is special-cased and therefore needs its own test
        for device in get_all_device_types():
            x = torch.randn(2, 3, device=device).t().rename_('N', 'C')
            self.assertEqual(x.contiguous().names, ('N', 'C'))

    def test_copy_transpose(self):
        # This type of copy is special-cased and therefore needs its own test
        def _test(self_names, other_names, expected_names):
            x = torch.empty(2, 5, names=self_names)
            y = torch.empty(5, 2).t().rename_(*other_names)
            x.copy_(y)
            self.assertEqual(x.names, expected_names)

        _test(('N', 'C'), ('N', 'C'), ('N', 'C'))
        _test(None, ('N', 'C'), ('N', 'C'))

    def test_rename_(self):
        tensor = torch.empty(1, 1, names=('N', 'C'))
        self.assertEqual(tensor.rename_(None).names, (None, None))
        self.assertEqual(tensor.rename_('H', 'W').names, ('H', 'W'))
        with self.assertRaisesRegex(RuntimeError, 'Number of names'):
            tensor.rename_('N', 'C', 'W')
        with self.assertRaisesRegex(RuntimeError, 'duplicate names'):
            tensor.rename_('N', 'N')

    def test_rename(self):
        tensor = torch.empty(1, 1, names=('N', 'C'))

        self.assertEqual(tensor.rename(None).names, (None, None))
        self.assertEqual(tensor.rename('H', 'W').names, ('H', 'W'))

        # Check that we didn't modify tensor.names
        self.assertEqual(tensor.names, ('N', 'C'))

        with self.assertRaisesRegex(RuntimeError, 'Number of names'):
            tensor.rename('N', 'C', 'W')
        with self.assertRaisesRegex(RuntimeError, 'duplicate names'):
            tensor.rename('N', 'N')

        with self.assertRaisesRegex(RuntimeError, 'either positional args or keyword args'):
            tensor.rename(None, N='batch')

        # rename returns a view on the tensor
        self.assertEqual(tensor.rename('H', 'W').data_ptr(), tensor.data_ptr())
        self.assertEqual(tensor.rename(None).data_ptr(), tensor.data_ptr())

    def test_rename_globber(self):
        scalar = torch.randn([])
        unnamed_tensor = torch.empty(1, 1, 1, 1)
        named_tensor = torch.empty(1, 1, 1, 1, names=('N', 'C', 'H', 'W'))

        self.assertEqual(scalar.rename(None).names, [])
        self.assertEqual(scalar.rename('...').names, [])

        # Check that it works with unnamed tensors
        self.assertEqual(unnamed_tensor.rename('...').names, unnamed_tensor.names)
        self.assertEqual(unnamed_tensor.rename('...', 'H', 'W').names,
                         [None, None, 'H', 'W'])
        self.assertEqual(unnamed_tensor.rename('N', '...', 'W').names,
                         ['N', None, None, 'W'])
        self.assertEqual(unnamed_tensor.rename('N', 'C', '...').names,
                         ['N', 'C', None, None])

        # Check that it works with named tensors
        self.assertEqual(named_tensor.rename('...').names, named_tensor.names)
        self.assertEqual(named_tensor.rename('...', 'width').names,
                         ['N', 'C', 'H', 'width'])
        self.assertEqual(named_tensor.rename('batch', 'channels', '...', 'width').names,
                         ['batch', 'channels', 'H', 'width'])
        self.assertEqual(named_tensor.rename('batch', '...').names,
                         ['batch', 'C', 'H', 'W'])

        # Test empty glob
        self.assertEqual(unnamed_tensor.rename('...', None, None, None, None).names,
                         [None, None, None, None])
        self.assertEqual(named_tensor.rename('N', 'C', 'H', '...', 'W').names,
                         ['N', 'C', 'H', 'W'])

        # Multiple globs throw
        with self.assertRaisesRegex(RuntimeError, 'More than one '):
            named_tensor.rename('...', 'channels', '...')

    def test_rename_rename_map(self):
        scalar = torch.randn([])
        unnamed_tensor = torch.empty(1, 1, 1, 1)
        named_tensor = torch.empty(1, 1, 1, 1, names=('N', 'C', 'H', 'W'))

        with self.assertRaisesRegex(RuntimeError, "dim 'N' does not exist"):
            scalar.rename(N='batch')
        with self.assertRaisesRegex(RuntimeError, "dim 'N' does not exist"):
            unnamed_tensor.rename(N='batch')
        with self.assertRaisesRegex(RuntimeError, "dim 'B' does not exist"):
            named_tensor.rename(B='batch')
        with self.assertRaisesRegex(RuntimeError, "dim 'B' does not exist"):
            named_tensor.rename(H='height', B='batch')

        self.assertEqual(named_tensor.rename(N='batch').data_ptr(),
                         named_tensor.data_ptr())
        self.assertEqual(named_tensor.rename(N='batch').names,
                         ['batch', 'C', 'H', 'W'])
        self.assertEqual(named_tensor.rename(N='batch', H='height').names,
                         ['batch', 'C', 'height', 'W'])

    def test_set_names_property(self):
        tensor = torch.empty(1, 1, names=('N', 'C'))

        tensor.names = None
        self.assertEqual(tensor.names, (None, None))

        tensor.names = ('N', 'W')
        self.assertEqual(tensor.names, ('N', 'W'))

        with self.assertRaisesRegex(RuntimeError, 'Number of names'):
            tensor.names = ['N', 'C', 'W']
        with self.assertRaisesRegex(RuntimeError, 'duplicate names'):
            tensor.names = ['N', 'N']

    def test_factory_edge_cases(self):
        for device in get_all_device_types():
            self._test_factory(torch.empty, device)

    def test_factory_coverage(self):
        def _test(factory, device):
            names = ('N', 'T', 'D')

            torch.manual_seed(0)
            result = factory(1, 2, 3, names=names, device=device)

            torch.manual_seed(0)
            expected = factory(1, 2, 3, device=device).rename_(*names)

            self.assertTensorDataAndNamesEqual(result, expected)

        supported = [
            torch.ones,
            torch.rand,
            torch.randn,
            torch.zeros,
        ]

        for op, device in itertools.product(supported, get_all_device_types()):
            _test(op, device)

        # Test torch.full
        for device in get_all_device_types():
            names = ('N', 'T', 'D')
            result = torch.full([1, 2, 3], 2., names=names, device=device)
            expected = torch.full([1, 2, 3], 2., device=device).rename_(*names)
            self.assertTensorDataAndNamesEqual(result, expected)

    def test_tensor_from_lists(self):
        names = ('N', 'C')
        tensor = torch.tensor([[1]], names=names)
        self.assertEqual(tensor.names, names)

        names = ('N',)
        tensor = torch.tensor([1], names=names)
        self.assertEqual(tensor.names, names)

        with self.assertRaisesRegex(RuntimeError, 'Number of names'):
            names = ('N', 'C')
            tensor = torch.tensor([1], names=names)

    @unittest.skipIf(not TEST_NUMPY, "no numpy")
    def test_tensor_from_numpy(self):
        import numpy as np
        arr = np.array([[1]])
        names = ('N', 'C')
        tensor = torch.tensor([[1]], names=names)
        self.assertEqual(tensor.names, names)

    def test_tensor_from_tensor(self):
        x = torch.randn(1, 1)
        names = ('N', 'C')
        tensor = torch.tensor(x, names=names)
        self.assertEqual(tensor.names, names)

    def test_tensor_from_named_tensor(self):
        x = torch.randn(1, 1, names=('N', 'D'))
        tensor = torch.tensor(x)
        self.assertEqual(tensor.names, ('N', 'D'))

        # there's no way to distinguish between names=None and not passing in names.
        # If the user passes in names=None they are asking for trouble.
        x = torch.randn(1, 1, names=('N', 'D'))
        tensor = torch.tensor(x, names=None)
        self.assertEqual(tensor.names, ('N', 'D'))

        x = torch.randn(1, 1, names=('N', 'D'))
        with self.assertRaisesRegex(RuntimeError, "Name mismatch"):
            tensor = torch.tensor(x, names=('N', 'C'))

    def test_size(self):
        t = torch.empty(2, 3, 5, names=('N', None, 'C'))
        self.assertEqual(t.size('N'), 2)
        self.assertEqual(t.size('C'), 5)
        with self.assertRaisesRegex(RuntimeError, 'Name \'channels\' not found in '):
            t.size('channels')
        with self.assertRaisesRegex(RuntimeError, 'Name \'N\' not found in '):
            torch.empty(2, 3, 4).size('N')

    def test_stride(self):
        t = torch.empty(2, 3, 5, names=('N', None, 'C'))
        self.assertEqual(t.stride('N'), 3 * 5)
        self.assertEqual(t.stride('C'), 1)
        with self.assertRaisesRegex(RuntimeError, 'Name \'channels\' not found in '):
            t.stride('channels')
        with self.assertRaisesRegex(RuntimeError, 'Name \'N\' not found in '):
            torch.empty(2, 3, 4).stride('N')

    def test_transpose_variants(self):
        t = torch.randn(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))
        self.assertEqual(t.transpose('N', 'C').names, ['C', 'N', 'H', 'W'])
        self.assertEqual(t.transpose(1, 3).names, ['N', 'W', 'H', 'C'])

        t = torch.randn(2, 3, names=('N', 'C'))
        self.assertEqual(t.t().names, ['C', 'N'])

    def test_resize(self):
        for device in get_all_device_types():
            named = torch.randn(2, names=('N',), device=device)
            named.resize_([2])
            self.assertEqual(named.names, ['N'])

            with self.assertRaisesRegex(RuntimeError, "Cannot resize named tensor"):
                named.resize_([3])

            other_named = torch.randn(2, names=('N',), device=device)
            named.resize_as_(other_named)
            self.assertEqual(other_named.names, ['N'])

            unnamed = torch.randn(2, device=device)
            with self.assertRaisesRegex(
                    RuntimeError, r'names .* are not the same as the computed output names'):
                named.resize_as_(unnamed)

            unnamed = torch.randn(1, device=device)
            unnamed.resize_as_(named)
            self.assertEqual(unnamed.names, ['N'])

    def test_cdist(self):
        for device in get_all_device_types():
            tensor = torch.randn(3, 1, 2, 7, names=('M', 'N', 'first_group', 'features'),
                                 device=device)
            other = torch.randn(5, 11, 7, names=('N', 'second_group', 'features'),
                                device=device)
            result = torch.cdist(tensor, other)
            self.assertEqual(result.names, ['M', 'N', 'first_group', 'second_group'])

    def test_info_smoke(self):
        # Smoke test for info functions / methods / attributes on named tensors.
        tensor = torch.empty(1, 1, names=('N', 'D'))

        tensor.device
        tensor.dtype
        tensor.get_device()
        tensor.is_complex()
        tensor.is_floating_point()
        tensor.is_nonzero()
        torch.is_same_size(tensor, tensor)
        torch.is_signed(tensor)
        tensor.layout
        tensor.numel()
        tensor.dim()
        tensor.element_size()
        tensor.is_contiguous()
        tensor.is_cuda
        tensor.is_leaf
        tensor.is_pinned()
        tensor.is_shared()
        tensor.is_sparse
        tensor.ndimension()
        tensor.nelement()
        tensor.shape
        tensor.size()
        tensor.size(1)
        tensor.storage()
        tensor.storage_offset()
        tensor.storage_type()
        tensor.stride()
        tensor.stride(1)
        tensor.data
        tensor.data_ptr()
        tensor.ndim
        tensor.item()
        tensor.type()
        tensor.is_shared()
        tensor.is_signed()

    def test_autograd_smoke(self):
        x = torch.randn(3, 3, names=('N', 'D'), requires_grad=True)

        y = x.clone()
        y.retain_grad()
        y.register_hook(lambda x: x)

        y.sum().backward()

        # autograd related attributes
        tensor = torch.empty(1, 1, names=('N', 'D'), requires_grad=True)
        tensor = tensor.relu()
        tensor.output_nr
        tensor.grad_fn
        tensor.requires_grad

    def test_split_fns_propagates_names(self):
        fns = [
            lambda x: x.split(1, 0),
            lambda x: x.split([1, 1], 1),
            lambda x: x.chunk(2, 0),
        ]

        for device in get_all_device_types():
            orig_tensor = torch.empty(2, 2, names=('N', 'D'), device=device)
            for fn in fns:
                splits = fn(orig_tensor)
                for split in splits:
                    self.assertEqual(split.names, orig_tensor.names)

    def test_any_all(self):
        for device in get_all_device_types():
            x = torch.zeros(3, dtype=torch.bool, device=device, names=('C',))
            self.assertEqual(x.any().names, [])
            self.assertEqual(x.all().names, [])

    def test_addcmul_addcdiv(self):
        for device in get_all_device_types():
            names = ['N']
            a = torch.rand(3, device=device, names=names)
            b = torch.rand(3, device=device, names=names)
            # avoid division by 0
            c = torch.rand(3, device=device, names=names).clamp_min_(0.1)
            out = torch.randn(3, device=device, names=names)

            self.assertEqual(torch.addcmul(a, b, c).names, names)
            self.assertEqual(torch.addcmul(a, b, c, out=out).names, names)
            self.assertEqual(a.addcmul_(b, c).names, names)

            self.assertEqual(torch.addcdiv(a, b, c).names, names)
            self.assertEqual(torch.addcdiv(a, b, c, out=out).names, names)
            self.assertEqual(a.addcdiv_(b, c).names, names)

    def test_binary_ops(self):
        def test_basic(op):
            a = torch.empty(2, 3, names=('N', 'C'))
            b = torch.empty(3, 2, names=('C', 'N'))
            c = torch.empty(3, names=('C',))
            d = torch.empty(5, names=('W',))

            self.assertEqual(op(a, a).names, ('N', 'C'))
            self.assertEqual(op(a, c).names, ('N', 'C'))
            # TODO: dynamo will throw a slightly different
            # error message because it's adding fake tensors
            # `must match the size of` portion is the dynamo error
            with self.assertRaisesRegex(RuntimeError, "do not match|must match the size of"):
                op(a, d)
            with self.assertRaisesRegex(RuntimeError, "do not match|must match the size of"):
                op(a, b)

        def test_wildcard(op):
            a = torch.empty(2, 3, names=('N', 'C'))
            c = torch.empty(2, 3, names=(None, 'C'))
            self.assertEqual(op(a, c).names, ('N', 'C'))

            b = torch.empty(2, 3)
            self.assertEqual(op(a, b).names, ('N', 'C'))

            d = torch.empty(2, 3, names=('C', None))
            with self.assertRaisesRegex(RuntimeError, "Misaligned"):
                op(d, c)

        def test_mixed_unnamed_named(op, is_inplace):
            named2 = torch.randn(1, 1, names=('N', 'C'))
            unnamed1 = torch.randn(1)
            unnamed2 = torch.randn(1, 1)
            unnamed3 = torch.randn(1, 1, 1)

            def compute_expected_names(tensor, other):
                assert tensor.has_names() ^ other.has_names()
                named = tensor if tensor.has_names() else other
                unnamed = other if tensor.has_names() else tensor
                unnamed_dim = unnamed.dim()
                if unnamed_dim > named.dim():
                    return [None] * (unnamed_dim - named.dim()) + list(named.names)
                else:
                    return named.names

            inputs = itertools.chain(
                itertools.product([named2], [unnamed1, unnamed2, unnamed3]),
                itertools.product([unnamed1, unnamed2, unnamed3], [named2]),
            )
            if is_inplace:
                # In-place ops have the constraint that they must not change shape.
                inputs = [(a, b) for (a, b) in inputs if a.dim() >= b.dim()]

            for tensor, other in inputs:
                expected_names = compute_expected_names(tensor, other)
                self.assertEqual(op(tensor, other).names, expected_names)

        def method(name, *args, **kwargs):
            return [Function(name, lambda a, b: getattr(a, name)(b, *args, **kwargs))]

        def function(name, *args, **kwargs):
            return [Function(name, lambda a, b: getattr(torch, name)(a, b, *args, **kwargs))]

        def out_function(name, *args, **kwargs):
            out_fn = getattr(torch, name)

            def fn(a, b):
                result = torch.empty([0], dtype=a.dtype, device=a.device)
                out_fn(a, b, *args, out=result, **kwargs)
                return result

            return [Function(name, fn)]

        def fn_method_and_inplace(name, *args, **kwargs):
            return (
                method(name, *args, **kwargs) +
                method(name + '_', *args, **kwargs) +
                out_function(name, *args, **kwargs)
            )

        tests = [
            fn_method_and_inplace('add'),
            fn_method_and_inplace('div'),
            fn_method_and_inplace('mul'),
            fn_method_and_inplace('sub'),
            fn_method_and_inplace('pow'),
            fn_method_and_inplace('atan2'),
            method('copy_'),
            function('floor_divide'),
            function('true_divide'),
        ]
        tests = flatten(tests)

        for name, op in tests:
            test_basic(op)
            test_wildcard(op)
            test_mixed_unnamed_named(op, is_inplace=name.endswith('_'))

    def test_logical_ops(self):
        # Implemented via TensorIterator, so just check that each version
        # (out-of-place, inplace, out=) propagates names.
        def zeros(*args, **kwargs):
            return torch.zeros(*args, dtype=torch.bool, **kwargs)

        for op in ('logical_xor', 'logical_and', 'logical_or'):
            self._test_name_inference(
                getattr(torch, op),
                (create('N:2,C:3', zeros), create('N:2,C:3', zeros)),
                expected_names=['N', 'C'])

            self._test_name_inference(
                getattr(Tensor, op + '_'),
                (create('N:2,C:3', zeros), create('N:2,C:3', zeros)),
                expected_names=['N', 'C'])

            self._test_name_inference(
                lambda out, x, y: getattr(torch, op)(x, y, out=out),
                (create('0', zeros), create('N:2,C:3', zeros), create('N:2,C:3', zeros)),
                expected_names=['N', 'C'])

    def test_pow_special(self):
        # There are a few pow cases that don't go through TensorIterator.
        # Test them here.
        for device in get_all_device_types():
            named = torch.randn(2, 3, names=('N', 'C'), device=device)
            unnamed = torch.randn([0], device=device)

            result = torch.pow(named, 0, out=unnamed.clone())
            self.assertEqual(result.names, named.names)

            result = torch.pow(named, 1, out=unnamed.clone())
            self.assertEqual(result.names, named.names)

            result = torch.pow(1, named, out=unnamed.clone())
            self.assertEqual(result.names, named.names)

    def test_out_fn_semantics(self):
        out_fn = torch.abs
        unnamed_tensor = torch.randn(3, 2)
        none_named_tensor = torch.randn(3, 2, names=(None, None))
        named_tensor = torch.randn(3, 2, names=('N', 'C'))
        partially_named_tensor = torch.randn(3, 2, names=('N', None))

        with self.assertRaisesRegex(RuntimeError, "Name mismatch"):
            out_fn(partially_named_tensor, out=named_tensor)
        with self.assertRaisesRegex(RuntimeError, "Name mismatch"):
            out_fn(named_tensor, out=partially_named_tensor)
        with self.assertRaisesRegex(RuntimeError, "Name mismatch"):
            out_fn(none_named_tensor, out=named_tensor)
        with self.assertRaisesRegex(RuntimeError, "Name mismatch"):
            out_fn(unnamed_tensor, out=named_tensor)

        output = torch.randn(3, 2)
        out_fn(unnamed_tensor, out=output)
        self.assertFalse(output.has_names())

        output = torch.randn(3, 2, names=(None, None))
        out_fn(named_tensor, out=output)
        self.assertEqual(output.names, named_tensor.names)

        output = torch.randn(3, 2)
        out_fn(named_tensor, out=output)
        self.assertEqual(output.names, named_tensor.names)

        output = torch.randn(3, 2, names=(None, None))
        out_fn(unnamed_tensor, out=output)
        self.assertFalse(output.has_names())

    def test_unary_propagate_names_fns(self):
        def _test(testcase, names=('N', 'D'), device='cpu'):
            sizes = [2] * len(names)
            tensor = torch.empty(sizes, names=names, device=device)
            try:
                out = testcase.lambd(tensor)
            except RuntimeError as err:
                # Get a better error message by catching the error and asserting.
                raise RuntimeError(f'{testcase.name}: {err}') from err
            self.assertEqual(out.names, tensor.names,
                             msg=testcase.name)

        def fn(name, *args, **kwargs):
            return [Function(name, lambda t: getattr(torch, name)(t, *args, **kwargs))]

        def method(name, *args, **kwargs):
            return [Function(name, lambda t: getattr(t, name)(*args, **kwargs))]

        def out_function(name, *args, **kwargs):
            out_fn = getattr(torch, name)

            def fn(tensor):
                result = torch.empty([0], dtype=tensor.dtype, device=tensor.device)
                out_fn(tensor, *args, out=result, **kwargs)
                return result

            return [Function(name + '_out', fn)]

        def fn_method_and_inplace(name, *args, **kwargs):
            return (
                method(name, *args, **kwargs) +
                method(name + '_', *args, **kwargs) +
                out_function(name, *args, **kwargs)
            )

        # All of these operate on 2x2 tensors.
        tests = [
            # unary pointwise
            fn_method_and_inplace('abs'),
            fn_method_and_inplace('acos'),
            fn_method_and_inplace('asin'),
            fn_method_and_inplace('atan'),
            fn_method_and_inplace('ceil'),
            fn_method_and_inplace('clamp', -1, 1),
            fn_method_and_inplace('clamp_min', -2),
            fn_method_and_inplace('clamp_max', 2),
            method('cauchy_'),
            method('clone'),
            method('contiguous'),
            fn_method_and_inplace('cos'),
            fn_method_and_inplace('cosh'),
            fn_method_and_inplace('digamma'),
            fn_method_and_inplace('erf'),
            fn_method_and_inplace('erfc'),
            fn_method_and_inplace('erfinv'),
            fn_method_and_inplace('exp'),
            fn_method_and_inplace('expm1'),
            method('exponential_'),
            fn_method_and_inplace('floor'),
            fn_method_and_inplace('frac'),
            method('geometric_', p=0.5),
            fn_method_and_inplace('lgamma'),
            fn_method_and_inplace('log'),
            fn_method_and_inplace('log10'),
            fn_method_and_inplace('log1p'),
            fn_method_and_inplace('log2'),
            method('log_normal_'),
            fn_method_and_inplace('neg'),
            method('normal_'),
            [Function('polygamma', lambda t: torch.polygamma(1, t))],
            method('polygamma_', 1),
            fn_method_and_inplace('reciprocal'),
            method('random_', 0, 1),
            method('random_', 1),
            method('random_'),
            method('relu_'),
            method('requires_grad_'),
            method('relu'),
            fn_method_and_inplace('round'),
            fn_method_and_inplace('rsqrt'),
            fn_method_and_inplace('sigmoid'),
            fn_method_and_inplace('sign'),
            fn_method_and_inplace('sin'),
            fn_method_and_inplace('sinh'),
            fn_method_and_inplace('sqrt'),
            fn_method_and_inplace('tan'),
            fn_method_and_inplace('tanh'),
            fn('threshold', 0, 1),
            fn('threshold_', 0, 1),
            out_function('threshold', 0, 1),
            fn_method_and_inplace('trunc'),
            method('uniform_'),
            method('zero_'),
            method('fill_', 1),
            method('fill_', torch.tensor(3.14)),

            # conversions
            method('to', dtype=torch.long),
            method('to', device='cpu'),
            method('to', torch.empty([])),
            method('bool'),
            method('byte'),
            method('char'),
            method('cpu'),
            method('double'),
            method('float'),
            method('long'),
            method('half'),
            method('int'),
            method('short'),
            method('type', dtype=torch.long),

            # cumsum and cumprod
            fn('cumsum', 0),
            fn('cumsum', 'D'),
            out_function('cumsum', 'D'),
            fn('cumprod', 0),
            fn('cumprod', 'D'),
            out_function('cumprod', 'D'),

            # views
            method('narrow', 0, 0, 1),

            # creation functions
            fn('empty_like'),
            fn('zeros_like'),
            fn('ones_like'),
            fn('full_like', 3.14),
            fn('rand_like'),
            fn('randn_like'),

            # bernoulli variants
            method('bernoulli_', 0.5),
            method('bernoulli_', torch.tensor(0.5)),

            method('softmax', dim=1),
            method('softmax', dim='D'),
            method('log_softmax', dim=1),
            method('log_softmax', dim='D'),

            [Function('F.dropout(inplace)', lambda t: F.dropout(t, p=0.5, inplace=True))],
            [Function('F.dropout(outplace)', lambda t: F.dropout(t, p=0.5, inplace=False))],
        ]
        tests = flatten(tests)

        for testcase, device in itertools.product(tests, get_all_device_types()):
            _test(testcase, device=device)

    def test_cummax_cummin(self):
        def test_ops(op):
            for device in get_all_device_types():
                names = ('N', 'D')
                tensor = torch.rand(2, 3, names=names, device=device)
                result = op(tensor, 0)
                self.assertEqual(result[0].names, names)
                self.assertEqual(result[1].names, names)
        test_ops(torch.cummax)
        test_ops(torch.cummin)

    def test_logcumsumexp(self):
        for device in get_all_device_types():
            names = ('N', 'D')
            tensor = torch.rand(2, 3, names=names, device=device)
            result = torch.logcumsumexp(tensor, 'D')
            self.assertEqual(result.names, names)

    def test_bitwise_not(self):
        for device in get_all_device_types():
            names = ('N', 'D')
            tensor = torch.zeros(2, 3, names=names, dtype=torch.bool, device=device)
            result = torch.empty(0, dtype=torch.bool, device=device)

            self.assertEqual(tensor.bitwise_not().names, names)
            self.assertEqual(torch.bitwise_not(tensor, out=result).names, names)
            self.assertEqual(tensor.bitwise_not_().names, names)

    def test_logical_not(self):
        for device in get_all_device_types():
            names = ('N', 'D')
            tensor = torch.zeros(2, 3, names=names, dtype=torch.bool, device=device)
            result = torch.empty(0, dtype=torch.bool, device=device)

            self.assertEqual(tensor.logical_not().names, names)
            self.assertEqual(torch.logical_not(tensor, out=result).names, names)
            self.assertEqual(tensor.logical_not_().names, names)

    def test_bernoulli(self):
        for device in get_all_device_types():
            names = ('N', 'D')
            tensor = torch.rand(2, 3, names=names, device=device)
            result = torch.empty(0, device=device)
            self.assertEqual(tensor.bernoulli().names, names)

            torch.bernoulli(tensor, out=result)
            self.assertEqual(result.names, names)

    def test_flatten(self):
        tensor = torch.randn(2, 3, 5, 7, 11, names=('N', 'C', 'D', 'H', 'W'))

        # basic
        out = tensor.flatten('D', 'W', 'features')
        self.assertEqual(out.names, ['N', 'C', 'features'])
        self.assertEqual(out.rename(None), tensor.rename(None).view(2, 3, -1))

        # int overload
        out = tensor.flatten(2, 4, 'features')
        self.assertEqual(out.names, ['N', 'C', 'features'])
        self.assertEqual(out.rename(None), tensor.rename(None).view(2, 3, -1))

        # list overload
        out = tensor.flatten(['D', 'H', 'W'], 'features')
        self.assertEqual(out.names, ['N', 'C', 'features'])
        self.assertEqual(out.rename(None), tensor.rename(None).view(2, 3, -1))

        # Non-contiguous flatten: N and H are not "adjacent" in memory.
        sentences = torch.randn(2, 3, 5, 7, names=('N', 'T', 'H', 'D'))
        sentences = sentences.transpose('T', 'H')
        out = sentences.flatten('N', 'H', 'N_H')
        self.assertEqual(out.names, ['N_H', 'T', 'D'])

        with self.assertRaisesRegex(RuntimeError, "Name 'L' not found in"):
            tensor.flatten(['D', 'L'], 'features')

        with self.assertRaisesRegex(RuntimeError, "must be consecutive in"):
            tensor.flatten(['D', 'W'], 'features')

        with self.assertRaisesRegex(RuntimeError, "must be consecutive in"):
            tensor.flatten(['H', 'D', 'W'], 'features')

    def test_flatten_nodims(self):
        tensor = torch.empty((2, 3))
        with self.assertRaisesRegex(RuntimeError, "cannot be empty"):
            tensor.flatten((), 'abcd')

    def test_flatten_index_error(self):
        tensor = torch.randn(1, 2)
        with self.assertRaisesRegex(IndexError,
                                    r"Dimension out of range \(expected to be in range of \[-2, 1\], but got 2\)"):
            tensor.flatten(0, 2)
        with self.assertRaisesRegex(IndexError,
                                    r"Dimension out of range \(expected to be in range of \[-2, 1\], but got 2\)"):
            tensor.flatten(0, 2, 'N')
        with self.assertRaisesRegex(RuntimeError,
                                    r"flatten\(\) has invalid args: start_dim cannot come after end_dim"):
            tensor.flatten(1, 0)
        with self.assertRaisesRegex(RuntimeError,
                                    r"flatten\(\) has invalid args: start_dim cannot come after end_dim"):
            tensor.flatten(1, 0, 'N')

    def test_unflatten(self):
        # test args: tensor, int, namedshape
        self.assertTrue(torch.equal(
            torch.ones(4, names=('A',)).unflatten('A', (('A', 2), ('B', 2))),
            torch.ones(2, 2, names=('A', 'B'))))
        self.assertTrue(torch.equal(
            torch.ones(4, names=('A',)).unflatten('A', [('A', 2), ('B', 2)]),
            torch.ones(2, 2, names=('A', 'B'))))
        self.assertTrue(torch.equal(
            torch.ones(4, names=('A',)).unflatten('A', (['A', 2], ['B', 2])),
            torch.ones(2, 2, names=('A', 'B'))))
        self.assertTrue(torch.equal(
            torch.ones(2, 10, names=('A', 'B')).unflatten('B', (['B1', -1],)),
            torch.ones(2, 10, names=('A', 'B1'))))
        self.assertTrue(torch.equal(
            torch.ones(2, 3 * 4 * 5 * 6, names=('A', 'B'))
                 .unflatten('B', (['B1', 3], ['B2', 4], ['B3', -1], ['B4', 6])),
            torch.ones(2, 3, 4, 5, 6, names=('A', 'B1', 'B2', 'B3', 'B4'))))
        self.assertTrue(torch.equal(
            torch.ones(2, 0, names=('A', 'B'))
                 .unflatten('B', (['B1', 3], ['B2', -1], ['B3', 4])),
            torch.ones(2, 3, 0, 4, names=('A', 'B1', 'B2', 'B3'))))

        # test args: namedtensor, str, namedshape
        self.assertTrue(torch.equal(
            torch.ones(2, 4, names=('A', 'B')).unflatten('B', (('B1', 2), ('B2', 2))),
            torch.ones(2, 2, 2, names=('A', 'B1', 'B2'))))

        # test invalid args: namedtensor, str, sizes
        with self.assertRaisesRegex(TypeError, r"unflatten\(\): argument 'dim' \(position 1\) must be int, not str"):
            torch.tensor([1], names=('A',)).unflatten('A', (1, 1))

        # test invalid args: namedtensor, int, sizes
        with self.assertRaisesRegex(RuntimeError, r"input is a named tensor but no names were given for unflattened sizes"):
            torch.tensor([1], names=("A",)).unflatten(0, (1, 1))

        with self.assertRaisesRegex(RuntimeError,
                                    r"Provided sizes \[3, -1\] don't multiply up to the "
                                    r"size of dim 1 \('B': 4\) in Tensor\['A', 'B'\]"):
            torch.ones(2, 4, names=('A', 'B')).unflatten('B', (('B1', 3), ('B2', -1)))

        with self.assertRaisesRegex(RuntimeError,
                                    r"the unspecified dimension size -1 can be any value and is ambiguous"):
            torch.ones(2, 0, names=('A', 'B')).unflatten('B', (('B1', 0), ('B2', -1)))

        tensor = torch.randn(7, 2 * 3 * 5, 11, names=('N', 'D', 'K'))

        # accepts OrderedDict
        out = tensor.unflatten('D', OrderedDict((('C', 2), ('H', 3), ('W', 5))))
        self.assertEqual(out.names, ('N', 'C', 'H', 'W', 'K'))
        self.assertEqual(out.shape, (7, 2, 3, 5, 11))

        # Unflatten left-most
        out = tensor.unflatten('N', (('N', 7), ('H', 1)))
        self.assertEqual(out.names, ('N', 'H', 'D', 'K'))
        self.assertEqual(out.shape, (7, 1, 2 * 3 * 5, 11))

        # Unflatten right-most
        out = tensor.unflatten('K', (('K', 11), ('H', 1)))
        self.assertEqual(out.names, ('N', 'D', 'K', 'H'))
        self.assertEqual(out.shape, (7, 2 * 3 * 5, 11, 1))

        with self.assertRaisesRegex(RuntimeError, "don't multiply up to"):
            tensor.unflatten('D', (('H', 3), ('W', 5)))

        with self.assertRaisesRegex(RuntimeError, 'sizes must be non-empty'):
            tensor.unflatten('D', None)

        with self.assertRaisesRegex(RuntimeError, 'non-empty'):
            tensor.unflatten('D', OrderedDict())

    def test_unsupported_op_error_msg(self):
        named = torch.randn(3, 3, names=('N', 'C'))
        with self.assertRaisesRegex(
                RuntimeError, r"pdist.+is not yet supported with named tensors"):
            torch.pdist(named)
        with self.assertRaisesRegex(
                RuntimeError, r"as_strided_.+is not yet supported with named tensors"):
            named.as_strided_((3, 3), (3, 1))

    def test_reduction_fns(self):
        def check_output(output, expected_names):
            if isinstance(output, torch.Tensor):
                self.assertEqual(output.names, expected_names)
                return
            for out in output:
                self.assertEqual(out.names, expected_names)

        def sum_all_outputs(output):
            if isinstance(output, torch.Tensor):
                return output.sum()
            result = 0
            for out in output:
                result = out + result
            return result.sum()

        def test_simple_reduce(op, device):
            t = torch.empty(2, 3, 5, names=('N', 'C', 'L'), device=device)
            check_output(op(t, 1), ['N', 'L'])
            check_output(op(t, -1), ['N', 'C'])
            check_output(op(t, 'C'), ['N', 'L'])
            ops_support_dim_none = [
                'sum',
                'mean',
                'std',
                'var',
                'std_mean',
                'var_mean',
                'nanmean',
                'nansum',
            ]
            if op.__name__ in ops_support_dim_none:
                check_output(op(t, None), [])
            else:
                with self.assertRaisesRegex(RuntimeError, 'Please look up dimensions by name'):
                    op(t, None)
            with self.assertRaisesRegex(RuntimeError, 'Name 
```



## High-Level Overview


This Python file contains 1 class(es) and 141 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestNamedTensor`

**Functions defined**: `pass_name_to_python_arg_parser`, `flatten`, `parse_compressed_namedshape`, `parse_name`, `create`, `out_fn`, `fn`, `test_aaa_must_run_first_check_experimental_warning`, `test_trivial`, `_test_name_inference`, `assertTensorDataAndNamesEqual`, `_test_factory`, `test_none_names_refcount`, `scope`, `test_has_names`, `test_py3_ellipsis`, `test_refine_names`, `test_detach`, `test_index_fill`, `test_equal`

**Key imports**: unittest, TestCase, run_tests, TEST_NUMPY, skipIfTorchDynamo, TEST_CUDA, get_all_device_types, namedtuple, OrderedDict, itertools, functools, torch, Tensor


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `torch.testing._internal.common_utils`: TestCase, run_tests, TEST_NUMPY
- `torch.testing._internal.common_cuda`: TEST_CUDA
- `torch.testing._internal.common_device_type`: get_all_device_types
- `collections`: namedtuple, OrderedDict
- `itertools`
- `functools`
- `torch`
- `torch.nn.functional as F`
- `multiprocessing.reduction`: ForkingPickler
- `pickle`
- `io`
- `sys`
- `warnings`
- `numpy as np`


## Code Patterns & Idioms

### Common Patterns

- **Abstract Base Classes**: Defines abstract interfaces
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

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/test_namedtensor.py
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

- **File Documentation**: `test_namedtensor.py_docs.md`
- **Keyword Index**: `test_namedtensor.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
