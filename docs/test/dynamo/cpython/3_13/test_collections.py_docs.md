# Documentation: `test/dynamo/cpython/3_13/test_collections.py`

## File Metadata

- **Path**: `test/dynamo/cpython/3_13/test_collections.py`
- **Size**: 99,317 bytes (96.99 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_collections.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

__TestCase = CPythonTestCase

# ======= END DYNAMO PATCH =======

"""Unit tests for collections.py."""

import array
import collections
import copy
import doctest
import inspect
import operator
import pickle
from random import choice, randrange
from itertools import product, chain, combinations
import string
import sys
from test import support
import types
import unittest

from collections import namedtuple, Counter, OrderedDict, _count_elements
from collections import UserDict, UserString, UserList
from collections import ChainMap
from collections import deque
from collections.abc import Awaitable, Coroutine
from collections.abc import AsyncIterator, AsyncIterable, AsyncGenerator
from collections.abc import Hashable, Iterable, Iterator, Generator, Reversible
from collections.abc import Sized, Container, Callable, Collection
from collections.abc import Set, MutableSet
from collections.abc import Mapping, MutableMapping, KeysView, ItemsView, ValuesView
from collections.abc import Sequence, MutableSequence
from collections.abc import ByteString, Buffer


class TestUserObjects(__TestCase):
    def _superset_test(self, a, b):
        self.assertGreaterEqual(
            set(dir(a)),
            set(dir(b)),
            '{a} should have all the methods of {b}'.format(
                a=a.__name__,
                b=b.__name__,
            ),
        )

    def _copy_test(self, obj):
        # Test internal copy
        obj_copy = obj.copy()
        self.assertIsNot(obj.data, obj_copy.data)
        self.assertEqual(obj.data, obj_copy.data)

        # Test copy.copy
        obj.test = [1234]  # Make sure instance vars are also copied.
        obj_copy = copy.copy(obj)
        self.assertIsNot(obj.data, obj_copy.data)
        self.assertEqual(obj.data, obj_copy.data)
        self.assertIs(obj.test, obj_copy.test)

    def test_str_protocol(self):
        self._superset_test(UserString, str)

    def test_list_protocol(self):
        self._superset_test(UserList, list)

    def test_dict_protocol(self):
        self._superset_test(UserDict, dict)

    def test_list_copy(self):
        obj = UserList()
        obj.append(123)
        self._copy_test(obj)

    def test_dict_copy(self):
        obj = UserDict()
        obj[123] = "abc"
        self._copy_test(obj)

    def test_dict_missing(self):
        with torch._dynamo.error_on_graph_break(False):
            class A(UserDict):
                def __missing__(self, key):
                    return 456
        self.assertEqual(A()[123], 456)
        # get() ignores __missing__ on dict
        self.assertIs(A().get(123), None)


################################################################################
### ChainMap (helper class for configparser and the string module)
################################################################################

class TestChainMap(__TestCase):

    def test_basics(self):
        c = ChainMap()
        c['a'] = 1
        c['b'] = 2
        d = c.new_child()
        d['b'] = 20
        d['c'] = 30
        self.assertEqual(d.maps, [{'b':20, 'c':30}, {'a':1, 'b':2}])  # check internal state
        self.assertEqual(d.items(), dict(a=1, b=20, c=30).items())    # check items/iter/getitem
        self.assertEqual(len(d), 3)                                   # check len
        for key in 'abc':                                             # check contains
            self.assertIn(key, d)
        for k, v in dict(a=1, b=20, c=30, z=100).items():             # check get
            self.assertEqual(d.get(k, 100), v)

        del d['b']                                                    # unmask a value
        self.assertEqual(d.maps, [{'c':30}, {'a':1, 'b':2}])          # check internal state
        self.assertEqual(d.items(), dict(a=1, b=2, c=30).items())     # check items/iter/getitem
        self.assertEqual(len(d), 3)                                   # check len
        for key in 'abc':                                             # check contains
            self.assertIn(key, d)
        for k, v in dict(a=1, b=2, c=30, z=100).items():              # check get
            self.assertEqual(d.get(k, 100), v)
        self.assertIn(repr(d), [                                      # check repr
            type(d).__name__ + "({'c': 30}, {'a': 1, 'b': 2})",
            type(d).__name__ + "({'c': 30}, {'b': 2, 'a': 1})"
        ])

        for e in d.copy(), copy.copy(d):                               # check shallow copies
            self.assertEqual(d, e)
            self.assertEqual(d.maps, e.maps)
            self.assertIsNot(d, e)
            self.assertIsNot(d.maps[0], e.maps[0])
            for m1, m2 in zip(d.maps[1:], e.maps[1:]):
                self.assertIs(m1, m2)

        # check deep copies
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            e = pickle.loads(pickle.dumps(d, proto))
            self.assertEqual(d, e)
            self.assertEqual(d.maps, e.maps)
            self.assertIsNot(d, e)
            for m1, m2 in zip(d.maps, e.maps):
                self.assertIsNot(m1, m2, e)
        for e in [copy.deepcopy(d),
                  eval(repr(d))
                ]:
            self.assertEqual(d, e)
            self.assertEqual(d.maps, e.maps)
            self.assertIsNot(d, e)
            for m1, m2 in zip(d.maps, e.maps):
                self.assertIsNot(m1, m2, e)

        f = d.new_child()
        f['b'] = 5
        self.assertEqual(f.maps, [{'b': 5}, {'c':30}, {'a':1, 'b':2}])
        self.assertEqual(f.parents.maps, [{'c':30}, {'a':1, 'b':2}])   # check parents
        self.assertEqual(f['b'], 5)                                    # find first in chain
        self.assertEqual(f.parents['b'], 2)                            # look beyond maps[0]

    def test_ordering(self):
        # Combined order matches a series of dict updates from last to first.
        # This test relies on the ordering of the underlying dicts.

        baseline = {'music': 'bach', 'art': 'rembrandt'}
        adjustments = {'art': 'van gogh', 'opera': 'carmen'}

        cm = ChainMap(adjustments, baseline)

        combined = baseline.copy()
        combined.update(adjustments)

        self.assertEqual(list(combined.items()), list(cm.items()))

    def test_constructor(self):
        self.assertEqual(ChainMap().maps, [{}])                        # no-args --> one new dict
        self.assertEqual(ChainMap({1:2}).maps, [{1:2}])                # 1 arg --> list

    def test_bool(self):
        self.assertFalse(ChainMap())
        self.assertFalse(ChainMap({}, {}))
        self.assertTrue(ChainMap({1:2}, {}))
        self.assertTrue(ChainMap({}, {1:2}))

    def test_missing(self):
        with torch._dynamo.error_on_graph_break(False):
            class DefaultChainMap(ChainMap):
                def __missing__(self, key):
                    return 999
        d = DefaultChainMap(dict(a=1, b=2), dict(b=20, c=30))
        for k, v in dict(a=1, b=2, c=30, d=999).items():
            self.assertEqual(d[k], v)                                  # check __getitem__ w/missing
        for k, v in dict(a=1, b=2, c=30, d=77).items():
            self.assertEqual(d.get(k, 77), v)                          # check get() w/ missing
        for k, v in dict(a=True, b=True, c=True, d=False).items():
            self.assertEqual(k in d, v)                                # check __contains__ w/missing
        self.assertEqual(d.pop('a', 1001), 1, d)
        self.assertEqual(d.pop('a', 1002), 1002)                       # check pop() w/missing
        self.assertEqual(d.popitem(), ('b', 2))                        # check popitem() w/missing
        with self.assertRaises(KeyError):
            d.popitem()

    def test_order_preservation(self):
        d = ChainMap(
                OrderedDict(j=0, h=88888),
                OrderedDict(),
                OrderedDict(i=9999, d=4444, c=3333),
                OrderedDict(f=666, b=222, g=777, c=333, h=888),
                OrderedDict(),
                OrderedDict(e=55, b=22),
                OrderedDict(a=1, b=2, c=3, d=4, e=5),
                OrderedDict(),
            )
        self.assertEqual(''.join(d), 'abcdefghij')
        self.assertEqual(list(d.items()),
            [('a', 1), ('b', 222), ('c', 3333), ('d', 4444),
             ('e', 55), ('f', 666), ('g', 777), ('h', 88888),
             ('i', 9999), ('j', 0)])

    def test_iter_not_calling_getitem_on_maps(self):
        with torch._dynamo.error_on_graph_break(False):
            class DictWithGetItem(UserDict):
                def __init__(self, *args, **kwds):
                    self.called = False
                    UserDict.__init__(self, *args, **kwds)
                def __getitem__(self, item):
                    self.called = True
                    UserDict.__getitem__(self, item)

        d = DictWithGetItem(a=1)
        c = ChainMap(d)
        d.called = False

        set(c)  # iterate over chain map
        self.assertFalse(d.called, '__getitem__ was called')

    def test_dict_coercion(self):
        d = ChainMap(dict(a=1, b=2), dict(b=20, c=30))
        self.assertEqual(dict(d), dict(a=1, b=2, c=30))
        self.assertEqual(dict(d.items()), dict(a=1, b=2, c=30))

    def test_new_child(self):
        'Tests for changes for issue #16613.'
        c = ChainMap()
        c['a'] = 1
        c['b'] = 2
        m = {'b':20, 'c': 30}
        d = c.new_child(m)
        self.assertEqual(d.maps, [{'b':20, 'c':30}, {'a':1, 'b':2}])  # check internal state
        self.assertIs(m, d.maps[0])

        # Use a different map than a dict
        with torch._dynamo.error_on_graph_break(False):
            class lowerdict(dict):
                def __getitem__(self, key):
                    if isinstance(key, str):
                        key = key.lower()
                    return dict.__getitem__(self, key)
                def __contains__(self, key):
                    if isinstance(key, str):
                        key = key.lower()
                    return dict.__contains__(self, key)

        c = ChainMap()
        c['a'] = 1
        c['b'] = 2
        m = lowerdict(b=20, c=30)
        d = c.new_child(m)
        self.assertIs(m, d.maps[0])
        for key in 'abc':                                             # check contains
            self.assertIn(key, d)
        for k, v in dict(a=1, B=20, C=30, z=100).items():             # check get
            self.assertEqual(d.get(k, 100), v)

        c = ChainMap({'a': 1, 'b': 2})
        d = c.new_child(b=20, c=30)
        self.assertEqual(d.maps, [{'b': 20, 'c': 30}, {'a': 1, 'b': 2}])

    def test_union_operators(self):
        cm1 = ChainMap(dict(a=1, b=2), dict(c=3, d=4))
        cm2 = ChainMap(dict(a=10, e=5), dict(b=20, d=4))
        cm3 = cm1.copy()
        d = dict(a=10, c=30)
        pairs = [('c', 3), ('p',0)]

        tmp = cm1 | cm2 # testing between chainmaps
        self.assertEqual(tmp.maps, [cm1.maps[0] | dict(cm2), *cm1.maps[1:]])
        cm1 |= cm2
        self.assertEqual(tmp, cm1)

        tmp = cm2 | d # testing between chainmap and mapping
        self.assertEqual(tmp.maps, [cm2.maps[0] | d, *cm2.maps[1:]])
        self.assertEqual((d | cm2).maps, [d | dict(cm2)])
        cm2 |= d
        self.assertEqual(tmp, cm2)

        # testing behavior between chainmap and iterable key-value pairs
        with self.assertRaises(TypeError):
            cm3 | pairs
        tmp = cm3.copy()
        cm3 |= pairs
        self.assertEqual(cm3.maps, [tmp.maps[0] | dict(pairs), *tmp.maps[1:]])

        # testing proper return types for ChainMap and it's subclasses
        class Subclass(ChainMap):
            pass

        class SubclassRor(ChainMap):
            def __ror__(self, other):
                return super().__ror__(other)

        tmp = ChainMap() | ChainMap()
        self.assertIs(type(tmp), ChainMap)
        self.assertIs(type(tmp.maps[0]), dict)
        tmp = ChainMap() | Subclass()
        self.assertIs(type(tmp), ChainMap)
        self.assertIs(type(tmp.maps[0]), dict)
        tmp = Subclass() | ChainMap()
        self.assertIs(type(tmp), Subclass)
        self.assertIs(type(tmp.maps[0]), dict)
        tmp = ChainMap() | SubclassRor()
        self.assertIs(type(tmp), SubclassRor)
        self.assertIs(type(tmp.maps[0]), dict)


################################################################################
### Named Tuples
################################################################################

TestNT = namedtuple('TestNT', 'x y z')    # type used for pickle tests

class TestNamedTuple(__TestCase):

    def test_factory(self):
        Point = namedtuple('Point', 'x y')
        self.assertEqual(Point.__name__, 'Point')
        self.assertEqual(Point.__slots__, ())
        self.assertEqual(Point.__module__, __name__)
        self.assertEqual(Point.__getitem__, tuple.__getitem__)
        self.assertEqual(Point._fields, ('x', 'y'))

        self.assertRaises(ValueError, namedtuple, 'abc%', 'efg ghi')       # type has non-alpha char
        self.assertRaises(ValueError, namedtuple, 'class', 'efg ghi')      # type has keyword
        self.assertRaises(ValueError, namedtuple, '9abc', 'efg ghi')       # type starts with digit

        self.assertRaises(ValueError, namedtuple, 'abc', 'efg g%hi')       # field with non-alpha char
        self.assertRaises(ValueError, namedtuple, 'abc', 'abc class')      # field has keyword
        self.assertRaises(ValueError, namedtuple, 'abc', '8efg 9ghi')      # field starts with digit
        self.assertRaises(ValueError, namedtuple, 'abc', '_efg ghi')       # field with leading underscore
        self.assertRaises(ValueError, namedtuple, 'abc', 'efg efg ghi')    # duplicate field

        namedtuple('Point0', 'x1 y2')   # Verify that numbers are allowed in names
        namedtuple('_', 'a b c')        # Test leading underscores in a typename

        nt = namedtuple('nt', 'the quick brown fox')                       # check unicode input
        self.assertNotIn("u'", repr(nt._fields))
        nt = namedtuple('nt', ('the', 'quick'))                           # check unicode input
        self.assertNotIn("u'", repr(nt._fields))

        self.assertRaises(TypeError, Point._make, [11])                     # catch too few args
        self.assertRaises(TypeError, Point._make, [11, 22, 33])             # catch too many args

    def test_defaults(self):
        Point = namedtuple('Point', 'x y', defaults=(10, 20))              # 2 defaults
        self.assertEqual(Point._field_defaults, {'x': 10, 'y': 20})
        self.assertEqual(Point(1, 2), (1, 2))
        self.assertEqual(Point(1), (1, 20))
        self.assertEqual(Point(), (10, 20))

        Point = namedtuple('Point', 'x y', defaults=(20,))                 # 1 default
        self.assertEqual(Point._field_defaults, {'y': 20})
        self.assertEqual(Point(1, 2), (1, 2))
        self.assertEqual(Point(1), (1, 20))

        Point = namedtuple('Point', 'x y', defaults=())                     # 0 defaults
        self.assertEqual(Point._field_defaults, {})
        self.assertEqual(Point(1, 2), (1, 2))
        with self.assertRaises(TypeError):
            Point(1)

        with self.assertRaises(TypeError):                                  # catch too few args
            Point()
        with self.assertRaises(TypeError):                                  # catch too many args
            Point(1, 2, 3)
        with self.assertRaises(TypeError):                                  # too many defaults
            Point = namedtuple('Point', 'x y', defaults=(10, 20, 30))
        with self.assertRaises(TypeError):                                  # non-iterable defaults
            Point = namedtuple('Point', 'x y', defaults=10)
        with self.assertRaises(TypeError):                                  # another non-iterable default
            Point = namedtuple('Point', 'x y', defaults=False)

        Point = namedtuple('Point', 'x y', defaults=None)                   # default is None
        self.assertEqual(Point._field_defaults, {})
        self.assertIsNone(Point.__new__.__defaults__, None)
        self.assertEqual(Point(10, 20), (10, 20))
        with self.assertRaises(TypeError):                                  # catch too few args
            Point(10)

        Point = namedtuple('Point', 'x y', defaults=[10, 20])               # allow non-tuple iterable
        self.assertEqual(Point._field_defaults, {'x': 10, 'y': 20})
        self.assertEqual(Point.__new__.__defaults__, (10, 20))
        self.assertEqual(Point(1, 2), (1, 2))
        self.assertEqual(Point(1), (1, 20))
        self.assertEqual(Point(), (10, 20))

        Point = namedtuple('Point', 'x y', defaults=iter([10, 20]))         # allow plain iterator
        self.assertEqual(Point._field_defaults, {'x': 10, 'y': 20})
        self.assertEqual(Point.__new__.__defaults__, (10, 20))
        self.assertEqual(Point(1, 2), (1, 2))
        self.assertEqual(Point(1), (1, 20))
        self.assertEqual(Point(), (10, 20))

    def test_readonly(self):
        Point = namedtuple('Point', 'x y')
        p = Point(11, 22)
        with self.assertRaises(AttributeError):
            p.x = 33
        with self.assertRaises(AttributeError):
            del p.x
        with self.assertRaises(TypeError):
            p[0] = 33
        with self.assertRaises(TypeError):
            del p[0]
        self.assertEqual(p.x, 11)
        self.assertEqual(p[0], 11)

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_factory_doc_attr(self):
        Point = namedtuple('Point', 'x y')
        self.assertEqual(Point.__doc__, 'Point(x, y)')
        Point.__doc__ = '2D point'
        self.assertEqual(Point.__doc__, '2D point')

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_field_doc(self):
        Point = namedtuple('Point', 'x y')
        self.assertEqual(Point.x.__doc__, 'Alias for field number 0')
        self.assertEqual(Point.y.__doc__, 'Alias for field number 1')
        Point.x.__doc__ = 'docstring for Point.x'
        self.assertEqual(Point.x.__doc__, 'docstring for Point.x')
        # namedtuple can mutate doc of descriptors independently
        Vector = namedtuple('Vector', 'x y')
        self.assertEqual(Vector.x.__doc__, 'Alias for field number 0')
        Vector.x.__doc__ = 'docstring for Vector.x'
        self.assertEqual(Vector.x.__doc__, 'docstring for Vector.x')

    @support.cpython_only
    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_field_doc_reuse(self):
        P = namedtuple('P', ['m', 'n'])
        Q = namedtuple('Q', ['o', 'p'])
        self.assertIs(P.m.__doc__, Q.o.__doc__)
        self.assertIs(P.n.__doc__, Q.p.__doc__)

    @support.cpython_only
    def test_field_repr(self):
        Point = namedtuple('Point', 'x y')
        self.assertEqual(repr(Point.x), "_tuplegetter(0, 'Alias for field number 0')")
        self.assertEqual(repr(Point.y), "_tuplegetter(1, 'Alias for field number 1')")

        Point.x.__doc__ = 'The x-coordinate'
        Point.y.__doc__ = 'The y-coordinate'

        self.assertEqual(repr(Point.x), "_tuplegetter(0, 'The x-coordinate')")
        self.assertEqual(repr(Point.y), "_tuplegetter(1, 'The y-coordinate')")

    def test_name_fixer(self):
        for spec, renamed in [
            [('efg', 'g%hi'),  ('efg', '_1')],                              # field with non-alpha char
            [('abc', 'class'), ('abc', '_1')],                              # field has keyword
            [('8efg', '9ghi'), ('_0', '_1')],                               # field starts with digit
            [('abc', '_efg'), ('abc', '_1')],                               # field with leading underscore
            [('abc', 'efg', 'efg', 'ghi'), ('abc', 'efg', '_2', 'ghi')],    # duplicate field
            [('abc', '', 'x'), ('abc', '_1', 'x')],                         # fieldname is a space
        ]:
            self.assertEqual(namedtuple('NT', spec, rename=True)._fields, renamed)

    def test_module_parameter(self):
        NT = namedtuple('NT', ['x', 'y'], module=collections)
        self.assertEqual(NT.__module__, collections)

    def test_instance(self):
        Point = namedtuple('Point', 'x y')
        p = Point(11, 22)
        self.assertEqual(p, Point(x=11, y=22))
        self.assertEqual(p, Point(11, y=22))
        self.assertEqual(p, Point(y=22, x=11))
        self.assertEqual(p, Point(*(11, 22)))
        self.assertEqual(p, Point(**dict(x=11, y=22)))
        self.assertRaises(TypeError, Point, 1)          # too few args
        self.assertRaises(TypeError, Point, 1, 2, 3)    # too many args
        with self.assertRaises(TypeError):              # wrong keyword argument
            Point(XXX=1, y=2)
        with self.assertRaises(TypeError):              # missing keyword argument
            Point(x=1)
        self.assertEqual(repr(p), 'Point(x=11, y=22)')
        self.assertNotIn('__weakref__', dir(p))
        self.assertEqual(p, Point._make([11, 22]))      # test _make classmethod
        self.assertEqual(p._fields, ('x', 'y'))         # test _fields attribute
        self.assertEqual(p._replace(x=1), (1, 22))      # test _replace method
        self.assertEqual(p._asdict(), dict(x=11, y=22)) # test _asdict method

        with self.assertRaises(TypeError):
            p._replace(x=1, error=2)

        # verify that field string can have commas
        Point = namedtuple('Point', 'x, y')
        p = Point(x=11, y=22)
        self.assertEqual(repr(p), 'Point(x=11, y=22)')

        # verify that fieldspec can be a non-string sequence
        Point = namedtuple('Point', ('x', 'y'))
        p = Point(x=11, y=22)
        self.assertEqual(repr(p), 'Point(x=11, y=22)')

    def test_tupleness(self):
        Point = namedtuple('Point', 'x y')
        p = Point(11, 22)

        self.assertIsInstance(p, tuple)
        self.assertEqual(p, (11, 22))                                       # matches a real tuple
        self.assertEqual(tuple(p), (11, 22))                                # coercible to a real tuple
        self.assertEqual(list(p), [11, 22])                                 # coercible to a list
        self.assertEqual(max(p), 22)                                        # iterable
        self.assertEqual(max(*p), 22)                                       # star-able
        x, y = p
        self.assertEqual(p, (x, y))                                         # unpacks like a tuple
        self.assertEqual((p[0], p[1]), (11, 22))                            # indexable like a tuple
        with self.assertRaises(IndexError):
            p[3]
        self.assertEqual(p[-1], 22)
        self.assertEqual(hash(p), hash((11, 22)))

        self.assertEqual(p.x, x)
        self.assertEqual(p.y, y)
        with self.assertRaises(AttributeError):
            p.z

    def test_odd_sizes(self):
        Zero = namedtuple('Zero', '')
        self.assertEqual(Zero(), ())
        self.assertEqual(Zero._make([]), ())
        self.assertEqual(repr(Zero()), 'Zero()')
        self.assertEqual(Zero()._asdict(), {})
        self.assertEqual(Zero()._fields, ())

        Dot = namedtuple('Dot', 'd')
        self.assertEqual(Dot(1), (1,))
        self.assertEqual(Dot._make([1]), (1,))
        self.assertEqual(Dot(1).d, 1)
        self.assertEqual(repr(Dot(1)), 'Dot(d=1)')
        self.assertEqual(Dot(1)._asdict(), {'d':1})
        self.assertEqual(Dot(1)._replace(d=999), (999,))
        self.assertEqual(Dot(1)._fields, ('d',))

    @support.requires_resource('cpu')
    def test_large_size(self):
        n = support.exceeds_recursion_limit()
        names = list(set(''.join([choice(string.ascii_letters)
                                  for j in range(10)]) for i in range(n)))
        n = len(names)
        Big = namedtuple('Big', names)
        b = Big(*range(n))
        self.assertEqual(b, tuple(range(n)))
        self.assertEqual(Big._make(range(n)), tuple(range(n)))
        for pos, name in enumerate(names):
            self.assertEqual(getattr(b, name), pos)
        repr(b)                                 # make sure repr() doesn't blow-up
        d = b._asdict()
        d_expected = dict(zip(names, range(n)))
        self.assertEqual(d, d_expected)
        b2 = b._replace(**dict([(names[1], 999),(names[-5], 42)]))
        b2_expected = list(range(n))
        b2_expected[1] = 999
        b2_expected[-5] = 42
        self.assertEqual(b2, tuple(b2_expected))
        self.assertEqual(b._fields, tuple(names))

    def test_pickle(self):
        p = TestNT(x=10, y=20, z=30)
        for module in (pickle,):
            loads = getattr(module, 'loads')
            dumps = getattr(module, 'dumps')
            for protocol in range(-1, module.HIGHEST_PROTOCOL + 1):
                q = loads(dumps(p, protocol))
                self.assertEqual(p, q)
                self.assertEqual(p._fields, q._fields)
                self.assertNotIn(b'OrderedDict', dumps(p, protocol))

    def test_copy(self):
        p = TestNT(x=10, y=20, z=30)
        for copier in copy.copy, copy.deepcopy:
            q = copier(p)
            self.assertEqual(p, q)
            self.assertEqual(p._fields, q._fields)

    def test_name_conflicts(self):
        # Some names like "self", "cls", "tuple", "itemgetter", and "property"
        # failed when used as field names.  Test to make sure these now work.
        T = namedtuple('T', 'itemgetter property self cls tuple')
        t = T(1, 2, 3, 4, 5)
        self.assertEqual(t, (1,2,3,4,5))
        newt = t._replace(itemgetter=10, property=20, self=30, cls=40, tuple=50)
        self.assertEqual(newt, (10,20,30,40,50))

       # Broader test of all interesting names taken from the code, old
       # template, and an example
        words = {'Alias', 'At', 'AttributeError', 'Build', 'Bypass', 'Create',
        'Encountered', 'Expected', 'Field', 'For', 'Got', 'Helper',
        'IronPython', 'Jython', 'KeyError', 'Make', 'Modify', 'Note',
        'OrderedDict', 'Point', 'Return', 'Returns', 'Type', 'TypeError',
        'Used', 'Validate', 'ValueError', 'Variables', 'a', 'accessible', 'add',
        'added', 'all', 'also', 'an', 'arg_list', 'args', 'arguments',
        'automatically', 'be', 'build', 'builtins', 'but', 'by', 'cannot',
        'class_namespace', 'classmethod', 'cls', 'collections', 'convert',
        'copy', 'created', 'creation', 'd', 'debugging', 'defined', 'dict',
        'dictionary', 'doc', 'docstring', 'docstrings', 'duplicate', 'effect',
        'either', 'enumerate', 'environments', 'error', 'example', 'exec', 'f',
        'f_globals', 'field', 'field_names', 'fields', 'formatted', 'frame',
        'function', 'functions', 'generate', 'get', 'getter', 'got', 'greater',
        'has', 'help', 'identifiers', 'index', 'indexable', 'instance',
        'instantiate', 'interning', 'introspection', 'isidentifier',
        'isinstance', 'itemgetter', 'iterable', 'join', 'keyword', 'keywords',
        'kwds', 'len', 'like', 'list', 'map', 'maps', 'message', 'metadata',
        'method', 'methods', 'module', 'module_name', 'must', 'name', 'named',
        'namedtuple', 'namedtuple_', 'names', 'namespace', 'needs', 'new',
        'nicely', 'num_fields', 'number', 'object', 'of', 'operator', 'option',
        'p', 'particular', 'pickle', 'pickling', 'plain', 'pop', 'positional',
        'property', 'r', 'regular', 'rename', 'replace', 'replacing', 'repr',
        'repr_fmt', 'representation', 'result', 'reuse_itemgetter', 's', 'seen',
        'self', 'sequence', 'set', 'side', 'specified', 'split', 'start',
        'startswith', 'step', 'str', 'string', 'strings', 'subclass', 'sys',
        'targets', 'than', 'the', 'their', 'this', 'to', 'tuple', 'tuple_new',
        'type', 'typename', 'underscore', 'unexpected', 'unpack', 'up', 'use',
        'used', 'user', 'valid', 'values', 'variable', 'verbose', 'where',
        'which', 'work', 'x', 'y', 'z', 'zip'}
        T = namedtuple('T', words)
        # test __new__
        values = tuple(range(len(words)))
        t = T(*values)
        self.assertEqual(t, values)
        t = T(**dict(zip(T._fields, values)))
        self.assertEqual(t, values)
        # test _make
        t = T._make(values)
        self.assertEqual(t, values)
        # exercise __repr__
        repr(t)
        # test _asdict
        self.assertEqual(t._asdict(), dict(zip(T._fields, values)))
        # test _replace
        t = T._make(values)
        newvalues = tuple(v*10 for v in values)
        newt = t._replace(**dict(zip(T._fields, newvalues)))
        self.assertEqual(newt, newvalues)
        # test _fields
        self.assertEqual(T._fields, tuple(words))
        # test __getnewargs__
        self.assertEqual(t.__getnewargs__(), values)

    def test_repr(self):
        A = namedtuple('A', 'x')
        self.assertEqual(repr(A(1)), 'A(x=1)')
        # repr should show the name of the subclass
        class B(A):
            pass
        self.assertEqual(repr(B(1)), 'B(x=1)')

    def test_keyword_only_arguments(self):
        # See issue 25628
        with self.assertRaises(TypeError):
            NT = namedtuple('NT', ['x', 'y'], True)

        NT = namedtuple('NT', ['abc', 'def'], rename=True)
        self.assertEqual(NT._fields, ('abc', '_1'))
        with self.assertRaises(TypeError):
            NT = namedtuple('NT', ['abc', 'def'], False, True)

    def test_namedtuple_subclass_issue_24931(self):
        with torch._dynamo.error_on_graph_break(False):
            class Point(namedtuple('_Point', ['x', 'y'])):
                pass

        a = Point(3, 4)
        self.assertEqual(a._asdict(), OrderedDict([('x', 3), ('y', 4)]))

        a.w = 5
        self.assertEqual(a.__dict__, {'w': 5})

    @support.cpython_only
    def test_field_descriptor(self):
        Point = namedtuple('Point', 'x y')
        p = Point(11, 22)
        self.assertTrue(inspect.isdatadescriptor(Point.x))
        self.assertEqual(Point.x.__get__(p), 11)
        self.assertRaises(AttributeError, Point.x.__set__, p, 33)
        self.assertRaises(AttributeError, Point.x.__delete__, p)

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                class NewPoint(tuple):
                    x = pickle.loads(pickle.dumps(Point.x, proto))
                    y = pickle.loads(pickle.dumps(Point.y, proto))

                np = NewPoint([1, 2])

                self.assertEqual(np.x, 1)
                self.assertEqual(np.y, 2)

    def test_new_builtins_issue_43102(self):
        obj = namedtuple('C', ())
        new_func = obj.__new__
        self.assertEqual(new_func.__globals__['__builtins__'], {})
        self.assertEqual(new_func.__builtins__, {})

    def test_match_args(self):
        Point = namedtuple('Point', 'x y')
        self.assertEqual(Point.__match_args__, ('x', 'y'))

    def test_non_generic_subscript(self):
        # For backward compatibility, subscription works
        # on arbitrary named tuple types.
        Group = collections.namedtuple('Group', 'key group')
        A = Group[int, list[int]]
        self.assertEqual(A.__origin__, Group)
        self.assertEqual(A.__parameters__, ())
        self.assertEqual(A.__args__, (int, list[int]))
        a = A(1, [2])
        self.assertIs(type(a), Group)
        self.assertEqual(a, (1, [2]))


################################################################################
### Abstract Base Classes
################################################################################

class ABCTestCase(__TestCase):

    def validate_abstract_methods(self, abc, *names):
        methodstubs = dict.fromkeys(names, lambda s, *args: 0)

        # everything should work will all required methods are present
        with torch._dynamo.error_on_graph_break(False):
            C = type('C', (abc,), methodstubs)
        C()

        # Dynamo raises a hard error here that we can't easily capture
        # Commenting this part as this would also fail in eager if a user
        # attempt to run the same code

        # instantiation should fail if a required method is missing
        # for name in names:
        #     stubs = methodstubs.copy()
        #     del stubs[name]
        #     C = type('C', (abc,), stubs)
        #     self.assertRaises(TypeError, C, name)

    def validate_isinstance(self, abc, name):
        stub = lambda s, *args: 0

        C = type('C', (object,), {'__hash__': None})
        setattr(C, name, stub)
        self.assertIsInstance(C(), abc)
        self.assertTrue(issubclass(C, abc))

        C = type('C', (object,), {'__hash__': None})
        self.assertNotIsInstance(C(), abc)
        self.assertFalse(issubclass(C, abc))

    def validate_comparison(self, instance):
        ops = ['lt', 'gt', 'le', 'ge', 'ne', 'or', 'and', 'xor', 'sub']
        operators = {}
        for op in ops:
            name = '__' + op + '__'
            operators[name] = getattr(operator, name)

        class Other:
            def __init__(self):
                self.right_side = False
            def __eq__(self, other):
                self.right_side = True
                return True
            __lt__ = __eq__
            __gt__ = __eq__
            __le__ = __eq__
            __ge__ = __eq__
            __ne__ = __eq__
            __ror__ = __eq__
            __rand__ = __eq__
            __rxor__ = __eq__
            __rsub__ = __eq__

        for name, op in operators.items():
            if not hasattr(instance, name):
                continue
            other = Other()
            op(instance, other)
            self.assertTrue(other.right_side,'Right side not called for %s.%s'
                            % (type(instance), name))

def _test_gen():
    yield

class TestOneTrickPonyABCs(ABCTestCase):

    def test_Awaitable(self):
        def gen():
            yield

        @types.coroutine
        def coro():
            yield

        async def new_coro():
            pass

        class Bar:
            def __await__(self):
                yield

        class MinimalCoro(Coroutine):
            def send(self, value):
                return value
            def throw(self, typ, val=None, tb=None):
                super().throw(typ, val, tb)
            def __await__(self):
                yield

        self.validate_abstract_methods(Awaitable, '__await__')

        non_samples = [None, int(), gen(), object()]
        for x in non_samples:
            self.assertNotIsInstance(x, Awaitable)
            self.assertFalse(issubclass(type(x), Awaitable), repr(type(x)))

        samples = [Bar(), MinimalCoro()]
        for x in samples:
            self.assertIsInstance(x, Awaitable)
            self.assertTrue(issubclass(type(x), Awaitable))

        c = coro()
        # Iterable coroutines (generators with CO_ITERABLE_COROUTINE
        # flag don't have '__await__' method, hence can't be instances
        # of Awaitable. Use inspect.isawaitable to detect them.
        self.assertNotIsInstance(c, Awaitable)

        c = new_coro()
        self.assertIsInstance(c, Awaitable)
        c.close() # avoid RuntimeWarning that coro() was not awaited

        class CoroLike: pass
        Coroutine.register(CoroLike)
        self.assertTrue(isinstance(CoroLike(), Awaitable))
        self.assertTrue(issubclass(CoroLike, Awaitable))
        CoroLike = None
        support.gc_collect() # Kill CoroLike to clean-up ABCMeta cache

    def test_Coroutine(self):
        def gen():
            yield

        @types.coroutine
        def coro():
            yield

        async def new_coro():
            pass

        class Bar:
            def __await__(self):
                yield

        class MinimalCoro(Coroutine):
            def send(self, value):
                return value
            def throw(self, typ, val=None, tb=None):
                super().throw(typ, val, tb)
            def __await__(self):
                yield

        self.validate_abstract_methods(Coroutine, '__await__', 'send', 'throw')

        non_samples = [None, int(), gen(), object(), Bar()]
        for x in non_samples:
            self.assertNotIsInstance(x, Coroutine)
            self.assertFalse(issubclass(type(x), Coroutine), repr(type(x)))

        samples = [MinimalCoro()]
        for x in samples:
            self.assertIsInstance(x, Awaitable)
            self.assertTrue(issubclass(type(x), Awaitable))

        c = coro()
        # Iterable coroutines (generators with CO_ITERABLE_COROUTINE
        # flag don't have '__await__' method, hence can't be instances
        # of Coroutine. Use inspect.isawaitable to detect them.
        self.assertNotIsInstance(c, Coroutine)

        c = new_coro()
        self.assertIsInstance(c, Coroutine)
        c.close() # avoid RuntimeWarning that coro() was not awaited

        class CoroLike:
            def send(self, value):
                pass
            def throw(self, typ, val=None, tb=None):
                pass
            def close(self):
                pass
            def __await__(self):
                pass
        self.assertTrue(isinstance(CoroLike(), Coroutine))
        self.assertTrue(issubclass(CoroLike, Coroutine))

        class CoroLike:
            def send(self, value):
                pass
            def close(self):
                pass
            def __await__(self):
                pass
        self.assertFalse(isinstance(CoroLike(), Coroutine))
        self.assertFalse(issubclass(CoroLike, Coroutine))

    def test_Hashable(self):
        # Check some non-hashables
        non_samples = [bytearray(), list(), set(), dict()]
        for x in non_samples:
            self.assertNotIsInstance(x, Hashable)
            self.assertFalse(issubclass(type(x), Hashable), repr(type(x)))
        # Check some hashables
        samples = [None,
                   int(), float(), complex(),
                   str(),
                   tuple(), frozenset(),
                   int, list, object, type, bytes()
                   ]
        for x in samples:
            self.assertIsInstance(x, Hashable)
            self.assertTrue(issubclass(type(x), Hashable), repr(type(x)))
        self.assertRaises(TypeError, Hashable)
        # Check direct subclassing
        class H(Hashable):
            def __hash__(self):
                return super().__hash__()
        self.assertEqual(hash(H()), 0)
        self.assertFalse(issubclass(int, H))
        self.validate_abstract_methods(Hashable, '__hash__')
        self.validate_isinstance(Hashable, '__hash__')

    def test_AsyncIterable(self):
        class AI:
            def __aiter__(self):
                return self
        self.assertTrue(isinstance(AI(), AsyncIterable))
        self.assertTrue(issubclass(AI, AsyncIterable))
        # Check some non-iterables
        non_samples = [None, object, []]
        for x in non_samples:
            self.assertNotIsInstance(x, AsyncIterable)
            self.assertFalse(issubclass(type(x), AsyncIterable), repr(type(x)))
        self.validate_abstract_methods(AsyncIterable, '__aiter__')
        self.validate_isinstance(AsyncIterable, '__aiter__')

    def test_AsyncIterator(self):
        class AI:
            def __aiter__(self):
                return self
            async def __anext__(self):
                raise StopAsyncIteration
        self.assertTrue(isinstance(AI(), AsyncIterator))
        self.assertTrue(issubclass(AI, AsyncIterator))
        non_samples = [None, object, []]
        # Check some non-iterables
        for x in non_samples:
            self.assertNotIsInstance(x, AsyncIterator)
            self.assertFalse(issubclass(type(x), AsyncIterator), repr(type(x)))
        # Similarly to regular iterators (see issue 10565)
        class AnextOnly:
            async def __anext__(self):
                raise StopAsyncIteration
        self.assertNotIsInstance(AnextOnly(), AsyncIterator)
        self.validate_abstract_methods(AsyncIterator, '__anext__', '__aiter__')

    def test_Iterable(self):
        # Check some non-iterables
        non_samples = [None, 42, 3.14, 1j]
        for x in non_samples:
            self.assertNotIsInstance(x, Iterable)
            self.assertFalse(issubclass(type(x), Iterable), repr(type(x)))
        # Check some iterables
        samples = [bytes(), str(),
                   tuple(), list(), set(), frozenset(), dict(),
                   dict().keys(), dict().items(), dict().values(),
                   _test_gen(),
                   (x for x in []),
                   ]
        for x in samples:
            self.assertIsInstance(x, Iterable)
            self.assertTrue(issubclass(type(x), Iterable), repr(type(x)))
        with torch._dynamo.error_on_graph_break(False):
            # Check direct subclassing
            class I(Iterable):
                def __iter__(self):
                    return super().__iter__()
        self.assertEqual(list(I()), [])
        self.assertFalse(issubclass(str, I))
        self.validate_abstract_methods(Iterable, '__iter__')
        self.validate_isinstance(Iterable, '__iter__')
        with torch._dynamo.error_on_graph_break(False):
            # Check None blocking
            class It:
                def __iter__(self): return iter([])
            class ItBlocked(It):
                __iter__ = None
        self.assertTrue(issubclass(It, Iterable))
        self.assertTrue(isinstance(It(), Iterable))
        self.assertFalse(issubclass(ItBlocked, Iterable))
        self.assertFalse(isinstance(ItBlocked(), Iterable))

    def test_Reversible(self):
        # Check some non-reversibles
        non_samples = [None, 42, 3.14, 1j, set(), frozenset()]
        for x in non_samples:
            self.assertNotIsInstance(x, Reversible)
            self.assertFalse(issubclass(type(x), Reversible), repr(type(x)))
        # Check some non-reversible iterables
        non_reversibles = [_test_gen(), (x for x in []), iter([]), reversed([])]
        for x in non_reversibles:
            self.assertNotIsInstance(x, Reversible)
            self.assertFalse(issubclass(type(x), Reversible), repr(type(x)))
        # Check some reversible iterables
        samples = [bytes(), str(), tuple(), list(), OrderedDict(),
                   OrderedDict().keys(), OrderedDict().items(),
                   OrderedDict().values(), Counter(), Counter().keys(),
                   Counter().items(), Counter().values(), dict(),
                   dict().keys(), dict().items(), dict().values()]
        for x in samples:
            self.assertIsInstance(x, Reversible)
            self.assertTrue(issubclass(type(x), Reversible), repr(type(x)))
        # Check also Mapping, MutableMapping, and Sequence
        self.assertTrue(issubclass(Sequence, Reversible), repr(Sequence))
        self.assertFalse(issubclass(Mapping, Reversible), repr(Mapping))
        self.assertFalse(issubclass(MutableMapping, Reversible), repr(MutableMapping))
        with torch._dynamo.error_on_graph_break(False):
            # Check direct subclassing
            class R(Reversible):
                def __iter__(self):
                    return iter(list())
                def __reversed__(self):
                    return iter(list())
        self.assertEqual(list(reversed(R())), [])
        self.assertFalse(issubclass(float, R))
        self.validate_abstract_methods(Reversible, '__reversed__', '__iter__')
        with torch._dynamo.error_on_graph_break(False):
            # Check reversible non-iterable (which is not Reversible)
            class RevNoIter:
                def __reversed__(self): return reversed([])
            class RevPlusIter(RevNoIter):
                def __iter__(self): return iter([])
        self.assertFalse(issubclass(RevNoIter, Reversible))
        self.assertFalse(isinstance(RevNoIter(), Reversible))
        self.assertTrue(issubclass(RevPlusIter, Reversible))
        self.assertTrue(isinstance(RevPlusIter(), Reversible))
        with torch._dynamo.error_on_graph_break(False):
            # Check None blocking
            class Rev:
                def __iter__(self): return iter([])
                def __reversed__(self): return reversed([])
            class RevItBlocked(Rev):
                __iter__ = None
            class RevRevBlocked(Rev):
                __reversed__ = None
        self.assertTrue(issubclass(Rev, Reversible))
        self.assertTrue(isinstance(Rev(), Reversible))
        self.assertFalse(issubclass(RevItBlocked, Reversible))
        self.assertFalse(isinstance(RevItBlocked(), Reversible))
        self.assertFalse(issubclass(RevRevBlocked, Reversible))
        self.assertFalse(isinstance(RevRevBlocked(), Reversible))

    def test_Collection(self):
        # Check some non-collections
        non_collections = [None, 42, 3.14, 1j, lambda x: 2*x]
        for x in non_collections:
            self.assertNotIsInstance(x, Collection)
            self.assertFalse(issubclass(type(x), Collection), repr(type(x)))
        # Check some non-collection iterables
        non_col_iterables = [_test_gen(), iter(b''), iter(bytearray()),
                             (x for x in [])]
        for x in non_col_iterables:
            self.assertNotIsInstance(x, Collection)
            self.assertFalse(issubclass(type(x), Collection), repr(type(x)))
        # Check some collections
        samples = [set(), frozenset(), dict(), bytes(), str(), tuple(),
                   list(), dict().keys(), dict().items(), dict().values()]
        for x in samples:
            self.assertIsInstance(x, Collection)
            self.assertTrue(issubclass(type(x), Collection), repr(type(x)))
        # Check also Mapping, MutableMapping, etc.
        self.assertTrue(issubclass(Sequence, Collection), repr(Sequence))
        self.assertTrue(issubclass(Mapping, Collection), repr(Mapping))
        self.assertTrue(issubclass(MutableMapping, Collection),
                                    repr(MutableMapping))
        self.assertTrue(issubclass(Set, Collection), repr(Set))
        self.assertTrue(issubclass(MutableSet, Collection), repr(MutableSet))
        self.assertTrue(issubclass(Sequence, Collection), repr(MutableSet))
        with torch._dynamo.error_on_graph_break(False):
            # Check direct subclassing
            class Col(Collection):
                def __iter__(self):
                    return iter(list())
                def __len__(self):
                    return 0
                def __contains__(self, item):
                    return False
            class DerCol(Col): pass
        self.assertEqual(list(iter(Col())), [])
        self.assertFalse(issubclass(list, Col))
        self.assertFalse(issubclass(set, Col))
        self.assertFalse(issubclass(float, Col))
        self.assertEqual(list(iter(DerCol())), [])
        self.assertFalse(issubclass(list, DerCol))
        self.assertFalse(issubclass(set, DerCol))
        self.assertFalse(issubclass(float, DerCol))
        self.validate_abstract_methods(Collection, '__len__', '__iter__',
                                                   '__contains__')
        # Check sized container non-iterable (which is not Collection) etc.
        with torch._dynamo.error_on_graph_break(False):
            class ColNoIter:
                def __len__(self): return 0
                def __contains__(self, item): return False
            class ColNoSize:
                def __iter__(self): return iter([])
                def __contains__(self, item): return False
            class ColNoCont:
                def __iter__(self): return iter([])
                def __len__(self): return 0
        self.assertFalse(issubclass(ColNoIter, Collection))
        self.assertFalse(isinstance(ColNoIter(), Collection))
        self.assertFalse(issubclass(ColNoSize, Collection))
        self.assertFalse(isinstance(ColNoSize(), Collection))
        self.assertFalse(issubclass(ColNoCont, Collection))
        self.assertFalse(isinstance(ColNoCont(), Collection))

        with torch._dynamo.error_on_graph_break(False):
            # Check None blocking
            class SizeBlock:
                def __iter__(self): return iter([])
                def __contains__(self): return False
                __len__ = None
            class IterBlock:
                def __len__(self): return 0
                def __contains__(self): return True
                __iter__ = None
        self.assertFalse(issubclass(SizeBlock, Collection))
        self.assertFalse(isinstance(SizeBlock(), Collection))
        self.assertFalse(issubclass(IterBlock, Collection))
        self.assertFalse(isinstance(IterBlock(), Collection))
        with torch._dynamo.error_on_graph_break(False):
            # Check None blocking in subclass
            class ColImpl:
                def __iter__(self):
                    return iter(list())
                def __len__(self):
                    return 0
                def __contains__(self, item):
                    return False
            class NonCol(ColImpl):
                __contains__ = None
        self.assertFalse(issubclass(NonCol, Collection))
        self.assertFalse(isinstance(NonCol(), Collection))


    def test_Iterator(self):
        non_samples = [None, 42, 3.14, 1j, b"", "", (), [], {}, set()]
        for x in non_samples:
            self.assertNotIsInstance(
```



## High-Level Overview

"""Unit tests for collections.py."""import arrayimport collectionsimport copyimport doctestimport inspectimport operatorimport picklefrom random import choice, randrangefrom itertools import product, chain, combinationsimport stringimport sysfrom test import supportimport typesimport unittestfrom collections import namedtuple, Counter, OrderedDict, _count_elementsfrom collections import UserDict, UserString, UserListfrom collections import ChainMapfrom collections import dequefrom collections.abc import Awaitable, Coroutinefrom collections.abc import AsyncIterator, AsyncIterable, AsyncGeneratorfrom collections.abc import Hashable, Iterable, Iterator, Generator, Reversiblefrom collections.abc import Sized, Container, Callable, Collectionfrom collections.abc import Set, MutableSetfrom collections.abc import Mapping, MutableMapping, KeysView, ItemsView, ValuesViewfrom collections.abc import Sequence, MutableSequencefrom collections.abc import ByteString, Buffer

This Python file contains 89 class(es) and 289 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestUserObjects`, `A`, `TestChainMap`, `DefaultChainMap`, `DictWithGetItem`, `lowerdict`, `Subclass`, `SubclassRor`, `TestNamedTuple`, `B`, `NewPoint`, `ABCTestCase`, `Other`, `TestOneTrickPonyABCs`, `Bar`, `MinimalCoro`, `CoroLike`, `Bar`, `MinimalCoro`, `CoroLike`

**Functions defined**: `_superset_test`, `_copy_test`, `test_str_protocol`, `test_list_protocol`, `test_dict_protocol`, `test_list_copy`, `test_dict_copy`, `test_dict_missing`, `__missing__`, `test_basics`, `test_ordering`, `test_constructor`, `test_bool`, `test_missing`, `__missing__`, `test_order_preservation`, `test_iter_not_calling_getitem_on_maps`, `__init__`, `__getitem__`, `test_dict_coercion`

**Key imports**: sys, torch, torch._dynamo.test_case, unittest, CPythonTestCase, run_tests, array, collections, copy, doctest


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo/cpython/3_13`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `sys`
- `torch`
- `torch._dynamo.test_case`
- `unittest`
- `torch.testing._internal.common_utils`: run_tests
- `array`
- `collections`
- `copy`
- `doctest`
- `inspect`
- `operator`
- `pickle`
- `random`: choice, randrange
- `itertools`: product, chain, combinations
- `string`
- `test`: support
- `types`
- `collections.abc`: Awaitable, Coroutine


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
- **Error Handling**: Includes exception handling
- **Asynchronous Programming**: Uses async/await


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized
- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/dynamo/cpython/3_13/test_collections.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/dynamo/cpython/3_13`):

- [`mapping_tests.diff_docs.md`](./mapping_tests.diff_docs.md)
- [`test_float.py_docs.md`](./test_float.py_docs.md)
- [`test_generators.py_docs.md`](./test_generators.py_docs.md)
- [`test_dict.py_docs.md`](./test_dict.py_docs.md)
- [`test_generator_stop.diff_docs.md`](./test_generator_stop.diff_docs.md)
- [`test_sort.diff_docs.md`](./test_sort.diff_docs.md)
- [`test_list.diff_docs.md`](./test_list.diff_docs.md)
- [`test_userdict.diff_docs.md`](./test_userdict.diff_docs.md)
- [`test_generators.diff_docs.md`](./test_generators.diff_docs.md)
- [`test_userlist.py_docs.md`](./test_userlist.py_docs.md)


## Cross-References

- **File Documentation**: `test_collections.py_docs.md`
- **Keyword Index**: `test_collections.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
