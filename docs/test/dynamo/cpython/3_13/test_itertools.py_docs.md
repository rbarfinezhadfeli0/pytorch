# Documentation: `test/dynamo/cpython/3_13/test_itertools.py`

## File Metadata

- **Path**: `test/dynamo/cpython/3_13/test_itertools.py`
- **Size**: 116,332 bytes (113.61 KB)
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
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_itertools.py

import torch
import torch._dynamo.test_case
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfTorchDynamo,
    slowTest,
)

__TestCase = CPythonTestCase

# ======= END DYNAMO PATCH =======

import doctest
import unittest
import itertools
from test import support
from test.support import threading_helper, script_helper
from itertools import *
import weakref
from decimal import Decimal
from fractions import Fraction
import operator
import random
import copy
import pickle
from functools import reduce
import sys
import struct
import threading
import gc
import warnings

def pickle_deprecated(testfunc):
    """ Run the test three times.
    First, verify that a Deprecation Warning is raised.
    Second, run normally but with DeprecationWarnings temporarily disabled.
    Third, run with warnings promoted to errors.
    """
    def inner(self):
        with self.assertWarns(DeprecationWarning):
            testfunc(self)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            testfunc(self)
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=DeprecationWarning)
            with self.assertRaises((DeprecationWarning, AssertionError, SystemError)):
                testfunc(self)

    return inner

maxsize = support.MAX_Py_ssize_t
minsize = -maxsize-1

@torch._dynamo.disable
def choice(*args):
    return random.choice(*args)

@torch._dynamo.disable
def randrange(*args):
    return random.randrange(*args)

def lzip(*args):
    return list(zip(*args))

def onearg(x):
    'Test function of one argument'
    return 2*x

def errfunc(*args):
    'Test function that raises an error'
    raise ValueError

def gen3():
    'Non-restartable source sequence'
    for i in (0, 1, 2):
        yield i

def isEven(x):
    'Test predicate'
    return x%2==0

def isOdd(x):
    'Test predicate'
    return x%2==1

def tupleize(*args):
    return args

def irange(n):
    for i in range(n):
        yield i

class StopNow:
    'Class emulating an empty iterable.'
    def __iter__(self):
        return self
    def __next__(self):
        raise StopIteration

def take(n, seq):
    'Convenience function for partially consuming a long of infinite iterable'
    return list(islice(seq, n))

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def fact(n):
    'Factorial'
    return prod(range(1, n+1))

# root level methods for pickling ability
def _testR(r):
    return r[0]

def _testR2(r):
    return r[2]

def underten(x):
    return x<10

picklecopiers = [lambda s, proto=proto: pickle.loads(pickle.dumps(s, proto))
                 for proto in range(pickle.HIGHEST_PROTOCOL + 1)]

class TestBasicOps(__TestCase):

    def pickletest(self, protocol, it, stop=4, take=1, compare=None):
        """Test that an iterator is the same after pickling, also when part-consumed"""
        def expand(it, i=0):
            # Recursively expand iterables, within sensible bounds
            if i > 10:
                raise RuntimeError("infinite recursion encountered")
            if isinstance(it, str):
                return it
            try:
                l = list(islice(it, stop))
            except TypeError:
                return it # can't expand it
            return [expand(e, i+1) for e in l]

        # Test the initial copy against the original
        dump = pickle.dumps(it, protocol)
        i2 = pickle.loads(dump)
        self.assertEqual(type(it), type(i2))
        a, b = expand(it), expand(i2)
        self.assertEqual(a, b)
        if compare:
            c = expand(compare)
            self.assertEqual(a, c)

        # Take from the copy, and create another copy and compare them.
        i3 = pickle.loads(dump)
        took = 0
        try:
            for i in range(take):
                next(i3)
                took += 1
        except StopIteration:
            pass #in case there is less data than 'take'
        dump = pickle.dumps(i3, protocol)
        i4 = pickle.loads(dump)
        a, b = expand(i3), expand(i4)
        self.assertEqual(a, b)
        if compare:
            c = expand(compare[took:])
            self.assertEqual(a, c);

    @pickle_deprecated
    def test_accumulate(self):
        self.assertEqual(list(accumulate(range(10))),               # one positional arg
                          [0, 1, 3, 6, 10, 15, 21, 28, 36, 45])
        self.assertEqual(list(accumulate(iterable=range(10))),      # kw arg
                          [0, 1, 3, 6, 10, 15, 21, 28, 36, 45])
        for typ in int, complex, Decimal, Fraction:                 # multiple types
            self.assertEqual(
                list(accumulate(map(typ, range(10)))),
                list(map(typ, [0, 1, 3, 6, 10, 15, 21, 28, 36, 45])))
        self.assertEqual(list(accumulate('abc')), ['a', 'ab', 'abc'])   # works with non-numeric
        self.assertEqual(list(accumulate([])), [])                  # empty iterable
        self.assertEqual(list(accumulate([7])), [7])                # iterable of length one
        self.assertRaises(TypeError, accumulate, range(10), 5, 6)   # too many args
        self.assertRaises(TypeError, accumulate)                    # too few args
        self.assertRaises(TypeError, accumulate, x=range(10))       # unexpected kwd arg
        self.assertRaises(TypeError, list, accumulate([1, []]))     # args that don't add

        s = [2, 8, 9, 5, 7, 0, 3, 4, 1, 6]
        self.assertEqual(list(accumulate(s, min)),
                         [2, 2, 2, 2, 2, 0, 0, 0, 0, 0])
        self.assertEqual(list(accumulate(s, max)),
                         [2, 8, 9, 9, 9, 9, 9, 9, 9, 9])
        self.assertEqual(list(accumulate(s, operator.mul)),
                         [2, 16, 144, 720, 5040, 0, 0, 0, 0, 0])
        with self.assertRaises(TypeError):
            list(accumulate(s, chr))                                # unary-operation
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.pickletest(proto, accumulate(range(10)))           # test pickling
            self.pickletest(proto, accumulate(range(10), initial=7))
        self.assertEqual(list(accumulate([10, 5, 1], initial=None)), [10, 15, 16])
        self.assertEqual(list(accumulate([10, 5, 1], initial=100)), [100, 110, 115, 116])
        self.assertEqual(list(accumulate([], initial=100)), [100])
        with self.assertRaises(TypeError):
            list(accumulate([10, 20], 100))

    def test_batched(self):
        self.assertEqual(list(batched('ABCDEFG', 3)),
                             [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)])
        self.assertEqual(list(batched('ABCDEFG', 2)),
                             [('A', 'B'), ('C', 'D'), ('E', 'F'), ('G',)])
        self.assertEqual(list(batched('ABCDEFG', 1)),
                            [('A',), ('B',), ('C',), ('D',), ('E',), ('F',), ('G',)])
        self.assertEqual(list(batched('ABCDEF', 2, strict=True)),
                             [('A', 'B'), ('C', 'D'), ('E', 'F')])

        with self.assertRaises(ValueError):         # Incomplete batch when strict
            list(batched('ABCDEFG', 3, strict=True))
        with self.assertRaises(TypeError):          # Too few arguments
            list(batched('ABCDEFG'))
        with self.assertRaises(TypeError):
            list(batched('ABCDEFG', 3, None))       # Too many arguments
        with self.assertRaises(TypeError):
            list(batched(None, 3))                  # Non-iterable input
        with self.assertRaises(TypeError):
            list(batched('ABCDEFG', 'hello'))       # n is a string
        with self.assertRaises(ValueError):
            list(batched('ABCDEFG', 0))             # n is zero
        with self.assertRaises(ValueError):
            list(batched('ABCDEFG', -1))            # n is negative

        data = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for n in range(1, 6):
            for i in range(len(data)):
                s = data[:i]
                batches = list(batched(s, n))
                with self.subTest(s=s, n=n, batches=batches):
                    # Order is preserved and no data is lost
                    self.assertEqual(''.join(chain(*batches)), s)
                    # Each batch is an exact tuple
                    self.assertTrue(all(type(batch) is tuple for batch in batches))
                    # All but the last batch is of size n
                    if batches:
                        last_batch = batches.pop()
                        self.assertTrue(all(len(batch) == n for batch in batches))
                        self.assertTrue(len(last_batch) <= n)
                        batches.append(last_batch)

    def test_chain(self):

        def chain2(*iterables):
            'Pure python version in the docs'
            for it in iterables:
                for element in it:
                    yield element

        for c in (chain, chain2):
            self.assertEqual(list(c('abc', 'def')), list('abcdef'))
            self.assertEqual(list(c('abc')), list('abc'))
            self.assertEqual(list(c('')), [])
            self.assertEqual(take(4, c('abc', 'def')), list('abcd'))
            self.assertRaises(TypeError, list,c(2, 3))

    def test_chain_from_iterable(self):
        self.assertEqual(list(chain.from_iterable(['abc', 'def'])), list('abcdef'))
        self.assertEqual(list(chain.from_iterable(['abc'])), list('abc'))
        self.assertEqual(list(chain.from_iterable([''])), [])
        self.assertEqual(take(4, chain.from_iterable(['abc', 'def'])), list('abcd'))
        self.assertRaises(TypeError, list, chain.from_iterable([2, 3]))
        self.assertEqual(list(islice(chain.from_iterable(repeat(range(5))), 2)), [0, 1])

    @pickle_deprecated
    def test_chain_reducible(self):
        for oper in [copy.deepcopy] + picklecopiers:
            it = chain('abc', 'def')
            self.assertEqual(list(oper(it)), list('abcdef'))
            self.assertEqual(next(it), 'a')
            self.assertEqual(list(oper(it)), list('bcdef'))

            self.assertEqual(list(oper(chain(''))), [])
            self.assertEqual(take(4, oper(chain('abc', 'def'))), list('abcd'))
            self.assertRaises(TypeError, list, oper(chain(2, 3)))
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.pickletest(proto, chain('abc', 'def'), compare=list('abcdef'))

    @pickle_deprecated
    def test_chain_setstate(self):
        self.assertRaises(TypeError, chain().__setstate__, ())
        self.assertRaises(TypeError, chain().__setstate__, [])
        self.assertRaises(TypeError, chain().__setstate__, 0)
        self.assertRaises(TypeError, chain().__setstate__, ([],))
        self.assertRaises(TypeError, chain().__setstate__, (iter([]), []))
        it = chain()
        it.__setstate__((iter(['abc', 'def']),))
        self.assertEqual(list(it), ['a', 'b', 'c', 'd', 'e', 'f'])
        it = chain()
        it.__setstate__((iter(['abc', 'def']), iter(['ghi'])))
        self.assertEqual(list(it), ['ghi', 'a', 'b', 'c', 'd', 'e', 'f'])

    @pickle_deprecated
    def test_combinations(self):
        self.assertRaises(TypeError, combinations, 'abc')       # missing r argument
        self.assertRaises(TypeError, combinations, 'abc', 2, 1) # too many arguments
        self.assertRaises(TypeError, combinations, None)        # pool is not iterable
        self.assertRaises(ValueError, combinations, 'abc', -2)  # r is negative

        for op in [lambda a:a] + picklecopiers:
            self.assertEqual(list(op(combinations('abc', 32))), [])     # r > n

            self.assertEqual(list(op(combinations('ABCD', 2))),
                             [('A','B'), ('A','C'), ('A','D'), ('B','C'), ('B','D'), ('C','D')])
            testIntermediate = combinations('ABCD', 2)
            next(testIntermediate)
            self.assertEqual(list(op(testIntermediate)),
                             [('A','C'), ('A','D'), ('B','C'), ('B','D'), ('C','D')])

            self.assertEqual(list(op(combinations(range(4), 3))),
                             [(0,1,2), (0,1,3), (0,2,3), (1,2,3)])
            testIntermediate = combinations(range(4), 3)
            next(testIntermediate)
            self.assertEqual(list(op(testIntermediate)),
                             [(0,1,3), (0,2,3), (1,2,3)])

        def combinations1(iterable, r):
            'Pure python version shown in the docs'
            pool = tuple(iterable)
            n = len(pool)
            if r > n:
                return
            indices = list(range(r))
            yield tuple(pool[i] for i in indices)
            while 1:
                for i in reversed(range(r)):
                    if indices[i] != i + n - r:
                        break
                else:
                    return
                indices[i] += 1
                for j in range(i+1, r):
                    indices[j] = indices[j-1] + 1
                yield tuple(pool[i] for i in indices)

        def combinations2(iterable, r):
            'Pure python version shown in the docs'
            pool = tuple(iterable)
            n = len(pool)
            for indices in permutations(range(n), r):
                if sorted(indices) == list(indices):
                    yield tuple(pool[i] for i in indices)

        def combinations3(iterable, r):
            'Pure python version from cwr()'
            pool = tuple(iterable)
            n = len(pool)
            for indices in combinations_with_replacement(range(n), r):
                if len(set(indices)) == r:
                    yield tuple(pool[i] for i in indices)

        for n in range(7):
            values = [5*x-12 for x in range(n)]
            for r in range(n+2):
                result = list(combinations(values, r))
                self.assertEqual(len(result), 0 if r>n else fact(n) / fact(r) / fact(n-r)) # right number of combs
                self.assertEqual(len(result), len(set(result)))         # no repeats
                self.assertEqual(result, sorted(result))                # lexicographic order
                for c in result:
                    self.assertEqual(len(c), r)                         # r-length combinations
                    self.assertEqual(len(set(c)), r)                    # no duplicate elements
                    self.assertEqual(list(c), sorted(c))                # keep original ordering
                    self.assertTrue(all(e in values for e in c))           # elements taken from input iterable
                    self.assertEqual(list(c),
                                     [e for e in values if e in c])      # comb is a subsequence of the input iterable
                self.assertEqual(result, list(combinations1(values, r))) # matches first pure python version
                self.assertEqual(result, list(combinations2(values, r))) # matches second pure python version
                self.assertEqual(result, list(combinations3(values, r))) # matches second pure python version

                for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.pickletest(proto, combinations(values, r))      # test pickling

    @support.bigaddrspacetest
    def test_combinations_overflow(self):
        with self.assertRaises((OverflowError, MemoryError)):
            combinations("AA", 2**29)

        # Test implementation detail:  tuple re-use
    @support.impl_detail("tuple reuse is specific to CPython")
    def test_combinations_tuple_reuse(self):
        self.assertEqual(len(set(map(id, combinations('abcde', 3)))), 1)
        self.assertNotEqual(len(set(map(id, list(combinations('abcde', 3))))), 1)

    @pickle_deprecated
    def test_combinations_with_replacement(self):
        cwr = combinations_with_replacement
        self.assertRaises(TypeError, cwr, 'abc')       # missing r argument
        self.assertRaises(TypeError, cwr, 'abc', 2, 1) # too many arguments
        self.assertRaises(TypeError, cwr, None)        # pool is not iterable
        self.assertRaises(ValueError, cwr, 'abc', -2)  # r is negative

        for op in [lambda a:a] + picklecopiers:
            self.assertEqual(list(op(cwr('ABC', 2))),
                             [('A','A'), ('A','B'), ('A','C'), ('B','B'), ('B','C'), ('C','C')])
            testIntermediate = cwr('ABC', 2)
            next(testIntermediate)
            self.assertEqual(list(op(testIntermediate)),
                             [('A','B'), ('A','C'), ('B','B'), ('B','C'), ('C','C')])


        def cwr1(iterable, r):
            'Pure python version shown in the docs'
            # number items returned:  (n+r-1)! / r! / (n-1)! when n>0
            pool = tuple(iterable)
            n = len(pool)
            if not n and r:
                return
            indices = [0] * r
            yield tuple(pool[i] for i in indices)
            while 1:
                for i in reversed(range(r)):
                    if indices[i] != n - 1:
                        break
                else:
                    return
                indices[i:] = [indices[i] + 1] * (r - i)
                yield tuple(pool[i] for i in indices)

        def cwr2(iterable, r):
            'Pure python version shown in the docs'
            pool = tuple(iterable)
            n = len(pool)
            for indices in product(range(n), repeat=r):
                if sorted(indices) == list(indices):
                    yield tuple(pool[i] for i in indices)

        def numcombs(n, r):
            if not n:
                return 0 if r else 1
            return fact(n+r-1) / fact(r)/ fact(n-1)

        for n in range(7):
            values = [5*x-12 for x in range(n)]
            for r in range(n+2):
                result = list(cwr(values, r))

                self.assertEqual(len(result), numcombs(n, r))           # right number of combs
                self.assertEqual(len(result), len(set(result)))         # no repeats
                self.assertEqual(result, sorted(result))                # lexicographic order

                regular_combs = list(combinations(values, r))           # compare to combs without replacement
                if n == 0 or r <= 1:
                    self.assertEqual(result, regular_combs)            # cases that should be identical
                else:
                    self.assertTrue(set(result) >= set(regular_combs))     # rest should be supersets of regular combs

                for c in result:
                    self.assertEqual(len(c), r)                         # r-length combinations
                    noruns = [k for k,v in groupby(c)]                  # combo without consecutive repeats
                    self.assertEqual(len(noruns), len(set(noruns)))     # no repeats other than consecutive
                    self.assertEqual(list(c), sorted(c))                # keep original ordering
                    self.assertTrue(all(e in values for e in c))           # elements taken from input iterable
                    self.assertEqual(noruns,
                                     [e for e in values if e in c])     # comb is a subsequence of the input iterable
                self.assertEqual(result, list(cwr1(values, r)))         # matches first pure python version
                self.assertEqual(result, list(cwr2(values, r)))         # matches second pure python version

                for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.pickletest(proto, cwr(values,r))               # test pickling

    @support.bigaddrspacetest
    def test_combinations_with_replacement_overflow(self):
        with self.assertRaises((OverflowError, MemoryError)):
            combinations_with_replacement("AA", 2**30)

        # Test implementation detail:  tuple re-use
    @support.impl_detail("tuple reuse is specific to CPython")
    def test_combinations_with_replacement_tuple_reuse(self):
        cwr = combinations_with_replacement
        self.assertEqual(len(set(map(id, cwr('abcde', 3)))), 1)
        self.assertNotEqual(len(set(map(id, list(cwr('abcde', 3))))), 1)

    def test_permutations(self):
        self.assertEqual(list(permutations('abc', 32)), [])     # r > n
        self.assertEqual(list(permutations(range(3), 2)),
                                           [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)])

        def permutations1(iterable, r=None):
            'Pure python version shown in the docs'
            pool = tuple(iterable)
            n = len(pool)
            r = n if r is None else r
            if r > n:
                return
            indices = list(range(n))
            cycles = list(range(n-r+1, n+1))[::-1]
            yield tuple(pool[i] for i in indices[:r])
            while n:
                for i in reversed(range(r)):
                    cycles[i] -= 1
                    if cycles[i] == 0:
                        indices[i:] = indices[i+1:] + indices[i:i+1]
                        cycles[i] = n - i
                    else:
                        j = cycles[i]
                        indices[i], indices[-j] = indices[-j], indices[i]
                        yield tuple(pool[i] for i in indices[:r])
                        break
                else:
                    return

        def permutations2(iterable, r=None):
            'Pure python version shown in the docs'
            pool = tuple(iterable)
            n = len(pool)
            r = n if r is None else r
            for indices in product(range(n), repeat=r):
                if len(set(indices)) == r:
                    yield tuple(pool[i] for i in indices)

        for n in range(5):
            values = [5*x-12 for x in range(n)]
            for r in range(n+2):
                result = list(permutations(values, r))
                self.assertEqual(len(result), 0 if r>n else fact(n) / fact(n-r))      # right number of perms
                self.assertEqual(len(result), len(set(result)))         # no repeats
                self.assertEqual(result, sorted(result))                # lexicographic order
                for p in result:
                    self.assertEqual(len(p), r)                         # r-length permutations
                    self.assertEqual(len(set(p)), r)                    # no duplicate elements
                    self.assertTrue(all(e in values for e in p))           # elements taken from input iterable
                self.assertEqual(result, list(permutations1(values, r))) # matches first pure python version
                self.assertEqual(result, list(permutations2(values, r))) # matches second pure python version
                if r == n:
                    self.assertEqual(result, list(permutations(values, None))) # test r as None
                    self.assertEqual(result, list(permutations(values)))       # test default r

    @support.bigaddrspacetest
    def test_permutations_overflow(self):
        with self.assertRaises((OverflowError, MemoryError)):
            permutations("A", 2**30)

    @support.impl_detail("tuple reuse is specific to CPython")
    def test_permutations_tuple_reuse(self):
        self.assertEqual(len(set(map(id, permutations('abcde', 3)))), 1)
        self.assertNotEqual(len(set(map(id, list(permutations('abcde', 3))))), 1)

    def test_combinatorics(self):
        # Test relationships between product(), permutations(),
        # combinations() and combinations_with_replacement().

        for n in range(6):
            s = 'ABCDEFG'[:n]
            for r in range(8):
                prod = list(product(s, repeat=r))
                cwr = list(combinations_with_replacement(s, r))
                perm = list(permutations(s, r))
                comb = list(combinations(s, r))

                # Check size
                self.assertEqual(len(prod), n**r)
                self.assertEqual(len(cwr), (fact(n+r-1) / fact(r)/ fact(n-1)) if n else (not r))
                self.assertEqual(len(perm), 0 if r>n else fact(n) / fact(n-r))
                self.assertEqual(len(comb), 0 if r>n else fact(n) / fact(r) / fact(n-r))

                # Check lexicographic order without repeated tuples
                self.assertEqual(prod, sorted(set(prod)))
                self.assertEqual(cwr, sorted(set(cwr)))
                self.assertEqual(perm, sorted(set(perm)))
                self.assertEqual(comb, sorted(set(comb)))

                # Check interrelationships
                self.assertEqual(cwr, [t for t in prod if sorted(t)==list(t)]) # cwr: prods which are sorted
                self.assertEqual(perm, [t for t in prod if len(set(t))==r])    # perm: prods with no dups
                self.assertEqual(comb, [t for t in perm if sorted(t)==list(t)]) # comb: perms that are sorted
                self.assertEqual(comb, [t for t in cwr if len(set(t))==r])      # comb: cwrs without dups
                self.assertEqual(comb, list(filter(set(cwr).__contains__, perm)))     # comb: perm that is a cwr
                self.assertEqual(comb, list(filter(set(perm).__contains__, cwr)))     # comb: cwr that is a perm
                self.assertEqual(comb, sorted(set(cwr) & set(perm)))            # comb: both a cwr and a perm

    @pickle_deprecated
    def test_compress(self):
        self.assertEqual(list(compress(data='ABCDEF', selectors=[1,0,1,0,1,1])), list('ACEF'))
        self.assertEqual(list(compress('ABCDEF', [1,0,1,0,1,1])), list('ACEF'))
        self.assertEqual(list(compress('ABCDEF', [0,0,0,0,0,0])), list(''))
        self.assertEqual(list(compress('ABCDEF', [1,1,1,1,1,1])), list('ABCDEF'))
        self.assertEqual(list(compress('ABCDEF', [1,0,1])), list('AC'))
        self.assertEqual(list(compress('ABC', [0,1,1,1,1,1])), list('BC'))
        n = 10000
        data = chain.from_iterable(repeat(range(6), n))
        selectors = chain.from_iterable(repeat((0, 1)))
        self.assertEqual(list(compress(data, selectors)), [1,3,5] * n)
        self.assertRaises(TypeError, compress, None, range(6))      # 1st arg not iterable
        self.assertRaises(TypeError, compress, range(6), None)      # 2nd arg not iterable
        self.assertRaises(TypeError, compress, range(6))            # too few args
        self.assertRaises(TypeError, compress, range(6), None)      # too many args

        # check copy, deepcopy, pickle
        for op in [lambda a:copy.copy(a), lambda a:copy.deepcopy(a)] + picklecopiers:
            for data, selectors, result1, result2 in [
                ('ABCDEF', [1,0,1,0,1,1], 'ACEF', 'CEF'),
                ('ABCDEF', [0,0,0,0,0,0], '', ''),
                ('ABCDEF', [1,1,1,1,1,1], 'ABCDEF', 'BCDEF'),
                ('ABCDEF', [1,0,1], 'AC', 'C'),
                ('ABC', [0,1,1,1,1,1], 'BC', 'C'),
                ]:

                self.assertEqual(list(op(compress(data=data, selectors=selectors))), list(result1))
                self.assertEqual(list(op(compress(data, selectors))), list(result1))
                testIntermediate = compress(data, selectors)
                if result1:
                    next(testIntermediate)
                    self.assertEqual(list(op(testIntermediate)), list(result2))

    @pickle_deprecated
    def test_count(self):
        self.assertEqual(lzip('abc',count()), [('a', 0), ('b', 1), ('c', 2)])
        self.assertEqual(lzip('abc',count(3)), [('a', 3), ('b', 4), ('c', 5)])
        self.assertEqual(take(2, lzip('abc',count(3))), [('a', 3), ('b', 4)])
        self.assertEqual(take(2, zip('abc',count(-1))), [('a', -1), ('b', 0)])
        self.assertEqual(take(2, zip('abc',count(-3))), [('a', -3), ('b', -2)])
        self.assertRaises(TypeError, count, 2, 3, 4)
        self.assertRaises(TypeError, count, 'a')
        self.assertEqual(take(3, count(maxsize)),
                        [maxsize, maxsize + 1, maxsize + 2])
        self.assertEqual(take(10, count(maxsize-5)),
                         list(range(maxsize-5, maxsize+5)))
        self.assertEqual(take(10, count(-maxsize-5)),
                         list(range(-maxsize-5, -maxsize+5)))
        self.assertEqual(take(3, count(3.25)), [3.25, 4.25, 5.25])
        self.assertEqual(take(3, count(3.25-4j)), [3.25-4j, 4.25-4j, 5.25-4j])
        self.assertEqual(take(3, count(Decimal('1.1'))),
                         [Decimal('1.1'), Decimal('2.1'), Decimal('3.1')])
        self.assertEqual(take(3, count(Fraction(2, 3))),
                         [Fraction(2, 3), Fraction(5, 3), Fraction(8, 3)])
        BIGINT = 1<<1000
        self.assertEqual(take(3, count(BIGINT)), [BIGINT, BIGINT+1, BIGINT+2])
        c = count(3)
        self.assertEqual(repr(c), 'count(3)')
        next(c)
        self.assertEqual(repr(c), 'count(4)')
        c = count(-9)
        self.assertEqual(repr(c), 'count(-9)')
        next(c)
        self.assertEqual(next(c), -8)
        self.assertEqual(repr(count(10.25)), 'count(10.25)')
        self.assertEqual(repr(count(10.0)), 'count(10.0)')

        self.assertEqual(repr(count(maxsize)), f'count({maxsize})')
        c = count(maxsize - 1)
        self.assertEqual(repr(c), f'count({maxsize - 1})')
        next(c)  # c is now at masize
        self.assertEqual(repr(c), f'count({maxsize})')
        next(c)
        self.assertEqual(repr(c), f'count({maxsize + 1})')

        self.assertEqual(type(next(count(10.0))), float)
        for i in (-sys.maxsize-5, -sys.maxsize+5 ,-10, -1, 0, 10, sys.maxsize-5, sys.maxsize+5):
            # Test repr
            r1 = repr(count(i))
            r2 = 'count(%r)'.__mod__(i)
            self.assertEqual(r1, r2)

        # check copy, deepcopy, pickle
        for value in -3, 3, maxsize-5, maxsize+5:
            c = count(value)
            self.assertEqual(next(copy.copy(c)), value)
            self.assertEqual(next(copy.deepcopy(c)), value)
            for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                self.pickletest(proto, count(value))

        #check proper internal error handling for large "step' sizes
        count(1, maxsize+5); sys.exc_info()

    @pickle_deprecated
    def test_count_with_step(self):
        self.assertEqual(lzip('abc',count(2,3)), [('a', 2), ('b', 5), ('c', 8)])
        self.assertEqual(lzip('abc',count(start=2,step=3)),
                         [('a', 2), ('b', 5), ('c', 8)])
        self.assertEqual(lzip('abc',count(step=-1)),
                         [('a', 0), ('b', -1), ('c', -2)])
        self.assertRaises(TypeError, count, 'a', 'b')
        self.assertEqual(lzip('abc',count(2,0)), [('a', 2), ('b', 2), ('c', 2)])
        self.assertEqual(lzip('abc',count(2,1)), [('a', 2), ('b', 3), ('c', 4)])
        self.assertEqual(lzip('abc',count(2,3)), [('a', 2), ('b', 5), ('c', 8)])
        self.assertEqual(take(20, count(maxsize-15, 3)), take(20, range(maxsize-15, maxsize+100, 3)))
        self.assertEqual(take(20, count(-maxsize-15, 3)), take(20, range(-maxsize-15,-maxsize+100, 3)))
        self.assertEqual(take(3, count(10, maxsize+5)),
                         list(range(10, 10+3*(maxsize+5), maxsize+5)))
        self.assertEqual(take(3, count(maxsize, 2)),
                         [maxsize, maxsize + 2, maxsize + 4])
        self.assertEqual(take(3, count(maxsize, maxsize)),
                         [maxsize, 2 * maxsize, 3 * maxsize])
        self.assertEqual(take(3, count(-maxsize, maxsize)),
                        [-maxsize, 0, maxsize])
        self.assertEqual(take(3, count(2, 1.25)), [2, 3.25, 4.5])
        self.assertEqual(take(3, count(2, 3.25-4j)), [2, 5.25-4j, 8.5-8j])
        self.assertEqual(take(3, count(Decimal('1.1'), Decimal('.1'))),
                         [Decimal('1.1'), Decimal('1.2'), Decimal('1.3')])
        self.assertEqual(take(3, count(Fraction(2,3), Fraction(1,7))),
                         [Fraction(2,3), Fraction(17,21), Fraction(20,21)])
        BIGINT = 1<<1000
        self.assertEqual(take(3, count(step=BIGINT)), [0, BIGINT, 2*BIGINT])
        self.assertEqual(repr(take(3, count(10, 2.5))), repr([10, 12.5, 15.0]))
        c = count(3, 5)
        self.assertEqual(repr(c), 'count(3, 5)')
        next(c)
        self.assertEqual(repr(c), 'count(8, 5)')
        c = count(-9, 0)
        self.assertEqual(repr(c), 'count(-9, 0)')
        next(c)
        self.assertEqual(repr(c), 'count(-9, 0)')
        c = count(-9, -3)
        self.assertEqual(repr(c), 'count(-9, -3)')
        next(c)
        self.assertEqual(repr(c), 'count(-12, -3)')
        self.assertEqual(repr(c), 'count(-12, -3)')
        self.assertEqual(repr(count(10.5, 1.25)), 'count(10.5, 1.25)')
        self.assertEqual(repr(count(10.5, 1)), 'count(10.5)')           # suppress step=1 when it's an int
        self.assertEqual(repr(count(10.5, 1.00)), 'count(10.5, 1.0)')   # do show float values lilke 1.0
        self.assertEqual(repr(count(10, 1.00)), 'count(10, 1.0)')
        c = count(10, 1.0)
        self.assertEqual(type(next(c)), int)
        self.assertEqual(type(next(c)), float)
        for i in (-sys.maxsize-5, -sys.maxsize+5 ,-10, -1, 0, 10, sys.maxsize-5, sys.maxsize+5):
            for j in  (-sys.maxsize-5, -sys.maxsize+5 ,-10, -1, 0, 1, 10, sys.maxsize-5, sys.maxsize+5):
                # Test repr
                r1 = repr(count(i, j))
                if j == 1:
                    r2 = ('count(%r)' % i)
                else:
                    r2 = ('count(%r, %r)' % (i, j))
                self.assertEqual(r1, r2)
                for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.pickletest(proto, count(i, j))

        c = count(maxsize -2, 2)
        self.assertEqual(repr(c), f'count({maxsize - 2}, 2)')
        next(c)  # c is now at masize
        self.assertEqual(repr(c), f'count({maxsize}, 2)')
        next(c)
        self.assertEqual(repr(c), f'count({maxsize + 2}, 2)')

        c = count(maxsize + 1, -1)
        self.assertEqual(repr(c), f'count({maxsize + 1}, -1)')
        next(c)  # c is now at masize
        self.assertEqual(repr(c), f'count({maxsize}, -1)')
        next(c)
        self.assertEqual(repr(c), f'count({maxsize - 1}, -1)')

    @threading_helper.requires_working_threading()
    def test_count_threading(self, step=1):
        # this test verifies multithreading consistency, which is
        # mostly for testing builds without GIL, but nice to test anyway
        count_to = 10_000
        num_threads = 10
        c = count(step=step)
        def counting_thread():
            for i in range(count_to):
                next(c)
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=counting_thread)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        self.assertEqual(next(c), count_to * num_threads * step)

    def test_count_with_step_threading(self):
        self.test_count_threading(step=5)

    def test_cycle(self):
        self.assertEqual(take(10, cycle('abc')), list('abcabcabca'))
        self.assertEqual(list(cycle('')), [])
        # self.assertRaises(TypeError, cycle)
        self.assertRaises(TypeError, cycle, 5)
        self.assertEqual(list(islice(cycle(gen3()),10)), [0,1,2,0,1,2,0,1,2,0])

    @pickle_deprecated
    def test_cycle_copy_pickle(self):
        # check copy, deepcopy, pickle
        c = cycle('abc')
        self.assertEqual(next(c), 'a')
        #simple copy currently not supported, because __reduce__ returns
        #an internal iterator
        #self.assertEqual(take(10, copy.copy(c)), list('bcabcabcab'))
        self.assertEqual(take(10, copy.deepcopy(c)), list('bcabcabcab'))
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertEqual(take(10, pickle.loads(pickle.dumps(c, proto))),
                             list('bcabcabcab'))
            next(c)
            self.assertEqual(take(10, pickle.loads(pickle.dumps(c, proto))),
                             list('cabcabcabc'))
            next(c)
            next(c)
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            self.pickletest(proto, cycle('abc'))

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            # test with partial consumed input iterable
            it = iter('abcde')
            c = cycle(it)
            _ = [next(c) for i in range(2)]      # consume 2 of 5 inputs
            p = pickle.dumps(c, proto)
            d = pickle.loads(p)                  # rebuild the cycle object
            self.assertEqual(take(20, d), list('cdeabcdeabcdeabcdeab'))

            # test with completely consumed input iterable
            it = iter('abcde')
            c = cycle(it)
            _ = [next(c) for i in range(7)]      # consume 7 of 5 inputs
            p = pickle.dumps(c, proto)
            d = pickle.loads(p)                  # rebuild the cycle object
            self.assertEqual(take(20, d), list('cdeabcdeabcdeabcdeab'))

    @pickle_deprecated
    def test_cycle_unpickle_compat(self):
        testcases = [
            b'citertools\ncycle\n(c__builtin__\niter\n((lI1\naI2\naI3\natRI1\nbtR((lI1\naI0\ntb.',
            b'citertools\ncycle\n(c__builtin__\niter\n(](K\x01K\x02K\x03etRK\x01btR(]K\x01aK\x00tb.',
            b'\x80\x02citertools\ncycle\nc__builtin__\niter\n](K\x01K\x02K\x03e\x85RK\x01b\x85R]K\x01aK\x00\x86b.',
            b'\x80\x03citertools\ncycle\ncbuiltins\niter\n](K\x01K\x02K\x03e\x85RK\x01b\x85R]K\x01aK\x00\x86b.',
            b'\x80\x04\x95=\x00\x00\x00\x00\x00\x00\x00\x8c\titertools\x8c\x05cycle\x93\x8c\x08builtins\x8c\x04iter\x93](K\x01K\x02K\x03e\x85RK\x01b\x85R]K\x01aK\x00\x86b.',

            b'citertools\ncycle\n(c__builtin__\niter\n((lp0\nI1\naI2\naI3\natRI1\nbtR(g0\nI1\ntb.',
            b'citertools\ncycle\n(c__builtin__\niter\n(]q\x00(K\x01K\x02K\x03etRK\x01btR(h\x00K\x01tb.',
            b'\x80\x02citertools\ncycle\nc__builtin__\niter\n]q\x00(K\x01K\x02K\x03e\x85RK\x01b\x85Rh\x00K\x01\x86b.',
            b'\x80\x03citertools\ncycle\ncbuiltins\niter\n]q\x00(K\x01K\x02K\x03e\x85RK\x01b\x85Rh\x00K\x01\x86b.',
            b'\x80\x04\x95<\x00\x00\x00\x00\x00\x00\x00\x8c\titertools\x8c\x05cycle\x93\x8c\x08builtins\x8c\x04iter\x93]\x94(K\x01K\x02K\x03e\x85RK\x01b\x85Rh\x00K\x01\x86b.',

            b'citertools\ncycle\n(c__builtin__\niter\n((lI1\naI2\naI3\natRI1\nbtR((lI1\naI00\ntb.',
            b'citertools\ncycle\n(c__builtin__\niter\n(](K\x01K\x02K\x03etRK\x01btR(]K\x01aI00\ntb.',
            b'\x80\x02citertools\ncycle\nc__builtin__\niter\n](K\x01K\x02K\x03e\x85RK\x01b\x85R]K\x01a\x89\x86b.',
            b'\x80\x03citertools\ncycle\ncbuiltins\niter\n](K\x01K\x02K\x03e\x85RK\x01b\x85R]K\x01a\x89\x86b.',
            b'\x80\x04\x95<\x00\x00\x00\x00\x00\x00\x00\x8c\titertools\x8c\x05cycle\x93\x8c\x08builtins\x8c\x04iter\x93](K\x01K\x02K\x03e\x85RK\x01b\x85R]K\x01a\x89\x86b.',

            b'citertools\ncycle\n(c__builtin__\niter\n((lp0\nI1\naI2\naI3\natRI1\nbtR(g0\nI01\ntb.',
            b'citertools\ncycle\n(c__builtin__\niter\n(]q\x00(K\x01K\x02K\x03etRK\x01btR(h\x00I01\ntb.',
            b'\x80\x02citertools\ncycle\nc__builtin__\niter\n]q\x00(K\x01K\x02K\x03e\x85RK\x01b\x85Rh\x00\x88\x86b.',
            b'\x80\x03citertools\ncycle\ncbuiltins\niter\n]q\x00(K\x01K\x02K\x03e\x85RK\x01b\x85Rh\x00\x88\x86b.',
            b'\x80\x04\x95;\x00\x00\x00\x00\x00\x00\x00\x8c\titertools\x8c\x05cycle\x93\x8c\x08builtins\x8c\x04iter\x93]\x94(K\x01K\x02K\x03e\x85RK\x01b\x85Rh\x00\x88\x86b.',
        ]
        assert len(testcases) == 20
        for t in testcases:
            it = pickle.loads(t)
            self.assertEqual(take(10, it), [2, 3, 1, 2, 3, 1, 2, 3, 1, 2])

    @pickle_deprecated
    def test_cycle_setstate(self):
        # Verify both modes for restoring state

        # Mode 0 is efficient.  It uses an incompletely consumed input
        # iterator to build a cycle object and then passes in state with
        # a list of previously consumed values.  There is no data
        # overlap between the two.
        c = cycle('defg')
        c.__setstate__((list('abc'), 0))
        self.assertEqual(take(20, c), list('defgabcdefgabcdefgab'))

        # Mode 1 is inefficient.  It starts with a cycle object built
        # from an iterator over the remaining elements in a partial
        # cycle and then passes in state with all of the previously
        # seen values (this overlaps values included in the iterator).
        c = cycle('defg')
        c.__setstate__((list('abcdefg'), 1))
        self.assertEqual(take(20, c), list('defgabcdefgabcdefgab'))

        # The first argument to setstate needs to be a tuple
        with self.assertRaises(TypeError):
            cycle('defg').__setstate__([list('abcdefg'), 0])

        # The first argument in the setstate tuple must be a list
        with self.assertRaises(TypeError):
            c = cycle('defg')
            c.__setstate__((tuple('defg'), 0))
        take(20, c)

        # The second argument in the setstate tuple must be an int
        with self.assertRaises(TypeError):
            cycle('defg').__setstate__((list('abcdefg'), 'x'))

        self.assertRaises(TypeError, cycle('').__setstate__, ())
        self.assertRaises(TypeError, cycle('').__setstate__, ([],))

    @pickle_deprecated
    def test_groupby(self):
        # Check whether it accepts arguments correctly
        self.assertEqual([], list(groupby([])))
        self.assertEqual([], list(groupby([], key=id)))
        self.assertRaises(TypeError, list, groupby('abc', []))
        self.assertRaises(TypeError, groupby, None)
        self.assertRaises(TypeError, groupby, 'abc', lambda x:x, 10)

        # Check normal input
        s = [(0, 10, 20), (0, 11,21), (0,12,21), (1,13,21), (1,14,22),
             (2,15,22), (3,16,23), (3,17,23)]
        dup = []
        for k, g in groupby(s, lambda r:r[0]):
            for elem in g:
                self.assertEqual(k, elem[0])
                dup.append(elem)
        self.assertEqual(s, dup)

        # Check normal pickled
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            dup = []
            for k, g in pickle.loads(pickle.dumps(groupby(s, _testR), proto)):
                for elem in g:
                    self.assertEqual(k, elem[0])
                    dup.append(elem)
            self.assertEqual(s, dup)

        # Check nested case
        dup = []
        for k, g in groupby(s, _testR):
            for ik, ig in groupby(g, _testR2):
                for elem in ig:
                    self.assertEqual(k, elem[0])
                    self.assertEqual(ik, elem[2])
                    dup.append(elem)
        self.assertEqual(s, dup)

        # Check nested and pickled
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            dup = []
            for k, g in pickle.loads(pickle.dumps(groupby(s, _testR), proto)):
                for ik, ig in pickle.loads(pickle.dumps(groupby(g, _testR2), proto)):
                    for elem in ig:
                        self.assertEqual(k, elem[0])
                        self.assertEqual(ik, elem[2])
                        dup.append(elem)
            self.assertEqual(s, dup)


        # Check case where inner iterator is not used
        keys = [k for k, g in groupby(s, _testR)]
        expectedkeys = set([r[0] for r in s])
        self.assertEqual(set(keys), expectedkeys)
        self.assertEqual(len(keys), len(expectedkeys))

        # Check case where inner iterator is used after advancing the groupby
        # iterator
        s = list(zip('AABBBAAAA', range(9)))
        it = groupby(s, _testR)
        _, g1 = next(it)
        _, g2 = next(it)
        _, g3 = next(it)
        self.assertEqual(list(g1), [])
        self.assertEqual(list(g2), [])
        self.assertEqual(next(g3), ('A', 5))
        list(it)  # exhaust the groupby iterator
        self.assertEqual(list(g3), [])

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            it = groupby(s, _testR)
            _, g = next(it)
            next(it)
            next(it)
            self.assertEqual(list(pickle.loads(pickle.dumps(g, proto))), [])

        # Exercise pipes and filters style
        s = 'abracadabra'
        # sort s | uniq
        r = [k for k, g in groupby(sorted(s))]
        self.assertEqual(r, ['a', 'b', 'c', 'd', 'r'])
        # sort s | uniq -d
        r = [k for k, g in groupby(sorted(s)) if list(islice(g,1,2))]
        self.assertEqual(r, ['a', 'b', 'r'])
        # sort s | uniq -c
        r = [(len(list(g)), k) for k, g in groupby(sorted(s))]
        self.assertEqual(r, [(5, 'a'), (2, 'b'), (1, 'c'), (1, 'd'), (2, 'r')])
        # sort s | uniq -c | sort -rn | head -3
        r = sorted([(len(list(g)) , k) for k, g in groupby(sorted(s))], reverse=True)[:3]
        self.assertEqual(r, [(5, 'a'), (2, 'r'), (2, 'b')])

        # iter.__next__ failure
        class ExpectedError(Exception):
            pass
        def delayed_raise(n=0):
            for i in range(n):
                yield 'yo'
            raise ExpectedError
        def gulp(iterable, keyp=None, func=list):
            return [func(g) for k, g in groupby(iterable, keyp)]

        # iter.__next__ failure on outer object
        self.assertRaises(ExpectedError, gulp, delayed_raise(0))
        # iter.__next__ failure on inner object
        self.assertRaises(ExpectedError, gulp, delayed_raise(1))

        # __eq__ failure
        class DummyCmp:
            def __eq__(self, dst):
                raise ExpectedError
        s = [DummyCmp(), DummyCmp(), None]

        # __eq__ failure on outer object
        self.assertRaises(ExpectedError, gulp, s, func=id)
        # __eq__ failure on inner object
        self.assertRaises(ExpectedError, gulp, s)

        # keyfunc failure
        def keyfunc(obj):
            if keyfunc.skip > 0:
                keyfunc.skip -= 1
                return obj
            else:
                raise ExpectedError

        # keyfunc failure on outer object
        keyfunc.skip = 0
        self.assertRaises(ExpectedError, gulp, [None], keyfunc)
        keyfunc.skip = 1
        self.assertRaises(ExpectedError, gulp, [None, None], keyfunc)

    def test_filter(self):
        self.assertEqual(list(filter(isEven, range(6))), [0,2,4])
        self.assertEqual(list(filter(None, [0,1,0,2,0])), [1,2])
        self.assertEqual(list(filter(bool, [0,1,0,2,0])), [1,2])
        self.assertEqual(take(4, filter(isEven, count())), [0,2,4,6])
        # these tests raise dynamo exceptions, not TypeError
        # self.assertRaises(TypeError, filter)
        # self.assertRaises(TypeError, filter, lambda x:x)
        # self.assertRaises(TypeError, filter, lambda x:x, range(6), 7)
        # self.assertRaises(TypeError, filter, isEven, 3)
        # dynamo raises Unsupported in this case
        # self.assertRaises(TypeError, next, filter(range(6), range(6)))

        # check copy, deepcopy, pickle
        # ans = [0,2,4]

        # c = filter(isEven, range(6))
        # self.assertEqual(list(copy.copy(c)), ans)
        # c = filter(isEven, range(6))
        # self.assertEqual(list(copy.deepcopy(c)), ans)
        # for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        #     c = filter(isEven, range(6))
        #     self.assertEqual(list(pickle.loads(pickle.dumps(c, proto))), ans)
        #     next(c)
        #     self.assertEqual(list(pickle.loads(pickle.dumps(c, proto))), ans[1:])
        # for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        #     c = filter(isEven, range(6))
        #     self.pickletest(proto, c)

    def test_filterfalse(self):
        self.assertEqual(list(filterfalse(isEven, range(6))), [1,3,5])
        self.assertEqual(list(filterfalse(None, [0,1,0,2,0])), [0,0,0])
        self.assertEqual(list(filterfalse(bool, [0,1,0,2,0])), [0,0,0])
        self.assertEqual(take(4, filterfalse(isEven, count())), [1,3,5,7])
        self.assertRaises(TypeError, filterfalse)
        self.assertRaises(TypeError, filterfalse, lambda x:x)
        self.assertRaises(TypeError, filterfalse, lambda x:x, range(6), 7)
        self.assertRaises(TypeError, filterfalse, isEven, 3)
        with torch._dynamo.error_on_graph_break(False):
            self.assertRaises(TypeError, next, filterfalse(range(6), range(6)))
            for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                self.pickletest(proto, filterfalse(isEven, range(6)))

    def test_zip(self):
        # XXX This is rather silly now that builtin zip() calls zip()...
        ans = [(x,y) for x, y in zip('abc',count())]
        self.assertEqual(ans, [('a', 0), ('b', 1), ('c', 2)])
        self.assertEqual(list(zip('abc', range(6))), lzip('abc', range(6)))
        self.assertEqual(list(zip('abcdef', range(3))), lzip('abcdef', range(3)))
        self.assertEqual(take(3,zip('abcdef', count())), lzip('abcdef', range(3)))
        self.assertEqual(list(zip('abcdef')), lzip('abcdef'))
        self.assertEqual(list(zip()), lzip())
        # self.assertRaises(TypeError, zip, 3)
        # self.assertRaises(TypeError, zip, range(3), 3)
        self.assertEqual([tuple(list(pair)) for pair in zip('abc', 'def')],
                         lzip('abc', 'def'))
        self.assertEqual([pair for pair in zip('abc', 'def')],
                         lzip('abc', 'def'))

    @support.impl_detail("tuple reuse is specific to CPython")
    @pickle_deprecated
    def test_zip_tuple_reuse(self):
        ids = list(map(id, zip('abc', 'def')))
        self.assertEqual(min(ids), max(ids))
        ids = list(map(id, list(zip('abc', 'def'))))
        self.assertEqual(len(dict.fromkeys(ids)), len(ids))

        # check copy, deepcopy, pickle
        ans = [(x,y) for x, y in copy.copy(zip('abc',count()))]
        self.assertEqual(ans, [('a', 0), ('b', 1), ('c', 2)])

        ans = [(x,y) for x, y in copy.deepcopy(zip('abc',c
```



## High-Level Overview

""" Run the test three times.    First, verify that a Deprecation Warning is raised.    Second, run normally but with DeprecationWarnings temporarily disabled.    Third, run with warnings promoted to errors.

This Python file contains 32 class(es) and 233 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `StopNow`, `TestBasicOps`, `ExpectedError`, `DummyCmp`, `BadIterable`, `Repeater`, `I`, `I`, `IntLike`, `I`, `I`, `TestExamples`, `TestPurePythonRoughEquivalents`, `_tee`, `TestGC`, `G`, `I`, `Ig`, `X`, `N`

**Functions defined**: `pickle_deprecated`, `inner`, `choice`, `randrange`, `lzip`, `onearg`, `errfunc`, `gen3`, `isEven`, `isOdd`, `tupleize`, `irange`, `__iter__`, `__next__`, `take`, `prod`, `fact`, `_testR`, `_testR2`, `underten`

**Key imports**: torch, torch._dynamo.test_case, CPythonTestCase, doctest, unittest, itertools, support, threading_helper, script_helper, weakref, Decimal


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/dynamo/cpython/3_13`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch._dynamo.test_case`
- `doctest`
- `unittest`
- `itertools`
- `test`: support
- `test.support`: threading_helper, script_helper
- `weakref`
- `decimal`: Decimal
- `fractions`: Fraction
- `operator`
- `random`
- `copy`
- `pickle`
- `functools`: reduce
- `sys`
- `struct`
- `threading`
- `gc`
- `warnings`
- `typing, copyreg, itertools`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Abstract Base Classes**: Defines abstract interfaces
- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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
python test/dynamo/cpython/3_13/test_itertools.py
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

- **File Documentation**: `test_itertools.py_docs.md`
- **Keyword Index**: `test_itertools.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
