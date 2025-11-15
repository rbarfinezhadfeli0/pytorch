# Documentation: `test/jit/test_freezing.py`

## File Metadata

- **Path**: `test/jit/test_freezing.py`
- **Size**: 122,116 bytes (119.25 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]
# ruff: noqa: F841

import io
import unittest
from itertools import product
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit._recursive import wrap_cpp_module
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_CUDNN
from torch.testing._internal.common_quantization import skipIfNoFBGEMM
from torch.testing._internal.common_quantized import override_quantized_engine
from torch.testing._internal.common_utils import (
    raise_on_run_directly,
    set_default_dtype,
    skipCUDAMemoryLeakCheckIf,
    skipIfTorchDynamo,
    TEST_WITH_ROCM,
)
from torch.testing._internal.jit_utils import JitTestCase
from torch.utils import mkldnn as mkldnn_utils


try:
    import torchvision

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

TEST_ROCM = torch.cuda.is_available() and torch.version.hip is not None


def removeExceptions(graph):
    for n in graph.findAllNodes("prim::RaiseException"):
        n.destroy()


class TestFreezing(JitTestCase):
    def test_freeze_module(self):
        class M(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 1  # folded
                self.b = 1.2  # folded
                self.c = "hello"  # folded
                self.c2 = "hi\xa1"  # not folded
                self.d = [1, 1]  # folded
                self.e = [1.0, 1.1]  # folded
                self.f = ["hello", "world"]  # folded
                self.f2 = [(1, "Over \u0e55\u0e57 57")]
                self.g = (
                    [1, 2],
                    3.2,
                    "4.4",
                    torch.tensor([5.5], requires_grad=True),
                )  # folded
                self.h = {"layer": [torch.tensor([7.7], requires_grad=True)]}
                self.h2 = {"layer\xb1": [torch.tensor([8.8], requires_grad=True)]}
                self.t = torch.tensor([1.2, 2.4], requires_grad=True)  # folded
                self.ts = [
                    torch.tensor([1.0, 2.0], requires_grad=True),
                    torch.tensor([3.0, 4.0], requires_grad=True),
                ]  # folded
                self.tt = [[torch.tensor([3.3, 2.3], requires_grad=True), None]]

            def forward(self, x):
                return (
                    str(self.a)
                    + str(self.b)
                    + self.c
                    + self.c2
                    + str(self.d)
                    + str(self.e)
                    + str(self.f)
                    + str(self.f2)
                    + str(self.g)
                    + str(self.h)
                    + str(self.h2)
                    + str(self.t)
                    + str(self.ts)
                    + str(self.tt)
                )

        m = torch.jit.script(M())
        m.eval()
        input = torch.randn(2, 2)
        output_s = m.forward(input)
        m._c = torch._C._freeze_module(m._c)
        buffer = io.BytesIO()
        torch.jit.save(m._c, buffer)
        buffer.seek(0)
        m2 = torch.jit.load(buffer)
        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #     tt = ...
        #   }
        #   ...
        # }
        self.assertFalse(m2._c.hasattr("a"))
        self.assertFalse(m2._c.hasattr("b"))
        self.assertFalse(m2._c.hasattr("c"))
        self.assertFalse(m2._c.hasattr("c2"))
        self.assertFalse(m2._c.hasattr("d"))
        self.assertFalse(m2._c.hasattr("e"))
        self.assertFalse(m2._c.hasattr("f"))
        self.assertFalse(m2._c.hasattr("f2"))
        self.assertFalse(m2._c.hasattr("g"))
        self.assertFalse(m2._c.hasattr("h"))
        self.assertFalse(m2._c.hasattr("h2"))
        self.assertFalse(m2._c.hasattr("t"))
        self.assertFalse(m2._c.hasattr("ts"))
        self.assertFalse(m2._c.hasattr("tt"))
        output_f = m2.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_submodule(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 11
                self.b = 2

            def forward(self, x):
                return self.a + self.b

        class SubModule2(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 12
                self.b = 2

            def forward(self, x):
                self.b = 30
                return self.a + self.b

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub1 = SubModule()
                self.sub2 = SubModule2()
                self.a = 3
                self.b = 4

            def forward(self, x):
                self.b = 20
                return self.sub1(x) + self.a + self.b + self.sub2(x)

        m = torch.jit.script(TestModule())
        m.eval()
        input = torch.randn(2, 2)
        output_s = m.forward(input)
        mf = torch.jit.freeze(m)

        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #     sub2 = ...
        #      b =
        #   }
        #   ...
        #   submodule {
        #     module m {
        #       attributes {
        #         sub2 = ...
        #         b =
        #       }
        #       ...
        #     }
        #   }
        # }
        mf = mf._c
        self.assertFalse(mf.hasattr("sub1"))
        self.assertFalse(mf.hasattr("a"))
        self.assertTrue(mf.hasattr("b"))
        self.assertTrue(mf.hasattr("sub2"))
        self.assertTrue(mf.sub2.hasattr("b"))  # verify b is preserved in sub2
        self.assertFalse(mf.sub2.hasattr("a"))  # verify a is removed in sub2
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_fork(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.ones(20, 20)
                self.b = torch.ones(20, 20)

            def forward(self, x):
                return self.a * self.b + x

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = SubModule()

            def forward(self, x):
                fut = torch.jit._fork(self.sub.forward, x)
                y_hat = self.sub(x)
                y = torch.jit._wait(fut)
                return y_hat + y

        m = torch.jit.script(TestModule())
        m.eval()
        input = torch.randn(20, 20)
        output_s = m.forward(input)
        mf = torch._C._freeze_module(m._c)

        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #   }
        #   ...
        #   submodule {
        #   }
        # }
        self.assertFalse(mf.hasattr("a"))
        self.assertFalse(mf.hasattr("b"))
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_nested_fork(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.ones(20, 20)
                self.b = torch.ones(20, 20)

            def forward(self, x):
                return self.a * self.b + x

        class SubModule2(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = SubModule()
                self.c = torch.ones(20, 20)

            def forward(self, x):
                fut = torch.jit._fork(self.sub.forward, x)
                y_hat = self.sub(x)
                y = torch.jit._wait(fut)
                return y_hat + y + self.c

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = SubModule2()
                self.d = 1

            def forward(self, x):
                fut = torch.jit._fork(self.sub.forward, x)
                y_hat = self.sub(x)
                y = torch.jit._wait(fut)
                self.d = 2
                return y_hat * y + self.d

        m = torch.jit.script(TestModule())
        m.eval()
        input = torch.randn(20, 20)
        output_s = m.forward(input)
        mf = torch._C._freeze_module(m._c)
        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #   }
        #   ...
        #   submodule {
        #   }
        # }
        self.assertFalse(mf.hasattr("a"))
        self.assertFalse(mf.hasattr("b"))
        self.assertFalse(mf.hasattr("c"))
        self.assertTrue(mf.hasattr("d"))
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_fork2(self):
        @torch.jit.script
        def foo(x):
            return x * 2

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.ones(20, 20)
                self.b = torch.ones(20, 20)

            def forward(self, x):
                fut = torch.jit._fork(foo, self.a)
                y_hat = foo(self.b)
                y = torch.jit._wait(fut)
                return y_hat + y

        m = torch.jit.script(TestModule())
        m.eval()
        input = torch.randn(2, 2)
        output_s = m.forward(input)
        mf = torch._C._freeze_module(m._c)

        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #     self.a = ...
        #     self.b = ..
        #   }
        #   ...
        #   submodule {
        #   }
        # }
        # TODO:  Although there are no mutation, the alias analysis
        # conservatively assumes there is a mutation because attributes are
        # passed to fork subgraph. both 'a' and 'b' are preserved.
        self.assertTrue(mf.hasattr("a"))
        self.assertFalse(mf.hasattr("b"))
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_fork_calling_module_method(self):
        @torch.jit.script
        def foo(x, y):
            return x * y

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.ones(20, 20)
                self.b = torch.ones(20, 20)

            @torch.jit.export
            def foo(self, x):
                return x * self.a

            @torch.jit.export
            def bar(self, x):
                return x * self.b

            def forward(self, x):
                fut = torch.jit._fork(self.foo, self.b)
                y_hat = self.bar(self.a)
                y = torch.jit._wait(fut)
                return y_hat + y

        m = torch.jit.script(TestModule())
        m.eval()
        input = torch.randn(2, 2)
        output_s = m.forward(input)
        mf = torch._C._freeze_module(m._c)
        # Check if frozen module looks as below:
        # module m {
        #   attributes {
        #     self.b = ..
        #   }
        #   ...
        # TODO:  Although there are no mutation, the alias analysis
        # conservatively assumes there is a mutation because attributes are
        # passed to fork subgraph. 'b' is preserved.
        self.assertFalse(mf.hasattr("a"))
        self.assertTrue(mf.hasattr("b"))
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_sharedclasstype(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor([1.1])
                self.b = torch.tensor([2.2])

            def forward(self, x):
                return self.a + self.b

            @torch.jit.export
            def modify_a(self, x):
                self.a[0] += 10
                return self.b

            @torch.jit.export
            def modify_b(self, x):
                self.b[0] += 20
                return self.a

        class SubModule2(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = SubModule()
                self.b = torch.tensor([3.3])

            def forward(self, x):
                y = self.sub.modify_b(x)
                return y + self.b

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub1 = SubModule()  # sub1 and sub2.sub shared same class type.
                self.sub2 = SubModule2()
                self.a = torch.tensor([4.4])

            def forward(self, x):
                z = self.sub1.modify_a(x)
                return self.sub2(x) + z + self.a

        m = torch.jit.script(TestModule())
        m.eval()
        input = torch.randn(2, 2)
        output_s = m.forward(input)
        mf = torch._C._freeze_module(m._c)

        # Checking if  Frozen module looks as  below
        # module mf {
        #   attributes {
        #     sub1 = ...
        #     sub2 = ...
        #   }
        #   ...
        #   submodules {
        #     module sub1 {
        #       attributes {
        #         a = ...
        #         b = ...
        #       }
        #       ...
        #     }
        #     module sub2 {
        #       attributes {
        #         sub = ...
        #       }
        #       ...
        #       submodule {
        #         module sub {
        #           attributes {
        #             a = ...
        #             b = ...
        #           }
        #           ...
        #         }
        #       }
        #     }
        #   }
        # }

        self.assertTrue(mf.hasattr("sub1"))
        self.assertTrue(mf.sub1.hasattr("a"))
        self.assertTrue(mf.sub1.hasattr("b"))
        self.assertFalse(mf.hasattr("a"))
        self.assertTrue(mf.hasattr("sub2"))
        self.assertTrue(mf.sub2.hasattr("sub"))
        self.assertFalse(mf.sub2.hasattr("b"))
        self.assertTrue(mf.sub2.sub.hasattr("a"))
        self.assertTrue(mf.sub2.sub.hasattr("b"))
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_nestedaliasing(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor([1.1])
                self.b = torch.tensor([2.2])

            def forward(self, x):
                return self.a + self.b

            @torch.jit.export
            def modify_a(self, x):
                self.a[0] = 10
                return self.b

            @torch.jit.export
            def modify_b(self, x):
                self.b[0] = 20
                return self.a

        Sub = SubModule()

        class SubModule2(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = Sub  # aliasing

            def forward(self, x):
                return self.sub.a

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub1 = Sub  # aliasing
                self.sub2 = SubModule2()

            def forward(self, x):
                z = self.sub1.modify_a(x)
                return self.sub2(x) + z

        m = torch.jit.script(TestModule())
        m.eval()
        mf = torch._C._freeze_module(m._c)
        self.assertTrue(mf.hasattr("sub1"))
        self.assertTrue(mf.sub1.hasattr("a"))
        self.assertFalse(mf.sub1.hasattr("b"))
        self.assertTrue(mf.hasattr("sub2"))
        self.assertTrue(mf.sub2.hasattr("sub"))
        self.assertTrue(
            mf.sub2.sub.hasattr("a")
        )  # Freezing detects that self.sub2.sub.a and self.sub1.a are alias
        self.assertFalse(mf.sub2.sub.hasattr("b"))
        input = torch.randn(2, 2)
        output_s = m.forward(input)
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    # FIXME: JIT is not honoring aliasing. 'Sub' module is copied. As a result
    # Eager and Script modules produce different output.
    def test_freeze_module_with_nestedaliasingscalar(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 1.1
                self.b = 2.2

            def forward(self, x):
                return self.a + self.b

            @torch.jit.export
            def modify_a(self, x):
                self.a = 10.0
                return self.b

            @torch.jit.export
            def modify_b(self, x):
                self.b = 20.0
                return self.a

        Sub = SubModule()

        class SubModule2(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = Sub  # aliasing

            def forward(self, x):
                return self.sub.a

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub1 = Sub  # aliasing
                self.sub2 = SubModule2()

            def forward(self, x):
                z = self.sub1.modify_a(x)
                return self.sub2(x) + z

        m = TestModule()
        ms = torch.jit.script(m)
        ms.eval()
        mf = torch._C._freeze_module(ms._c)
        self.assertTrue(mf.hasattr("sub1"))
        self.assertTrue(mf.sub1.hasattr("a"))
        self.assertFalse(mf.sub1.hasattr("b"))
        # sub2 is fully folded because self.sub1 and self.sub2.sub are not alias (Scripting bug)
        self.assertFalse(mf.hasattr("sub2"))
        input = torch.randn(2, 2)
        output = m.forward(input)
        output_s = ms.forward(input)
        output_f = mf.forward(input)
        # Should be equal
        self.assertNotEqual(output, output_s)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_preserve_sub_module(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor([1.1])
                self.b = 2.2

            def forward(self, x):
                return self.a

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub1 = SubModule()  # aliasing
                self.sub2 = SubModule()

            def forward(self, x):
                return self.sub2(x) + self.sub1(x)

        m = TestModule()
        ms = torch.jit.script(m)
        ms.eval()
        mf = torch._C._freeze_module(ms._c, ["sub1"])

        # Test that 'sub1' is preserved entirely and 'sub2' is completely folded
        self.assertTrue(mf.hasattr("sub1"))
        self.assertTrue(mf.sub1.hasattr("a"))
        self.assertTrue(mf.sub1.hasattr("b"))
        self.assertFalse(mf.hasattr("sub2"))
        input = torch.randn(2, 2)
        output_s = ms.forward(input)
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_preserve_sub_module_and_mutation(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor([1.1])
                self.b = 2.2

            def forward(self, x):
                self.a[0] = 3.3
                return self.a

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub1 = SubModule()  # aliasing
                self.sub2 = SubModule()

            def forward(self, x):
                return self.sub2(x) + self.sub1(x)

        m = TestModule()
        ms = torch.jit.script(m)
        ms.eval()
        mf = torch._C._freeze_module(ms._c, ["sub1"])

        # Test that be both sub1 and sub1 are preserved and 'b' is preserved
        # even if it is not used. To fulfill user request to preserve 'sub1'
        self.assertTrue(mf.hasattr("sub1"))
        self.assertTrue(mf.sub1.hasattr("a"))
        self.assertTrue(mf.sub1.hasattr("b"))
        self.assertTrue(mf.hasattr("sub2"))
        self.assertTrue(mf.sub2.hasattr("a"))
        self.assertTrue(mf.sub2.hasattr("b"))
        input = torch.randn(2, 2)
        output_s = ms.forward(input)
        output_f = mf.forward(input)
        self.assertEqual(output_s, output_f)

    def test_freeze_module_with_helperfunction(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 11
                self.b = 2

            def forward(self, x):
                return self.a + self.b

        class TestModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = SubModule()
                self.a = 3
                self.b = 4

            def forward(self, x):
                self.b = 20
                return self._forward(x) + self.a + self.b

            def _forward(self, x):
                return self.sub(x)

        m = torch.jit.script(TestModule())
        m.eval()
        input = torch.randn(2, 2)
        mf = torch._C._freeze_module(m._c)
        self.assertFalse(mf.hasattr("sub"))
        self.assertFalse(mf.hasattr("a"))
        self.assertTrue(mf.hasattr("b"))
        with self.assertRaisesRegex(
            AttributeError, "TestModule (.*) does not have a field with name '_forward'"
        ):
            mf._forward(x)  # noqa: F821

    def test_freeze_module_with_inplace_mutable(self):
        class FreezeMe(torch.jit.ScriptModule):
            def __init__(self) -> None:
                super().__init__()
                self.a = [11, 22]

            @torch.jit.script_method
            def forward(self, x):
                for i in range(3):
                    self.a.append(i)
                return self.a

        m = FreezeMe()
        m.eval()
        m_f = torch._C._freeze_module(m._c)
        self.assertTrue(m_f.hasattr("a"))
        m.forward(torch.tensor([3]))
        out = m_f.forward(torch.tensor([5]))
        expected = [11, 22, 0, 1, 2, 0, 1, 2]
        self.assertEqual(out, expected)

    # Mutable attributes
    def test_freeze_module_with_mutable_list(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = [1, 2]

            def forward(self, x):
                return self.a

        m = FreezeMe()
        m.eval()
        m.a.append(3)
        m_s = torch.jit.script(m)
        v = m_s.a
        v.append(4)
        m_s.a = v
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        # Post-freezing mutating m_s.a  does not affect m_f (m_f has its own copy).
        v = m_s.a
        v.append(5)
        m_s.a = v
        self.assertFalse(m_f.hasattr("a"))
        out = m_f.forward(torch.tensor([5]))
        expected = [1, 2, 3, 4]
        self.assertEqual(out, expected)

    def test_freeze_module_with_mutable_dict(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = {"layer": "4"}

            def forward(self, x):
                return self.a

            @torch.jit.export
            def modify_a(self, x):
                self.a["layer"] = self.a["layer"] + "1"
                return self.a

        m = FreezeMe()
        m.eval()
        m.a["layer2"] = "3"
        m_s = torch.jit.script(m)
        t = torch.tensor(5)
        m_s.modify_a(t)
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        m.a["layer2"] += "2"
        m_s.modify_a(t)
        self.assertFalse(m_f.hasattr("a"))
        out = m_f.forward(t)
        expected = {"layer": "411", "layer2": "3"}
        self.assertEqual(out, expected)

    def test_freeze_module_with_mutable_tensor(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor([1.0, 2.0, 3.0])

            def forward(self, x):
                return self.a

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.a[1] += 3.0
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        # Post-freezing tensor attribute mutations affect m_f.
        # FIXME: deep copy all folded attributes so that m_f has full ownership.
        m_s.a[0] += 5.0
        self.assertFalse(m_f.hasattr("a"))
        out = m_f.forward(torch.tensor([5]))
        expected = [6.0, 5.0, 3.0]
        self.assertEqual(out, expected)

    def test_freeze_module_with_tuple(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = (torch.tensor([1, 2, 3, 4, 5, 6]), "hi")

            def forward(self, x):
                if x[0] == 2.0:
                    self.a[0][0] = 10
                return self.a[0].sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        inp = torch.tensor([2.0])
        expected = m_s.forward(inp)
        m_s.a[0][0] = 1
        m_f = torch._C._freeze_module(m_s._c)
        self.assertFalse(m_f.hasattr("a"))
        out = m_f.forward(inp)
        self.assertEqual(out, expected)

    def test_freeze_module_with_tensor(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor([1, 2, 3, 4, 5, 6])

            def forward(self, x):
                x = self.a.view(2, 3)
                x[0][0] += 10
                return self.a.sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        inp = torch.tensor([5])
        expected = m_s.forward(inp)
        m_f = torch._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr("a"))
        m_f.a[0] -= 10
        out = m_f.forward(inp)
        self.assertEqual(out, expected)

    def test_freeze_module_with_list(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = [torch.tensor([1, 2, 3, 4, 5, 6])]

            def forward(self, x):
                self.a[0][1] += 10
                return self.a[0].sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        inp = torch.tensor([5])
        expected = m_s.forward(inp)
        m_s.a[0][1] -= 10
        m_f = torch._C._freeze_module(m_s._c)
        self.assertFalse(m_f.hasattr("a"))
        out = m_f.forward(inp)
        self.assertEqual(out, expected)

    def test_freeze_module_with_aliased_tensor_attr(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor([1, 2, 3, 4, 5, 6])
                self.b = self.a.view(2, 3)

            def forward(self, x):
                self.b[1] += 10
                return self.a.sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr("a"))
        inp = torch.tensor([5])
        out = m_f.forward(inp)
        expected = torch.tensor(51)  # 1+2+3+14+15+16
        self.assertEqual(out, expected)

    def test_freeze_module_with_aliased_tensor_attr2(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor([1, 2, 3, 4, 5, 6])
                self.b = {"layer": ([self.a.view(2, 3), torch.tensor([10])], 20)}
                self.c = ([self.a.view(2, 3), torch.tensor([10])], 20)
                self.d = (self.a.view(2, 3), 20)

            def forward(self, x):
                self.d[0][0] += 10
                return self.a.sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        inp = torch.tensor([5])
        expected = m_s.forward(inp)
        with self.assertRaisesRegex(
            RuntimeError, "module contains attributes values that overlaps"
        ):
            m_f = torch._C._freeze_module(m_s._c)

    def test_freeze_module_with_aliased_tensor_attr3(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor([1, 2, 3, 4, 5, 6])
                self.b = [self.a, torch.tensor([10])]

            def forward(self, x):
                self.a[1] += 10
                return self.b[0].sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        inp = torch.tensor([5])
        expected = m_s.forward(inp)
        m_f = torch._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr("a"))
        self.assertTrue(m_f.hasattr("b"))
        out = m_f.forward(inp)
        expected += 10  # account for  self.a += 10.
        self.assertEqual(out, expected)

    def test_freeze_module_with_aliased_tensor_attr4(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor([1, 2, 3, 4, 5, 6])
                self.b = [self.a, torch.tensor([10])]

            def forward(self, x):
                self.b[0][0] += 10
                return self.a.sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        inp = torch.tensor([5])
        expected = m_s.forward(inp)
        m_s.a[0] -= 10
        with self.assertRaisesRegex(
            RuntimeError, "module contains attributes values that overlaps"
        ):
            m_f = torch._C._freeze_module(m_s._c)

    def test_freeze_module_with_overlapping_attrs(self):
        a = torch.tensor([1, 2, 3, 4, 5, 6])

        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.b = [a.view(3, 2), torch.tensor([10])]
                self.c = (20, a.view(2, 3))

            def forward(self, x):
                self.b[0][0] += 10
                return self.c[1].sum()

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        inp = torch.tensor([5])
        expected = m_s.forward(inp)
        a[0] -= 10
        with self.assertRaisesRegex(
            RuntimeError, "module contains attributes values that overlaps"
        ):
            m_f = torch._C._freeze_module(m_s._c)

    def test_freeze_module_with_aliased_attr(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = [1, 2, 3, 4, 5, 6]
                self.b = self.a
                self.c = (self.a, 10)

            def forward(self, x):
                self.b[1] += 10
                return str(self.a) + str(self.c)

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        # FIXME: It should be assertTrue. Currently scripting is making a copy for setting self.b (see #33034)
        self.assertFalse(m_f.hasattr("a"))
        self.assertFalse(m_f.hasattr("c"))
        inp = torch.tensor([5])
        out = m_f.forward(inp)
        expected = m_s.forward(inp)
        self.assertEqual(out, expected)

    # Check attribute a is preserved. Alias analysis detects that 'a' has output writers.
    # In this example, 'a' is not mutated. However, we do not track which sub
    # values of a composite ivalue is mutated.
    def test_freeze_module_with_aliased_attr2(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = [1, 2, 3, 4, 5, 6]
                self.b = ([11], [10])

            def forward(self, x):
                v = self.a
                self.b = (v, [12])
                v2 = self.b[1]
                v2.append(7)
                return str(v) + str(v2)

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr("a"))
        inp = torch.tensor([5])
        out = m_f.forward(inp)
        expected = m.forward(inp)
        self.assertEqual(out, expected)

    def test_freeze_module_with_aliased_attr3(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = [1, 2, 3, 4, 5, 6]
                self.b = ([11], [10])

            def forward(self, x):
                v = self.a
                v2 = (v, [12])
                v3 = v2[0]
                v3.append(7)
                return str(self.a)

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr("a"))
        inp = torch.tensor([5])
        out = m_f.forward(inp)
        expected = m.forward(inp)
        self.assertEqual(out, expected)

    def test_freeze_module_return_self(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor([1.0, 2.0, 3.0])

            def forward(self, x):
                return self

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        with self.assertRaisesRegex(
            RuntimeError, "attempted to freeze a module that return itself"
        ):
            m_f = torch._C._freeze_module(m_s._c)

    def test_freeze_module_inlining(self):
        @torch.jit.script  # noqa: B903
        class Obj:  # noqa: B903
            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y

        class Mod(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.obj = Obj(2, 3)

            def forward(self, i: int):
                print(self.obj)
                return i

        mod = torch.jit.freeze(torch.jit.script(Mod().eval()))
        obj = mod.graph.findNode("prim::Constant")
        self.assertTrue(torch._C._jit_object_is_non_holding(obj))

        buffer = io.BytesIO()
        torch.jit.save(mod, buffer)
        buffer.seek(0)

        loaded = torch.jit.load(buffer)
        obj = mod.graph.findNode("prim::Constant")
        self.assertTrue(torch._C._jit_object_is_non_holding(obj))

    def test_freeze_module_return_sub_module(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)

            def forward(self, x):
                return self.conv1

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c)
        self.assertTrue(m_f.hasattr("conv1"))

    def test_freeze_module_no_forward(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(10, 1)

            @torch.jit.export
            def foo(self, x):
                return self.lin(x)

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        m_f = torch._C._freeze_module(m_s._c, preservedAttrs=["foo"])
        input = torch.ones(10)
        self.assertEqual(m_s.foo(input), m_f.foo(input))

    def test_freeze_no_forward(self):
        class FreezeMe(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(10, 1)

            @torch.jit.export
            def foo(self, x):
                return self.lin(x)

        m = FreezeMe()
        m_s = torch.jit.script(m)
        m_s.eval()
        m_f = torch.jit.freeze(m_s, preserved_attrs=["foo"])
        input = torch.ones(10)
        self.assertEqual(m_s.foo(input), m_f.foo(input))

    def test_freeze_module_in_training_mode(self):
        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                self.dropout1 = nn.Dropout2d(0.25)
                self.dropout2 = nn.Dropout2d(0.5)
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = nn.functional.relu(x)
                x = self.conv2(x)
                x = nn.functional.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = nn.functional.relu(x)
                x = self.dropout2(x)
                x = self.fc2(x)
                output = nn.functional.log_softmax(x, dim=1)
                return output

        model = torch.jit.script(Net())
        model.train()
        mTrain_freezed = torch._C._freeze_module(model._c)
        # verify mTrain_freezed looks exactly as:
        # module {
        #   attributes {
        #     conv1 = ...
        #     conv2 = ...
        #     dropout1 = ...
        #     dropout2 = ...
        #     fc1 = ...
        #     fc2 = ...
        #   }
        #   ...
        #   submodules {
        #     module conv1 {
        #       attributes {
        #          weight = ...
        #          bias = ...
        #       }
        #       ...
        #     }
        #     module conv2 {
        #       attributes {
        #          weight = ...
        #          bias = ...
        #       }
        #       ...
        #     }
        #     module dropout1 {
        #       attributes {
        #          training = ...
        #       }
        #       ...
        #     }
        #     module dropout2 {
        #       attributes {
        #          training = ...
        #       }
        #       ...
        #     }
        #     module fc1 {
        #       attributes {
        #          weight = ...
        #          bias = ...
        #       }
        #       ...
        #     }
        #     module fc2 {
        #       attributes {
        #          weight = ...
        #          bias = ...
        #       }
        #       ...
        #     }
        self.assertFalse(mTrain_freezed.hasattr("training"))
        self.assertTrue(mTrain_freezed.hasattr("conv1"))
        self.assertFalse(mTrain_freezed.conv1.hasattr("training"))
        self.assertTrue(mTrain_freezed.conv1.hasattr("weight"))
        self.assertTrue(mTrain_freezed.conv1.hasattr("bias"))
        self.assertTrue(mTrain_freezed.hasattr("conv2"))
        self.assertFalse(mTrain_freezed.conv2.hasattr("training"))
        self.assertTrue(mTrain_freezed.conv2.hasattr("weight"))
        self.assertTrue(mTrain_freezed.conv2.hasattr("bias"))
        self.assertTrue(mTrain_freezed.hasattr("dropout1"))
        self.assertTrue(mTrain_freezed.dropout1.hasattr("training"))
        self.assertTrue(mTrain_freezed.hasattr("dropout2"))
        self.assertTrue(mTrain_freezed.dropout2.hasattr("training"))
        self.assertTrue(mTrain_freezed.hasattr("fc1"))
        self.assertTrue(mTrain_freezed.fc1.hasattr("weight"))
        self.assertTrue(mTrain_freezed.fc1.hasattr("bias"))
        self.assertTrue(mTrain_freezed.hasattr("fc2"))
        self.assertTrue(mTrain_freezed.fc2.hasattr("weight"))
        self.assertTrue(mTrain_freezed.fc2.hasattr("bias"))
        model.eval()
        mEval_freezed = torch._C._freeze_module(model._c)
        self.assertFalse(mEval_freezed.hasattr("conv1"))
        self.assertFalse(mEval_freezed.hasattr("conv2"))
        self.assertFalse(mEval_freezed.hasattr("dropout1"))
        self.assertFalse(mEval_freezed.hasattr("training"))
        self.assertFalse(mEval_freezed.hasattr("fc1"))
        self.assertFalse(mEval_freezed.hasattr("dropout2"))
        self.assertFalse(mEval_freezed.hasattr("fc2"))
        with self.assertRaisesRegex(
            AttributeError, "does not have a field with name 'state_dict'"
        ):
            print(mEval_freezed.state_dict())
        buffer = io.BytesIO()
        torch.jit.save(mEval_freezed, buffer)
        buffer.seek(0)
        m = torch.jit.load(buffer)
        FileCheck().check_not("GetAttr[name=").run(m._c._get_method("forward").graph)
        m2 = torch._C._freeze_module(model._c, preserveParameters=True)
        self.assertTrue(m2.hasattr("conv1"))
        self.assertTrue(m2.hasattr("conv2"))
        self.assertFalse(m2.hasattr("dropout1"))
        self.assertFalse(m2.hasattr("training"))
        self.assertTrue(m2.hasattr("fc1"))
        self.assertFalse(m2.hasattr("dropout2"))
        self.assertTrue(m2.hasattr("fc2"))

    def test_freeze_module_detach_gradient(self):
        mod = nn.Conv2d(8, 3, 4, 2, 1)
        self.assertTrue(mod.weight.requires_grad)
        smod = torch.jit.script(mod)
        smod.eval()
        fmod = torch._C._freeze_module(smod._c)
        self.assertTrue(mod.weight.requires_grad)
        self.assertTrue(smod.weight.requires_grad)
        self.assertFalse(fmod.hasattr("weight"))
        inp = torch.ones(1, 8, 32, 32)
        out1 = fmod.forward(inp)
        # FIXME: frozen module mutated from outside (original module).
        with torch.no_grad():
            smod.weight[0, 0, 0, 0] += 100.0
        out2 = fmod.forward(inp)
        out3 = smod(inp)
        self.assertNotEqual(out1, out2)
        self.assertEqual(out2, out3)

    def test_freeze_module_with_user_preserved_attr(self):
        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor([1.1])
                self.b = torch.tensor([2.2])

            def forward(self, x):
                return self.a + self.b

        m = torch.jit.script(Module())
        m.eval()
        fm = torch._C._freeze_module(m._c, ["a"])
        # Attribute "a" is preserved
        self.assertTrue(fm.hasattr("a"))
        self.assertFalse(fm.hasattr("b"))

    def test_freeze_module_with_user_preserved_method(self):
        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor([1.1])
                self.b = torch.tensor([2.2])

            def forward(self, x):
                return self.a + self.b

            @torch.jit.export
            def modify_a(self, x):
                self.a[0] += 10
                return self.b

            @torch.jit.export
            def modify_b(self, x):
                self.b[0] += 20
                return self.a

        m = torch.jit.script(Module())
        m.eval()
        fm = torch._C._freeze_module(m._c, ["modify_a"])
        # Both attribute "a" and method "modify_a" are preserved
        self.assertTrue(fm.hasattr("a"))
        self.assertFalse(fm.hasattr("b"))
        input = torch.randn(2, 2)
        expected = m.forward(input)
        out = fm.forward(input)
        self.assertEqual(out, expected)

    def test_freeze_module_with_user_preserved_method2(self):
        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor([1.1])
                self.b = torch.tensor([2.2])

            def forward(self, x):
                self.b += 10
                return self.a + self.b

            @torch.jit.export
            def modify_a(self, x):
                self.a[0] += 10
                return self.b + self.a

        m = torch.jit.script(Module())
        m.eval()
        fm = torch._C._freeze_module(m._c, ["modify_a"])
        FileCheck().check('prim::GetAttr[name="a"]').run(fm.forward.graph)
        FileCheck().check('prim::GetAttr[name="b"]').run(fm.modify_a.graph)

    def test_freeze_module_with_user_preserved_attribute_on_submodule(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 1
                self.b = 2

            def forward(self):
                return self.a + self.b

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub1 = SubModule()
                self.sub2 = SubModule()

            def forward(self):
                return self.sub1() + self.sub2()

        m = torch.jit.script(Module())
        m.eval()
        m = torch.jit.freeze(m, preserved_attrs=["sub1.a", "sub2.a"])
        fm = m._c

        self.assertTrue(fm.hasattr("sub1"))
        self.assertTrue(fm.sub1.hasattr("a"))
        self.assertFalse(fm.sub1.hasattr("b"))
        self.assertTrue(fm.hasattr("sub2"))
        self.assertTrue(fm.sub2.hasattr("a"))
        self.assertFalse(fm.sub2.hasattr("b"))
        self.assertEqual(m(), 6)
        m.sub1.a += 1
        self.assertEqual(m(), 7)

    def test_freeze_module_with_user_preserved_attribute_on_unused_submodule(self):
        class SubModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = 1
                self.b = 2

            def forward(self):
                return self.a + self.b

            @torch.jit.export
            def method_a(self):
                return 42

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = SubModule()

            def forward(self):
                return 1

        m = torch.jit.script(Module())
        m.eval()
        fm = torch.jit.freeze(m, preserved_attrs=["sub.a", "sub.method_a"])._c

        self.assertTrue(fm.hasattr("sub"))
        self.assertTrue(fm.sub.hasattr("a"))
        self.assertFalse(fm.sub.hasattr("b"))
        self.assertTrue(fm.sub._has_method("method_a"))

    def test_freeze_module_with_user_preserved_method_on_submodule(self):
        class SubModule(nn.Module):
            def forward(self, x):
                return self.method_a(x) + self.method_b(x)

            def method_a(self, x):
                return x * x

            def method_b(self, x):
                return x + x

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sub = SubModule()

            def forward(self, x):
                return self.sub(x)

        m = torch.jit.script(Module())
        m.eval()
        fm = torch.jit.freeze(m, preserved_attrs=["sub.method_a"])._c

        self.assertTrue(fm.hasattr("sub"))
        self.assertTrue(fm.sub._has_method("method_a"))
        self.assertFalse(fm.sub._has_method("method_b"))

    @skipIfNoFBGEMM
    def test_module_with_shared_type_instances(self):
        class Child(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 1, 1).to(dtype=torch.float32)

            def forward(self, x):
                x = self.conv1(x)
                return x

        class Parent(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.conv1 = nn.Conv2d(1, 1, 1).to(dtype=torch.float32)
                self.child = Child()
                self.child2 = Child()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.conv1(x)
                x = self.child(x)
                x = self.child2(x)
                x = self.dequant(x)
                return x

        def _static_quant(model):
            qModel = torch.ao.quantization.QuantWrapper(model)
            qModel.qconfig = torch.ao.quantization.default_qconfig
            torch.ao.quantization.prepare(qModel, inplace=True)
            qModel(torch.rand(4, 1, 4, 4, dtype=torch.float32))
            torch.ao.quantization.convert(qModel, inplace=True)
            return model

        with override_quantized_engine("fbgemm"):
            data = torch.randn(4, 1, 4, 4, dtype=torch.float32)
            m = Parent().to(torch.float32)
            m = _static_quant(m)
            m = torch.jit.script(m)
            m.eval()
            torch._C._jit_pass_inline(m.graph)
            m_frozen = wrap_cpp_module(torch._C._freeze_module(m._c))
            # Earlier bug resulted in _packed_params set to false.
            FileCheck().check_not("_packed_params = False").run(
                m_frozen._c.dump_to_str(True, True, False)
            )

            m_res = m(data)
            # It used to segfault while running frozen module.
            m_frozen_res = m_frozen(data)
            self.assertEqual(m_res, m_frozen_res)

    def test_module_getattr_indirection(self):
        @torch.jit.script
        class ValHolder:
            def __init__(self, val: int):
                self.val: int = val

        class Mod(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mod1 = ValHolder(1)
                self.mod2 = ValHolder(2)

            def forward(self, cond: bool):
                if cond:
                    mod = self.mod1
                else:
                    mod = self.mod2
                return mod.val

        mod = Mod()
        mod.eval()
        frozen_mod = torch.jit.freeze(torch.jit.script(mod))
        mod_eager = Mod()
        self.assertEqual(mod_eager(True), frozen_mod(True))
        self.assertEqual(mod_eager(False), frozen_mod(False))

    def test_freeze_module_with_non_static_module_container_index(self):

```



## High-Level Overview


This Python file contains 137 class(es) and 373 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestFreezing`, `M`, `SubModule`, `SubModule2`, `TestModule`, `SubModule`, `TestModule`, `SubModule`, `SubModule2`, `TestModule`, `TestModule`, `TestModule`, `SubModule`, `SubModule2`, `TestModule`, `SubModule`, `SubModule2`, `TestModule`, `SubModule`, `SubModule2`

**Functions defined**: `removeExceptions`, `test_freeze_module`, `__init__`, `forward`, `test_freeze_module_with_submodule`, `__init__`, `forward`, `__init__`, `forward`, `__init__`, `forward`, `test_freeze_module_with_fork`, `__init__`, `forward`, `__init__`, `forward`, `test_freeze_module_with_nested_fork`, `__init__`, `forward`, `__init__`

**Key imports**: io, unittest, product, Any, torch, torch.nn as nn, torch.nn.functional as F, wrap_cpp_module, FileCheck, TEST_CUDA, TEST_CUDNN


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `io`
- `unittest`
- `itertools`: product
- `typing`: Any
- `torch`
- `torch.nn as nn`
- `torch.nn.functional as F`
- `torch.jit._recursive`: wrap_cpp_module
- `torch.testing`: FileCheck
- `torch.testing._internal.common_cuda`: TEST_CUDA, TEST_CUDNN
- `torch.testing._internal.common_quantization`: skipIfNoFBGEMM
- `torch.testing._internal.common_quantized`: override_quantized_engine
- `torch.testing._internal.jit_utils`: JitTestCase
- `torch.utils`: mkldnn as mkldnn_utils
- `torchvision`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Error Handling**: Includes exception handling
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/jit/test_freezing.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/jit`):

- [`test_dataclasses.py_docs.md`](./test_dataclasses.py_docs.md)
- [`test_recursive_script.py_docs.md`](./test_recursive_script.py_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_python_builtins.py_docs.md`](./test_python_builtins.py_docs.md)
- [`test_functional_blocks.py_docs.md`](./test_functional_blocks.py_docs.md)
- [`test_hooks_modules.py_docs.md`](./test_hooks_modules.py_docs.md)
- [`mydecorator.py_docs.md`](./mydecorator.py_docs.md)
- [`test_union.py_docs.md`](./test_union.py_docs.md)
- [`test_python_bindings.py_docs.md`](./test_python_bindings.py_docs.md)
- [`test_parametrization.py_docs.md`](./test_parametrization.py_docs.md)


## Cross-References

- **File Documentation**: `test_freezing.py_docs.md`
- **Keyword Index**: `test_freezing.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
