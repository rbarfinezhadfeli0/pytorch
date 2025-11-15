# Documentation: `docs/test/test_opaque_obj.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_opaque_obj.py_docs.md`
- **Size**: 12,801 bytes (12.50 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_opaque_obj.py`

## File Metadata

- **Path**: `test/test_opaque_obj.py`
- **Size**: 9,699 bytes (9.47 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: custom-operators"]
import copy

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._library.fake_class_registry import maybe_to_fake_obj
from torch._library.opaque_object import (
    get_payload,
    make_opaque,
    OpaqueType,
    set_payload,
)
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


class OpaqueQueue:
    def __init__(self, queue: list[torch.Tensor], init_tensor_: torch.Tensor) -> None:
        super().__init__()
        self.queue = queue
        self.init_tensor_ = init_tensor_

        # For testing purposes
        self._push_counter = 0
        self._pop_counter = 0
        self._size_counter = 0

    def push(self, tensor: torch.Tensor) -> None:
        self._push_counter += 1
        self.queue.append(tensor)

    def pop(self) -> torch.Tensor:
        self._pop_counter += 1
        if len(self.queue) > 0:
            return self.queue.pop(0)
        return self.init_tensor_

    def size(self) -> int:
        self._size_counter += 1
        return len(self.queue)

    def __eq__(self, other):
        if len(self.queue) != len(other.queue):
            return False
        for q1, q2 in zip(self.queue, other.queue):
            if not torch.allclose(q1, q2):
                return False
        return torch.allclose(self.init_tensor_, other.init_tensor_)


class TestOpaqueObject(TestCase):
    def setUp(self):
        self.lib = torch.library.Library("_TestOpaqueObject", "FRAGMENT")  # noqa: TOR901

        torch.library.define(
            "_TestOpaqueObject::queue_push",
            "(__torch__.torch.classes.aten.OpaqueObject a, Tensor b) -> ()",
            tags=torch.Tag.pt2_compliant_tag,
            lib=self.lib,
        )

        @torch.library.impl(
            "_TestOpaqueObject::queue_push", "CompositeExplicitAutograd", lib=self.lib
        )
        def push_impl(q: torch._C.ScriptObject, b: torch.Tensor) -> None:
            queue = get_payload(q)
            assert isinstance(queue, OpaqueQueue)
            queue.push(b)

        @torch.library.register_fake("_TestOpaqueObject::queue_push", lib=self.lib)
        def push_impl_fake(q: torch._C.ScriptObject, b: torch.Tensor) -> None:
            pass

        self.lib.define(
            "queue_pop(__torch__.torch.classes.aten.OpaqueObject a) -> Tensor",
        )

        def pop_impl(q: torch._C.ScriptObject) -> torch.Tensor:
            queue = get_payload(q)
            assert isinstance(queue, OpaqueQueue)
            return queue.pop()

        self.lib.impl("queue_pop", pop_impl, "CompositeExplicitAutograd")

        def pop_impl_fake(q: torch._C.ScriptObject) -> torch.Tensor:
            # This is not accurate since the queue could have tensors that are
            # not rank 1
            ctx = torch._custom_op.impl.get_ctx()
            u0 = ctx.new_dynamic_size()
            return torch.empty(u0)

        self.lib._register_fake("queue_pop", pop_impl_fake)

        @torch.library.custom_op(
            "_TestOpaqueObject::queue_size",
            mutates_args=[],
        )
        def size_impl(q: OpaqueType) -> int:
            queue = get_payload(q)
            assert isinstance(queue, OpaqueQueue)
            return queue.size()

        @size_impl.register_fake
        def size_impl_fake(q: torch._C.ScriptObject) -> int:
            ctx = torch._custom_op.impl.get_ctx()
            u0 = ctx.new_dynamic_size()
            return u0

        super().setUp()

    def tearDown(self):
        self.lib._destroy()

        super().tearDown()

    def test_creation(self):
        queue = OpaqueQueue([], torch.zeros(3))
        obj = make_opaque(queue)
        self.assertTrue(isinstance(obj, torch._C.ScriptObject))
        self.assertEqual(str(obj._type()), "__torch__.torch.classes.aten.OpaqueObject")

        # obj.payload stores a direct reference to this python queue object
        payload = get_payload(obj)
        self.assertEqual(payload, queue)
        queue.push(torch.ones(3))
        self.assertEqual(payload.size(), 1)

    def test_ops(self):
        queue = OpaqueQueue([], torch.zeros(3))
        obj = make_opaque()
        set_payload(obj, queue)

        torch.ops._TestOpaqueObject.queue_push(obj, torch.ones(3) + 1)
        self.assertEqual(queue.size(), 1)
        size = torch.ops._TestOpaqueObject.queue_size(obj)
        self.assertEqual(size, queue.size())
        popped = torch.ops._TestOpaqueObject.queue_pop(obj)
        self.assertEqual(popped, torch.ones(3) + 1)
        self.assertEqual(queue.size(), 0)

    def test_eq(self):
        self.assertTrue(make_opaque("moo") == make_opaque("moo"))
        self.assertFalse(make_opaque("moo") == make_opaque("mop"))

        q1 = OpaqueQueue([torch.ones(3)], torch.zeros(3))
        q2 = OpaqueQueue([torch.ones(3)], torch.zeros(3))
        obj1 = make_opaque(q1)
        obj2 = make_opaque(q2)
        self.assertTrue(obj1 == obj1)
        self.assertTrue(q1 == q2)
        self.assertTrue(obj1 == obj2)

    def test_deepcopy(self):
        q1 = OpaqueQueue([torch.ones(3), torch.ones(3) * 2], torch.zeros(3))
        obj1 = make_opaque(q1)

        obj2 = copy.deepcopy(obj1)
        q2 = get_payload(obj2)

        self.assertTrue(q1 is not q2)
        self.assertTrue(q1 == q2)

    def test_bad_fake(self):
        torch.library.define(
            "_TestOpaqueObject::bad_fake",
            "(__torch__.torch.classes.aten.OpaqueObject q, Tensor x) -> Tensor",
            lib=self.lib,
        )

        def f(q, x):
            torch.ops._TestOpaqueObject.bad_fake(q, x)
            return x.cos()

        def bad_fake1(q: torch._C.ScriptObject, b: torch.Tensor) -> torch.Tensor:
            payload = get_payload(q)
            return b * payload

        torch.library.register_fake(
            "_TestOpaqueObject::bad_fake", bad_fake1, lib=self.lib
        )

        with FakeTensorMode() as fake_mode:
            obj = make_opaque(1)
            fake_obj = maybe_to_fake_obj(fake_mode, obj)
            x = torch.ones(3)

            with self.assertRaisesRegex(
                ValueError,
                "get_payload: this function was called with a FakeScriptObject",
            ):
                torch.ops._TestOpaqueObject.bad_fake(fake_obj, x)

        def bad_fake2(q: torch._C.ScriptObject, b: torch.Tensor) -> torch.Tensor:
            set_payload(q, 2)
            return torch.empty_like(b)

        torch.library.register_fake(
            "_TestOpaqueObject::bad_fake", bad_fake2, lib=self.lib, allow_override=True
        )

        with FakeTensorMode() as fake_mode:
            obj = make_opaque(1)
            fake_obj = maybe_to_fake_obj(fake_mode, obj)
            x = torch.ones(3)

            with self.assertRaisesRegex(
                ValueError,
                "set_payload: this function was called with a FakeScriptObject",
            ):
                torch.ops._TestOpaqueObject.bad_fake(fake_obj, x)

    @parametrize("make_fx_tracing_mode", ["fake", "symbolic"])
    def test_make_fx(self, make_fx_tracing_mode):
        class M(torch.nn.Module):
            def forward(self, queue, x):
                torch.ops._TestOpaqueObject.queue_push(queue, x.tan())
                torch.ops._TestOpaqueObject.queue_push(queue, x.cos())
                torch.ops._TestOpaqueObject.queue_push(queue, x.sin())
                pop1 = torch.ops._TestOpaqueObject.queue_pop(queue)
                size1 = torch.ops._TestOpaqueObject.queue_size(queue)
                pop2 = torch.ops._TestOpaqueObject.queue_pop(queue)
                size2 = torch.ops._TestOpaqueObject.queue_size(queue)
                x_cos = pop1 + size1
                x_sin = pop2 - size2
                return x_sin + x_cos

        q1 = OpaqueQueue([], torch.empty(0).fill_(-1))
        obj1 = make_opaque(q1)
        q2 = OpaqueQueue([], torch.empty(0).fill_(-1))
        obj2 = make_opaque(q2)

        x = torch.ones(2, 3)
        gm = make_fx(M(), tracing_mode=make_fx_tracing_mode)(obj1, x)
        self.assertTrue(torch.allclose(gm(obj1, x), M()(obj2, x)))
        self.assertEqual(q1._push_counter, 3)
        self.assertEqual(q1._pop_counter, 2)
        self.assertEqual(q1._size_counter, 2)
        self.assertEqual(q1.size(), 1)
        self.assertExpectedInline(
            gm.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1):
    tan = torch.ops.aten.tan.default(arg1_1)
    queue_push = torch.ops._TestOpaqueObject.queue_push.default(arg0_1, tan);  tan = queue_push = None
    cos = torch.ops.aten.cos.default(arg1_1)
    queue_push_1 = torch.ops._TestOpaqueObject.queue_push.default(arg0_1, cos);  cos = queue_push_1 = None
    sin = torch.ops.aten.sin.default(arg1_1);  arg1_1 = None
    queue_push_2 = torch.ops._TestOpaqueObject.queue_push.default(arg0_1, sin);  sin = queue_push_2 = None
    queue_pop = torch.ops._TestOpaqueObject.queue_pop.default(arg0_1)
    queue_size = torch.ops._TestOpaqueObject.queue_size.default(arg0_1)
    queue_pop_1 = torch.ops._TestOpaqueObject.queue_pop.default(arg0_1)
    queue_size_1 = torch.ops._TestOpaqueObject.queue_size.default(arg0_1);  arg0_1 = None
    add = torch.ops.aten.add.Tensor(queue_pop, queue_size);  queue_pop = queue_size = None
    sub = torch.ops.aten.sub.Tensor(queue_pop_1, queue_size_1);  queue_pop_1 = queue_size_1 = None
    add_1 = torch.ops.aten.add.Tensor(sub, add);  sub = add = None
    return add_1
    """,
        )


instantiate_parametrized_tests(TestOpaqueObject)

if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 3 class(es) and 24 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `OpaqueQueue`, `TestOpaqueObject`, `M`

**Functions defined**: `__init__`, `push`, `pop`, `size`, `__eq__`, `setUp`, `push_impl`, `push_impl_fake`, `pop_impl`, `pop_impl_fake`, `size_impl`, `size_impl_fake`, `tearDown`, `test_creation`, `test_ops`, `test_eq`, `test_deepcopy`, `test_bad_fake`, `f`, `bad_fake1`

**Key imports**: copy, torch, run_tests, TestCase, maybe_to_fake_obj, FakeTensorMode, make_fx


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `torch`
- `torch._dynamo.test_case`: run_tests, TestCase
- `torch._library.fake_class_registry`: maybe_to_fake_obj
- `torch._subclasses.fake_tensor`: FakeTensorMode
- `torch.fx.experimental.proxy_tensor`: make_fx


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/test_opaque_obj.py
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

- **File Documentation**: `test_opaque_obj.py_docs.md`
- **Keyword Index**: `test_opaque_obj.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

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
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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
python docs/test/test_opaque_obj.py_docs.md
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

- **File Documentation**: `test_opaque_obj.py_docs.md_docs.md`
- **Keyword Index**: `test_opaque_obj.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
