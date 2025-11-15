# Documentation: `test/distributed/_tools/test_mod_tracker.py`

## File Metadata

- **Path**: `test/distributed/_tools/test_mod_tracker.py`
- **Size**: 7,020 bytes (6.86 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

from copy import copy

import torch
from torch.distributed._tools.mod_tracker import ModTracker
from torch.testing._internal.common_utils import run_tests, TestCase, xfailIfTorchDynamo
from torch.utils.checkpoint import checkpoint


class TestModTracker(TestCase):
    # "https://github.com/pytorch/pytorch/issues/127112
    @xfailIfTorchDynamo
    def test_module_hierarchy(self):
        seen_fw = []
        seen_bw = []

        class Foo(torch.nn.Module):
            def forward(self, x):
                x = x["a"].relu_()
                seen_fw.append((copy(tracker.parents), tracker.is_bw))
                x.register_hook(
                    lambda grad: seen_bw.append((copy(tracker.parents), tracker.is_bw))
                )
                return {"a": torch.mm(x, x)}

        class Mod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = Foo()
                self.b = torch.nn.ModuleDict({"nest": Foo()})
                self.c = torch.nn.ModuleList([Foo()])

            def forward(self, x):
                x = self.c[0](x)
                return self.b["nest"](self.a(x))

        mod = Mod()

        with ModTracker() as tracker:
            mod({"a": torch.randn(10, 10, requires_grad=True).clone()})[
                "a"
            ].sum().backward()
            mod({"a": torch.randn(10, 10, requires_grad=True).clone()})[
                "a"
            ].sum().backward()

        self.assertEqual(
            seen_fw,
            [
                ({"Global", "Mod", "Mod.c.0"}, False),
                ({"Global", "Mod", "Mod.a"}, False),
                ({"Global", "Mod", "Mod.b.nest"}, False),
                ({"Global", "Mod", "Mod.c.0"}, False),
                ({"Global", "Mod", "Mod.a"}, False),
                ({"Global", "Mod", "Mod.b.nest"}, False),
            ],
        )

        self.assertEqual(
            seen_bw,
            [
                ({"Global", "Mod", "Mod.b.nest"}, True),
                ({"Global", "Mod", "Mod.a"}, True),
                ({"Global", "Mod", "Mod.c.0"}, True),
                ({"Global", "Mod", "Mod.b.nest"}, True),
                ({"Global", "Mod", "Mod.a"}, True),
                ({"Global", "Mod", "Mod.c.0"}, True),
            ],
        )

    def test_bw_detection(self):
        mod = torch.nn.Linear(2, 2)

        with ModTracker() as tracker:
            mod(torch.rand(2, requires_grad=True)).sum().backward()
            self.assertFalse(tracker.is_bw)
            self.assertEqual(tracker.parents, {"Global"})

    @xfailIfTorchDynamo
    def test_user_hooks(self):
        class Bar(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.foo(x).relu_()

        mt = ModTracker()
        test_op = []

        def hook(mod, hook_name):
            mfqn = mt.get_known_fqn(mod) if mod is not None else None
            test_op.append((hook_name, mfqn, mfqn in mt.parents, mt.is_bw))

        mod = Bar()

        mt.register_user_hooks(
            lambda m, inp: hook(m, "pre_fw"),
            lambda m, inp, op: hook(m, "post_fw"),
            lambda m, gop: hook(m, "pre_bw"),
            lambda m, ginp: hook(m, "post_bw"),
        )
        with mt:
            mod(torch.rand(10, 10, requires_grad=True)).sum().backward()
        expected_op = [
            ("pre_fw", "Bar", True, False),
            ("pre_fw", "Bar.foo", True, False),
            ("post_fw", "Bar.foo", True, False),
            ("post_fw", "Bar", True, False),
            ("pre_bw", "Bar", True, True),
            ("pre_bw", "Bar.foo", True, True),
            ("post_bw", "Bar", True, True),
            ("post_bw", "Bar.foo", True, True),
        ]
        self.assertEqual(test_op, expected_op)

        with self.assertRaises(AssertionError):
            mt.register_user_hooks(lambda x, y: x, None, None, None)

        test_op.clear()
        with mt:
            loss = mod(torch.rand(10, 10, requires_grad=True)).sum()
            del mod
            loss.backward()
        expected_op = [
            ("pre_fw", "Bar", True, False),
            ("pre_fw", "Bar.foo", True, False),
            ("post_fw", "Bar.foo", True, False),
            ("post_fw", "Bar", True, False),
            ("pre_bw", None, False, True),
            ("pre_bw", None, False, True),
            ("post_bw", None, False, True),
            ("post_bw", None, False, True),
        ]
        self.assertEqual(test_op, expected_op)

    @xfailIfTorchDynamo
    def test_ac(self):
        class Foo(torch.nn.Module):
            def __init__(self, n_layers: int, dim: int, use_ac: bool = False):
                super().__init__()
                self.linears = torch.nn.ModuleList()
                self.use_ac = use_ac
                for _ in range(n_layers):
                    self.linears.append(torch.nn.Linear(dim, dim))

            def forward(self, x):
                for i, block in enumerate(self.linears):
                    if i >= 1 and self.use_ac:
                        x = checkpoint(
                            block, x, preserve_rng_state=True, use_reentrant=False
                        )
                    else:
                        x = block(x)
                    assert x is not None
                    x = torch.nn.functional.relu(x)
                return x

        bsz = 2
        dim = 8
        n_layers = 2
        test_op = []

        def hook(mod, mt, hook_name):
            mfqn = mt.get_known_fqn(mod) if mod is not None else None
            test_op.append((hook_name, mfqn, mfqn in mt.parents, mt.is_bw))

        mt = ModTracker()
        mt.register_user_hooks(
            lambda m, i: hook(m, mt, "pre_fw"),
            lambda m, i, o: hook(m, mt, "post_fw"),
            lambda m, go: hook(m, mt, "pre_bw"),
            lambda m, gi: hook(m, mt, "post_bw"),
        )
        model = Foo(n_layers, dim, True)
        x = torch.randn(bsz, dim)
        with mt:
            model(x).sum().backward()

        expected_op = [
            ("pre_fw", "Foo", True, False),
            ("pre_fw", "Foo.linears.0", True, False),
            ("post_fw", "Foo.linears.0", True, False),
            ("pre_fw", "Foo.linears.1", True, False),
            ("post_fw", "Foo.linears.1", True, False),
            ("post_fw", "Foo", True, False),
            ("pre_bw", "Foo", True, True),
            ("pre_bw", "Foo.linears.1", True, True),
            ("pre_fw", "Foo.linears.1", True, True),
            ("post_fw", "Foo.linears.1", True, True),
            ("post_bw", "Foo.linears.1", True, True),
            ("pre_bw", "Foo.linears.0", True, True),
            ("post_bw", "Foo.linears.0", True, True),
            ("post_bw", "Foo", True, True),
        ]
        self.assertEqual(test_op, expected_op)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 5 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestModTracker`, `Foo`, `Mod`, `Bar`, `Foo`

**Functions defined**: `test_module_hierarchy`, `forward`, `__init__`, `forward`, `test_bw_detection`, `test_user_hooks`, `__init__`, `forward`, `hook`, `test_ac`, `__init__`, `forward`, `hook`

**Key imports**: copy, torch, ModTracker, run_tests, TestCase, xfailIfTorchDynamo, checkpoint


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/_tools`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`: copy
- `torch`
- `torch.distributed._tools.mod_tracker`: ModTracker
- `torch.testing._internal.common_utils`: run_tests, TestCase, xfailIfTorchDynamo
- `torch.utils.checkpoint`: checkpoint


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
python test/distributed/_tools/test_mod_tracker.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/_tools`):

- [`test_fsdp2_mem_tracker.py_docs.md`](./test_fsdp2_mem_tracker.py_docs.md)
- [`test_runtime_estimator.py_docs.md`](./test_runtime_estimator.py_docs.md)
- [`test_memory_tracker.py_docs.md`](./test_memory_tracker.py_docs.md)
- [`test_sac_estimator.py_docs.md`](./test_sac_estimator.py_docs.md)
- [`test_sac_ilp.py_docs.md`](./test_sac_ilp.py_docs.md)
- [`test_mem_tracker.py_docs.md`](./test_mem_tracker.py_docs.md)
- [`test_fake_collectives.py_docs.md`](./test_fake_collectives.py_docs.md)


## Cross-References

- **File Documentation**: `test_mod_tracker.py_docs.md`
- **Keyword Index**: `test_mod_tracker.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
