# Documentation: `test/fx/test_fx_node_hook.py`

## File Metadata

- **Path**: `test/fx/test_fx_node_hook.py`
- **Size**: 3,373 bytes (3.29 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: fx"]
import torch
from torch.fx import symbolic_trace
from torch.testing._internal.common_utils import TestCase


class TestFXNodeHook(TestCase):
    def test_hooks_for_node_update(self):
        global create_node_hook1_called
        global create_node_hook2_called
        global erase_node_hook1_called
        global erase_node_hook2_called
        global replace_node_hook1_called
        global replace_node_hook2_called
        create_node_hook1_called = False
        create_node_hook2_called = False
        erase_node_hook1_called = False
        erase_node_hook2_called = False
        replace_node_hook1_called = False
        replace_node_hook2_called = False

        def fn(a, b, c):
            x = torch.nn.functional.linear(a, b)
            x = x + c
            return x.cos()

        def create_node_hook1(node):
            global create_node_hook1_called
            create_node_hook1_called = True

        def create_node_hook2(node):
            global create_node_hook2_called
            create_node_hook2_called = True

        def erase_node_hook1(node):
            global erase_node_hook1_called
            erase_node_hook1_called = True

        def erase_node_hook2(node):
            global erase_node_hook2_called
            erase_node_hook2_called = True

        def replace_node_hook1(old, new, user):
            global replace_node_hook1_called
            self.assertEqual(old.name, "a")
            self.assertEqual(new, "a_1")
            self.assertEqual(user.name, "linear")
            replace_node_hook1_called = True

        def replace_node_hook2(old, new, user):
            global replace_node_hook2_called
            replace_node_hook2_called = True

        gm = symbolic_trace(fn)
        gm._register_create_node_hook(create_node_hook1)
        gm._register_create_node_hook(create_node_hook2)
        gm._register_erase_node_hook(erase_node_hook1)
        gm._register_erase_node_hook(erase_node_hook2)
        gm._register_replace_node_hook(replace_node_hook1)
        gm._register_replace_node_hook(replace_node_hook2)

        graph = gm.graph
        node_a = None
        for node in graph.find_nodes(op="placeholder"):
            node_a = node
            break
        assert node_a is not None
        # This will create a new node
        node_a_copy = graph.node_copy(node_a)
        node_a.replace_all_uses_with(node_a_copy)
        graph.erase_node(node_a)

        assert (
            create_node_hook1_called
            and create_node_hook2_called
            and erase_node_hook1_called
            and erase_node_hook2_called
            and replace_node_hook1_called
            and replace_node_hook2_called
        )

        gm._unregister_create_node_hook(create_node_hook1)
        gm._unregister_create_node_hook(create_node_hook2)
        gm._unregister_erase_node_hook(erase_node_hook1)
        gm._unregister_erase_node_hook(erase_node_hook2)
        gm._unregister_replace_node_hook(replace_node_hook1)
        gm._unregister_replace_node_hook(replace_node_hook2)

        assert gm._create_node_hooks == []
        assert gm._erase_node_hooks == []
        assert gm._replace_hooks == []


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )

```



## High-Level Overview


This Python file contains 1 class(es) and 8 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestFXNodeHook`

**Functions defined**: `test_hooks_for_node_update`, `fn`, `create_node_hook1`, `create_node_hook2`, `erase_node_hook1`, `erase_node_hook2`, `replace_node_hook1`, `replace_node_hook2`

**Key imports**: torch, symbolic_trace, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/fx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.fx`: symbolic_trace
- `torch.testing._internal.common_utils`: TestCase


## Code Patterns & Idioms

### Common Patterns

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
python test/fx/test_fx_node_hook.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/fx`):

- [`test_dynamism.py_docs.md`](./test_dynamism.py_docs.md)
- [`test_dce_pass.py_docs.md`](./test_dce_pass.py_docs.md)
- [`test_source_matcher_utils.py_docs.md`](./test_source_matcher_utils.py_docs.md)
- [`test_graph_pickler.py_docs.md`](./test_graph_pickler.py_docs.md)
- [`named_tup.py_docs.md`](./named_tup.py_docs.md)
- [`test_cse_pass.py_docs.md`](./test_cse_pass.py_docs.md)
- [`test_fx_traceback.py_docs.md`](./test_fx_traceback.py_docs.md)
- [`test_gradual_type.py_docs.md`](./test_gradual_type.py_docs.md)
- [`test_pass_infra.py_docs.md`](./test_pass_infra.py_docs.md)
- [`test_lazy_graph_module.py_docs.md`](./test_lazy_graph_module.py_docs.md)


## Cross-References

- **File Documentation**: `test_fx_node_hook.py_docs.md`
- **Keyword Index**: `test_fx_node_hook.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
