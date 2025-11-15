# Documentation: `docs/test/jit/test_python_bindings.py_docs.md`

## File Metadata

- **Path**: `docs/test/jit/test_python_bindings.py_docs.md`
- **Size**: 6,602 bytes (6.45 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/jit/test_python_bindings.py`

## File Metadata

- **Path**: `test/jit/test_python_bindings.py`
- **Size**: 3,616 bytes (3.53 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: jit"]

import torch
from torch.testing import FileCheck
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestPythonBindings(JitTestCase):
    def test_cu_get_functions(self):
        @torch.jit.script
        def test_get_python_cu_fn(x: torch.Tensor):
            return 2 * x

        cu = torch.jit._state._python_cu
        self.assertTrue(
            "test_get_python_cu_fn" in (str(fn.name) for fn in cu.get_functions())
        )

    def test_cu_create_function(self):
        @torch.jit.script
        def fn(x: torch.Tensor):
            return 2 * x

        cu = torch._C.CompilationUnit()
        cu.create_function("test_fn", fn.graph)

        inp = torch.randn(5)

        self.assertEqual(inp * 2, cu.find_function("test_fn")(inp))
        self.assertEqual(cu.find_function("doesnt_exist"), None)
        self.assertEqual(inp * 2, cu.test_fn(inp))
        with self.assertRaises(AttributeError):
            cu.doesnt_exist(inp)

    def test_invalidation(self):
        @torch.jit.script
        def test_invalidation_fn(x: torch.Tensor):
            return 2 * x

        gr = test_invalidation_fn.graph.copy()
        n = gr.insertNode(gr.create("prim::profile"))
        v = n.output()
        # check that they work
        str((n, v))
        torch._C._jit_pass_dce(gr)
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            str(n)
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            str(v)

    def test_graph_iterator_keepalive(self):
        @torch.jit.script
        def test_iterator_keepalive_fn(x: torch.Tensor):
            return 2 * x

        # the list would segfault before because inlined_graph
        # is temporary and had been deleted (see issue #50454)
        n = test_iterator_keepalive_fn.inlined_graph.nodes()
        list(n)
        i = test_iterator_keepalive_fn.inlined_graph.inputs()
        list(i)
        o = test_iterator_keepalive_fn.inlined_graph.outputs()
        list(o)

    def test_aliasdb(self):
        @torch.jit.script
        def test_aliasdb_fn(x: torch.Tensor):
            return 2 * x

        gr = test_aliasdb_fn.graph.copy()
        alias_db = gr.alias_db()
        self.assertTrue("WILDCARD" in str(alias_db))
        self.assertTrue("digraph alias_db" in alias_db.to_graphviz_str())

    def test_graph_create(self):
        gr = torch._C.Graph()
        with self.assertRaises(ValueError):
            gr.create("prim::Constant", [None])

    def test_add_input(self):
        gr = torch._C.Graph()
        foo_value = gr.addInput("foo")
        assert foo_value in gr.inputs()

    def test_canonicalize(self):
        ir = """
graph(%p207 : Tensor,
      %1 : Tensor,
      %p407 : int):
  %11 : Tensor = aten::view_expand_placeholder(%1)
  %12 : Tensor = aten::pointwise_placeholder(%11, %p207, %p407)
  %13 : Tensor = aten::view_expand_placeholder(%12)
  %14 : Tensor = aten::pointwise_placeholder(%13)
  return (%14)
        """

        graph1 = torch._C.parse_ir(ir)
        graph1 = torch._C._jit_pass_canonicalize(graph1, True)

        graph2 = torch._C.parse_ir(ir)
        graph2 = torch._C._jit_pass_canonicalize(graph2)

        self.assertEqual(str(graph1), str(graph2))
        FileCheck().check("%p207").check_not("%14").run(graph1)

        graph3 = torch._C.parse_ir(ir)
        graph3 = torch._C._jit_pass_canonicalize(graph3, False)
        FileCheck().check_not("%p207").run(graph3)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")

```



## High-Level Overview


This Python file contains 1 class(es) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestPythonBindings`

**Functions defined**: `test_cu_get_functions`, `test_get_python_cu_fn`, `test_cu_create_function`, `fn`, `test_invalidation`, `test_invalidation_fn`, `test_graph_iterator_keepalive`, `test_iterator_keepalive_fn`, `test_aliasdb`, `test_aliasdb_fn`, `test_graph_create`, `test_add_input`, `test_canonicalize`

**Key imports**: torch, FileCheck, raise_on_run_directly, JitTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.testing`: FileCheck
- `torch.testing._internal.common_utils`: raise_on_run_directly
- `torch.testing._internal.jit_utils`: JitTestCase


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/jit/test_python_bindings.py
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
- [`test_parametrization.py_docs.md`](./test_parametrization.py_docs.md)


## Cross-References

- **File Documentation**: `test_python_bindings.py_docs.md`
- **Keyword Index**: `test_python_bindings.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/jit`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/jit/test_python_bindings.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/jit`):

- [`test_attr.py_kw.md_docs.md`](./test_attr.py_kw.md_docs.md)
- [`test_parametrization.py_kw.md_docs.md`](./test_parametrization.py_kw.md_docs.md)
- [`test_hooks.py_kw.md_docs.md`](./test_hooks.py_kw.md_docs.md)
- [`test_dataclasses.py_docs.md_docs.md`](./test_dataclasses.py_docs.md_docs.md)
- [`test_aten_pow.py_kw.md_docs.md`](./test_aten_pow.py_kw.md_docs.md)
- [`test_misc.py_docs.md_docs.md`](./test_misc.py_docs.md_docs.md)
- [`test_graph_rewrite_passes.py_kw.md_docs.md`](./test_graph_rewrite_passes.py_kw.md_docs.md)
- [`test_module_containers.py_kw.md_docs.md`](./test_module_containers.py_kw.md_docs.md)
- [`test_complex.py_kw.md_docs.md`](./test_complex.py_kw.md_docs.md)
- [`test_types.py_kw.md_docs.md`](./test_types.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_python_bindings.py_docs.md_docs.md`
- **Keyword Index**: `test_python_bindings.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
