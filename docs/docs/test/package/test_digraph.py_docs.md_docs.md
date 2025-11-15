# Documentation: `docs/test/package/test_digraph.py_docs.md`

## File Metadata

- **Path**: `docs/test/package/test_digraph.py_docs.md`
- **Size**: 7,708 bytes (7.53 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/package/test_digraph.py`

## File Metadata

- **Path**: `test/package/test_digraph.py`
- **Size**: 3,697 bytes (3.61 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: package/deploy"]

from torch.package._digraph import DiGraph
from torch.testing._internal.common_utils import run_tests


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestDiGraph(PackageTestCase):
    """Test the DiGraph structure we use to represent dependencies in PackageExporter"""

    def test_successors(self):
        g = DiGraph()
        g.add_edge("foo", "bar")
        g.add_edge("foo", "baz")
        g.add_node("qux")

        self.assertIn("bar", list(g.successors("foo")))
        self.assertIn("baz", list(g.successors("foo")))
        self.assertEqual(len(list(g.successors("qux"))), 0)

    def test_predecessors(self):
        g = DiGraph()
        g.add_edge("foo", "bar")
        g.add_edge("foo", "baz")
        g.add_node("qux")

        self.assertIn("foo", list(g.predecessors("bar")))
        self.assertIn("foo", list(g.predecessors("baz")))
        self.assertEqual(len(list(g.predecessors("qux"))), 0)

    def test_successor_not_in_graph(self):
        g = DiGraph()
        with self.assertRaises(ValueError):
            g.successors("not in graph")

    def test_predecessor_not_in_graph(self):
        g = DiGraph()
        with self.assertRaises(ValueError):
            g.predecessors("not in graph")

    def test_node_attrs(self):
        g = DiGraph()
        g.add_node("foo", my_attr=1, other_attr=2)
        self.assertEqual(g.nodes["foo"]["my_attr"], 1)
        self.assertEqual(g.nodes["foo"]["other_attr"], 2)

    def test_node_attr_update(self):
        g = DiGraph()
        g.add_node("foo", my_attr=1)
        self.assertEqual(g.nodes["foo"]["my_attr"], 1)

        g.add_node("foo", my_attr="different")
        self.assertEqual(g.nodes["foo"]["my_attr"], "different")

    def test_edges(self):
        g = DiGraph()
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g.add_edge(1, 3)
        g.add_edge(4, 5)

        edge_list = list(g.edges)
        self.assertEqual(len(edge_list), 4)

        self.assertIn((1, 2), edge_list)
        self.assertIn((2, 3), edge_list)
        self.assertIn((1, 3), edge_list)
        self.assertIn((4, 5), edge_list)

    def test_iter(self):
        g = DiGraph()
        g.add_node(1)
        g.add_node(2)
        g.add_node(3)

        nodes = set()
        nodes.update(g)

        self.assertEqual(nodes, {1, 2, 3})

    def test_contains(self):
        g = DiGraph()
        g.add_node("yup")

        self.assertTrue("yup" in g)
        self.assertFalse("nup" in g)

    def test_contains_non_hashable(self):
        g = DiGraph()
        self.assertFalse([1, 2, 3] in g)

    def test_forward_closure(self):
        g = DiGraph()
        g.add_edge("1", "2")
        g.add_edge("2", "3")
        g.add_edge("5", "4")
        g.add_edge("4", "3")
        self.assertTrue(g.forward_transitive_closure("1") == {"1", "2", "3"})
        self.assertTrue(g.forward_transitive_closure("4") == {"4", "3"})

    def test_all_paths(self):
        g = DiGraph()
        g.add_edge("1", "2")
        g.add_edge("1", "7")
        g.add_edge("7", "8")
        g.add_edge("8", "3")
        g.add_edge("2", "3")
        g.add_edge("5", "4")
        g.add_edge("4", "3")

        result = g.all_paths("1", "3")
        # to get rid of indeterminism
        actual = {i.strip("\n") for i in result.split(";")[2:-1]}
        expected = {
            '"2" -> "3"',
            '"1" -> "7"',
            '"7" -> "8"',
            '"1" -> "2"',
            '"8" -> "3"',
        }
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""Test the DiGraph structure we use to represent dependencies in PackageExporter"""    def test_successors(self):        g = DiGraph()        g.add_edge("foo", "bar")        g.add_edge("foo", "baz")        g.add_node("qux")        self.assertIn("bar", list(g.successors("foo")))        self.assertIn("baz", list(g.successors("foo")))        self.assertEqual(len(list(g.successors("qux"))), 0)    def test_predecessors(self):        g = DiGraph()        g.add_edge("foo", "bar")        g.add_edge("foo", "baz")        g.add_node("qux")        self.assertIn("foo", list(g.predecessors("bar")))        self.assertIn("foo", list(g.predecessors("baz")))        self.assertEqual(len(list(g.predecessors("qux"))), 0)    def test_successor_not_in_graph(self):        g = DiGraph()        with self.assertRaises(ValueError):            g.successors("not in graph")    def test_predecessor_not_in_graph(self):        g = DiGraph()        with self.assertRaises(ValueError):            g.predecessors("not in graph")    def test_node_attrs(self):        g = DiGraph()        g.add_node("foo", my_attr=1, other_attr=2)        self.assertEqual(g.nodes["foo"]["my_attr"], 1)

This Python file contains 1 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestDiGraph`

**Functions defined**: `test_successors`, `test_predecessors`, `test_successor_not_in_graph`, `test_predecessor_not_in_graph`, `test_node_attrs`, `test_node_attr_update`, `test_edges`, `test_iter`, `test_contains`, `test_contains_non_hashable`, `test_forward_closure`, `test_all_paths`

**Key imports**: DiGraph, run_tests, PackageTestCase, PackageTestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/package`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch.package._digraph`: DiGraph
- `torch.testing._internal.common_utils`: run_tests
- `.common`: PackageTestCase
- `common`: PackageTestCase


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


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
python test/package/test_digraph.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/package`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_directory_reader.py_docs.md`](./test_directory_reader.py_docs.md)
- [`test_dependency_api.py_docs.md`](./test_dependency_api.py_docs.md)
- [`module_a.py_docs.md`](./module_a.py_docs.md)
- [`test_model.py_docs.md`](./test_model.py_docs.md)
- [`module_a_remapped_path.py_docs.md`](./module_a_remapped_path.py_docs.md)
- [`test_glob_group.py_docs.md`](./test_glob_group.py_docs.md)
- [`test_load_bc_packages.py_docs.md`](./test_load_bc_packages.py_docs.md)
- [`test_mangling.py_docs.md`](./test_mangling.py_docs.md)


## Cross-References

- **File Documentation**: `test_digraph.py_docs.md`
- **Keyword Index**: `test_digraph.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/package`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/package`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Error Handling**: Includes exception handling


## Performance Considerations

### Performance Notes

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
python docs/test/package/test_digraph.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/package`):

- [`test_mangling.py_docs.md_docs.md`](./test_mangling.py_docs.md_docs.md)
- [`test_analyze.py_docs.md_docs.md`](./test_analyze.py_docs.md_docs.md)
- [`test_misc.py_docs.md_docs.md`](./test_misc.py_docs.md_docs.md)
- [`test_resources.py_kw.md_docs.md`](./test_resources.py_kw.md_docs.md)
- [`generate_bc_packages.py_kw.md_docs.md`](./generate_bc_packages.py_kw.md_docs.md)
- [`test_package_fx.py_docs.md_docs.md`](./test_package_fx.py_docs.md_docs.md)
- [`test_repackage.py_kw.md_docs.md`](./test_repackage.py_kw.md_docs.md)
- [`test_importer.py_kw.md_docs.md`](./test_importer.py_kw.md_docs.md)
- [`test_repackage.py_docs.md_docs.md`](./test_repackage.py_docs.md_docs.md)
- [`common.py_docs.md_docs.md`](./common.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_digraph.py_docs.md_docs.md`
- **Keyword Index**: `test_digraph.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
