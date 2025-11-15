# Documentation: `docs/test/fx/test_dynamism.py_docs.md`

## File Metadata

- **Path**: `docs/test/fx/test_dynamism.py_docs.md`
- **Size**: 8,623 bytes (8.42 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/fx/test_dynamism.py`

## File Metadata

- **Path**: `test/fx/test_dynamism.py`
- **Size**: 5,706 bytes (5.57 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: fx"]

import torch
from torch.fx.experimental._dynamism import track_dynamism_across_examples
from torch.testing._internal.common_utils import TestCase


class TestDynamism(TestCase):
    def test_dynamic_tensor(self):
        ex1 = {"x": 1, "y": torch.ones(1, 1), "z": {0: torch.ones(1)}}
        ex2 = {"x": 2, "y": torch.ones(2, 1), "z": {0: torch.ones(2)}}
        ex3 = {"x": 3, "y": torch.ones(3, 1), "z": {0: torch.ones(3)}}
        ex4 = {"x": 4, "y": torch.ones(4, 1), "z": {0: torch.ones(4)}}
        ex5 = {"x": 5, "y": torch.ones(5, 1), "z": {0: torch.ones(5)}}
        examples = [ex1, ex2, ex3, ex4, ex5]

        result = track_dynamism_across_examples(examples)
        expected = {
            "x": {"L['x']": (True,)},
            "y": {"L['y']": (True, False)},
            "z": {"L['z'][0]": (True,)},
        }
        self.assertEqual(result, expected)

    def test_dynamic_tensor_deeply_nested(self):
        ex1 = {"z": {"z": {"z": {"z": {0: torch.ones(1)}}}}}
        ex2 = {"z": {"z": {"z": {"z": {0: torch.ones(2)}}}}}
        ex3 = {"z": {"z": {"z": {"z": {0: torch.ones(3)}}}}}
        ex4 = {"z": {"z": {"z": {"z": {0: torch.ones(4)}}}}}
        ex5 = {"z": {"z": {"z": {"z": {0: torch.ones(5)}}}}}
        examples = [ex1, ex2, ex3, ex4, ex5]

        result = track_dynamism_across_examples(examples)
        expected = {
            "z": {
                "L['z']['z']['z']['z'][0]": (True,),
            },
        }
        self.assertEqual(result, expected)

    def test_mixed_dynamism(self):
        ex1 = {"a": torch.ones(1, 2), "b": [torch.ones(1), 3], "c": {"d": 42}}
        ex2 = {"a": torch.ones(2, 2), "b": [torch.ones(2), 4], "c": {"d": 42}}
        ex3 = {"a": torch.ones(3, 2), "b": [torch.ones(3), 5], "c": {"d": 42}}
        ex4 = {"a": torch.ones(4, 2), "b": [torch.ones(4), 6], "c": {"d": 42}}
        ex5 = {"a": torch.ones(5, 2), "b": [torch.ones(5), 7], "c": {"d": 42}}
        examples = [ex1, ex2, ex3, ex4, ex5]

        result = track_dynamism_across_examples(examples)
        expected = {
            "a": {"L['a']": (True, False)},
            "b": {"L['b'][0]": (True,), "L['b'][1]": (True,)},
            "c": {"L['c']['d']": (False,)},
        }
        self.assertEqual(result, expected)

    def test_nn_module(self):
        class Y(torch.nn.Module):
            def __init__(self, n_input, n_output):
                super().__init__()
                self.compress = torch.nn.Linear(n_input, n_output)
                self.x = n_input

            def forward(self, x):
                return self.compress(x) * self.x

        class M(torch.nn.Module):
            def __init__(self, n_input, n_output):
                self.n_input = n_input
                self.n_output = n_output
                super().__init__()
                self.y = Y(n_input, n_output)

            def forward(self, x):
                return self.y(x)

        model1 = M(3210, 30)
        model2 = M(3211, 30)

        result = track_dynamism_across_examples(
            [
                {"self": model1},
                {"self": model2},
            ]
        )
        expected = {
            "self": {
                "L['self']['_modules']['y']['_modules']['compress']['_parameters']['weight']": (
                    False,
                    True,
                ),
                "L['self']['_modules']['y']['_modules']['compress']['_parameters']['bias']": (
                    False,
                ),
                "L['self']['_modules']['y']['_modules']['compress']['bias']": (False,),
                "L['self']['_modules']['y']['_modules']['compress']['in_features']": (
                    True,
                ),
                "L['self']['_modules']['y']['_modules']['compress']['out_features']": (
                    False,
                ),
                "L['self']['_modules']['y']['_modules']['compress']['weight']": (
                    False,
                    True,
                ),
                "L['self']['_modules']['y']['x']": (True,),
                "L['self']['n_input']": (True,),
                "L['self']['n_output']": (False,),
            }
        }
        self.assertEqual(result, expected)

    def test_property_not_implemented(self):
        class ModuleWithNotImplementedProperty(torch.nn.Module):
            def __init__(self, x, y):
                super().__init__()
                self.linear = torch.nn.Linear(x, y)

            @property
            def not_implemented_property(self):
                raise NotImplementedError("This property is not implemented")

        module1 = ModuleWithNotImplementedProperty(10, 10)
        module2 = ModuleWithNotImplementedProperty(10, 10)

        result = track_dynamism_across_examples(
            [
                {"self": module1},
                {"self": module2},
            ]
        )

        expected = {
            "self": {
                "L['self']['_modules']['linear']['_parameters']['weight']": (
                    False,
                    False,
                ),
                "L['self']['_modules']['linear']['_parameters']['bias']": (False,),
                "L['self']['_modules']['linear']['bias']": (False,),
                "L['self']['_modules']['linear']['in_features']": (False,),
                "L['self']['_modules']['linear']['out_features']": (False,),
                "L['self']['_modules']['linear']['weight']": (False, False),
            }
        }

        self.assertEqual(result, expected)


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )

```



## High-Level Overview


This Python file contains 4 class(es) and 11 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestDynamism`, `Y`, `M`, `ModuleWithNotImplementedProperty`

**Functions defined**: `test_dynamic_tensor`, `test_dynamic_tensor_deeply_nested`, `test_mixed_dynamism`, `test_nn_module`, `__init__`, `forward`, `__init__`, `forward`, `test_property_not_implemented`, `__init__`, `not_implemented_property`

**Key imports**: torch, track_dynamism_across_examples, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/fx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.fx.experimental._dynamism`: track_dynamism_across_examples
- `torch.testing._internal.common_utils`: TestCase


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
python test/fx/test_dynamism.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/fx`):

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

- **File Documentation**: `test_dynamism.py_docs.md`
- **Keyword Index**: `test_dynamism.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/fx`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/fx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

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
python docs/test/fx/test_dynamism.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/fx`):

- [`named_tup.py_kw.md_docs.md`](./named_tup.py_kw.md_docs.md)
- [`test_dynamism.py_kw.md_docs.md`](./test_dynamism.py_kw.md_docs.md)
- [`test_fx_traceback.py_docs.md_docs.md`](./test_fx_traceback.py_docs.md_docs.md)
- [`test_fx_xform_observer.py_docs.md_docs.md`](./test_fx_xform_observer.py_docs.md_docs.md)
- [`test_pass_infra.py_kw.md_docs.md`](./test_pass_infra.py_kw.md_docs.md)
- [`test_fx_xform_observer.py_kw.md_docs.md`](./test_fx_xform_observer.py_kw.md_docs.md)
- [`test_fx_node_hook.py_kw.md_docs.md`](./test_fx_node_hook.py_kw.md_docs.md)
- [`test_partitioner_order.py_docs.md_docs.md`](./test_partitioner_order.py_docs.md_docs.md)
- [`test_subgraph_rewriter.py_kw.md_docs.md`](./test_subgraph_rewriter.py_kw.md_docs.md)
- [`test_fx_split.py_docs.md_docs.md`](./test_fx_split.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_dynamism.py_docs.md_docs.md`
- **Keyword Index**: `test_dynamism.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
