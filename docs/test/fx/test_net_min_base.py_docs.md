# Documentation: `test/fx/test_net_min_base.py`

## File Metadata

- **Path**: `test/fx/test_net_min_base.py`
- **Size**: 4,098 bytes (4.00 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: fx"]

from unittest import mock

import torch
from torch.fx.passes.net_min_base import (
    _MinimizerBase,
    _MinimizerSettingBase,
    FxNetMinimizerResultMismatchError,
)
from torch.fx.passes.tools_common import Names
from torch.testing._internal.common_utils import TestCase


class TestNetMinBaseBlock(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Setup test fixtures for each test method

        class SimpleModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
                self.linear2 = torch.nn.Linear(5, 5)
                self.relu = torch.nn.ReLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.linear(x)
                x = self.linear2(x)
                x = self.relu(x)
                return x

        self.compare_fn = mock.MagicMock()

        self.module = torch.fx.symbolic_trace(SimpleModule())
        self.sample_input = (torch.randn(2, 10),)
        self.settings = _MinimizerSettingBase(traverse_method="block")
        self.minimizer = _MinimizerBase(
            module=self.module,
            sample_input=self.sample_input,
            settings=self.settings,
            compare_fn=self.compare_fn,
        )
        self.report = []

    def assert_problematic_nodes(self, culprit_names: Names) -> None:
        """
        Quick helper function to assert that a set of nodes (when present together in a subgraph) cause a discrepancy
        """
        with mock.patch("torch.fx.passes.net_min_base._MinimizerBase._run_and_compare"):

            def run_and_compare_side_effect(
                split_module: torch.fx.GraphModule,
                submod_name: str,
                output_names: Names,
                report_idx: int = -1,
            ) -> None:
                submodule = getattr(split_module, submod_name)

                # Remove input/output layer
                names = set([node.name for node in submodule.graph.nodes][1:-1])
                if set(culprit_names) <= names:
                    raise FxNetMinimizerResultMismatchError

            self.minimizer._run_and_compare.side_effect = run_and_compare_side_effect

            # Every single node should be a discrepancy
            culprits = self.minimizer.minimize()
            self.assertEqual({node.name for node in culprits}, set(culprit_names))

    def test_no_discrepancy(self) -> None:
        # No discrepancies should handle gracefully with an empty set
        with (
            mock.patch("torch.fx.passes.net_min_base._MinimizerBase.run_a"),
            mock.patch("torch.fx.passes.net_min_base._MinimizerBase.run_b"),
        ):
            # Have both run_a and run_b return the same result
            return_value = torch.zeros((2, 5))
            self.minimizer.run_a.return_value = return_value
            self.minimizer.run_b.return_value = return_value
            self.compare_fn.return_value = (0, True)

            # There should be no discrepancy between the two, and thus we should receive an empty set
            culprits = self.minimizer.minimize()
            self.assertEqual(culprits, set())

    def test_all_nodes_discrepancy(self) -> None:
        self.assert_problematic_nodes(["linear", "linear2", "relu"])

    def test_first_node_discrepancy(self) -> None:
        self.assert_problematic_nodes(["linear"])

    def test_last_node_discrepancy(self) -> None:
        self.assert_problematic_nodes(["relu"])

    def test_middle_node_discrepancy(self) -> None:
        self.assert_problematic_nodes(["linear2"])

    def test_contiguous_partial_discrepancy_end(self) -> None:
        self.assert_problematic_nodes(["linear2", "relu"])

    def test_continugous_partial_discrepancy_beginning(self) -> None:
        self.assert_problematic_nodes(["linear", "linear2"])


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )

```



## High-Level Overview

"""        Quick helper function to assert that a set of nodes (when present together in a subgraph) cause a discrepancy

This Python file contains 2 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestNetMinBaseBlock`, `SimpleModule`

**Functions defined**: `setUp`, `__init__`, `forward`, `assert_problematic_nodes`, `run_and_compare_side_effect`, `test_no_discrepancy`, `test_all_nodes_discrepancy`, `test_first_node_discrepancy`, `test_last_node_discrepancy`, `test_middle_node_discrepancy`, `test_contiguous_partial_discrepancy_end`, `test_continugous_partial_discrepancy_beginning`

**Key imports**: mock, torch, Names, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/fx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`: mock
- `torch`
- `torch.fx.passes.tools_common`: Names
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
python test/fx/test_net_min_base.py
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

- **File Documentation**: `test_net_min_base.py_docs.md`
- **Keyword Index**: `test_net_min_base.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
