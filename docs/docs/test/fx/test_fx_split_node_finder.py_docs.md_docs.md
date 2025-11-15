# Documentation: `docs/test/fx/test_fx_split_node_finder.py_docs.md`

## File Metadata

- **Path**: `docs/test/fx/test_fx_split_node_finder.py_docs.md`
- **Size**: 9,923 bytes (9.69 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/fx/test_fx_split_node_finder.py`

## File Metadata

- **Path**: `test/fx/test_fx_split_node_finder.py`
- **Size**: 6,794 bytes (6.63 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```python
# Owner(s): ["module: fx"]

# pyre-strict
import os
import shutil
import sys
import tempfile

import torch
from torch.fx.passes.operator_support import OperatorSupportBase
from torch.fx.passes.splitter_base import (
    ALL_SUFFIX,
    ENV_FX_NET_ACC_SPLITTER_TRACKER_TRACKED_NODES,
    FxNetAccNodesFinder,
    NodeEventTracker,
    NODES_SUFFIX,
    ShapeProp,
)
from torch.testing._internal.common_utils import TestCase


# Wrappepr function to make it supported
@torch.fx.wrap
def sup_f(x):
    return x


class TestFxSplitNodeFinder(TestCase):
    def setUp(self):
        super().setUp()
        self.save_path = sys.path[:]
        self.tmpdir = tempfile.mkdtemp()
        sys.path.insert(0, self.tmpdir)

    def tearDown(self):
        sys.path[:] = self.save_path
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _testTrackerBasics(self, tracker):
        """
        Test the basic functionalities of the tracker by putting it into a
        node finder and examine the events generated after the finder is called.
        """

        def getEvents(tracker, node):
            return [tracker.events[idx] for idx in tracker.node_events[node.name]]

        class IsNodeSupported(OperatorSupportBase):
            def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
                return "sup_" in node.name

        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                x = sup_f(x)
                y = sup_f(y)
                b = x + y  # non-supported to break graph
                return sup_f(b)

        gm = torch.fx.symbolic_trace(TestModule())
        ShapeProp(gm).propagate(*(torch.rand((2, 2)), 3))

        finder = FxNetAccNodesFinder(gm, IsNodeSupported(), False)
        finder.tracker = tracker
        acc_nodes = finder()
        # check that acc nodes events are as expected
        for node in gm.graph.nodes:
            if node.name == "sup_f_1":
                # this node should be removed from acc nodes.
                self.assertFalse(node in acc_nodes)
                events = getEvents(tracker, node)
                # 2 events.
                self.assertEqual(len(events), 2)
                # 1st event is init_acc as supported operator
                self.assertTrue(
                    events[0].desc.startswith(
                        "init_acc|callable_and_operator_supported"
                    )
                )
                # 2nd event is del_acc as non-tensor output with cpu user
                self.assertTrue(
                    events[1].desc.startswith("acc_del|non_tensor_output_with_cpu_user")
                )
            elif node.name.startswith("sup_f"):
                # other supported nodes should remain in acc nodes.
                self.assertTrue(node in acc_nodes)
                events = getEvents(tracker, node)
                self.assertEqual(len(events), 1)
                self.assertTrue(
                    events[0].desc.startswith(
                        "init_acc|callable_and_operator_supported"
                    )
                )
            else:
                # other nodes are on cpu.
                self.assertFalse(node in acc_nodes)

    def _validate_file_content(self, filepath, expected_lines):
        """
        Validate the content of the file.
        Args:
            filepath: the path of the file to be validated.
            expected_lines: the expected lines of the file.
        Returns:
            None
        """
        with open(filepath) as f:
            idx = 0
            for line in f:
                self.assertEqual(line.rstrip("\n"), expected_lines[idx])
                idx += 1
                self.assertTrue(idx <= len(expected_lines))
            self.assertEqual(idx, len(expected_lines))

    def _assert_events_file(self, events_file):
        self._validate_file_content(
            events_file,
            [
                "Node: x:",
                "  x: init_cpu|not_callable #",
                "Node: y:",
                "  y: init_cpu|not_callable #",
                "Node: sup_f:",
                "  sup_f: init_acc|callable_and_operator_supported #",
                "Node: sup_f_1:",
                "  sup_f_1: init_acc|callable_and_operator_supported #",
                "  sup_f_1: acc_del|non_tensor_output_with_cpu_user add",
                "Node: add:",
                "  add: init_cpu|operator_support #",
                "Node: sup_f_2:",
                "  sup_f_2: init_acc|callable_and_operator_supported #",
                "Node: output:",
                "  output: init_cpu|not_callable #",
            ],
        )

    def _testTrackerMode(self, mode):
        """
        Test the tracker with different modes.
        Args:
            mode: the mode to be tested.
            - 0: no local dump
            - 1: dump all events to file
            - 2: dump specific nodes in recursive manner
            - 3: dump all nodes with more than 1 event in recursive manner.
        """
        tmp_dump_base_path = self.tmpdir + "/" + str(mode)
        tracker = NodeEventTracker(
            mode,  # mode: just enable the tracker without dumping
            tmp_dump_base_path,  # dump path
        )
        events_file = tmp_dump_base_path + ALL_SUFFIX
        nodes_file = tmp_dump_base_path + NODES_SUFFIX
        self.assertFalse(os.path.exists(events_file))
        self.assertFalse(os.path.exists(nodes_file))
        self._testTrackerBasics(tracker)

        if mode == 0:
            # Make sure there are no files dumped
            self.assertFalse(os.path.exists(events_file))
            self.assertFalse(os.path.exists(nodes_file))
        elif mode == 1:
            self._assert_events_file(events_file)
            self.assertFalse(os.path.exists(nodes_file))
        elif mode == 2:
            self._assert_events_file(events_file)
            self._validate_file_content(
                nodes_file,
                ["|-sup_f_2: init_acc|callable_and_operator_supported #"],
            )
        elif mode == 3:
            self._assert_events_file(events_file)
            self._validate_file_content(
                nodes_file,
                [
                    "|-sup_f_1: init_acc|callable_and_operator_supported #",
                    "|-sup_f_1: acc_del|non_tensor_output_with_cpu_user add",
                    "| |-add: init_cpu|operator_support #",
                ],
            )

    def testMode0(self):
        self._testTrackerMode(0)

    def testMode1(self):
        self._testTrackerMode(1)

    def testMode2(self):
        os.environ[ENV_FX_NET_ACC_SPLITTER_TRACKER_TRACKED_NODES] = "sup_f_2"
        self._testTrackerMode(2)

    def testMode3(self):
        self._testTrackerMode(3)

```



## High-Level Overview

"""        Test the basic functionalities of the tracker by putting it into a        node finder and examine the events generated after the finder is called.

This Python file contains 3 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestFxSplitNodeFinder`, `IsNodeSupported`, `TestModule`

**Functions defined**: `sup_f`, `setUp`, `tearDown`, `_testTrackerBasics`, `getEvents`, `is_node_supported`, `forward`, `_validate_file_content`, `_assert_events_file`, `_testTrackerMode`, `testMode0`, `testMode1`, `testMode2`, `testMode3`

**Key imports**: os, shutil, sys, tempfile, torch, OperatorSupportBase, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/fx`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `shutil`
- `sys`
- `tempfile`
- `torch`
- `torch.fx.passes.operator_support`: OperatorSupportBase
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
python test/fx/test_fx_split_node_finder.py
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

- **File Documentation**: `test_fx_split_node_finder.py_docs.md`
- **Keyword Index**: `test_fx_split_node_finder.py_kw.md`
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
python docs/test/fx/test_fx_split_node_finder.py_docs.md
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

- **File Documentation**: `test_fx_split_node_finder.py_docs.md_docs.md`
- **Keyword Index**: `test_fx_split_node_finder.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
