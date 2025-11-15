# Documentation: `test/distributed/optim/test_named_optimizer.py`

## File Metadata

- **Path**: `test/distributed/optim/test_named_optimizer.py`
- **Size**: 15,000 bytes (14.65 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
from torch.distributed.optim import _NamedOptimizer


def _run_model_training(model_optim_lists):
    for _ in range(2):
        x = torch.rand(5, 8)
        for model_optim_list in model_optim_lists:
            model = model_optim_list[0]
            optim_list = model_optim_list[1]
            y = model(x)
            y.sum().backward()
            for optim in optim_list:
                optim.step()


class TestDummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))


class NamedOptimizerTest(unittest.TestCase):
    def _compare_state_dict_group(self, group, named_group, assert_equal=True):
        for key, val in group.items():
            if key != "params":
                self.assertTrue(
                    key in named_group, f"{key} not in named optimizer state dict"
                )
                err_msg = (
                    f"{key} state not equal" if assert_equal else f"{key} state equal"
                )
                if isinstance(val, torch.Tensor):
                    fn = self.assertTrue if assert_equal else self.assertFalse
                    fn(torch.allclose(val, named_group[key]), err_msg)
                else:
                    fn = self.assertEqual if assert_equal else self.assertNotEqual
                    fn(val, named_group[key], err_msg)

    def _compare_param_groups(self, param_groups_1, param_groups_2):
        self.assertTrue(isinstance(param_groups_1, list))
        self.assertTrue(isinstance(param_groups_2, list))
        for groups in zip(param_groups_1, param_groups_2):
            self._compare_param_group(groups[0], groups[1])

    def _compare_param_group(self, group_1, group_2):
        self.assertTrue(isinstance(group_1, dict))
        self.assertTrue(isinstance(group_2, dict))
        for key, val in group_1.items():
            self.assertTrue(key in group_2)
            if key != "params":
                self.assertEqual(val, group_2[key])
            else:
                for tensors in zip(val, group_2[key]):
                    self.assertTrue(torch.allclose(tensors[0], tensors[1]))

    def test_state_dict(self):
        """Check that NamedOptimizer exposes the expected state dict
        interface."""
        m = TestDummyModel()
        m_dup = TestDummyModel()
        optim = torch.optim.SGD(
            m.parameters(),
            lr=1e-2,
            momentum=0.9,
        )

        named_optim = _NamedOptimizer(
            m_dup.named_parameters(),
            torch.optim.SGD,
            lr=1e-2,
            momentum=0.9,
        )
        self._compare_param_groups(optim.param_groups, named_optim.param_groups)

        _run_model_training([(m, [optim]), (m_dup, [named_optim])])
        self._compare_param_groups(optim.param_groups, named_optim.param_groups)

        sd = optim.state_dict()
        named_sd = named_optim.state_dict()

        # Compare "state" in optim state dict
        self._compare_state_dict_group(
            sd["state"][0],
            named_sd["state"]["net1.0.weight"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd["state"][3],
            named_sd["state"]["net2.0.bias"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd["state"][4],
            named_sd["state"]["net3.weight"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd["state"][7],
            named_sd["state"]["net4.1.bias"],
            assert_equal=True,
        )

    def test_state_dict_multi_param_group(self):
        """Check that NamedOptimizer exposes the expected state dict
        interface when multiple param groups are specified."""
        m = TestDummyModel()
        m_dup = TestDummyModel()
        optim_1 = torch.optim.SGD(
            [
                {"params": m.net1.parameters()},
                {"params": m.net3.parameters(), "lr": 1e-3},
            ],
            lr=1e-2,
            momentum=0.9,
        )

        optim_2 = torch.optim.Adam(
            [
                {"params": m.net2.parameters()},
                {"params": m.net4.parameters(), "lr": 1e-5},
            ]
        )

        named_optim_1 = _NamedOptimizer(
            m_dup.named_parameters(),
            torch.optim.SGD,
            [
                {"params": m_dup.net1.parameters()},
                {"params": m_dup.net3.parameters(), "lr": 1e-3},
            ],
            lr=1e-2,
            momentum=0.9,
        )

        named_optim_2 = _NamedOptimizer(
            m_dup.named_parameters(),
            torch.optim.Adam,
            [
                {"params": m_dup.net2.parameters()},
                {"params": m_dup.net4.parameters(), "lr": 1e-5},
            ],
        )
        self._compare_param_groups(optim_1.param_groups, named_optim_1.param_groups)
        self._compare_param_groups(optim_2.param_groups, named_optim_2.param_groups)

        _run_model_training(
            [(m, [optim_1, optim_2]), (m_dup, [named_optim_1, named_optim_2])]
        )
        self._compare_param_groups(optim_1.param_groups, named_optim_1.param_groups)
        self._compare_param_groups(optim_2.param_groups, named_optim_2.param_groups)
        sd_1 = optim_1.state_dict()
        sd_2 = optim_2.state_dict()
        named_sd_1 = named_optim_1.state_dict()
        named_sd_2 = named_optim_2.state_dict()

        # Compare "state" in optim state dict
        self._compare_state_dict_group(
            sd_1["state"][0],
            named_sd_1["state"]["net1.0.weight"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd_2["state"][1],
            named_sd_2["state"]["net2.0.bias"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd_1["state"][2],
            named_sd_1["state"]["net3.weight"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd_2["state"][3],
            named_sd_2["state"]["net4.1.bias"],
            assert_equal=True,
        )

        # Compare "param_groups" in optim state dict
        self._compare_state_dict_group(
            sd_1["param_groups"][0],
            named_sd_1["param_groups"][0],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            sd_2["param_groups"][1], named_sd_2["param_groups"][1], assert_equal=True
        )

    def test_load_state_dict(self):
        """Check that NamedOptimizer's load_state_dict works as expected."""
        m = TestDummyModel()
        named_optim_1 = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            lr=1e-2,
            momentum=0.9,
        )

        _run_model_training([(m, [named_optim_1])])
        state_dict_to_load = named_optim_1.state_dict()

        named_optim_2 = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            lr=1e-2,
            momentum=0.6,
        )

        _run_model_training([(m, [named_optim_2])])
        state_dict_before_load = named_optim_2.state_dict()

        # Compare "state" in optim state dict
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net1.0.weight"],
            state_dict_before_load["state"]["net1.0.weight"],
            assert_equal=False,
        )
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net2.0.bias"],
            state_dict_before_load["state"]["net2.0.bias"],
            assert_equal=False,
        )
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net3.weight"],
            state_dict_before_load["state"]["net3.weight"],
            assert_equal=False,
        )
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net4.1.bias"],
            state_dict_before_load["state"]["net4.1.bias"],
            assert_equal=False,
        )

        named_optim_2.load_state_dict(state_dict_to_load)
        state_dict_after_load = named_optim_2.state_dict()

        # Compare "state" in optim state dict
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net1.0.weight"],
            state_dict_after_load["state"]["net1.0.weight"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net2.0.bias"],
            state_dict_after_load["state"]["net2.0.bias"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net3.weight"],
            state_dict_after_load["state"]["net3.weight"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net4.1.bias"],
            state_dict_after_load["state"]["net4.1.bias"],
            assert_equal=True,
        )

    def test_load_state_dict_conditional_training(self):
        """Check that NamedOptimizer load_state_dict works under conditional training case."""
        m = TestDummyModel()
        named_optim_1 = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            [
                {"params": m.net1.parameters()},
                {"params": m.net3.parameters(), "lr": 1e-3},
            ],
            lr=1e-2,
            momentum=0.9,
        )

        _run_model_training([(m, [named_optim_1])])
        state_dict_to_load = named_optim_1.state_dict()

        named_optim_2 = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            lr=1e-2,
            momentum=0.6,
        )

        _run_model_training([(m, [named_optim_2])])
        named_optim_2.load_state_dict(state_dict_to_load)
        state_dict_after_load = named_optim_2.state_dict()

        # Compare "state" in optim state dict
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net1.0.weight"],
            state_dict_after_load["state"]["net1.0.weight"],
            assert_equal=True,
        )
        self._compare_state_dict_group(
            state_dict_to_load["state"]["net3.weight"],
            state_dict_after_load["state"]["net3.weight"],
            assert_equal=True,
        )

    def test_load_state_dict_error(self):
        m = TestDummyModel()
        named_optim_1 = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            lr=1e-2,
            momentum=0.9,
        )

        _run_model_training([(m, [named_optim_1])])
        state_dict_to_load = named_optim_1.state_dict()

        named_optim_2 = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            lr=1e-2,
            momentum=0.6,
        )

        err_msg = (
            "Expects the optim to be initialized before load but found not initialized"
        )
        with self.assertRaisesRegex(ValueError, err_msg):
            named_optim_2.load_state_dict(state_dict_to_load)

    def test_add_param_group(self):
        m = TestDummyModel()
        m_dup = TestDummyModel()
        optim = torch.optim.SGD(
            [
                {"params": m.net1.parameters()},
                {"params": m.net3.parameters(), "lr": 1e-3},
            ],
            lr=1e-2,
            momentum=0.9,
        )
        named_optim = _NamedOptimizer(
            m_dup.named_parameters(),
            torch.optim.SGD,
            [
                {"params": m_dup.net1.parameters()},
                {"params": m_dup.net3.parameters(), "lr": 1e-3},
            ],
            lr=1e-2,
            momentum=0.9,
        )

        _run_model_training([(m, [optim]), (m_dup, [named_optim])])
        self._compare_param_groups(optim.param_groups, named_optim.param_groups)

        optim.add_param_group({"params": m.net2.parameters(), "lr": 1e-5})
        named_optim.add_param_group({"params": m_dup.net2.parameters(), "lr": 1e-5})
        _run_model_training([(m, [optim]), (m_dup, [named_optim])])
        self._compare_param_groups(optim.param_groups, named_optim.param_groups)

        optim.add_param_group({"params": m.net4[1].weight, "lr": 1e-3})
        named_optim.add_param_group({"params": m_dup.net4[1].weight, "lr": 1e-3})
        _run_model_training([(m, [optim]), (m_dup, [named_optim])])
        self._compare_param_groups(optim.param_groups, named_optim.param_groups)

    def test_add_param_group_error(self):
        m = TestDummyModel()
        named_optim = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            [
                {"params": m.net1.parameters()},
                {"params": m.net3.parameters(), "lr": 1e-3},
            ],
            lr=1e-2,
            momentum=0.9,
        )

        err_msg = "some parameters are not in the module"
        with self.assertRaisesRegex(ValueError, err_msg):
            named_optim.add_param_group({"params": [torch.ones(8, 1)], "lr": 1e-5})

    def test_init_state(self):
        m = TestDummyModel()
        named_optim = _NamedOptimizer(
            m.named_parameters(),
            torch.optim.SGD,
            [
                {"params": m.net1.parameters()},
                {"params": m.net3.parameters(), "lr": 1e-3},
            ],
            lr=1e-2,
            momentum=0.9,
        )
        named_sd = named_optim.state_dict()
        self.assertTrue(m.net1[0].weight.grad is None)
        self.assertTrue(len(named_sd["state"]) == 0)
        named_optim.init_state()
        named_sd = named_optim.state_dict()
        self.assertTrue(m.net1[0].weight.grad is not None)
        self.assertTrue("momentum_buffer" in named_sd["state"]["net1.0.weight"])
        self.assertFalse(
            torch.all(named_sd["state"]["net1.0.weight"]["momentum_buffer"]).item()
        )
        self.assertFalse(
            torch.all(named_sd["state"]["net1.0.bias"]["momentum_buffer"]).item()
        )
        self.assertTrue(m.net3.bias.grad is not None)
        self.assertTrue("momentum_buffer" in named_sd["state"]["net3.bias"])
        self.assertFalse(
            torch.all(named_sd["state"]["net3.bias"]["momentum_buffer"]).item()
        )
        self.assertFalse(
            torch.all(named_sd["state"]["net3.weight"]["momentum_buffer"]).item()
        )

```



## High-Level Overview


This Python file contains 2 class(es) and 14 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestDummyModel`, `NamedOptimizerTest`

**Functions defined**: `_run_model_training`, `__init__`, `forward`, `_compare_state_dict_group`, `_compare_param_groups`, `_compare_param_group`, `test_state_dict`, `test_state_dict_multi_param_group`, `test_load_state_dict`, `test_load_state_dict_conditional_training`, `test_load_state_dict_error`, `test_add_param_group`, `test_add_param_group_error`, `test_init_state`

**Key imports**: unittest, torch, torch.nn as nn, _NamedOptimizer


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/optim`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `unittest`
- `torch`
- `torch.nn as nn`
- `torch.distributed.optim`: _NamedOptimizer


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
python test/distributed/optim/test_named_optimizer.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/optim`):

- [`test_zero_redundancy_optimizer.py_docs.md`](./test_zero_redundancy_optimizer.py_docs.md)
- [`test_apply_optimizer_in_backward.py_docs.md`](./test_apply_optimizer_in_backward.py_docs.md)


## Cross-References

- **File Documentation**: `test_named_optimizer.py_docs.md`
- **Keyword Index**: `test_named_optimizer.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
