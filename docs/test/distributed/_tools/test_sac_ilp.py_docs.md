# Documentation: `test/distributed/_tools/test_sac_ilp.py`

## File Metadata

- **Path**: `test/distributed/_tools/test_sac_ilp.py`
- **Size**: 10,224 bytes (9.98 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This file is a **utility or tool script**. This appears to be a **test file**. Contains **unit tests** using Python testing frameworks. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]
import copy
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.ilp_utils import (
    aggregate_stats,
    get_peak_memory_runtime_baseline,
    ModuleInfo,
    parse_module_info,
)
from torch.distributed._tools.mem_tracker import _ModState, MemTracker
from torch.distributed._tools.runtime_estimator import RuntimeEstimator
from torch.distributed._tools.sac_estimator import SACEstimator, SACStats
from torch.distributed._tools.sac_ilp import (
    get_optimal_checkpointing_policy_per_module,
    sac_milp,
)
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    MI300_ARCH,
    run_tests,
    skipIfRocmArch,
    skipIfTorchDynamo,
    TestCase,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)


class TestSACILP(TestCase):
    def setUp(self):
        super().setUp()
        self.device = torch.cuda.current_device()
        self.estimate_mode = "operator-level-cost-model"

    def _init_model_input_optimizer(
        self,
    ) -> tuple[torch.nn.Module, torch.optim.Optimizer, torch.Tensor]:
        bsz = 8
        model_args = ModelArgs(
            n_layers=4,
            n_heads=12,
            vocab_size=8192,
            max_seq_len=1024,
            dim=768,
            dropout_p=0.1,
        )
        with torch.device(self.device):
            model = Transformer(model_args)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        inp = torch.randint(
            0, model_args.vocab_size, (bsz, model_args.max_seq_len), device=self.device
        )
        return (model, optimizer, inp)

    def _run_and_get_memTracker(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        inp: torch.Tensor,
    ) -> MemTracker:
        mem_tracker = MemTracker()
        mem_tracker.track_external(model, optimizer)
        with mem_tracker as mt:
            for iter_idx in range(2):  # running twice to initialize optimizer
                output = model(inp)
                output.sum().backward()
                if iter_idx == 1:
                    last_snapshot = mt.get_tracker_snapshot("current")
                optimizer.step()
                optimizer.zero_grad()
                if iter_idx == 0:
                    mt.reset_mod_stats()
        assert last_snapshot is not None
        for mod_stats in mem_tracker.memory_tracking.values():
            # postprocessing due to the fact that for ModTracker, the post backward hook
            # is not being called for modules whose inputs don't require gradients
            # TODO: fix this in ModTracker and ensure it does not lead to any perf regression
            if _ModState.POST_BW not in mod_stats.snapshots:
                mod_stats.snapshots.setdefault(_ModState.POST_BW, []).append(
                    copy.deepcopy(last_snapshot)
                )
        return mem_tracker

    def _run_and_get_runtime_estimator(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        inp: torch.Tensor,
    ) -> RuntimeEstimator:
        def _run_one_step() -> None:
            output = model(inp)
            output.sum().backward()
            optimizer.step()
            optimizer.zero_grad()

        # Initializing optimizer states and warm-up
        _run_one_step()

        runtime_estimator = RuntimeEstimator()
        with runtime_estimator(estimate_mode_type=self.estimate_mode):
            _run_one_step()  # We use only one iteration for estimation
        return runtime_estimator

    def _run_and_get_sac_estimator(
        self,
        model: torch.nn.Module,
        inp: torch.Tensor,
    ) -> SACEstimator:
        sac_estimator = SACEstimator()
        with sac_estimator(estimate_mode_type=self.estimate_mode):
            loss = model(inp).sum()
        loss.backward()
        return sac_estimator

    def _collect_module_info_with_fake_tensor_mode(self) -> ModuleInfo:
        with FakeTensorMode():
            model, optimizer, inp = self._init_model_input_optimizer()
            mem_tracker = self._run_and_get_memTracker(model, optimizer, inp)
            runtime_estimator = self._run_and_get_runtime_estimator(
                model, optimizer, inp
            )
            sac_estimator = self._run_and_get_sac_estimator(model, inp)
            mod_info = aggregate_stats(
                model,
                mem_tracker,
                runtime_estimator,
                sac_estimator,
                torch.device(self.device),
            )
        return mod_info

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    @skipIfRocmArch(MI300_ARCH)
    def test_sac_ilp_case1(self):
        """
        This is a case where the memory budget is either binding or too tight,
        meaning that with some AC, the model can fit into GPU memory.
        """
        mod_info = self._collect_module_info_with_fake_tensor_mode()
        g = parse_module_info(mod_info)

        peak_mem, compute_time = get_peak_memory_runtime_baseline(g)
        self.assertAlmostEqual(peak_mem / 2583888896, 1, delta=0.05)

        ac_decisions, recomputation_time, _ = sac_milp(
            g, memory_budget=1.6, world_size=4
        )

        # The solution should AC all four transformer layers. On A100 machine, the percentage of
        # activation memory to discard is 0.5232 for three layers and is 0.7964 for the fourth layer.
        # Due to symmetry, the layer that has 0.7964 can be any of the first three layers. On CI,
        # due to machine variance and difference in flops, the results can be different -- e.g.,
        # the ratios are  0.672, 0.5646, 0.5646, 0.5646 for the four transformer layers for test
        # linux-jammy-cuda11.8-py3.10-gcc9 / test (distributed, 1, 3, lf.linux.8xlarge.nvidia.gpu).
        # and recomputation_time = 58.14; compute_time = 902.26
        modules_to_ac = set(ac_decisions.keys())
        sorted_discard_ratio = sorted(ac_decisions.values())
        self.assertEqual(
            modules_to_ac,
            {"Transformer.layers." + str(i) for i in range(4)},  # n_layers=4
        )
        self.assertAlmostEqual(sorted_discard_ratio[0], 0.55, delta=0.05)
        self.assertAlmostEqual(sorted_discard_ratio[1], 0.55, delta=0.05)
        self.assertAlmostEqual(sorted_discard_ratio[2], 0.55, delta=0.05)
        self.assertAlmostEqual(sum(sorted_discard_ratio), 2.35, delta=0.05)
        self.assertAlmostEqual(ac_decisions["Transformer.layers.3"], 0.55, delta=0.05)

        # On A100 machine, recomputation_time is 6.97 ms and compute_time is 97.97 ms.
        # Since runtime is device_flops dependent, so we only check the ratio
        self.assertAlmostEqual(
            (recomputation_time / compute_time) / (6.97 / 97.97), 1, delta=0.25
        )

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_sac_ilp_case2(self):
        """
        This is a case where the memory budget is not binding, meaning that no
        AC is needed to fit the model into memory.
        """
        mod_info = self._collect_module_info_with_fake_tensor_mode()
        g = parse_module_info(mod_info)
        ac_decisions, recomputation_time, peak_mem = sac_milp(
            g, memory_budget=2.4, world_size=4
        )
        self.assertDictEqual(ac_decisions, {})
        self.assertEqual(recomputation_time, 0)
        self.assertGreater(peak_mem, 1)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_sac_ilp_case3(self):
        """
        This is a case where the memory budget is too tight, meaning that even with
        aggressive AC, the model cannot fit into memory.
        """
        mod_info = self._collect_module_info_with_fake_tensor_mode()
        g = parse_module_info(mod_info)
        ac_decisions, recomputation_time, peak_mem = sac_milp(
            g, memory_budget=0.8, world_size=4
        )
        self.assertEqual(ac_decisions, {})
        self.assertEqual(recomputation_time, 0)
        self.assertEqual(peak_mem, -1)


class TestOptimalCheckpointingPolicy(TestCase):
    # tests are adapted from tests in xformers
    # https://github.com/facebookresearch/xformers/blob/c6c0ac31f1b08542a0bc27278c6ed10f825f6963/tests/test_checkpoint.py#L222
    def setUp(self):
        super().setUp()
        data = [
            ("aten.copy_", 5, 0),
            ("aten.add", 5, 100),
            ("aten.div", 8, 100),
            ("aten.mm", 15, 120),
            ("aten.native_dropout", 15, 0),
            ("aten.linear", 9, 100),
            ("aten.t", 1, 0),
            ("aten.relu_", 5, 0),
        ]
        self.sac_stats = SACStats(
            func_names=[x[0] for x in data],
            runtimes=[x[1] for x in data],
            memory=[x[2] for x in data],
            view_like_ops=[6],
            rand_ops=[4],
            saved_autograd_ops=[],  # not needed for SAC decisions
            inplace_ops=[(0, 0), (7, 5)],
            force_store_random=False,
        )

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_get_optimial_checkpointing_policy_per_module(self):
        for memory_budget, optimal_soln in [
            (0, [1, 0, 0, 0, 1, 0, 0, 0]),
            (100 / 420, [1, 0, 0, 0, 1, 1, 0, 1]),
            (120 / 420, [1, 0, 0, 1, 1, 0, 0, 0]),
            (200 / 420, [1, 0, 1, 0, 1, 1, 0, 1]),
            (220 / 420, [1, 0, 0, 1, 1, 1, 0, 1]),
            (320 / 420, [1, 0, 1, 1, 1, 1, 0, 1]),
            (420 / 420, [1, 1, 1, 1, 1, 1, 0, 1]),
        ]:
            soln = get_optimal_checkpointing_policy_per_module(
                sac_stats=self.sac_stats, memory_budget=memory_budget
            )
            self.assertEqual(optimal_soln, soln)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 12 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestSACILP`, `TestOptimalCheckpointingPolicy`

**Functions defined**: `setUp`, `_init_model_input_optimizer`, `_run_and_get_memTracker`, `_run_and_get_runtime_estimator`, `_run_one_step`, `_run_and_get_sac_estimator`, `_collect_module_info_with_fake_tensor_mode`, `test_sac_ilp_case1`, `test_sac_ilp_case2`, `test_sac_ilp_case3`, `setUp`, `test_get_optimial_checkpointing_policy_per_module`

**Key imports**: copy, unittest, torch, FakeTensorMode, _ModState, MemTracker, RuntimeEstimator, SACEstimator, SACStats, TEST_CUDA


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/_tools`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `copy`
- `unittest`
- `torch`
- `torch._subclasses.fake_tensor`: FakeTensorMode
- `torch.distributed._tools.mem_tracker`: _ModState, MemTracker
- `torch.distributed._tools.runtime_estimator`: RuntimeEstimator
- `torch.distributed._tools.sac_estimator`: SACEstimator, SACStats
- `torch.testing._internal.common_cuda`: TEST_CUDA


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/distributed/_tools/test_sac_ilp.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/_tools`):

- [`test_fsdp2_mem_tracker.py_docs.md`](./test_fsdp2_mem_tracker.py_docs.md)
- [`test_runtime_estimator.py_docs.md`](./test_runtime_estimator.py_docs.md)
- [`test_mod_tracker.py_docs.md`](./test_mod_tracker.py_docs.md)
- [`test_memory_tracker.py_docs.md`](./test_memory_tracker.py_docs.md)
- [`test_sac_estimator.py_docs.md`](./test_sac_estimator.py_docs.md)
- [`test_mem_tracker.py_docs.md`](./test_mem_tracker.py_docs.md)
- [`test_fake_collectives.py_docs.md`](./test_fake_collectives.py_docs.md)


## Cross-References

- **File Documentation**: `test_sac_ilp.py_docs.md`
- **Keyword Index**: `test_sac_ilp.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
