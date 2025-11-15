# Documentation: `docs/test/optim/test_lrscheduler.py_docs.md`

## File Metadata

- **Path**: `docs/test/optim/test_lrscheduler.py_docs.md`
- **Size**: 52,875 bytes (51.64 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/optim/test_lrscheduler.py`

## File Metadata

- **Path**: `test/optim/test_lrscheduler.py`
- **Size**: 101,506 bytes (99.13 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: optimizer", "module: LrScheduler" ]
# ruff: noqa: F841
import copy
import math
import pickle
import tempfile
import types
import warnings
from functools import partial

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch.optim import Adam, Rprop, SGD
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    EPOCH_DEPRECATION_WARNING,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    LRScheduler,
    MultiplicativeLR,
    MultiStepLR,
    OneCycleLR,
    PolynomialLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)
from torch.optim.swa_utils import SWALR
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    load_tests,
    parametrize,
    skipIfTorchDynamo,
    TestCase,
)


# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127


class TestLRScheduler(TestCase):
    class SchedulerTestNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 1, 1)
            self.conv2 = torch.nn.Conv2d(1, 1, 1)

        def forward(self, x):
            return self.conv2(F.relu(self.conv1(x)))

    class LambdaLRTestObject:
        def __init__(self, value):
            self.value = value

        def __call__(self, epoch):
            return self.value * epoch

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return self.__dict__ == other.__dict__
            else:
                return False

    exact_dtype = True

    def setUp(self):
        super().setUp()
        self.net = self.SchedulerTestNet()
        self.opt = SGD(
            [
                {"params": self.net.conv1.parameters()},
                {"params": self.net.conv2.parameters(), "lr": torch.tensor(0.5)},
            ],
            lr=0.05,
        )

    def _check_warning_is_epoch_deprecation_warning(self, w, *, num_warnings: int = 1):
        """This function swallows the epoch deprecation warning which is produced when we
        call `scheduler.step(epoch)` with some not `None` value of `epoch`.
        this is deprecated, and this function will need to be removed/updated when
        the schedulers no longer accept the parameter at all.
        """
        self.assertEqual(len(w), num_warnings)
        for warning in w:
            self.assertEqual(len(warning.message.args), 1)
            self.assertEqual(warning.message.args[0], EPOCH_DEPRECATION_WARNING)

    def test_error_when_getlr_has_epoch(self):
        class MultiStepLR(torch.optim.lr_scheduler.LRScheduler):
            def __init__(self, optimizer, gamma, milestones, last_epoch=-1):
                self.init_lr = [group["lr"] for group in optimizer.param_groups]
                self.gamma = gamma
                self.milestones = milestones
                super().__init__(optimizer, last_epoch)

            def get_lr(self, step):
                global_step = self.last_epoch
                gamma_power = (
                    [0]
                    + [i + 1 for i, m in enumerate(self.milestones) if global_step >= m]
                )[-1]
                return [init_lr * (self.gamma**gamma_power) for init_lr in self.init_lr]

        optimizer = SGD([torch.rand(1)], lr=1)

        with self.assertRaises(TypeError):
            scheduler = MultiStepLR(optimizer, gamma=1, milestones=[10, 20])

    @skipIfTorchDynamo(
        "Torchdynamo keeps references to optim in the guards and the stack of the graph break frames"
    )
    def test_no_cyclic_references(self):
        import gc

        param = Parameter(torch.empty(10))
        optim = SGD([param], lr=0.5)
        scheduler = LambdaLR(optim, lambda epoch: 1.0)
        del scheduler

        self.assertTrue(
            len(gc.get_referrers(optim)) == 0,
            "Optimizer should contain no cyclic references",
        )

        gc.collect()
        del optim
        self.assertEqual(
            gc.collect(), 0, msg="Optimizer should be garbage-collected on __del__"
        )

    @skipIfTorchDynamo(
        "Torchdynamo keeps references to optim in the guards and the stack of the graph break frames"
    )
    def test_no_cyclic_references_in_step(self):
        import gc
        import weakref

        def run():
            param = torch.empty(10, requires_grad=True)
            optim = SGD(params=[param], lr=0.5)
            scheduler = LambdaLR(optim, lambda epoch: 1.0)
            param.sum().backward()
            optim.step()
            scheduler.step()

            return weakref.ref(scheduler)

        # To ensure that there are no reference cycles in scheduler,
        # we need to turn off the garbage collector. Since gc will
        # automatically collect unreachable objects.
        gc.disable()
        ref = run()

        assert ref() is None
        gc.enable()  # restore

    def test_old_pattern_warning(self):
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        def old_pattern():
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        self.assertWarnsRegex(UserWarning, r"how-to-adjust-learning-rate", old_pattern)

    def test_old_pattern_warning_with_arg(self):
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        def old_pattern2():
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        self.assertWarnsRegex(UserWarning, r"how-to-adjust-learning-rate", old_pattern2)

    def test_old_pattern_warning_resuming(self):
        epochs = 35
        for group in self.opt.param_groups:
            group["initial_lr"] = 0.01

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3, last_epoch=10)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        def old_pattern():
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        self.assertWarnsRegex(UserWarning, r"how-to-adjust-learning-rate", old_pattern)

    def test_old_pattern_warning_resuming_with_arg(self):
        epochs = 35
        for group in self.opt.param_groups:
            group["initial_lr"] = 0.01

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3, last_epoch=10)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        def old_pattern2():
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        self.assertWarnsRegex(UserWarning, r"how-to-adjust-learning-rate", old_pattern2)

    def test_old_pattern_warning_with_overridden_optim_step(self):
        epochs = 35
        for group in self.opt.param_groups:
            group["initial_lr"] = 0.01

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3, last_epoch=10)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        # emulate use-case with optimizer.step overridden
        import types

        old_step = self.opt.step

        def new_step(o, *args, **kwargs):
            retval = old_step(*args, **kwargs)
            return retval

        self.opt.step = types.MethodType(new_step, self.opt)

        def old_pattern2():
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        self.assertWarnsRegex(UserWarning, r"how-to-adjust-learning-rate", old_pattern2)

    def test_new_pattern_no_warning(self):
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            for _ in range(epochs):
                self.opt.step()
                scheduler.step()
            self.assertTrue(len(ws) == 0, "No warning should be raised")

    def test_new_pattern_no_warning_with_arg(self):
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            for _ in range(epochs):
                self.opt.step()
                scheduler.step()
            self.assertTrue(len(ws) == 0, "No warning should be raised")

    def test_new_pattern_no_warning_with_overridden_optim_step(self):
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        # emulate use-case with optimizer.step overridden
        import types

        old_step = self.opt.step

        def new_step(o, *args, **kwargs):
            retval = old_step(*args, **kwargs)
            return retval

        self.opt.step = types.MethodType(new_step, self.opt)

        def new_pattern():
            for _ in range(epochs):
                self.opt.step()
                scheduler.step()

        self.assertWarnsRegex(
            UserWarning, r"`optimizer.step\(\)` has been overridden", new_pattern
        )

    def _test_lr_is_constant_for_constant_epoch(self, scheduler):
        l = []

        for _ in range(10):
            scheduler.optimizer.step()
            with warnings.catch_warnings(record=True) as w:
                scheduler.step(2)
                self._check_warning_is_epoch_deprecation_warning(w)

            l.append(self.opt.param_groups[0]["lr"])
        self.assertEqual(min(l), max(l))

    def test_step_lr_is_constant_for_constant_epoch(self):
        scheduler = StepLR(self.opt, 2)
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    def test_exponential_lr_is_constant_for_constant_epoch(self):
        scheduler = ExponentialLR(self.opt, gamma=0.9)
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    def test_constantlr_is_constant_for_constant_epoch(self):
        scheduler = ConstantLR(self.opt)
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    def test_linear_linearlr_is_constant_for_constant_epoch(self):
        scheduler = LinearLR(self.opt)
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    def test_polynomial_lr_is_constant_for_constant_epoch(self):
        scheduler = PolynomialLR(self.opt, power=0.9)
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    def test_step_lr(self):
        # lr = 0.05     if epoch < 3
        # lr = 0.005    if 30 <= epoch < 6
        # lr = 0.0005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 3 + [0.005] * 3 + [0.0005] * 3 + [0.00005] * 3
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
        self._test(scheduler, targets, epochs)

    def test_get_last_lr_step_lr(self):
        from torch.nn import Parameter

        epochs = 10
        optimizer = SGD([Parameter(torch.randn(2, 2, requires_grad=True))], 0.1)
        targets = [[0.1] * 3 + [0.01] * 3 + [0.001] * 3 + [0.0001]]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_get_last_lr_multi_step_lr(self):
        # lr = 0.05     if epoch < 2
        # lr = 0.005    if 2 <= epoch < 5
        # lr = 0.0005   if 5 <= epoch < 9
        # lr = 0.00005   if 9 <= epoch
        epochs = 10
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005] * 1
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_raise_error_when_last_epoch_is_greater_than_0_and_initial_lr_is_not_specified(
        self,
    ):
        optimizer = SGD([Parameter(torch.randn(2, 2, requires_grad=True))], 0.1)
        with self.assertRaisesRegex(
            KeyError,
            r"param \'initial_lr\' is not specified in param_groups\[0\] when resuming scheduler with last_epoch >= 0",
        ):
            StepLR(optimizer, step_size=3, gamma=0.1, last_epoch=1)

    def test_multi_step_lr(self):
        # lr = 0.05     if epoch < 2
        # lr = 0.005    if 2 <= epoch < 5
        # lr = 0.0005   if epoch < 9
        # lr = 0.00005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005] * 3
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        self._test(scheduler, targets, epochs)

    def test_multi_step_lr_with_epoch(self):
        # lr = 0.05     if epoch < 2
        # lr = 0.005    if 2 <= epoch < 5
        # lr = 0.0005   if epoch < 9
        # lr = 0.00005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005] * 3
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        self._test_with_epoch(scheduler, targets, epochs)

    def test_get_last_lr_constantlr(self):
        # lr = 0.025     if epoch < 5
        # lr = 0.005    if 5 <= epoch
        epochs = 10
        single_targets = [0.025] * 5 + [0.05] * 5
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = ConstantLR(self.opt, factor=1.0 / 2, total_iters=5)
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_get_last_lr_linearlr(self):
        # lr = 0.025     if epoch == 0
        # lr = 0.03125   if epoch == 1
        # lr = 0.0375    if epoch == 2
        # lr = 0.04375   if epoch == 3
        # lr = 0.005     if 4 <= epoch
        epochs = 10
        start_factor = 1.0 / 4
        end_factor = 3.0 / 5
        iters = 4
        interpolation = [
            start_factor + i * (end_factor - start_factor) / iters for i in range(iters)
        ]
        single_targets = [x * 0.05 for x in interpolation] + [0.05 * end_factor] * (
            epochs - iters
        )
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = LinearLR(
            self.opt,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=iters,
        )
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_constantlr(self):
        # lr = 0.025     if epoch < 5
        # lr = 0.005    if 5 <= epoch
        epochs = 10
        single_targets = [0.025] * 5 + [0.05] * 5
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = ConstantLR(self.opt, factor=1.0 / 2, total_iters=5)
        self._test(scheduler, targets, epochs)

    def test_linearlr(self):
        # lr = 0.025     if epoch == 0
        # lr = 0.03125   if epoch == 1
        # lr = 0.0375    if epoch == 2
        # lr = 0.04375   if epoch == 3
        # lr = 0.005     if 4 <= epoch
        epochs = 10
        start_factor = 1.0 / 2
        iters = 4
        interpolation = [
            start_factor + i * (1 - start_factor) / iters for i in range(iters)
        ]
        single_targets = [x * 0.05 for x in interpolation] + [0.05] * (epochs - iters)
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = LinearLR(self.opt, start_factor=start_factor, total_iters=iters)
        self._test(scheduler, targets, epochs)

    def test_linearlr_start_factor_limits1(self):
        start_factor = 0.0
        iters = 4
        with self.assertRaises(ValueError):
            LinearLR(self.opt, start_factor=start_factor, total_iters=iters)

    def test_linearlr_start_factor_limits2(self):
        start_factor = 1.1
        iters = 4
        with self.assertRaises(ValueError):
            LinearLR(self.opt, start_factor=start_factor, total_iters=iters)

    def test_constantlr_with_epoch(self):
        # lr = 0.025     if epoch < 5
        # lr = 0.005    if 5 <= epoch
        epochs = 10
        single_targets = [0.025] * 5 + [0.05] * 5
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = ConstantLR(self.opt, factor=1.0 / 2, total_iters=5)
        self._test_with_epoch(scheduler, targets, epochs)

    def test_linearlr_with_epoch(self):
        # lr = 0.025     if epoch == 0
        # lr = 0.03125   if epoch == 1
        # lr = 0.0375    if epoch == 2
        # lr = 0.04375   if epoch == 3
        # lr = 0.005     if 4 <= epoch
        epochs = 10
        start_factor = 1.0 / 2
        end_factor = 1.0
        iters = 4
        interpolation = [
            start_factor + i * (end_factor - start_factor) / iters for i in range(iters)
        ]
        single_targets = [x * 0.05 for x in interpolation] + [0.05] * (epochs - iters)
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = LinearLR(self.opt, start_factor=start_factor, total_iters=iters)
        self._test_with_epoch(scheduler, targets, epochs)

    def test_exp_lr(self):
        epochs = 10
        single_targets = [0.05 * (0.9**x) for x in range(epochs)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = ExponentialLR(self.opt, gamma=0.9)
        self._test(scheduler, targets, epochs)

    def test_poly_lr(self):
        epochs = 10
        power = 0.9
        total_iters = 5
        single_targets = [
            (1.0 - x / total_iters) ** power * 0.05 for x in range(total_iters)
        ] + [0.0] * (epochs - total_iters)
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = PolynomialLR(self.opt, power=power, total_iters=total_iters)
        self._test(scheduler, targets, epochs)

    def test_cos_anneal_lr(self):
        epochs = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        self._test(scheduler, targets, epochs)

    def test_closed_form_step_lr(self):
        scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
        closed_form_scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_linearlr(self):
        scheduler = LinearLR(
            self.opt, start_factor=1.0 / 3, end_factor=0.7, total_iters=4
        )
        closed_form_scheduler = LinearLR(
            self.opt, start_factor=1.0 / 3, end_factor=0.7, total_iters=4
        )
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_constantlr(self):
        scheduler = ConstantLR(self.opt, factor=1.0 / 3, total_iters=4)
        closed_form_scheduler = ConstantLR(self.opt, factor=1.0 / 3, total_iters=4)
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_multi_step_lr(self):
        scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        closed_form_scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_exp_lr(self):
        scheduler = ExponentialLR(self.opt, gamma=0.9)
        closed_form_scheduler = ExponentialLR(self.opt, gamma=0.9)
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_poly_lr(self):
        scheduler = PolynomialLR(self.opt, power=0.9)
        closed_form_scheduler = PolynomialLR(self.opt, power=0.9)
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_cos_anneal_lr(self):
        eta_min = 1e-10
        epochs = 20
        T_max = 5
        scheduler = CosineAnnealingLR(self.opt, T_max=T_max, eta_min=eta_min)
        closed_form_scheduler = CosineAnnealingLR(
            self.opt, T_max=T_max, eta_min=eta_min
        )
        self._test_against_closed_form(scheduler, closed_form_scheduler, epochs)

    def test_cos_anneal_lr_continue(self):
        eta_min = 0.1
        T_max = 5
        scheduler = CosineAnnealingLR(self.opt, T_max=T_max, eta_min=eta_min)
        self.opt.step()
        scheduler.step()
        original_lrs = scheduler._last_lr
        new_scheduler = CosineAnnealingLR(
            self.opt, T_max=T_max, eta_min=eta_min, last_epoch=0
        )
        new_lrs = new_scheduler._last_lr
        torch.testing.assert_close(original_lrs, new_lrs, rtol=1e-4, atol=1e-5)

    def test_reduce_lr_on_plateau1(self):
        epochs = 10
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 20]
        metrics = [10 - i * 0.0167 for i in range(20)]
        scheduler = ReduceLROnPlateau(
            self.opt,
            threshold_mode="abs",
            mode="min",
            threshold=0.01,
            patience=5,
            cooldown=5,
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau2(self):
        epochs = 22
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 6 + [0.05] * 7 + [0.005] * 7 + [0.0005] * 2]
        metrics = [10 - i * 0.0165 for i in range(22)]
        scheduler = ReduceLROnPlateau(
            self.opt,
            patience=5,
            cooldown=0,
            threshold_mode="abs",
            mode="min",
            threshold=0.1,
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau3(self):
        epochs = 22
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * (2 + 6) + [0.05] * (5 + 6) + [0.005] * 4]
        metrics = [-0.8] * 2 + [-0.234] * 20
        scheduler = ReduceLROnPlateau(
            self.opt, mode="max", patience=5, cooldown=5, threshold_mode="abs"
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau4(self):
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 20]
        metrics = [1.5 * (1.025**i) for i in range(20)]  # 1.025 > 1.1**0.25
        scheduler = ReduceLROnPlateau(
            self.opt, mode="max", patience=3, threshold_mode="rel", threshold=0.1
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau5(self):
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 6 + [0.05] * (5 + 6) + [0.005] * 4]
        metrics = [1.5 * (1.005**i) for i in range(20)]
        scheduler = ReduceLROnPlateau(
            self.opt,
            mode="max",
            threshold_mode="rel",
            threshold=0.1,
            patience=5,
            cooldown=5,
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau6(self):
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 20]
        metrics = [1.5 * (0.85**i) for i in range(20)]
        scheduler = ReduceLROnPlateau(
            self.opt, mode="min", threshold_mode="rel", threshold=0.1
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau7(self):
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 6 + [0.05] * (5 + 6) + [0.005] * 4]
        metrics = [1] * 7 + [0.6] + [0.5] * 12
        scheduler = ReduceLROnPlateau(
            self.opt,
            mode="min",
            threshold_mode="rel",
            threshold=0.1,
            patience=5,
            cooldown=5,
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau8(self):
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 6 + [0.4] * 14, [0.5] * 6 + [0.3] * 14]
        metrics = [1.5 * (1.005**i) for i in range(20)]
        scheduler = ReduceLROnPlateau(
            self.opt,
            mode="max",
            threshold_mode="rel",
            min_lr=[0.4, 0.3],
            threshold=0.1,
            patience=5,
            cooldown=5,
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau_get_last_lr_before_step(self):
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        scheduler = ReduceLROnPlateau(
            self.opt,
        )
        self.assertEqual(
            scheduler.get_last_lr(), [0.5 for param_group in self.opt.param_groups]
        )

    def test_reduce_lr_on_plateau_preserves_lr_type(self):
        # Ensures that tensor lrs are preserved, preventing recompilations.
        types = [type(group["lr"]) for group in self.opt.param_groups]
        scheduler = ReduceLROnPlateau(self.opt, mode="min", patience=0)
        scheduler.step(1.0)
        scheduler.step(2.0)  # Triggers scheduler._reduce_lr
        for group, type_ in zip(self.opt.param_groups, types):
            self.assertEqual(type(group["lr"]), type_)

    def test_sequentiallr1(self):
        epochs = 19
        schedulers = [None] * 2
        targets = [
            [0.05, 0.04, 0.032]
            + [0.05 for x in range(4)]
            + [0.05 * 0.1 for x in range(4)]
            + [0.05 * 0.01 for x in range(4)]
            + [0.05 * 0.001 for x in range(4)]
        ]
        milestones = [3]
        schedulers[0] = ExponentialLR(self.opt, gamma=0.8)
        schedulers[1] = StepLR(self.opt, gamma=0.1, step_size=4)
        scheduler = SequentialLR(self.opt, schedulers=schedulers, milestones=milestones)
        self._test(scheduler, targets, epochs)

    def test_sequentiallr2(self):
        epochs = 13
        schedulers = [None] * 2
        targets = [[0.005, 0.005, 0.005] + [0.05 * 0.9**x for x in range(10)]]
        milestones = [3]
        schedulers[0] = ConstantLR(self.opt, factor=0.1, total_iters=3)
        schedulers[1] = ExponentialLR(self.opt, gamma=0.9)
        scheduler = SequentialLR(self.opt, schedulers=schedulers, milestones=milestones)
        self._test(scheduler, targets, epochs)

    def test_sequentiallr3(self):
        epochs = 12
        schedulers = [None] * 3
        targets = [
            [0.005, 0.005, 0.005]
            + [0.05, 0.04, 0.032]
            + [0.05, 0.05, 0.005, 0.005, 0.0005, 0.0005]
        ]
        milestones = [3, 6]
        schedulers[0] = ConstantLR(self.opt, factor=0.1, total_iters=3)
        schedulers[1] = ExponentialLR(self.opt, gamma=0.8)
        schedulers[2] = StepLR(self.opt, gamma=0.1, step_size=2)
        scheduler = SequentialLR(self.opt, schedulers=schedulers, milestones=milestones)
        self._test(scheduler, targets, epochs)

    def test_sequentiallr4(self):
        optimizer = SGD([torch.tensor(0.5)], lr=0.1)
        prev_lr = optimizer.param_groups[0]["lr"]

        schedulers = [
            torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1),
            torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1),
        ]
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers, milestones=[10]
        )

        new_lr = optimizer.param_groups[0]["lr"]

        # Ensure that multiple schedulers does not affect the initial learning rate
        self.assertEqual(prev_lr, new_lr)

    def test_sequentiallr5(self):
        """
        Test SequentialLR with a ChainedScheduler.
        """
        epochs = 10
        schedulers = []
        milestones = []

        targets = [
            [0.0005, 0.0014, 0.0023, 0.0032, 0.0041]
            + [0.025, 0.025, 0.025, 0.025, 0.025]
        ]

        const_sched = ConstantLR(optimizer=self.opt, factor=0.1, total_iters=5)
        lin_sched = LinearLR(optimizer=self.opt, start_factor=0.1, total_iters=5)
        milestones.append(5)

        chained = ChainedScheduler([lin_sched, const_sched])
        schedulers.append(chained)

        const_sched2 = ConstantLR(optimizer=self.opt, factor=0.5, total_iters=5)
        schedulers.append(const_sched2)

        scheduler = SequentialLR(self.opt, schedulers=schedulers, milestones=milestones)
        self._test(scheduler, targets, epochs)

    def test_sequentiallr_no_warnings(self):
        scheduler1 = LinearLR(self.opt, start_factor=0.5, end_factor=0.1, total_iters=5)
        scheduler2 = ExponentialLR(self.opt, gamma=0.9)
        scheduler = SequentialLR(
            self.opt, schedulers=[scheduler1, scheduler2], milestones=[5]
        )

        for _ in range(10):
            self.opt.step()
            with warnings.catch_warnings(record=True) as ws:
                scheduler.step()
                self.assertTrue(len(ws) == 0, "No warning should be raised")

    def test_get_last_lr_sequentiallr(self):
        epochs = 12
        milestones = [3, 6]
        schedulers = [None] * 3
        schedulers[0] = ConstantLR(self.opt, factor=0.1, total_iters=3)
        schedulers[1] = ExponentialLR(self.opt, gamma=0.8)
        schedulers[2] = StepLR(self.opt, gamma=0.1, step_size=2)
        scheduler = SequentialLR(self.opt, schedulers=schedulers, milestones=milestones)
        constant_lr_target = [0.005] * 3
        exponential_lr_target = [0.05, 0.04, 0.032]
        step_lr_target = [0.05, 0.05, 0.005, 0.005, 0.0005, 0.0005]
        single_targets = constant_lr_target + exponential_lr_target + step_lr_target
        targets = [single_targets, [x * 10 for x in single_targets]]
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_sequentiallr_does_not_alias_lr_and_initial_lr(self):
        # The TestLRScheduler object uses self.opt to avoid instantiating a new optimizer for each test.
        # self.opt has a float lr, and we need to use a Tensor lr to ensure that a former SequentialLR bug is fixed.
        # For more context, see https://github.com/pytorch/pytorch/issues/162359
        old_opt = self.opt
        lr = torch.tensor(2.0)
        self.opt = SGD(self.net.parameters(), lr=lr)
        milestone = 4
        epochs = 8
        start, end = 0.1, 0.8

        schedulers = [
            LinearLR(self.opt, start, end, total_iters=milestone),
            LinearLR(self.opt, end, start, total_iters=epochs - milestone),
        ]
        targets = [[0.2, 0.55, 0.9, 1.25, 1.6, 1.25, 0.9, 0.55]]

        scheduler = SequentialLR(self.opt, schedulers, milestones=[milestone])
        self._test(scheduler, targets, epochs)
        self.opt = old_opt

    def test_chained_lr2_get_last_lr_before_step(self):
        schedulers = [
            LinearLR(self.opt, start_factor=0.4, total_iters=3),
            MultiStepLR(self.opt, milestones=[4, 8, 10], gamma=0.1),
        ]
        scheduler = ChainedScheduler(schedulers)
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_chained_lr1(self):
        epochs = 10
        schedulers = [None] * 1
        targets = [[0.05] * 3 + [0.005] * 3 + [0.0005] * 3 + [0.00005] * 3]
        schedulers[0] = StepLR(self.opt, gamma=0.1, step_size=3)
        scheduler = ChainedScheduler(schedulers)
        self._test([scheduler], targets, epochs)
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_chained_lr2(self):
        epochs = 10
        schedulers = [None] * 1
        targets = [[0.02, 0.03, 0.04] + [0.05] * 9]
        schedulers[0] = LinearLR(self.opt, start_factor=0.4, total_iters=3)
        scheduler = ChainedScheduler(schedulers)
        self._test([scheduler], targets, epochs)
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_chained_lr3(self):
        epochs = 10
        schedulers = [None] * 2
        targets = [
            [0.02, 0.03, 0.04, 0.05] + [0.005] * 4 + [0.0005] * 3 + [0.00005] * 3
        ]
        schedulers[0] = LinearLR(self.opt, start_factor=0.4, total_iters=3)
        schedulers[1] = MultiStepLR(self.opt, milestones=[4, 8, 10], gamma=0.1)
        scheduler = ChainedScheduler(schedulers)
        self._test([scheduler], targets, epochs)
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_chained_lr4(self):
        epochs = 9
        schedulers = [None] * 3
        targets = [
            [0.05 * 0.2 * 0.9**x for x in range(3)]
            + [0.05 * 0.2 * 0.9**3 * 0.1]
            + [0.05 * 0.9**x * 0.1 for x in range(4, 6)]
            + [0.05 * 0.9**x * 0.01 for x in range(6, 9)]
        ]
        schedulers[0] = ExponentialLR(self.opt, gamma=0.9)
        schedulers[1] = ConstantLR(self.opt, factor=0.2, total_iters=4)
        schedulers[2] = StepLR(self.opt, gamma=0.1, step_size=3)
        scheduler = ChainedScheduler(schedulers)
        self._test([scheduler], targets, epochs)
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_chained_lr5(self):
        def poly_lr(lr: float):
            return [
                (lr * ((1.0 - x / total_iters) ** power)) for x in range(total_iters)
            ] + [0.0] * (epochs - total_iters)

        schedulers = [None] * 2
        epochs = 10
        power = 0.9
        total_iters = 5
        const_factor = 0.1
        single_targets = [x * const_factor for x in poly_lr(lr=0.05)]
        targets = [single_targets, [x * const_factor for x in poly_lr(0.5)]]
        schedulers[0] = PolynomialLR(self.opt, power=power, total_iters=total_iters)
        schedulers[1] = ConstantLR(self.opt, factor=const_factor)
        scheduler = ChainedScheduler(schedulers)
        self._test(scheduler, targets, epochs)
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_compound_step_and_multistep_lr(self):
        epochs = 10
        schedulers = [None] * 2
        schedulers[0] = StepLR(self.opt, gamma=0.1, step_size=3)
        schedulers[1] = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        targets = [[0.05] * 2 + [0.005] * 1 + [5e-4] * 2 + [5e-5] + [5e-6] * 3 + [5e-8]]
        self._test(schedulers, targets, epochs)

    def test_compound_step_and_exp_lr(self):
        epochs = 10
        schedulers = [None] * 2
        single_targets = [0.05 * (0.9**x) for x in range(3)]
        single_targets += [0.005 * (0.9**x) for x in range(3, 6)]
        single_targets += [0.0005 * (0.9**x) for x in range(6, 9)]
        single_targets += [0.00005 * (0.9**x) for x in range(9, 12)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = StepLR(self.opt, gamma=0.1, step_size=3)
        schedulers[1] = ExponentialLR(self.opt, gamma=0.9)
        self._test(schedulers, targets, epochs)

    def test_compound_exp_and_multistep_lr(self):
        epochs = 10
        schedulers = [None] * 2
        single_targets = [0.05 * (0.9**x) for x in range(2)]
        single_targets += [0.005 * (0.9**x) for x in range(2, 5)]
        single_targets += [0.0005 * (0.9**x) for x in range(5, 9)]
        single_targets += [0.00005 * (0.9**x) for x in range(9, 11)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        schedulers[1] = ExponentialLR(self.opt, gamma=0.9)
        self._test(schedulers, targets, epochs)

    def test_compound_exp_and_linearlr(self):
        epochs = 10
        iters = 4
        start_factor = 0.4
        end_factor = 0.9
        schedulers = [None] * 2
        single_targets = [0.05 * (0.9**x) for x in range(11)]
        for i in range(iters):
            single_targets[i] *= start_factor + i / iters * (end_factor - start_factor)
        for i in range(iters, 11):
            single_targets[i] *= end_factor
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = LinearLR(
            self.opt,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=iters,
        )
        schedulers[1] = ExponentialLR(self.opt, gamma=0.9)
        self._test(schedulers, targets, epochs)

    def test_compound_step_and_constantlr(self):
        epochs = 10
        iters = 4
        factor = 0.4
        schedulers = [None] * 2
        single_targets = (
            [0.05 * 0.4] * 3
            + [0.005 * 0.4]
            + [0.005] * 2
            + [0.0005] * 3
            + [0.00005] * 3
        )
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = StepLR(self.opt, gamma=0.1, step_size=3)
        schedulers[1] = ConstantLR(self.opt, factor=0.4, total_iters=4)
        self._test(schedulers, targets, epochs)

    def test_compound_linearlr_and_multistep_lr(self):
        epochs = 10
        iters = 4
        start_factor = 0.4
        schedulers = [None] * 2
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005] * 2
        for i in range(iters):
            single_targets[i] *= start_factor + i / iters * (1 - start_factor)
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        schedulers[1] = LinearLR(self.opt, start_factor=start_factor, total_iters=iters)
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_step_lr(self):
        epochs = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        single_targets = [x * 0.1 ** (i // 3) for i, x in enumerate(single_targets)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers = [None] * 2
        schedulers[0] = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        schedulers[1] = StepLR(self.opt, gamma=0.1, step_size=3)
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_multistep_lr(self):
        epochs = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        multipliers = [1] * 2 + [0.1] * 3 + [0.01] * 4 + [0.001]
        single_targets = [x * y for x, y in zip(single_targets, multipliers)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers = [None] * 2
        schedulers[0] = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        schedulers[1] = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_linearlr(self):
        epochs = 10
        iters = 4
        start_factor = 0.4
        eta_min = 1e-10
        schedulers = [None] * 2
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        for i in range(iters):
            single_targets[i] *= start_factor + i / iters * (1 - start_factor)
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = LinearLR(self.opt, start_factor=start_factor, total_iters=iters)
        schedulers[1] = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_exp_lr(self):
        epochs = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        multipliers = [0.1**i for i in range(epochs)]
        single_targets = [x * y for x, y in zip(single_targets, multipliers)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers = [None] * 2
        schedulers[0] = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        schedulers[1] = ExponentialLR(self.opt, gamma=0.1)
        self._test(schedulers, targets, epochs)

    def test_compound_reduce_lr_on_plateau1(self):
        epochs = 10
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        single_targets = [0.5] * 20
        multipliers = [0.1 ** (i // 3) for i in range(20)]
        single_targets = [x * y for x, y in zip(multipliers, single_targets)]
        targets = [single_targets]
        targets = targets[1:]  # test runs step before checking lr
        metrics = [10 - i * 0.0167 for i in range(20)]
        schedulers = [None, None]
        schedulers[0] = ReduceLROnPlateau(
            self.opt,
            threshold_mode="abs",
            mode="min",
            threshold=0.01,
            patience=5,
            cooldown=5,
        )
        schedulers[1] = StepLR(self.opt, gamma=0.1, step_size=3)
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_compound_reduce_lr_on_plateau2(self):
        epochs = 22
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        single_targets = [0.5] * 6 + [0.05] * 7 + [0.005] * 7 + [0.0005] * 2
        multipliers = [1] * 3 + [0.1] * 5 + [0.01] * 4 + [0.001] * 10
        single_targets = [x * y for x, y in zip(single_targets, multipliers)]
        targets = [single_targets]
        targets = targets[1:]  # test runs step before checking lr
        metrics = [10 - i * 0.0165 for i in range(22)]
        schedulers = [None] * 2
        schedulers[0] = ReduceLROnPlateau(
            self.opt,
            patience=5,
            cooldown=0,
            threshold_mode="abs",
            mode="min",
            threshold=0.1,
        )
        schedulers[1] = MultiStepLR(self.opt, gamma=0.1, milestones=[3, 8, 12])
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_compound_reduce_lr_on_plateau3(self):
        epochs = 22
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        single_targets = [0.5] * (2 + 6) + [0.05] * (5 + 6) + [0.005] * 4
        multipliers = [0.1**i for i in range(epochs)]
        single_targets = [x * y for x, y in zip(multipliers, single_targets)]
        targets = [single_targets]
        targets = targets[1:]  # test runs step before checking lr
        metrics = [-0.8] * 2 + [-0.234] * 20
        schedulers = [None, None]
        schedulers[0] = ReduceLROnPlateau(
            self.opt, mode="max", patience=5, cooldown=5, threshold_mode="abs"
        )
        schedulers[1] = ExponentialLR(self.opt, gamma=0.1)
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_compound_reduce_lr_on_plateau4(self):
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.05
        epochs = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        targets = [single_targets]
        targets = targets[1:]  # test runs step before checking lr
        metrics = [1.5 * (1.025**i) for i in range(20)]  # 1.025 > 1.1**0.25
        schedulers = [None, None]
        schedulers[0] = ReduceLROnPlateau(
            self.opt, mode="max", patience=3, threshold_mode="rel", threshold=0.1
        )
        schedulers[1] = CosineAnnealingLR(self.opt, epochs, eta_min)
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_compound_reduce_lr_on_plateau5(self):
        iters = 4
        start_factor = 0.4
        epochs = 22
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        single_targets = [0.5] * 6 + [0.05] * 7 + [0.005] * 7 + [0.0005] * 2
        multipliers = [1] * 22
        for i in range(iters):
            multipliers[i] *= start_factor + i / iters * (1 - start_factor)
        single_targets = [x * y for x, y in zip(single_targets, multipliers)]
        targets = [single_targets]
        targets = targets[1:]  # test runs step before checking lr
        metrics = [10 - i * 0.0165 for i in range(22)]
        schedulers = [None] * 2
        schedulers[0] = ReduceLROnPlateau(
            self.opt,
            patience=5,
            cooldown=0,
            threshold_mode="abs",
            mode="min",
            threshold=0.1,
        )
        schedulers[1] = LinearLR(self.opt, start_factor=start_factor, total_iters=iters)
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_cycle_lr_invalid_mode(self):
        with self.assertRaises(ValueError):
            scheduler = CyclicLR(self.opt, base_lr=0, max_lr=0, mode="CATS")

    def test_cycle_lr_triangular_mode_one_lr(self):
        lr_target = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        momentum_target = [5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicLR(
            self.opt,
            base_lr=1,
            max_lr=5,
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=1,
            max_momentum=5,
            mode="triangular",
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    def test_cycle_lr_triangular_mode_one_lr_no_momentum(self):
        lr_target = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        lr_targets = [lr_target, lr_target]
        momentum_target = [self.opt.defaults["momentum"]] * len(lr_target)
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicLR(
            self.opt,
            base_lr=1,
            max_lr=5,
            step_size_up=4,
            cycle_momentum=False,
            mode="triangular",
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    def test_cycle_lr_triangular2_mode_one_lr(self):
        lr_target = [
            1,
            2,
            3,
            4,
            5,
            4,
            3,
            2,
            1,
            1.5,
            2.0,
            2.5,
            3.0,
            2.5,
            2.0,
            1.5,
            1,
            1.25,
            1.50,
            1.75,
            2.00,
            1.75,
        ]
        momentum_target = [
            5.0,
            4.0,
            3.0,
            2.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            4.5,
            4.0,
            3.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            4.75,
            4.5,
            4.25,
            4.0,
            4.25,
        ]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicLR(
            self.opt,
            base_lr=1,
            max_lr=5,
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=1,
            max_momentum=5,
            mode="triangular2",
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    def test_cycle_lr_exp_range_mode_one_lr(self):
        base_lr, max_lr = 1, 5
        di
```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/optim`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/optim`, which is part of the **testing infrastructure**.



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
python docs/test/optim/test_lrscheduler.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/optim`):

- [`test_optim.py_docs.md_docs.md`](./test_optim.py_docs.md_docs.md)
- [`test_swa_utils.py_kw.md_docs.md`](./test_swa_utils.py_kw.md_docs.md)
- [`test_lrscheduler.py_kw.md_docs.md`](./test_lrscheduler.py_kw.md_docs.md)
- [`test_optim.py_kw.md_docs.md`](./test_optim.py_kw.md_docs.md)
- [`test_swa_utils.py_docs.md_docs.md`](./test_swa_utils.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_lrscheduler.py_docs.md_docs.md`
- **Keyword Index**: `test_lrscheduler.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
